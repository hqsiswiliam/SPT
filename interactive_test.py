import argparse
import glob
import json
import locale
import os
import random
import re
import time
from multiprocessing import freeze_support

import deepspeed
import torch
from dotenv import load_dotenv
from torch.utils.data import DistributedSampler

from dataset.dataset import PersonaChatDataset
from utils.dist_helper import setup
from utils.format_inputs import TASK_TYPE
from utils.parser_helper import str2bool

os.environ["PYTHONIOENCODING"] = "utf-8"
myLocale = locale.setlocale(category=locale.LC_ALL, locale="C.UTF-8")
load_dotenv()

argparse = argparse.ArgumentParser()
argparse.add_argument('--model_path', type=str, default=None)
argparse.add_argument('--path_pattern', type=str, default=None)
argparse.add_argument('--batch_size', type=int)
argparse.add_argument('--valid_path', type=str, default=None)
argparse.add_argument('--local_rank', type=int, default=-1)
argparse.add_argument('--skip_exists', type=str2bool, default=False)
argparse.add_argument('--selection_noise', type=float, default=None)
parser = deepspeed.add_config_arguments(argparse)
args = argparse.parse_args()
_cmd_args = parser.parse_args()
freeze_support()

VICUNA_PREFIX = 'PATH_TO_VICUNA'


def test_process(model_paths, batch_size, valid_path, skip_exists, selection_noise, cmd_args):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    with open(cmd_args.deepspeed_config) as json_file:
        ds_config = json.load(json_file)
        del cmd_args.deepspeed_config

    setup()
    for model_path in model_paths:
        try:
            if selection_noise is not None:
                save_dir = os.sep.join(
                    model_path.split(os.sep)[:-1]) + os.sep + f'evaluation_result_selection_noise={selection_noise}.pkl'
            else:
                save_dir = os.sep.join(model_path.split(os.sep)[:-1]) + os.sep + 'evaluation_result.pkl'
            if os.path.exists(save_dir) and (skip_exists):
                continue
            print(
                f"Start setup rank {deepspeed.comm.get_local_rank()} of {world_size} on GPU {torch.cuda.current_device()}")

            ckpt = torch.load(os.sep.join(model_path.split(os.sep)[:-1]) + os.sep + 'checkpoint_best.pth',
                              map_location=f'cpu')
            config = ckpt['config']
            ds_config['train_micro_batch_size_per_gpu'] = batch_size
            load_precision = '32'
            if config.model.load_bit == 16:
                ds_config['float16']['enabled'] = True
                load_precision = 'fp16'
            if 'llama' in config.model.model_name.lower():
                ds_config['float16']['enabled'] = False
                ds_config['bf16']['enabled'] = True
                load_precision = 'bf16'
            load_bit_map = {
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
                '32': torch.float32}

            if config.model.model_type == 'selective_pt':
                from models.selective_llm_chat import SelectLLMChat as LLMChat
            else:
                from models.llm_chat import LLMChat
            if 'vicuna' in config.model.model_name and (not os.path.exists(config.model.model_name)):
                config.model.model_name = VICUNA_PREFIX + os.sep + config.model.model_name.split(os.sep)[-1]
            _model = LLMChat(config, batch_size)
            left_tokenizer = _model.left_tokenizer
            right_tokenizer = _model.right_tokenizer
            print(f'LOADING {model_path} with {load_precision} precision')
            model_engine, _, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=_model,
                                                         config=ds_config,
                                                         )
            model_engine.load_checkpoint(model_path, load_module_strict=False, load_optimizer_states=False,
                                         load_lr_scheduler_states=False,
                                         load_module_only=True)
            valid_path_file = valid_path
            if valid_path_file is None:
                valid_path_file = config.dataset.valid
            if config.dataset.test.__class__ is str:
                valid_path_file = config.dataset.test
                print('using train split from personachat')
            task_type = TASK_TYPE(config.training.task_type)

            valid_dataset = PersonaChatDataset(valid_path_file, max_context_turns=config.dataset.max_context_turns)
            from dataset.dataset import get_dataloader
            max_new_token = 32
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, shuffle=False,
                                               drop_last=False)
            valid_dataloader = get_dataloader(valid_dataset, batch_size, num_workers=0, sampler=valid_sampler)

            context_input = []
            persona_list = []
            dist_pred_text = [None for _ in range(world_size)]
            dist_gt_text = [None for _ in range(world_size)]
            pred_text = []
            gt_text = []
            selected_prompts = []
            print('Please enter your input:')
            first_setence = input()
            chosen_persona = random.choice([p['persona'] for p in valid_dataset.turns_data])
            history = [f"Q: {first_setence}"]
            history_with_prompt_idx = [f"USER: {first_setence}"]
            selected_prompts = []
            while True:
                data = {'context_input': [history],
                        'persona_list': [chosen_persona],
                        'target': ['not use']}
                _, text, batch_selected_prompts = LLMChat.test_step(model_engine, data, left_tokenizer,
                                                                    right_tokenizer,
                                                                    config, max_new_tokens=max_new_token,
                                                                    tqdm_instance=None,
                                                                    selection_noise=None,
                                                                    no_repeat_ngram_size=4,
                                                                    top_p=0.9,
                                                                    num_beams=10)
                response = text[0].strip()
                search_result = re.search('R:|Q:|Summary:|\n|\:', response)
                if search_result is not None:
                    response = response[:search_result.span()[0]]
                response = response.strip()

                selected_prompts.append(batch_selected_prompts.item())
                history += [f"R: {response}"]
                history_with_prompt_idx += [f"SPT: {response} [SPT Index: {batch_selected_prompts.item()}]"]
                history_str = "\n".join(history_with_prompt_idx)
                print_str = f"""
Persona: {' '.join(chosen_persona)}
Dialogue: 
{history_str}
                    """
                print(print_str)
                print('Please enter your input:')
                user_input = input()
                if user_input == 'r':
                    history = history[:-1]
                    history_with_prompt_idx = history_with_prompt_idx[:-1]
                    continue
                if user_input == 'exit':
                    exit()
                elif user_input == 'save':
                    os.makedirs('interactive_dialog', exist_ok=True)
                    with open('interactive_dialog/'+time.strftime('%Y-%m-%d-%H%M')+'.txt', 'w') as file:
                        file.write(print_str)
                    history = []
                    history_with_prompt_idx = []
                    chosen_persona = random.choice([p['persona'] for p in valid_dataset.turns_data])
                    print('Please enter your input:')
                    user_input = input()
                elif user_input == 'clear':
                    history = []
                    history_with_prompt_idx = []
                    chosen_persona = random.choice([p['persona'] for p in valid_dataset.turns_data])
                    print('Please enter your input:')
                    user_input = input()
                history += [f"Q: {user_input}"]
                history_with_prompt_idx += [f"USER: {user_input}"]

        except Exception as e:
            save_dir = os.sep.join(model_path.split(os.sep)[:-1]) + os.sep + "test_error.txt"
            print(f'WRITING TESTING ERROR! ERROR: {str(e)}')
            with open(save_dir, 'w') as file:
                file.write(str(e))
        deepspeed.comm.barrier()
    deepspeed.comm.barrier()


model_path_arg = args.model_path
model_paths = [model_path_arg]
if len(glob.glob(model_path_arg + os.sep + 'ds_ckpt' + os.sep + '*')):
    model_paths = [model_path_arg + os.sep + 'ds_ckpt']
elif not model_path_arg.endswith('.pth'):
    import glob

    path_pattern = args.path_pattern
    if path_pattern is not None:
        model_paths = glob.glob(f'{model_path_arg}/{path_pattern}/ds_ckpt/*/*.pt')
    else:
        model_paths = glob.glob(f'{model_path_arg}/*/ds_ckpt/*/*.pt')
    model_paths = list(set([os.sep.join(p.split(os.sep)[:-2]) for p in model_paths]))
    print(model_paths)
num_of_gpus = torch.cuda.device_count()
print(f"{num_of_gpus} GPUs available")
test_process(model_paths, args.batch_size, args.valid_path,
             args.skip_exists, args.selection_noise, cmd_args=_cmd_args)
deepspeed.comm.barrier()
deepspeed.comm.destroy_process_group()
print('Test Ends')
