import argparse
import glob
import json
import locale
import os
import re
from functools import reduce
from multiprocessing import freeze_support

import deepspeed
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from torch.utils.data import DistributedSampler
from tqdm import tqdm

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
                save_dir = os.sep.join(model_path.split(os.sep)[:-1]) + os.sep + f'evaluation_result_selection_noise={selection_noise}.pkl'
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
            tqdm_iterator = tqdm(valid_dataloader, total=len(valid_dataloader))
            selected_prompts = []
            for data in tqdm_iterator:
                _, text, batch_selected_prompts = LLMChat.test_step(model_engine, data, left_tokenizer,
                                                                    right_tokenizer,
                                                                    config, max_new_tokens=max_new_token,
                                                                    tqdm_instance=tqdm_iterator,
                                                                    selection_noise=selection_noise)
                if batch_selected_prompts.__class__ != list:
                    selected_prompts += (batch_selected_prompts.detach().cpu().tolist())

                context_input += data['context_input']
                persona_list += data['persona_list']
                pred_text += text
                gt_text += data['target']

            clean_preds = []
            for pred in pred_text:
                search_result = re.search('R:|Q:|Summary:|\n|\:', pred)
                if search_result is not None:
                    clean_preds.append(pred[:search_result.span()[0]])
                else:
                    clean_preds.append(pred)
            pred_text = clean_preds
            dist.all_gather_object(dist_pred_text, pred_text)
            dist.all_gather_object(dist_gt_text, gt_text)
            pred_text = reduce(lambda x, y: x + y, dist_pred_text)
            gt_text = reduce(lambda x, y: x + y, dist_gt_text)
            from evaluation import bleu_score, f1_score, normalize_answer
            bleu = bleu_score(pred_text, [gt_text])
            import pickle

            result = {
                'context_input': context_input,
                'persona_list': persona_list,
                'pred_text': pred_text,
                'gt_text': gt_text,
                'bleu': bleu,
            }
            from collections import Counter
            counter = Counter(selected_prompts)
            if deepspeed.comm.get_local_rank() == 0:
                print('bleu: ', bleu)
                with open(save_dir, 'wb') as file:
                    pickle.dump(result, file)
                with open(save_dir.replace('.pkl', '.txt'), 'w', encoding='utf-8') as file:
                    file.write('bleu: ' + str(bleu) + '\n')
                    if len(selected_prompts) > 0:
                        file.write('selected prompt: ' + str(counter) + '\n')
                    for i in range(len(context_input)):
                        if context_input[i].__class__ == list:
                            file.write('context: ' + str(u' '.join(context_input[i]).encode('utf-8')) + '\n')
                        else:
                            file.write('context: ' + str(context_input[i].encode('utf-8')) + '\n')
                        file.write('persona: ' + str(u' '.join(persona_list[i]).encode('utf-8')) + '\n')
                        file.write('pred: ' + pred_text[i] + '\n')
                        file.write('gt: ' + gt_text[i] + '\n')
                        if len(selected_prompts) > 0:
                            file.write('selected prompt: ' + str(selected_prompts[i]) + '\n')
                        file.write('\n')
        except Exception as e:
            save_dir = os.sep.join(model_path.split(os.sep)[:-1]) + os.sep + "test_error.txt"
            print(f'WRITING TESTING ERROR! ERROR: {str(e)}')
            with open(save_dir, 'w') as file:
                file.write(str(e))
        deepspeed.comm.barrier()
    deepspeed.comm.barrier()


model_path_arg = args.model_path
model_paths = [model_path_arg]
if len(glob.glob(model_path_arg+os.sep+'ds_ckpt'+os.sep+'*')):
    model_paths = [model_path_arg+os.sep+'ds_ckpt']
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
# if not model_path_arg.endswith('.pth'):
#     evaluate_folder(model_path_arg, skip_exists=args.skip_exists)
print('Test Ends')
