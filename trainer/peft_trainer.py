import json
import os
import time

import deepspeed
import torch
from pytictoc import TicToc
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import PersonaChatDataset
from utils.dist_helper import setup
from utils.format_inputs import TASK_TYPE
from utils.seed_everything import seed_everything


def save_checkpoint(model, optimizer, config, filename):
    torch.save({
        # 'model_state_dict': model.module.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, filename)


def train_generator(config, batch_size, lr,
                    num_workers,
                    epoch,
                    gradient_clipping, seed, save_model,
                    training_ratio, cmd_args, shuffle_train=True,warmup_config=None,
                    ckpt_path=None):
    with open(cmd_args.deepspeed_config) as json_file:
        ds_config = json.load(json_file)
        del cmd_args.deepspeed_config
        ds_config['train_micro_batch_size_per_gpu'] = batch_size
        ds_config['optimizer']['params']['lr'] = lr
        if config.model.load_bit == 16:
            ds_config['float16']['enabled'] = True
        if config.model.load_bit == 'bf16':
            ds_config['bf16']['enabled'] = True
        if gradient_clipping > 0:
            ds_config['gradient_clipping'] = gradient_clipping

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if config.model.model_type == 'selective_pt':
        from models.selective_llm_chat import SelectLLMChat as LLMChat
    else:
        from models.llm_chat import LLMChat
    seed_everything(seed)
    # initialize the distributed environment
    # time setup function using tictoc
    t = TicToc()
    t.tic()
    setup()
    # print(f"Time for setup is {t.tocvalue()} seconds")
    config.training.learning_rate = float(lr)
    # Create model and move it to GPU

    task_type: str = config.training.task_type
    enum_task = TASK_TYPE(task_type)
    train_dataset = PersonaChatDataset(config.dataset.train, max_context_turns=config.dataset.max_context_turns,
                                       training_ratio=training_ratio,
                                       only_longest=config.training.only_longest,
                                       task_type=enum_task)
    valid_dataset = PersonaChatDataset(config.dataset.valid, max_context_turns=config.dataset.max_context_turns,
                                       task_type=enum_task)
    from dataset.dataset import get_dataloader
    if warmup_config is not None:
        warmup_config["params"]['warmup_num_steps'] = int(len(train_dataset)/batch_size * warmup_config["params"]['warmup_ratio'] / world_size)
        warmup_config["params"]['warmup_num_steps'] = int(len(train_dataset)/batch_size * warmup_config["params"]['warmup_ratio'] / world_size)
        warmup_config["params"]['total_num_steps'] = int(len(train_dataset)/batch_size)/world_size
        del warmup_config["params"]['warmup_ratio']
        ds_config['scheduler'] = warmup_config
    _pt_model = LLMChat(config, batch_size=batch_size, ds_config=ds_config)

    # ddp_model = DDP(_pt_model, device_ids=[0], output_device=0, find_unused_parameters=False)
    left_tokenizer = _pt_model.left_tokenizer
    right_tokenizer = _pt_model.right_tokenizer
    # So there are always training samples
    right_tokenizer.truncation_side = 'left'
    # If it is lengthy, cut the right side
    left_tokenizer.truncation_side = 'right'
    # Create distributed sampler
    all_params = [p for p in _pt_model.parameters()]
    require_grads = [p for p in all_params if p.requires_grad]
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(args=cmd_args,
                                                                        model=_pt_model,
                                                                        model_parameters=require_grads,
                                                                        training_data=train_dataset,
                                                                        config=ds_config,
                                                                        )
    if ckpt_path is not None:
        model_engine.load_checkpoint(ckpt_path, load_module_strict=False, load_optimizer_states=True,
                                         load_lr_scheduler_states=True,
                                         load_module_only=False)

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, shuffle=False,
                                       drop_last=False)

    valid_dataloader = get_dataloader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      sampler=valid_sampler)

    if enum_task in [TASK_TYPE.GENERATE_RESPONSE, TASK_TYPE.GENERATE_PERSONA]:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle=shuffle_train,
                                        drop_last=False)
        train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                        sampler=train_sampler)


    # You might want to adjust this depending on your specific requirements
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if config.training.log_dir.__class__ is str:
        logdir = f"{config.training.log_dir}/{config.exp_name}_{time.strftime('%Y-%m-%d-%H%M')}"
    else:
        logdir = f"runs/{config.exp_name}_{time.strftime('%Y-%m-%d-%H%M')}"
    # Tensorboard logger
    writer = SummaryWriter(log_dir=logdir)
    best_valid_loss = 65535
    # Training Loop
    counter = 0
    valid_counter = 0
    for _epoch in range(epoch):
        model_engine.train()
        total_loss = 0.0
        gathered_train_loss = [torch.zeros(1, dtype=torch.float32, device=model_engine.device) for _ in range(world_size)]
        train_iter = tqdm(train_dataloader, total=len(train_dataloader), desc=f'epoch: {_epoch}')
        total_steps_per_epoch = len(train_dataloader)
        total_steps = total_steps_per_epoch*epoch
        for idx, inputs in enumerate(train_iter):
            current_step = idx+_epoch*total_steps_per_epoch
            current_training_percent = current_step/total_steps
            model_engine.zero_grad()
            loss = LLMChat.training_step(model_engine, inputs, left_tokenizer, right_tokenizer, config,
                                          mode=config.training.mode, task_type=enum_task, training_process=current_training_percent)
            skipped = False
            params = []
            if deepspeed.comm.get_local_rank() in [-1, 0]:
                for n, p in model_engine.named_parameters():
                    if p.requires_grad:
                        params.append(p)
                norm = torch.stack([p.norm() for p in params]).sum()
                print(f'NORM: {norm}')
            if loss.isnan():
               model_engine.backward(loss.new_zeros(loss.shape, requires_grad=True))
               skipped = True
               print(inputs)
               raise ValueError('Meet NaN in training!')
            else:
                model_engine.backward(loss)
                if gradient_clipping > 0:
                    model_engine.gradient_clipping()

            model_engine.step()

            total_loss += loss.item()
            writer.add_scalar(f'Loss-{deepspeed.comm.get_local_rank()}/train', loss.item(), counter)
            counter += 1
            train_iter.set_postfix_str(f'loss: {loss.item()}'+(" (Skipped)" if skipped else ""))
        outputs_valid_losses = [torch.zeros(1, dtype=torch.float32, device=model_engine.device) for _ in range(world_size)]
        valid_loss = []
        for inputs in tqdm(valid_dataloader, total=len(valid_dataloader), desc='valid'):
            model_engine.eval()
            with torch.no_grad():
                loss = LLMChat.validation_step(model_engine, inputs, left_tokenizer, right_tokenizer, config,
                                               mode=config.training.mode, task_type=enum_task)
                valid_loss.append(loss.item())
                writer.add_scalar(f'Loss-{deepspeed.comm.get_local_rank()}/valid', loss.item(), valid_counter)
                valid_counter += 1
        deepspeed.comm.all_gather(outputs_valid_losses, torch.tensor(valid_loss).mean().to(model_engine.device))
        gathered_valid_loss = torch.stack(outputs_valid_losses).mean()
        deepspeed.comm.all_gather(gathered_train_loss, torch.tensor(total_loss / len(train_dataloader), device=model_engine.device))
        writer.add_scalar(f'Loss-{deepspeed.comm.get_local_rank()}/total_train',   torch.stack(gathered_train_loss).mean(), _epoch)

        writer.add_scalar(f'Loss-{deepspeed.comm.get_local_rank()}/total_valid', gathered_valid_loss, _epoch)
        deepspeed.comm.barrier()
        print(
            f'\nepoch: {_epoch}, train_loss: {total_loss / len(train_dataloader)}, valid_loss: {gathered_valid_loss}\n')
        if best_valid_loss > gathered_valid_loss and save_model:
            # Save pt_model checkpoint
            if model_engine.global_rank == 0:
                print(f"Saving model checkpoint with valid loss {gathered_valid_loss}")
                save_checkpoint(model_engine, optimizer, config, f'{logdir}/checkpoint_best.pth')
            model_engine.save_checkpoint(f'{logdir}/ds_ckpt', tag='best', exclude_frozen_parameters=True)
            best_valid_loss = gathered_valid_loss


    deepspeed.comm.destroy_process_group()
