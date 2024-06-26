import argparse
import os
from multiprocessing import freeze_support

import deepspeed
import torch
from dotenv import load_dotenv
from transformers.utils import logging

from trainer.peft_trainer import train_generator
from utils.config import get_config
from utils.parser_helper import str2bool

load_dotenv()
logging.set_verbosity_error()
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('medium')


def set_model_config(the_config, value, key):
    if value is not None:
        the_config.model[key] = value


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--find_batch', type=str2bool, default=False)
parser.add_argument('--find_lr', type=str2bool, default=False)
parser.add_argument('--bf16', type=str2bool, default=True)
parser.add_argument('--auto_scale_batch_size', type=str2bool, default=False)
parser.add_argument('--train_after_tune', type=str2bool, default=False)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--scheduler_patience', type=int, default=10)
parser.add_argument('--scheduler_monitor', type=str, default='train_loss', choices=['train_loss'])
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--grad_clip', type=float, default=-1)
parser.add_argument('--save_model', type=str2bool, default=True)
parser.add_argument('--shuffle_train', type=str2bool, default=True)
parser.add_argument('--training_ratio', type=float, default=1.0)
parser.add_argument('--adding_noise', type=float, default=None)

# parser.add_argument('--retriever_type', type=str, default=None,
#                     choices=['bert-base-uncased', 'albert-base-v2'])

parser.add_argument('--tokenizer_parallel', type=str2bool, default=True)
parser.add_argument('--do_test', type=str2bool, default=False)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--mode', default=None, type=str, choices=['normal', 'causal', None])
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser.add_argument('--selective_loss_weight', type=float, default=None)
parser.add_argument('--contrastive_weight', type=float, default=None)
parser.add_argument('--log_dir', type=str, default=None)

parser.add_argument('--warmup_type', type=str, default=None)
parser.add_argument('--warmup_min', type=float, default=0)
parser.add_argument('--warmup_ratio', type=float, default=0.05)

parser.add_argument('--ckpt_path', type=str, default=None)




parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()

freeze_support()
args = parser.parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.tokenizer_parallel else "false"
config = get_config(args.config)
if args.exp_name is not None:
    config.exp_name = args.exp_name
elif config.exp_name.__class__ != str:
    config.exp_name = args.config.split(os.sep)[-1][:-4]
if args.lr is not None:
    config.exp_name += f'_LR={args.lr}'
if args.selective_loss_weight is not None:
    config.training.selective_loss_weight = args.selective_loss_weight
    config.exp_name += f'_SLW={args.selective_loss_weight}'
if args.contrastive_weight is not None:
    config.training.contrastive_weight = args.contrastive_weight
    config.exp_name += f'_CTRW={args.contrastive_weight}'
if args.adding_noise is not None:
    config.training.adding_noise = args.adding_noise
    config.exp_name += f'_NOISE={args.adding_noise}'
if args.training_ratio < 1.0:
    config.exp_name += f'_PTRAIN={args.training_ratio}'
# Done model config
generator_type = config.model.generator_type
if args.mode is not None:
    config.training.mode = args.mode
if args.epoch is not None:
    config.training.num_epoch = args.epoch
epoch = config.training.num_epoch
if args.log_dir is not None:
    config.training.log_dir = args.log_dir
if 'llama-2' in config.model.model_name.lower():
    folder_name = config.model.model_name.split('/')[-1]
    config.model.model_name = os.getenv('LLAMA2_PATH')+'/'+folder_name
warmup_config = None
if args.warmup_type is not None:
    warmup_config = {
        "type": args.warmup_type,
        "params": {
            # "warmup_min_lr": args.warmup_min,
            # "warmup_max_lr": args.lr,
            "warmup_ratio": args.warmup_ratio
        }
    }
    config.exp_name += f'_WP={args.warmup_type}@{args.warmup_ratio}'
if __name__ == '__main__':
    train_generator(config, args.batch, args.lr, args.num_workers,
                                    epoch, args.grad_clip, args.seed, args.save_model,
                                    args.training_ratio, cmd_args=cmd_args, shuffle_train=args.shuffle_train,
                    warmup_config=warmup_config, ckpt_path=args.ckpt_path)
    # num_of_gpus = torch.cuda.device_count()
    # print(f"{num_of_gpus} GPUs available")
    # mp.spawn(train_generator, args=(config, args.batch, args.lr, args.num_workers,
    #                                 epoch, args.grad_clip, num_of_gpus, args.seed, args.save_model,
    #                                 args.training_ratio), nprocs=num_of_gpus)

# train_generator(args.local_rank, config,
#                 batch_size=args.batch,
#                 lr=args.lr,
#                 num_workers=args.num_workers,
#                 epoch=args.epoch,
#                 gradient_clipping=args.grad_clip)
