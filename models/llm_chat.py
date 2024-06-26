import torch
from peft import get_peft_model, LoraConfig, PromptTuningConfig, TaskType, PrefixTuningConfig
from torch import nn, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from utils.format_inputs import TASK_TYPE
from utils.format_inputs import format_causal_personachat_input, format_personachat_input, \
    format_generate_persona_input
from utils.model_helpers import print_trainable_parameters


class LLMChat(nn.Module):
    def __init__(self, config, batch_size, ds_config=None):
        if ds_config is not None:
            _hfdsc = HfDeepSpeedConfig(ds_config)
        super(LLMChat, self).__init__()
        self.model_name = config.model.model_name
        self.load_bit = config.model.load_bit
        self.left_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        original_vocab_size = len(self.left_tokenizer)
        if config.training.mode != 'causal':
            self.left_tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                                    'bos_token': '[BOS]',
                                                    'eos_token': '[EOS]',
                                                    'unk_token': '[UNK]',
                                                    'sep_token': '[SEP]',
                                                    'cls_token': '[CLS]',
                                                    'mask_token': '[MASK]'})
        self.left_tokenizer.padding_side = 'left'
        self.left_tokenizer.truncation_side = 'left'
        self.right_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if config.training.mode != 'causal':
            self.right_tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                                     'bos_token': '[BOS]',
                                                     'eos_token': '[EOS]',
                                                     'unk_token': '[UNK]',
                                                     'sep_token': '[SEP]',
                                                     'cls_token': '[CLS]',
                                                     'mask_token': '[MASK]'})
        self.right_tokenizer.padding_side = 'right'
        self.right_tokenizer.truncation_side = 'right'
        if self.left_tokenizer.pad_token is None and config.model.pad_token == 'bos':
            self.left_tokenizer.pad_token = self.left_tokenizer.bos_token
            self.right_tokenizer.pad_token = self.right_tokenizer.bos_token
        elif self.left_tokenizer.pad_token_id is None:
            self.left_tokenizer.pad_token = self.left_tokenizer.eos_token
            self.right_tokenizer.pad_token = self.right_tokenizer.eos_token
        self.batch_size = batch_size
        load_bit_map = {4: {'load_in_4bit': True,
                            'bnb_4bit_compute_dtype': torch.bfloat16},
                        8: {'load_in_8bit': True},
                        16: {'torch_dtype': torch.float16},
                        32: {'torch_dtype': torch.float32}}
        assert config.model.load_bit in [16, 32], 'deepspeed is not friendly with bnb!'
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            **load_bit_map[config.model.load_bit],
        )
        if config.training.mode != 'causal':
            model.resize_token_embeddings(len(self.left_tokenizer))
        # for m in model.children():
        #     if hasattr(m, 'gradient_checkpointing_enable'):
        #         m.gradient_checkpointing_enable()
        model.gradient_checkpointing_enable()
        if config.model.peft_config is not None:
            for param in model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later
                if param.ndim == 1:
                    # cast the small parameters (e.g. layernorm) to fp32 for stability
                    param.data = param.data.to(torch.float32)
            model.enable_input_require_grads()

            # # enable special token embedding params, since we resized the vocabulary
            # for name, param in model.named_parameters():
            #     if 'embed_tokens' in name:
            #         param[original_vocab_size:].requires_grad = True

            class CastOutputToFloat(nn.Sequential):
                def forward(self, x): return super().forward(x).to(torch.float32)

            if config.model.peft_type == 'prompt_tuning':
                peft_config = PromptTuningConfig(
                    **config.model.peft_config,
                    task_type=TaskType.CAUSAL_LM,
                )
            elif config.model.peft_type == 'prefix_tuning':
                peft_config = PrefixTuningConfig(
                    **config.model.peft_config,
                    task_type=TaskType.CAUSAL_LM,
                )
            else:
                peft_config = LoraConfig(**config.model.peft_config)
            model.lm_head = CastOutputToFloat(model.lm_head)
            model = get_peft_model(model, peft_config)
        self.using_nn_modulelist = False
        if config.model.using_nn_modulelist.__class__ is bool and config.model.using_nn_modulelist:
            self.using_nn_modulelist = config.model.using_nn_modulelist
            self.model = nn.ModuleList([model])
        else:
            self.model = model
        if config.model.add_extra_layers.__class__ is bool and config.model.add_extra_layers:
            self.prompt_normalizer = nn.Linear(
                self.model[0].prompt_encoder.default.embedding.weight.shape[1],
                self.model[0].word_embeddings.weight.shape[1])
            self.score_activation = nn.Softplus(threshold=1, beta=10)
        self.learning_rate = config.training.learning_rate
        self.warmup_steps = config.training.warmup_steps
        self.config = config
        self.find_batch = False
        print_trainable_parameters(self)

    def print_llm_trainable_parameters(self):
        print_trainable_parameters(self.model)

    @autocast('cuda')
    def forward(self, x):
        if self.config._non_exists == 1:
            self.prompt_normalizer(x)
            self.score_activation(x)
        for k in x.keys():
            x[k] = x[k].cuda()
        if self.find_batch:
            x['attention_mask'] = x['attention_mask'].new_ones(x['attention_mask'].shape)
        if self.using_nn_modulelist:
            if self.config.model.using_output_stack.__class__ is bool and self.config.model.using_output_stack:
                _outputs = [_model(**x) for _model in self.model]
                _logits = torch.stack([_output['logits'] for _output in _outputs])
                return {'logits': _logits}
            return self.model[0](**x)
        return self.model(**x)

    def on_train_start(self) -> None:
        self.print_llm_trainable_parameters()

    @staticmethod
    def training_step(model, batch, left_tokenizer, right_tokenizer, config, find_batch=False, mode='normal',
                      task_type=TASK_TYPE.GENERATE_RESPONSE, **_kwargs):
        assert mode in ['normal', 'causal']
        if task_type == TASK_TYPE.GENERATE_PERSONA and mode == 'normal':
            lm_input, lm_target = format_generate_persona_input(batch, left_tokenizer, right_tokenizer,
                                                                config)
        elif task_type == TASK_TYPE.GENERATE_RESPONSE and mode == 'causal':
            lm_input, lm_target = format_causal_personachat_input(batch, left_tokenizer, right_tokenizer,
                                                                  config)
        elif task_type == TASK_TYPE.GENERATE_RESPONSE and mode == 'normal':
            lm_input, lm_target = format_personachat_input(batch, left_tokenizer, right_tokenizer, config)
        else:
            raise NotImplementedError('mode and task_type not implemented')
        output = model(lm_input)
        if find_batch:
            loss = nn.CrossEntropyLoss()(output['logits'].view(-1, output['logits'].shape[-1]),
                                         lm_target.cuda().view(-1))
        else:
            if config.model.peft_type == 'prompt_tuning':
                virtual_tokens = config.model.peft_config.num_virtual_tokens
                batch_size = lm_target.size()[0]
                _lm_target = torch.cat(
                    (lm_target.new_ones((batch_size, virtual_tokens)) * left_tokenizer.pad_token_id, lm_target), dim=1)
            else:
                _lm_target = lm_target
            loss = nn.CrossEntropyLoss(ignore_index=left_tokenizer.pad_token_id)(
                output['logits'].view(-1, output['logits'].shape[-1]),
                _lm_target.cuda().view(-1))
        # self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        if config.training.normalize_loss.__class__ == bool and config.training.normalize_loss.__class__:
            model.module.normalize()
        return loss

    def normalize(self):
        raise NotImplementedError('normalize trainable weights needs implementation')
        return None

    @staticmethod
    def validation_step(model, batch, left_tokenizer, right_tokenizer, config, task_type, mode='normal'):
        loss = LLMChat.training_step(model, batch, left_tokenizer, right_tokenizer, config, task_type=task_type,
                                     find_batch=False, mode=mode)
        return loss

    def on_test_start(self) -> None:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        peft_weight = get_peft_model_state_dict(self.model).copy()
        peft_config = self.model.peft_config
        del self.model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        self.model = get_peft_model(model, peft_config['default'])
        set_peft_model_state_dict(self.model, peft_weight, adapter_name='default')
        self.model.merge_and_unload()
        self.model.eval()

    @staticmethod
    @autocast('cuda')
    def test_step(model, batch, left_tokenizer, right_tokenizer, config, max_new_tokens=16, tqdm_instance=None, **kwargs):
        model.eval()
        task_type = TASK_TYPE(config.training.task_type)
        with torch.no_grad():
            if config.training.mode == 'causal':
                lm_input, lm_target, inference_tokenized = format_causal_personachat_input(batch,
                                                                                           left_tokenizer,
                                                                                           right_tokenizer,
                                                                                           config,
                                                                                           for_test=True)
            else:
                lm_input, lm_target, inference_tokenized = format_personachat_input(batch, left_tokenizer,
                                                                                    right_tokenizer, config,
                                                                                    for_test=True)
            inference_tokenized.to('cuda')
            model_for_generation = None
            if 'deepspeed' in str(model.__class__):
                model_for_generation = model.module.model
            else:
                model_for_generation = model.model
            if model_for_generation.__class__ is nn.ModuleList:
                model_for_generation = model_for_generation[0]
            # adding do_sample=False to avoid inf error!
            raw_output = model_for_generation.generate(**inference_tokenized, max_new_tokens=max_new_tokens,
                                                       do_sample=False)
            trunc_output = raw_output[:, inference_tokenized['input_ids'].shape[1]:]
            if trunc_output[trunc_output >= len(left_tokenizer)].size()[0] > 0:
                trunc_output[trunc_output >= len(left_tokenizer)] = left_tokenizer.pad_token_id
            text_output = right_tokenizer.batch_decode(trunc_output, skip_special_tokens=True)
            return trunc_output, text_output, []
