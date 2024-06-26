import deepspeed
import deepspeed
import torch
import transformers
from peft import get_peft_model, PromptTuningConfig, TaskType, PrefixTuningConfig
from torch import nn, autocast
from torch.functional import F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from utils.format_inputs import TASK_TYPE
from utils.format_inputs import format_causal_personachat_input, format_personachat_input, format_generate_persona_input
from utils.model_helpers import print_trainable_parameters


class SelectLLMChat(nn.Module):
    def __init__(self, config, batch_size, ds_config=None):
        super(SelectLLMChat, self).__init__()
        if ds_config is not None:
            _hfdsc = HfDeepSpeedConfig(ds_config)
        peft_type = config.model.peft_type
        self.peft_type = peft_type
        assert config.model.peft_type in ['prompt_tuning', 'prefix_tuning',
                                          ], 'only prompt tuning is supported!'
        K = config.model.K
        self.K = K
        self.ensemble_training = config.training.ensemble
        self.model_name = config.model.model_name
        self.load_bit = config.model.load_bit
        self.left_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
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
        if self.left_tokenizer.pad_token is None and config.model.pad_token=='bos':
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
                        'bf16': {'torch_dtype': torch.bfloat16},
                        32: {'torch_dtype': torch.float32}}
        assert config.model.load_bit in [16, 32, 'bf16'], 'deepspeed is not friendly with bnb!'
        model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                **load_bit_map[config.model.load_bit]
            )
        if config.training.mode != 'causal':
            model.resize_token_embeddings(len(self.left_tokenizer))
        model.gradient_checkpointing_enable()
        if config.model.peft_config is not None:
            for param in model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later
                if param.ndim == 1:
                    # cast the small parameters (e.g. layernorm) to fp32 for stability
                    param.data = param.data.to(torch.float32)
            model.enable_input_require_grads()
        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)

        model.lm_head = CastOutputToFloat(model.lm_head)
        self.model = model
        models = []
        peft_config = None
        for _ in range(K):
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
                raise NotImplementedError()
            _peft_model = get_peft_model(model, peft_config)
            models.append(_peft_model)
        self.models = nn.ModuleList(models)
        self.learning_rate = config.training.learning_rate
        self.warmup_steps = config.training.warmup_steps
        self.config = config
        self.find_batch = False
        self.retriever = None
        if config.model.retriever.retriever_type == 'transformer_encoder':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.models[0].word_embeddings.weight.shape[1],
                                                       nhead=config.model.retriever.n_head)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.model.retriever.num_layers)
            self.retriever = transformer_encoder
        if config.model.peft_type in ['prompt_tuning'] and config.model.normalizer.__class__ is not str:
            class DoNothing(nn.Sequential):
                def forward(self, x): return x

            self.prompt_normalizer = DoNothing()
        elif config.model.normalizer == 'linear':
            if config.model.peft_type in ['prompt_tuning', 'prefix_tuning']:
                _d_peft = self.models[0].prompt_encoder.default.embedding.weight.shape[1]
            else:
                raise NotImplementedError('check here!')
            self.prompt_normalizer = nn.Linear(_d_peft, _d_peft)

        if config.model.score_activation == 'softplus':
            self.score_activation = nn.Softplus(threshold=1, beta=10)
        elif config.model.score_activation == 'relu':
            self.score_activation = nn.ReLU()
        elif config.model.score_activation == 'leaky_relu':
            self.score_activation = nn.LeakyReLU()
        else:
            self.score_activation = nn.Softplus(threshold=1, beta=10)
            # raise NotImplementedError()
        self.retriever_on = ['extra']
        if config.model.retriever.retriever_on.__class__ is list:
            self.retriever_on = config.model.retriever.retriever_on
        if config.training.all_tunable.__class__ is bool and config.training.all_tunable:
            for param in self.parameters():
                param.requires_grad = True
        print_trainable_parameters(self)
        self.contrastive_metric = None
        if config.training.contrastive_metric.__class__ is str:
            self.contrastive_metric = config.training.contrastive_metric
        self.contrastive_threshold = 0.0
        if config.training.contrastive_threshold.__class__ is float:
            self.contrastive_threshold = config.training.contrastive_threshold
        self.config = config
        self.annealing_nll = False
        self.annealing_scalar = 0.0
        if self.config.training.annealing_nll.__class__ == bool:
            self.annealing_nll = self.config.training.annealing_nll
            self.annealing_scalar = self.config.training.annealing_scalar


    def print_llm_trainable_parameters(self):
        print_trainable_parameters(self.model)

    def retrieve_based_on_input_x(self, x, K):
        return self.retrieve_prompts(x, K)

    @autocast('cuda')
    def retrieve_prompts(self, x, K):
        batch_size = x['input_ids'].shape[0]
        input_ids = x['input_ids']
        spawned_x = input_ids.repeat(K, 1)
        if self.models[0].base_model.__class__ == transformers.models.llama.modeling_llama.LlamaForCausalLM:
            spawned_x_emb = self.models[0].base_model.model.embed_tokens(spawned_x)
        else:
            spawned_x_emb = self.models[0].base_model.model.decoder.embed_tokens(spawned_x)
        if spawned_x_emb.shape[-1] != self.models[0].config.hidden_size:
            # need project_in here
            spawned_x_emb = self.models[0].base_model.model.decoder.project_in(spawned_x_emb)
        prompt_embeddings = torch.stack([_model.prompt_encoder.default.embedding.weight for _model in self.models])
        if self.retriever is not None:
            if 'extra' in self.retriever_on:
                prompt_embeddings = self.retriever(self.prompt_normalizer(prompt_embeddings))
            if 'lm' in self.retriever_on:
                spawned_x_emb = self.retriever(spawned_x_emb)
        spawned_x_emb_mean = spawned_x_emb.mean(dim=1)
        prompt_embeddings_mean = prompt_embeddings.mean(dim=1)
        if self.retriever is None:
            normalizer_on = self.config.model.normalizer_on
            if normalizer_on.__class__ is not list:
                prompt_embeddings_mean = self.prompt_normalizer(prompt_embeddings_mean)
            if 'prompt' in normalizer_on:
                prompt_embeddings_mean = self.prompt_normalizer(prompt_embeddings_mean)
            if 'lm' in normalizer_on:
                spawned_x_emb_mean = self.prompt_normalizer(spawned_x_emb_mean)
        prompt_embeddings_mean_spawn = torch.repeat_interleave(prompt_embeddings_mean, batch_size, dim=0)
        sim_scores = self.score_activation(
            torch.nn.CosineSimilarity()(prompt_embeddings_mean_spawn, spawned_x_emb_mean))
        return sim_scores

    @autocast('cuda')
    def forward(self, x, mode='training'):
        for k in x.keys():
            x[k] = x[k].cuda(device=deepspeed.comm.get_local_rank())
        if self.find_batch:
            x['attention_mask'] = x['attention_mask'].new_ones(x['attention_mask'].shape)
        if mode == 'training':
            if self.config.training.skip_retrieval.__class__ is bool and self.config.training.skip_retrieval:
                sim_scores = None
            else:
                sim_scores = self.retrieve_based_on_input_x(x, self.K)
            # get pt embeddings
            _outputs = [_model(**x) for _model in self.models]
            _logits = torch.stack([_output['logits'] for _output in _outputs])
            return {'logits': _logits, 'sim_scores': sim_scores}
        else:
            raise NotImplementedError('validation and testing not implemented')

    def on_train_start(self) -> None:
        self.print_llm_trainable_parameters()
        deepspeed.zero.Init()

    @staticmethod
    def training_step(model, batch, left_tokenizer, right_tokenizer, config, mode='normal',
                      task_type=TASK_TYPE.GENERATE_RESPONSE, training_process=0.0):
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
        output = model.module(dict(lm_input))
        # suppose batch=2, K=3, the logits is presented interleave:
        # [0,1]
        # [0,1]
        # [0,1]
        logits = output['logits']  # (K*Batch,SeqLen,VocabSize)
        logits = logits.view(-1, logits.shape[2], logits.shape[3])
        sim_scores = output['sim_scores']
        batch_size = lm_target.size()[0]
        if config.model.peft_type == 'prompt_tuning':
            virtual_tokens = config.model.peft_config.num_virtual_tokens
            _lm_target = torch.cat(
                (lm_target.new_ones((batch_size, virtual_tokens)) * left_tokenizer.pad_token_id, lm_target), dim=1)
        else:
            _lm_target = lm_target
        _lm_target_spawn = _lm_target.repeat(config.model.K, 1)
        losses = nn.CrossEntropyLoss(ignore_index=left_tokenizer.pad_token_id, reduction='none')(
            logits.view(-1, logits.shape[-1]),
            _lm_target_spawn.cuda(device=deepspeed.comm.get_local_rank()).view(-1))
        if config.training.only_nll.__class__ == bool and config.training.only_nll:
            return losses[losses != 0].mean()

        reshaped_losses = losses.view(logits.shape[0], logits.shape[1]).detach().clone()
        reshaped_losses = torch.stack([_losses[_losses != 0].mean() for _losses in reshaped_losses.detach().clone()])
        # reshaped_losses = reshaped_losses.clone().detach().mean(dim=1)

        softmaxed_neg_losses = nn.Softmax(dim=0)(
            -reshaped_losses.view(config.model.K, batch_size) / config.training.tau_gold).permute(1, 0)
        if config.training.adding_noise.__class__ is float:
            noise = torch.randn_like(softmaxed_neg_losses, device=softmaxed_neg_losses.device)
            softmaxed_neg_losses = softmaxed_neg_losses + config.training.adding_noise * noise
        logsoftmaxed_sim_scores = F.log_softmax(sim_scores.view(config.model.K, batch_size) / config.training.tau_sim,
                                                dim=0).permute(1, 0)
        kldiv_loss = nn.KLDivLoss(reduction='batchmean')(logsoftmaxed_sim_scores,
                                                         softmaxed_neg_losses)
        selective_loss_weight = 1.0
        if config.training.annealing_nll.__class__ is bool and config.training.annealing_nll:
            _ann_scalar = config.training.annealing_scalar * (1 - training_process)
            _sim_score = torch.clamp(_ann_scalar * nn.Softmax(-1)(sim_scores),
                                     config.training.annealing_min, config.training.annealing_max).detach()
            losses = torch.einsum('ab,a->ab', losses.view(logits.shape[0], logits.shape[1]), _sim_score).view(-1)

        if config.training.selective_loss_weight.__class__ == float:
            selective_loss_weight = config.training.selective_loss_weight
        if config.training.selective_loss.__class__ == bool and (config.training.selective_loss == False):
            loss = losses[losses != 0].mean()
        elif config.training.disable_nll.__class__ is bool and config.training.disable_nll:
            loss = selective_loss_weight * kldiv_loss
        else:
            loss = losses[losses != 0].mean() + selective_loss_weight * kldiv_loss

        if model.module.ensemble_training:
            K = config.model.K
            enb_losses = []
            for data_idx in range(batch_size):
                data_indices = [data_idx + (batch_size * inc) for inc in range(K)]
                ensemble_preds = logits[data_indices, :, :]
                ensemble_sims = sim_scores[data_indices]
                normed_preds = ensemble_sims.unsqueeze(-1).unsqueeze(-1).mul(ensemble_preds)
                normed_preds = normed_preds.sum(dim=0)
                _target = _lm_target_spawn[data_indices, :]
                assert _target.unique(dim=0).shape[0] == 1, 'error in resemble the preds'
                enb_loss = nn.CrossEntropyLoss(ignore_index=left_tokenizer.pad_token_id)(normed_preds,
                                                                                         _target[0].cuda(
                                                                                             device=deepspeed.comm.get_local_rank()))
                enb_losses.append(enb_loss)
            loss += torch.stack(enb_losses).mean()
        if model.module.contrastive_metric:
            ctr_losses = []
            from sacrebleu import BLEU
            ctr_metrics = BLEU(effective_order=True)
            batch_persona = [' '.join(row) for row in batch['persona_list']]
            statics = {}
            # Dim here
            #     x1 x2
            # p1 s11 s21
            # p2 s12 s22
            # p3 s13 s23
            permuted_sim_scores = sim_scores.unsqueeze(0).view(model.module.K, batch_size)
            if model.module.contrastive_metric == 'bleu':
                for idx in range(len(batch_persona) - 1):
                    for jdx in range(idx + 1, len(batch_persona)):
                        iele = batch_persona[idx]
                        jele = batch_persona[jdx]
                        scores = ctr_metrics.sentence_score(iele, [jele]).score
                        idist = permuted_sim_scores[:, idx]
                        jdist = permuted_sim_scores[:, jdx]
                        cosine_emb_loss = nn.CosineEmbeddingLoss()
                        if scores > model.module.contrastive_threshold:
                            cosine_target = 1
                        else:
                            cosine_target = -1
                        cos_loss = cosine_emb_loss(idist, jdist, torch.tensor(cosine_target))
                        ctr_losses.append(cos_loss)
                        statics[(idx, jdx)] = {'iele': iele, 'jele': jele, 'scores': scores,
                                               'idist': idist,
                                               'jdist': jdist, 'cos_emb_loss': cos_loss}
            if len(ctr_losses) != 0:
                ctr_losses_pt = torch.stack(ctr_losses).mean()
                loss += config.training.contrastive_weight * ctr_losses_pt
            else:
                print(f'CTR ERROR: {statics}')
        return loss

    @staticmethod
    def validation_step(model, batch, left_tokenizer, right_tokenizer, config, task_type, mode='normal'):
        loss = SelectLLMChat.training_step(model, batch, left_tokenizer, right_tokenizer, config, task_type=task_type,
                                           mode=mode, training_process=0.0)
        return loss

    @staticmethod
    @autocast('cuda')
    def test_step(model, batch, left_tokenizer, right_tokenizer, config, max_new_tokens=16, tqdm_instance: tqdm = None,
                  selection_noise=None, **gen_kwargs):
        model.eval()
        with torch.no_grad():
            if config.training.mode == 'causal':
                lm_input, lm_target, inference_tokenized = format_causal_personachat_input(batch,
                                                                                           left_tokenizer,
                                                                                           right_tokenizer,
                                                                                           config,
                                                                                           for_test=True)
            else:
                lm_input, lm_target, inference_tokenized = format_personachat_input(batch, left_tokenizer,
                                                                                    right_tokenizer,
                                                                                    config,
                                                                                    for_test=True)
            inference_tokenized.to('cuda')
            if 'deepspeed' in str(model.__class__):
                batch_size = inference_tokenized['input_ids'].shape[0]
                sim_scores = model.module.retrieve_based_on_input_x(inference_tokenized, config.model.K)
                sim_scores = sim_scores.reshape(config.model.K, batch_size).permute(1, 0)
                if selection_noise:
                    noise = torch.randn_like(sim_scores, device=sim_scores.device)
                    sim_scores = sim_scores + selection_noise * noise
                selected_prompts = torch.argmax(sim_scores, dim=1)
                if tqdm_instance is not None:
                    tqdm_instance.set_postfix_str(f"selected prompts: {selected_prompts}")
                detached_selected_prompts = selected_prompts.detach().cpu().numpy()
                selected_prompts_set = set(detached_selected_prompts)
                output_dicts = {}
                # adding do_sample=False to avoid inf error!
                for key in selected_prompts_set:
                    outputs = model.module.models[key].generate(
                        input_ids=inference_tokenized['input_ids'],
                        attention_mask=inference_tokenized['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        **gen_kwargs
                    )
                    output_dicts[key] = outputs.detach().cpu()
                raw_output = []
                for idx, prompt_idx in enumerate(detached_selected_prompts):
                    raw_output.append(output_dicts[prompt_idx][idx][inference_tokenized['input_ids'].shape[1]:])
                # raw_output = torch.stack(raw_output).squeeze(1)
                trunc_output = raw_output
                text_output = right_tokenizer.batch_decode(trunc_output, skip_special_tokens=True)
                return trunc_output, text_output, selected_prompts
            else:
                raise NotImplementedError('not implemented')
