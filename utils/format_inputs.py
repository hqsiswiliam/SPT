from enum import Enum

import torch


class TASK_TYPE(Enum):
    GENERATE_RESPONSE = 'generate_response'
    GENERATE_PERSONA = 'generate_persona'



def format_personachat_input(batch, left_tokenizer, right_tokenizer, config, for_test=False, find_batch=False):
    batch_size = len(batch['context_input'])
    pad_token_id = left_tokenizer.pad_token_id
    targets = [t.strip() for t in batch['target']]
    eos_token = left_tokenizer.eos_token
    concat_context = [' '.join(context) for context in batch['context_input']]
    concat_persona = [' '.join(persona) for persona in batch['persona_list']]
    concat_input = [f'#persona#{persona}#context#{context}' for persona, context in
                    zip(concat_persona, concat_context)]
    inference_tokenized = None
    bos_token = left_tokenizer.bos_token
    if for_test:
        inference_input = [f'#persona#{persona}#context#{context}{bos_token}' for persona, context in
                           zip(concat_persona, concat_context)]
        inference_tokenized = left_tokenizer(inference_input, add_special_tokens=False, return_tensors='pt',
                                             padding='max_length', truncation=True,
                                             max_length=config.dataset.max_token_length - 16)
    # processing target
    _target_with_bos = [f'{bos_token}{target}{eos_token}' for target in targets]
    _target_with_bos_pt = right_tokenizer(_target_with_bos,
                                          add_special_tokens=False, return_tensors='pt', \
                                          padding=True)
    _target_pt = _target_with_bos_pt.copy()
    _target_pt['input_ids'] = torch.cat((_target_pt['input_ids'][:, 1:],
                                         _target_pt['input_ids'].new_ones(batch_size, 1) * pad_token_id), dim=1)
    _target_pt['attention_mask'] = torch.cat((_target_pt['attention_mask'][:, 1:],
                                              _target_pt['attention_mask'].new_zeros(batch_size, 1)), dim=1)
    # processing concat
    context_pt = left_tokenizer(concat_input, add_special_tokens=False, return_tensors='pt',
                                padding='max_length', truncation=True,
                                max_length=config.dataset.max_token_length)
    input_pt = torch.cat((context_pt['input_ids'], _target_with_bos_pt['input_ids']),
                         dim=1)[:, -config.dataset.max_token_length:]
    input_attn = torch.cat((context_pt['attention_mask'], _target_with_bos_pt['attention_mask']),
                           dim=1)[:, -config.dataset.max_token_length:]
    lm_input = {'input_ids': input_pt, 'attention_mask': input_attn}
    if find_batch:
        lm_target = torch.cat((context_pt['input_ids'],
                               _target_pt['input_ids']), dim=1)[:, -config.dataset.max_token_length:]
    else:
        lm_target = torch.cat((context_pt['input_ids'] * 0 - 1,
                               _target_pt['input_ids']), dim=1)[:, -config.dataset.max_token_length:]
    if for_test:
        return lm_input, lm_target, inference_tokenized
    return lm_input, lm_target


# Template Type:
# 0: </s>

def format_causal_personachat_input(batch, left_tokenizer, right_tokenizer, config, for_test=False,
                                    find_batch=False, template_type=0):
    template_types = [
        '{cinput} R: {target}',
        '{cinput} R: [COMPLETE] the answer for [COMPLETE] is {target}'
    ]
    bos_token = left_tokenizer.bos_token
    eos_token = left_tokenizer.eos_token
    batch_size = len(batch['context_input'])
    pad_token_id = right_tokenizer.pad_token_id
    targets = [t.strip() for t in batch['target']]
    concat_context = [' '.join(context) for context in batch['context_input']]
    concat_persona = [' '.join(persona) for persona in batch['persona_list']]
    concat_input = [f'given persona: {persona}; context: {context}' for persona, context in
                    zip(concat_persona, concat_context)]
    concat_input_target = [template_types[template_type].format(cinput=cinput, target=target) for cinput, target in
                           zip(concat_input, targets)]
    bos_concat_input = [f'{bos_token}{cinput}{eos_token}' for cinput in concat_input_target]
    lm_input = right_tokenizer(bos_concat_input, add_special_tokens=False, return_tensors='pt',
                               padding='max_length', truncation=True,
                               max_length=config.dataset.max_token_length)
    lm_target = lm_input.copy()
    lm_target = torch.cat((lm_target['input_ids'][:, 1:], lm_target['input_ids'].new_full(
        (batch_size, 1), pad_token_id)), dim=1)
    # lm_target['attention_mask'] = torch.cat(
    #     (lm_target['attention_mask'][:, 1:], lm_target['attention_mask'].new_full(
    #         (batch_size, 1), 0)), dim=1)
    # freeze persona
    if config.training.freeze_persona.__class__ is bool and config.training.freeze_persona:
        for _lm_target in lm_target:
            if 'given persona:' not in left_tokenizer.decode(_lm_target):
                continue
            _tokens = left_tokenizer.convert_ids_to_tokens(_lm_target)
            _token_ids = _lm_target
            _token_idx = None
            for idx in range(0, len(_tokens) - 1):
                if _tokens[idx].endswith('context') and _tokens[idx + 1].endswith(':'):
                    _token_idx = idx
                    break
                _token_ids[idx] = left_tokenizer.pad_token_id
    # freeze context
    if config.training.freeze_context.__class__ is bool and config.training.freeze_context:
        for _lm_target in lm_target:
            _tokens = left_tokenizer.convert_ids_to_tokens(_lm_target)
            _token_ids = _lm_target
            _start_idx = None
            _end_idx = None
            for idx in range(0, len(_tokens) - 1):
                if _tokens[idx].endswith('context') and _tokens[idx + 1].endswith(':'):
                    _start_idx = idx
                if _tokens[idx].endswith('R') and _tokens[idx + 1].endswith(':'):
                    _end_idx = idx + 2
            if _start_idx is None or _end_idx is None:
                continue
            for idx in range(_start_idx, _end_idx):
                _token_ids[idx] = left_tokenizer.pad_token_id

    if for_test:
        inference_input = [template_types[template_type].format(cinput=cinput, target='') for cinput in concat_input]
        bos_concat_input = [f'{bos_token}{cinput}' for cinput in inference_input]
        inference_tokenized = left_tokenizer(bos_concat_input, add_special_tokens=False
                                             , return_tensors='pt',
                                             padding=True, truncation=True,
                                             max_length=config.dataset.max_token_length)
        return lm_input, lm_target, inference_tokenized
    return lm_input, lm_target


def format_generate_persona_input(batch, left_tokenizer, right_tokenizer, config, for_test=False, find_batch=False):
    batch_size = len(batch['context_input'])
    pad_token_id = left_tokenizer.pad_token_id
    targets = [' '.join(persona) for persona in batch['persona_list']]
    eos_token = left_tokenizer.eos_token
    concat_context = [' '.join(context) for context in batch['context_input']]
    concat_input = [f'#context#{context}' for context in
                    concat_context]
    inference_tokenized = None
    bos_token = left_tokenizer.bos_token
    if for_test:
        inference_input = [f'#context#{context}{bos_token}' for context in
                           concat_context]
        inference_tokenized = left_tokenizer(inference_input, add_special_tokens=False, return_tensors='pt',
                                             padding='max_length', truncation=True,
                                             max_length=config.dataset.max_token_length - 16)
    # processing target
    _target_with_bos = [f'{bos_token}{target}{eos_token}' for target in targets]
    _target_with_bos_pt = right_tokenizer(_target_with_bos,
                                          add_special_tokens=False, return_tensors='pt',
                                          padding=True)
    _target_pt = _target_with_bos_pt.copy()
    _target_pt['input_ids'] = torch.cat((_target_pt['input_ids'][:, 1:],
                                         _target_pt['input_ids'].new_ones(batch_size, 1) * pad_token_id), dim=1)
    _target_pt['attention_mask'] = torch.cat((_target_pt['attention_mask'][:, 1:],
                                              _target_pt['attention_mask'].new_zeros(batch_size, 1)), dim=1)
    # processing concat
    context_pt = left_tokenizer(concat_input, add_special_tokens=False, return_tensors='pt',
                                padding='max_length', truncation=True,
                                max_length=config.dataset.max_token_length)
    input_pt = torch.cat((context_pt['input_ids'], _target_with_bos_pt['input_ids']),
                         dim=1)[:, -config.dataset.max_token_length:]
    input_attn = torch.cat((context_pt['attention_mask'], _target_with_bos_pt['attention_mask']),
                           dim=1)[:, -config.dataset.max_token_length:]
    lm_input = {'input_ids': input_pt, 'attention_mask': input_attn}
    if find_batch:
        lm_target = torch.cat((context_pt['input_ids'],
                               _target_pt['input_ids']), dim=1)[:, -config.dataset.max_token_length:]
    else:
        lm_target = torch.cat((context_pt['input_ids'] * 0 - 1,
                               _target_pt['input_ids']), dim=1)[:, -config.dataset.max_token_length:]
    if for_test:
        return lm_input, lm_target, inference_tokenized
    return lm_input, lm_target