import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset.dataset_helper import read_personachat_split
from utils.format_inputs import TASK_TYPE


class PersonaChatDataset(torch.utils.data.Dataset):
    # longest first for batch finder
    def __init__(self, data_path, max_context_turns=-1,
                 add_role_indicator=True, only_longest=False, training_ratio=1.0,
                 task_type=TASK_TYPE.GENERATE_RESPONSE):
        self.path = data_path
        self.add_role_indicator = add_role_indicator
        self.max_context_turns = max_context_turns
        self.turns_data = read_personachat_split(data_path, only_longest=only_longest)
        self.only_longest = only_longest
        self.training_ratio = training_ratio
        if training_ratio < 1.0:
            self.turns_data = self.turns_data[:int(len(self.turns_data) * training_ratio)]
        self.task_type = task_type
        # # For debug only
        # os.makedirs("data_logs", exist_ok=True)
        # random_num = random.randint(0, 100000)
        # self.file = open(f"data_logs/{random_num}_{data_path.split(os.sep)[-1]}", 'w')
        # # add id to turns_data
        # self.turns_data = [{'id': idx, **turn} for idx, turn in enumerate(self.turns_data)]
        # self.file.write(f"total_turns: {len(self.turns_data)}\n")

    def sort_longest_first(self):
        self.turns_data = sorted(self.turns_data, key=lambda x: len(
            (' '.join(x['persona']) + ' '.join(x['context']) + x['response']).split(' ')), reverse=True)

    def __getitem__(self, idx):
        # self.file.write(str(idx) + "\n")
        # self.file.flush()
        input_data = self.turns_data[idx]
        persona_list = input_data['persona']
        target = input_data['response']
        context_input = input_data['context']
        if self.add_role_indicator:
            roled_context_input = [['Q: ', 'R: '][c_idx % 2] + context for c_idx, context in enumerate(context_input)]
            context_input = roled_context_input
        if self.max_context_turns != -1:
            truncated_context = context_input[-(self.max_context_turns * 2 - 1):]
            context_input = truncated_context
        if self.only_longest:
            context_input = context_input[:-1]
        return {
            'context_input': context_input,
            'persona_list': persona_list,
            'target': target
        }

    def __len__(self):
        return len(self.turns_data)


# class HGPersonaChatDataset(PersonaChatDataset):
#     def __init__(self, data_path, max_context_turns=-1,
#                  add_role_indicator=True, only_longest=False, tokenizer=None):
#         super().__init__(data_path, max_context_turns, add_role_indicator, only_longest)
#         self.tokenizer = tokenizer
#
#     def __getitem__(self, idx):
#         data = super().__getitem__(idx)
#         input = "P: " + ' '.join(data['persona_list']) + " C: " + ' '.join(data['context_input']) + " R: " + data[
#             'target']
#         tokenized = self.tokenizer(input)
#         return {**data, **tokenized}


def collate_fn(sample_list):
    dont_be_a_tensor = ['context_input', 'persona_list', 'target']
    to_be_flattened = [*dont_be_a_tensor]
    data = {}
    for key in to_be_flattened:
        if key not in sample_list[0].keys():
            continue
        if sample_list[0][key] is None:
            continue
        flatten_samples = [sample[key] for sample in sample_list]
        if flatten_samples[-1].__class__ == str or key in dont_be_a_tensor:
            data[key] = flatten_samples
        else:
            data[key] = torch.tensor(flatten_samples)
    return data


def collate_fn_straight(sample_list):
    sample_list = collate_fn(sample_list)
    return sample_list


def collate_fn_straight_with_fn(fn):
    def build_collate_fn(sample_list):
        sample_list = collate_fn(sample_list)
        sample_list_processed = fn(sample_list)
        return {**sample_list, **sample_list_processed}

    return build_collate_fn


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=None, collate_fn=None, sampler=None):
    if num_workers is None:
        num_workers = batch_size // 4
    # num_workers = min(num_workers, batch_size)
    if collate_fn == None:
        _collate_fn = collate_fn_straight
    else:
        _collate_fn = collate_fn_straight_with_fn(collate_fn)
    return DataLoader(dataset, batch_size=batch_size,
                      collate_fn=_collate_fn,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      sampler=sampler)


def get_lightening_dataloader(dataset, batch_size, shuffle=False, num_workers=None):
    return LitDataModule(batch_size, dataset, shuffle, num_workers)


class LitDataModule(LightningDataModule):
    def __init__(self, batch_size, dataset, shuffle, num_workers):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])
        # or
        self.batch_size = batch_size
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          collate_fn=collate_fn_straight,
                          shuffle=self.hparams.shuffle,
                          num_workers=self.hparams.num_workers)

if __name__ == '__main__':
    import json
    train_ds = PersonaChatDataset(data_path='data_file/ConvAI2/train_self_original_no_cands.txt',
                                  )
    from tqdm import tqdm

    jsonfy_data = []

    for data in tqdm(train_ds):
        context_input = "\n".join(data['context_input'])
        persona_input = '\n'.join(data['persona_list'])
        jsonfy_data.append({
            "instruction": f"""Given the dialog history between Q and R is:
{context_input}

Given the personality of the R as: 
{persona_input}

Please response to Q according to both the dialog history and the R's personality.
Now, the R would say:""",
            "input": "",
            "output": data['target'],
            "answer": "",
        })
    with open('data_file/train.json', 'w') as writer:
        json.dump(jsonfy_data, writer)
    jsonfy_data = []
    del train_ds

    train_ds = PersonaChatDataset(data_path='data_file/ConvAI2/valid_self_original_no_cands.txt',
                                  )

    for data in tqdm(train_ds):
        context_input = "\n".join(data['context_input'])
        persona_input = '\n'.join(data['persona_list'])
        jsonfy_data.append({
            "instruction": f"""Given the dialog history between Q and R is:
{context_input}

Given the personality of the R as: 
{persona_input}

Please response to Q according to both the dialog history and the R's personality.
Now, the R would say:""",
            "input": "",
            "output": data['target'],
            "answer": "",
        })
    with open('data_file/valid.json', 'w') as writer:
        json.dump(jsonfy_data, writer)
    with open('data_file/test.json', 'w') as writer:
        json.dump(jsonfy_data, writer)