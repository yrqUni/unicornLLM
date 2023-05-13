import numpy as np
import os
from tqdm import tqdm
import copy

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

from . import raw_datasets

IGNORE_INDEX = -100

def get_raw_dataset(dataset_name, data_input_path, output_path, seed, local_rank, eval_split):
    if "unicorn" in dataset_name.lower():
        return raw_datasets.unicorn(output_path, data_input_path, seed, local_rank, eval_split)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )

def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

class PromptDataset(Dataset):

    def __init__(self, chosen_dataset,
                 pad_token_id, ) -> None:
        super().__init__()
        self.chosen_dataset = chosen_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        length = len(self.chosen_dataset)
        return length

    def __getitem__(self, idx):
        return {
            "input_ids": self.chosen_dataset[idx]["input_ids"],
            "attention_mask": self.chosen_dataset[idx]["attention_mask"],
            "labels": self.chosen_dataset[idx]["labels"]
        }


def pad_tensors_to_max_length(input_tensor, max_length, pad_token_id):
    padded_tensor = pad_token_id * torch.ones((max_length,), dtype=input_tensor.dtype, device=input_tensor.device)
    padded_tensor[-input_tensor.shape[0]:] = input_tensor
    return padded_tensor


def create_dataset_split(current_dataset, raw_dataset, tokenizer, max_seq_len=512):
    def _addrole_masklabel_tokenize(source, f_idx):
        '''
        add speaker and concatenate the sentences
        {
            "id": "uniq_sample_id",
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "assistant", "value": "你好，有什么可以帮助你的吗？"},
                {"from": "human", "value": "今天天气怎么样？"},
                {"from": "assistant", "value": "不好意思，我无法回答你的问题，因为我不知道你的位置信息，同时我目前还无法获取到最新的天气信息。"}
            ]
        }
        tokenizer_bloomz.encode("你好，有什么可以帮助你的吗？") == [41381, 355, 37242, 205599, 7336, 10468]
        tokenizer_llama.encode("你好，有什么可以帮助你的吗？") == [1, 29871, 30919, 31076, 30214, 30417, 231, 190, 131, 31882, 30682, 30651, 232, 187, 177, 31931, 30919, 30210, 232, 147, 154, 30882]
        '''

        conversation = ''
        input_ids = []
        labels = []
        for idx, sentence in enumerate(source):
            sentence_from = sentence["from"].lower()
            # 使用本函数注释的样本格式
            sentence_value = raw_dataset.get_prompt_and_chosen_unicorn(sentence, sentence_from)
            conversation += sentence_value
            sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)  # do not add bos_token_id
            label = copy.deepcopy(sentence_ids) if sentence_from != 'human' else [IGNORE_INDEX] * len(sentence_ids)
            input_ids += sentence_ids
            labels += label
            if sentence_from != 'human':
                # add eos at every end of assistant sentence
                input_ids += [tokenizer.eos_token_id]  # make sure eos_token_id is correct
                labels += [tokenizer.eos_token_id]
            else:
                # add bos at every begin of human sentence
                input_ids = [tokenizer.bos_token_id] + input_ids
                labels = [tokenizer.bos_token_id] + labels
        labels[0] = IGNORE_INDEX  # 第一位bos mask if use bos_token_id
        return input_ids, labels, conversation

    chosen_dataset = []
    filter_nums = 0
    assert tokenizer.padding_side == "left"  # We need add eos_token_id at the last position of input_ids
         
    train_phase = 1
    if train_phase == 1:
        for i, tmp_data in tqdm(enumerate(current_dataset), total=len(current_dataset), unit="example"):
        # for i, tmp_data in enumerate(current_dataset):
            source = raw_dataset.get_conversations(tmp_data)
            input_ids, labels, conversation = _addrole_masklabel_tokenize(source, i)
            input_ids = input_ids[:max_seq_len - 1]
            labels = labels[:max_seq_len - 1]
            if not any(x > IGNORE_INDEX for x in labels) or "Human" not in conversation:
                filter_nums += 1
                continue

            attention_mask = [1] * len(input_ids)
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            labels = torch.LongTensor(labels)

            chosen_token = {
                "input_ids": pad_tensors_to_max_length(input_ids, max_seq_len, tokenizer.pad_token_id),
                "attention_mask": pad_tensors_to_max_length(attention_mask, max_seq_len, tokenizer.pad_token_id),
                "labels": pad_tensors_to_max_length(labels, max_seq_len, IGNORE_INDEX)
            }
            chosen_dataset.append(chosen_token)
    else:
        raise ValueError("Only supported SFT")
    print(f'filter sample nums: {filter_nums}')
    return PromptDataset(chosen_dataset, tokenizer.pad_token_id)

def create_dataset(local_rank, dataset_name, data_input_path, output_path, seed, tokenizer, max_seq_len, eval_split):
    raw_dataset = get_raw_dataset(dataset_name, data_input_path, output_path, seed, local_rank, eval_split)
    train_dataset = raw_dataset.get_train_data()
    train_dataset = create_dataset_split(train_dataset, raw_dataset, tokenizer, max_seq_len)
    eval_dataset = raw_dataset.get_eval_data()
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, tokenizer, max_seq_len)
    return train_dataset, eval_dataset

def create_prompt_dataset(local_rank,
                          dataset_name,
                          data_input_path,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          eval_split):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(dataset_name)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
    fname = "_".join(fname.split("/"))
    fname = str(hash(fname))  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    # Skip creating cache if we found it on all the nodes.
    if buf_create_cache.item() == 0:
        return torch.load(train_fname), torch.load(eval_fname)
    else:
        if len(dataset_name) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, dataset_name[0], data_input_path[0],output_path,
                seed, tokenizer, max_seq_len, eval_split)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_name in dataset_name:
                train_dataset, eval_dataset = create_dataset(
                    local_rank, d_name, data_input_path, output_path,
                    seed, tokenizer, max_seq_len, eval_split)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        if local_rank <= 0:
            torch.save(train_dataset, train_fname)
            torch.save(eval_dataset, eval_fname)
        return train_dataset, eval_dataset


