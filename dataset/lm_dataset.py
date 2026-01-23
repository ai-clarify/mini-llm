import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast
from array import array
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from mlx_train.data import Bin2DDataset, BinDataset, detect_bin_format
except Exception:
    Bin2DDataset = None
    BinDataset = None
    detect_bin_format = None


def _infer_data_format(path: str, data_format: str) -> str:
    fmt = str(data_format).lower()
    if fmt == "auto":
        if detect_bin_format is not None:
            fmt = detect_bin_format(path) or "jsonl"
        else:
            fmt = "jsonl"
    return fmt


def _ids_to_tensor(ids, dtype=torch.long) -> torch.Tensor:
    if isinstance(ids, memoryview):
        arr = np.frombuffer(ids, dtype=np.uint32)
        return torch.as_tensor(arr, dtype=dtype)
    if isinstance(ids, array):
        arr = np.asarray(ids, dtype=np.uint32)
        return torch.as_tensor(arr, dtype=dtype)
    return torch.as_tensor(ids, dtype=dtype)


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, *, data_format: str = "auto", bin_cache: str = "mmap"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = None
        self._bin_dataset = None
        self._bin_format = _infer_data_format(data_path, data_format)
        self._bin_cache = bin_cache
        if self._bin_format in ("bin", "bin2d"):
            if self._bin_format == "bin2d":
                if Bin2DDataset is None:
                    raise RuntimeError("Bin2DDataset unavailable; ensure mlx_train is on PYTHONPATH")
                self._bin_dataset = Bin2DDataset(data_path, cache=bin_cache)
                if self._bin_dataset.seq_len > 0:
                    self.max_length = min(self.max_length, int(self._bin_dataset.seq_len))
            else:
                if BinDataset is None:
                    raise RuntimeError("BinDataset unavailable; ensure mlx_train is on PYTHONPATH")
                self._bin_dataset = BinDataset(data_path, cache=bin_cache)
        else:
            self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        if self._bin_dataset is not None:
            return len(self._bin_dataset)
        return len(self.samples)

    def __getitem__(self, index):
        if self._bin_dataset is not None:
            ids = self._bin_dataset.get_ids(index)
            input_ids = _ids_to_tensor(ids, dtype=torch.long)
            target_len = int(self.max_length) + 1
            if input_ids.numel() < target_len:
                pad_id = int(self.tokenizer.pad_token_id)
                pad = torch.full((target_len - input_ids.numel(),), pad_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad], dim=0)
            elif input_ids.numel() > target_len:
                input_ids = input_ids[:target_len]

            loss_mask = (input_ids != int(self.tokenizer.pad_token_id)).long()
            X = input_ids[:-1].clone().detach()
            Y = input_ids[1:].clone().detach()
            loss_mask = loss_mask[1:].clone().detach()
            return X, Y, loss_mask

        sample = self.samples[index]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = input_ids[:-1].clone().detach()
        Y = input_ids[1:].clone().detach()
        loss_mask = loss_mask[1:].clone().detach().long()
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, *, data_format: str = "auto", bin_cache: str = "mmap"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = None
        self._bin_dataset = None
        self._bin_format = _infer_data_format(jsonl_path, data_format)
        self._bin_cache = bin_cache
        if self._bin_format in ("bin", "bin2d"):
            if self._bin_format == "bin2d":
                if Bin2DDataset is None:
                    raise RuntimeError("Bin2DDataset unavailable; ensure mlx_train is on PYTHONPATH")
                self._bin_dataset = Bin2DDataset(jsonl_path, cache=bin_cache)
                if self._bin_dataset.seq_len > 0:
                    self.max_length = min(self.max_length, int(self._bin_dataset.seq_len))
            else:
                if BinDataset is None:
                    raise RuntimeError("BinDataset unavailable; ensure mlx_train is on PYTHONPATH")
                self._bin_dataset = BinDataset(jsonl_path, cache=bin_cache)
        else:
            self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        if self._bin_dataset is not None:
            return len(self._bin_dataset)
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        if self._bin_dataset is not None:
            ids = self._bin_dataset.get_ids(index)
            input_ids = _ids_to_tensor(ids, dtype=torch.long)
            target_len = int(self.max_length) + 1
            if input_ids.numel() < target_len:
                pad_id = int(self.tokenizer.pad_token_id)
                pad = torch.full((target_len - input_ids.numel(),), pad_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad], dim=0)
            elif input_ids.numel() > target_len:
                input_ids = input_ids[:target_len]

            X = input_ids[:-1].clone().detach()
            Y = input_ids[1:].clone().detach()
            pos = None
            if hasattr(self._bin_dataset, "get_label_pos"):
                pos = self._bin_dataset.get_label_pos(index)
            if pos is None:
                loss_mask = self._generate_loss_mask(input_ids.tolist())
                loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
            else:
                loss_mask = torch.zeros_like(Y, dtype=torch.long)
                for p in pos:
                    if 0 <= int(p) < loss_mask.numel():
                        loss_mask[int(p)] = 1
            return X, Y, loss_mask

        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass
