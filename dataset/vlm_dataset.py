"""Dataset helpers for MiniLLM vision-language fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset

from model.model_vlm import MiniLLMVLM


class VLMDataset(Dataset):
    """Simple JSONL based dataset used for VLM SFT and pre-training."""

    def __init__(
        self,
        jsonl_path: str,
        images_path: str,
        tokenizer,
        preprocess=None,
        max_length: int = 512,
        image_special_token: str = "@" * 196,
    ) -> None:
        super().__init__()
        self.samples = self._load_data(jsonl_path)
        self.images_path = Path(images_path)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length
        self.image_token = image_special_token
        self.bos_id = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def _load_data(self, path: str) -> List[dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _create_chat_prompt(self, conversations: List[dict]) -> str:
        messages = []
        for idx, turn in enumerate(conversations):
            role = "user" if idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"].replace("<image>", self.image_token)})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        prompt = self._create_chat_prompt(sample["conversations"])
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        label_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        mask_tensor = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in sample["image"].split(","):
            image_name = image_name.strip()
            image_path = self.images_path / image_name
            if not image_path.exists():
                candidates = list(self.images_path.glob(f"**/{image_name}"))
                if not candidates:
                    raise FileNotFoundError(f"Image {image_name} not found under {self.images_path}")
                image_path = candidates[0]
            image = Image.open(image_path)
            image_tensor = MiniLLMVLM.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        pixel_values = torch.stack(image_tensors, dim=0)

        return input_tensor, label_tensor, mask_tensor, pixel_values
