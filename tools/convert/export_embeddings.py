#!/usr/bin/env python
"""Convert JSONL datasets to token tensors using RustBPE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import torch

from tokenizer import RustBPETokenizer
from utils.conversation import flatten_conversation, normalize_conversation


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pad(sequence: Sequence[int], length: int, pad_id: int) -> List[int]:
    seq = list(sequence[:length])
    if len(seq) < length:
        seq.extend([pad_id] * (length - len(seq)))
    return seq


def build_pretrain_tensor(
    tokenizer: RustBPETokenizer,
    record: dict,
    max_length: int,
    pad_id: int,
    append_eos: bool,
) -> Tuple[List[int], List[int], List[int]]:
    if "text" in record:
        text = str(record["text"])
    else:
        try:
            messages = normalize_conversation(record)
        except Exception:
            return [], [], []
        text = flatten_conversation(messages)
    ids = [tokenizer.get_bos_token_id()] + tokenizer.encode(text)
    if append_eos:
        ids.append(tokenizer.encode_special("<|assistant_end|>"))
    if len(ids) < 2:
        return [], [], []
    inputs = pad(ids[:-1], max_length, pad_id)
    labels = pad(ids[1:], max_length, pad_id)
    loss_mask = [1] * min(len(ids) - 1, max_length)
    loss_mask.extend([0] * (max_length - len(loss_mask)))
    return inputs, labels, loss_mask


def build_sft_tensor(
    tokenizer: RustBPETokenizer,
    record: dict,
    max_length: int,
    pad_id: int,
) -> Tuple[List[int], List[int], List[int]]:
    try:
        messages = normalize_conversation(record)
    except Exception:
        return [], [], []
    ids, mask = tokenizer.render_conversation(messages, max_tokens=max_length + 1)
    if len(ids) < 2:
        return [], [], []
    inputs = pad(ids[:-1], max_length, pad_id)
    labels = pad(ids[1:], max_length, pad_id)
    loss_mask = pad(mask[1:], max_length, 0)
    return inputs, labels, loss_mask


def build_dpo_tensor(
    tokenizer: RustBPETokenizer,
    record: dict,
    max_length: int,
    pad_id: int,
) -> Dict[str, List[int]]:
    chosen = record.get("chosen") or record.get("chosen_conversation")
    rejected = record.get("rejected") or record.get("rejected_conversation")
    if chosen is None or rejected is None:
        return {}
    try:
        chosen_messages = normalize_conversation({"conversations": chosen})
        rejected_messages = normalize_conversation({"conversations": rejected})
    except Exception:
        return {}
    chosen_ids, chosen_mask = tokenizer.render_conversation(chosen_messages, max_tokens=max_length + 1)
    rejected_ids, rejected_mask = tokenizer.render_conversation(rejected_messages, max_tokens=max_length + 1)
    if len(chosen_ids) < 2 or len(rejected_ids) < 2:
        return {}
    payload = {
        "chosen_input_ids": pad(chosen_ids[:-1], max_length, pad_id),
        "chosen_labels": pad(chosen_ids[1:], max_length, pad_id),
        "chosen_loss_mask": pad(chosen_mask[1:], max_length, 0),
        "rejected_input_ids": pad(rejected_ids[:-1], max_length, pad_id),
        "rejected_labels": pad(rejected_ids[1:], max_length, pad_id),
        "rejected_loss_mask": pad(rejected_mask[1:], max_length, 0),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export token tensors for training")
    parser.add_argument("phase", choices=["pretrain", "sft", "dpo"], help="Training phase")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output torch file")
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="Directory of the trained RustBPE tokenizer")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--pad-id", type=int, default=0, help="Padding token id")
    parser.add_argument("--append-eos", action="store_true", help="Append assistant end token in pretrain mode")
    args = parser.parse_args()

    tokenizer = RustBPETokenizer.from_directory(args.tokenizer_dir)
    records = list(iter_jsonl(args.input))
    print(f"Loaded {len(records)} records from {args.input}")

    if args.phase == "pretrain":
        data = [
            build_pretrain_tensor(tokenizer, record, args.max_length, args.pad_id, args.append_eos)
            for record in records
        ]
        data = [item for item in data if item[0]]
        if not data:
            raise RuntimeError("No valid pretrain samples found")
        inputs = torch.tensor([item[0] for item in data], dtype=torch.int32)
        labels = torch.tensor([item[1] for item in data], dtype=torch.int32)
        loss_mask = torch.tensor([item[2] for item in data], dtype=torch.int32)
        torch.save({"input_ids": inputs, "labels": labels, "loss_mask": loss_mask}, args.output)
        print(f"Saved {inputs.shape[0]} pretrain samples to {args.output}")
        return

    if args.phase == "sft":
        data = [
            build_sft_tensor(tokenizer, record, args.max_length, args.pad_id)
            for record in records
        ]
        data = [item for item in data if item[0]]
        if not data:
            raise RuntimeError("No valid SFT samples found")
        inputs = torch.tensor([item[0] for item in data], dtype=torch.int32)
        labels = torch.tensor([item[1] for item in data], dtype=torch.int32)
        loss_mask = torch.tensor([item[2] for item in data], dtype=torch.int32)
        torch.save({"input_ids": inputs, "labels": labels, "loss_mask": loss_mask}, args.output)
        print(f"Saved {inputs.shape[0]} SFT samples to {args.output}")
        return

    if args.phase == "dpo":
        data = [
            build_dpo_tensor(tokenizer, record, args.max_length, args.pad_id)
            for record in records
        ]
        data = [item for item in data if item]
        if not data:
            raise RuntimeError("No valid DPO samples found")
        payload = {key: torch.tensor([item[key] for item in data], dtype=torch.int32) for key in data[0].keys()}
        torch.save(payload, args.output)
        print(f"Saved {len(data)} DPO preference pairs to {args.output}")
        return

    raise ValueError(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    main()
