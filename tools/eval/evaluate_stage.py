#!/usr/bin/env python
"""Lightweight evaluation helpers for the MiniLLM training pipeline."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

# Suppress pynvml deprecation warning from torch.cuda
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from dataset.lm_dataset import DPODataset, PretrainDataset, SFTDataset
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM


def load_model(checkpoint: Path, hidden_size: int, num_hidden_layers: int, device: torch.device, *, use_moe: bool = False) -> MiniLLMForCausalLM:
    config = MiniLLMConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=use_moe)
    model = MiniLLMForCausalLM(config)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
    loss = loss.view(labels.size())
    mask = mask.to(loss.dtype)
    loss = (loss * mask).sum()
    tokens = mask.sum()
    return loss, tokens


def evaluate_pretrain(model: MiniLLMForCausalLM, dataset: PretrainDataset, device: torch.device, *, max_samples: int, batch_size: int) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            x, y, mask = (t.to(device) for t in batch)
            logits = model(x).logits
            loss, tokens = masked_cross_entropy(logits, y, mask)
            total_loss += loss
            total_tokens += tokens
            processed += x.size(0)
            if processed >= max_samples:
                break
    avg_loss = (total_loss / total_tokens).item()
    perplexity = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": perplexity}


def evaluate_sft(model: MiniLLMForCausalLM, dataset: SFTDataset, device: torch.device, *, max_samples: int, batch_size: int) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            x, y, mask = (t.to(device) for t in batch)
            logits = model(x).logits
            loss, tokens = masked_cross_entropy(logits, y, mask)
            total_loss += loss
            total_tokens += tokens
            processed += x.size(0)
            if processed >= max_samples:
                break
    avg_loss = (total_loss / total_tokens).item()
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": ppl}


def evaluate_dpo(model: MiniLLMForCausalLM, dataset: DPODataset, device: torch.device, *, max_samples: int, batch_size: int) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chosen_total = torch.zeros(1, device=device)
    reject_total = torch.zeros(1, device=device)
    comparisons = 0
    with torch.no_grad():
        for batch in loader:
            x_chosen = batch["x_chosen"].to(device)
            x_rejected = batch["x_rejected"].to(device)
            y_chosen = batch["y_chosen"].to(device)
            y_rejected = batch["y_rejected"].to(device)
            mask_chosen = batch["mask_chosen"].to(device)
            mask_rejected = batch["mask_rejected"].to(device)

            logits_chosen = model(x_chosen).logits
            logits_rejected = model(x_rejected).logits

            chosen_loss, chosen_tokens = masked_cross_entropy(logits_chosen, y_chosen, mask_chosen)
            reject_loss, reject_tokens = masked_cross_entropy(logits_rejected, y_rejected, mask_rejected)

            if chosen_tokens.item() == 0 or reject_tokens.item() == 0:
                continue

            batch_size = x_chosen.size(0)
            chosen_avg = -(chosen_loss / chosen_tokens)
            reject_avg = -(reject_loss / reject_tokens)

            chosen_total += chosen_avg * batch_size
            reject_total += reject_avg * batch_size
            comparisons += batch_size

            if comparisons >= max_samples:
                break

    if comparisons == 0:
        return {"win_rate": 0.0, "reward_margin": 0.0}

    avg_chosen = chosen_total.item() / comparisons
    avg_reject = reject_total.item() / comparisons
    margin = avg_chosen - avg_reject
    win_rate = 0.5 * (math.tanh(margin) + 1.0)
    return {"win_rate": win_rate, "reward_margin": margin}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniLLM checkpoints on small held-out splits")
    parser.add_argument("--stage", choices=["pretrain", "sft", "dpo"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-file", type=Path, help="Append evaluation summaries to this JSONL file")
    parser.add_argument("--tensorboard-dir", type=Path, help="Optional TensorBoard log directory for evaluation metrics")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("./model/")

    if args.stage == "pretrain":
        dataset = PretrainDataset(str(args.data_path), tokenizer, max_length=args.max_seq_len)
    elif args.stage == "sft":
        dataset = SFTDataset(str(args.data_path), tokenizer, max_length=args.max_seq_len)
    else:
        dataset = DPODataset(str(args.data_path), tokenizer, max_length=args.max_seq_len)

    model = load_model(checkpoint, args.hidden_size, args.num_hidden_layers, device, use_moe=args.use_moe)

    if args.stage == "pretrain":
        metrics = evaluate_pretrain(model, dataset, device, max_samples=args.max_samples, batch_size=args.batch_size)
    elif args.stage == "sft":
        metrics = evaluate_sft(model, dataset, device, max_samples=args.max_samples, batch_size=args.batch_size)
    else:
        metrics = evaluate_dpo(model, dataset, device, max_samples=args.max_samples, batch_size=args.batch_size)

    summary = {"stage": args.stage, **metrics}
    print(json.dumps(summary, ensure_ascii=False))

    if args.results_file:
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        with args.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if args.tensorboard_dir:
        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        with SummaryWriter(log_dir=str(args.tensorboard_dir)) as writer:
            for key, value in metrics.items():
                writer.add_scalar(f"{args.stage}/{key}", value)


if __name__ == "__main__":
    main()
