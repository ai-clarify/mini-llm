#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _count_jsonl_lines(path: Path, *, max_lines: Optional[int] = None) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if max_lines is not None and n >= max_lines:
                break
    return n


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def _split_prompt(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant = i
            break
    if last_assistant is None:
        return messages, []
    return messages[:last_assistant], messages[last_assistant:]


class SyntheticChatDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, *, max_seq_len: int) -> None:
        self.records = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        conversations = record.get("conversations") or record.get("messages")
        if not conversations:
            raise ValueError("Missing conversations/messages in JSONL record")

        prompt_msgs, _ = _split_prompt(conversations)
        full_text = _apply_chat_template(self.tokenizer, conversations, add_generation_prompt=False)
        prompt_text = _apply_chat_template(self.tokenizer, prompt_msgs, add_generation_prompt=True)

        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids

        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]

        attention_mask = [1] * len(full_ids)
        loss_mask = [0] * len(full_ids)
        start = min(len(prompt_ids), len(full_ids))
        for i in range(start, len(full_ids)):
            loss_mask[i] = 1

        pad_len = self.max_seq_len - len(full_ids)
        if pad_len > 0:
            full_ids += [self.pad_id] * pad_len
            attention_mask += [0] * pad_len
            loss_mask += [0] * pad_len

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float32),
        )


class LowRankHead(nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, int(rank), bias=False)
        self.out = nn.Linear(int(rank), vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.proj(x))


class Eagle3Speculator(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        spec_len: int,
        spec_layers: int,
        spec_heads: int,
        dropout: float,
        init_weight: Optional[torch.Tensor],
        head_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=spec_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(spec_layers))
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.heads = nn.ModuleList(
                [LowRankHead(hidden_size=hidden_size, vocab_size=vocab_size, rank=rank) for _ in range(int(spec_len))]
            )
        else:
            self.heads = nn.ModuleList([nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(int(spec_len))])
            if init_weight is not None:
                for head in self.heads:
                    head.weight.data.copy_(init_weight)

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> List[torch.Tensor]:
        x = hidden
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        seq_len = x.shape[1]
        causal_mask = None
        if seq_len > 1:
            # Prevent peeking at future positions during multi-token training.
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def _spec_loss(
    logits_list: List[torch.Tensor],
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_loss = torch.tensor(0.0, device=input_ids.device)
    total_tokens = torch.tensor(0.0, device=input_ids.device)
    vocab_size = logits_list[0].shape[-1] if logits_list else 0
    for i, logits in enumerate(logits_list):
        offset = i + 1
        if input_ids.shape[1] <= offset:
            continue
        labels = input_ids[:, offset:]
        pred = logits[:, :-offset, :]
        label_mask = loss_mask[:, offset:] * attention_mask[:, offset:]
        if label_mask.sum() == 0:
            continue
        loss = F.cross_entropy(pred.reshape(-1, vocab_size), labels.reshape(-1), reduction="none")
        loss = loss.view_as(labels)
        total_loss += (loss * label_mask).sum()
        total_tokens += label_mask.sum()
    if total_tokens.item() == 0:
        return total_loss, total_tokens
    return total_loss / total_tokens, total_tokens


def _resolve_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _count_params_torch(model: nn.Module) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _auto_spec_config(param_count: Optional[int]) -> Tuple[int, int]:
    if not param_count or param_count <= 0:
        return 1, 1
    params_b = float(param_count) / 1e9
    if params_b <= 1.0:
        return 1, 1
    if params_b <= 3.0:
        return 2, 1
    if params_b <= 7.0:
        return 3, 2
    if params_b <= 13.0:
        return 4, 2
    return 5, 2


def _resolve_spec_config(
    spec_len: Optional[int], spec_layers: Optional[int], *, param_count: Optional[int]
) -> Tuple[int, int]:
    auto_len, auto_layers = _auto_spec_config(param_count)
    if spec_len is None or int(spec_len) <= 0:
        spec_len = auto_len
    if spec_layers is None or int(spec_layers) <= 0:
        spec_layers = auto_layers
    return int(spec_len), int(spec_layers)


def _default_head_rank(hidden_size: int) -> int:
    return max(32, min(256, int(hidden_size) // 8))


def _save_checkpoint(
    *, step: int, speculator: nn.Module, optimizer: torch.optim.Optimizer, out_dir: Path
) -> None:
    ckpt_dir = out_dir / "checkpoints" / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(speculator.state_dict(), ckpt_dir / "speculator.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    (ckpt_dir / "train_state.json").write_text(
        json.dumps({"step": step, "timestamp": time.time()}, indent=2),
        encoding="utf-8",
    )
    print(f"[ckpt] saved {ckpt_dir}")


def _load_minillm_config(path: Optional[str]) -> MiniLLMConfig:
    if not path:
        return MiniLLMConfig()
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MiniLLM config must be a JSON object: {cfg_path}")
    return MiniLLMConfig(**data)


def _forward_hidden_qwen3(
    target: nn.Module, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    out = target.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.last_hidden_state


def _forward_hidden_minillm(
    target: nn.Module, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    hidden, _, _ = target.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return hidden


def _ensure_synth_data(args) -> None:
    data_path = Path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_jsonl_lines(data_path, max_lines=args.gen_total_samples)
    if existing >= args.min_samples:
        print(f"[data] found {existing} samples at {data_path}, skip generation")
        return
    target_total = max(int(args.gen_total_samples), int(args.min_samples))
    cmd = [
        sys.executable,
        "-m",
        "mlx_train.distill_data_ollama",
        "--out_jsonl",
        str(data_path),
        "--ollama_url",
        args.ollama_url,
        "--ollama_model",
        args.ollama_model,
        "--target_total_samples",
        str(target_total),
        "--num_workers",
        str(args.gen_workers),
        "--max_new_tokens",
        str(args.gen_max_new_tokens),
        "--temperature",
        str(args.gen_temperature),
        "--top_p",
        str(args.gen_top_p),
    ]
    print(f"[gen] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an EAGLE-3 style speculator for Qwen3-0.6B or MiniLLM using pure synthetic data."
    )
    parser.add_argument("--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--minillm_ckpt", type=str, default=None)
    parser.add_argument("--minillm_config", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument("--data_path", type=str, default="out/distill_ollama_qwen3_0.6b/synth.jsonl")
    parser.add_argument("--out_dir", type=str, default="out/eagle3_speculator/qwen3_0.6b")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--head_rank",
        type=int,
        default=None,
        help="Low-rank speculator head size (auto if unset; set 0 to disable).",
    )
    parser.add_argument(
        "--early_stop_loss",
        type=float,
        default=None,
        help="Stop early when loss <= this value (disabled if unset).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Consecutive steps meeting early_stop_loss before stopping (0 = immediate).",
    )
    parser.add_argument("--spec_len", type=int, default=None, help="Draft length for speculator (auto if unset).")
    parser.add_argument("--spec_layers", type=int, default=None, help="Transformer layers in speculator (auto if unset).")
    parser.add_argument("--spec_heads", type=int, default=0)
    parser.add_argument("--spec_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min_samples", type=int, default=20000)

    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_model", type=str, default="qwen3:0.6b")
    parser.add_argument("--gen_total_samples", type=int, default=20000)
    parser.add_argument("--gen_workers", type=int, default=8)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)
    parser.add_argument("--gen_temperature", type=float, default=0.2)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--no_auto_generate", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if not args.no_auto_generate:
        _ensure_synth_data(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)

    if args.target_arch == "minillm":
        tokenizer = AutoTokenizer.from_pretrained(args.minillm_tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "right"
        cfg = _load_minillm_config(args.minillm_config)
        target = MiniLLMForCausalLM(cfg)
        if args.minillm_ckpt:
            state = torch.load(args.minillm_ckpt, map_location=device)
            target.load_state_dict(state, strict=False)
        else:
            print("[warn] MiniLLM checkpoint not provided; using random weights")
        target = target.to(device=device, dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "right"
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad = False

    param_count = _count_params_torch(target)
    auto_spec = (
        args.spec_len is None
        or args.spec_layers is None
        or int(args.spec_len) <= 0
        or int(args.spec_layers) <= 0
    )
    spec_len, spec_layers = _resolve_spec_config(
        args.spec_len, args.spec_layers, param_count=param_count
    )
    args.spec_len = spec_len
    args.spec_layers = spec_layers
    if auto_spec:
        if param_count:
            print(
                f"[speculator] auto spec_len={spec_len} spec_layers={spec_layers} "
                f"(target_params={param_count / 1e9:.2f}B)"
            )
        else:
            print(f"[speculator] auto spec_len={spec_len} spec_layers={spec_layers}")

    hidden_size = int(target.config.hidden_size)
    vocab_size = int(target.config.vocab_size)
    spec_heads = int(args.spec_heads)
    if spec_heads <= 0:
        cfg_heads = int(target.config.num_attention_heads)
        if cfg_heads > 0:
            spec_heads = cfg_heads
        else:
            spec_heads = max(1, hidden_size // 64)
        while spec_heads > 1 and hidden_size % spec_heads != 0:
            spec_heads -= 1

    init_weight = target.lm_head.weight.detach().clone()

    if args.head_rank is None:
        head_rank = _default_head_rank(hidden_size)
    elif int(args.head_rank) <= 0:
        head_rank = None
    else:
        head_rank = int(args.head_rank)
    speculator = Eagle3Speculator(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        spec_len=args.spec_len,
        spec_layers=args.spec_layers,
        spec_heads=spec_heads,
        dropout=args.spec_dropout,
        init_weight=init_weight,
        head_rank=head_rank,
    ).to(device)

    trainable = sum(p.numel() for p in speculator.parameters() if p.requires_grad)
    head_note = f" head_rank={head_rank}" if head_rank else ""
    print(
        f"[speculator] params={trainable / 1e6:.2f}M spec_len={args.spec_len} "
        f"spec_layers={args.spec_layers}{head_note}"
    )

    dataset = SyntheticChatDataset(Path(args.data_path), tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    data_iter = _infinite_loader(loader)

    optimizer = torch.optim.AdamW(speculator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    config = {
        "target_arch": args.target_arch,
        "target_model": args.target_model,
        "minillm_ckpt": args.minillm_ckpt,
        "minillm_config": args.minillm_config,
        "minillm_tokenizer": args.minillm_tokenizer,
        "data_path": args.data_path,
        "max_seq_len": args.max_seq_len,
        "spec_len": args.spec_len,
        "spec_layers": args.spec_layers,
        "spec_heads": spec_heads,
        "head_rank": head_rank,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "dtype": args.dtype,
        "early_stop_loss": args.early_stop_loss,
        "early_stop_patience": args.early_stop_patience,
    }
    (out_dir / "speculator_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start_time = time.time()
    accum_steps = int(args.accum_steps)
    if accum_steps <= 0:
        raise ValueError("--accum_steps must be >= 1")

    optimizer.zero_grad(set_to_none=True)
    early_stop_loss = args.early_stop_loss
    early_stop_patience = int(args.early_stop_patience or 0)
    if early_stop_loss is not None and early_stop_patience <= 0:
        early_stop_patience = 1
    early_stop_hits = 0
    last_step = 0
    try:
        for step in range(1, int(args.max_steps) + 1):
            input_ids, attention_mask, loss_mask = next(data_iter)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loss_mask = loss_mask.to(device)
            with torch.no_grad():
                if args.target_arch == "minillm":
                    hidden = _forward_hidden_minillm(target, input_ids, attention_mask)
                else:
                    hidden = _forward_hidden_qwen3(target, input_ids, attention_mask)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits_list = speculator(hidden, attention_mask)
                loss, tokens = _spec_loss(logits_list, input_ids, loss_mask, attention_mask)
                if tokens.item() == 0:
                    continue
                loss = loss / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_val = float(loss.item() * accum_steps)
            stop_training = False
            if early_stop_loss is not None:
                if loss_val <= float(early_stop_loss):
                    early_stop_hits += 1
                    if early_stop_hits >= early_stop_patience:
                        print(
                            f"[train] early stop at step={step} loss={loss_val:.4f} "
                            f"target={float(early_stop_loss):.4f}"
                        )
                        stop_training = True
                else:
                    early_stop_hits = 0

            if step % accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(speculator.parameters(), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                tok_s = tokens.item() / max(elapsed, 1e-6)
                print(f"[train] step={step} loss={loss_val:.4f} tok/s={tok_s:.2f}")
                start_time = time.time()

            saved = False
            if step % args.save_interval == 0 or step == int(args.max_steps):
                _save_checkpoint(step=step, speculator=speculator, optimizer=optimizer, out_dir=out_dir)
                saved = True

            last_step = step
            if stop_training:
                if not saved:
                    _save_checkpoint(step=step, speculator=speculator, optimizer=optimizer, out_dir=out_dir)
                break
    except KeyboardInterrupt:
        if last_step > 0:
            print(f"[train] interrupted at step={last_step}, saving checkpoint")
            _save_checkpoint(step=last_step, speculator=speculator, optimizer=optimizer, out_dir=out_dir)
        else:
            print("[train] interrupted before first step; no checkpoint saved")


if __name__ == "__main__":
    main()
