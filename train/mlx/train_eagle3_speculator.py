#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer.mlx.common import Eagle3Speculator, _default_model_dir

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
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    return "\n\n".join([str(m.get("content", "")) for m in messages])


def _split_prompt(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant = i
            break
    if last_assistant is None:
        return messages, []
    return messages[:last_assistant], messages[last_assistant:]


class SyntheticChatDataset:
    def __init__(self, jsonl_path: Path, tokenizer, *, max_seq_len: int) -> None:
        self.samples: List[Tuple[List[int], List[int], List[float]]] = []
        self.max_seq_len = int(max_seq_len)
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                conversations = record.get("conversations") or record.get("messages")
                if not conversations:
                    continue

                prompt_msgs, _ = _split_prompt(conversations)
                full_text = _apply_chat_template(tokenizer, conversations, add_generation_prompt=False)
                prompt_text = _apply_chat_template(tokenizer, prompt_msgs, add_generation_prompt=True)

                full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

                if len(full_ids) > self.max_seq_len:
                    full_ids = full_ids[: self.max_seq_len]

                attention_mask = [1] * len(full_ids)
                loss_mask = [0.0] * len(full_ids)
                start = min(len(prompt_ids), len(full_ids))
                for i in range(start, len(full_ids)):
                    loss_mask[i] = 1.0

                pad_len = self.max_seq_len - len(full_ids)
                if pad_len > 0:
                    full_ids += [int(self.pad_id)] * pad_len
                    attention_mask += [0] * pad_len
                    loss_mask += [0.0] * pad_len

                self.samples.append((full_ids, attention_mask, loss_mask))

    def __len__(self) -> int:
        return len(self.samples)

    def sample_batch(self, *, batch_size: int, rng: random.Random) -> Tuple[mx.array, mx.array, mx.array]:
        if not self.samples:
            raise ValueError("Dataset is empty")
        idxs = [rng.randrange(len(self.samples)) for _ in range(int(batch_size))]
        batch_ids = [self.samples[i][0] for i in idxs]
        batch_attn = [self.samples[i][1] for i in idxs]
        batch_loss = [self.samples[i][2] for i in idxs]
        return (
            mx.array(batch_ids, dtype=mx.int32),
            mx.array(batch_attn, dtype=mx.float32),
            mx.array(batch_loss, dtype=mx.float32),
        )


def _spec_loss(
    logits_list: List[mx.array],
    input_ids: mx.array,
    loss_mask: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = mx.array(0.0, dtype=mx.float32)
    for i, logits in enumerate(logits_list):
        offset = i + 1
        if input_ids.shape[1] <= offset:
            continue
        labels = input_ids[:, offset:]
        pred = logits[:, :-offset, :]
        label_mask = loss_mask[:, offset:] * attention_mask[:, offset:]
        vocab = int(pred.shape[-1])
        loss = nn.losses.cross_entropy(
            pred.reshape(-1, vocab),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        total_loss = total_loss + mx.sum(loss * label_mask)
        total_tokens = total_tokens + mx.sum(label_mask)

    denom = mx.maximum(total_tokens, mx.array(1.0, dtype=mx.float32))
    return total_loss / denom


def _resolve_dtype(name: str):
    name = str(name).lower()
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16
    if name == "float32":
        return mx.float32
    raise ValueError(f"Unsupported dtype: {name}")


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


def _count_tokens(loss_mask: mx.array, attention_mask: mx.array, *, spec_len: int) -> float:
    tokens = 0.0
    for i in range(int(spec_len)):
        offset = i + 1
        if loss_mask.shape[1] <= offset:
            continue
        mask = loss_mask[:, offset:] * attention_mask[:, offset:]
        tokens += float(mx.sum(mask).item())
    return tokens


def save_optimizer_state(optimizer: optim.Optimizer, path: str) -> None:
    flat: Dict[str, Any] = {}
    mlx_utils.tree_flatten(optimizer.state, destination=flat)
    mx.savez(path, **flat)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an EAGLE-3 style speculator for Qwen3-0.6B (MLX backend, pure synthetic data)."
    )
    parser.add_argument("--hf_repo", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="out/distill_ollama_qwen3_0.6b/synth.jsonl")
    parser.add_argument("--out_dir", type=str, default="out/eagle3_speculator_mlx/qwen3_0.6b")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--spec_len", type=int, default=7)
    parser.add_argument("--spec_layers", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16")
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

    mx.random.seed(int(args.seed))
    rng = random.Random(int(args.seed))

    if not args.no_auto_generate:
        _ensure_synth_data(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_lm import load
    except ImportError as exc:
        raise ImportError("Missing mlx-lm. Install via `python3 -m pip install mlx-lm`.") from exc

    if args.model_dir:
        model_path = Path(args.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model dir not found: {model_path}")
        model_source = str(model_path)
    else:
        default_dir = _default_model_dir(args.hf_repo)
        model_source = str(default_dir) if default_dir.exists() else args.hf_repo

    target, tokenizer = load(model_source, revision=args.revision)
    target.eval()

    if getattr(tokenizer, "pad_token_id", None) is None:
        try:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        except Exception:
            pass

    init_weight = None
    if hasattr(target, "args") and getattr(target.args, "tie_word_embeddings", False):
        init_weight = mx.array(target.model.embed_tokens.weight)
    elif hasattr(target, "lm_head"):
        init_weight = mx.array(target.lm_head.weight)

    if not hasattr(target, "args"):
        raise AttributeError("Target model missing args; expected mlx-lm Qwen3 model")

    speculator = Eagle3Speculator(
        args=target.args,
        spec_len=args.spec_len,
        spec_layers=args.spec_layers,
        init_weight=init_weight,
    )

    dtype = _resolve_dtype(args.dtype)
    speculator.apply(lambda p: p.astype(dtype))
    speculator.train()

    trainable = sum(p.size for p in speculator.trainable_parameters())
    print(f"[speculator] params={trainable / 1e6:.2f}M spec_len={args.spec_len} spec_layers={args.spec_layers}")

    dataset = SyntheticChatDataset(Path(args.data_path), tokenizer, max_seq_len=args.max_seq_len)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; check --data_path")

    optimizer = optim.AdamW(learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay))
    optimizer.init(speculator.trainable_parameters())

    def loss_wrapped(input_ids: mx.array, attention_mask: mx.array, loss_mask: mx.array) -> mx.array:
        hidden = target.model(input_ids, cache=None)
        hidden = mx.stop_gradient(hidden)
        if hidden.dtype != dtype:
            hidden = hidden.astype(dtype)
        logits_list = speculator(hidden, attention_mask=attention_mask)
        return _spec_loss(logits_list, input_ids, loss_mask, attention_mask)

    value_and_grad = nn.value_and_grad(speculator, loss_wrapped)

    config = {
        "hf_repo": args.hf_repo,
        "model_source": model_source,
        "data_path": args.data_path,
        "max_seq_len": args.max_seq_len,
        "spec_len": args.spec_len,
        "spec_layers": args.spec_layers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "dtype": args.dtype,
    }
    (out_dir / "speculator_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start_time = time.time()
    accum_steps = int(args.accum_steps)
    if accum_steps <= 0:
        raise ValueError("--accum_steps must be >= 1")

    grad_template = mlx_utils.tree_map(lambda p: mx.zeros_like(p), speculator.trainable_parameters())

    for step in range(1, int(args.max_steps) + 1):
        loss_sum = mx.array(0.0, dtype=mx.float32)
        grad_accum = mlx_utils.tree_map(lambda p: p, grad_template)
        tokens_seen = 0.0

        for _ in range(accum_steps):
            input_ids, attention_mask, loss_mask = dataset.sample_batch(
                batch_size=args.batch_size, rng=rng
            )
            tokens_seen += _count_tokens(loss_mask, attention_mask, spec_len=args.spec_len)
            loss, grads = value_and_grad(input_ids, attention_mask, loss_mask)
            loss_sum = loss_sum + loss.astype(mx.float32)
            grad_accum = mlx_utils.tree_map(lambda a, b: a + b, grad_accum, grads)

        grad_accum = mlx_utils.tree_map(lambda g: g / float(accum_steps), grad_accum)
        if args.grad_clip > 0:
            grad_accum, grad_norm = optim.clip_grad_norm(grad_accum, max_norm=float(args.grad_clip))
        else:
            grad_norm = mx.array(0.0, dtype=mx.float32)

        optimizer.update(speculator, grad_accum)
        mx.eval(loss_sum, grad_norm)

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tok_s = tokens_seen / max(elapsed, 1e-6)
            loss_val = float((loss_sum / float(accum_steps)).item())
            print(f"[train] step={step} loss={loss_val:.4f} tok/s={tok_s:.2f}")
            start_time = time.time()

        if step % args.save_interval == 0 or step == int(args.max_steps):
            ckpt_dir = out_dir / "checkpoints" / f"step_{step:08d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_path = ckpt_dir / "speculator.safetensors"
            tmp_model = Path(os.fspath(model_path) + ".tmp.safetensors")
            speculator.save_weights(os.fspath(tmp_model))
            os.replace(tmp_model, model_path)

            opt_path = ckpt_dir / "optimizer.npz"
            tmp_opt = Path(os.fspath(opt_path) + ".tmp.npz")
            save_optimizer_state(optimizer, os.fspath(tmp_opt))
            os.replace(tmp_opt, opt_path)

            state_path = ckpt_dir / "train_state.json"
            state_path.write_text(
                json.dumps({"step": step, "timestamp": time.time()}, indent=2),
                encoding="utf-8",
            )
            print(f"[ckpt] saved {ckpt_dir}")


if __name__ == "__main__":
    main()
