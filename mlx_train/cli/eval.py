from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx

from ..config import MiniLLMConfig
from ..data import make_batch_iterator, resolve_jsonl_paths
from ..download import resolve_data_path_spec
from ..models import MiniLLMForCausalLM
from ..ops.loss import chunked_ce_loss_sum_and_tokens
from ..nn.lora import merge_lora


def load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data).__name__}")
    return MiniLLMConfig.from_dict(data)


def loss_sum_and_tokens(
    model: MiniLLMForCausalLM, x: mx.array, y: mx.array, loss_mask: mx.array
) -> Tuple[mx.array, mx.array]:
    hidden = model.model(x)  # [B, T, H]
    return chunked_ce_loss_sum_and_tokens(
        hidden=hidden,
        lm_head_weight=model.model.embed_tokens.weight,
        labels=y,
        loss_mask=loss_mask,
        chunk_size=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) eval: average loss / perplexity")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory produced by mlx_train/train.py (contains model.safetensors + config.json).",
    )
    parser.add_argument("--task", type=str, default="pretrain", choices=["pretrain", "sft"])
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL file/dir/glob; can be comma-separated. Supports minimind:* specs (downloads if needed).",
    )
    parser.add_argument("--data_dir", type=str, default="./dataset/minimind")
    parser.add_argument("--hf_repo_id", type=str, default="jingyaogong/minimind_dataset")
    parser.add_argument("--hf_endpoint", type=str, default=None)
    parser.add_argument(
        "--data_source",
        type=str,
        default=os.environ.get("MINIMIND_DATA_SOURCE", "modelscope"),
        help="Dataset source for `minimind:*` specs: modelscope|hf (default: MINIMIND_DATA_SOURCE or modelscope).",
    )
    parser.add_argument(
        "--ms_dataset_repo",
        type=str,
        default=os.environ.get("MINIMIND_DATA_REPO", "gongjy/minimind_dataset"),
        help="ModelScope dataset repo used by `minimind:*` specs (default: MINIMIND_DATA_REPO or gongjy/minimind_dataset).",
    )
    parser.add_argument(
        "--ms_cache_dir",
        type=str,
        default=os.environ.get("MINIMIND_MS_CACHE"),
        help="ModelScope cache directory (default: MINIMIND_MS_CACHE).",
    )
    parser.add_argument("--ms_revision", type=str, default=None, help="Optional ModelScope revision/tag.")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--max_download_mb", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle_buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Evaluate at most N batches (0 = full dataset; default: 100).",
    )
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="Compile the forward loss for faster eval.")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Failed to import `transformers`. Install MLX training deps via "
            "`python3 -m pip install -r mlx_train/requirements.txt`."
        ) from e

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint must be a directory: {ckpt}")

    resolved = resolve_data_path_spec(
        args.data_path,
        task=args.task,
        data_dir=args.data_dir,
        hf_repo_id=args.hf_repo_id,
        hf_endpoint=args.hf_endpoint,
        force_download=bool(args.force_download),
        max_download_mb=int(args.max_download_mb),
        data_source=args.data_source,
        ms_repo_id=args.ms_dataset_repo or args.hf_repo_id,
        ms_cache_dir=args.ms_cache_dir,
        ms_revision=args.ms_revision,
    )
    paths = resolve_jsonl_paths(resolved)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    cfg = load_config(ckpt)
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(ckpt / "model.safetensors"))
    model.eval()
    if int(cfg.lora_r) > 0:
        merge_lora(model)

    def eval_step(x: mx.array, y: mx.array, m: mx.array) -> Tuple[mx.array, mx.array]:
        return loss_sum_and_tokens(model, x, y, m)

    eval_fn = eval_step
    if args.compile:
        eval_fn = mx.compile(eval_fn, inputs={"model": model})
        x0 = mx.zeros((args.batch_size, args.seq_len), dtype=mx.int32)
        y0 = mx.zeros((args.batch_size, args.seq_len), dtype=mx.int32)
        m0 = mx.ones((args.batch_size, args.seq_len), dtype=mx.float32)
        out0 = eval_fn(x0, y0, m0)
        mx.eval(out0)

    it = make_batch_iterator(
        paths=paths,
        tokenizer=tokenizer,
        task=args.task,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )

    total_loss = 0.0
    total_tokens = 0.0
    batches = 0
    t0 = time.time()
    for batch in it:
        x = mx.array(batch.x, dtype=mx.int32)
        y = mx.array(batch.y, dtype=mx.int32)
        m = mx.array(batch.loss_mask, dtype=mx.float32)
        loss_sum, tokens = eval_fn(x, y, m)
        mx.eval(loss_sum, tokens)

        total_loss += float(loss_sum.item())
        total_tokens += float(tokens.item())
        batches += 1

        if args.log_interval > 0 and batches % args.log_interval == 0:
            avg = total_loss / max(total_tokens, 1.0)
            elapsed = max(time.time() - t0, 1e-9)
            tok_s = total_tokens / elapsed
            print(f"[eval] batches={batches} avg_loss={avg:.4f} tok/s={tok_s:.0f}")

        if args.max_batches > 0 and batches >= args.max_batches:
            break

    avg_loss = total_loss / max(total_tokens, 1.0)
    ppl = math.exp(min(avg_loss, 100.0))
    elapsed = max(time.time() - t0, 1e-9)
    tok_s = total_tokens / elapsed
    print(
        f"[eval] done batches={batches} tokens={int(total_tokens)} avg_loss={avg_loss:.4f} ppl={ppl:.2f} tok/s={tok_s:.0f}"
    )


if __name__ == "__main__":
    main()
