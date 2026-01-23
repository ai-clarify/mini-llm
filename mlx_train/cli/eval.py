from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from ..config import MiniLLMConfig
from ..data import (
    Bin2DDataset,
    BinDataset,
    detect_bin_format,
    make_batch_iterator,
    resolve_bin_prefix,
    resolve_jsonl_paths,
)
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


def _run_cmd_first(cmds: Sequence[Sequence[str]]) -> Optional[str]:
    for cmd in cmds:
        try:
            return subprocess.check_output(list(cmd), text=True).strip()
        except Exception:
            continue
    return None


def _read_vm_stat() -> Optional[str]:
    return _run_cmd_first(
        [
            ["/usr/bin/vm_stat"],
            ["/usr/sbin/vm_stat"],
            ["vm_stat"],
        ]
    )


def _parse_vm_stat() -> Tuple[Optional[int], Dict[str, int]]:
    out = _read_vm_stat()
    if not out:
        return None, {}
    page_size = None
    pages: Dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if "page size of" in line and "bytes" in line:
            m = re.search(r"page size of (\d+) bytes", line)
            if m:
                page_size = int(m.group(1))
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip().strip(".")
        try:
            pages[key] = int(val)
        except ValueError:
            continue
    return page_size, pages


def _estimate_available_mem_bytes(*, mode: str = "available") -> Optional[int]:
    page_size, pages = _parse_vm_stat()
    if page_size is None or not pages:
        return None
    mode = str(mode).lower().strip()
    if mode == "free":
        keys = ("pages free", "pages speculative")
    else:
        keys = ("pages free", "pages inactive", "pages speculative", "pages purgeable")
    avail_pages = 0
    for key in keys:
        if key in pages:
            avail_pages += pages[key]
    if avail_pages <= 0:
        return None
    return int(avail_pages) * int(page_size)


def _estimate_bin_bytes(path: str) -> Optional[int]:
    prefix = resolve_bin_prefix(path)
    meta_path = Path(prefix + ".meta.json")
    files: List[Path] = []
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            base_dir = meta_path.parent
            ids = meta.get("ids", {})
            ids_bin = ids.get("bin")
            if ids_bin:
                files.append(base_dir / str(ids_bin))
            labels = meta.get("labels", {})
            lbl_bin = labels.get("bin")
            if lbl_bin:
                files.append(base_dir / str(lbl_bin))
        except Exception:
            files = []
    if not files:
        ids_bin = Path(prefix + ".ids.bin")
        if ids_bin.exists():
            files.append(ids_bin)
        lbl_bin = Path(prefix + ".lbl.bin")
        if lbl_bin.exists():
            files.append(lbl_bin)
    if not files:
        return None
    total = 0
    for p in files:
        try:
            total += int(p.stat().st_size)
        except Exception:
            continue
    return total or None


def _choose_bin_cache(
    path: str,
    cache: str,
    *,
    avail_bytes: Optional[int],
    reserve_mb: int = 2048,
    ratio: float = 0.35,
) -> str:
    cache = str(cache).lower().strip()
    if cache != "auto":
        return cache
    bin_bytes = _estimate_bin_bytes(path)
    if bin_bytes is None or avail_bytes is None:
        return "mmap"
    reserve = int(reserve_mb) * 1024 * 1024
    budget = max(0, int(avail_bytes) - reserve)
    if budget <= 0:
        return "mmap"
    if bin_bytes <= int(budget * float(ratio)):
        return "memory"
    return "mmap"


def loss_sum_and_tokens(
    model: MiniLLMForCausalLM, x: mx.array, y: mx.array, loss_mask: mx.array
) -> Tuple[mx.array, mx.array]:
    hidden = model.model(x)  # [B, T, H]
    return chunked_ce_loss_sum_and_tokens(
        hidden=hidden,
        lm_head_weight=model.lm_head_weight(),
        labels=y,
        loss_mask=loss_mask,
        chunk_size=0,
        logit_softcap=float(model.config.logit_softcap),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) eval: average loss / perplexity")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["auto", "hf", "rustbpe"],
        default="auto",
        help="Tokenizer backend: auto (prefer RustBPE), hf, or rustbpe.",
    )
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
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["auto", "jsonl", "bin", "bin2d"],
        default="auto",
        help="Dataset format: jsonl, bin (mlx bin format), bin2d (fixed-length), or auto-detect.",
    )
    parser.add_argument(
        "--bin_cache",
        type=str,
        choices=["mmap", "memory", "auto"],
        default="mmap",
        help="Binary dataset cache mode: mmap, memory, or auto (choose based on free RAM).",
    )
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

    from .tokenizer_utils import load_tokenizer

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
    bin_dataset: Optional[BinDataset] = None
    bin2d_dataset: Optional[Bin2DDataset] = None
    paths: Sequence[str] = []
    fmt = str(args.data_format)
    if fmt == "auto":
        detected = detect_bin_format(str(resolved))
        fmt = detected if detected is not None else "jsonl"
    if fmt == "bin2d":
        avail_for_cache = _estimate_available_mem_bytes(mode="available")
        bin_cache = _choose_bin_cache(
            str(resolved), str(args.bin_cache), avail_bytes=avail_for_cache
        )
        if str(args.bin_cache) == "auto":
            print(f"[data] bin_cache=auto resolved to {bin_cache}")
        bin2d_dataset = Bin2DDataset(str(resolved), cache=str(bin_cache))
    elif fmt == "bin":
        avail_for_cache = _estimate_available_mem_bytes(mode="available")
        bin_cache = _choose_bin_cache(
            str(resolved), str(args.bin_cache), avail_bytes=avail_for_cache
        )
        if str(args.bin_cache) == "auto":
            print(f"[data] bin_cache=auto resolved to {bin_cache}")
        bin_dataset = BinDataset(str(resolved), cache=str(bin_cache))
    else:
        paths = resolve_jsonl_paths(resolved)

    tokenizer = load_tokenizer(
        args.tokenizer_path,
        tokenizer_type=str(args.tokenizer_type),
    )
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
        bin_dataset=bin_dataset,
        bin2d_dataset=bin2d_dataset,
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
