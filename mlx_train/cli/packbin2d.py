from __future__ import annotations

import argparse
import json
import os
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast
from multiprocessing import Pool, cpu_count

from ..data import _encode_sample, _generate_sft_loss_mask, iter_jsonl, resolve_jsonl_paths
from ..download import resolve_data_path_spec


def _as_ids(obj: Dict[str, Any]) -> list[int] | None:
    raw = obj.get("ids")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"`ids` must be a list, got: {type(raw).__name__}")
    return [int(t) for t in raw]


def _write_int32_list(f, values: Sequence[int]) -> None:
    arr = array("I", [int(v) for v in values])
    if sys.byteorder != "little":
        arr.byteswap()
    f.write(arr.tobytes())


# Global worker state (initialized per process)
_worker_tokenizer = None
_worker_task = None
_worker_stride = None
_worker_retokenize = False


def _init_worker(tokenizer_path: str, tokenizer_type: str, task: str, stride: int, retokenize: bool):
    """Initialize tokenizer in each worker process."""
    global _worker_tokenizer, _worker_task, _worker_stride, _worker_retokenize
    from .tokenizer_utils import load_tokenizer
    _worker_tokenizer = load_tokenizer(tokenizer_path, tokenizer_type=tokenizer_type)
    _worker_task = task
    _worker_stride = stride
    _worker_retokenize = retokenize


def _process_sample(obj: Dict[str, Any]) -> List[int]:
    """Process a single sample in worker process."""
    global _worker_tokenizer, _worker_task, _worker_stride, _worker_retokenize

    ids = None if _worker_retokenize else _as_ids(obj)
    if ids is None:
        ids = _encode_sample(obj, tokenizer=_worker_tokenizer, task=_worker_task)
    ids = list(ids)

    if len(ids) > _worker_stride:
        ids = ids[:_worker_stride]
    elif len(ids) < _worker_stride:
        ids = ids + [int(_worker_tokenizer.pad_token_id)] * (_worker_stride - len(ids))

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack JSONL into fixed-length MLX bin2d format")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["auto", "hf", "rustbpe"],
        default="auto",
        help="Tokenizer backend: auto (prefer RustBPE), hf, or rustbpe.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL file/dir/glob; can be comma-separated; supports minimind:* specs and URLs.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Download/cache directory when `data_path` contains URLs or `minimind:*` specs.",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default="jingyaogong/minimind_dataset",
        help="HuggingFace dataset repo used by `minimind:*` specs.",
    )
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
    parser.add_argument(
        "--max_download_mb",
        type=int,
        default=2048,
        help="Safety guard for remote dataset downloads (MB); set 0 to disable.",
    )
    parser.add_argument("--task", type=str, choices=["pretrain", "sft", "r1"], default="pretrain")
    parser.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help="Sequence length (writes fixed length seq_len+1 ids per sample).",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        required=True,
        help="Output prefix for .ids2d.bin/.lbl.bin/.lbl.idx/.meta.json",
    )
    parser.add_argument(
        "--with_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store label positions for sparse_loss (sft/r1 only).",
    )
    parser.add_argument(
        "--retokenize",
        action="store_true",
        help="Always re-tokenize even if input already has `ids`.",
    )
    parser.add_argument("--log_interval", type=int, default=5000)
    parser.add_argument("--show_progress", action="store_true", help="Show tqdm progress bar if available.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of parallel workers for tokenization (0=auto, 1=single-threaded).",
    )
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for parallel processing.")

    args = parser.parse_args()

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues

    from .tokenizer_utils import load_tokenizer

    tokenizer = load_tokenizer(
        args.tokenizer_path,
        tokenizer_type=str(args.tokenizer_type),
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id.")

    data_spec = resolve_data_path_spec(
        args.data_path,
        task=args.task,
        data_dir=args.data_dir,
        hf_repo_id=args.hf_dataset_repo,
        hf_endpoint=args.hf_endpoint,
        force_download=bool(args.force_download),
        max_download_mb=int(args.max_download_mb),
        data_source=args.data_source,
        ms_repo_id=args.ms_dataset_repo or args.hf_dataset_repo,
        ms_cache_dir=args.ms_cache_dir,
        ms_revision=args.ms_revision,
    )
    paths = resolve_jsonl_paths(data_spec)
    print(f"[packbin2d] Found {len(paths)} file(s), using {num_workers} workers", flush=True)

    # Try to count total lines for progress bar
    total_lines = None
    if args.show_progress:
        try:
            print("[packbin2d] Counting lines...", end="", flush=True)
            total_lines = 0
            for p in paths:
                with open(p, "r", encoding="utf-8") as f:
                    total_lines += sum(1 for _ in f)
            print(f" {total_lines:,} lines", flush=True)
        except Exception:
            print(" (failed to count)", flush=True)
            total_lines = None

    # Try to use tqdm for progress
    tqdm_cls = None
    if args.show_progress:
        try:
            from tqdm import tqdm
            tqdm_cls = tqdm
        except ImportError:
            print("[packbin2d] Install tqdm for progress bar: pip install tqdm")

    out_prefix = str(args.out_prefix)
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    ids_bin_path = out_prefix + ".ids2d.bin"
    lbl_bin_path = out_prefix + ".lbl.bin"
    lbl_idx_path = out_prefix + ".lbl.idx"
    meta_path = out_prefix + ".meta.json"

    with_labels = bool(args.with_labels) and str(args.task) in ("sft", "r1")
    bos_id: Sequence[int] = []
    eos_id: Sequence[int] = []
    if with_labels:
        bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant", add_special_tokens=False)
        eos_id = tokenizer.encode(f"{tokenizer.eos_token}", add_special_tokens=False)

    n = 0
    lbl_offset = 0
    t0 = time.time()
    seq_len = int(args.seq_len)
    stride = seq_len + 1

    # Create progress bar if available
    pbar = None
    if tqdm_cls is not None:
        pbar = tqdm_cls(total=total_lines, desc="Processing", unit=" samples", dynamic_ncols=True)

    with open(ids_bin_path, "wb") as ids_bin:
        lbl_bin = open(lbl_bin_path, "wb") if with_labels else None
        lbl_idx = open(lbl_idx_path, "wb") if with_labels else None

        try:
            if num_workers > 1:
                # Parallel processing
                with Pool(
                    processes=num_workers,
                    initializer=_init_worker,
                    initargs=(args.tokenizer_path, str(args.tokenizer_type), args.task, stride, bool(args.retokenize)),
                ) as pool:
                    data_iter = iter_jsonl(paths)

                    for ids in pool.imap(_process_sample, data_iter, chunksize=args.chunk_size):
                        _write_int32_list(ids_bin, ids)

                        if with_labels and lbl_bin is not None and lbl_idx is not None:
                            loss_mask = _generate_sft_loss_mask(ids, bos_id, eos_id, len(ids))
                            mask = loss_mask[1:]
                            pos = [i for i, v in enumerate(mask) if int(v) != 0]
                            _write_int32_list(lbl_bin, pos)
                            lbl_idx.write(
                                (int(lbl_offset)).to_bytes(8, byteorder="little", signed=False)
                                + (int(len(pos))).to_bytes(8, byteorder="little", signed=False)
                            )
                            lbl_offset += len(pos)

                        n += 1
                        if pbar is not None:
                            pbar.update(1)
                        elif int(args.log_interval) > 0 and n % int(args.log_interval) == 0:
                            dt = time.time() - t0
                            rate = n / max(dt, 1e-6)
                            eta = (total_lines - n) / rate if total_lines else 0
                            eta_str = f" ETA: {eta/60:.1f}min" if total_lines else ""
                            pct = f"{100*n/total_lines:.1f}%" if total_lines else "?"
                            msg = f"\r[packbin2d] {n:,}/{total_lines or '?':,} ({pct}) {rate:.0f}/s{eta_str}    "
                            sys.stdout.write(msg)
                            sys.stdout.flush()
            else:
                # Single-threaded processing (original code path)
                for obj in cast(Iterable[Dict[str, Any]], iter_jsonl(paths)):
                    ids = None if bool(args.retokenize) else _as_ids(obj)
                    if ids is None:
                        ids = _encode_sample(obj, tokenizer=tokenizer, task=args.task)
                    ids = list(ids)
                    if len(ids) > stride:
                        ids = ids[:stride]
                    elif len(ids) < stride:
                        ids = ids + [int(tokenizer.pad_token_id)] * (stride - len(ids))

                    _write_int32_list(ids_bin, ids)

                    if with_labels and lbl_bin is not None and lbl_idx is not None:
                        loss_mask = _generate_sft_loss_mask(ids, bos_id, eos_id, len(ids))
                        mask = loss_mask[1:]
                        pos = [i for i, v in enumerate(mask) if int(v) != 0]
                        _write_int32_list(lbl_bin, pos)
                        lbl_idx.write(
                            (int(lbl_offset)).to_bytes(8, byteorder="little", signed=False)
                            + (int(len(pos))).to_bytes(8, byteorder="little", signed=False)
                        )
                        lbl_offset += len(pos)

                    n += 1
                    if pbar is not None:
                        pbar.update(1)
                    elif int(args.log_interval) > 0 and n % int(args.log_interval) == 0:
                        dt = time.time() - t0
                        rate = n / max(dt, 1e-6)
                        eta = (total_lines - n) / rate if total_lines else 0
                        eta_str = f" ETA: {eta/60:.1f}min" if total_lines else ""
                        pct = f"{100*n/total_lines:.1f}%" if total_lines else "?"
                        msg = f"\r[packbin2d] {n:,}/{total_lines or '?':,} ({pct}) {rate:.0f}/s{eta_str}    "
                        sys.stdout.write(msg)
                        sys.stdout.flush()
        finally:
            if pbar is not None:
                pbar.close()
            if lbl_bin is not None:
                lbl_bin.close()
            if lbl_idx is not None:
                lbl_idx.close()

    # Clear the progress line if we were using inline progress
    if pbar is None and args.show_progress and n > 0:
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    meta = {
        "version": 1,
        "format": "bin2d",
        "task": str(args.task),
        "num_samples": int(n),
        "byteorder": sys.byteorder,
        "seq_len": int(seq_len),
        "stride": int(stride),
        "ids": {"bin": os.path.basename(ids_bin_path), "dtype": "int32"},
        "labels": None,
    }
    if with_labels:
        meta["labels"] = {
            "bin": os.path.basename(lbl_bin_path),
            "idx": os.path.basename(lbl_idx_path),
            "dtype": "int32",
        }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"[done] wrote={meta_path} n={n:,} {n / max(dt, 1e-6):.0f} lines/s ({dt:.1f}s)")


if __name__ == "__main__":
    main()
