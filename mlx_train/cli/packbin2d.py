from __future__ import annotations

import argparse
import json
import os
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, cast

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
    parser.add_argument("--log_interval", type=int, default=10000)

    args = parser.parse_args()

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
    with open(ids_bin_path, "wb") as ids_bin:
        lbl_bin = open(lbl_bin_path, "wb") if with_labels else None
        lbl_idx = open(lbl_idx_path, "wb") if with_labels else None
        try:
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
                if int(args.log_interval) > 0 and n % int(args.log_interval) == 0:
                    dt = time.time() - t0
                    print(f"[packbin2d] n={n} {n / max(dt, 1e-6):.0f} lines/s")
        finally:
            if lbl_bin is not None:
                lbl_bin.close()
            if lbl_idx is not None:
                lbl_idx.close()

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
    print(f"[done] wrote={meta_path} n={n} {n / max(dt, 1e-6):.0f} lines/s")


if __name__ == "__main__":
    main()
