from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, Sequence, cast

from ..data import iter_jsonl, resolve_jsonl_paths
from ..download import resolve_data_path_spec


def _encode_pretrain_sample(tokenizer, obj: Dict[str, Any]) -> str:
    if "text" in obj:
        return str(obj["text"])
    if "conversations" in obj:
        conversations = cast(Sequence[Dict[str, Any]], obj["conversations"])
        return tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
    return json.dumps(obj, ensure_ascii=False)


def _encode_sft_sample(tokenizer, obj: Dict[str, Any]) -> str:
    if "conversations" not in obj:
        raise ValueError("SFT task expects JSONL lines with a `conversations` field.")
    return tokenizer.apply_chat_template(
        cast(Sequence[Dict[str, Any]], obj["conversations"]),
        tokenize=False,
        add_generation_prompt=False,
    )


def _encode_dpo_pair(tokenizer, obj: Dict[str, Any]) -> tuple[list[int], list[int]]:
    if "chosen" not in obj or "rejected" not in obj:
        raise ValueError("DPO task expects JSONL lines with `chosen` and `rejected` fields.")
    chosen = cast(Sequence[Dict[str, Any]], obj["chosen"])
    rejected = cast(Sequence[Dict[str, Any]], obj["rejected"])
    chosen_prompt = tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
    rejected_prompt = tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
    chosen_ids = tokenizer.encode(chosen_prompt, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected_prompt, add_special_tokens=False)
    return chosen_ids, rejected_ids


def _as_ids(obj: Dict[str, Any]) -> list[int] | None:
    if "ids" not in obj:
        return None
    raw = obj["ids"]
    if not isinstance(raw, list):
        raise ValueError(f"`ids` must be a list, got: {type(raw).__name__}")
    return [int(t) for t in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretokenize JSONL datasets into `ids` for MLX training")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL file/dir/glob; can be comma-separated; supports `minimind:*` specs and URLs.",
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
    parser.add_argument("--task", type=str, choices=["pretrain", "sft", "r1", "dpo"], default="pretrain")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=0,
        help="Optional clip length (keeps up to seq_len+1 tokens, 0 = no clipping).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output JSONL path (will be overwritten if exists).",
    )
    parser.add_argument(
        "--keep_fields",
        action="store_true",
        help="Keep original JSON fields in output (default: write only `ids`).",
    )
    parser.add_argument(
        "--retokenize",
        action="store_true",
        help="Always re-tokenize even if input already has `ids`.",
    )
    parser.add_argument("--log_interval", type=int, default=10000)

    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Failed to import `transformers`. Install MLX training deps via "
            "`python3 -m pip install -r mlx_train/requirements.txt`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id.")

    data_spec = resolve_data_path_spec(
        args.data_path,
        task=args.task,
        data_dir=args.data_dir,
        hf_repo_id=args.hf_dataset_repo,
        hf_endpoint=args.hf_endpoint,
        force_download=args.force_download,
        max_download_mb=args.max_download_mb,
        data_source=args.data_source,
        ms_repo_id=args.ms_dataset_repo or args.hf_dataset_repo,
        ms_cache_dir=args.ms_cache_dir,
        ms_revision=args.ms_revision,
    )
    paths = resolve_jsonl_paths(data_spec)
    out_path = str(args.out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for obj in cast(Iterable[Dict[str, Any]], iter_jsonl(paths)):
            ids: list[int] | None = None
            if not bool(args.retokenize):
                ids = _as_ids(obj)

            out_obj: Dict[str, Any]
            if args.task == "dpo":
                if ids is not None:
                    raise ValueError("DPO pretokenize expects raw `chosen`/`rejected` fields, not `ids`.")
                chosen_ids, rejected_ids = _encode_dpo_pair(tokenizer, obj)
                if int(args.seq_len) > 0:
                    chosen_ids = chosen_ids[: int(args.seq_len) + 1]
                    rejected_ids = rejected_ids[: int(args.seq_len) + 1]
                if bool(args.keep_fields):
                    out_obj = dict(obj)
                    out_obj["chosen_ids"] = chosen_ids
                    out_obj["rejected_ids"] = rejected_ids
                else:
                    out_obj = {"chosen_ids": chosen_ids, "rejected_ids": rejected_ids}
            else:
                if ids is None:
                    if args.task == "pretrain":
                        text = _encode_pretrain_sample(tokenizer, obj)
                    else:
                        text = _encode_sft_sample(tokenizer, obj)
                    ids = tokenizer.encode(text, add_special_tokens=False)

                if int(args.seq_len) > 0:
                    ids = ids[: int(args.seq_len) + 1]

                if bool(args.keep_fields):
                    out_obj = dict(obj)
                    out_obj["ids"] = ids
                else:
                    out_obj = {"ids": ids}

            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n += 1
            if int(args.log_interval) > 0 and n % int(args.log_interval) == 0:
                dt = time.time() - t0
                print(f"[pretokenize] n={n} {n / max(dt, 1e-6):.0f} lines/s")

    dt = time.time() - t0
    print(f"[done] wrote={out_path} n={n} {n / max(dt, 1e-6):.0f} lines/s")


if __name__ == "__main__":
    main()
