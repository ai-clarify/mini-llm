#!/usr/bin/env python
"""Train a RustBPE tokenizer on MiniLLM datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tokenizer import RustBPETokenizer
from utils.conversation import conversation_to_template, normalize_conversation


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def yield_text(record: dict, data_format: str) -> Iterator[str]:
    if data_format == "pretrain":
        if "text" in record:
            yield str(record["text"])
            return
        try:
            messages = normalize_conversation(record)
        except Exception:
            return
        yield conversation_to_template(messages, include_bos=True)
        return

    if data_format == "sft":
        try:
            messages = normalize_conversation(record)
        except Exception:
            return
        yield conversation_to_template(messages, include_bos=True)
        return

    if data_format == "dpo":
        chosen = record.get("chosen") or record.get("chosen_conversation")
        rejected = record.get("rejected") or record.get("rejected_conversation")
        if chosen is None or rejected is None:
            return
        try:
            chosen_messages = normalize_conversation({"conversations": chosen})
            rejected_messages = normalize_conversation({"conversations": rejected})
        except Exception:
            return
        yield conversation_to_template(chosen_messages, include_bos=True)
        yield conversation_to_template(rejected_messages, include_bos=True)
        return

    raise ValueError(f"Unknown data format: {data_format}")


def stream_text(paths: Sequence[Path], data_format: str) -> Iterator[str]:
    for path in paths:
        for record in iter_jsonl(path):
            yield from yield_text(record, data_format)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RustBPE tokenizer")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files")
    parser.add_argument("--output", type=Path, default=Path("out/tokenizer"), help="Directory to save the tokenizer")
    parser.add_argument("--format", choices=["pretrain", "sft", "dpo"], default="pretrain", help="Dataset format")
    parser.add_argument("--vocab-size", type=int, default=6400, help="Tokenizer vocabulary size including special tokens")
    args = parser.parse_args()

    print(f"Training RustBPE tokenizer on {len(args.inputs)} file(s)...")
    tokenizer = RustBPETokenizer.train_from_iterator(stream_text(args.inputs, args.format), args.vocab_size)
    tokenizer.save(args.output)
    print(f"Tokenizer saved to {args.output}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
