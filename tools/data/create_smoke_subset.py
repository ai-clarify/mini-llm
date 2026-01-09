#!/usr/bin/env python3
"""Create small JSONL slices for local smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a truncated JSONL file for smoke testing")
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=32, help="Maximum number of lines to copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero for smoke testing")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            copied += 1
            if copied >= args.limit:
                break

    if copied == 0:
        raise RuntimeError(f"No records copied from {args.input}")


if __name__ == "__main__":
    main()
