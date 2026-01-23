from __future__ import annotations

import os
from typing import Any


def load_tokenizer(tokenizer_path: str, *, tokenizer_type: str = "auto") -> Any:
    kind = str(tokenizer_type).lower().strip()
    if kind not in ("auto", "hf", "rustbpe"):
        raise ValueError(f"Unknown tokenizer_type={tokenizer_type}")

    tok_dir = os.fspath(tokenizer_path)
    wants_rust = kind == "rustbpe" or kind == "auto"
    if wants_rust:
        try:
            from tokenizer import RustBPETokenizer

            has_rust = os.path.exists(os.path.join(tok_dir, "tokenizer.pkl"))
            if kind == "rustbpe" or has_rust:
                return RustBPETokenizer.from_directory(tok_dir)
        except Exception as exc:
            if kind == "rustbpe":
                raise
            print(f"[warn] rustbpe tokenizer unavailable ({exc}); falling back to HF")

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tok_dir)
