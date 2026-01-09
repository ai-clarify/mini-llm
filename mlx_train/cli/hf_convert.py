import argparse
from pathlib import Path


def _safe_name(repo: str) -> str:
    safe = repo.strip().lower().replace("/", "_")
    for ch in (":", "-", "."):
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def _default_out_dir(repo: str) -> Path:
    return Path("out") / "mlx_hf" / _safe_name(repo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF Qwen3 (or any mlx-lm model) to MLX format")
    parser.add_argument("--hf_repo", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "out_name",
        nargs="?",
        help="Optional output dir name (joined with --out_dir if provided).",
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--q_bits", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=64)
    parser.add_argument("--q_mode", type=str, default="affine")
    parser.add_argument("--dtype", type=str, default=None, help="float16|bfloat16|float32")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if args.out_name:
            out_dir = out_dir / args.out_name
    else:
        if args.out_name:
            out_dir = Path(args.out_name)
        else:
            out_dir = _default_out_dir(args.hf_repo)

    if out_dir.exists():
        if args.skip_if_exists:
            print(f"[hf_convert] skip (exists): {out_dir}")
            return
        raise FileExistsError(
            f"Output dir exists: {out_dir}. Remove it or use --skip_if_exists."
        )

    try:
        from mlx_lm import convert
    except ImportError as exc:
        raise ImportError(
            "Missing `mlx-lm`. Install via `python3 -m pip install mlx-lm`."
        ) from exc

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.dtype and args.dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    convert(
        hf_path=args.hf_repo,
        mlx_path=str(out_dir),
        quantize=bool(args.quantize),
        q_group_size=int(args.q_group_size) if args.q_group_size else 64,
        q_bits=int(args.q_bits) if args.q_bits else 4,
        q_mode=str(args.q_mode),
        dtype=args.dtype if args.dtype else None,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
    )

    print(f"[hf_convert] done: {out_dir}")


if __name__ == "__main__":
    main()
