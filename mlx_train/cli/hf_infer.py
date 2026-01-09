import argparse
from pathlib import Path
from typing import Optional


def _safe_name(repo: str) -> str:
    safe = repo.strip().lower().replace("/", "_")
    for ch in (":", "-", "."):
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def _default_model_dir(repo: str) -> Path:
    return Path("out") / "mlx_hf" / _safe_name(repo)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX inference for HF-converted models (via mlx-lm)")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--hf_repo", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise ImportError(
            "Missing `mlx-lm`. Install via `python3 -m pip install mlx-lm`."
        ) from exc

    if args.seed is not None:
        mx.random.seed(int(args.seed))

    model_dir = Path(args.model_dir) if args.model_dir else _default_model_dir(args.hf_repo)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model dir not found: {model_dir}. Convert first via `python3 -m mlx_train.hf_convert`."
        )

    model, tokenizer = load(str(model_dir))

    prompt_text = args.prompt
    if not args.no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampler = make_sampler(temp=float(args.temperature), top_p=float(args.top_p))
    text = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=int(args.max_tokens),
        sampler=sampler,
        verbose=bool(args.verbose),
    )
    if not args.verbose:
        print(text)


if __name__ == "__main__":
    main()
