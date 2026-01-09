#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.models import qwen3 as qwen3_model
    from mlx_lm.models.base import create_attention_mask
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "Missing MLX dependencies. Install via `python3 -m pip install mlx-lm`. "
        "Note: mlx-lm currently pins transformers==5.0.0rc1; use a clean venv if needed."
    ) from exc


def _safe_name(repo: str) -> str:
    safe = repo.strip().lower().replace("/", "_")
    for ch in (":", "-", "."):
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def _default_model_dir(repo: str) -> Path:
    return Path("out") / "mlx_hf" / _safe_name(repo)


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    return "\n\n".join([str(m.get("content", "")) for m in messages])


def sample_next_token(logits: mx.array, *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits).item())
    logits = logits / float(temperature)
    if top_p >= 1.0:
        token = mx.random.categorical(logits, axis=-1)
        return int(token.item())

    sorted_idx = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)

    remove = cumprobs > float(top_p)
    remove = mx.concatenate([mx.array([False]), remove[:-1]], axis=-1)
    neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
    filtered_logits = mx.where(remove, neg_inf, sorted_logits)
    picked = mx.random.categorical(filtered_logits, axis=-1)
    picked_i = int(picked.item())
    return int(sorted_idx[picked_i].item())


def _project_logits(target: nn.Module, hidden: mx.array) -> mx.array:
    if hasattr(target, "args") and getattr(target.args, "tie_word_embeddings", False):
        return target.model.embed_tokens.as_linear(hidden)
    if hasattr(target, "lm_head"):
        return target.lm_head(hidden)
    if hasattr(target, "model") and hasattr(target.model, "lm_head"):
        return target.model.lm_head(hidden)
    raise AttributeError("Unable to locate LM head for target model")


def _pick_latest_checkpoint(ckpt_root: Path) -> Optional[Path]:
    if not ckpt_root.is_dir():
        return None
    best = None
    best_step = -1
    for child in ckpt_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("step_"):
            continue
        try:
            step = int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if step > best_step:
            best_step = step
            best = child
    return best


class Eagle3Speculator(nn.Module):
    def __init__(
        self,
        *,
        args: qwen3_model.ModelArgs,
        spec_len: int,
        spec_layers: int,
        init_weight: Optional[mx.array],
    ) -> None:
        super().__init__()
        self.layers = [qwen3_model.TransformerBlock(args=args) for _ in range(int(spec_layers))]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.heads = [nn.Linear(args.hidden_size, args.vocab_size, bias=False) for _ in range(int(spec_len))]
        if init_weight is not None:
            for head in self.heads:
                head.weight = init_weight

    def __call__(self, hidden: mx.array, attention_mask: Optional[mx.array] = None) -> List[mx.array]:
        x = hidden
        if attention_mask is not None:
            x = x * attention_mask[..., None]
        mask = None
        if x.shape[1] > 1:
            mask = create_attention_mask(x, cache=None)
        for layer in self.layers:
            x = layer(x, mask=mask, cache=None)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def load_speculator(
    *,
    target: nn.Module,
    speculator_dir: Path,
    speculator_ckpt: Optional[Path],
    spec_len: int,
    spec_layers: int,
) -> Tuple[Eagle3Speculator, int]:
    cfg_path = speculator_dir / "speculator_config.json"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        spec_len = int(cfg.get("spec_len", spec_len))
        spec_layers = int(cfg.get("spec_layers", spec_layers))

    init_weight = None
    if hasattr(target, "args") and getattr(target.args, "tie_word_embeddings", False):
        init_weight = target.model.embed_tokens.weight
    elif hasattr(target, "lm_head"):
        init_weight = target.lm_head.weight

    if not hasattr(target, "args"):
        raise AttributeError("Target model missing args; expected mlx-lm Qwen3 model")

    speculator = Eagle3Speculator(
        args=target.args,
        spec_len=spec_len,
        spec_layers=spec_layers,
        init_weight=init_weight,
    )

    if speculator_ckpt is None:
        latest = _pick_latest_checkpoint(speculator_dir / "checkpoints")
        if latest is None:
            raise FileNotFoundError(f"No checkpoints under {speculator_dir}/checkpoints")
        speculator_ckpt = latest / "speculator.safetensors"
    if not speculator_ckpt.is_file():
        raise FileNotFoundError(f"Speculator checkpoint not found: {speculator_ckpt}")

    speculator.load_weights(str(speculator_ckpt))
    speculator.eval()
    return speculator, spec_len


def _load_target(
    *,
    model_dir: Optional[str],
    hf_repo: str,
    revision: Optional[str],
) -> Tuple[nn.Module, Any]:
    path: str
    if model_dir:
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model dir not found: {model_path}")
        path = str(model_path)
    else:
        model_path = _default_model_dir(hf_repo)
        path = str(model_path) if model_path.exists() else hf_repo
    model, tokenizer = load(path, revision=revision)
    model.eval()
    return model, tokenizer


def baseline_decode(
    *,
    target: nn.Module,
    input_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    cache: Optional[List[Any]] = None,
    last_hidden: Optional[mx.array] = None,
) -> List[int]:
    output_ids = list(input_ids)
    if cache is None or last_hidden is None:
        cache = mlx_cache.make_prompt_cache(target.model)
        prompt = mx.array([output_ids], dtype=mx.int32)
        hidden = target.model(prompt, cache=cache)
        mx.eval(hidden)
        last_hidden = hidden[:, -1:, :]

    for _ in range(int(max_new_tokens)):
        logits = _project_logits(target, last_hidden)[:, -1, :]
        mx.eval(logits)
        token = sample_next_token(logits, temperature=temperature, top_p=top_p)
        output_ids.append(int(token))
        if eos_token_id is not None and int(token) == int(eos_token_id):
            break
        step = mx.array([[int(token)]], dtype=mx.int32)
        hidden = target.model(step, cache=cache)
        mx.eval(hidden)
        last_hidden = hidden[:, -1:, :]

    return output_ids


def speculative_decode(
    *,
    target: nn.Module,
    speculator: Eagle3Speculator,
    input_ids: List[int],
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
    optimized: bool,
    max_consecutive_misses: int = 2,
) -> List[int]:
    output_ids = list(input_ids)
    produced = 0
    consecutive_misses = 0

    cache = mlx_cache.make_prompt_cache(target.model) if use_cache else None
    prompt = mx.array([output_ids], dtype=mx.int32)
    hidden = target.model(prompt, cache=cache)
    mx.eval(hidden)
    last_hidden = hidden[:, -1:, :]

    while produced < int(max_new_tokens):
        logits_list = speculator(last_hidden)
        mx.eval(*logits_list)
        draft_tokens = [
            sample_next_token(logits_list[i][0, -1, :], temperature=temperature, top_p=top_p)
            for i in range(int(spec_len))
        ]

        if use_cache:
            draft_arr = mx.array([draft_tokens], dtype=mx.int32)
            block_hidden = target.model(draft_arr, cache=cache)
            block_logits = _project_logits(target, block_hidden)
            mx.eval(block_logits)
        else:
            full = mx.array([output_ids + draft_tokens], dtype=mx.int32)
            full_hidden = target.model(full, cache=None)
            full_logits = _project_logits(target, full_hidden)
            mx.eval(full_logits)
            block_logits = full_logits[:, -len(draft_tokens) :, :]
            block_hidden = full_hidden[:, -len(draft_tokens) :, :]

        posterior_tokens = [
            sample_next_token(block_logits[0, i, :], temperature=temperature, top_p=top_p)
            for i in range(len(draft_tokens))
        ]

        accept_len = 0
        for i in range(len(draft_tokens)):
            if draft_tokens[i] == posterior_tokens[i]:
                accept_len += 1
            else:
                break

        if accept_len == 0:
            consecutive_misses += 1
        else:
            consecutive_misses = 0

        new_tokens = list(draft_tokens[:accept_len])
        if accept_len < len(draft_tokens):
            new_tokens.append(posterior_tokens[accept_len])

        remaining = int(max_new_tokens) - produced
        if len(new_tokens) > remaining:
            new_tokens = new_tokens[:remaining]

        if not new_tokens:
            break

        output_ids.extend(int(t) for t in new_tokens)
        produced += len(new_tokens)

        if eos_token_id is not None and int(eos_token_id) in new_tokens:
            eos_idx = new_tokens.index(int(eos_token_id))
            tail = len(new_tokens) - eos_idx - 1
            if tail > 0:
                output_ids = output_ids[:-tail]
            break

        if use_cache:
            if accept_len == len(draft_tokens):
                last_hidden = block_hidden[:, -1:, :]
            else:
                if not mlx_cache.can_trim_prompt_cache(cache):
                    cache = mlx_cache.make_prompt_cache(target.model)
                    prompt = mx.array([output_ids], dtype=mx.int32)
                    hidden = target.model(prompt, cache=cache)
                    mx.eval(hidden)
                    last_hidden = hidden[:, -1:, :]
                else:
                    mlx_cache.trim_prompt_cache(cache, len(draft_tokens) - accept_len)
                    step = mx.array([[int(new_tokens[-1])]], dtype=mx.int32)
                    hidden = target.model(step, cache=cache)
                    mx.eval(hidden)
                    last_hidden = hidden[:, -1:, :]
        else:
            prompt = mx.array([output_ids], dtype=mx.int32)
            hidden = target.model(prompt, cache=None)
            mx.eval(hidden)
            last_hidden = hidden[:, -1:, :]

        if optimized and consecutive_misses >= max_consecutive_misses:
            break

    if optimized and produced < int(max_new_tokens):
        output_ids = baseline_decode(
            target=target,
            input_ids=output_ids,
            max_new_tokens=int(max_new_tokens) - produced,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            cache=cache if use_cache else None,
            last_hidden=last_hidden,
        )

    return output_ids


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--hf_repo", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--speculator_dir", type=str, default="out/eagle3_speculator_mlx/qwen3_0.6b")
    parser.add_argument("--speculator_ckpt", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--spec_len", type=int, default=7)
    parser.add_argument("--spec_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no_speculator", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")
    return parser


def run_cli(*, optimized: bool) -> None:
    parser = build_arg_parser(
        description="Speculative decoding with EAGLE-3 speculator (MLX backend)."
        if not optimized
        else "Speculative decoding with EAGLE-3 speculator (MLX backend, optimized)."
    )
    args = parser.parse_args()

    mx.random.seed(int(args.seed))

    target, tokenizer = _load_target(
        model_dir=args.model_dir,
        hf_repo=args.hf_repo,
        revision=args.revision,
    )

    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    if args.no_chat_template:
        prompt_text = args.prompt
    else:
        prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    speculator = None
    spec_len = args.spec_len
    if not args.no_speculator:
        speculator_dir = Path(args.speculator_dir)
        speculator_ckpt = Path(args.speculator_ckpt) if args.speculator_ckpt else None
        speculator, spec_len = load_speculator(
            target=target,
            speculator_dir=speculator_dir,
            speculator_ckpt=speculator_ckpt,
            spec_len=args.spec_len,
            spec_layers=args.spec_layers,
        )

    start = time.time()
    if speculator is None:
        output_ids = baseline_decode(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        output_ids = speculative_decode(
            target=target,
            speculator=speculator,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            spec_len=spec_len,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=not args.no_cache,
            optimized=optimized,
        )
    elapsed = time.time() - start
    out_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(out_text)
    print(f"[time] {elapsed:.2f}s")
