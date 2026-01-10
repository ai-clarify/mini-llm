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
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("Missing MLX core dependencies. Install mlx first.") from exc


def _load_qwen3_deps():
    try:
        from mlx_lm import load as qwen3_load
        from mlx_lm.models import cache as qwen3_cache
        from mlx_lm.models import qwen3 as qwen3_model
        from mlx_lm.models.base import create_attention_mask
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "Missing mlx-lm for Qwen3 inference/training. Install via `python3 -m pip install mlx-lm`. "
            "Note: mlx-lm currently pins transformers==5.0.0rc1; use a clean venv if needed."
        ) from exc
    return qwen3_load, qwen3_cache, qwen3_model, create_attention_mask


def _load_minillm_deps():
    try:
        from transformers import AutoTokenizer
        from mlx_train.config import MiniLLMConfig
        from mlx_train.models import MiniLLMForCausalLM
        from mlx_train.models.minillm import MiniLLMBlock, RMSNorm
    except Exception as exc:  # pragma: no cover - dependency check
        raise ImportError("Missing MiniLLM MLX dependencies; ensure mlx_train and transformers are available.") from exc
    return AutoTokenizer, MiniLLMConfig, MiniLLMForCausalLM, MiniLLMBlock, RMSNorm


def _safe_name(repo: str) -> str:
    safe = repo.strip().lower().replace("/", "_")
    for ch in (":", "-", "."):
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def _default_model_dir(repo: str) -> Path:
    return Path("out") / "mlx_hf" / _safe_name(repo)


def _count_params_mlx(model: nn.Module) -> Optional[int]:
    try:
        return int(sum(int(p.size) for p in model.parameters()))
    except Exception:
        return None


def _auto_spec_config(param_count: Optional[int]) -> Tuple[int, int]:
    if not param_count or param_count <= 0:
        return 1, 1
    params_b = float(param_count) / 1e9
    if params_b <= 1.0:
        return 1, 1
    if params_b <= 3.0:
        return 2, 1
    if params_b <= 7.0:
        return 3, 2
    if params_b <= 13.0:
        return 4, 2
    return 5, 2


def _resolve_spec_config(
    spec_len: Optional[int], spec_layers: Optional[int], *, param_count: Optional[int]
) -> Tuple[int, int]:
    auto_len, auto_layers = _auto_spec_config(param_count)
    if spec_len is None or int(spec_len) <= 0:
        spec_len = auto_len
    if spec_layers is None or int(spec_layers) <= 0:
        spec_layers = auto_layers
    return int(spec_len), int(spec_layers)


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


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


def _project_logits_qwen3(target: nn.Module, hidden: mx.array) -> mx.array:
    if target.args.tie_word_embeddings:
        return target.model.embed_tokens.as_linear(hidden)
    return target.lm_head(hidden)


def _project_logits_minillm(target: nn.Module, hidden: mx.array) -> mx.array:
    return hidden @ target.model.embed_tokens.weight.transpose()


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


_QWEN3_DEPS = None
_MINILLM_DEPS = None


def _get_qwen3_deps():
    global _QWEN3_DEPS
    if _QWEN3_DEPS is None:
        _QWEN3_DEPS = _load_qwen3_deps()
    return _QWEN3_DEPS


def _get_minillm_deps():
    global _MINILLM_DEPS
    if _MINILLM_DEPS is None:
        _MINILLM_DEPS = _load_minillm_deps()
    return _MINILLM_DEPS


class LowRankHead(nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, int(rank), bias=False)
        self.out = nn.Linear(int(rank), vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out(self.proj(x))


class Qwen3Speculator(nn.Module):
    def __init__(
        self,
        *,
        args: Any,
        spec_len: int,
        spec_layers: int,
        init_weight: Optional[mx.array],
        head_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        _, _, qwen3_model, create_attention_mask = _get_qwen3_deps()
        self._create_attention_mask = create_attention_mask
        self.layers = [qwen3_model.TransformerBlock(args=args) for _ in range(int(spec_layers))]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.heads = [LowRankHead(hidden_size=args.hidden_size, vocab_size=args.vocab_size, rank=rank) for _ in range(int(spec_len))]
        else:
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
            mask = self._create_attention_mask(x, cache=None)
        for layer in self.layers:
            x = layer(x, mask=mask, cache=None)
        x = self.norm(x)
        return [head(x) for head in self.heads]


class MiniLLMSpeculator(nn.Module):
    def __init__(
        self,
        *,
        config: Any,
        spec_len: int,
        spec_layers: int,
        init_weight: Optional[mx.array],
        head_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        _, _, _, MiniLLMBlock, RMSNorm = _get_minillm_deps()
        self.layers = [MiniLLMBlock(i, config) for i in range(int(spec_layers))]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.heads = [
                LowRankHead(hidden_size=config.hidden_size, vocab_size=config.vocab_size, rank=rank)
                for _ in range(int(spec_len))
            ]
        else:
            self.heads = [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(int(spec_len))]
            if init_weight is not None:
                for head in self.heads:
                    head.weight = init_weight

    def __call__(self, hidden: mx.array, attention_mask: Optional[mx.array] = None) -> List[mx.array]:
        x = hidden
        if attention_mask is not None:
            x = x * attention_mask[..., None]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def build_speculator(
    *,
    target_arch: str,
    target: nn.Module,
    spec_len: int,
    spec_layers: int,
    head_rank: Optional[int] = None,
) -> nn.Module:
    init_weight = None
    if target_arch == "minillm":
        init_weight = target.model.embed_tokens.weight
    else:
        if target.args.tie_word_embeddings:
            init_weight = target.model.embed_tokens.weight
        else:
            init_weight = target.lm_head.weight

    if target_arch == "minillm":
        return MiniLLMSpeculator(
            config=target.config,
            spec_len=spec_len,
            spec_layers=spec_layers,
            init_weight=init_weight,
            head_rank=head_rank,
        )

    return Qwen3Speculator(
        args=target.args,
        spec_len=spec_len,
        spec_layers=spec_layers,
        init_weight=init_weight,
        head_rank=head_rank,
    )


def load_speculator(
    *,
    target_arch: str,
    target: nn.Module,
    speculator_dir: Path,
    speculator_ckpt: Optional[Path],
    spec_len: int,
    spec_layers: int,
    head_rank: Optional[int],
) -> Tuple[nn.Module, int]:
    cfg_path = speculator_dir / "speculator_config.json"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        spec_len = int(cfg.get("spec_len", spec_len))
        spec_layers = int(cfg.get("spec_layers", spec_layers))
        target_arch = str(cfg.get("target_arch", target_arch))
        if "head_rank" in cfg:
            head_rank = cfg.get("head_rank", head_rank)

    speculator = build_speculator(
        target_arch=target_arch,
        target=target,
        spec_len=spec_len,
        spec_layers=spec_layers,
        head_rank=head_rank,
    )

    if speculator_ckpt is None:
        latest = _pick_latest_checkpoint(speculator_dir / "checkpoints")
        if latest is None:
            raise FileNotFoundError(f"No checkpoints under {speculator_dir}/checkpoints")
        speculator_ckpt = latest / "speculator.safetensors"
    if not speculator_ckpt.is_file():
        candidates = [
            p
            for p in speculator_ckpt.parent.iterdir()
            if p.is_file() and p.name.endswith(".safetensors")
        ]
        if len(candidates) == 1:
            speculator_ckpt = candidates[0]
            print(f"[warn] using fallback checkpoint: {speculator_ckpt}", flush=True)
        else:
            raise FileNotFoundError(f"Speculator checkpoint not found: {speculator_ckpt}")

    speculator.load_weights(str(speculator_ckpt))
    speculator.eval()
    return speculator, spec_len


def _load_target(
    *,
    target_arch: str,
    model_dir: Optional[str],
    hf_repo: str,
    revision: Optional[str],
    minillm_ckpt_dir: Optional[str],
    minillm_tokenizer: str,
) -> Tuple[nn.Module, Any]:
    if target_arch == "minillm":
        if not minillm_ckpt_dir:
            raise ValueError("--minillm_ckpt_dir is required when target_arch=minillm")
        AutoTokenizer, MiniLLMConfig, MiniLLMForCausalLM, _, _ = _get_minillm_deps()
        ckpt_path = Path(minillm_ckpt_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MiniLLM checkpoint dir not found: {ckpt_path}")
        cfg_path = ckpt_path / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing MiniLLM config.json in {ckpt_path}")
        cfg = MiniLLMConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        model = MiniLLMForCausalLM(cfg)
        weights_path = ckpt_path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing MiniLLM weights: {weights_path}")
        model.load_weights(str(weights_path))
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(minillm_tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        return model, tokenizer

    qwen3_load, _, _, _ = _get_qwen3_deps()
    path: str
    if model_dir:
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model dir not found: {model_path}")
        path = str(model_path)
    else:
        model_path = _default_model_dir(hf_repo)
        path = str(model_path) if model_path.exists() else hf_repo
    try:
        model, tokenizer = qwen3_load(path, revision=revision)
    except AttributeError as exc:
        if "keys" in str(exc) and "list" in str(exc):
            model, tokenizer = qwen3_load(
                path,
                revision=revision,
                tokenizer_config={"extra_special_tokens": {}},
            )
        else:
            raise
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
    _, mlx_cache, _, _ = _get_qwen3_deps()
    output_ids = list(input_ids)
    if cache is None or last_hidden is None:
        cache = mlx_cache.make_prompt_cache(target.model)
        prompt = mx.array([output_ids], dtype=mx.int32)
        hidden = target.model(prompt, cache=cache)
        mx.eval(hidden)
        last_hidden = hidden[:, -1:, :]

    for _ in range(int(max_new_tokens)):
        logits = _project_logits_qwen3(target, last_hidden)[:, -1, :]
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
    speculator: nn.Module,
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
    _, mlx_cache, _, _ = _get_qwen3_deps()
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
            prev_hidden = mx.concatenate([last_hidden, block_hidden], axis=1)[:, :-1, :]
            block_logits = _project_logits_qwen3(target, prev_hidden)
            mx.eval(block_logits)
        else:
            full = mx.array([output_ids + draft_tokens], dtype=mx.int32)
            full_hidden = target.model(full, cache=None)
            block_hidden = full_hidden[:, -len(draft_tokens) :, :]
            prev_hidden = mx.concatenate([last_hidden, block_hidden], axis=1)[:, :-1, :]
            block_logits = _project_logits_qwen3(target, prev_hidden)
            mx.eval(block_logits)

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


def baseline_decode_minillm(
    *,
    target: nn.Module,
    input_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> List[int]:
    output_ids = list(input_ids)
    for _ in range(int(max_new_tokens)):
        prompt = mx.array([output_ids], dtype=mx.int32)
        hidden = target.model(prompt)
        logits = _project_logits_minillm(target, hidden)[:, -1, :]
        mx.eval(logits)
        token = sample_next_token(logits, temperature=temperature, top_p=top_p)
        output_ids.append(int(token))
        if eos_token_id is not None and int(token) == int(eos_token_id):
            break
    return output_ids


def speculative_decode_minillm(
    *,
    target: nn.Module,
    speculator: nn.Module,
    input_ids: List[int],
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    optimized: bool,
    max_consecutive_misses: int = 2,
) -> List[int]:
    output_ids = list(input_ids)
    produced = 0
    consecutive_misses = 0

    while produced < int(max_new_tokens):
        prompt = mx.array([output_ids], dtype=mx.int32)
        hidden = target.model(prompt)
        mx.eval(hidden)
        last_hidden = hidden[:, -1:, :]

        logits_list = speculator(last_hidden)
        mx.eval(*logits_list)
        draft_tokens = [
            sample_next_token(logits_list[i][0, -1, :], temperature=temperature, top_p=top_p)
            for i in range(int(spec_len))
        ]

        full = mx.array([output_ids + draft_tokens], dtype=mx.int32)
        full_hidden = target.model(full)
        block_hidden = full_hidden[:, -len(draft_tokens) :, :]
        prev_hidden = mx.concatenate([last_hidden, block_hidden], axis=1)[:, :-1, :]
        block_logits = _project_logits_minillm(target, prev_hidden)
        mx.eval(block_logits)

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

        if optimized and consecutive_misses >= max_consecutive_misses:
            break

    if optimized and produced < int(max_new_tokens):
        output_ids = baseline_decode_minillm(
            target=target,
            input_ids=output_ids,
            max_new_tokens=int(max_new_tokens) - produced,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    return output_ids


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3")
    parser.add_argument("--hf_repo", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--minillm_ckpt_dir", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument("--speculator_dir", type=str, default="out/eagle3_speculator_mlx/qwen3_0.6b")
    parser.add_argument("--speculator_ckpt", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--spec_len", type=int, default=None, help="Draft length for speculator (auto if unset).")
    parser.add_argument("--spec_layers", type=int, default=None, help="Transformer layers in speculator (auto if unset).")
    parser.add_argument(
        "--head_rank",
        type=int,
        default=None,
        help="Low-rank speculator head size (overrides config if set; full head if unset).",
    )
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
        target_arch=args.target_arch,
        model_dir=args.model_dir,
        hf_repo=args.hf_repo,
        revision=args.revision,
        minillm_ckpt_dir=args.minillm_ckpt_dir,
        minillm_tokenizer=args.minillm_tokenizer,
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
    spec_len, spec_layers = _resolve_spec_config(
        args.spec_len, args.spec_layers, param_count=_count_params_mlx(target)
    )
    head_rank = args.head_rank if args.head_rank is not None and int(args.head_rank) > 0 else None
    if not args.no_speculator:
        speculator_dir = Path(args.speculator_dir)
        speculator_ckpt = Path(args.speculator_ckpt) if args.speculator_ckpt else None
        speculator, spec_len = load_speculator(
            target_arch=args.target_arch,
            target=target,
            speculator_dir=speculator_dir,
            speculator_ckpt=speculator_ckpt,
            spec_len=spec_len,
            spec_layers=spec_layers,
            head_rank=head_rank,
        )

    start = time.time()
    if args.target_arch == "minillm":
        if speculator is None:
            output_ids = baseline_decode_minillm(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            output_ids = speculative_decode_minillm(
                target=target,
                speculator=speculator,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                spec_len=spec_len,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                optimized=optimized,
            )
    else:
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
