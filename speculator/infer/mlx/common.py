#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("Missing MLX core dependencies. Install mlx first.") from exc


@dataclass(frozen=True)
class SpecStats:
    total_accept: int
    total_draft: int
    accepted_output: int
    zero_accept: int
    steps: int
    spec_time_s: float
    target_time_s: float
    target_prefill_time_s: float
    target_verify_time_s: float
    target_generate_time_s: float
    target_generated: int


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
        return 2, 2
    params_b = float(param_count) / 1e9
    if params_b <= 1.0:
        return 4, 2
    if params_b <= 3.0:
        return 3, 2
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


def _select_feature_layers(num_layers: int, *, count: int = 4) -> List[int]:
    if num_layers <= 0:
        return []
    if num_layers <= count:
        return list(range(int(num_layers)))
    step = (num_layers - 1) / float(count - 1)
    layers = [int(round(step * i)) for i in range(int(count))]
    selected: List[int] = []
    for idx in layers:
        if not selected or idx != selected[-1]:
            selected.append(idx)
    return selected


def _resolve_feature_layers(feature_layers: Optional[List[int]], *, num_layers: int) -> List[int]:
    if feature_layers:
        return [int(i) for i in feature_layers]
    return _select_feature_layers(num_layers)


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def sample_next_token(logits: mx.array, *, temperature: float, top_p: float) -> int:
    logits = logits.reshape(-1)
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


def _sample_next_token_batch(logits: mx.array, *, temperature: float, top_p: float) -> mx.array:
    """Sample tokens from batched logits. Time O(kV) best/avg/worst, O(kV log V) when top_p<1; space O(kV)."""
    if int(logits.size) == 0:
        return mx.array([], dtype=mx.int32)
    logits = logits.reshape(int(logits.shape[0]), -1)
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    scaled = logits / float(temperature)
    if top_p >= 1.0:
        return mx.random.categorical(scaled, axis=-1)

    sorted_idx = mx.argsort(-scaled, axis=-1)
    sorted_logits = mx.take_along_axis(scaled, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)
    keep_prefix = mx.concatenate(
        [mx.ones((int(logits.shape[0]), 1), dtype=mx.bool_), cumprobs[:, :-1] <= float(top_p)],
        axis=-1,
    )
    neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
    filtered_logits = mx.where(keep_prefix, sorted_logits, neg_inf)
    picked = mx.random.categorical(filtered_logits, axis=-1)
    tokens = mx.take_along_axis(sorted_idx, picked[:, None], axis=-1).squeeze(-1)
    return tokens


def _token_prob_from_logits(
    logits: mx.array,
    token: int,
    *,
    temperature: float,
    top_p: float,
) -> float:
    """Token probability under (temp, top_p). Time O(V) best/avg/worst, space O(V)."""
    logits = logits.reshape(-1)
    token_id = int(token)
    if temperature <= 0:
        return 1.0 if int(mx.argmax(logits).item()) == token_id else 0.0
    scaled = logits / float(temperature)
    if top_p >= 1.0:
        probs = mx.softmax(scaled, axis=-1)
        return float(probs[token_id].item())
    sorted_idx = mx.argsort(-scaled, axis=-1)
    sorted_logits = mx.take_along_axis(scaled, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)
    remove = cumprobs > float(top_p)
    remove = mx.concatenate([mx.array([False]), remove[:-1]], axis=-1)
    neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
    filtered_logits = mx.where(remove, neg_inf, sorted_logits)
    filtered_probs = mx.softmax(filtered_logits, axis=-1)
    mask = sorted_idx == token_id
    prob = mx.sum(filtered_probs * mask)
    return float(prob.item())


def _token_probs_from_logits_batch(
    logits: mx.array,
    tokens: mx.array,
    *,
    temperature: float,
    top_p: float,
) -> mx.array:
    """Token probs for batched logits. Time O(kV) best/avg/worst, O(kV log V) when top_p<1; space O(kV)."""
    if int(logits.size) == 0:
        return mx.array([], dtype=mx.float32)
    logits = logits.reshape(int(logits.shape[0]), -1)
    tokens = tokens.reshape(-1)
    if temperature <= 0:
        argmax = mx.argmax(logits, axis=-1)
        return mx.where(argmax == tokens, mx.ones_like(tokens, dtype=mx.float32), mx.zeros_like(tokens, dtype=mx.float32))

    scaled = logits / float(temperature)
    if top_p >= 1.0:
        probs = mx.softmax(scaled, axis=-1)
        return mx.take_along_axis(probs, tokens[:, None], axis=-1).squeeze(-1)

    sorted_idx = mx.argsort(-scaled, axis=-1)
    sorted_logits = mx.take_along_axis(scaled, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)
    keep_prefix = mx.concatenate(
        [mx.ones((int(logits.shape[0]), 1), dtype=mx.bool_), cumprobs[:, :-1] <= float(top_p)],
        axis=-1,
    )
    neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
    filtered_logits = mx.where(keep_prefix, sorted_logits, neg_inf)
    filtered_probs = mx.softmax(filtered_logits, axis=-1)
    mask = sorted_idx == tokens[:, None]
    probs = mx.sum(filtered_probs * mask, axis=-1)
    return probs


def _accept_reject_block(
    *,
    draft_tokens: List[int],
    draft_logits: mx.array,
    target_logits: mx.array,
    temperature: float,
    top_p: float,
) -> Tuple[int, List[int], bool]:
    """Reject-sampling acceptance for a draft block. Time O(kV) best/avg, O(kV log V) when top_p<1; space O(kV)."""
    if not draft_tokens:
        return 0, [], False
    draft_tokens_arr = mx.array(draft_tokens, dtype=mx.int32)
    q_probs = _token_probs_from_logits_batch(
        draft_logits, draft_tokens_arr, temperature=temperature, top_p=top_p
    )
    p_probs = _token_probs_from_logits_batch(
        target_logits, draft_tokens_arr, temperature=temperature, top_p=top_p
    )
    zero = mx.zeros_like(p_probs)
    accept_probs = mx.where(q_probs <= 0.0, zero, mx.minimum(1.0, p_probs / q_probs))
    accept_draws = mx.random.uniform(shape=accept_probs.shape)
    accepted = accept_draws < accept_probs
    all_accepted = bool(mx.all(accepted).item())
    if all_accepted:
        return len(draft_tokens), draft_tokens, False

    reject_idx = int(mx.argmax(mx.logical_not(accepted)).item())
    token = sample_next_token(target_logits[reject_idx], temperature=temperature, top_p=top_p)
    new_tokens = list(draft_tokens[:reject_idx]) + [int(token)]
    return reject_idx, new_tokens, True


def _project_logits_qwen3(target: nn.Module, hidden: mx.array) -> mx.array:
    if target.args.tie_word_embeddings:
        return target.model.embed_tokens.as_linear(hidden)
    return target.lm_head(hidden)


def _project_logits_minillm(target: nn.Module, hidden: mx.array) -> mx.array:
    return hidden @ target.model.embed_tokens.weight.transpose()


def _qwen3_forward_hidden_states(
    target: nn.Module,
    input_ids: mx.array,
    *,
    cache: Optional[List[Any]],
    layer_ids: List[int],
) -> Tuple[mx.array, List[mx.array]]:
    _, _, _, create_attention_mask = _get_qwen3_deps()
    model = target.model
    h = model.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])
    layer_index = {int(layer_id): idx for idx, layer_id in enumerate(layer_ids)}
    hiddens: List[Optional[mx.array]] = [None] * len(layer_ids)
    for idx, (layer, c) in enumerate(zip(model.layers, cache)):
        h = layer(h, mask, c)
        if idx in layer_index:
            hiddens[layer_index[idx]] = h
    h = model.norm(h)
    return h, [hidden for hidden in hiddens if hidden is not None]


def _minillm_forward_hidden_states(
    target: nn.Module,
    input_ids: mx.array,
    *,
    attention_mask: Optional[mx.array],
    layer_ids: List[int],
) -> Tuple[mx.array, List[mx.array]]:
    model = target.model
    h = model.embed_tokens(input_ids)
    h = model.dropout(h)
    layer_index = {int(layer_id): idx for idx, layer_id in enumerate(layer_ids)}
    hiddens: List[Optional[mx.array]] = [None] * len(layer_ids)
    for idx, layer in enumerate(model.layers):
        h = layer(h, start_pos=0, attention_mask=attention_mask)
        if idx in layer_index:
            hiddens[layer_index[idx]] = h
    h = model.norm(h)
    return h, [hidden for hidden in hiddens if hidden is not None]


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


class FeatureFusion(nn.Module):
    def __init__(self, *, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.projs = [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(int(num_layers))]
        self.weights = mx.zeros((int(num_layers),), dtype=mx.float32)

    def __call__(self, hiddens: List[mx.array]) -> mx.array:
        weights = mx.softmax(self.weights)
        fused = None
        for idx, (proj, hidden) in enumerate(zip(self.projs, hiddens)):
            contrib = proj(hidden) * weights[idx]
            fused = contrib if fused is None else fused + contrib
        return fused


class Qwen3Speculator(nn.Module):
    def __init__(
        self,
        *,
        args: Any,
        spec_layers: int,
        feature_layers: List[int],
        init_weight: Optional[mx.array],
        head_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        _, _, qwen3_model, create_attention_mask = _get_qwen3_deps()
        self._create_attention_mask = create_attention_mask
        self.feature_layers = [int(i) for i in feature_layers]
        self.fusion = FeatureFusion(hidden_size=args.hidden_size, num_layers=len(self.feature_layers))
        self.layers = [qwen3_model.TransformerBlock(args=args) for _ in range(int(spec_layers))]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.head = LowRankHead(hidden_size=args.hidden_size, vocab_size=args.vocab_size, rank=rank)
        else:
            self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
            if init_weight is not None:
                self.head.weight = init_weight

    def fuse(self, hiddens: List[mx.array], attention_mask: Optional[mx.array] = None) -> mx.array:
        fused = self.fusion(hiddens)
        if attention_mask is not None:
            fused = fused * attention_mask[..., None]
        return fused

    def decode(self, *, fused_context: mx.array, token_embeds: Optional[mx.array]) -> mx.array:
        if token_embeds is None or int(token_embeds.shape[1]) == 0:
            x = fused_context
        else:
            x = mx.concatenate([fused_context, token_embeds], axis=1)
        mask = None
        if x.shape[1] > 1:
            mask = self._create_attention_mask(x, cache=None)
        for layer in self.layers:
            x = layer(x, mask=mask, cache=None)
        x = self.norm(x)
        logits = self.head(x)
        return logits[:, -1, :]


class MiniLLMSpeculator(nn.Module):
    def __init__(
        self,
        *,
        config: Any,
        spec_layers: int,
        feature_layers: List[int],
        init_weight: Optional[mx.array],
        head_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        _, _, _, MiniLLMBlock, RMSNorm = _get_minillm_deps()
        self.feature_layers = [int(i) for i in feature_layers]
        self.fusion = FeatureFusion(hidden_size=config.hidden_size, num_layers=len(self.feature_layers))
        self.layers = [MiniLLMBlock(i, config) for i in range(int(spec_layers))]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.head = LowRankHead(hidden_size=config.hidden_size, vocab_size=config.vocab_size, rank=rank)
        else:
            self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if init_weight is not None:
                self.head.weight = init_weight

    def fuse(self, hiddens: List[mx.array], attention_mask: Optional[mx.array] = None) -> mx.array:
        fused = self.fusion(hiddens)
        if attention_mask is not None:
            fused = fused * attention_mask[..., None]
        return fused

    def decode(self, *, fused_context: mx.array, token_embeds: Optional[mx.array]) -> mx.array:
        if token_embeds is None or int(token_embeds.shape[1]) == 0:
            x = fused_context
        else:
            x = mx.concatenate([fused_context, token_embeds], axis=1)
        for layer in self.layers:
            x = layer(x, start_pos=0, attention_mask=None)
        x = self.norm(x)
        logits = self.head(x)
        return logits[:, -1, :]


def build_speculator(
    *,
    target_arch: str,
    target: nn.Module,
    spec_len: int,
    spec_layers: int,
    feature_layers: Optional[List[int]] = None,
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
        layers = _resolve_feature_layers(feature_layers, num_layers=int(target.config.num_hidden_layers))
        return MiniLLMSpeculator(
            config=target.config,
            spec_layers=spec_layers,
            feature_layers=layers,
            init_weight=init_weight,
            head_rank=head_rank,
        )

    layers = _resolve_feature_layers(feature_layers, num_layers=int(target.args.num_hidden_layers))
    return Qwen3Speculator(
        args=target.args,
        spec_layers=spec_layers,
        feature_layers=layers,
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
    feature_layers: Optional[List[int]] = None
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        spec_len = int(cfg.get("spec_len", spec_len))
        spec_layers = int(cfg.get("spec_layers", spec_layers))
        target_arch = str(cfg.get("target_arch", target_arch))
        if "feature_layers" in cfg:
            feature_layers = [int(i) for i in cfg.get("feature_layers") or []]
        if "head_rank" in cfg:
            head_rank = cfg.get("head_rank", head_rank)

    speculator = build_speculator(
        target_arch=target_arch,
        target=target,
        spec_len=spec_len,
        spec_layers=spec_layers,
        feature_layers=feature_layers,
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


def _speculative_decode_qwen3(
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
    max_consecutive_misses: int,
    collect_stats: bool,
) -> Tuple[List[int], Optional[SpecStats]]:
    _, mlx_cache, _, _ = _get_qwen3_deps()
    output_ids = list(input_ids)
    produced = 0
    consecutive_misses = 0
    total_accept = 0
    total_draft = 0
    accepted_output = 0
    zero_accept = 0
    steps = 0
    spec_time_s = 0.0
    target_time_s = 0.0
    target_prefill_time_s = 0.0
    target_verify_time_s = 0.0
    target_generate_time_s = 0.0
    target_generated = 0

    cache = mlx_cache.make_prompt_cache(target.model) if use_cache else None
    prompt = mx.array([output_ids], dtype=mx.int32)
    t0 = time.perf_counter()
    hidden, layer_hiddens = _qwen3_forward_hidden_states(
        target,
        prompt,
        cache=cache,
        layer_ids=speculator.feature_layers,
    )
    mx.eval(hidden)
    prefill_s = time.perf_counter() - t0
    target_time_s += prefill_s
    target_prefill_time_s += prefill_s
    last_hidden = hidden[:, -1:, :]
    last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]

    while produced < int(max_new_tokens):
        remaining = int(max_new_tokens) - produced
        block_len = min(int(spec_len), int(remaining))
        fused_context = speculator.fuse(last_layer_hiddens, attention_mask=None)
        draft_tokens: List[int] = []
        draft_logits_list: List[mx.array] = []
        token_embeds = None
        for _ in range(int(block_len)):
            t0 = time.perf_counter()
            logits = speculator.decode(fused_context=fused_context, token_embeds=token_embeds)
            mx.eval(logits)
            spec_time_s += time.perf_counter() - t0
            token = sample_next_token(logits[0], temperature=temperature, top_p=top_p)
            draft_tokens.append(int(token))
            draft_logits_list.append(logits)
            token_embed = target.model.embed_tokens(mx.array([[int(token)]], dtype=mx.int32))
            token_embeds = token_embed if token_embeds is None else mx.concatenate([token_embeds, token_embed], axis=1)
        draft_logits = mx.concatenate(draft_logits_list, axis=0)

        if use_cache:
            draft_arr = mx.array([draft_tokens], dtype=mx.int32)
            t0 = time.perf_counter()
            block_hidden, block_layer_hiddens = _qwen3_forward_hidden_states(
                target,
                draft_arr,
                cache=cache,
                layer_ids=speculator.feature_layers,
            )
            prev_hidden = mx.concatenate([last_hidden, block_hidden], axis=1)[:, :-1, :]
            block_logits = _project_logits_qwen3(target, prev_hidden)
            mx.eval(block_logits)
            verify_s = time.perf_counter() - t0
            target_time_s += verify_s
            target_verify_time_s += verify_s
        else:
            full = mx.array([output_ids + draft_tokens], dtype=mx.int32)
            t0 = time.perf_counter()
            full_hidden, full_layer_hiddens = _qwen3_forward_hidden_states(
                target,
                full,
                cache=None,
                layer_ids=speculator.feature_layers,
            )
            block_hidden = full_hidden[:, -len(draft_tokens) :, :]
            block_layer_hiddens = [h[:, -len(draft_tokens) :, :] for h in full_layer_hiddens]
            prev_hidden = mx.concatenate([last_hidden, block_hidden], axis=1)[:, :-1, :]
            block_logits = _project_logits_qwen3(target, prev_hidden)
            mx.eval(block_logits)
            verify_s = time.perf_counter() - t0
            target_time_s += verify_s
            target_verify_time_s += verify_s

        accept_len, new_tokens, rejected = _accept_reject_block(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            target_logits=block_logits[0],
            temperature=temperature,
            top_p=top_p,
        )

        bonus_added = False
        if not rejected and accept_len == len(draft_tokens) and remaining > len(draft_tokens):
            t0 = time.perf_counter()
            bonus_logits = _project_logits_qwen3(target, block_hidden[:, -1:, :])[:, -1, :]
            mx.eval(bonus_logits)
            gen_s = time.perf_counter() - t0
            target_time_s += gen_s
            target_generate_time_s += gen_s
            bonus_token = sample_next_token(bonus_logits, temperature=temperature, top_p=top_p)
            new_tokens.append(int(bonus_token))
            bonus_added = True

        if collect_stats:
            total_accept += int(accept_len)
            total_draft += int(block_len)
            steps += 1
            if accept_len == 0:
                zero_accept += 1

        token_sources = [True] * int(accept_len)
        if rejected or bonus_added:
            token_sources.append(False)
        eos_hit = False
        if eos_token_id is not None and int(eos_token_id) in new_tokens:
            eos_idx = new_tokens.index(int(eos_token_id))
            new_tokens = new_tokens[: eos_idx + 1]
            token_sources = token_sources[: eos_idx + 1]
            eos_hit = True
        if collect_stats:
            accepted_step = sum(1 for src in token_sources if src)
            accepted_output += int(accepted_step)
            target_generated += int(len(token_sources) - accepted_step)

        if accept_len == 0:
            consecutive_misses += 1
        else:
            consecutive_misses = 0

        output_ids.extend(int(t) for t in new_tokens)
        produced += len(new_tokens)

        if eos_hit:
            break

        if use_cache:
            if accept_len == len(draft_tokens):
                if bonus_added:
                    step = mx.array([[int(new_tokens[-1])]], dtype=mx.int32)
                    t0 = time.perf_counter()
                    hidden, layer_hiddens = _qwen3_forward_hidden_states(
                        target,
                        step,
                        cache=cache,
                        layer_ids=speculator.feature_layers,
                    )
                    mx.eval(hidden)
                    gen_s = time.perf_counter() - t0
                    target_time_s += gen_s
                    target_generate_time_s += gen_s
                    last_hidden = hidden[:, -1:, :]
                    last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]
                else:
                    last_hidden = block_hidden[:, -1:, :]
                    last_layer_hiddens = [h[:, -1:, :] for h in block_layer_hiddens]
            else:
                if not mlx_cache.can_trim_prompt_cache(cache):
                    cache = mlx_cache.make_prompt_cache(target.model)
                    prompt = mx.array([output_ids], dtype=mx.int32)
                    t0 = time.perf_counter()
                    hidden, layer_hiddens = _qwen3_forward_hidden_states(
                        target,
                        prompt,
                        cache=cache,
                        layer_ids=speculator.feature_layers,
                    )
                    mx.eval(hidden)
                    prefill_s = time.perf_counter() - t0
                    target_time_s += prefill_s
                    target_prefill_time_s += prefill_s
                    last_hidden = hidden[:, -1:, :]
                    last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]
                else:
                    mlx_cache.trim_prompt_cache(cache, len(draft_tokens) - accept_len)
                    step = mx.array([[int(new_tokens[-1])]], dtype=mx.int32)
                    t0 = time.perf_counter()
                    hidden, layer_hiddens = _qwen3_forward_hidden_states(
                        target,
                        step,
                        cache=cache,
                        layer_ids=speculator.feature_layers,
                    )
                    mx.eval(hidden)
                    gen_s = time.perf_counter() - t0
                    target_time_s += gen_s
                    target_generate_time_s += gen_s
                    last_hidden = hidden[:, -1:, :]
                    last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]
        else:
            prompt = mx.array([output_ids], dtype=mx.int32)
            t0 = time.perf_counter()
            hidden, layer_hiddens = _qwen3_forward_hidden_states(
                target,
                prompt,
                cache=None,
                layer_ids=speculator.feature_layers,
            )
            mx.eval(hidden)
            prefill_s = time.perf_counter() - t0
            target_time_s += prefill_s
            target_prefill_time_s += prefill_s
            last_hidden = hidden[:, -1:, :]
            last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]

        if optimized and consecutive_misses >= max_consecutive_misses:
            break

    if optimized and produced < int(max_new_tokens):
        t0 = time.perf_counter()
        before_len = len(output_ids)
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
        gen_s = time.perf_counter() - t0
        target_time_s += gen_s
        target_generate_time_s += gen_s
        if collect_stats:
            target_generated += len(output_ids) - before_len

    stats = None
    if collect_stats:
        stats = SpecStats(
            total_accept=int(total_accept),
            total_draft=int(total_draft),
            accepted_output=int(accepted_output),
            zero_accept=int(zero_accept),
            steps=int(steps),
            spec_time_s=float(spec_time_s),
            target_time_s=float(target_time_s),
            target_prefill_time_s=float(target_prefill_time_s),
            target_verify_time_s=float(target_verify_time_s),
            target_generate_time_s=float(target_generate_time_s),
            target_generated=int(target_generated),
        )
    return output_ids, stats


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
    output_ids, _ = _speculative_decode_qwen3(
        target=target,
        speculator=speculator,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        spec_len=spec_len,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        use_cache=use_cache,
        optimized=optimized,
        max_consecutive_misses=max_consecutive_misses,
        collect_stats=False,
    )
    return output_ids


def speculative_decode_with_stats(
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
) -> Tuple[List[int], SpecStats]:
    output_ids, stats = _speculative_decode_qwen3(
        target=target,
        speculator=speculator,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        spec_len=spec_len,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        use_cache=use_cache,
        optimized=optimized,
        max_consecutive_misses=max_consecutive_misses,
        collect_stats=True,
    )
    return output_ids, stats


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


def _speculative_decode_minillm(
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
    max_consecutive_misses: int,
    collect_stats: bool,
) -> Tuple[List[int], Optional[SpecStats]]:
    output_ids = list(input_ids)
    produced = 0
    consecutive_misses = 0
    total_accept = 0
    total_draft = 0
    accepted_output = 0
    zero_accept = 0
    steps = 0
    spec_time_s = 0.0
    target_time_s = 0.0
    target_prefill_time_s = 0.0
    target_verify_time_s = 0.0
    target_generate_time_s = 0.0
    target_generated = 0

    while produced < int(max_new_tokens):
        prompt = mx.array([output_ids], dtype=mx.int32)
        t0 = time.perf_counter()
        hidden, layer_hiddens = _minillm_forward_hidden_states(
            target,
            prompt,
            attention_mask=None,
            layer_ids=speculator.feature_layers,
        )
        mx.eval(hidden)
        prefill_s = time.perf_counter() - t0
        target_time_s += prefill_s
        target_prefill_time_s += prefill_s
        last_hidden = hidden[:, -1:, :]
        last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]

        remaining = int(max_new_tokens) - produced
        block_len = min(int(spec_len), int(remaining))
        fused_context = speculator.fuse(last_layer_hiddens, attention_mask=None)
        draft_tokens: List[int] = []
        draft_logits_list: List[mx.array] = []
        token_embeds = None
        for _ in range(int(block_len)):
            t0 = time.perf_counter()
            logits = speculator.decode(fused_context=fused_context, token_embeds=token_embeds)
            mx.eval(logits)
            spec_time_s += time.perf_counter() - t0
            token = sample_next_token(logits[0], temperature=temperature, top_p=top_p)
            draft_tokens.append(int(token))
            draft_logits_list.append(logits)
            token_embed = target.model.embed_tokens(mx.array([[int(token)]], dtype=mx.int32))
            token_embeds = token_embed if token_embeds is None else mx.concatenate([token_embeds, token_embed], axis=1)
        draft_logits = mx.concatenate(draft_logits_list, axis=0)

        full = mx.array([output_ids + draft_tokens], dtype=mx.int32)
        t0 = time.perf_counter()
        full_hidden, _ = _minillm_forward_hidden_states(
            target,
            full,
            attention_mask=None,
            layer_ids=speculator.feature_layers,
        )
        block_hidden = full_hidden[:, -len(draft_tokens) :, :]
        prev_hidden = mx.concatenate([last_hidden, block_hidden], axis=1)[:, :-1, :]
        block_logits = _project_logits_minillm(target, prev_hidden)
        mx.eval(block_logits)
        verify_s = time.perf_counter() - t0
        target_time_s += verify_s
        target_verify_time_s += verify_s

        accept_len, new_tokens, rejected = _accept_reject_block(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            target_logits=block_logits[0],
            temperature=temperature,
            top_p=top_p,
        )

        bonus_added = False
        if not rejected and accept_len == len(draft_tokens) and remaining > len(draft_tokens):
            t0 = time.perf_counter()
            bonus_logits = _project_logits_minillm(target, block_hidden[:, -1:, :])[:, -1, :]
            mx.eval(bonus_logits)
            gen_s = time.perf_counter() - t0
            target_time_s += gen_s
            target_generate_time_s += gen_s
            bonus_token = sample_next_token(bonus_logits, temperature=temperature, top_p=top_p)
            new_tokens.append(int(bonus_token))
            bonus_added = True

        if collect_stats:
            total_accept += int(accept_len)
            total_draft += int(block_len)
            steps += 1
            if accept_len == 0:
                zero_accept += 1

        token_sources = [True] * int(accept_len)
        if rejected or bonus_added:
            token_sources.append(False)
        eos_hit = False
        if eos_token_id is not None and int(eos_token_id) in new_tokens:
            eos_idx = new_tokens.index(int(eos_token_id))
            new_tokens = new_tokens[: eos_idx + 1]
            token_sources = token_sources[: eos_idx + 1]
            eos_hit = True
        if collect_stats:
            accepted_step = sum(1 for src in token_sources if src)
            accepted_output += int(accepted_step)
            target_generated += int(len(token_sources) - accepted_step)

        if accept_len == 0:
            consecutive_misses += 1
        else:
            consecutive_misses = 0

        output_ids.extend(int(t) for t in new_tokens)
        produced += len(new_tokens)

        if eos_hit:
            break

        if optimized and consecutive_misses >= max_consecutive_misses:
            break

    if optimized and produced < int(max_new_tokens):
        t0 = time.perf_counter()
        before_len = len(output_ids)
        output_ids = baseline_decode_minillm(
            target=target,
            input_ids=output_ids,
            max_new_tokens=int(max_new_tokens) - produced,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
        gen_s = time.perf_counter() - t0
        target_time_s += gen_s
        target_generate_time_s += gen_s
        if collect_stats:
            target_generated += len(output_ids) - before_len

    stats = None
    if collect_stats:
        stats = SpecStats(
            total_accept=int(total_accept),
            total_draft=int(total_draft),
            accepted_output=int(accepted_output),
            zero_accept=int(zero_accept),
            steps=int(steps),
            spec_time_s=float(spec_time_s),
            target_time_s=float(target_time_s),
            target_prefill_time_s=float(target_prefill_time_s),
            target_verify_time_s=float(target_verify_time_s),
            target_generate_time_s=float(target_generate_time_s),
            target_generated=int(target_generated),
        )
    return output_ids, stats


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
    output_ids, _ = _speculative_decode_minillm(
        target=target,
        speculator=speculator,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        spec_len=spec_len,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        optimized=optimized,
        max_consecutive_misses=max_consecutive_misses,
        collect_stats=False,
    )
    return output_ids


def speculative_decode_minillm_with_stats(
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
) -> Tuple[List[int], SpecStats]:
    output_ids, stats = _speculative_decode_minillm(
        target=target,
        speculator=speculator,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        spec_len=spec_len,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        optimized=optimized,
        max_consecutive_misses=max_consecutive_misses,
        collect_stats=True,
    )
    return output_ids, stats


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
    parser.add_argument("--temperature", type=float, default=0.8)
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
    parser.add_argument("--no_chat_template", action="store_true")
    return parser


def run_cli() -> None:
    parser = build_arg_parser(
        description="Speculative decoding with EAGLE-3 speculator (MLX backend)."
    )
    args = parser.parse_args()
    optimized = True
    use_cache = True

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
                use_cache=use_cache,
                optimized=optimized,
            )
    elapsed = time.time() - start
    out_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(out_text)
    print(f"[time] {elapsed:.2f}s")
