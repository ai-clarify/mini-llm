#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN

from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _default_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _maybe_adjust_dtype_for_device(device: torch.device, dtype: torch.dtype) -> torch.dtype:
    if device.type == "mps" and dtype == torch.bfloat16:
        try:
            torch.empty((1,), device=device, dtype=dtype)
        except Exception:
            return torch.float16
    return dtype


def _count_params_torch(model: nn.Module) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _auto_spec_config(param_count: Optional[int]) -> Tuple[int, int]:
    if not param_count or param_count <= 0:
        return 2, 2
    params_b = float(param_count) / 1e9
    if params_b <= 1.0:
        return 4, 2
    if params_b <= 3.0:
        return 4, 2
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
    target_prefill_calls: int
    target_verify_calls: int
    target_generate_calls: int
    target_generated: int


@dataclass(frozen=True)
class Eagle3HFConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    hidden_act: str
    head_dim: Optional[int] = None


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


def _load_minillm_config(path: Optional[str]) -> MiniLLMConfig:
    if not path:
        return MiniLLMConfig()
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MiniLLM config must be a JSON object: {cfg_path}")
    return MiniLLMConfig(**data)


def _extract_hidden_state(output: Any) -> torch.Tensor:
    return output.last_hidden_state


def _extract_hidden_layers(output: Any, layer_ids: List[int]) -> List[torch.Tensor]:
    hidden_states = output.hidden_states
    if not hidden_states:
        return [output.last_hidden_state for _ in layer_ids]
    offset = 1 if len(hidden_states) > max(layer_ids) + 1 else 0
    return [hidden_states[int(layer_id) + offset] for layer_id in layer_ids]


def _project_logits_minillm(target: MiniLLMForCausalLM, hidden: torch.Tensor) -> torch.Tensor:
    return target.lm_head(hidden)


def _minillm_forward_hidden_states(
    target: MiniLLMForCausalLM,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    layer_ids: List[int],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    model = target.model
    h = model.embed_tokens(input_ids)
    h = model.dropout(h)
    seq_len = int(input_ids.shape[1])
    position_embeddings = (
        model.freqs_cos[:seq_len],
        model.freqs_sin[:seq_len],
    )
    layer_index = {int(layer_id): idx for idx, layer_id in enumerate(layer_ids)}
    hiddens: List[Optional[torch.Tensor]] = [None] * len(layer_ids)
    for idx, layer in enumerate(model.layers):
        h, _ = layer(
            h,
            position_embeddings,
            past_key_value=None,
            use_cache=False,
            attention_mask=attention_mask,
        )
        if idx in layer_index:
            hiddens[layer_index[idx]] = h
    h = model.norm(h)
    return h, [hidden for hidden in hiddens if hidden is not None]


def _embed_tokens(target: AutoModelForCausalLM, token_ids: List[int]) -> Optional[torch.Tensor]:
    if not token_ids:
        return None
    device = next(target.parameters()).device
    token_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)
    return target.get_input_embeddings()(token_tensor)

def _load_target_and_tokenizer(args, device: torch.device, dtype: torch.dtype):
    if args.target_arch == "minillm":
        tokenizer = AutoTokenizer.from_pretrained(args.minillm_tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "right"

        cfg = _load_minillm_config(args.minillm_config)
        target = MiniLLMForCausalLM(cfg)
        if args.minillm_ckpt:
            state = torch.load(args.minillm_ckpt, map_location=device)
            target.load_state_dict(state, strict=False)
        else:
            print("[warn] MiniLLM checkpoint not provided; using random weights")
        target = target.to(device=device, dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "right"

        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, trust_remote_code=True, torch_dtype=dtype
        ).to(device)

    target.eval()
    return target, tokenizer


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


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


def _clone_past_key_values(past):
    if past is None:
        return None
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        DynamicCache = None
    if DynamicCache is not None and isinstance(past, DynamicCache):
        legacy = past.to_legacy_cache()
        cloned = tuple((k.clone(), v.clone()) for (k, v) in legacy)
        return DynamicCache.from_legacy_cache(cloned)
    if hasattr(past, "to_legacy_cache") and hasattr(type(past), "from_legacy_cache"):
        legacy = past.to_legacy_cache()
        cloned = tuple((k.clone(), v.clone()) for (k, v) in legacy)
        return type(past).from_legacy_cache(cloned)
    return tuple((k.clone(), v.clone()) for (k, v) in past)


def sample_next_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / float(temperature)
    if top_p >= 1.0:
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > float(top_p)
    if mask[..., 0].item():
        mask[..., 0] = False
    sorted_logits = torch.where(mask, torch.tensor(-1e9, device=logits.device), sorted_logits)
    probs = torch.softmax(sorted_logits, dim=-1)
    picked = torch.multinomial(probs, num_samples=1)
    return int(sorted_idx.gather(-1, picked).item())


def _token_prob_from_logits(
    logits: torch.Tensor,
    token: int,
    *,
    temperature: float,
    top_p: float,
) -> float:
    """Token probability under (temp, top_p). Time O(V) best/avg/worst, space O(V)."""
    token_id = int(token)
    logits = logits.reshape(-1)
    if temperature <= 0:
        return 1.0 if int(torch.argmax(logits, dim=-1).item()) == token_id else 0.0
    scaled = logits / float(temperature)
    if top_p >= 1.0:
        probs = torch.softmax(scaled, dim=-1)
        return float(probs[token_id].item())
    sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > float(top_p)
    if mask[..., 0].item():
        mask[..., 0] = False
    filtered_logits = torch.where(
        mask, torch.tensor(-1e9, device=logits.device), sorted_logits
    )
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    token_mask = sorted_idx == token_id
    prob = (filtered_probs * token_mask).sum()
    return float(prob.item())


def _token_probs_from_logits_batch(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Token probs for batched logits. Time O(kV) best/avg, O(kV log V) when top_p<1; space O(kV)."""
    if logits.numel() == 0:
        return torch.empty((0,), device=logits.device, dtype=torch.float32)
    logits = logits.reshape(int(logits.shape[0]), -1)
    tokens = tokens.reshape(-1)
    if temperature <= 0:
        argmax = torch.argmax(logits, dim=-1)
        return torch.where(
            argmax == tokens,
            torch.ones_like(tokens, dtype=torch.float32),
            torch.zeros_like(tokens, dtype=torch.float32),
        )
    scaled = logits / float(temperature)
    if top_p >= 1.0:
        probs = torch.softmax(scaled, dim=-1)
        return probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    keep_prefix = torch.cat(
        [
            torch.ones((int(logits.shape[0]), 1), device=logits.device, dtype=torch.bool),
            cumprobs[:, :-1] <= float(top_p),
        ],
        dim=-1,
    )
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=sorted_logits.dtype)
    filtered_logits = torch.where(keep_prefix, sorted_logits, neg_inf)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    mask = sorted_idx == tokens.unsqueeze(-1)
    return torch.sum(filtered_probs * mask, dim=-1)


def _accept_reject_block(
    *,
    draft_tokens: List[int],
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> Tuple[int, List[int], bool]:
    """Reject-sampling acceptance for a draft block. Time O(kV) best/avg, O(kV log V) when top_p<1; space O(kV)."""
    if not draft_tokens:
        return 0, [], False
    token_tensor = torch.tensor(draft_tokens, device=draft_logits.device, dtype=torch.long)
    q_probs = _token_probs_from_logits_batch(
        draft_logits, token_tensor, temperature=temperature, top_p=top_p
    )
    p_probs = _token_probs_from_logits_batch(
        target_logits, token_tensor, temperature=temperature, top_p=top_p
    )
    accept_probs = torch.where(
        q_probs <= 0.0,
        torch.zeros_like(p_probs),
        torch.minimum(torch.ones_like(p_probs), p_probs / q_probs),
    )
    accept_draws = torch.rand_like(accept_probs)
    accepted = accept_draws < accept_probs
    if bool(torch.all(accepted).item()):
        return len(draft_tokens), draft_tokens, False

    reject_idx = int(torch.argmax(torch.logical_not(accepted).to(torch.int32)).item())
    token = sample_next_token(target_logits[reject_idx], temperature=temperature, top_p=top_p)
    new_tokens = list(draft_tokens[:reject_idx]) + [int(token)]
    return reject_idx, new_tokens, True


class LowRankHead(nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, int(rank), bias=False)
        self.out = nn.Linear(int(rank), vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.proj(x))


class FeatureFusion(nn.Module):
    def __init__(self, *, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(int(num_layers))]
        )
        self.weights = nn.Parameter(torch.zeros(int(num_layers)))

    def forward(self, hiddens: List[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.weights, dim=0)
        fused = None
        for idx, (proj, hidden) in enumerate(zip(self.projs, hiddens)):
            contrib = proj(hidden) * weights[idx]
            fused = contrib if fused is None else fused + contrib
        return fused


class Eagle3Speculator(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        spec_layers: int,
        spec_heads: int,
        dropout: float,
        feature_layers: List[int],
        init_weight: Optional[torch.Tensor],
        head_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.feature_layers = [int(i) for i in feature_layers]
        self.fusion = FeatureFusion(hidden_size=hidden_size, num_layers=len(self.feature_layers))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=spec_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(spec_layers))
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.head = LowRankHead(hidden_size=hidden_size, vocab_size=vocab_size, rank=rank)
        else:
            self.head = nn.Linear(hidden_size, vocab_size, bias=False)
            if init_weight is not None:
                self.head.weight.data.copy_(init_weight)

    def fuse(self, hiddens: List[torch.Tensor], attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        fused = self.fusion(hiddens)
        if attention_mask is not None:
            fused = fused * attention_mask.unsqueeze(-1)
        return fused

    def decode(self, *, fused_context: torch.Tensor, token_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        if token_embeds is None or int(token_embeds.shape[1]) == 0:
            x = fused_context
        else:
            x = torch.cat([fused_context, token_embeds], dim=1)
        seq_len = x.shape[1]
        causal_mask = None
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=None)
        x = self.norm(x)
        logits = self.head(x)
        return logits[:, -1, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class _Eagle3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, *, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = int(seq_len)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x: torch.Tensor, *, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(seq_len) > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=int(seq_len), device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class _Eagle3LinearScalingRotaryEmbedding(_Eagle3RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> None:
        self.scaling_factor = float(scaling_factor)
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)

    def _set_cos_sin_cache(self, *, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = int(seq_len)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class _Eagle3DynamicNTKScalingRotaryEmbedding(_Eagle3RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> None:
        self.scaling_factor = float(scaling_factor)
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)

    def _set_cos_sin_cache(self, *, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = int(seq_len)
        if int(seq_len) > int(self.max_position_embeddings):
            base = self.base * (
                (self.scaling_factor * float(seq_len) / float(self.max_position_embeddings))
                - (self.scaling_factor - 1.0)
            ) ** (self.dim / max(self.dim - 2, 1))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class _Eagle3LlamaAttention(nn.Module):
    def __init__(self, config: Eagle3HFConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = int(config.max_position_embeddings)

        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope(config)

    def _init_rope(self, config: Eagle3HFConfig) -> None:
        if config.rope_scaling is None:
            self.rotary_emb = _Eagle3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=config.rope_theta,
            )
            return
        scaling_type = str(config.rope_scaling.get("type", ""))
        scaling_factor = float(config.rope_scaling.get("factor", 1.0))
        if scaling_type == "linear":
            self.rotary_emb = _Eagle3LinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=config.rope_theta,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = _Eagle3DynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=config.rope_theta,
                scaling_factor=scaling_factor,
            )
        else:
            self.rotary_emb = _Eagle3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=config.rope_theta,
            )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class _Eagle3LlamaMLP(nn.Module):
    def __init__(self, config: Eagle3HFConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[str(config.hidden_act)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Eagle3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(hidden_size)))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class _Eagle3LlamaDecoderLayer(nn.Module):
    def __init__(self, config: Eagle3HFConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.self_attn = _Eagle3LlamaAttention(config)
        self.mlp = _Eagle3LlamaMLP(config)
        self.hidden_norm = _Eagle3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = _Eagle3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _Eagle3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Eagle3HFSpeculator(nn.Module):
    def __init__(
        self,
        *,
        config: Eagle3HFConfig,
        target_hidden_size: int,
        target_vocab_size: int,
        draft_vocab_size: int,
        feature_layers: List[int],
    ) -> None:
        super().__init__()
        self.feature_layers = [int(i) for i in feature_layers]
        self.hidden_size = int(config.hidden_size)
        self.target_vocab_size = int(target_vocab_size)
        self.draft_vocab_size = int(draft_vocab_size)
        self.fc = nn.Linear(
            int(target_hidden_size) * len(self.feature_layers),
            self.hidden_size,
            bias=False,
        )
        self.midlayer = _Eagle3LlamaDecoderLayer(config)
        self.norm = _Eagle3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(self.target_vocab_size, dtype=torch.bool))
        self.register_buffer(
            "_draft_ids", torch.arange(self.draft_vocab_size, dtype=torch.long), persistent=False
        )

    def fuse(self, hiddens: List[torch.Tensor], attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not hiddens:
            raise ValueError("HF speculator requires non-empty hidden states.")
        fused = torch.cat(hiddens, dim=-1).to(self.fc.weight.dtype)
        fused = self.fc(fused)
        if attention_mask is not None:
            fused = fused * attention_mask.unsqueeze(-1)
        return fused

    def _map_draft_logits(self, draft_logits: torch.Tensor) -> torch.Tensor:
        target_ids = self._draft_ids + self.d2t
        target_logits = torch.full(
            (draft_logits.shape[0], self.target_vocab_size),
            float("-inf"),
            device=draft_logits.device,
            dtype=draft_logits.dtype,
        )
        target_logits.index_copy_(1, target_ids, draft_logits)
        return target_logits

    def decode(self, *, fused_context: torch.Tensor, token_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        if token_embeds is None or int(token_embeds.shape[1]) == 0:
            token_embeds = torch.zeros_like(fused_context)
        if fused_context.shape[1] != token_embeds.shape[1]:
            fused_context = fused_context.expand(-1, token_embeds.shape[1], -1)
        dtype = self.lm_head.weight.dtype
        fused_context = fused_context.to(dtype)
        token_embeds = token_embeds.to(dtype)

        seq_len = int(token_embeds.shape[1])
        attention_mask = None
        if seq_len > 1:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=token_embeds.device),
                diagonal=1,
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        position_ids = torch.arange(seq_len, device=token_embeds.device).unsqueeze(0)
        position_ids = position_ids.expand(token_embeds.shape[0], -1)

        hidden_states = fused_context
        if hidden_states.shape[-1] != self.hidden_size:
            hidden_states = self.fc(hidden_states)
        hidden_states = self.midlayer(
            input_emb=token_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.norm(hidden_states)
        draft_logits = self.lm_head(hidden_states[:, -1, :])
        return self._map_draft_logits(draft_logits)


def _parse_hf_eagle3_config(cfg: Dict[str, Any]) -> Eagle3HFConfig:
    hidden_size = int(cfg.get("hidden_size", 0))
    intermediate_size = int(cfg.get("intermediate_size", 0))
    num_attention_heads = int(cfg.get("num_attention_heads", 0))
    if hidden_size <= 0 or intermediate_size <= 0 or num_attention_heads <= 0:
        raise ValueError("Invalid HF speculator config: missing core dimensions.")
    num_key_value_heads = int(cfg.get("num_key_value_heads", num_attention_heads))
    max_position_embeddings = int(cfg.get("max_position_embeddings", 2048))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_scaling = cfg.get("rope_scaling")
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        rope_scaling = None
    hidden_act = str(cfg.get("hidden_act", "silu"))
    head_dim = cfg.get("head_dim")
    head_dim = int(head_dim) if head_dim is not None else None
    return Eagle3HFConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        hidden_act=hidden_act,
        head_dim=head_dim,
    )


def _find_hf_speculator_ckpt(speculator_dir: Path) -> Optional[Path]:
    candidate = speculator_dir / "pytorch_model.bin"
    if candidate.is_file():
        return candidate
    return None


def _load_hf_speculator(
    *,
    target: AutoModelForCausalLM,
    speculator_dir: Path,
    speculator_ckpt: Path,
    spec_len: int,
    feature_layers: Optional[List[int]],
) -> Tuple[Eagle3HFSpeculator, int]:
    cfg_path = speculator_dir / "config.json"
    if not cfg_path.is_file():
        cfg_path = speculator_ckpt.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json for HF speculator: {speculator_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    hf_cfg = _parse_hf_eagle3_config(cfg)

    state = torch.load(speculator_ckpt, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"HF speculator checkpoint must be a state dict: {speculator_ckpt}")
    fc_weight = state.get("fc.weight")
    if fc_weight is None:
        raise KeyError("HF speculator checkpoint missing fc.weight.")
    if int(fc_weight.shape[0]) != hf_cfg.hidden_size:
        raise ValueError("HF speculator hidden_size does not match checkpoint.")

    target_hidden_size = int(getattr(target.config, "hidden_size", hf_cfg.hidden_size))
    if target_hidden_size <= 0:
        target_hidden_size = hf_cfg.hidden_size
    feature_count = int(fc_weight.shape[1]) // int(target_hidden_size)
    if int(fc_weight.shape[1]) % int(target_hidden_size) != 0:
        raise ValueError("HF speculator fc input does not align with target hidden size.")

    if feature_layers is None:
        num_layers = int(getattr(target.config, "num_hidden_layers", 0))
        if num_layers <= 0:
            raise ValueError("Target model missing num_hidden_layers for HF speculator.")
        feature_layers = _select_feature_layers(num_layers, count=feature_count)
    if len(feature_layers) != feature_count:
        raise ValueError("HF speculator feature_layers count does not match checkpoint.")

    expected_fc_in = int(target_hidden_size) * len(feature_layers)
    if expected_fc_in != int(fc_weight.shape[1]):
        raise ValueError("HF speculator feature_layers do not match fc input width.")

    draft_vocab_size = int(cfg.get("draft_vocab_size") or 0)
    if draft_vocab_size <= 0:
        lm_head_weight = state.get("lm_head.weight")
        if lm_head_weight is None:
            raise KeyError("HF speculator checkpoint missing lm_head.weight.")
        draft_vocab_size = int(lm_head_weight.shape[0])

    speculator = Eagle3HFSpeculator(
        config=hf_cfg,
        target_hidden_size=target_hidden_size,
        target_vocab_size=int(target.config.vocab_size),
        draft_vocab_size=draft_vocab_size,
        feature_layers=feature_layers,
    )
    speculator.load_state_dict(state, strict=True)
    target_ids = speculator._draft_ids + speculator.d2t
    if int(target_ids.max().item()) >= speculator.target_vocab_size or int(target_ids.min().item()) < 0:
        raise ValueError("HF speculator token mapping is out of target vocab range.")
    speculator = speculator.to(dtype=next(target.parameters()).dtype)
    speculator.eval()
    return speculator, spec_len


def load_speculator(
    *,
    target: AutoModelForCausalLM,
    speculator_dir: Path,
    speculator_ckpt: Optional[Path],
    spec_len: int,
    spec_layers: int,
    spec_heads: int,
    head_rank: Optional[int],
    dropout: float,
) -> Tuple[Eagle3Speculator, int]:
    cfg_path = speculator_dir / "speculator_config.json"
    feature_layers: Optional[List[int]] = None
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        spec_len = int(cfg.get("spec_len", spec_len))
        spec_layers = int(cfg.get("spec_layers", spec_layers))
        spec_heads = int(cfg.get("spec_heads", spec_heads))
        if "feature_layers" in cfg:
            feature_layers = [int(i) for i in cfg.get("feature_layers") or []]
        if "head_rank" in cfg:
            head_rank = cfg.get("head_rank", head_rank)

    hidden_size = int(target.config.hidden_size)
    vocab_size = int(target.config.vocab_size)
    if spec_heads <= 0:
        cfg_heads = int(target.config.num_attention_heads)
        if cfg_heads > 0:
            spec_heads = cfg_heads
        else:
            spec_heads = max(1, hidden_size // 64)
        while spec_heads > 1 and hidden_size % spec_heads != 0:
            spec_heads -= 1

    if speculator_ckpt is not None and speculator_ckpt.suffix == ".bin":
        return _load_hf_speculator(
            target=target,
            speculator_dir=speculator_dir,
            speculator_ckpt=speculator_ckpt,
            spec_len=spec_len,
            feature_layers=feature_layers,
        )

    if speculator_ckpt is None:
        latest = _pick_latest_checkpoint(speculator_dir / "checkpoints")
        if latest is None:
            hf_ckpt = _find_hf_speculator_ckpt(speculator_dir)
            if hf_ckpt is not None:
                return _load_hf_speculator(
                    target=target,
                    speculator_dir=speculator_dir,
                    speculator_ckpt=hf_ckpt,
                    spec_len=spec_len,
                    feature_layers=feature_layers,
                )
            raise FileNotFoundError(f"No checkpoints under {speculator_dir}/checkpoints")
        speculator_ckpt = latest / "speculator.pt"
    if not speculator_ckpt.is_file():
        raise FileNotFoundError(f"Speculator checkpoint not found: {speculator_ckpt}")

    init_weight = target.lm_head.weight.detach().clone()

    feature_layers = _resolve_feature_layers(
        feature_layers, num_layers=int(target.config.num_hidden_layers)
    )

    speculator = Eagle3Speculator(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        spec_layers=spec_layers,
        spec_heads=spec_heads,
        dropout=dropout,
        feature_layers=feature_layers,
        init_weight=init_weight,
        head_rank=head_rank,
    )

    state = torch.load(speculator_ckpt, map_location="cpu")
    speculator.load_state_dict(state, strict=True)
    speculator.eval()
    return speculator, spec_len


@torch.inference_mode()
def baseline_decode(
    *,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
) -> torch.Tensor:
    device = input_ids.device
    output_ids = input_ids.clone()
    out = target(input_ids=output_ids, use_cache=bool(use_cache), return_dict=True)
    last_logits = out.logits[:, -1, :]
    past = out.past_key_values if use_cache else None
    for _ in range(int(max_new_tokens)):
        token = sample_next_token(last_logits, temperature=temperature, top_p=top_p)
        output_ids = torch.cat([output_ids, torch.tensor([[token]], device=device, dtype=torch.long)], dim=1)
        if eos_token_id is not None and token == eos_token_id:
            break
        if use_cache:
            out = target(input_ids=output_ids[:, -1:], past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            last_logits = out.logits[:, -1, :]
        else:
            out = target(input_ids=output_ids, use_cache=False, return_dict=True)
            last_logits = out.logits[:, -1, :]
    return output_ids


@torch.inference_mode()
def _speculative_decode_qwen3(
    *,
    target: AutoModelForCausalLM,
    speculator: Eagle3Speculator,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
    optimized: bool,
    max_consecutive_misses: int = 2,
    collect_stats: bool,
) -> Tuple[torch.Tensor, Optional[SpecStats]]:
    device = input_ids.device
    output_ids = input_ids.clone()
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
    target_prefill_calls = 0
    target_verify_calls = 0
    target_generate_calls = 0
    target_generated = 0

    t0 = time.perf_counter()
    out = target(
        input_ids=output_ids,
        use_cache=bool(use_cache),
        output_hidden_states=True,
        return_dict=True,
    )
    prefill_s = time.perf_counter() - t0
    target_time_s += prefill_s
    target_prefill_time_s += prefill_s
    target_prefill_calls += 1
    past = out.past_key_values if use_cache else None
    layer_hiddens = _extract_hidden_layers(out, speculator.feature_layers)
    last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]
    last_logits = out.logits[:, -1, :]

    while produced < int(max_new_tokens):
        remaining = int(max_new_tokens) - produced
        block_len = min(int(spec_len), int(remaining))
        fused_context = speculator.fuse(last_layer_hiddens, attention_mask=None)
        draft_tokens: List[int] = []
        draft_logits_steps: List[torch.Tensor] = []
        token_embeds = None
        for _ in range(int(block_len)):
            t0 = time.perf_counter()
            logits = speculator.decode(fused_context=fused_context, token_embeds=token_embeds)
            spec_time_s += time.perf_counter() - t0
            token = sample_next_token(logits[0], temperature=temperature, top_p=top_p)
            draft_tokens.append(int(token))
            draft_logits_steps.append(logits)
            token_embed = _embed_tokens(target, [int(token)])
            token_embeds = token_embed if token_embeds is None else torch.cat([token_embeds, token_embed], dim=1)
        draft_logits = torch.cat(draft_logits_steps, dim=0)

        draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
        if use_cache:
            past_snapshot = _clone_past_key_values(past)
            t0 = time.perf_counter()
            block_out = target(
                input_ids=draft_tensor,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            verify_s = time.perf_counter() - t0
            target_time_s += verify_s
            target_verify_time_s += verify_s
            target_verify_calls += 1
            block_logits = block_out.logits
            block_layer_hiddens = _extract_hidden_layers(block_out, speculator.feature_layers)
        else:
            t0 = time.perf_counter()
            block_out = target(
                input_ids=torch.cat([output_ids, draft_tensor], dim=1),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            verify_s = time.perf_counter() - t0
            target_time_s += verify_s
            target_verify_time_s += verify_s
            target_verify_calls += 1
            block_logits = block_out.logits[:, -len(draft_tokens) :, :]
            block_layer_hiddens = [
                h[:, -len(draft_tokens) :, :] for h in _extract_hidden_layers(block_out, speculator.feature_layers)
            ]
            past_snapshot = None
        shifted_logits = torch.cat([last_logits.unsqueeze(1), block_logits[:, :-1, :]], dim=1)
        accept_len, new_tokens, rejected = _accept_reject_block(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            target_logits=shifted_logits[0],
            temperature=temperature,
            top_p=top_p,
        )

        bonus_added = False
        if not rejected and accept_len == len(draft_tokens) and remaining > len(draft_tokens):
            bonus_logits = block_logits[:, -1, :]
            bonus_token = sample_next_token(bonus_logits, temperature=temperature, top_p=top_p)
            new_tokens.append(int(bonus_token))
            bonus_added = True

        if collect_stats:
            total_accept += int(accept_len)
            total_draft += int(block_len)
            steps += 1
            if accept_len == 0:
                zero_accept += 1

        if accept_len == 0:
            consecutive_misses += 1
        else:
            consecutive_misses = 0

        token_sources = [True] * int(accept_len)
        if rejected or bonus_added:
            token_sources.append(False)

        remaining = int(max_new_tokens) - produced
        if len(new_tokens) > remaining:
            new_tokens = new_tokens[:remaining]
            token_sources = token_sources[:remaining]

        eos_hit = False
        if eos_token_id is not None and eos_token_id in new_tokens:
            eos_idx = new_tokens.index(eos_token_id)
            new_tokens = new_tokens[: eos_idx + 1]
            token_sources = token_sources[: eos_idx + 1]
            eos_hit = True
        if collect_stats:
            accepted_step = sum(1 for src in token_sources if src)
            accepted_output += int(accepted_step)
            target_generated += int(len(token_sources) - accepted_step)

        if not new_tokens:
            break

        output_ids = torch.cat(
            [output_ids, torch.tensor([new_tokens], dtype=torch.long, device=device)], dim=1
        )
        produced += len(new_tokens)

        if eos_hit:
            break

        if use_cache:
            if accept_len == len(draft_tokens):
                past = block_out.past_key_values
                last_layer_hiddens = [h[:, -1:, :] for h in block_layer_hiddens]
                last_logits = block_logits[:, -1, :]
                if bonus_added:
                    bonus_tensor = torch.tensor(
                        [[new_tokens[-1]]], dtype=torch.long, device=device
                    )
                    t0 = time.perf_counter()
                    bonus_out = target(
                        input_ids=bonus_tensor,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    gen_s = time.perf_counter() - t0
                    target_time_s += gen_s
                    target_generate_time_s += gen_s
                    target_generate_calls += 1
                    past = bonus_out.past_key_values
                    last_layer_hiddens = [
                        h[:, -1:, :] for h in _extract_hidden_layers(bonus_out, speculator.feature_layers)
                    ]
                    last_logits = bonus_out.logits[:, -1, :]
            else:
                if past_snapshot is None:
                    t0 = time.perf_counter()
                    out = target(
                        input_ids=output_ids,
                        use_cache=bool(use_cache),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    prefill_s = time.perf_counter() - t0
                    target_time_s += prefill_s
                    target_prefill_time_s += prefill_s
                    target_prefill_calls += 1
                    past = out.past_key_values if use_cache else None
                    last_layer_hiddens = [
                        h[:, -1:, :] for h in _extract_hidden_layers(out, speculator.feature_layers)
                    ]
                    last_logits = out.logits[:, -1, :]
                else:
                    past = past_snapshot
                    accept_tensor = torch.tensor([new_tokens], dtype=torch.long, device=device)
                    t0 = time.perf_counter()
                    accept_out = target(
                        input_ids=accept_tensor,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    gen_s = time.perf_counter() - t0
                    target_time_s += gen_s
                    target_generate_time_s += gen_s
                    target_generate_calls += 1
                    past = accept_out.past_key_values
                    last_layer_hiddens = [
                        h[:, -1:, :] for h in _extract_hidden_layers(accept_out, speculator.feature_layers)
                    ]
                    last_logits = accept_out.logits[:, -1, :]
        else:
            t0 = time.perf_counter()
            out = target(
                input_ids=output_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            prefill_s = time.perf_counter() - t0
            target_time_s += prefill_s
            target_prefill_time_s += prefill_s
            target_prefill_calls += 1
            last_layer_hiddens = [
                h[:, -1:, :] for h in _extract_hidden_layers(out, speculator.feature_layers)
            ]
            last_logits = out.logits[:, -1, :]

        if optimized and consecutive_misses >= max_consecutive_misses:
            break

    if optimized and produced < int(max_new_tokens):
        t0 = time.perf_counter()
        before_len = int(output_ids.shape[1])
        fallback_ids = baseline_decode(
            target=target,
            input_ids=output_ids,
            max_new_tokens=int(max_new_tokens) - produced,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
        )
        gen_s = time.perf_counter() - t0
        target_time_s += gen_s
        target_generate_time_s += gen_s
        if collect_stats:
            target_generated += int(fallback_ids.shape[1]) - before_len
            target_generate_calls += int(fallback_ids.shape[1]) - before_len
        output_ids = fallback_ids

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
            target_prefill_calls=int(target_prefill_calls),
            target_verify_calls=int(target_verify_calls),
            target_generate_calls=int(target_generate_calls),
            target_generated=int(target_generated),
        )
    return output_ids, stats


@torch.inference_mode()
def _speculative_decode_minillm(
    *,
    target: MiniLLMForCausalLM,
    speculator: Eagle3Speculator,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    optimized: bool,
    max_consecutive_misses: int = 2,
    collect_stats: bool,
) -> Tuple[torch.Tensor, Optional[SpecStats]]:
    device = input_ids.device
    output_ids = input_ids.clone()
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
    target_prefill_calls = 0
    target_verify_calls = 0
    target_generate_calls = 0
    target_generated = 0

    while produced < int(max_new_tokens):
        t0 = time.perf_counter()
        hidden, layer_hiddens = _minillm_forward_hidden_states(
            target,
            output_ids,
            attention_mask=None,
            layer_ids=speculator.feature_layers,
        )
        prefill_s = time.perf_counter() - t0
        target_time_s += prefill_s
        target_prefill_time_s += prefill_s
        target_prefill_calls += 1
        last_hidden = hidden[:, -1:, :]
        last_layer_hiddens = [h[:, -1:, :] for h in layer_hiddens]

        remaining = int(max_new_tokens) - produced
        block_len = min(int(spec_len), int(remaining))
        fused_context = speculator.fuse(last_layer_hiddens, attention_mask=None)
        draft_tokens: List[int] = []
        draft_logits_steps: List[torch.Tensor] = []
        token_embeds = None
        for _ in range(int(block_len)):
            t0 = time.perf_counter()
            logits = speculator.decode(fused_context=fused_context, token_embeds=token_embeds)
            spec_time_s += time.perf_counter() - t0
            token = sample_next_token(logits[0], temperature=temperature, top_p=top_p)
            draft_tokens.append(int(token))
            draft_logits_steps.append(logits)
            token_embed = _embed_tokens(target, [int(token)])
            token_embeds = token_embed if token_embeds is None else torch.cat([token_embeds, token_embed], dim=1)
        draft_logits = torch.cat(draft_logits_steps, dim=0)

        draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
        full_ids = torch.cat([output_ids, draft_tensor], dim=1)
        t0 = time.perf_counter()
        full_hidden, _ = _minillm_forward_hidden_states(
            target,
            full_ids,
            attention_mask=None,
            layer_ids=speculator.feature_layers,
        )
        verify_s = time.perf_counter() - t0
        target_time_s += verify_s
        target_verify_time_s += verify_s
        target_verify_calls += 1
        block_hidden = full_hidden[:, -len(draft_tokens) :, :]
        prev_hidden = torch.cat([last_hidden, block_hidden], dim=1)[:, :-1, :]
        block_logits = _project_logits_minillm(target, prev_hidden)

        accept_len, new_tokens, rejected = _accept_reject_block(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            target_logits=block_logits[0],
            temperature=temperature,
            top_p=top_p,
        )

        bonus_added = False
        if not rejected and accept_len == len(draft_tokens) and remaining > len(draft_tokens):
            bonus_logits = _project_logits_minillm(target, block_hidden[:, -1:, :])[:, -1, :]
            bonus_token = sample_next_token(bonus_logits, temperature=temperature, top_p=top_p)
            new_tokens.append(int(bonus_token))
            bonus_added = True

        if collect_stats:
            total_accept += int(accept_len)
            total_draft += int(block_len)
            steps += 1
            if accept_len == 0:
                zero_accept += 1

        if accept_len == 0:
            consecutive_misses += 1
        else:
            consecutive_misses = 0

        token_sources = [True] * int(accept_len)
        if rejected or bonus_added:
            token_sources.append(False)
        eos_hit = False
        if eos_token_id is not None and eos_token_id in new_tokens:
            eos_idx = new_tokens.index(eos_token_id)
            new_tokens = new_tokens[: eos_idx + 1]
            token_sources = token_sources[: eos_idx + 1]
            eos_hit = True
        if collect_stats:
            accepted_step = sum(1 for src in token_sources if src)
            accepted_output += int(accepted_step)
            target_generated += int(len(token_sources) - accepted_step)

        output_ids = torch.cat(
            [output_ids, torch.tensor([new_tokens], dtype=torch.long, device=device)], dim=1
        )
        produced += len(new_tokens)

        if eos_hit:
            break

        if optimized and consecutive_misses >= max_consecutive_misses:
            break

    if optimized and produced < int(max_new_tokens):
        t0 = time.perf_counter()
        before_len = int(output_ids.shape[1])
        output_ids = baseline_decode(
            target=target,
            input_ids=output_ids,
            max_new_tokens=int(max_new_tokens) - produced,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            use_cache=False,
        )
        gen_s = time.perf_counter() - t0
        target_time_s += gen_s
        target_generate_time_s += gen_s
        if collect_stats:
            target_generated += int(output_ids.shape[1]) - before_len
            target_generate_calls += int(output_ids.shape[1]) - before_len

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
            target_prefill_calls=int(target_prefill_calls),
            target_verify_calls=int(target_verify_calls),
            target_generate_calls=int(target_generate_calls),
            target_generated=int(target_generated),
        )
    return output_ids, stats


@torch.inference_mode()
def speculative_decode(
    *,
    target: AutoModelForCausalLM,
    speculator: Eagle3Speculator,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
    optimized: bool,
    max_consecutive_misses: int = 2,
) -> torch.Tensor:
    output_ids, _ = speculative_decode_with_stats(
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
    )
    return output_ids


@torch.inference_mode()
def speculative_decode_with_stats(
    *,
    target: AutoModelForCausalLM,
    speculator: Eagle3Speculator,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
    optimized: bool,
    max_consecutive_misses: int = 2,
) -> Tuple[torch.Tensor, SpecStats]:
    if isinstance(target, MiniLLMForCausalLM):
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
    else:
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
    if stats is None:
        raise RuntimeError("Speculative decode stats missing")
    return output_ids, stats


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--minillm_ckpt", type=str, default=None)
    parser.add_argument("--minillm_config", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument("--speculator_dir", type=str, default="out/eagle3_speculator/qwen3_0.6b")
    parser.add_argument("--speculator_ckpt", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=_default_device_name())
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--spec_len", type=int, default=None, help="Draft length for speculator (auto if unset).")
    parser.add_argument("--spec_layers", type=int, default=None, help="Transformer layers in speculator (auto if unset).")
    parser.add_argument("--spec_heads", type=int, default=0)
    parser.add_argument(
        "--head_rank",
        type=int,
        default=None,
        help="Low-rank speculator head size (overrides config if set; full head if unset).",
    )
    parser.add_argument("--spec_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no_speculator", action="store_true")
    return parser


def run_cli() -> None:
    parser = build_arg_parser(
        description="Speculative decoding with EAGLE-3 speculator (Torch backend)."
    )
    args = parser.parse_args()
    optimized = True
    use_cache = True

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    dtype = _maybe_adjust_dtype_for_device(device, dtype)

    target, tokenizer = _load_target_and_tokenizer(args, device, dtype)

    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    speculator = None
    spec_len, spec_layers = _resolve_spec_config(
        args.spec_len, args.spec_layers, param_count=_count_params_torch(target)
    )
    head_rank = args.head_rank if args.head_rank is not None and int(args.head_rank) > 0 else None
    if not args.no_speculator:
        speculator_dir = Path(args.speculator_dir)
        speculator_ckpt = Path(args.speculator_ckpt) if args.speculator_ckpt else None
        speculator, spec_len = load_speculator(
            target=target,
            speculator_dir=speculator_dir,
            speculator_ckpt=speculator_ckpt,
            spec_len=spec_len,
            spec_layers=spec_layers,
            spec_heads=args.spec_heads,
            head_rank=head_rank,
            dropout=args.spec_dropout,
        )
        speculator = speculator.to(device)

    start = time.time()
    if speculator is None:
        output_ids = baseline_decode(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
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
    out_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(out_text)
    print(f"[time] {elapsed:.2f}s")
