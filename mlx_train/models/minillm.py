from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..config import MiniLLMConfig
from ..nn.lora import LoRAConfig, apply_lora
from ..ops import metal as metal_ops

if TYPE_CHECKING:
    from ..trace import ActivationTracer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        if metal_ops.enabled():
            try:
                return metal_ops.rms_norm(x, self.weight, self.eps)
            except Exception:
                pass
        x_fp32 = x.astype(mx.float32)
        norm = x_fp32 * mx.rsqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + self.eps)
        return norm.astype(x.dtype) * self.weight.astype(x.dtype)


def yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def yarn_find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int) -> Tuple[int, int]:
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_idx: int, max_idx: int, dim: int) -> mx.array:
    if min_idx == max_idx:
        max_idx += 1
    linear = (mx.arange(dim, dtype=mx.float32) - float(min_idx)) / float(max_idx - min_idx)
    return mx.clip(linear, 0, 1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_rope_freqs(
    *, dim: int, end: int, base: float, rope_scaling: Optional[dict]
) -> Tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)))
    mscale = 1.0
    if rope_scaling is not None:
        scaling_type = rope_scaling.get("type", "")
        factor = rope_scaling.get("factor", 1.0)
        if scaling_type == "linear":
            inv_freq = inv_freq / float(factor)
        elif scaling_type == "yarn":
            orig_max = rope_scaling.get("original_max_position_embeddings", end)
            beta_fast = rope_scaling.get("beta_fast", 32)
            beta_slow = rope_scaling.get("beta_slow", 1)
            if end > orig_max:
                low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
                inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
                inv_freq = inv_freq / float(factor) * (1 - inv_freq_mask) + inv_freq * inv_freq_mask
            mscale = yarn_get_mscale(factor, rope_scaling.get("mscale", 1.0)) / yarn_get_mscale(
                factor, rope_scaling.get("mscale_all_dim", 0.0)
            )
    t = mx.arange(end, dtype=mx.float32)
    freqs = t[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs) * float(mscale)
    sin = mx.sin(freqs) * float(mscale)
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)
    return cos, sin


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    def rotate_half(x: mx.array) -> mx.array:
        half = int(x.shape[-1] // 2)
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _topk_with_indices(x: mx.array, *, k: int) -> Tuple[mx.array, mx.array]:
    if k <= 0:
        raise ValueError("k must be > 0")
    k = int(k)
    last_dim = int(x.shape[-1])
    if k > last_dim:
        k = last_dim
    idx = mx.argpartition(-x, kth=k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(x, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    return vals, idx


def _make_causal_mask(q_len: int, k_len: int) -> mx.array:
    mask = mx.triu(mx.ones((int(q_len), int(k_len)), dtype=mx.float32), k=1)
    return mask * mx.array(-1e9, dtype=mx.float32)


def _build_index_mask(topk_idx: mx.array, *, k_len: int) -> mx.array:
    eye = mx.eye(int(k_len), dtype=mx.float32)
    one_hot = eye[topk_idx]
    allow = mx.sum(one_hot, axis=2) > 0
    neg_inf = mx.array(-1e9, dtype=mx.float32)
    return mx.where(allow, mx.array(0.0, dtype=mx.float32), neg_inf)


def _resolve_attention_mask(
    attention_mask: Optional[mx.array],
    *,
    q_len: int,
    k_len: int,
    topk_idx: Optional[mx.array],
    use_causal: bool,
) -> Optional[mx.array | str]:
    mask = None
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            pad = (1.0 - attention_mask.astype(mx.float32)) * mx.array(-1e9, dtype=mx.float32)
            mask = pad
        else:
            if attention_mask.dtype == mx.bool_:
                mask = mx.where(attention_mask, mx.array(0.0, dtype=mx.float32), mx.array(-1e9, dtype=mx.float32))
            else:
                mask = attention_mask
            if attention_mask.ndim == 4:
                use_causal = False

    if topk_idx is not None:
        index_mask = _build_index_mask(topk_idx, k_len=int(k_len))
        index_mask = index_mask[:, None, :, :]
        mask = index_mask if mask is None else mask + index_mask

    if use_causal:
        causal = _make_causal_mask(int(q_len), int(k_len))[None, None, :, :]
        mask = causal if mask is None else mask + causal

    if mask is None and use_causal:
        return "causal"
    return mask


class MiniLLMIndexer(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        cfg = config.finalize()
        self.enabled = bool(cfg.index_n_heads) and int(cfg.index_topk) > 0
        self.index_topk = int(cfg.index_topk)
        self.n_heads = int(cfg.index_n_heads)
        self.head_dim = int(cfg.index_head_dim)
        if not self.enabled:
            self.q_proj = None
            self.k_proj = None
            return
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: Optional[mx.array],
        past_len: int,
    ) -> Optional[mx.array]:
        """Return per-query top-k key indices for DSA masking.

        Complexity: O(B * H * T^2) time, O(B * T^2) space for score tensors,
        where B is batch size, H index heads, and T sequence length.
        """
        if not self.enabled or self.q_proj is None or self.k_proj is None:
            return None
        if int(past_len) > 0:
            return None
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(hidden_states).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = mx.matmul(q.astype(mx.float32), k.astype(mx.float32).transpose(0, 1, 3, 2))
        scores = scores * float(self.scale)
        if attention_mask is not None and attention_mask.ndim == 2:
            pad = (1.0 - attention_mask.astype(mx.float32)) * mx.array(-1e9, dtype=mx.float32)
            scores = scores + pad
        scores = mx.mean(scores, axis=1)
        k_len = int(scores.shape[-1])
        topk = min(int(self.index_topk), int(k_len))
        if topk >= k_len:
            return None
        _, topk_idx = _topk_with_indices(scores, k=topk)
        return topk_idx


class MiniLLMAttention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        cfg = config.finalize()
        self.n_heads = cfg.num_attention_heads
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.q_head_dim = int(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        self.v_head_dim = cfg.v_head_dim
        self.scale = self.q_head_dim ** -0.5
        self.kv_lora_rank = cfg.kv_lora_rank

        self.q_lora_rank = cfg.q_lora_rank if cfg.q_lora_rank and cfg.q_lora_rank > 0 else None
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(cfg.hidden_size, self.q_lora_rank, bias=cfg.attention_bias)
            self.q_a_norm = RMSNorm(self.q_lora_rank, eps=cfg.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.n_heads * self.q_head_dim, bias=False)

        self.kv_a_proj = nn.Linear(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            bias=cfg.attention_bias,
        )
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank, eps=cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        if cfg.rope_scaling is not None and cfg.rope_scaling.get("type") == "yarn":
            mscale_all_dim = cfg.rope_scaling.get("mscale_all_dim", 0.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(cfg.rope_scaling.get("factor", 1.0), mscale_all_dim)
                self.scale = self.scale * mscale * mscale
        self.indexer = MiniLLMIndexer(cfg)

    def __call__(
        self, x: mx.array, *, start_pos: int = 0, attention_mask: Optional[mx.array] = None, cos: mx.array, sin: mx.array
    ) -> mx.array:
        bsz, seq_len, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.reshape(bsz, seq_len, self.n_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        kv = self.kv_a_proj(x)
        kv, k_pe = mx.split(kv, [self.kv_lora_rank], axis=-1)
        kv = self.kv_a_norm(kv)
        kv = self.kv_b_proj(kv)
        kv = kv.reshape(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(0, 2, 1, 3)
        k_nope, v = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        k_pe = k_pe.reshape(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos[:seq_len], sin[:seq_len])

        q_states = mx.concatenate([q_nope, q_pe], axis=-1)
        k_states = mx.concatenate([k_nope, mx.repeat(k_pe, repeats=self.n_heads, axis=1)], axis=-1)

        topk_idx = self.indexer(x, attention_mask=attention_mask, past_len=int(start_pos))
        mask = _resolve_attention_mask(
            attention_mask,
            q_len=int(seq_len),
            k_len=int(k_states.shape[2]),
            topk_idx=topk_idx,
            use_causal=attention_mask is None,
        )

        out = mx.fast.scaled_dot_product_attention(q_states, k_states, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.n_heads * self.v_head_dim)
        out = self.o_proj(out)
        return self.resid_dropout(out)

    def forward_with_cache(
        self,
        x: mx.array,
        *,
        start_pos: int,
        cache: "LayerKVCache",
        attention_mask: Optional[mx.array] = None,
        trace: Optional["ActivationTracer"] = None,
        layer_id: Optional[int] = None,
        cos: mx.array,
        sin: mx.array,
    ) -> Tuple[mx.array, "LayerKVCache"]:
        bsz, seq_len, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.reshape(bsz, seq_len, self.n_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        kv = self.kv_a_proj(x)
        kv, k_pe = mx.split(kv, [self.kv_lora_rank], axis=-1)
        kv = self.kv_a_norm(kv)
        kv = self.kv_b_proj(kv)
        kv = kv.reshape(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(0, 2, 1, 3)
        k_nope, v = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        k_pe = k_pe.reshape(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos[:seq_len], sin[:seq_len])

        q_states = mx.concatenate([q_nope, q_pe], axis=-1)
        k_states = mx.concatenate([k_nope, mx.repeat(k_pe, repeats=self.n_heads, axis=1)], axis=-1)

        if trace is not None and layer_id is not None:
            trace.record_qkv(layer_id=int(layer_id), start_pos=int(start_pos), q=q_states, k=k_states, v=v)

        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got: {start_pos}")
        if int(start_pos) + int(seq_len) > int(cache.k.shape[2]):
            raise ValueError(
                f"KV cache too small: need {int(start_pos) + int(seq_len)} "
                f"but cache_len={int(cache.k.shape[2])}"
            )

        k_cache = mx.slice_update(cache.k, k_states, start_indices=mx.array([int(start_pos)]), axes=(2,))
        v_cache = mx.slice_update(cache.v, v, start_indices=mx.array([int(start_pos)]), axes=(2,))

        total = int(start_pos) + int(seq_len)
        k_full = mx.slice(
            k_cache,
            start_indices=mx.array([0]),
            axes=(2,),
            slice_size=(int(bsz), int(self.n_heads), int(total), int(self.q_head_dim)),
        )
        v_full = mx.slice(
            v_cache,
            start_indices=mx.array([0]),
            axes=(2,),
            slice_size=(int(bsz), int(self.n_heads), int(total), int(self.v_head_dim)),
        )

        topk_idx = self.indexer(x, attention_mask=attention_mask, past_len=int(start_pos))
        use_causal = attention_mask is None and int(start_pos) == 0
        mask = _resolve_attention_mask(
            attention_mask,
            q_len=int(seq_len),
            k_len=int(k_full.shape[2]),
            topk_idx=topk_idx,
            use_causal=use_causal,
        )

        if trace is not None and trace.cfg is not None and bool(trace.cfg.record_attn):
            q_for_trace = q_states
            if not bool(trace.cfg.record_attn_all_queries):
                q_for_trace = q_states[:, :, -1:, :]
                query_positions = [int(start_pos) + int(seq_len) - 1]
            else:
                query_positions = list(range(int(start_pos), int(start_pos) + int(seq_len)))

            scores = mx.matmul(
                q_for_trace.astype(mx.float32),
                k_full.astype(mx.float32).transpose(0, 1, 3, 2),
            )
            scores = scores * float(self.scale)
            if attention_mask is None and int(seq_len) > 1:
                key_pos = mx.arange(int(total))[None, :]
                qpos = mx.array([int(p) for p in query_positions], dtype=mx.int32)[:, None]
                allow = key_pos <= qpos
                neg_inf = mx.array(-1e9, dtype=scores.dtype)
                scores = mx.where(allow[None, None, :, :], scores, neg_inf)
            if topk_idx is not None:
                idx_mask = _build_index_mask(topk_idx, k_len=int(total))
                scores = scores + idx_mask[:, None, :, :]
            attn_w = mx.softmax(scores, axis=-1)
            if trace is not None and layer_id is not None:
                trace.record_attn(
                    layer_id=int(layer_id),
                    start_pos=int(start_pos),
                    attn=attn_w,
                    query_positions=query_positions,
                )

        out = mx.fast.scaled_dot_product_attention(q_states, k_full, v_full, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.n_heads * self.v_head_dim)
        out = self.o_proj(out)
        return self.resid_dropout(out), LayerKVCache(k=k_cache, v=v_cache)


class FeedForward(nn.Module):
    def __init__(self, config: MiniLLMConfig, *, intermediate_size: Optional[int] = None):
        super().__init__()
        cfg = config.finalize()
        inter = intermediate_size if intermediate_size is not None else cfg.intermediate_size
        if inter is None:
            raise ValueError("intermediate_size is None after finalize()")
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.hidden_act != "silu":
            raise ValueError(f"Only silu is supported right now, got: {cfg.hidden_act}")

    def __call__(
        self,
        x: mx.array,
        *,
        trace: Optional["ActivationTracer"] = None,
        layer_id: Optional[int] = None,
        start_pos: int = 0,
    ) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if metal_ops.enabled():
            try:
                act = metal_ops.silu_mul(gate, up)
            except Exception:
                act = nn.silu(gate) * up
        else:
            act = nn.silu(gate) * up
        if trace is not None and layer_id is not None:
            trace.record_mlp_act(layer_id=int(layer_id), start_pos=int(start_pos), act=act)
        return self.dropout(self.down_proj(act))


class MoEGate(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        cfg = config.finalize()
        if cfg.n_routed_experts is None:
            raise ValueError("n_routed_experts must be set for MoE")
        self.top_k = int(cfg.num_experts_per_tok)
        self.n_routed_experts = int(cfg.n_routed_experts)
        self.n_group = int(cfg.n_group or 1)
        self.topk_group = int(cfg.topk_group or self.n_group)
        self.scoring_func = cfg.scoring_func
        self.norm_topk_prob = bool(cfg.norm_topk_prob)
        self.routed_scaling_factor = float(cfg.routed_scaling_factor)
        self.aux_loss_alpha = float(cfg.aux_loss_alpha)
        self.seq_aux = bool(cfg.seq_aux)
        limit = math.sqrt(6.0 / float(cfg.hidden_size + cfg.hidden_size))
        self.weight = mx.random.uniform(
            low=-limit,
            high=limit,
            shape=(self.n_routed_experts, cfg.hidden_size),
            dtype=mx.float32,
        )

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        bsz, seq_len, dim = hidden_states.shape
        flat = hidden_states.reshape(-1, dim)
        scores = flat @ self.weight.transpose()
        if self.scoring_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.scoring_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")
        original_scores = scores

        if self.n_group > 1:
            scores = scores.reshape(-1, self.n_group, self.n_routed_experts // self.n_group)
            if self.topk_group < self.n_group:
                if self.scoring_func == "sigmoid":
                    group_scores = mx.sum(mx.topk(scores, k=2, axis=-1), axis=-1)
                else:
                    group_scores = mx.max(scores, axis=-1)
                _, top_groups = _topk_with_indices(group_scores, k=self.topk_group)
                one_hot = mx.eye(self.n_group, dtype=mx.float32)[top_groups]
                selected = mx.sum(one_hot, axis=1) > 0
                mask = ~selected
                scores = mx.where(mask[..., None], mx.array(-1e9, dtype=scores.dtype), scores)
            scores = scores.reshape(-1, self.n_routed_experts)

        topk_weight, topk_idx = _topk_with_indices(scores, k=self.top_k)
        topk_idx = topk_idx.astype(mx.int32)
        if self.scoring_func == "sigmoid" or (self.top_k > 1 and self.norm_topk_prob):
            denom = mx.maximum(mx.sum(topk_weight, axis=-1, keepdims=True), mx.array(1e-9, dtype=topk_weight.dtype))
            topk_weight = topk_weight / denom
        topk_weight = topk_weight * float(self.routed_scaling_factor)

        aux_loss = mx.array(0.0, dtype=mx.float32)
        if self.training and self.aux_loss_alpha > 0.0:
            scores_for_aux = original_scores
            topk_idx_for_aux = topk_idx.reshape(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.reshape(bsz, seq_len, -1)
                one_hot = mx.eye(self.n_routed_experts, dtype=scores_for_aux.dtype)[topk_idx_for_aux]
                ce = mx.mean(one_hot, axis=1) * float(self.n_routed_experts)
                aux_loss = mx.mean(mx.sum(ce * mx.mean(scores_for_seq_aux, axis=1), axis=1))
                aux_loss = aux_loss * float(self.aux_loss_alpha)
            else:
                flat_idx = topk_idx_for_aux.reshape(-1)
                mask_ce = mx.eye(self.n_routed_experts, dtype=scores_for_aux.dtype)[flat_idx]
                ce = mx.mean(mask_ce, axis=0)
                pi = mx.mean(scores_for_aux, axis=0)
                fi = ce * float(self.n_routed_experts)
                aux_loss = mx.sum(pi * fi) * float(self.aux_loss_alpha)

        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        cfg = config.finalize()
        if cfg.n_routed_experts is None:
            raise ValueError("n_routed_experts must be set for MoE")
        self.experts = [
            FeedForward(cfg, intermediate_size=cfg.moe_intermediate_size) for _ in range(int(cfg.n_routed_experts))
        ]
        self.gate = MoEGate(cfg)
        self.shared_expert = None
        if cfg.n_shared_experts and cfg.n_shared_experts > 0:
            shared_dim = int(cfg.n_shared_experts * cfg.moe_intermediate_size)
            self.shared_expert = FeedForward(cfg, intermediate_size=shared_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.aux_loss = mx.array(0.0, dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        *,
        trace: Optional["ActivationTracer"] = None,
        layer_id: Optional[int] = None,
        start_pos: int = 0,
    ) -> mx.array:
        """Compute sparse MoE output with per-token top-k expert weights.

        Complexity: O(T * K * D * I) time and O(T * K * (D + I)) space,
        where T is tokens, K experts per token, D hidden size, I intermediate size.
        """
        identity = x
        orig_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        topk_idx, topk_weight, aux_loss = self.gate(x)
        gate_w = mx.stack([expert.gate_proj.weight for expert in self.experts], axis=0)
        up_w = mx.stack([expert.up_proj.weight for expert in self.experts], axis=0)
        down_w = mx.stack([expert.down_proj.weight for expert in self.experts], axis=0)
        gate_sel = mx.take(gate_w, topk_idx, axis=0)
        up_sel = mx.take(up_w, topk_idx, axis=0)
        down_sel = mx.take(down_w, topk_idx, axis=0)
        gate = mx.einsum("td,tkid->tki", flat, gate_sel)
        up = mx.einsum("td,tkid->tki", flat, up_sel)
        act = nn.silu(gate) * up
        if trace is not None and layer_id is not None:
            act_weighted = mx.sum(act * topk_weight[..., None], axis=1)
            act_weighted = act_weighted.reshape(orig_shape[0], orig_shape[1], -1)
            trace.record_mlp_act(layer_id=int(layer_id), start_pos=int(start_pos), act=act_weighted)
        out = mx.einsum("tki,tkdi->tkd", act, down_sel)
        out = self.dropout(out)
        out = mx.sum(out * topk_weight[..., None], axis=1)
        out = out.reshape(*orig_shape)
        if self.shared_expert is not None:
            out = out + self.shared_expert(identity, trace=trace, layer_id=layer_id, start_pos=start_pos)
        self.aux_loss = aux_loss
        return out


class MiniLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniLLMConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = MiniLLMAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_attn_gate = bool(config.use_attn_gate)
        self.attn_gate_logit = (
            mx.array(float(config.attn_gate_init), dtype=mx.float32) if bool(config.use_attn_gate) else None
        )
        if _is_moe_layer(layer_id, config):
            self.mlp = MoEFeedForward(config)
        else:
            self.mlp = FeedForward(config)

    def __call__(
        self,
        x: mx.array,
        *,
        start_pos: int = 0,
        attention_mask: Optional[mx.array] = None,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        h = self.self_attn(self.input_layernorm(x), start_pos=start_pos, attention_mask=attention_mask, cos=cos, sin=sin)
        if self.attn_gate_logit is not None:
            gate = mx.sigmoid(self.attn_gate_logit).astype(x.dtype)
            x = x + gate * h
        else:
            x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x), start_pos=start_pos)
        return x

    def forward_with_cache(
        self,
        x: mx.array,
        *,
        start_pos: int,
        cache: "LayerKVCache",
        attention_mask: Optional[mx.array] = None,
        trace: Optional["ActivationTracer"] = None,
        cos: mx.array,
        sin: mx.array,
    ) -> Tuple[mx.array, "LayerKVCache"]:
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="x_in_rms", start_pos=int(start_pos), x=x)

        x_attn_in = self.input_layernorm(x)
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="attn_in_rms", start_pos=int(start_pos), x=x_attn_in)

        h, cache = self.self_attn.forward_with_cache(
            x_attn_in,
            start_pos=int(start_pos),
            cache=cache,
            attention_mask=attention_mask,
            trace=trace,
            layer_id=int(self.layer_id),
            cos=cos,
            sin=sin,
        )
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="attn_out_rms", start_pos=int(start_pos), x=h)

        if self.attn_gate_logit is not None:
            gate = mx.sigmoid(self.attn_gate_logit).astype(x.dtype)
            x = x + gate * h
        else:
            x = x + h
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="x_mid_rms", start_pos=int(start_pos), x=x)

        x_mlp_in = self.post_attention_layernorm(x)
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="mlp_in_rms", start_pos=int(start_pos), x=x_mlp_in)

        mlp_out = self.mlp(x_mlp_in, trace=trace, layer_id=int(self.layer_id), start_pos=int(start_pos))
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="mlp_out_rms", start_pos=int(start_pos), x=mlp_out)

        x = x + mlp_out
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="x_out_rms", start_pos=int(start_pos), x=x)

        return x, cache


def _is_moe_layer(layer_id: int, config: MiniLLMConfig) -> bool:
    if not config.use_moe or config.n_routed_experts is None:
        return False
    if layer_id < config.first_k_dense_replace:
        return False
    freq = max(int(config.moe_layer_freq), 1)
    return ((layer_id - config.first_k_dense_replace) % freq) == 0


class MiniLLMModel(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config.finalize()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.layers = [MiniLLMBlock(i, self.config) for i in range(self.config.num_hidden_layers)]
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.checkpoint_every_n: int = 0

        self.freqs_cos, self.freqs_sin = precompute_rope_freqs(
            dim=self.config.qk_rope_head_dim,
            end=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
        )

    def __call__(self, input_ids: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        h = self.embed_tokens(input_ids)
        h = self.dropout(h)
        start_pos = 0
        ckpt_n = int(self.checkpoint_every_n)
        use_ckpt = self.training and ckpt_n > 0
        cos = self.freqs_cos[start_pos : start_pos + int(input_ids.shape[1])]
        sin = self.freqs_sin[start_pos : start_pos + int(input_ids.shape[1])]
        for i, layer in enumerate(self.layers):
            if use_ckpt and (i % ckpt_n == 0):
                h = mx.checkpoint(
                    lambda x, m, c, s, st: layer(x, start_pos=st, attention_mask=m, cos=c, sin=s)
                )(h, attention_mask, cos, sin, start_pos)
            else:
                h = layer(h, start_pos=start_pos, attention_mask=attention_mask, cos=cos, sin=sin)
        return self.norm(h)

    def forward_with_cache(
        self,
        input_ids: mx.array,
        *,
        start_pos: int,
        cache: List["LayerKVCache"],
        attention_mask: Optional[mx.array] = None,
        trace: Optional["ActivationTracer"] = None,
    ) -> Tuple[mx.array, List["LayerKVCache"]]:
        if len(cache) != len(self.layers):
            raise ValueError(f"KV cache layers mismatch: got {len(cache)} want {len(self.layers)}")

        if trace is not None:
            trace.on_input_ids(start_pos=int(start_pos), input_ids=input_ids)

        h = self.embed_tokens(input_ids)
        h = self.dropout(h)
        new_cache: List[LayerKVCache] = []
        cos = self.freqs_cos[start_pos : start_pos + int(input_ids.shape[1])]
        sin = self.freqs_sin[start_pos : start_pos + int(input_ids.shape[1])]
        for layer, layer_cache in zip(self.layers, cache):
            h, layer_cache = layer.forward_with_cache(
                h, start_pos=int(start_pos), cache=layer_cache, attention_mask=attention_mask, trace=trace, cos=cos, sin=sin
            )
            new_cache.append(layer_cache)
        return self.norm(h), new_cache


class MTPPredictor(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        cfg = config.finalize()
        inter = int(cfg.mtp_intermediate_size or cfg.hidden_size)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        act = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(act))


class MiniLLMForCausalLM(nn.Module):
    def __init__(self, config: Optional[MiniLLMConfig] = None):
        super().__init__()
        self.config = (config or MiniLLMConfig()).finalize()
        metal_ops.set_enabled(bool(self.config.use_metal_kernels))
        self.model = MiniLLMModel(self.config)
        self.mtp_layers = [MTPPredictor(self.config) for _ in range(int(self.config.num_nextn_predict_layers or 0))]

        if int(self.config.lora_r) > 0:
            self.freeze(recurse=True)
            targets = tuple(t.strip() for t in str(self.config.lora_targets).split(",") if t.strip())
            apply_lora(
                self,
                cfg=LoRAConfig(
                    r=int(self.config.lora_r),
                    alpha=float(self.config.lora_alpha),
                    dropout=float(self.config.lora_dropout),
                    target_modules=targets,
                ),
                verbose=False,
            )

    def _mtp_hidden(self, hidden: mx.array) -> List[mx.array]:
        mtp_hidden: List[mx.array] = []
        h = hidden
        for layer in self.mtp_layers:
            h = h + layer(h)
            mtp_hidden.append(h)
        return mtp_hidden

    def __call__(self, input_ids: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        h = self.model(input_ids, attention_mask=attention_mask)
        logits = h @ self.model.embed_tokens.weight.transpose()
        return logits

    def forward_with_mtp_hidden(
        self, input_ids: mx.array, *, attention_mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, List[mx.array]]:
        h = self.model(input_ids, attention_mask=attention_mask)
        return h, self._mtp_hidden(h)

    def forward_with_cache(
        self,
        input_ids: mx.array,
        *,
        start_pos: int,
        cache: List["LayerKVCache"],
        attention_mask: Optional[mx.array] = None,
        trace: Optional["ActivationTracer"] = None,
    ) -> Tuple[mx.array, List["LayerKVCache"]]:
        h, cache = self.model.forward_with_cache(
            input_ids, start_pos=int(start_pos), cache=cache, attention_mask=attention_mask, trace=trace
        )
        logits = h @ self.model.embed_tokens.weight.transpose()
        return logits, cache

    def allocate_kv_cache(self, *, batch_size: int, max_seq_len: int) -> List["LayerKVCache"]:
        dtype = self.model.embed_tokens.weight.dtype
        return allocate_kv_cache(self.config, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype)


@dataclass(frozen=True)
class LayerKVCache:
    k: mx.array
    v: mx.array


def allocate_kv_cache(
    config: MiniLLMConfig, *, batch_size: int, max_seq_len: int, dtype: mx.Dtype
) -> List[LayerKVCache]:
    cfg = config.finalize()
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")

    caches: List[LayerKVCache] = []
    for _ in range(cfg.num_hidden_layers):
        k = mx.zeros((int(batch_size), int(cfg.num_attention_heads), int(max_seq_len), int(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)), dtype=dtype)
        v = mx.zeros((int(batch_size), int(cfg.num_attention_heads), int(max_seq_len), int(cfg.v_head_dim)), dtype=dtype)
        caches.append(LayerKVCache(k=k, v=v))
    return caches


def count_parameters(params: Dict[str, Any]) -> int:
    def _count(obj: Any) -> int:
        if isinstance(obj, mx.array):
            return int(obj.size)
        if isinstance(obj, dict):
            return sum(_count(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_count(v) for v in obj)
        return 0

    return _count(params)


def parameters_bytes(params: Dict[str, Any]) -> int:
    def _bytes(obj: Any) -> int:
        if isinstance(obj, mx.array):
            return int(obj.size) * int(obj.dtype.size)
        if isinstance(obj, dict):
            return sum(_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_bytes(v) for v in obj)
        return 0

    return _bytes(params)
