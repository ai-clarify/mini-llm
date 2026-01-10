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
    def __init__(self, dim: int, eps: float = 1e-5):
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


class Attention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, *, start_pos: int = 0, attention_mask: Optional[mx.array] = None) -> mx.array:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=start_pos)
        k = self.rope(k, offset=start_pos)

        if attention_mask is None:
            mask: Optional[mx.array | str] = "causal"
        else:
            # Accept additive/boolean masks broadcastable to [B, N, T, T].
            # For strict alignment with the PyTorch training scripts, you can pass None.
            mask = attention_mask

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.n_heads * self.head_dim)
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
    ) -> Tuple[mx.array, "LayerKVCache"]:
        """
        KV-cached attention for inference.

        - Prefill: call with `start_pos=0` and `x` containing the whole prompt.
        - Decode: call with `x` being the last generated token (T=1) and `start_pos` equal to its absolute position.
        """

        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=int(start_pos))
        k = self.rope(k, offset=int(start_pos))
        if trace is not None and layer_id is not None:
            trace.record_qkv(layer_id=int(layer_id), start_pos=int(start_pos), q=q, k=k, v=v)

        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got: {start_pos}")
        if int(start_pos) + int(seq_len) > int(cache.k.shape[2]):
            raise ValueError(
                f"KV cache too small: need {int(start_pos) + int(seq_len)} "
                f"but cache_len={int(cache.k.shape[2])}"
            )

        k_cache = mx.slice_update(cache.k, k, start_indices=mx.array([int(start_pos)]), axes=(2,))
        v_cache = mx.slice_update(cache.v, v, start_indices=mx.array([int(start_pos)]), axes=(2,))

        total = int(start_pos) + int(seq_len)
        _, _, _, head_dim = k_cache.shape
        k_full = mx.slice(
            k_cache,
            start_indices=mx.array([0]),
            axes=(2,),
            slice_size=(int(bsz), int(self.n_kv_heads), int(total), int(head_dim)),
        )
        v_full = mx.slice(
            v_cache,
            start_indices=mx.array([0]),
            axes=(2,),
            slice_size=(int(bsz), int(self.n_kv_heads), int(total), int(head_dim)),
        )

        if attention_mask is None and int(start_pos) > 0:
            # For decode (T=1), the query token is the last position; attending to all cached keys is safe.
            mask: Optional[mx.array | str] = None
        else:
            mask = "causal" if attention_mask is None else attention_mask

        if trace is not None and trace.cfg is not None and bool(trace.cfg.record_attn):
            # Record attention weights (top-k keys per head) for interpretability.
            # This is an extra compute path and is only enabled when tracing.
            rep = int(self.n_heads // self.n_kv_heads)
            k_rep = mx.repeat(k_full, repeats=rep, axis=1)
            q_for_trace = q
            if not bool(trace.cfg.record_attn_all_queries):
                q_for_trace = q[:, :, -1:, :]
                query_positions = [int(start_pos) + int(seq_len) - 1]
            else:
                query_positions = list(range(int(start_pos), int(start_pos) + int(seq_len)))

            scores = mx.matmul(
                q_for_trace.astype(mx.float32),
                k_rep.astype(mx.float32).transpose(0, 1, 3, 2),
            )
            scores = scores * float(self.scale)
            if attention_mask is None and int(seq_len) > 1:
                # Causal mask for multi-token prefill: key_pos <= query_pos (absolute positions).
                key_pos = mx.arange(int(total))[None, :]
                qpos = mx.array([int(p) for p in query_positions], dtype=mx.int32)[:, None]
                allow = key_pos <= qpos
                neg_inf = mx.array(-1e9, dtype=scores.dtype)
                scores = mx.where(allow[None, None, :, :], scores, neg_inf)
            attn_w = mx.softmax(scores, axis=-1)
            if trace is not None and layer_id is not None:
                trace.record_attn(
                    layer_id=int(layer_id),
                    start_pos=int(start_pos),
                    attn=attn_w,
                    query_positions=query_positions,
                )

        out = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        return self.resid_dropout(out), LayerKVCache(k=k_cache, v=v_cache)


class FeedForward(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        config = config.finalize()
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            raise ValueError("intermediate_size is None after finalize()")
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if config.hidden_act != "silu":
            raise ValueError(f"Only silu is supported right now, got: {config.hidden_act}")

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


class MiniLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniLLMConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_attn_gate = bool(config.use_attn_gate)
        self.attn_gate_logit = (
            mx.array(float(config.attn_gate_init), dtype=mx.float32) if bool(config.use_attn_gate) else None
        )
        if config.use_moe:
            raise NotImplementedError("MoE is not implemented for the MLX path yet.")
        self.mlp = FeedForward(config)

    def __call__(self, x: mx.array, *, start_pos: int = 0, attention_mask: Optional[mx.array] = None) -> mx.array:
        h = self.self_attn(self.input_layernorm(x), start_pos=start_pos, attention_mask=attention_mask)
        if self.attn_gate_logit is not None:
            gate = mx.sigmoid(self.attn_gate_logit).astype(x.dtype)
            x = x + gate * h
        else:
            x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def forward_with_cache(
        self,
        x: mx.array,
        *,
        start_pos: int,
        cache: "LayerKVCache",
        attention_mask: Optional[mx.array] = None,
        trace: Optional["ActivationTracer"] = None,
    ) -> Tuple[mx.array, "LayerKVCache"]:
        if trace is not None:
            trace.record_hidden(layer_id=int(self.layer_id), name="x_in_rms", start_pos=int(start_pos), x=x)

        x_attn_in = self.input_layernorm(x)
        if trace is not None:
            trace.record_hidden(
                layer_id=int(self.layer_id), name="attn_in_rms", start_pos=int(start_pos), x=x_attn_in
            )

        h, cache = self.self_attn.forward_with_cache(
            x_attn_in,
            start_pos=int(start_pos),
            cache=cache,
            attention_mask=attention_mask,
            trace=trace,
            layer_id=int(self.layer_id),
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


class MiniLLMModel(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config.finalize()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.layers = [MiniLLMBlock(i, self.config) for i in range(self.config.num_hidden_layers)]
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.checkpoint_every_n: int = 0

    def __call__(self, input_ids: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        # input_ids: [B, T]
        h = self.embed_tokens(input_ids)
        h = self.dropout(h)
        start_pos = 0
        ckpt_n = int(self.checkpoint_every_n)
        use_ckpt = self.training and ckpt_n > 0
        for i, layer in enumerate(self.layers):
            if use_ckpt and (i % ckpt_n == 0):
                h = mx.checkpoint(lambda x, m: layer(x, start_pos=start_pos, attention_mask=m))(h, attention_mask)
            else:
                h = layer(h, start_pos=start_pos, attention_mask=attention_mask)
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
        for layer, layer_cache in zip(self.layers, cache):
            h, layer_cache = layer.forward_with_cache(
                h, start_pos=int(start_pos), cache=layer_cache, attention_mask=attention_mask, trace=trace
            )
            new_cache.append(layer_cache)
        return self.norm(h), new_cache


class MiniLLMForCausalLM(nn.Module):
    def __init__(self, config: Optional[MiniLLMConfig] = None):
        super().__init__()
        self.config = (config or MiniLLMConfig()).finalize()
        metal_ops.set_enabled(bool(self.config.use_metal_kernels))
        self.model = MiniLLMModel(self.config)

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

    def __call__(self, input_ids: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        h = self.model(input_ids, attention_mask=attention_mask)  # [B, T, H]
        # Weight tying: lm_head.weight == embed_tokens.weight
        logits = h @ self.model.embed_tokens.weight.transpose()  # [B, T, V]
        return logits

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
    # [B, n_kv_heads, max_seq_len, head_dim]
    k: mx.array
    v: mx.array


def allocate_kv_cache(
    config: MiniLLMConfig, *, batch_size: int, max_seq_len: int, dtype: mx.Dtype
) -> List[LayerKVCache]:
    cfg = config.finalize()
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads if cfg.num_key_value_heads is not None else cfg.num_attention_heads
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")

    caches: List[LayerKVCache] = []
    for _ in range(cfg.num_hidden_layers):
        k = mx.zeros((int(batch_size), int(n_kv), int(max_seq_len), int(head_dim)), dtype=dtype)
        v = mx.zeros((int(batch_size), int(n_kv), int(max_seq_len), int(head_dim)), dtype=dtype)
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
