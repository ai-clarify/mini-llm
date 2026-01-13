#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as mlx_cache
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention

from speculator.infer.mlx.common import (
    SpecStats,
    _project_logits_qwen3,
    _qwen3_forward_hidden_states,
    sample_next_token,
)


@dataclass(frozen=True)
class Eagle3Config:
    vocab_size: int
    draft_vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Eagle3Config":
        return cls(
            vocab_size=int(data.get("vocab_size", 32000)),
            draft_vocab_size=int(data.get("draft_vocab_size", data.get("vocab_size", 32000))),
            hidden_size=int(data.get("hidden_size", 4096)),
            intermediate_size=int(data.get("intermediate_size", 11008)),
            num_attention_heads=int(data.get("num_attention_heads", 32)),
            num_key_value_heads=int(data.get("num_key_value_heads", data.get("num_attention_heads", 32))),
            head_dim=int(data.get("head_dim", 0)) or 0,
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            rope_scaling=data.get("rope_scaling"),
        )


def _resolve_head_dim(cfg: Eagle3Config) -> int:
    if cfg.head_dim > 0:
        return int(cfg.head_dim)
    return int(cfg.hidden_size // cfg.num_attention_heads)


def _apply_rope_with_positions(
    x: mx.array,
    positions: mx.array,
    *,
    base: float,
    scale: float,
) -> mx.array:
    """Apply RoPE with explicit positions. Time O(TD) best/avg/worst, space O(TD)."""
    _, _, seq_len, head_dim = x.shape
    half = int(head_dim // 2)
    inv_freq = 1.0 / (float(base) ** (mx.arange(0, half) / float(half)))
    angles = positions.reshape(seq_len, 1) * float(scale) * inv_freq.reshape(1, half)
    cos = mx.cos(angles)[None, None, :, :]
    sin = mx.sin(angles)[None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return mx.concatenate([out1, out2], axis=-1)


class Eagle3Attention(nn.Module):
    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        head_dim = _resolve_head_dim(cfg)
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_key_value_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(head_dim)
        self.scale = float(head_dim) ** -0.5
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * head_dim, self.hidden_size, bias=False)

        rope_scale = 1.0
        if cfg.rope_scaling:
            if str(cfg.rope_scaling.get("type", "default")) == "linear":
                rope_scale = 1.0 / float(cfg.rope_scaling.get("factor", 1.0))
            else:
                raise NotImplementedError("Unsupported rope_scaling for Eagle3 MLX drafter.")
        self.rope_base = float(cfg.rope_theta)
        self.rope_scale = float(rope_scale)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[mlx_cache.KVCache] = None,
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = int(cache.offset) if cache is not None else 0
        if positions is None:
            positions = mx.arange(offset, offset + int(seq_len))
        q = _apply_rope_with_positions(q, positions, base=self.rope_base, scale=self.rope_scale)
        k = _apply_rope_with_positions(k, positions, base=self.rope_base, scale=self.rope_scale)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Eagle3MLP(nn.Module):
    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        self.self_attn = Eagle3Attention(cfg)
        self.mlp = Eagle3MLP(cfg)
        self.hidden_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        *,
        input_emb: mx.array,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array],
        cache: Optional[mlx_cache.KVCache],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = mx.concatenate([input_emb, hidden_states], axis=-1)
        hidden_states = residual + self.self_attn(hidden_states, mask=attention_mask, cache=cache)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Eagle3Drafter(nn.Module):
    def __init__(
        self,
        *,
        cfg: Eagle3Config,
        total_tokens: int,
        depth: int,
        top_k: int,
        threshold: float,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(cfg.hidden_size)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.draft_vocab_size, bias=False)
        self.fc = nn.Linear(cfg.hidden_size * 3, cfg.hidden_size, bias=False)
        self.midlayer = Eagle3DecoderLayer(cfg)
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.d2t = mx.zeros((cfg.draft_vocab_size,), dtype=mx.int64)
        self.t2d = mx.zeros((cfg.vocab_size,), dtype=mx.bool_)
        self.top_k = int(top_k)
        self.total_tokens = int(total_tokens) - 1
        self.depth = int(depth)
        self.threshold = float(math.log(max(threshold, 1e-9)))
        self.tree_mask = None
        self.tree_mask_init = None
        self.position_ids = None
        self.initial_position_id = 0
        self.stable_kv: Optional[mlx_cache.KVCache] = None

    def init_tree(self) -> None:
        self.tree_mask_init = mx.eye(self.top_k).astype(mx.bool_)[None, None]
        self.position_ids = mx.zeros((self.top_k,), dtype=mx.int32)

    def reset(self) -> None:
        self.tree_mask = None

    def reset_kv(self) -> None:
        self.stable_kv = None

    def _prepare_decoder_attention_mask(self, seq_len: int, past_len: int) -> Optional[mx.array]:
        if seq_len <= 1:
            return None
        mask = create_causal_mask(seq_len, offset=int(past_len))
        if self.tree_mask is not None:
            tree = self.tree_mask
            if int(tree.ndim) == 4:
                tree = tree[0, 0]
            tree_rows, tree_cols = int(tree.shape[0]), int(tree.shape[1])
            if tree_rows <= int(mask.shape[0]) and tree_cols <= int(mask.shape[1]):
                tail_rows = mask[-tree_rows:, :]
                tail_left = tail_rows[:, : int(mask.shape[1]) - tree_cols]
                tail_right = tail_rows[:, -tree_cols:] & tree
                tail_rows = mx.concatenate([tail_left, tail_right], axis=1)
                head_rows = mask[: int(mask.shape[0]) - tree_rows, :]
                mask = tail_rows if head_rows.size == 0 else mx.concatenate([head_rows, tail_rows], axis=0)
        return mask

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        input_ids: mx.array,
        past_key_values: Optional[mlx_cache.KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[mlx_cache.KVCache], Optional[mx.array]]:
        if int(hidden_states.shape[-1]) != int(self.hidden_size):
            hidden_states = self.fc(hidden_states)
        input_embeds = self.embed_tokens(input_ids)
        past_len = int(past_key_values.offset) if past_key_values is not None else 0
        mask = self._prepare_decoder_attention_mask(int(input_ids.shape[1]), past_len)
        hidden_states = self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            attention_mask=mask,
            cache=past_key_values if use_cache else None,
        )
        return hidden_states, past_key_values, None

    def _get_topk_tokens(self, hidden: mx.array) -> Tuple[mx.array, mx.array]:
        logits = self.lm_head(self.norm(hidden))
        log_probs = nn.log_softmax(logits, axis=-1)
        if self.threshold < 0.0:
            cutoff = mx.array(self.threshold, dtype=log_probs.dtype)
            neg_inf = mx.array(-1e9, dtype=log_probs.dtype)
            log_probs = mx.where(log_probs < cutoff, neg_inf, log_probs)
        order = mx.argsort(-log_probs, axis=-1)
        topk_idx = order[:, : self.top_k]
        topk_vals = mx.take_along_axis(log_probs, topk_idx, axis=-1)
        return topk_idx, topk_vals[0]

    def _get_initial_hidden(
        self, hidden_states: mx.array, input_ids: mx.array
    ) -> Tuple[mx.array, mlx_cache.KVCache, Optional[mx.array]]:
        if self.stable_kv is not None:
            kv_len = int(self.stable_kv.offset)
            input_ids = input_ids[:, kv_len:]
            if kv_len > 0:
                hidden_states = hidden_states[:, kv_len:]
            outputs = self(hidden_states, input_ids=input_ids, past_key_values=self.stable_kv, use_cache=True)
        else:
            outputs = self(hidden_states, input_ids=input_ids, past_key_values=None, use_cache=True)
        out_hidden, past_key_values, early_stop_signal = outputs
        return out_hidden[:, -1], past_key_values, early_stop_signal

    def _process_tree_level(
        self,
        level: int,
        tree_mask: mx.array,
        input_hidden: mx.array,
        input_ids: mx.array,
        scores: mx.array,
        topk_cs_index: mx.array,
        scores_list: List[mx.array],
        parents_list: List[mx.array],
        ss_token: List[mx.array],
        past_key_values: mlx_cache.KVCache,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mlx_cache.KVCache]:
        self.tree_mask = tree_mask
        position_ids = self.position_ids + int(self.initial_position_id)
        out_hidden, past_key_values, _ = self(
            input_hidden,
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        self.initial_position_id += 1

        bias1 = self.top_k if level > 0 else 0
        bias2 = max(0, level - 1)
        bias = 1 + (self.top_k ** 2) * bias2 + bias1
        parents = topk_cs_index + int(bias)
        parents_list.append(parents)

        topk_index, topk_p = self._get_topk_tokens(out_hidden[0])
        cu_scores = topk_p + scores[:, None]
        flat = cu_scores.reshape(-1)
        order = mx.argsort(-flat)
        topk_cs_index = order[: self.top_k]
        scores = mx.take_along_axis(flat, topk_cs_index, axis=0)

        out_ids = topk_cs_index // self.top_k
        input_hidden = out_hidden[:, out_ids]
        input_ids = mx.take(topk_index.reshape(-1), topk_cs_index)[None]
        if int(self.cfg.vocab_size) == int(self.cfg.draft_vocab_size):
            ss_token.append(topk_index)
        else:
            input_ids = input_ids + mx.take(self.d2t, input_ids)
            ss_token.append(topk_index + mx.take(self.d2t, topk_index))
        scores_list.append(cu_scores)
        tree_mask = mx.concatenate([tree_mask[:, :, out_ids], self.tree_mask_init], axis=3)
        return tree_mask, input_hidden, input_ids, scores, topk_cs_index, past_key_values

    def _build_tree_mask(
        self, top_indices: mx.array, parents_list: Sequence[mx.array]
    ) -> Tuple[mx.array, mx.array]:
        top_indices_list = [int(x) for x in top_indices.tolist()]
        parents = mx.concatenate(list(parents_list), axis=0)
        parent_tokens = parents[top_indices // int(self.top_k)].tolist()
        mask_index_list: List[int] = []
        for parent in parent_tokens:
            if parent == 0:
                mask_index_list.append(-1)
                continue
            target = int(parent) - 1
            idx = 0
            while idx < len(top_indices_list) and top_indices_list[idx] < target:
                idx += 1
            mask_index_list.append(idx)
        mask_index_list = [idx + 1 for idx in mask_index_list]

        tree_mask = [[False] * (self.total_tokens + 1) for _ in range(self.total_tokens + 1)]
        for i in range(self.total_tokens + 1):
            tree_mask[i][0] = True
        for i in range(self.total_tokens):
            parent = mask_index_list[i]
            if parent >= 0:
                tree_mask[i + 1] = [
                    tree_mask[i + 1][j] or tree_mask[parent][j]
                    for j in range(self.total_tokens + 1)
                ]
            tree_mask[i + 1][i + 1] = True

        tree_position_ids = [sum(row) - 1 for row in tree_mask]
        return (
            mx.array(tree_mask, dtype=mx.bool_)[None, None],
            mx.array(tree_position_ids, dtype=mx.int32),
        )

    def _generate_retrieve_indices(
        self, tree_position_ids: mx.array, top_indices: mx.array, parents_list: Sequence[mx.array]
    ) -> mx.array:
        top_indices_list = [int(x) for x in top_indices.tolist()]
        parents = mx.concatenate(list(parents_list), axis=0)
        parent_tokens = parents[top_indices // int(self.top_k)].tolist()
        mask_index_list: List[int] = []
        for parent in parent_tokens:
            if parent == 0:
                mask_index_list.append(-1)
                continue
            target = int(parent) - 1
            idx = 0
            while idx < len(top_indices_list) and top_indices_list[idx] < target:
                idx += 1
            mask_index_list.append(idx)
        mask_index_list = [idx + 1 for idx in mask_index_list]

        noleaf = sorted(set(mask_index_list))
        leaf_num = self.total_tokens - (len(noleaf) - 1)
        max_depth = int(mx.max(tree_position_ids).item()) + 1
        retrieve_indices = [[-1] * max_depth for _ in range(leaf_num)]
        position_ids_list = tree_position_ids.tolist()
        rid = 0
        for i in range(self.total_tokens + 1):
            if i not in noleaf:
                cid = i
                depth = int(position_ids_list[i])
                for j in range(depth, -1, -1):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1
        return mx.array(retrieve_indices, dtype=mx.int32)

    def _apply_logits_processor(self, retrieve_indices: mx.array) -> mx.array:
        maxitem = int(self.total_tokens) + 5

        def custom_sort(row: List[int]) -> List[int]:
            return [x if x >= 0 else maxitem for x in row]

        rows = sorted(retrieve_indices.tolist(), key=custom_sort)
        return mx.array(rows, dtype=mx.int32)

    def topk_generate(
        self,
        *,
        hidden_states: mx.array,
        input_ids: mx.array,
        apply_logits_processor: bool = False,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array]]:
        scores_list: List[mx.array] = []
        parents_list: List[mx.array] = []
        ss_token: List[mx.array] = []

        input_ids = input_ids.astype(mx.int32)
        sample_token = input_ids[:, -1]
        input_ids = input_ids[:, 1:]
        self.initial_position_id = int(input_ids.shape[1])
        self.reset()

        last_hidden, past_key_values, early_stop_signal = self._get_initial_hidden(hidden_states, input_ids)
        self.stable_kv = past_key_values

        topk_index, scores = self._get_topk_tokens(last_hidden)
        scores_list.append(scores[None])
        parents_list.append(mx.zeros((1,), dtype=mx.int32))

        if int(self.cfg.vocab_size) == int(self.cfg.draft_vocab_size):
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            mapped_tokens = topk_index + mx.take(self.d2t, topk_index)
            ss_token.append(mapped_tokens)
            input_ids = mapped_tokens

        input_hidden = mx.repeat(last_hidden[None], self.top_k, axis=1)
        tree_mask = self.tree_mask_init
        topk_cs_index = mx.arange(self.top_k)

        for i in range(int(self.depth)):
            (
                tree_mask,
                input_hidden,
                input_ids,
                scores,
                topk_cs_index,
                past_key_values,
            ) = self._process_tree_level(
                i,
                tree_mask,
                input_hidden,
                input_ids,
                scores,
                topk_cs_index,
                scores_list,
                parents_list,
                ss_token,
                past_key_values,
            )

        all_scores = mx.concatenate(scores_list, axis=0).reshape(-1)
        order = mx.argsort(-all_scores)
        top_indices = mx.sort(order[: self.total_tokens])
        all_tokens = mx.concatenate(ss_token, axis=0).reshape(-1)
        draft_tokens = mx.take_along_axis(all_tokens, top_indices, axis=0)
        draft_tokens = mx.concatenate([sample_token, draft_tokens], axis=0)

        tree_mask, tree_position_ids = self._build_tree_mask(top_indices, parents_list)
        retrieve_indices = self._generate_retrieve_indices(tree_position_ids, top_indices, parents_list)
        if apply_logits_processor:
            retrieve_indices = self._apply_logits_processor(retrieve_indices)
        return (
            draft_tokens[None],
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            early_stop_signal,
        )


def _load_eagle3_config(path: Path) -> Eagle3Config:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Eagle3Config.from_dict(data)


def _load_eagle3_state_dict(path: Path) -> Dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported eagle3 state dict: {path}")
    return state


def load_eagle3_drafter(
    *,
    eagle3_dir: Path,
    target,
    total_tokens: int,
    depth: int,
    top_k: int,
    threshold: float,
) -> Eagle3Drafter:
    cfg_path = eagle3_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing eagle3 config.json: {cfg_path}")
    cfg = _load_eagle3_config(cfg_path)
    drafter = Eagle3Drafter(
        cfg=cfg,
        total_tokens=total_tokens,
        depth=depth,
        top_k=top_k,
        threshold=threshold,
    )
    drafter.init_tree()

    emb_weight = target.model.embed_tokens.weight
    drafter.embed_tokens.weight = emb_weight

    state_path = eagle3_dir / "pytorch_model.bin"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing eagle3 weights: {state_path}")
    state = _load_eagle3_state_dict(state_path)

    def to_mx(tensor, *, dtype=None):
        arr = tensor.detach().cpu().numpy()
        mx_arr = mx.array(arr)
        if dtype is not None:
            mx_arr = mx_arr.astype(dtype)
        return mx_arr

    dtype = emb_weight.dtype
    if "d2t" in state:
        drafter.d2t = to_mx(state["d2t"], dtype=mx.int64)
    if "t2d" in state:
        drafter.t2d = to_mx(state["t2d"], dtype=mx.bool_)

    weight_map = {
        "midlayer.self_attn.q_proj.weight": drafter.midlayer.self_attn.q_proj,
        "midlayer.self_attn.k_proj.weight": drafter.midlayer.self_attn.k_proj,
        "midlayer.self_attn.v_proj.weight": drafter.midlayer.self_attn.v_proj,
        "midlayer.self_attn.o_proj.weight": drafter.midlayer.self_attn.o_proj,
        "midlayer.mlp.gate_proj.weight": drafter.midlayer.mlp.gate_proj,
        "midlayer.mlp.up_proj.weight": drafter.midlayer.mlp.up_proj,
        "midlayer.mlp.down_proj.weight": drafter.midlayer.mlp.down_proj,
        "midlayer.hidden_norm.weight": drafter.midlayer.hidden_norm,
        "midlayer.input_layernorm.weight": drafter.midlayer.input_layernorm,
        "midlayer.post_attention_layernorm.weight": drafter.midlayer.post_attention_layernorm,
        "norm.weight": drafter.norm,
        "fc.weight": drafter.fc,
        "lm_head.weight": drafter.lm_head,
    }
    for key, module in weight_map.items():
        if key not in state:
            continue
        value = state[key]
        if hasattr(module, "weight"):
            module.weight = to_mx(value, dtype=dtype)
        else:
            module.weight = to_mx(value, dtype=dtype)

    return drafter


def _resolve_eagle3_layers(num_layers: int) -> List[int]:
    if num_layers <= 0:
        return []
    layers = [2, num_layers // 2, max(num_layers - 3, 0)]
    selected: List[int] = []
    for idx in layers:
        if idx not in selected and 0 <= idx < num_layers:
            selected.append(idx)
    return selected


def _build_tree_attention_mask(
    *,
    seq_len: int,
    offset: int,
    tree_mask: Optional[mx.array],
) -> Optional[mx.array]:
    if seq_len <= 1:
        return None
    mask = create_causal_mask(int(seq_len), offset=int(offset))
    if tree_mask is not None:
        tail = mask[:, int(offset) :]
        mask = mx.concatenate([mask[:, :int(offset)], tail & tree_mask], axis=1)
    return mask


def _qwen3_forward_hidden_states_tree(
    target,
    input_ids: mx.array,
    *,
    cache: Optional[List[Any]],
    position_ids: mx.array,
    tree_mask: Optional[mx.array],
    layer_ids: Sequence[int],
) -> Tuple[mx.array, List[mx.array]]:
    model = target.model
    h = model.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(model.layers)
    offset = int(cache[0].offset) if cache and cache[0] is not None else 0
    mask = _build_tree_attention_mask(
        seq_len=int(h.shape[1]),
        offset=offset,
        tree_mask=tree_mask,
    )
    layer_index = {int(layer_id): idx for idx, layer_id in enumerate(layer_ids)}
    hiddens: List[Optional[mx.array]] = [None] * len(layer_ids)
    for idx, (layer, c) in enumerate(zip(model.layers, cache)):
        x = layer.input_layernorm(h)
        h = h + _qwen3_attention_with_positions(layer.self_attn, x, mask=mask, cache=c, positions=position_ids)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        if idx in layer_index:
            hiddens[layer_index[idx]] = h
    h = model.norm(h)
    return h, [hidden for hidden in hiddens if hidden is not None]


def _qwen3_attention_with_positions(
    attn,
    x: mx.array,
    *,
    mask: Optional[mx.array],
    cache: Optional[Any],
    positions: mx.array,
) -> mx.array:
    bsz, seq_len, _ = x.shape
    q = attn.q_proj(x).reshape(bsz, seq_len, attn.n_heads, -1)
    k = attn.k_proj(x).reshape(bsz, seq_len, attn.n_kv_heads, -1)
    v = attn.v_proj(x).reshape(bsz, seq_len, attn.n_kv_heads, -1)
    q = attn.q_norm(q).transpose(0, 2, 1, 3)
    k = attn.k_norm(k).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    if not isinstance(attn.rope, nn.RoPE):
        raise NotImplementedError("Tree decoding only supports nn.RoPE for Qwen3 in MLX.")
    q = _apply_rope_with_positions(q, positions, base=attn.rope.base, scale=attn.rope.scale)
    k = _apply_rope_with_positions(k, positions, base=attn.rope.base, scale=attn.rope.scale)
    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    out = scaled_dot_product_attention(q, k, v, cache=cache, scale=attn.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
    return attn.o_proj(out)


def _topk_logits_to_distribution(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    """Return top-k token ids + probs. Time O(k log k) best/avg/worst, space O(k)."""
    if temperature > 0:
        logits = logits / float(temperature)
    logits = logits.astype(mx.float32)
    vocab = int(logits.shape[-1])
    k = int(top_k)
    if k <= 0 or k > vocab:
        k = vocab
    if k <= 0:
        return [0], [1.0]

    indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    values = mx.take_along_axis(logits, indices, axis=-1)
    order = mx.argsort(-values, axis=-1)
    values = mx.take_along_axis(values, order, axis=-1)
    indices = mx.take_along_axis(indices, order, axis=-1)
    probs = mx.softmax(values, axis=-1)
    if top_p < 1.0:
        cum = mx.cumsum(probs, axis=-1)
        keep = mx.concatenate(
            [mx.array([True], dtype=mx.bool_), cum[:-1] <= float(top_p)],
            axis=-1,
        )
        zeros = mx.zeros_like(probs)
        probs = mx.where(keep, probs, zeros)
        total = float(mx.sum(probs).item())
        if total > 0.0:
            probs = probs / total
        else:
            head = mx.array([1.0], dtype=probs.dtype)
            tail = mx.zeros((k - 1,), dtype=probs.dtype) if k > 1 else mx.array([], dtype=probs.dtype)
            probs = mx.concatenate([head, tail], axis=-1)
    return indices.astype(mx.int32).tolist(), probs.astype(mx.float32).tolist()


def _sample_from_sparse(top_idx: List[int], probs: List[float], rng: random.Random) -> int:
    if not top_idx:
        return 0
    r = rng.random()
    cum = 0.0
    for tok, p in zip(top_idx, probs):
        cum += p
        if r <= cum:
            return int(tok)
    return int(top_idx[-1])


def _evaluate_posterior(
    *,
    logits: mx.array,
    candidates: mx.array,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: random.Random,
) -> Tuple[int, int, int]:
    candidates_np = candidates.astype(mx.int32).tolist()
    num_cand = len(candidates_np)
    cand_len = len(candidates_np[0]) if candidates_np else 0
    if cand_len == 0:
        return 0, 0, 0
    if temperature <= 0:
        argmax = mx.argmax(logits[:, :-1, :], axis=-1).astype(mx.int32).tolist()
        accept_lengths = []
        for row in range(num_cand):
            ok = 0
            for j in range(1, cand_len):
                target_token = int(argmax[row][j - 1])
                if candidates_np[row][j] == target_token:
                    ok += 1
                else:
                    break
            accept_lengths.append(ok)
        accept_length = max(accept_lengths)
        best_candidate = accept_lengths.index(accept_length) if accept_length > 0 else 0
        sample_token = int(mx.argmax(logits[best_candidate, accept_length, :]).item())
        return best_candidate, accept_length, sample_token

    if int(top_k) <= 0:
        top_k = min(128, int(logits.shape[-1]))

    accept_length = 1
    accept_cand = candidates_np[0][:1]
    best_candidate = 0
    adjustflag = False
    last_top_idx: Optional[List[int]] = None
    last_probs: Optional[List[float]] = None
    for i in range(1, cand_len):
        if i != accept_length:
            break
        is_eq = [row[:accept_length] == accept_cand for row in candidates_np]
        if True not in is_eq:
            break
        fi = is_eq.index(True)
        top_idx, probs = _topk_logits_to_distribution(
            logits[fi, i - 1, :],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        idx_map = {tok: pos for pos, tok in enumerate(top_idx)}
        last_top_idx = top_idx
        last_probs = probs
        candidates_set: set[int] = set()
        for j in range(num_cand):
            if not is_eq[j]:
                continue
            xi = int(candidates_np[j][i])
            if xi in candidates_set or xi == -1:
                continue
            candidates_set.add(xi)
            r = rng.random()
            pos = idx_map.get(xi)
            px = probs[pos] if pos is not None else 0.0
            if r <= px:
                accept_cand = accept_cand + [xi]
                accept_length += 1
                best_candidate = j
                break
            adjustflag = True
            if pos is not None:
                probs[pos] = 0.0
                total = sum(probs)
                if total > 0.0:
                    probs = [p / total for p in probs]
                last_probs = probs
    if adjustflag and accept_length != cand_len and last_top_idx and last_probs is not None:
        sample_token = _sample_from_sparse(last_top_idx, last_probs, rng)
    else:
        top_idx, probs = _topk_logits_to_distribution(
            logits[best_candidate, accept_length - 1, :],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        sample_token = _sample_from_sparse(top_idx, probs, rng)
    return best_candidate, accept_length - 1, sample_token


def eagle3_decode_with_stats(
    *,
    target,
    drafter: Eagle3Drafter,
    input_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: Optional[int],
    seed: int,
) -> Tuple[List[int], SpecStats]:
    output_ids = list(input_ids)
    produced = 0
    steps = 0
    total_accept = 0
    total_draft = 0
    accepted_output = 0
    zero_accept = 0
    target_generated = 0
    spec_time_s = 0.0
    target_time_s = 0.0
    target_prefill_time_s = 0.0
    target_verify_time_s = 0.0
    target_generate_time_s = 0.0
    target_prefill_calls = 0
    target_verify_calls = 0
    target_generate_calls = 0

    cache = mlx_cache.make_prompt_cache(target.model)
    prompt = mx.array([output_ids], dtype=mx.int32)
    layer_ids = _resolve_eagle3_layers(len(target.model.layers))
    drafter.reset_kv()
    t0 = time.perf_counter()
    hidden, layer_hiddens = _qwen3_forward_hidden_states(
        target,
        prompt,
        cache=cache,
        layer_ids=layer_ids,
    )
    mx.eval(hidden)
    prefill_s = time.perf_counter() - t0
    target_time_s += prefill_s
    target_prefill_time_s += prefill_s
    target_prefill_calls += 1

    logits = _project_logits_qwen3(target, hidden)[:, -1, :]
    mx.eval(logits)
    token = sample_next_token(logits, temperature=temperature, top_p=top_p)
    target_generated += 1
    draft_input = mx.array([output_ids + [int(token)]], dtype=mx.int32)
    hidden_concat = mx.concatenate(layer_hiddens, axis=-1)
    full_hidden_concat = hidden_concat

    t0 = time.perf_counter()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = drafter.topk_generate(
        hidden_states=full_hidden_concat,
        input_ids=draft_input,
        apply_logits_processor=temperature > 0.0,
    )
    mx.eval(draft_tokens)
    spec_time_s += time.perf_counter() - t0

    rng = random.Random(int(seed))

    while produced < int(max_new_tokens):
        tree_len = int(draft_tokens.shape[1])
        position_ids = tree_position_ids + int(len(output_ids))
        t0 = time.perf_counter()
        tree_hidden, tree_layers = _qwen3_forward_hidden_states_tree(
            target,
            draft_tokens,
            cache=cache,
            position_ids=position_ids,
            tree_mask=tree_mask[0, 0] if tree_mask is not None else None,
            layer_ids=layer_ids,
        )
        tree_logits = _project_logits_qwen3(target, tree_hidden)
        mx.eval(tree_logits)
        verify_s = time.perf_counter() - t0
        target_time_s += verify_s
        target_verify_time_s += verify_s
        target_verify_calls += 1

        safe_indices = mx.where(retrieve_indices < 0, tree_len - 1, retrieve_indices)
        draft_exp = draft_tokens[:, None, :]
        cand_indices = safe_indices[None, :, :]
        candidates = mx.take_along_axis(draft_exp, cand_indices, axis=2)[0]

        logits_exp = tree_logits[:, None, :, :]
        logit_indices = safe_indices[None, :, :, None]
        logits_candidates = mx.take_along_axis(logits_exp, logit_indices, axis=2)[0]

        hidden_concat = mx.concatenate(tree_layers, axis=-1)
        hidden_exp = hidden_concat[:, None, :, :]
        hidden_indices = safe_indices[None, :, :, None]
        retrieved_hidden = mx.take_along_axis(hidden_exp, hidden_indices, axis=2)

        best_candidate, accept_length, sample_token = _evaluate_posterior(
            logits=logits_candidates,
            candidates=candidates,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            rng=rng,
        )

        steps += 1
        total_draft += max(tree_len - 1, 0)
        total_accept += int(accept_length)
        if int(accept_length) == 0:
            zero_accept += 1

        new_tokens = candidates[best_candidate, : int(accept_length) + 1].tolist()
        append_hidden = retrieved_hidden[:, best_candidate, : int(accept_length) + 1]
        output_ids.extend(int(t) for t in new_tokens)
        produced += len(new_tokens)
        accepted_output += len(new_tokens)

        eos_hit = False
        if eos_token_id is not None and int(eos_token_id) in new_tokens:
            eos_idx = new_tokens.index(int(eos_token_id))
            output_ids = output_ids[: -len(new_tokens) + eos_idx + 1]
            produced -= len(new_tokens) - (eos_idx + 1)
            accepted_output -= len(new_tokens) - (eos_idx + 1)
            new_tokens = new_tokens[: eos_idx + 1]
            append_hidden = append_hidden[:, : eos_idx + 1]
            eos_hit = True

        full_hidden_concat = mx.concatenate([full_hidden_concat, append_hidden], axis=1)

        if not mlx_cache.can_trim_prompt_cache(cache):
            cache = mlx_cache.make_prompt_cache(target.model)
            prompt = mx.array([output_ids], dtype=mx.int32)
            t0 = time.perf_counter()
            hidden, layer_hiddens = _qwen3_forward_hidden_states(
                target,
                prompt,
                cache=cache,
                layer_ids=layer_ids,
            )
            mx.eval(hidden)
            gen_s = time.perf_counter() - t0
            target_time_s += gen_s
            target_generate_time_s += gen_s
            target_generate_calls += 1
            full_hidden_concat = mx.concatenate(layer_hiddens, axis=-1)
        else:
            if tree_len > 0:
                mlx_cache.trim_prompt_cache(cache, tree_len)
            step = mx.array([new_tokens], dtype=mx.int32)
            t0 = time.perf_counter()
            hidden, layer_hiddens = _qwen3_forward_hidden_states(
                target,
                step,
                cache=cache,
                layer_ids=layer_ids,
            )
            mx.eval(hidden)
            gen_s = time.perf_counter() - t0
            target_time_s += gen_s
            target_generate_time_s += gen_s
            target_generate_calls += 1

        if eos_hit or produced >= int(max_new_tokens):
            break

        draft_input = mx.array([output_ids + [int(sample_token)]], dtype=mx.int32)
        t0 = time.perf_counter()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = drafter.topk_generate(
            hidden_states=full_hidden_concat,
            input_ids=draft_input,
            apply_logits_processor=temperature > 0.0,
        )
        mx.eval(draft_tokens)
        spec_time_s += time.perf_counter() - t0
        target_generated += 1

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
