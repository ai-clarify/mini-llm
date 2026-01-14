"""MiniLLM model upgraded to DeepSeek-V3.2-style architecture.

This implementation focuses on MLA (multi-head latent attention), MoE routing,
YaRN-style RoPE scaling, and optional MTP (multi-token prediction) heads while
keeping a small-parameter-friendly default config.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniLLMConfig(PretrainedConfig):
    model_type = "minillm"

    def __init__(
        self,
        *,
        vocab_size: int = 6400,
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 32768,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        inference_rope_scaling: bool = False,
        # MLA
        q_lora_rank: int = 256,
        kv_lora_rank: int = 128,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        # MoE
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: Optional[int] = None,
        n_shared_experts: int = 0,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        topk_method: str = "noaux_tc",
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.001,
        seq_aux: bool = True,
        moe_layer_freq: int = 1,
        first_k_dense_replace: int = 0,
        # MTP
        num_nextn_predict_layers: int = 1,
        mtp_intermediate_size: Optional[int] = None,
        mtp_loss_weight: float = 0.1,
        # DSA / indexer
        index_n_heads: int = 0,
        index_head_dim: int = 32,
        index_topk: int = 0,
        # misc
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.inference_rope_scaling = inference_rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_intermediate_size = mtp_intermediate_size
        self.mtp_loss_weight = mtp_loss_weight

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.inference_rope_scaling and self.rope_scaling is None:
            self.rope_scaling = {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 0.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            }

        if self.intermediate_size is None:
            intermediate = int(self.hidden_size * 8 / 3)
            self.intermediate_size = 64 * ((intermediate + 63) // 64)
        if self.moe_intermediate_size is None:
            moe_intermediate = max(1, int(self.hidden_size // 4))
            self.moe_intermediate_size = 64 * ((moe_intermediate + 63) // 64)
        if self.mtp_intermediate_size is None:
            mtp_intermediate = max(1, int(self.hidden_size * 2))
            self.mtp_intermediate_size = 64 * ((mtp_intermediate + 63) // 64)

        if self.n_routed_experts is None and self.use_moe:
            self.n_routed_experts = 4
        if not self.use_moe:
            self.n_routed_experts = None
            self.n_shared_experts = 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).type_as(x)


def yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def yarn_find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int) -> Tuple[int, int]:
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_idx: int, max_idx: int, dim: int) -> torch.Tensor:
    if min_idx == max_idx:
        max_idx += 1
    linear = (torch.arange(dim, dtype=torch.float32) - min_idx) / (max_idx - min_idx)
    return torch.clamp(linear, 0, 1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_rope_freqs(
    *, dim: int, end: int, base: float, rope_scaling: Optional[dict]
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    mscale = 1.0
    if rope_scaling is not None:
        scaling_type = rope_scaling.get("type", "")
        factor = rope_scaling.get("factor", 1.0)
        if scaling_type == "linear":
            inv_freq = inv_freq / factor
        elif scaling_type == "yarn":
            orig_max = rope_scaling.get("original_max_position_embeddings", end)
            beta_fast = rope_scaling.get("beta_fast", 32)
            beta_slow = rope_scaling.get("beta_slow", 1)
            if end > orig_max:
                low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
                inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
                inv_freq = inv_freq / factor * (1 - inv_freq_mask) + inv_freq * inv_freq_mask
            mscale = yarn_get_mscale(factor, rope_scaling.get("mscale", 1.0)) / yarn_get_mscale(
                factor, rope_scaling.get("mscale_all_dim", 0.0)
            )
    t = torch.arange(end, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs) * mscale
    sin = torch.sin(freqs) * mscale
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MiniLLMIndexer(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.enabled = bool(config.index_n_heads) and int(config.index_topk) > 0
        self.index_topk = int(config.index_topk)
        self.n_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        if not self.enabled:
            self.q_proj = None
            self.k_proj = None
            return
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        past_len: int,
    ) -> Optional[torch.Tensor]:
        if not self.enabled or self.q_proj is None or self.k_proj is None:
            return None
        if past_len > 0:
            return None
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None and attention_mask.dim() == 2:
            pad = (1.0 - attention_mask[:, None, None, :]) * -1e9
            scores = scores + pad
        scores = scores.mean(dim=1)
        k_len = scores.shape[-1]
        topk = min(int(self.index_topk), int(k_len))
        if topk >= k_len:
            return None
        _, topk_idx = torch.topk(scores, k=topk, dim=-1)
        return topk_idx


class MiniLLMAttention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank and config.q_lora_rank > 0 else None
        self.kv_lora_rank = config.kv_lora_rank
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=config.attention_bias)
            self.q_a_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.softmax_scale = self.q_head_dim ** -0.5
        if config.rope_scaling is not None and config.rope_scaling.get("type") == "yarn":
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(config.rope_scaling.get("factor", 1.0), mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        self.indexer = MiniLLMIndexer(config)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.shape
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_norm(kv)
        kv = self.kv_b_proj(kv)
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        cos = cos[:q_len]
        sin = sin[:q_len]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present = (key_states, value_states) if use_cache else None

        k_len = key_states.shape[2]
        past_len = k_len - q_len

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.softmax_scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad = (1.0 - attention_mask[:, None, None, :]) * -1e9
                attn_weights = attn_weights + pad
            else:
                attn_weights = attn_weights + attention_mask

        if q_len > 1:
            q_positions = torch.arange(q_len, device=attn_weights.device) + past_len
            k_positions = torch.arange(k_len, device=attn_weights.device)
            causal = k_positions[None, :] > q_positions[:, None]
            attn_weights = attn_weights + causal.to(attn_weights.dtype) * -1e9

        topk_idx = self.indexer(hidden_states=x, attention_mask=attention_mask, past_len=past_len)
        if topk_idx is not None:
            index_mask = attn_weights.new_full((bsz, q_len, k_len), float("-inf"))
            index_mask.scatter_(-1, topk_idx, 0.0)
            attn_weights = attn_weights + index_mask.unsqueeze(1)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.resid_dropout(self.o_proj(attn_output))
        return attn_output, present


class FeedForward(nn.Module):
    def __init__(self, config: MiniLLMConfig, *, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        if config.n_routed_experts is None:
            raise ValueError("n_routed_experts must be set for MoE")
        self.top_k = int(config.num_experts_per_tok)
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_group = int(config.n_group or 1)
        self.topk_group = int(config.topk_group or self.n_group)
        self.scoring_func = config.scoring_func
        self.aux_loss_alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = hidden_states.shape
        flat = hidden_states.view(-1, dim)
        scores = F.linear(flat, self.weight, None)
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            raise NotImplementedError(f"Unsupported scoring function: {self.scoring_func}")
        original_scores = scores

        if self.n_group > 1:
            scores = scores.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            if self.topk_group < self.n_group:
                if self.scoring_func == "sigmoid":
                    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
                else:
                    group_scores = scores.amax(dim=-1)
                top_groups = group_scores.topk(self.topk_group, dim=-1)[1]
                mask = torch.ones_like(group_scores, dtype=torch.bool)
                mask.scatter_(1, top_groups, False)
                scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
            scores = scores.reshape(-1, self.n_routed_experts)

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.scoring_func == "sigmoid" or (self.top_k > 1 and self.norm_topk_prob):
            denom = topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            topk_weight = topk_weight / denom
        topk_weight = topk_weight * self.routed_scaling_factor

        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training and self.aux_loss_alpha > 0.0:
            scores_for_aux = original_scores
            topk_idx_for_aux = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux, torch.ones_like(topk_idx_for_aux, dtype=ce.dtype))
                ce = ce.div(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.aux_loss_alpha

        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        if config.n_routed_experts is None:
            raise ValueError("n_routed_experts must be set for MoE")
        self.config = config
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)]
        )
        self.gate = MoEGate(config)
        self.shared_expert = None
        if config.n_shared_experts and config.n_shared_experts > 0:
            shared_dim = config.n_shared_experts * config.moe_intermediate_size
            self.shared_expert = FeedForward(config, intermediate_size=shared_dim)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x_rep = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x_rep)
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    y[mask] = expert(x_rep[mask])
            y = y.view(-1, self.config.num_experts_per_tok, x.shape[-1])
            y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
        else:
            y = self._moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1))
        y = y.view(*orig_shape)
        if self.shared_expert is not None:
            y = y + self.shared_expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def _moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount(minlength=len(self.experts)).cumsum(0).cpu().numpy()
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache


MOEFeedForward = MoEFeedForward


class MiniLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniLLMConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = MiniLLMAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if _is_moe_layer(layer_id, config):
            self.mlp = MoEFeedForward(config)
        else:
            self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states,
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class MTPPredictor(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        inter = int(config.mtp_intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


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
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniLLMBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        rope_dim = config.qk_rope_head_dim
        freqs_cos, freqs_sin = precompute_rope_freqs(
            dim=rope_dim,
            end=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        if input_ids is None:
            raise ValueError("input_ids must not be None")
        batch_size, seq_len = input_ids.shape
        if past_key_values is None:
            past_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.layers)
        else:
            past_list = [
                None if kv is None or kv[0] is None or kv[1] is None else kv for kv in past_key_values
            ]
        first_past = past_list[0]
        start_pos = int(first_past[0].shape[2]) if first_past is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_key_value in zip(self.layers, past_list):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        for layer in self.layers:
            if isinstance(layer.mlp, MoEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss.to(aux_loss.dtype)

        return hidden_states, presents, aux_loss


class MiniLLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniLLMConfig

    def __init__(self, config: Optional[MiniLLMConfig] = None):
        self.config = config or MiniLLMConfig()
        super().__init__(self.config)
        self.model = MiniLLMModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.mtp_layers = nn.ModuleList(
            [MTPPredictor(self.config) for _ in range(int(self.config.num_nextn_predict_layers or 0))]
        )
        self.OUT = CausalLMOutputWithPast()

    def _mtp_logits(
        self,
        hidden_states: torch.Tensor,
        *,
        slice_indices: Union[slice, torch.Tensor],
    ) -> List[torch.Tensor]:
        mtp_logits: List[torch.Tensor] = []
        if not self.mtp_layers:
            return mtp_logits
        mtp_hidden = hidden_states
        for layer in self.mtp_layers:
            mtp_hidden = mtp_hidden + layer(mtp_hidden)
            mtp_logits.append(self.lm_head(mtp_hidden[:, slice_indices, :]))
        return mtp_logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        slice_indices: Union[slice, torch.Tensor]
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        else:
            slice_indices = logits_to_keep

        logits = self.lm_head(hidden[:, slice_indices, :])
        mtp_logits = self._mtp_logits(hidden, slice_indices=slice_indices)

        self.OUT.__setitem__("last_hidden_state", hidden)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("past_key_values", past_kvs)
        self.OUT.__setitem__("mtp_logits", mtp_logits)
        return self.OUT
