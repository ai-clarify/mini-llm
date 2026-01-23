from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional


@dataclass
class MiniLLMConfig:
    # Keep names aligned with `model/model_minillm.py::MiniLLMConfig`.
    dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    max_position_embeddings: int = 32768
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 6400
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    inference_rope_scaling: bool = False
    attention_dropout: float = 0.0
    attention_bias: bool = False
    logit_softcap: float = 0.0
    tie_word_embeddings: bool = True

    # MLA
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    value_mix: float = 0.0
    partial_key_offset: int = 0
    paired_heads: bool = False
    attn_window: int = 0
    attn_global_tokens: int = 0
    sparse_attn_gate: bool = False
    sparse_attn_gate_topk: int = 0

    # MoE
    use_moe: bool = False
    num_experts_per_tok: int = 2
    n_routed_experts: Optional[int] = None
    n_shared_experts: int = 0
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0

    # MTP
    num_nextn_predict_layers: int = 1
    mtp_intermediate_size: Optional[int] = None
    mtp_loss_weight: float = 0.1

    # Residual scaling / init
    residual_scale: float = 1.0
    residual_decay: float = 0.0
    zero_init_residual: bool = False

    # Skip/value embedding tricks
    embed_skip_scale: float = 0.0
    embed_skip_gate: bool = False
    skip_connections: Optional[List[List[int]]] = None
    skip_scale: float = 1.0
    skip_gate: bool = False
    value_embed_count: int = 0
    value_embed_scale: float = 0.0
    value_embed_gate: bool = False
    value_embed_repeat_ends: bool = True

    # Input augmentation
    smear: bool = False
    smear_scale: float = 0.0
    bigram_hash_size: int = 0
    bigram_hash_scale: float = 0.0
    bigram_hash_base: int = 1000003

    # Back-out / untying schedule
    back_out_ratio: float = 0.0
    back_out_scale: float = 1.0
    untie_lm_head_at_ratio: float = 0.0

    # DSA / indexer (optional)
    index_n_heads: int = 0
    index_head_dim: int = 32
    index_topk: int = 0

    # Custom Metal fused kernels (MLX path)
    use_metal_kernels: bool = True

    # Gated attention (optional)
    use_attn_gate: bool = False
    attn_gate_init: float = 4.0  # logit; sigmoid(4) ~= 0.982

    # LoRA (MLX path)
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    def finalize(self) -> "MiniLLMConfig":
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) must be divisible by num_attention_heads({self.num_attention_heads})"
            )

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

        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MiniLLMConfig":
        """
        Construct a config from a dict while ignoring unknown keys.

        This makes checkpoint `config.json` forward/backward compatible across
        versions (newer keys won't break older code and vice versa).
        """
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered).finalize()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def minillm_200mb() -> MiniLLMConfig:
    """
    ~200M params preset (DeepSeek-V3.2-style, small-scale).

    Note: fp16/bf16 weight size will be ~400MB (order of magnitude).
    """

    return MiniLLMConfig(
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        vocab_size=6400,
        max_position_embeddings=32768,
        rope_theta=10000.0,
        dropout=0.0,
        use_moe=True,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_layer_freq=1,
        first_k_dense_replace=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        scoring_func="softmax",
    ).finalize()
