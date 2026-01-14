DeepSeek-V3.2-Style MiniLLM (MLX) Architecture Guide
====================================================

This document explains how the MiniLLM MLX backend is upgraded to a
DeepSeek-V3.2-style architecture. The explanation is "easy first, deeper later",
and the code snippets are simplified for clarity. For exact behavior, see the
referenced source files.

Scope and reference code
------------------------
- MLX model core: `mlx_train/models/minillm.py`
- MLX config: `mlx_train/config.py`
- MTP training loss: `mlx_train/cli/train.py`
- MTP decode (speculative): `speculator/infer/mlx/common.py`
- Torch parity reference: `model/model_minillm.py`

Quick mental model
------------------
1) MLA attention:
   - Split Q into (NOPE + ROPE) parts.
   - Compress Q/KV with low-rank (LoRA-style) projections.
   - Apply RoPE only to the ROPE part.
2) MoE MLP:
   - Gate selects top-k experts per token.
   - Only selected experts are evaluated (vectorized gather).
3) YaRN RoPE scaling:
   - Adjusts RoPE frequencies for long-context.
4) MTP heads:
   - Predict multiple future tokens with residual predictor layers.
5) DSA/indexer:
   - Optional top-k key selection to mask attention.

----------------------------------------
1) MLA (Multi-Head Latent Attention)
----------------------------------------
MLA compresses Q/KV with low-rank projections and splits RoPE/NOPE
subspaces to reduce compute while keeping expressiveness.

Simplified (from `mlx_train/models/minillm.py::MiniLLMAttention`):
```python
# Q path (with optional low-rank)
q = q_proj(x)                  # or q_b_proj(rms_norm(q_a_proj(x)))
q = q.reshape(B, T, H, Dq).transpose(0, 2, 1, 3)
q_nope, q_rope = split(q, [D_nope], axis=-1)

# KV path (low-rank for K/V plus separate RoPE part)
kv = kv_a_proj(x)              # [B, T, kv_rank + D_rope]
kv, k_rope = split(kv, [kv_rank], axis=-1)
kv = kv_b_proj(rms_norm(kv))   # [B, T, H*(D_nope + Dv)]
k_nope, v = split(kv, [D_nope], axis=-1)

# RoPE on rope-part only
q_rope, k_rope = rope(q_rope, k_rope, cos, sin)
q = concat(q_nope, q_rope)
k = concat(k_nope, repeat(k_rope, H))

attn = sdp_attention(q, k, v, mask=mask)
out = o_proj(attn)
```
Reference: `mlx_train/models/minillm.py` (MiniLLMAttention.__call__,
MiniLLMAttention.forward_with_cache).

Notes:
- `q_lora_rank` and `kv_lora_rank` control the low-rank compression.
- `qk_nope_head_dim` and `qk_rope_head_dim` define the split sizes.
- `v_head_dim` is separate from Q/K head dims.
- Attention scale includes YaRN adjustment when enabled.

----------------------------------------
2) YaRN RoPE scaling
----------------------------------------
YaRN extends RoPE to longer contexts by scaling frequencies and the
softmax temperature.

Simplified (from `precompute_rope_freqs` and attention scale logic):
```python
inv_freq = 1 / (base ** (arange(0, dim, 2) / dim))
if rope_scaling["type"] == "yarn":
    inv_freq = mix(inv_freq / factor, inv_freq, ramp_mask)
    mscale = yarn_get_mscale(factor, mscale_cfg)
cos = cos(freqs) * mscale
sin = sin(freqs) * mscale
softmax_scale *= mscale * mscale  # attention temperature adjustment
```
Reference: `mlx_train/models/minillm.py` (precompute_rope_freqs,
MiniLLMAttention.__init__).

----------------------------------------
3) MoE (Mixture-of-Experts) MLP
----------------------------------------
MoE routes each token to top-k experts. In MLX we implement a sparse,
vectorized path (no scatter ops are available).

Simplified (from `MoEGate` + `MoEFeedForward`):
```python
# Gate
scores = softmax(x @ gate_weight)              # [T, E]
topk_idx, topk_w = topk(scores, k=K)           # [T, K]
topk_w = normalize(topk_w) * routed_scale

# Expert compute (vectorized gather)
gate_w = gather(gate_proj_weight, topk_idx)    # [T, K, I, D]
up_w   = gather(up_proj_weight, topk_idx)
down_w = gather(down_proj_weight, topk_idx)

gate = einsum(x, gate_w)
up   = einsum(x, up_w)
act  = silu(gate) * up
out  = einsum(act, down_w)

out = sum(out * topk_w[..., None], axis=1)     # [T, D]
```
Reference: `mlx_train/models/minillm.py` (MoEGate.__call__,
MoEFeedForward.__call__).

Notes:
- `num_experts_per_tok` = K, `n_routed_experts` = E.
- Optional shared expert adds a dense path on top.
- Auxiliary routing loss (seq-aux or token-aux) is implemented.

----------------------------------------
4) MTP (Multi-Token Prediction)
----------------------------------------
MTP adds predictor layers that each try to predict a further future token.

Simplified (from `MTPPredictor` and `_mtp_hidden`):
```python
h = model_hidden
mtp_hidden = []
for predictor in mtp_layers:
    h = h + predictor(h)
    mtp_hidden.append(h)
```
Reference: `mlx_train/models/minillm.py` (MTPPredictor, MiniLLMForCausalLM._mtp_hidden).

Training loss (from `mlx_train/cli/train.py`):
```python
loss = CE(logits_t, labels_t)
for k, mtp_h in enumerate(mtp_hidden):
    offset = k + 2
    loss += CE(mtp_h[:, :-offset], labels[:, offset:]) / num_layers
loss *= mtp_loss_weight
```
Reference: `mlx_train/cli/train.py` (loss_fn).

MTP speculative decoding (from `speculator/infer/mlx/common.py`):
```python
next_token = sample(last_logits)
draft_tokens = [sample(mtp_logits_i) for i in range(spec_len-1)]
verify tokens with cached target model one by one
```
Reference: `speculator/infer/mlx/common.py` (_mtp_speculative_decode_minillm).

----------------------------------------
5) DSA / Indexer (Top-k key masking)
----------------------------------------
The indexer optionally selects top-k keys per query and masks attention
to those keys. It is only applied at prefill (past_len == 0), matching
the torch side.

Simplified (from `MiniLLMIndexer` + `_resolve_attention_mask`):
```python
scores = (Q @ K.T) * scale
scores = mean_over_heads(scores)           # [B, T, T]
topk_idx = topk(scores, k=index_topk)

index_mask = one_hot(topk_idx, K)          # [B, T, K]
mask = pad_mask + causal_mask + index_mask
attn = sdp_attention(q, k, v, mask=mask)
```
Reference: `mlx_train/models/minillm.py` (MiniLLMIndexer, _resolve_attention_mask).

Notes:
- `index_n_heads`, `index_head_dim`, `index_topk` control this path.
- If `index_topk >= seq_len`, indexer is disabled for that batch.

----------------------------------------
6) Forward path and KV cache
----------------------------------------
KV cache uses MLA key dimensions:
- K shape: `[B, H, T, D_nope + D_rope]`
- V shape: `[B, H, T, D_v]`

Reference: `mlx_train/models/minillm.py` (allocate_kv_cache,
MiniLLMAttention.forward_with_cache, MiniLLMModel.forward_with_cache).

----------------------------------------
7) Config mapping (MLX)
----------------------------------------
Key config fields are aligned with the torch config:
- MLA: `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`
- MoE: `use_moe`, `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts`,
  `aux_loss_alpha`, `seq_aux`, `moe_layer_freq`, `first_k_dense_replace`
- RoPE: `rope_theta`, `rope_scaling`, `inference_rope_scaling`
- MTP: `num_nextn_predict_layers`, `mtp_intermediate_size`, `mtp_loss_weight`
- DSA: `index_n_heads`, `index_head_dim`, `index_topk`

Reference: `mlx_train/config.py` and `model/model_minillm.py`.

----------------------------------------
8) MLX-specific considerations
----------------------------------------
- MoE is sparse but implemented via vectorized gather + einsum, because MLX
  lacks scatter ops.
- Attention masking supports:
  - `attention_mask` as 2D padding mask (B, T)
  - full 4D additive masks
  - optional causal + indexer masks
- `mx.fast.scaled_dot_product_attention` is used for speed and supports
  additive masks.

----------------------------------------
9) How to read or extend the code
----------------------------------------
If you need to match the torch path closely:
- Compare `mlx_train/models/minillm.py` with `model/model_minillm.py`.
- Start with these classes:
  - `MiniLLMAttention`
  - `MoEGate` / `MoEFeedForward`
  - `MiniLLMIndexer`
  - `MTPPredictor`

If you add new features, update:
- `mlx_train/config.py` (config surface)
- `mlx_train/cli/train.py` (training losses)
- `speculator/infer/mlx/common.py` (inference / MTP)

----------------------------------------
10) Testing and validation
----------------------------------------
Minimal MLX tests:
- `tests/test_mlx_moe_indexer.py`

Smoke pipeline:
- `SMOKE_CLEAN=0 OUT=out/mlx_smoke bash scripts/run_mlx.sh --smoke-test`

MTP decode:
- `.venv_mlx/bin/python speculator/infer/mlx/common.py --target_arch minillm --minillm_ckpt_dir out/mlx_smoke/sft/checkpoints/step_00000005 --minillm_tokenizer ./model --prompt "hi" --spec_len 2`
