MiniLLM（MLX）DeepSeek-V3.2 架构说明
====================================

本文档详细介绍 MLX 端 MiniLLM 升级为 DeepSeek‑V3.2 风格后的架构实现。
内容从易到难、逐步深入；代码片段均为“简化表达”，便于理解。
需要对照真实实现时，请参考文中标注的源码路径。

适用范围与源码引用
------------------
- MLX 模型核心：`mlx_train/models/minillm.py`
- MLX 配置：`mlx_train/config.py`
- MTP 训练损失：`mlx_train/cli/train.py`
- MTP 投机解码：`speculator/infer/mlx/common.py`
- Torch 对照实现：`model/model_minillm.py`

一页速览（心智模型）
--------------------
1) MLA 注意力：
   - Q 拆成 NOPE + ROPE 两块。
   - Q/KV 使用低秩投影压缩。
   - RoPE 只作用于 ROPE 部分。
2) MoE MLP：
   - 路由器对每个 token 选 top‑k 专家。
   - 只计算被选中的专家（向量化 gather）。
3) YaRN RoPE：
   - 长上下文外推用的频率缩放。
4) MTP 头：
   - 多层 predictor 预测未来多个 token。
5) DSA/Indexer：
   - 可选 top‑k key 掩码，限制注意力范围。

----------------------------------------
1) MLA（Multi‑Head Latent Attention）
----------------------------------------
MLA 的核心是：Q/KV 低秩投影 + Q 拆分（NOPE/ROPE），仅对 ROPE 部分做旋转。

简化版（对应 `mlx_train/models/minillm.py::MiniLLMAttention`）：
```python
# Q 路径（可选低秩）
q = q_proj(x)                  # 或 q_b_proj(rms_norm(q_a_proj(x)))
q = q.reshape(B, T, H, Dq).transpose(0, 2, 1, 3)
q_nope, q_rope = split(q, [D_nope], axis=-1)

# KV 路径（低秩 + 独立 RoPE 分量）
kv = kv_a_proj(x)              # [B, T, kv_rank + D_rope]
kv, k_rope = split(kv, [kv_rank], axis=-1)
kv = kv_b_proj(rms_norm(kv))   # [B, T, H*(D_nope + Dv)]
k_nope, v = split(kv, [D_nope], axis=-1)

# RoPE 只作用于 rope‑part
q_rope, k_rope = rope(q_rope, k_rope, cos, sin)
q = concat(q_nope, q_rope)
k = concat(k_nope, repeat(k_rope, H))

attn = sdp_attention(q, k, v, mask=mask)
out = o_proj(attn)
```
源码参考：`mlx_train/models/minillm.py`（`MiniLLMAttention.__call__`、
`MiniLLMAttention.forward_with_cache`）。

关键配置：
- `q_lora_rank` / `kv_lora_rank`：低秩投影 rank。
- `qk_nope_head_dim` / `qk_rope_head_dim`：Q 拆分的维度。
- `v_head_dim`：V 维度。
- `rope_scaling`：启用 YaRN 时会调整 softmax scale。

----------------------------------------
2) YaRN RoPE（长上下文外推）
----------------------------------------
YaRN 通过调整频率与注意力温度实现更长上下文。

简化版（对应 `precompute_rope_freqs` 与注意力 scale）：
```python
inv_freq = 1 / (base ** (arange(0, dim, 2) / dim))
if rope_scaling["type"] == "yarn":
    inv_freq = mix(inv_freq / factor, inv_freq, ramp_mask)
    mscale = yarn_get_mscale(factor, mscale_cfg)
cos = cos(freqs) * mscale
sin = sin(freqs) * mscale
softmax_scale *= mscale * mscale
```
源码参考：`mlx_train/models/minillm.py`（`precompute_rope_freqs`、
`MiniLLMAttention.__init__`）。

----------------------------------------
3) MoE（Mixture‑of‑Experts）MLP
----------------------------------------
每个 token 选 top‑k 专家，MLX 侧用向量化 gather/einsum 实现“稀疏计算”。

简化版（对应 `MoEGate` + `MoEFeedForward`）：
```python
# Gate
scores = softmax(x @ gate_weight)              # [T, E]
topk_idx, topk_w = topk(scores, k=K)           # [T, K]
topk_w = normalize(topk_w) * routed_scale

# 专家计算（向量化 gather）
gate_w = gather(gate_proj_weight, topk_idx)    # [T, K, I, D]
up_w   = gather(up_proj_weight, topk_idx)
down_w = gather(down_proj_weight, topk_idx)

gate = einsum(x, gate_w)
up   = einsum(x, up_w)
act  = silu(gate) * up
out  = einsum(act, down_w)

out = sum(out * topk_w[..., None], axis=1)     # [T, D]
```
源码参考：`mlx_train/models/minillm.py`（`MoEGate.__call__`、
`MoEFeedForward.__call__`）。

说明：
- `num_experts_per_tok` = K，`n_routed_experts` = E。
- 共享专家（shared expert）作为可选残差路径。
- 支持 seq‑aux 或 token‑aux 的路由辅助损失。

----------------------------------------
4) MTP（Multi‑Token Prediction）
----------------------------------------
MTP 通过多个 predictor 层预测更远的未来 token。

简化版（`MTPPredictor` + `_mtp_hidden`）：
```python
h = model_hidden
mtp_hidden = []
for predictor in mtp_layers:
    h = h + predictor(h)
    mtp_hidden.append(h)
```
源码参考：`mlx_train/models/minillm.py`（`MTPPredictor`、
`MiniLLMForCausalLM._mtp_hidden`）。

训练损失（`mlx_train/cli/train.py`）：
```python
loss = CE(logits_t, labels_t)
for k, mtp_h in enumerate(mtp_hidden):
    offset = k + 2
    loss += CE(mtp_h[:, :-offset], labels[:, offset:]) / num_layers
loss *= mtp_loss_weight
```

MTP 投机解码（`speculator/infer/mlx/common.py`）：
```python
next_token = sample(last_logits)
draft_tokens = [sample(mtp_logits_i) for i in range(spec_len-1)]
逐步用 target cache 验证 draft token
```

----------------------------------------
5) DSA / Indexer（top‑k key 掩码）
----------------------------------------
Indexer 只在 prefill 阶段（`past_len == 0`）启用：
为每个 query 选 top‑k keys，然后把注意力掩码限制在这些 keys 上。

简化版（`MiniLLMIndexer` + `_resolve_attention_mask`）：
```python
scores = (Q @ K.T) * scale
scores = mean_over_heads(scores)           # [B, T, T]
topk_idx = topk(scores, k=index_topk)

index_mask = one_hot(topk_idx, K)          # [B, T, K]
mask = pad_mask + causal_mask + index_mask
attn = sdp_attention(q, k, v, mask=mask)
```
源码参考：`mlx_train/models/minillm.py`（`MiniLLMIndexer`、
`_resolve_attention_mask`）。

----------------------------------------
6) KV Cache 形状
----------------------------------------
KV cache 和 MLA 对齐：
- K: `[B, H, T, D_nope + D_rope]`
- V: `[B, H, T, D_v]`

源码参考：`mlx_train/models/minillm.py`（`allocate_kv_cache`、
`MiniLLMAttention.forward_with_cache`、
`MiniLLMModel.forward_with_cache`）。

----------------------------------------
7) 配置映射（MLX）
----------------------------------------
关键字段一览（与 Torch 对齐）：
- MLA：`q_lora_rank`、`kv_lora_rank`、`qk_nope_head_dim`、
  `qk_rope_head_dim`、`v_head_dim`
- MoE：`use_moe`、`n_routed_experts`、`num_experts_per_tok`、
  `n_shared_experts`、`aux_loss_alpha`、`seq_aux`、
  `moe_layer_freq`、`first_k_dense_replace`
- RoPE：`rope_theta`、`rope_scaling`、`inference_rope_scaling`
- MTP：`num_nextn_predict_layers`、`mtp_intermediate_size`、
  `mtp_loss_weight`
- DSA：`index_n_heads`、`index_head_dim`、`index_topk`

源码参考：`mlx_train/config.py` 与 `model/model_minillm.py`。

----------------------------------------
8) MLX 侧注意事项
----------------------------------------
- MLX 没有 scatter 操作，因此 MoE 用向量化 gather + einsum 实现稀疏计算。
- 注意力掩码支持：
  - 2D padding mask（B, T）
  - 4D additive mask
  - 可选 causal + indexer mask 叠加
- 使用 `mx.fast.scaled_dot_product_attention` 进行高效注意力计算。

----------------------------------------
9) 扩展与对齐建议
----------------------------------------
要与 Torch 细节完全一致：
- 对比 `mlx_train/models/minillm.py` 与 `model/model_minillm.py`。
- 重点看：
  - `MiniLLMAttention`
  - `MoEGate` / `MoEFeedForward`
  - `MiniLLMIndexer`
  - `MTPPredictor`

----------------------------------------
10) 测试与验证
----------------------------------------
最小测试：
- `tests/test_mlx_moe_indexer.py`

Smoke pipeline：
- `SMOKE_CLEAN=0 OUT=out/mlx_smoke bash scripts/run_mlx.sh --smoke-test`

MTP 推理：
- `.venv_mlx/bin/python speculator/infer/mlx/common.py --target_arch minillm --minillm_ckpt_dir out/mlx_smoke/sft/checkpoints/step_00000005 --minillm_tokenizer ./model --prompt "hi" --spec_len 2`
