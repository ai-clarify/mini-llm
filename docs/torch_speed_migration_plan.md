# Torch 侧速度优化迁移方案（MiniLLM）

目标：把 MLX 已验证的吞吐优化“体系化迁移”到 PyTorch 训练链路，在 **不破坏现有脚本 API** 的前提下，尽可能提高 tok/s。

约束：
- 不新增运行时依赖（除非用户明确批准）。
- 保持现有 CLI 兼容，新增参数必须有默认值。
- 保留 `out/` checkpoint 目录结构与日志格式。

---

## 1) 数据管线迁移（最高优先级）

### 1.1 支持 bin2d（与 MLX 共享）
**动机**：bin2d 固定长度 + 连续内存读取显著减少 tokenizer/JSON 解析开销。

**实现**：
- 新增 `dataset/bin2d_dataset.py`：
  - `Bin2DDataset` 使用 `mmap` 或 `read_bytes` 方式读取 `*.ids2d.bin` 与 `*.lbl.*`。
  - 与 MLX 的 `meta.json` 兼容（相同字段）。
- 训练脚本新增 `--data_format`（`jsonl|bin|bin2d`）和 `--bin_cache`（`mmap|memory`）。

### 1.2 支持 `ids` 预分词 JSONL
**动机**：JSONL 中若已有 `ids` 字段则跳过 tokenizer。
**实现**：`SFTDataset/PretrainDataset` 支持 `ids`（并保持 loss_mask 逻辑）。

### 1.3 Streaming + shuffle buffer
**动机**：避免一次性加载全部数据；降低内存并稳定吞吐。
**实现**：
- 新增 `IterableDataset` 流式读取 JSONL/bin2d。
- `--shuffle_buffer`：维持固定 buffer 随机替换输出（与 MLX 一致）。
- DDP shard：按 rank/step 取样，避免跨进程重复。

### 1.4 Bucketing 减少 padding
**动机**：大幅减少无效 token 计算。
**实现**：
- `--bucket_sizes 256,512,...`
- collate 时按 bucket 分组（保证 batch 内长度一致）。

---

## 2) Sparse Loss 迁移（SFT 关键）

### 2.1 label positions 预计算
**动机**：仅在 loss_mask==1 位置计算 CE，避免全量 [B,T,V]。
**实现**：
- bin2d 写 `label_pos` 与 `label_pos_mask`。
- JSONL 路径运行时生成 label_pos（可缓存）。

### 2.2 Sparse gather loss
**实现**：
- 在 loss 处仅 gather label_pos 位置 logits。
- 支持 `--label_bucket_sizes` 对 label 数分桶（减少 padding）。

---

## 3) 编译与内核路径

### 3.1 `torch.compile`
**动机**：减少 Python overhead，提升 kernel 融合。
**实现**：新增 `--compile` 开关（默认 off，按需启用）。

### 3.2 SDPA / FlashAttention
**动机**：注意力是吞吐热点。
**实现**：
- 优先走 `torch.nn.functional.scaled_dot_product_attention`（自动选择 flash/mem-efficient）。
- 若未启用 flash，则 fallback 原实现。

---

## 4) Optimizer/AMP

### 4.1 Fused AdamW
**动机**：减少优化器 overhead。
**实现**：
- 优先 `torch.optim.AdamW(fused=True, foreach=True)`（版本允许时）。

### 4.2 AMP 与 TF32
**动机**：更高吞吐。
**实现**：
- `torch.cuda.amp.autocast` + GradScaler。
- `torch.set_float32_matmul_precision("high")`（若启用 TF32）。

---

## 5) 训练调度/系统

### 5.1 Grad Accum / Batch schedule
**动机**：通过减少优化器调用次数提高吞吐。
**实现**：已有 `accum_steps` + `batch_size_schedule` 机制，新增到 torch 侧。

### 5.2 预取与 pinned memory
**动机**：CPU 供数与 GPU 训练并行。
**实现**：
- `DataLoader(num_workers, prefetch_factor, persistent_workers, pin_memory=True)`
- `non_blocking=True` 传输。

---

## 6) 可观测性与基准

### 6.1 统一 tok/s 统计
**实现**：在 torch 训练脚本中输出 `tok/s` 与 `step_ms`（与 MLX 对齐）。

### 6.2 Profiling
**实现**：`torch.profiler` 或 NVTX 标记关键路径（data/fwd/bwd/opt）。

---

## 7) 交付顺序（最小风险）

1. **bin2d Dataset + data_format**（提升最大）
2. **shuffle_buffer + bucketing**（稳定性/吞吐）
3. **sparse_loss + label_pos**（SFT 大幅收益）
4. **SDPA + compile**（内核层）
5. **fused AdamW + AMP**（最后优化）

---

## 8) 预期收益与风险

- 预期：数据管线与 sparse loss 将提供最大吞吐收益（对 SFT）。
- 风险：torch.compile 与 flash attention 在某些设备/版本不稳定，需要做 fallback。
- 不新增依赖的前提下，提升主要来自 **数据与 loss 侧**。

