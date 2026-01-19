# run.sh 脚本使用指南

## 概述

`scripts/run.sh` 是 MiniGPT 训练的主入口脚本，支持完整的三阶段训练流程：
1. **Pretrain** - 预训练
2. **SFT** - 监督微调
3. **DPO** - 直接偏好优化

## 命令行选项

```bash
scripts/run.sh [OPTIONS]
```

### 可用选项

| 选项 | 说明 |
|-----|------|
| `--smoke-test` | 快速烟雾测试模式（CPU，小数据集） |
| `--skip-pretrain` | 如果有好的 checkpoint 则跳过 pretrain |
| `--force-pretrain` | 强制执行 pretrain（覆盖 --skip-pretrain） |
| `-h, --help` | 显示帮助信息 |

---

## 使用场景

### 1. 完整训练（从零开始）

```bash
# 本地环境
scripts/run.sh

# 云端环境（自动处理数据）
scripts/run.sh
```

**流程**:
- Pretrain (2 epochs)
- SFT (使用高质量数据集)
- DPO (偏好优化)

---

### 2. 跳过 Pretrain（节省时间）

**使用场景**: 已有训练好的 pretrain checkpoint，想直接进行 SFT

```bash
scripts/run.sh --skip-pretrain
```

**脚本行为**:
1. 检查是否存在 pretrain checkpoint
2. 验证 checkpoint 文件大小和完整性
3. 如果 checkpoint 有效，跳过 pretrain 阶段
4. 直接使用现有 checkpoint 进行 SFT 训练

**检测位置**（按优先级）:
1. `$MINILLM_PRETRAINED_PATH` 环境变量指定的路径
2. `/openbayes/home/out/pretrain_512.pth` (云端)
3. `out/pretrain_512.pth` (本地)

**示例输出**:
```
[stage] Found existing pretrain checkpoint: out/pretrain_512.pth
[stage] Checking checkpoint quality...
[stage] Checkpoint looks valid (size: 45678KB)
[stage] Skipping pretrain stage, will use existing checkpoint for SFT
[stage] Pretrain stage skipped
[eval] Running quick evaluation on existing pretrain checkpoint
[stage] Starting SFT
```

---

### 3. 强制 Pretrain

**使用场景**: 即使有 checkpoint 也要重新训练（例如：更新了数据集）

```bash
scripts/run.sh --force-pretrain
```

**脚本行为**:
- 忽略所有现有 checkpoint
- 从零开始 pretrain
- 覆盖旧的 checkpoint 文件

---

### 4. 烟雾测试（快速验证）

**使用场景**: 本地开发，验证训练流程是否正常工作

```bash
scripts/run.sh --smoke-test
```

**测试配置**:
- 设备: CPU
- 数据: 极小子集（pretrain 64条，SFT 16条，DPO 8条）
- 步数: 每阶段 4 步
- 评估: 8 个样本

**预期运行时间**: 5-10 分钟

---

### 5. 组合使用

```bash
# 烟雾测试 + 跳过 pretrain
scripts/run.sh --smoke-test --skip-pretrain

# 注意：--force-pretrain 会覆盖 --skip-pretrain
scripts/run.sh --skip-pretrain --force-pretrain  # 实际会执行 pretrain
```

---

## 云端环境特性

### 自动数据处理

在 OpenBayes 云端环境，脚本会自动：

1. **检测云端环境**
   ```
   [cloud] Cloud environment detected, checking for input data...
   ```

2. **查找输入数据**
   - `/openbayes/input/input0/final/sft_mini_512.cleaned.jsonl`
   - `/openbayes/input/input0/sft_mini_512.jsonl`

3. **自动生成高质量数据集**
   ```
   [cloud] Auto-generating high-quality SFT dataset...
   [cloud] Processing complete: kept 1,132,794, removed 16,976
   ```

4. **使用高质量数据训练**
   - 输出: `data/final/sft_high_quality.jsonl`

### 云端 + 跳过 Pretrain

```bash
# 理想的云端训练流程
scripts/run.sh --skip-pretrain
```

**优势**:
- 自动处理输入数据生成高质量数据集
- 使用现有 pretrain checkpoint（如果有）
- 节省训练时间和资源

---

## 环境变量

可以通过环境变量自定义训练行为：

### 数据路径

```bash
# 自定义 SFT 数据
SFT_JSON=/path/to/custom_sft.jsonl scripts/run.sh

# 自定义 Pretrain 数据
PRETRAIN_JSON=/path/to/pretrain.jsonl scripts/run.sh

# 自定义 DPO 数据
DPO_JSON=/path/to/dpo.jsonl scripts/run.sh
```

### 模型配置

```bash
# 512 维模型（默认）
MODEL_HIDDEN_SIZE=512 scripts/run.sh

# 1024 维模型
MODEL_HIDDEN_SIZE=1024 MODEL_NUM_LAYERS=16 scripts/run.sh

# 使用 MoE 架构
USE_MOE=true scripts/run.sh
```

### 输出目录

```bash
# 自定义输出目录
OUT_DIR=/path/to/output scripts/run.sh

# 自定义 TensorBoard 目录
TF_DIR=/path/to/tensorboard scripts/run.sh
```

### Checkpoint 与恢复

默认输出布局对齐 MLX：

```
out/
  pretrain/
    checkpoints/step_00000010/
    pretrain_512.pth
  sft/
    checkpoints/step_00000010/
    full_sft_512.pth
  dpo/
    checkpoints/step_00000010/
    rlhf_512.pth
  r1/
    checkpoints/step_00000010/
    reason_512.pth
```

兼容文件仍会链接到根目录（如 `out/pretrain_512.pth`），便于旧脚本使用。
当存在 step checkpoint 时，脚本会自动断点续训（恢复 optimizer/step/seed）。

可选环境变量：

```bash
# 每阶段保存间隔（默认 200）
SAVE_INTERVAL=200

# 每阶段保留最近 N 个 step checkpoints（默认 3）
KEEP_LAST=3
```

### TensorBoard 自动启动

默认启用自动启动 TensorBoard（可关闭）：

```bash
# 关闭自动启动
TB_AUTO=0

# 自定义 host/port
TB_HOST=127.0.0.1
TB_PORT=6006
```

### Pretrain Checkpoint

```bash
# 指定预训练模型路径
MINILLM_PRETRAINED_PATH=/path/to/pretrain.pth scripts/run.sh
```

---

## 完整示例

### 示例 1: 本地开发测试

```bash
# 快速烟雾测试
scripts/run.sh --smoke-test

# 完整训练（小模型）
MODEL_HIDDEN_SIZE=256 MODEL_NUM_LAYERS=4 scripts/run.sh
```

### 示例 2: 云端首次训练

```bash
# 在 OpenBayes 上首次运行
# 假设 /openbayes/input/input0 已挂载数据
scripts/run.sh
```

**流程**:
1. 检测云端环境
2. 自动处理输入数据生成高质量 SFT 数据集
3. Pretrain (2 epochs)
4. SFT (使用高质量数据)
5. DPO

### 示例 3: 云端继续训练

```bash
# 假设之前已经完成了 pretrain
# /openbayes/home/out/pretrain_512.pth 存在
scripts/run.sh --skip-pretrain
```

**流程**:
1. 检测到现有 pretrain checkpoint
2. 验证 checkpoint 有效
3. 跳过 pretrain
4. 直接进行 SFT + DPO

### 示例 4: 使用自定义数据

```bash
# 本地：使用自己准备的高质量数据
SFT_JSON=data/my_custom_sft.jsonl \
PRETRAIN_JSON=data/my_pretrain.jsonl \
scripts/run.sh --skip-pretrain
```

### 示例 5: 强制重新训练

```bash
# 更新了 pretrain 数据，需要重新训练
PRETRAIN_JSON=data/pretrain_v2.jsonl scripts/run.sh --force-pretrain
```

---

## Checkpoint 管理

### Checkpoint 位置

训练会生成以下 checkpoint 文件：

```
out/
├── pretrain_512.pth       # Pretrain checkpoint
├── full_sft_512.pth       # SFT checkpoint
└── rlhf_512.pth           # DPO checkpoint
```

### Checkpoint 自动加载

脚本会自动查找和加载 checkpoint（按优先级）：

1. **环境变量**: `$MINILLM_PRETRAINED_PATH`
2. **云端目录**: `/openbayes/home/out/`
3. **本地目录**: `$OUT_DIR/`

### 手动管理

```bash
# 备份 checkpoint
cp out/pretrain_512.pth out/pretrain_512_backup.pth

# 使用备份的 checkpoint
MINILLM_PRETRAINED_PATH=out/pretrain_512_backup.pth scripts/run.sh --skip-pretrain

# 清除所有 checkpoint（重新开始）
rm -f out/*.pth
scripts/run.sh
```

---

## 故障排除

### 问题 1: Pretrain checkpoint 找不到

```bash
[stage] No pretrain checkpoint found, will train from scratch
```

**解决方案**:
- 确保之前的训练已完成
- 检查 `out/pretrain_512.pth` 是否存在
- 或者移除 `--skip-pretrain` 选项

### 问题 2: Checkpoint 损坏

```bash
[stage] Checkpoint appears corrupted or too small, will retrain
```

**解决方案**:
- 删除损坏的 checkpoint: `rm out/pretrain_512.pth`
- 重新训练: `scripts/run.sh --force-pretrain`

### 问题 3: 云端数据未找到

```bash
[data] Required dataset not found: /openbayes/input/input0/sft_mini_512.jsonl
```

**解决方案**:
- 确保数据已正确挂载到 `/openbayes/input/input0`
- 或者使用环境变量指定数据路径:
  ```bash
  SFT_JSON=/path/to/your/data.jsonl scripts/run.sh
  ```

---

## 性能优化建议

### 1. 使用 --skip-pretrain

如果有可用的 pretrain checkpoint，使用此选项可以节省数小时的训练时间。

### 2. 云端环境优化

```bash
# 利用云端自动数据处理 + 跳过 pretrain
scripts/run.sh --skip-pretrain
```

### 3. 调整批量大小

```bash
# 如果 GPU 内存充足，增加批量大小
SFT_ARGS="--batch_size 32" scripts/run.sh
```

### 4. 并行数据加载

```bash
# 增加数据加载 worker
PRETRAIN_ARGS="--num_workers 4" scripts/run.sh
```

---

## 总结

`scripts/run.sh` 提供了灵活的训练控制：

- ✓ **自动化**: 云端自动处理数据
- ✓ **可跳过阶段**: 使用现有 checkpoint 节省时间
- ✓ **灵活配置**: 环境变量自定义所有参数
- ✓ **安全检查**: 自动验证 checkpoint 完整性
- ✓ **开发友好**: 烟雾测试快速验证

**推荐工作流**:

1. **首次训练**: `scripts/run.sh`
2. **后续迭代**: `scripts/run.sh --skip-pretrain`
3. **本地测试**: `scripts/run.sh --smoke-test`
4. **更新数据**: `scripts/run.sh --force-pretrain`
