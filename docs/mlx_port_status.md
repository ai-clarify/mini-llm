# MLX 迁移要点覆盖情况

梳理自 `unsloth-mlx` 的核心设计，并标记在本仓库中的落地状态，便于后续对齐与补全。

## LoRA 低开销适配 + 冻结策略

- **目标与实现方式**：短名→全路径映射，`linear_to_lora_layers` 一次性替换，并在应用前 `freeze()` 底座，只解冻 LoRA 权重（参见 `unsloth_mlx/model.py::MLXModelWrapper._apply_lora`）。
- **当前状态**：部分（**✓ 已冻结底座/叶子名匹配/参数计数；缺短名→全路径映射与 `linear_to_lora_layers` 批量替换**）。我们的 `mlx_train/nn/lora.py` 在替换时会冻结底层 `nn.Linear`，支持目标层叶子名匹配，并输出可训练参数量；但未做“短名→全路径”映射，也未使用 `linear_to_lora_layers` 批量替换。后续可补充路径映射与批量替换以减少分支判断和显存占用。

## 按需启用梯度检查点

- **目标与实现方式**：显式参数 + LoRA 配置共同决定，仅必要时开启（`unsloth_mlx/sft_trainer.py::_should_use_grad_checkpoint`）。
- **当前状态**：未实现（**✗**）。目前仅支持 `checkpoint_every_n`（逐层重算），未与 LoRA 配置联动。需要新增“参数 + 配置”联合判定的开关。

## 原生 MLX 训练优先、CLI 兜底

- **目标与实现方式**：优先走 `mlx_lm.tuner.train`，失败再回退 `mlx_lm.lora` CLI（`unsloth_mlx/sft_trainer.py::train`）。
- **当前状态**：未对齐（**✗**）。现有训练循环完全自研（`mlx_train/cli/train.py`），无 `mlx_lm` 入口与 CLI 兜底。若要对齐，可添加后端选择与自动回退机制。

## LR 调度与优化器集中管理

- **目标与实现方式**：统一封装 cosine/linear/constant 调度与 `optim.AdamW`（`unsloth_mlx/sft_trainer.py::_get_lr_schedule`）。
- **当前状态**：部分（**✓ 已有 cosine 调度与优化器封装；缺 linear/constant 策略工厂化**）。`mlx_train/cli/train.py` 内自带 cosine 退火和优化器创建，但未抽象为可选策略（linear/constant），也未集中到独立调度工厂。可引入策略枚举与工厂以便复用。

## 数据管线本地缓存

- **目标与实现方式**：加载后用 `CacheDataset` 缓存，减少重复 tokenization（`unsloth_mlx/sft_trainer.py::_train_native`）。
- **当前状态**：部分（**✓ 已有预分词缓存；缺懒加载 CacheDataset 风格的迭代缓存**）。已增加 `pretokenize_jsonl` 与 `--cache_tokenized`（`mlx_train/data/__init__.py`, `mlx_train/cli/train.py`），可复用 token ids；但未引入类似 `CacheDataset` 的按需懒缓存。若需进一步优化，可在迭代器内加入分批缓存。

## 多种 RLHF 损失的高效实现

- **目标与实现方式**：DPO/ORPO/KTO/SimPO/GRPO 等基于 MLX 的高效损失，支持 reference log-probs 与多生成（`unsloth_mlx/losses.py`, `unsloth_mlx/rl_trainers.py`）。
- **当前状态**：未实现（**✗**）。当前 RL/对齐训练仍依赖现有 PyTorch 实现。若计划迁移到 MLX，需要补全对应损失与数据流程。

## 推理模式切换与 KV Cache

- **目标与实现方式**：`for_inference`/`enable_inference_mode` 开启 KV cache、关闭训练开销（`unsloth_mlx/model.py`）。
- **当前状态**：未实现（**✗**）。暂无显式推理模式切换或 KV cache 控制；需在推理入口添加快捷封装。

## 量化与统一内存利用

- **目标与实现方式**：加载 4bit/8bit 预量化模型并利用 MLX 统一内存（`unsloth_mlx/model.py`, README）。
- **当前状态**：未实现（**✗**）。当前 MLX 路径未提供量化加载选项；可在模型加载流程中增加量化与 dtype 选择，以降低显存占用。
