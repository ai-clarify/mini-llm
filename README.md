# MiniLLM

> 本仓库由 [MiniMind](https://github.com/jingyaogong/minimind) 项目重构而来，在保留“从零实现轻量级 LLM”教学目标的同时，补全了数据、训练、评估与部署的一体化流程。

中文 | [English](./README_en.md)

MiniLLM 提供一套可在本地 GPU 或低成本云端环境运行的轻量级大语言模型 (LLM) 训练与部署方案。相比原始 MiniMind，我们对代码结构、脚本接口、数据处理管线进行了完整的再设计，使得学习者可以：

- 以统一的配置完成 **预训练 → 监督微调 (SFT) → 偏好对齐 (DPO/GRPO/PPO/SPO) → 蒸馏** 的整条训练链路；
- 通过 `scripts/run.sh` 在单机单卡、DDP 或 DeepSpeed 环境下一键复现；
- 使用内置的数据清洗、质量评估与去重工具快速构造高质量数据集；
- 以 WebUI、OpenAI 协议 API 等方式部署推理服务，并兼容 llama.cpp / vLLM / Ollama 等生态。

---

## 🔍 仓库概览

```text
.
├── data/                # 标准化的数据缓存目录（预训练、SFT、偏好对齐数据）
├── dataset/             # 公开数据集示例与脚本
├── docs/                # 深入文档与操作手册
├── model/               # MiniLLM Dense 与 MoE 模型实现
├── tokenizer/           # RustBPE 分词训练与词表
├── trainer/             # 预训练/SFT/LoRA/DPO/GRPO/PPO/SPO/蒸馏脚本
├── scripts/             # 一键训练、推理与工具脚本（含 run.sh 主入口）
├── utils/               # 公共工具与评估脚本
└── requirements.txt     # 依赖清单
```

> 详细的目录与模块说明请参考 [docs/README.md](./docs/README.md) 与 [docs/booklet_cn.md](./docs/booklet_cn.md)。

---

## 🚀 快速开始

### 1. 环境准备

```bash
conda create -n minillm python=3.10 -y
conda activate minillm
pip install -r requirements.txt
```

- 推荐使用带有 ≥12GB 显存的 NVIDIA GPU；
- 默认使用清华镜像下载依赖，可根据需要修改 `pyproject.toml` 中的 `[tool.pip]` 配置。

### 2. 数据准备

1. 将原始语料放置到 `dataset/` 或自定义目录；
2. 运行 `scripts/prepare_data.sh`（或参考 [docs/data_processing_pipeline.md](./docs/data_processing_pipeline.md)）完成去重、分词、质量过滤；
3. 处理后的 JSONL/CSV 文件会自动同步到 `data/final/`，供训练脚本使用。

### 3. 一键训练

`scripts/run.sh` 是推荐的训练入口，可根据需求组合参数：

```bash
# 完整三阶段：预训练 → SFT → DPO
scripts/run.sh

# 仅执行 SFT + DPO（跳过预训练）
scripts/run.sh --skip-pretrain

# 烟雾测试（CPU，小数据集）
scripts/run.sh --smoke-test
```

### 3.1 MLX 一键训练（≈200M params 预设）

如果你使用 Apple Silicon 并希望用 **MLX** 跑通训练/推理流程，可使用：

```bash
# 完整流程：自动下载 MiniMind 数据 -> 预训练 -> SFT
bash scripts/run_mlx.sh

# Smoke：小数据 + 少量步数 + 推理
bash scripts/run_mlx.sh --smoke-test
```

更多说明见 `mlx_train/README.md`。

脚本会自动：

- 检测本地或云端环境并选择合适的数据/Checkpoint；
- 输出到 `out/` 目录，包括最新模型权重与评估日志；
- 调用 `trainer/` 中的对应 Python 脚本执行各阶段训练；
- 在云端（如 OpenBayes）自动构建高质量 SFT 数据集。

更多参数说明见 [docs/run_script_options.md](./docs/run_script_options.md)。

### 4. 可视化与监控

- 训练默认集成 [SwanLab](https://swanlab.cn) 与 [Weights & Biases](https://wandb.ai)；
- 可通过环境变量 `MINILLM_USE_SWANLAB` / `WANDB_API_KEY` 控制启用；
- 日志与配置保存在 `out/logs/`，便于二次分析。
- 新增 `scripts/dashboard/` 仪表盘（灵感来自 [llm-madness](https://github.com/MaxHastings/llm-madness)），一站式查看配置模板、数据集与 `out/` 内的运行记录：
  ```bash
  python -m scripts.dashboard.app --host 0.0.0.0 --port 8008
  ```
  打开页面后可以：
  - 浏览 `configs/dashboard/*.json` 中的预设管线/训练配置；
  - 勾选数据文件并合并成 `out/datasets/<name>/combined.jsonl` 快照，自动写入 manifest；
  - 聚合 `out/` 目录下的 checkpoint、TensorBoard 标量并绘制曲线，便于快速比对训练状态。

### 5. (可选) 启动 WebUI

```bash
streamlit run scripts/web_demo.py
```

---

## 🧠 模型与特性

- **模型结构**：实现了密集 (Dense) 与专家混合 (MoE) 两套架构，兼容 Rope、YaRN 等位置编码策略；
- **训练策略**：
  - 预训练支持原生 PyTorch、DeepSpeed、FlashAttention；
  - 指令微调支持全量微调与 LoRA；
  - 对齐阶段包含 DPO、PPO、GRPO、SPO，可按需选择；
  - 提供蒸馏脚本，支持 MiniLLM-Reason 等推理模型；
- **部署能力**：提供 Streamlit WebUI、OpenAI 协议服务端、导出到 llama.cpp/vLLM/Ollama 的转换工具；
- **评估支持**：集成 C-Eval、CMMLU、OpenBookQA 等基准评测脚本。

已有的模型权重、数据与演示请参见：

- [ModelScope: MiniLLM-Reasoning](https://www.modelscope.cn/studios/gongjy/MiniLLM-Reasoning)
- [ModelScope: MiniLLM](https://www.modelscope.cn/studios/gongjy/MiniLLM)
- [Bilibili 视频介绍](https://www.bilibili.com/video/BV12dHPeqE72)

---

## 📚 学习资源

- [docs/booklet_cn.md](./docs/booklet_cn.md)：MiniLLM 全流程小册子，涵盖数据构建、RustBPE、训练与部署；
- [docs/guides/](./docs/guides)：针对不同阶段（预训练、SFT、对齐、部署）的分步教程；
- [docs/changelog/](./docs/changelog)：历史更新记录；
- [analyze_cleaned_data.py](./analyze_cleaned_data.py)：数据质量分析示例脚本。

---

## 🤝 贡献指南

欢迎通过 Issue 或 Pull Request 反馈问题、共享数据与改进方案。

1. Fork 仓库并新建分支；
2. 遵循 [docs/CODE_OF_CONDUCT.md](./docs/CODE_OF_CONDUCT.md)；
3. 使用 `scripts/run.sh --smoke-test` 或 `pytest` 进行最小化验证；
4. 提交 PR 时请附上关键日志或截图，帮助我们快速复现。

---

## 📄 许可协议

本项目采用 [MIT License](./LICENSE)。在引用或再发布模型与数据时，请遵守相应数据集或权重的许可证要求。
