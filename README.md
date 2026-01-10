<p align="center">
  <img src="assets/minillm-icon.svg" alt="MiniLLM logo" width="96" height="96"/>
</p>
<h1 align="center">MiniLLM</h1>
<p align="center">轻量级 LLM 训练、对齐、部署一体化项目，面向从 0 到 1 的学习与复现。</p>
<p align="center">
  <a href="./README_en.md">English</a> ·
  <a href="./docs/README.md">Docs</a> ·
  <a href="./docs/booklet_cn.md">Booklet</a>
</p>
<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg"/>
  <img alt="platform" src="https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey.svg"/>
</p>

> 本仓库由 [MiniMind](https://github.com/jingyaogong/minimind) 项目重构而来，保留“从零实现轻量级 LLM”的教学目标，并补全数据、训练、评估与部署流程。

---

## ✨ 特性

- 端到端训练链路：预训练 → SFT → 偏好对齐（DPO/GRPO/PPO/SPO）→ 蒸馏
- 训练与推理：原生 PyTorch + DeepSpeed + MLX（Apple Silicon）
- 数据管线：清洗、去重、质量评估、RustBPE 分词
- 部署方式：Streamlit WebUI、OpenAI 协议 API、llama.cpp/vLLM/Ollama 导出
- 评估工具：C-Eval、CMMLU、OpenBookQA 等基准评测

---

## 🚀 快速开始

### 1) 环境准备

```bash
conda create -n minillm python=3.10 -y
conda activate minillm
pip install -r requirements.txt
```

如果下载较慢，可使用清华源：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2) 数据准备

- 将原始语料放在 `dataset/` 或自定义目录
- 运行 `scripts/prepare_data.sh` 完成去重、分词、过滤
- 处理后的数据会同步到 `data/` 供训练脚本使用

### 3) 一键训练

```bash
# 预训练 → SFT → DPO
scripts/run.sh

# 跳过预训练，仅执行 SFT + DPO
scripts/run.sh --skip-pretrain

# 烟雾测试（CPU + 小数据）
scripts/run.sh --smoke-test
```

### 4) WebUI

```bash
python -m streamlit run scripts/web_demo.py
```

训练日志、权重与评估输出默认保存在 `out/`。

---

## 🍎 MLX（Apple Silicon）

```bash
# 自动跑通下载数据 → 预训练 → SFT
bash scripts/run_mlx.sh

# Smoke Test
bash scripts/run_mlx.sh --smoke-test
```

MLX 产物默认写入 `out/mlx`，WebUI 会自动解析最新 `step_` checkpoint。

---

## 🧪 蒸馏（可选）

### MLX 一键蒸馏（Ollama 教师模型）

```bash
# 需要先启动 ollama serve，并拉取教师模型（如 qwen3:0.6b）
bash scripts/run_mlx_distill_ollama.sh
```

可通过环境变量调整：

```bash
OLLAMA_MODEL=qwen3:0.6b DATA_JSONL=out/distill_ollama_qwen3_0.6b/synth.jsonl OUT_DIR=out/mlx_distill/qwen3_0.6b_sft \
  bash scripts/run_mlx_distill_ollama.sh
```

### EAGLE-3 speculator（Qwen3-0.6B / MiniLLM，纯合成数据）

> - speculator 默认会根据目标模型大小自动设置；可用 `--spec_len`/`--spec_layers` 显式覆盖。
> - `--head_rank` 默认自动设置（hidden_size/8，范围 32-256）；可显式指定或设为 0 关闭低秩头。

#### Qwen3-0.6B（Torch）

```bash
# Torch：自动生成合成数据 + 训练 EAGLE-3 style speculator
python train/torch/train_eagle3_speculator.py
# Torch：基准对比（baseline vs speculator）
python infer/torch/bench.py --max_samples 16
```

#### Qwen3-0.6B（MLX）

```bash
# MLX：自动生成合成数据 + 训练 speculator
python train/mlx/train_eagle3_speculator.py --hf_repo Qwen/Qwen3-0.6B
# MLX：基准对比（baseline vs speculator）
python infer/mlx/bench.py --hf_repo Qwen/Qwen3-0.6B --max_samples 16
```

#### MiniLLM（Torch）

```bash
# Torch：训练（指定 MiniLLM checkpoint + tokenizer）
python train/torch/train_eagle3_speculator.py \
  --target_arch minillm \
  --minillm_ckpt out/pretrain_512.pth \
  --minillm_tokenizer ./model
# Torch：基准对比
python infer/torch/bench.py \
  --target_arch minillm \
  --minillm_ckpt out/pretrain_512.pth \
  --minillm_tokenizer ./model
```

#### MiniLLM（MLX）

```bash
# MLX：训练（使用 mlx_train 产出的 checkpoint 目录）
python train/mlx/train_eagle3_speculator.py \
  --target_arch minillm \
  --minillm_ckpt_dir out/mlx/sft/checkpoints/step_00000050 \
  --minillm_tokenizer ./model
# MLX：基准对比
python infer/mlx/bench.py \
  --target_arch minillm \
  --minillm_ckpt_dir out/mlx/sft/checkpoints/step_00000050 \
  --minillm_tokenizer ./model
```

> MLX 推理/训练依赖 `mlx-lm`（当前与 transformers==5.0.0rc1 绑定），建议使用独立虚拟环境。

### PyTorch 蒸馏训练

```bash
# 默认读取 out/ 中的 full_sft_512.pth（学生）与 full_sft_768.pth（教师）
python trainer/train_distillation.py --data_path dataset/sft_xxx.jsonl --out_dir out
```

---

## 🧪 推理与部署

- **OpenAI 兼容 API**：`python scripts/serve_openai_api.py`（默认端口 8998）
- **评测/推理脚本**：`python eval_model.py --model_mode 1`
- **训练监控面板**：`python -m scripts.dashboard.app --host 0.0.0.0 --port 8008`

---

## 🧭 仓库结构

```text
.
├── apps/                # 服务与 UI（OpenAI API / WebUI / Dashboard）
├── data/                # 数据缓存目录
├── dataset/             # 公开数据集示例与脚本
├── docs/                # 文档与指南
├── infer/               # Speculator 推理入口（torch/mlx）
├── mlx_train/           # MLX 训练与推理
├── model/               # MiniLLM Dense/MoE 实现
├── pipelines/           # 一键训练/推理流水线脚本（主逻辑）
├── scripts/             # 脚本与工具
├── tokenizer/           # RustBPE 分词与词表
├── trainer/             # 训练/对齐/蒸馏脚本
├── train/               # Speculator 训练入口（torch/mlx）
├── tools/               # 数据/评测/转换/分词等工具脚本
└── utils/               # 公共工具与评估脚本
```

---

## 📚 资源与文档

- [docs/README.md](./docs/README.md)：文档入口与导航
- [docs/booklet_cn.md](./docs/booklet_cn.md)：完整中文小册子
- [docs/changelog/CHANGELOG.md](./docs/changelog/CHANGELOG.md)：版本记录
- [ModelScope: MiniMind-Reasoning](https://www.modelscope.cn/studios/gongjy/minimind-reasoning)
- [ModelScope: MiniMind](https://www.modelscope.cn/studios/gongjy/minimind)
- [Bilibili 视频介绍](https://www.bilibili.com/video/BV12dHPeqE72)

---

## 🤝 贡献指南

欢迎通过 Issue 或 Pull Request 反馈问题和改进建议。请先阅读 [docs/CODE_OF_CONDUCT.md](./docs/CODE_OF_CONDUCT.md)，并参考 [AGENTS.md](./AGENTS.md) 了解项目约定。

---

## 📄 许可协议

本项目采用 [MIT License](./LICENSE)。在引用或再发布模型与数据时，请遵守相应许可证要求。
