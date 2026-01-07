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

## 🧪 推理与部署

- **OpenAI 兼容 API**：`python scripts/serve_openai_api.py`（默认端口 8998）
- **评测/推理脚本**：`python eval_model.py --model_mode 1`
- **训练监控面板**：`python -m scripts.dashboard.app --host 0.0.0.0 --port 8008`

---

## 🧭 仓库结构

```text
.
├── data/                # 数据缓存目录
├── dataset/             # 公开数据集示例与脚本
├── docs/                # 文档与指南
├── model/               # MiniLLM Dense/MoE 实现
├── tokenizer/           # RustBPE 分词与词表
├── trainer/             # 训练/对齐/蒸馏脚本
├── scripts/             # 一键训练/推理/工具脚本
├── mlx_train/           # MLX 训练与推理
└── utils/               # 公共工具与评估脚本
```

---

## 📚 资源与文档

- [docs/README.md](./docs/README.md)：文档入口与导航
- [docs/booklet_cn.md](./docs/booklet_cn.md)：完整中文小册子
- [docs/changelog/CHANGELOG.md](./docs/changelog/CHANGELOG.md)：版本记录
- [ModelScope: MiniLLM-Reasoning](https://www.modelscope.cn/studios/gongjy/MiniLLM-Reasoning)
- [ModelScope: MiniLLM](https://www.modelscope.cn/studios/gongjy/MiniLLM)
- [Bilibili 视频介绍](https://www.bilibili.com/video/BV12dHPeqE72)

---

## 🤝 贡献指南

欢迎通过 Issue 或 Pull Request 反馈问题和改进建议。请先阅读 [docs/CODE_OF_CONDUCT.md](./docs/CODE_OF_CONDUCT.md)。

---

## 📄 许可协议

本项目采用 [MIT License](./LICENSE)。在引用或再发布模型与数据时，请遵守相应许可证要求。
