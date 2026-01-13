<p align="center">
  <img src="assets/minillm-icon.svg" alt="MiniLLM logo" width="96" height="96"/>
</p>
<h1 align="center">MiniLLM</h1>
<p align="center">Lightweight LLM training, alignment, and deployment from scratch.</p>
<p align="center">
  <a href="./README.md">中文</a> ·
  <a href="./docs/README.md">Docs</a> ·
  <a href="./docs/booklet_cn.md">Booklet (CN)</a>
</p>
<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg"/>
  <img alt="platform" src="https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey.svg"/>
</p>

> This repository is a refactor of [MiniMind](https://github.com/jingyaogong/minimind), preserving the “build a lightweight LLM from scratch” learning goal while completing the data, training, evaluation, and deployment pipeline.

---

## ✨ Highlights

- End-to-end pipeline: Pretrain → SFT → Preference alignment (DPO/GRPO/PPO/SPO) → Distillation
- Training & inference: native PyTorch + DeepSpeed + MLX (Apple Silicon)
- Data tooling: cleaning, deduplication, quality scoring, RustBPE tokenization
- Deployment: Streamlit WebUI, OpenAI-compatible API, export to llama.cpp/vLLM/Ollama
- Evaluation: C-Eval, CMMLU, OpenBookQA and more

---

## 🚀 Quick Start

### 1) Environment

```bash
conda create -n minillm python=3.10 -y
conda activate minillm
pip install -r requirements.txt
```

If downloads are slow, use the Tsinghua PyPI mirror:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2) Data

- Put raw corpora under `dataset/` or a custom directory
- Run `scripts/prepare_data.sh` for cleaning, tokenization, and filtering
- Processed data is synced to `data/` for training

### 3) Training

```bash
# Pretrain → SFT → DPO
scripts/run.sh

# Skip pretraining, run SFT + DPO only
scripts/run.sh --skip-pretrain

# CPU smoke test
scripts/run.sh --smoke-test
```

### 4) WebUI

```bash
python -m streamlit run scripts/web_demo.py
```

Outputs and logs land in `out/` by default.

---

## 🍎 MLX (Apple Silicon)

```bash
# Download data → pretrain → SFT
bash scripts/run_mlx.sh

# Smoke test
bash scripts/run_mlx.sh --smoke-test
```

MLX checkpoints are stored under `out/mlx`, and the WebUI auto-resolves the latest `step_` checkpoint.

---

## 🧪 Distillation (Optional)

### One-click MLX distillation (Ollama teacher)

```bash
# Start ollama serve and pull a teacher model first (e.g. qwen3:0.6b)
bash scripts/run_mlx_distill_ollama.sh
```

Override with environment variables:

```bash
OLLAMA_MODEL=qwen3:0.6b DATA_JSONL=out/distill_ollama_qwen3_0.6b/synth.jsonl OUT_DIR=out/mlx_distill/qwen3_0.6b_sft \
  bash scripts/run_mlx_distill_ollama.sh
```

### PyTorch distillation training

```bash
# Expects out/full_sft_512.pth (student) and out/full_sft_768.pth (teacher) by default
python trainer/train_distillation.py --data_path dataset/sft_xxx.jsonl --out_dir out
```

### EAGLE-3 speculator (Qwen3-0.6B, pure synthetic data)

> Speculator defaults auto-scale to target model size; override with `--spec_len`/`--spec_layers` if needed.
> `--head_rank` defaults to hidden_size/8 (clamped to 32-256); override or set 0 to disable the low-rank head.
> MLX training auto-resumes from the latest checkpoint under `out_dir`; add `--no_resume` to start fresh.

```bash
# Torch: auto-generate synthetic data + train EAGLE-3 style speculator
python speculator/train/torch/train_eagle3_speculator.py
# Torch: benchmark (baseline vs speculator)
python speculator/infer/torch/bench.py --max_samples 16
```

```bash
# MLX: auto-generate synthetic data + train speculator
python speculator/train/mlx/train_eagle3_speculator.py --hf_repo Qwen/Qwen3-0.6B
# MLX: benchmark (baseline vs speculator)
python speculator/infer/mlx/bench.py --hf_repo Qwen/Qwen3-0.6B --max_samples 16
```

```bash
# Qwen3-1.7B + AngelSlim EAGLE-3 weights (MLX, no training)
# 1) Optional: convert Qwen3-1.7B to MLX weights (avoid first-time HF conversion)
python -m mlx_train.cli.hf_convert --hf_repo Qwen/Qwen3-1.7B --out_dir out/mlx_hf/qwen_qwen3_1_7b

# 2) Download AngelSlim EAGLE-3 drafter weights
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="AngelSlim/Qwen3-1.7B_eagle3",
    local_dir="out/eagle3_speculator_hf/angelslim_qwen3_1_7b_eagle3",
)
PY

# 3) MLX: benchmark (baseline vs EAGLE-3 weights)
python speculator/infer/mlx/bench.py \
  --hf_repo Qwen/Qwen3-1.7B \
  --model_dir out/mlx_hf/qwen_qwen3_1_7b \
  --eagle3_dir out/eagle3_speculator_hf/angelslim_qwen3_1_7b_eagle3 \
  --max_samples 16
```

```bash
# Qwen3-1.7B + AngelSlim EAGLE-3 weights (Torch, no training)
# 1) Download AngelSlim EAGLE-3 drafter weights
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="AngelSlim/Qwen3-1.7B_eagle3",
    local_dir="out/eagle3_speculator_hf/angelslim_qwen3_1_7b_eagle3",
)
PY

# 2) Torch: benchmark (baseline vs EAGLE-3 weights)
# Note: AngelSlim EAGLE-3 requires --eagle3_dir (tree decode). --speculator_dir is only for MiniLLM-trained speculators.
python speculator/infer/torch/bench.py \
  --target_arch qwen3 \
  --target_model Qwen/Qwen3-1.7B \
  --eagle3_dir out/eagle3_speculator_hf/angelslim_qwen3_1_7b_eagle3 \
  --max_samples 16
```

> MLX inference/training requires `mlx-lm` (currently pinned to transformers==5.0.0rc1). Use a clean venv if needed.

---

## 🧪 Inference & Serving

- **OpenAI-compatible API**: `python scripts/serve_openai_api.py` (default port 8998)
- **CLI evaluation**: `python eval_model.py --model_mode 1`
- **Training dashboard**: `python -m scripts.dashboard.app --host 0.0.0.0 --port 8008`

---

## 🧭 Repository Layout

```text
.
├── apps/                # Services & UI (OpenAI API / WebUI / Dashboard)
├── data/                # Data cache
├── dataset/             # Public dataset examples
├── docs/                # Documentation
├── speculator/          # Speculator training/inference entrypoints (torch/mlx)
├── mlx_train/           # MLX training and inference
├── model/               # MiniLLM Dense/MoE implementations
├── pipelines/           # One-click pipeline scripts (main logic)
├── scripts/             # Scripts and utilities
├── tokenizer/           # RustBPE tokenizer assets
├── trainer/             # Training/alignment/distillation scripts
├── tools/               # Data/eval/convert/tokenizer utilities
└── utils/               # Shared utilities and evaluation helpers
```

---

## 📚 Docs & Resources

- [docs/README.md](./docs/README.md) - Documentation index
- [docs/booklet_cn.md](./docs/booklet_cn.md) - Full pipeline booklet (CN)
- [docs/changelog/CHANGELOG.md](./docs/changelog/CHANGELOG.md) - Changelog
- [ModelScope: MiniMind-Reasoning](https://www.modelscope.cn/studios/gongjy/minimind-reasoning)
- [ModelScope: MiniMind](https://www.modelscope.cn/studios/gongjy/minimind)
- [Bilibili overview](https://www.bilibili.com/video/BV12dHPeqE72)

---

## 🤝 Contributing

Issues and pull requests are welcome. Please read [docs/CODE_OF_CONDUCT.md](./docs/CODE_OF_CONDUCT.md) and [AGENTS.md](./AGENTS.md) before contributing.

---

## 📄 License

Licensed under the [MIT License](./LICENSE). Please respect the original dataset/model licenses when redistributing.
