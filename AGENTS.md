# AGENTS

This file guides automation and contributors working in this repository.

## Project Snapshot

MiniLLM provides end-to-end training pipelines (pretrain -> SFT -> inference) for
both PyTorch and MLX (Apple Silicon), plus data preparation utilities and eval.

## Architecture Map

- `model/`: core model definitions and configs.
- `trainer/`: PyTorch training, SFT, DPO, distillation entrypoints.
- `mlx_train/`: MLX training, inference, demo, data utilities.
- `tokenizer/`: RustBPE tokenizer and related tooling.
- `scripts/`: orchestration scripts and utilities.
- `configs/`: dashboard/job configs.
- `docs/`: docs, guides, changelog, and code of conduct.
- `dataset/`, `data/`, `out/`: data caches and generated outputs (not for VCS).

## Golden Paths

- MLX full pipeline: `scripts/run_mlx.sh`
  - Defaults to `pretrain_hq.jsonl` + `sft_mini_512.jsonl` (SMALL=1).
  - Full SFT data: set `SMALL=0`.
  - Short env names only:
    - Core: `PY`, `VENV`, `UV`, `OUT`, `DATA`
    - Data: `PRE_DATA`, `SFT_DATA`, `R1_DATA`, `AUTO_DL`, `DL_MAX`, `DPO_DL`
    - Training: `PRE_*`, `SFT_*`, `R1_*`, `PRESET`, `DTYPE`
    - Inference: `INF_*`
    - Gated attention: `GATE`, `GATE_INIT`
- Smoke test: `scripts/run_mlx.sh --smoke-test`
- PyTorch pipeline: `scripts/run.sh`

## Conventions

- Treat scripts as public APIs; avoid breaking flags or output layouts.
- Preserve checkpoint layout under `out/mlx/{pretrain,sft,r1}`.
- Keep large artifacts in `out/` or `dataset/`; do not commit them.
- Prefer minimal changes with clear docs updates.

## Validation

- MLX: `scripts/run_mlx.sh --smoke-test`
- Tests: `python -m pytest tests/test_vlm_smoke.py` (if deps available)
