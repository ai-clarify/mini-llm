# AGENTS

This file guides automation and contributors working in this repository.

## Project Snapshot

MiniLLM provides end-to-end training pipelines (pretrain -> SFT -> inference) for
both PyTorch and MLX (Apple Silicon), plus data preparation utilities and eval.

## Architecture Map

- `model/`: core model definitions and configs.
- `trainer/`: PyTorch training, SFT, DPO, distillation entrypoints.
- `mlx_train/`: MLX training, inference, demo, data utilities.
- `speculator/`: EAGLE-3 speculator training/inference (torch/mlx).
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
- Avoid unnecessary defensive code; if context guarantees invariants, use direct access instead of `getattr` or guard clauses.
- Deliver professional, well-considered results; ensure changes are coherent, documented, and ready to run.
- Benchmark outputs should include progress and actionable failure guidance.
- Behave as a top-tier algorithm engineer: keep a sharp experimental mindset and proactively analyze every dataset.

## Implementation standards for algorithms

- Prefer pure functions and immutable inputs where reasonable.
- Must handle edge cases explicitly: empty input, single element, duplicates, negative numbers, very large values, already-sorted, reverse-sorted.
- Complexity must be stated in docstring: time and space, include best/average/worst when relevant.
- Determinism required: do not rely on hash iteration order; if randomness is needed, seed it and expose the seed parameter.
- Avoid premature micro-optimizations; optimize only with a failing benchmark or a measured hotspot.

## Testing standards

- Every new algorithm needs tests that cover:
  - representative examples
  - edge cases
  - randomized/property-style tests when appropriate (keep deterministic seeds)
  - regression test for any bug fix
- Keep tests fast; if a test is slow, mark it and justify.

## Style conventions

- Follow existing naming and docstring style in nearby files.
- Prefer explicit types for public functions.
- Keep functions small; extract helpers instead of deeply nested logic.
- No print debugging in committed code.

## Dependency policy and boundaries

- Do not add new runtime dependencies without asking first.
- Do not change public APIs (function names/signatures) unless the task explicitly requires it.
- Never commit secrets, tokens, or local paths.
- Do not edit generated files or vendor directories (if present).

## Validation

- MLX: `scripts/run_mlx.sh --smoke-test`
- Tests: `python -m pytest tests/test_vlm_smoke.py` (if deps available)
