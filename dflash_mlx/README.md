# dflash_mlx (MLX / macOS)

A lightweight MLX benchmark for DFlash-style speculative decoding on macOS.
It compares non-spec decoding (block_size=1) vs block speculative decoding
(block_size=N) using MiniLLM MLX checkpoints.

Note: the official DFlash draft weights are only available for Qwen3-4B/8B.
For a smaller model on Mac, use the MiniLLM MLX "tiny" preset and run the
benchmark with a smaller checkpoint.

## Quick start

1) Prepare a MLX checkpoint (example: tiny smoke test). Note that smoke-test
cleans outputs by default; set `CLEANUP_SMOKE=0` or provide `OUT=...` to keep them.

```bash
CLEANUP_SMOKE=0 bash scripts/run_mlx.sh --smoke-test
```

This produces checkpoints under `out/mlx_smoke` (e.g. `out/mlx_smoke/sft/checkpoints/step_XXXXXX`).

2) Run the MLX speculative benchmark

```bash
TARGET_CKPT=out/mlx_smoke/sft/checkpoints/step_XXXXXX \
BLOCK_SIZE=8 MAX_SAMPLES=8 MAX_NEW_TOKENS=128 \
DRAFT_LAYERS=2 \
bash dflash_mlx/run_benchmark.sh
```

## Options

- `TARGET_CKPT`: MLX checkpoint dir (required).
- `DRAFT_CKPT`: optional separate draft checkpoint dir.
- `DRAFT_LAYERS`: if no `DRAFT_CKPT`, slice the first N layers from target weights.
- `BLOCK_SIZE`: speculative block length.
- `DATASET`: dataset name (gsm8k/math500/...) or a .jsonl file with `turns`/`prompt`/`text`. Default is `demo` (no extra deps).

## Notes

- Draft slicing uses the first N layers from the target weights (no extra training).
- For Hugging Face datasets (e.g. gsm8k), install `datasets`: `python3 -m pip install datasets`.
- This MLX version follows standard speculative decoding; it does not implement
  the diffusion draft architecture from the original DFlash repo.
