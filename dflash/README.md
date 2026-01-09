# DFlash (local reproduction)

This folder mirrors the DFlash block-diffusion draft design for speculative decoding.
It keeps the target model weights unchanged ("old" weights) and adds a new draft
model format for speculative decoding, then benchmarks against non-spec decode
(block_size=1).

Source design: https://github.com/z-lab/dflash

## Quick start

1) Install deps (root `requirements.txt` already includes `torch`, `transformers`, `datasets`):

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

2) Run a single-GPU benchmark:

```bash
python dflash/benchmark.py \
  --block-size 16 \
  --dataset gsm8k \
  --max-samples 32 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 1024 \
  --temperature 1.0
```

The script reports decoding speedup and acceptance length. It always compares
block_size=1 (non-spec) vs block_size=N (spec).

## Notes

- This reproduction currently targets Qwen3-based models because the draft
  architecture relies on `transformers.models.qwen3` modules.
- Multi-GPU usage follows `torchrun` (see `dflash/run_benchmark.sh`).
