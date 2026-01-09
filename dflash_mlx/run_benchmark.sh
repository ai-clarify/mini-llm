#!/usr/bin/env bash
set -euo pipefail

if [ -z "${TARGET_CKPT:-}" ]; then
  echo "[error] TARGET_CKPT is required (path to MLX checkpoint dir)" >&2
  exit 1
fi

DATASET=${DATASET:-demo}
MAX_SAMPLES=${MAX_SAMPLES:-16}
BLOCK_SIZE=${BLOCK_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOKENIZER_PATH=${TOKENIZER_PATH:-./model}
DRAFT_CKPT=${DRAFT_CKPT:-}
DRAFT_LAYERS=${DRAFT_LAYERS:-}
DTYPE=${DTYPE:-float16}

cmd=(python dflash_mlx/benchmark.py \
  --target_ckpt "$TARGET_CKPT" \
  --dataset "$DATASET" \
  --max_samples "$MAX_SAMPLES" \
  --block_size "$BLOCK_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dtype "$DTYPE")

if [ -n "$DRAFT_CKPT" ]; then
  cmd+=(--draft_ckpt "$DRAFT_CKPT")
elif [ -n "$DRAFT_LAYERS" ]; then
  cmd+=(--draft_layers "$DRAFT_LAYERS")
fi

"${cmd[@]}"
