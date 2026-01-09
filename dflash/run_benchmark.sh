#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B}
DRAFT=${DRAFT:-z-lab/Qwen3-4B-DFlash-b16}
DATASET=${DATASET:-gsm8k}
MAX_SAMPLES=${MAX_SAMPLES:-32}
BLOCK_SIZE=${BLOCK_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-1.0}
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-29500}

if [ "$NPROC" -gt 1 ]; then
  torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    dflash/benchmark.py \
    --block-size "$BLOCK_SIZE" \
    --dataset "$DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --model-name-or-path "$MODEL" \
    --draft-name-or-path "$DRAFT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
else
  python dflash/benchmark.py \
    --block-size "$BLOCK_SIZE" \
    --dataset "$DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --model-name-or-path "$MODEL" \
    --draft-name-or-path "$DRAFT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
fi
