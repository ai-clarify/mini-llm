#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-"$ROOT_DIR/.venv_mlx/bin/python"}
DATA=${DATA:-"$ROOT_DIR/dataset"}
OUT=${OUT:-"$ROOT_DIR/out"}

EXTRA_ARGS=("$@")

"$PY" -m mlx_train.train \
  --data_path "$DATA/minimind/sft_mini_512_2d.meta.json" --data_format bin2d --bin_cache memory \
  --task sft --preset tiny --seq_len 512 --batch_size 72 --accum_steps 1 \
  --prefetch_batches 6 --shuffle_buffer 512 \
  --dtype bfloat16 --sparse_loss --label_bucket_sizes 64,128,256,512 \
  --paired_heads --memory_limit_mb 14000 \
  --tokenizer_type auto \
  --out_dir "$OUT/mlx_tiny_fast" \
  "${EXTRA_ARGS[@]}"
