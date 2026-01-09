#!/usr/bin/env bash
set -euo pipefail

HF_REPO=${HF_REPO:-Qwen/Qwen3-0.6B}
PROMPT=${PROMPT:-"Hello"}
SYSTEM=${SYSTEM:-}
MAX_TOKENS=${MAX_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-}
QUANTIZE=${QUANTIZE:-0}
Q_BITS=${Q_BITS:-4}
Q_GROUP_SIZE=${Q_GROUP_SIZE:-64}
Q_MODE=${Q_MODE:-affine}
DTYPE=${DTYPE:-}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}

safe_name=$(echo "$HF_REPO" | tr '[:upper:]' '[:lower:]' | tr '/:.-' '____' | tr -s '_')
OUT_DIR=${OUT_DIR:-out/mlx_hf/$safe_name}

if [ -d "$OUT_DIR" ]; then
  if { [ ! -s "$OUT_DIR/model.safetensors" ] && [ ! -s "$OUT_DIR/weights.npz" ]; } || \
     [ ! -s "$OUT_DIR/config.json" ]; then
    echo "[error] $OUT_DIR exists but is missing weights/config.json" >&2
    exit 1
  fi
else
  cmd=(python3 -m mlx_train.hf_convert --hf_repo "$HF_REPO" --out_dir "$OUT_DIR")
  if [ "$QUANTIZE" = "1" ]; then
    cmd+=(--quantize --q_bits "$Q_BITS" --q_group_size "$Q_GROUP_SIZE" --q_mode "$Q_MODE")
  fi
  if [ -n "$DTYPE" ]; then
    cmd+=(--dtype "$DTYPE")
  fi
  if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    cmd+=(--trust_remote_code)
  fi
  "${cmd[@]}"
fi

infer_cmd=(python3 -m mlx_train.hf_infer --model_dir "$OUT_DIR" --prompt "$PROMPT" --max_tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P")
if [ -n "$SYSTEM" ]; then
  infer_cmd+=(--system "$SYSTEM")
fi
if [ -n "$SEED" ]; then
  infer_cmd+=(--seed "$SEED")
fi

"${infer_cmd[@]}"
