#!/usr/bin/env bash
set -euo pipefail

on_interrupt() {
  echo
  echo "[abort] Interrupted."
  exit 130
}
trap on_interrupt INT

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x ".venv_mlx/bin/python" ]; then
    PY=".venv_mlx/bin/python"
  else
    PY="python3"
  fi
fi

# Prefer `/root/gsm8k` inside containers; fall back to a workspace dir when not writable.
if [ -z "${DATASET_DIR+x}" ]; then
  if [ -d "/root/gsm8k" ]; then
    DATASET_DIR="/root/gsm8k"
  else
    if mkdir -p "/root/gsm8k" >/dev/null 2>&1; then
      DATASET_DIR="/root/gsm8k"
    else
      DATASET_DIR="dataset/gsm8k"
    fi
  fi
fi
SPLIT=${SPLIT:-train}
TOKENIZER_PATH=${TOKENIZER_PATH:-./model}
CHECKPOINT=${CHECKPOINT:-}
OUT_DIR=${OUT_DIR:-out/rl_gsm8k}
RESET_OUT=${RESET_OUT:-0} # 1 = move existing OUT_DIR aside and start fresh

NUM_ROLLOUTS=${NUM_ROLLOUTS:-512}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-1}
MIN_POSITIVE=${MIN_POSITIVE:-0}
MAX_TOTAL_ROLLOUTS=${MAX_TOTAL_ROLLOUTS:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
SEED=${SEED:-1337}

SEQ_LEN=${SEQ_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_STEPS=${MAX_STEPS:-1000}
ITERS=${ITERS:-5}
STEPS_PER_ITER=${STEPS_PER_ITER:-} # optional; default = ceil(MAX_STEPS / ITERS)
CLEAN_BUFFERS=${CLEAN_BUFFERS:-1}  # 1 = overwrite each iter buffer

DOWNLOAD=${DOWNLOAD:-auto} # auto|0|1
REPO_ID=${REPO_ID:-zhuzilin/gsm8k}

# Optional SFT warmup to bootstrap reward positives.
WARMUP_STEPS=${WARMUP_STEPS:-200}          # 0 = disable
WARMUP_LIMIT=${WARMUP_LIMIT:-2048}         # training samples to write (0/empty = all)
WARMUP_SEQ_LEN=${WARMUP_SEQ_LEN:-1024}
WARMUP_BATCH_SIZE=${WARMUP_BATCH_SIZE:-4}
WARMUP_LR=${WARMUP_LR:-1e-4}
WARMUP_ANSWER_MODE=${WARMUP_ANSWER_MODE:-full} # full|final
WARMUP_OUT_DIR=${WARMUP_OUT_DIR:-$OUT_DIR/warmup_sft}
WARMUP_JSONL=${WARMUP_JSONL:-$WARMUP_OUT_DIR/gsm8k_sft.jsonl}
WARMUP_REGEN=${WARMUP_REGEN:-0} # 1 = re-generate JSONL even if exists

# Allow passing checkpoint as the first positional arg:
#   bash scripts/run_mlx_rl_gsm8k.sh out/mlx/sft/checkpoints/step_XXXXXXXX
if [ -z "$CHECKPOINT" ] && [ "${1:-}" != "" ] && [[ "${1:-}" != -* ]]; then
  if [ -d "$1" ]; then
    CHECKPOINT="$1"
    shift
  fi
fi

if [ -z "$CHECKPOINT" ]; then
  echo "[error] CHECKPOINT is required (e.g. out/mlx/sft/checkpoints/step_XXXXXXXX)" >&2
  exit 2
fi

BASE_CKPT="$CHECKPOINT"

if [ "$RESET_OUT" = "1" ] && [ -d "$OUT_DIR" ]; then
  ts=$(date +"%Y%m%d_%H%M%S" 2>/dev/null || true)
  if [ -z "$ts" ]; then
    ts=$("$PY" - <<'PY'
import time
print(time.strftime("%Y%m%d_%H%M%S"))
PY
)
  fi
  mv "$OUT_DIR" "${OUT_DIR}.bak_${ts}"
  echo "[reset] moved existing OUT_DIR -> ${OUT_DIR}.bak_${ts}"
fi

mkdir -p "$OUT_DIR"
BUFFER_PATH=${BUFFER_PATH:-$OUT_DIR/buffer.jsonl}

if [ "$DOWNLOAD" = "auto" ]; then
  if [ -d "$DATASET_DIR" ] && find "$DATASET_DIR" -type f \( -name "*train*.parquet" -o -name "*train*.jsonl" -o -name "*train*.json" \) | head -n 1 | grep -q .; then
    DOWNLOAD=0
  else
    DOWNLOAD=1
  fi
fi

if [ "$DOWNLOAD" = "1" ]; then
  "$PY" -m mlx_train.rl_gsm8k.download \
    --repo_id "$REPO_ID" \
    --repo_type dataset \
    --local_dir "$DATASET_DIR"
elif [ ! -d "$DATASET_DIR" ]; then
  echo "[error] dataset_dir not found: $DATASET_DIR (set DOWNLOAD=1 or DATASET_DIR=...)" >&2
  exit 2
fi

BUFFER_DIR=${BUFFER_DIR:-$OUT_DIR/buffer}
mkdir -p "$BUFFER_DIR"

if [ -z "${STEPS_PER_ITER}" ]; then
  if [ "$ITERS" -le 0 ]; then
    STEPS_PER_ITER=0
  else
    # ceil(MAX_STEPS / ITERS)
    STEPS_PER_ITER=$(( (MAX_STEPS + ITERS - 1) / ITERS ))
  fi
fi

latest_ckpt() {
  ls -d "$OUT_DIR"/checkpoints/step_* 2>/dev/null | sort | tail -n 1 || true
}

latest_ckpt_in() {
  local dir="$1"
  ls -d "$dir"/checkpoints/step_* 2>/dev/null | sort | tail -n 1 || true
}

ckpt_step() {
  local path="$1"
  local base
  base=$(basename "$path")
  echo "${base#step_}" | sed 's/^0*//; s/^$/0/'
}

existing_rl_ckpt=$(latest_ckpt)
if [ -n "$existing_rl_ckpt" ]; then
  echo "[resume] found existing checkpoint under OUT_DIR: $existing_rl_ckpt"
fi

if [ "$WARMUP_STEPS" != "0" ] && [ "$WARMUP_STEPS" -gt 0 ]; then
  if [ -n "$existing_rl_ckpt" ]; then
    echo "[warmup] skip (OUT_DIR already has checkpoints). To warmup from scratch, set RESET_OUT=1 or OUT_DIR=..."
  else
    mkdir -p "$WARMUP_OUT_DIR"
    if [ "$WARMUP_REGEN" = "1" ] || [ ! -f "$WARMUP_JSONL" ]; then
      warmup_limit_arg=()
      if [ -n "${WARMUP_LIMIT}" ] && [ "$WARMUP_LIMIT" != "0" ]; then
        warmup_limit_arg=(--limit "$WARMUP_LIMIT")
      fi
      "$PY" -m mlx_train.rl_gsm8k.prepare_sft \
        --dataset_dir "$DATASET_DIR" \
        --split "$SPLIT" \
        --out_path "$WARMUP_JSONL" \
        --answer_mode "$WARMUP_ANSWER_MODE" \
        "${warmup_limit_arg[@]}"
    fi

    warmup_ckpt=$(latest_ckpt_in "$WARMUP_OUT_DIR")
    warmup_step=0
    if [ -n "$warmup_ckpt" ]; then
      warmup_step=$(ckpt_step "$warmup_ckpt")
    fi

    if [ "$warmup_step" -ge "$WARMUP_STEPS" ] && [ -n "$warmup_ckpt" ]; then
      echo "[warmup] reuse ckpt=$warmup_ckpt (step=$warmup_step >= warmup_steps=$WARMUP_STEPS)"
    else
      if [ -n "$warmup_ckpt" ]; then
        echo "[warmup] resume=$warmup_ckpt -> max_steps=$WARMUP_STEPS"
        "$PY" -m mlx_train.train \
          --task sft \
          --tokenizer_path "$TOKENIZER_PATH" \
          --data_path "$WARMUP_JSONL" \
          --resume "$warmup_ckpt" \
          --out_dir "$WARMUP_OUT_DIR" \
          --seq_len "$WARMUP_SEQ_LEN" \
          --batch_size "$WARMUP_BATCH_SIZE" \
          --learning_rate "$WARMUP_LR" \
          --epochs 999999 \
          --max_steps "$WARMUP_STEPS"
      else
        echo "[warmup] init_from=$BASE_CKPT -> max_steps=$WARMUP_STEPS"
        "$PY" -m mlx_train.train \
          --task sft \
          --tokenizer_path "$TOKENIZER_PATH" \
          --data_path "$WARMUP_JSONL" \
          --init_from "$BASE_CKPT" \
          --out_dir "$WARMUP_OUT_DIR" \
          --seq_len "$WARMUP_SEQ_LEN" \
          --batch_size "$WARMUP_BATCH_SIZE" \
          --learning_rate "$WARMUP_LR" \
          --epochs 999999 \
          --max_steps "$WARMUP_STEPS"
      fi

      warmup_ckpt=$(latest_ckpt_in "$WARMUP_OUT_DIR")
      if [ -z "$warmup_ckpt" ]; then
        echo "[error] warmup finished but no checkpoint found under $WARMUP_OUT_DIR/checkpoints" >&2
        exit 2
      fi
    fi

    CHECKPOINT="$warmup_ckpt"
  fi
fi

echo "[rl] base_ckpt=$BASE_CKPT init_ckpt=$CHECKPOINT dataset=$DATASET_DIR split=$SPLIT out=$OUT_DIR iters=$ITERS max_steps=$MAX_STEPS num_rollouts=$NUM_ROLLOUTS spp=$SAMPLES_PER_PROMPT min_pos=$MIN_POSITIVE warmup_steps=$WARMUP_STEPS"

for ((iter=0; iter<ITERS; iter++)); do
  cur_ckpt=$(latest_ckpt)
  if [ -z "$cur_ckpt" ]; then
    rollout_ckpt="$CHECKPOINT"
    cur_step=0
  else
    rollout_ckpt="$cur_ckpt"
    cur_step=$(ckpt_step "$cur_ckpt")
  fi

  if [ "$cur_step" -ge "$MAX_STEPS" ]; then
    echo "[loop] reached max_steps=$MAX_STEPS (cur_step=$cur_step); stop" >&2
    break
  fi

  next_step=$((cur_step + STEPS_PER_ITER))
  if [ "$next_step" -gt "$MAX_STEPS" ]; then
    next_step="$MAX_STEPS"
  fi

  iter_tag=$(printf "%03d" "$iter")
  iter_buffer="$BUFFER_DIR/iter_${iter_tag}.jsonl"
  if [ "$CLEAN_BUFFERS" = "1" ]; then
    rm -f "$iter_buffer" 2>/dev/null || true
  fi

  rollout_seed=$((SEED + iter * 10007))
  echo "[loop] iter=$iter_tag rollout_ckpt=$rollout_ckpt steps=${cur_step}->${next_step} buffer=$iter_buffer"
  "$PY" -m mlx_train.rl_gsm8k.rollout \
    --dataset_dir "$DATASET_DIR" \
    --split "$SPLIT" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --checkpoint "$rollout_ckpt" \
    --out_buffer "$iter_buffer" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --samples_per_prompt "$SAMPLES_PER_PROMPT" \
    --min_positive "$MIN_POSITIVE" \
    --max_total_rollouts "$MAX_TOTAL_ROLLOUTS" \
    --seed "$rollout_seed" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_new_tokens "$MAX_NEW_TOKENS"

  if [ -z "$cur_ckpt" ]; then
    echo "[train] iter=$iter_tag init_from=$CHECKPOINT out=$OUT_DIR buffer=$iter_buffer max_steps=$next_step"
    "$PY" -m mlx_train.rl_gsm8k.train \
      --tokenizer_path "$TOKENIZER_PATH" \
      --buffer_path "$iter_buffer" \
      --init_from "$CHECKPOINT" \
      --out_dir "$OUT_DIR" \
      --seq_len "$SEQ_LEN" \
      --batch_size "$BATCH_SIZE" \
      --max_steps "$next_step" \
      "$@"
  else
    echo "[train] iter=$iter_tag resume=$cur_ckpt out=$OUT_DIR buffer=$iter_buffer max_steps=$next_step"
    "$PY" -m mlx_train.rl_gsm8k.train \
      --tokenizer_path "$TOKENIZER_PATH" \
      --buffer_path "$iter_buffer" \
      --resume "$cur_ckpt" \
      --out_dir "$OUT_DIR" \
      --seq_len "$SEQ_LEN" \
      --batch_size "$BATCH_SIZE" \
      --max_steps "$next_step" \
      "$@"
  fi
done

RUN_EVAL=${RUN_EVAL:-1}
RUN_BENCH=${RUN_BENCH:-1}

LATEST_CKPT=$(latest_ckpt)
if [ -z "$LATEST_CKPT" ] || [ ! -d "$LATEST_CKPT" ]; then
  echo "[warn] no checkpoint found under $OUT_DIR/checkpoints; skip eval/bench" >&2
  exit 0
fi

if [ "$RUN_EVAL" = "1" ]; then
  GSM8K_EVAL_SPLIT=${GSM8K_EVAL_SPLIT:-test}
  GSM8K_EVAL_NUM=${GSM8K_EVAL_NUM:-200} # 0 = all
  GSM8K_EVAL_BUFFER=${GSM8K_EVAL_BUFFER:-$OUT_DIR/eval_gsm8k_${GSM8K_EVAL_SPLIT}.jsonl}

  echo "[eval] gsm8k split=$GSM8K_EVAL_SPLIT num=$GSM8K_EVAL_NUM ckpt=$LATEST_CKPT"
  "$PY" -m mlx_train.rl_gsm8k.rollout \
    --dataset_dir "$DATASET_DIR" \
    --split "$GSM8K_EVAL_SPLIT" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --checkpoint "$LATEST_CKPT" \
    --out_buffer "$GSM8K_EVAL_BUFFER" \
    --num_rollouts "$GSM8K_EVAL_NUM" \
    --seed "$SEED" \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --log_every 50
fi

if [ "$RUN_BENCH" = "1" ]; then
  BENCH_SUITE=${BENCH_SUITE:-all}
  BENCH_MAX_NEW_TOKENS=${BENCH_MAX_NEW_TOKENS:-32}

  echo "[bench] suite=$BENCH_SUITE ckpt=$LATEST_CKPT"
  "$PY" -m mlx_train.bench \
    --checkpoint "$LATEST_CKPT" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --suite "$BENCH_SUITE" \
    --seed "$SEED" \
    --max_new_tokens "$BENCH_MAX_NEW_TOKENS" \
    --no_ollama
fi
