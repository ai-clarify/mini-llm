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

if ! "$PY" -c "import requests" >/dev/null 2>&1; then
  echo "[error] Python deps missing. Install: pip install -r mlx_train/requirements.txt" >&2
  exit 1
fi

# [tb] auto-start
TB_AUTO=${TB_AUTO:-1}
TB_PORT=${TB_PORT:-6006}
TB_HOST=${TB_HOST:-127.0.0.1}

find_tensorboard_dir() {
  local tb_dir=""
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --tensorboard_dir)
        tb_dir="${args[$((i+1))]:-}"
        i=$((i+1))
        ;;
      --tensorboard_dir=*)
        tb_dir="${args[$i]#*=}"
        ;;
    esac
  done
  echo "$tb_dir"
}

TB_LOGDIR=$(find_tensorboard_dir "$@")
if [ -n "$TB_LOGDIR" ] && [ "$TB_AUTO" != "0" ]; then
  if "$PY" -m tensorboard --version >/dev/null 2>&1; then
    mkdir -p "$TB_LOGDIR"
    TB_PORT_IN_USE=$("$PY" - "$TB_HOST" "$TB_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
    print("0")
except OSError:
    print("1")
finally:
    sock.close()
PY
)
    if [ "$TB_PORT_IN_USE" = "1" ]; then
      TB_PORT=$("$PY" - "$TB_HOST" <<'PY'
import socket
import sys

host = sys.argv[1]
sock = socket.socket()
sock.bind((host, 0))
port = sock.getsockname()[1]
sock.close()
print(port)
PY
)
    fi
    "$PY" -m tensorboard --logdir "$TB_LOGDIR" --host "$TB_HOST" --port "$TB_PORT" >/dev/null 2>&1 &
    echo "[tensorboard] http://$TB_HOST:$TB_PORT (pid=$!)"
  else
    echo "[warn] TensorBoard not available; install tensorboard to enable TB_AUTO" >&2
  fi
fi

OLLAMA_URL=${OLLAMA_URL:-http://127.0.0.1:11434}
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3:0.6b}

DATA_JSONL=${DATA_JSONL:-out/distill_ollama_qwen3_0.6b/synth.jsonl}
OUT_DIR=${OUT_DIR:-out/mlx_distill/qwen3_0.6b_sft}

# Cold-start: generate some data first, then start training + background generation.
COLD_SAMPLES=${COLD_SAMPLES:-512}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-20000}   # 0 = keep generating forever
GEN_WORKERS=${GEN_WORKERS:-8}

# Training defaults (override via env vars).
PRESET=${PRESET:-200mb}
DTYPE=${DTYPE:-bfloat16}
SEQ_LEN=${SEQ_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-2}
ACCUM_STEPS=${ACCUM_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}
LOG_INTERVAL=${LOG_INTERVAL:-10}
KEEP_LAST_CHECKPOINTS=${KEEP_LAST_CHECKPOINTS:-3}
INIT_FROM=${INIT_FROM:-}

mkdir -p "$(dirname "$DATA_JSONL")"
mkdir -p "$OUT_DIR"

if command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "[error] Ollama not reachable at $OLLAMA_URL (expected ollama serve running)" >&2
    exit 1
  fi
fi

echo "[distill] ollama=$OLLAMA_URL model=$OLLAMA_MODEL"
echo "[distill] data=$DATA_JSONL out=$OUT_DIR"
echo "[distill] cold_samples=$COLD_SAMPLES total_samples=$TOTAL_SAMPLES gen_workers=$GEN_WORKERS"

if [ "${COLD_SAMPLES}" -gt 0 ]; then
  echo "[distill] cold-generate ${COLD_SAMPLES} samples..."
  "$PY" -m mlx_train.distill_data_ollama \
    --out_jsonl "$DATA_JSONL" \
    --ollama_url "$OLLAMA_URL" \
    --ollama_model "$OLLAMA_MODEL" \
    --num_workers "$GEN_WORKERS" \
    --cold_min_samples "$COLD_SAMPLES"
fi

echo "[distill] start background generation..."
"$PY" -m mlx_train.distill_data_ollama \
  --out_jsonl "$DATA_JSONL" \
  --ollama_url "$OLLAMA_URL" \
  --ollama_model "$OLLAMA_MODEL" \
  --num_workers "$GEN_WORKERS" \
  --target_total_samples "$TOTAL_SAMPLES" &
GEN_PID=$!

cleanup() {
  if kill -0 "$GEN_PID" >/dev/null 2>&1; then
    echo "[distill] stopping generator (pid=$GEN_PID)..."
    kill "$GEN_PID" >/dev/null 2>&1 || true
    wait "$GEN_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

TRAIN_ARGS=(
  --data_path "$DATA_JSONL"
  --task sft
  --preset "$PRESET"
  --dtype "$DTYPE"
  --seq_len "$SEQ_LEN"
  --batch_size "$BATCH_SIZE"
  --accum_steps "$ACCUM_STEPS"
  --epochs 1000000
  --max_steps "$MAX_STEPS"
  --save_interval "$SAVE_INTERVAL"
  --log_interval "$LOG_INTERVAL"
  --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
  --out_dir "$OUT_DIR"
)
if [ -n "$INIT_FROM" ]; then
  TRAIN_ARGS+=(--init_from "$INIT_FROM")
fi

echo "[train] $PY -m mlx_train.train ${TRAIN_ARGS[*]}"
"$PY" -m mlx_train.train "${TRAIN_ARGS[@]}" "$@"
