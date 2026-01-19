#!/usr/bin/env bash
set -euo pipefail

on_interrupt() {
  echo
  echo "[abort] Interrupted (SIGINT)."
  exit 130
}

trap on_interrupt INT

usage() {
  cat <<'USAGE'
Usage: scripts/run_mlx.sh [OPTIONS]

Runs a one-click MLX pipeline (pretrain -> SFT -> infer) aligned with MiniMind datasets.

Options:
  --smoke-test       Tiny fast run (downloads minimind:smoke, runs infer).
  --download-only    Only download datasets, then exit.
  --infer-only       Skip training; run inference using the latest checkpoint (OUT/, or OUT/{sft,pretrain}/).
  --infer-demo       Alias of `--infer-only` (kept for backward compatibility).
  --infer-checkpoint PATH
                    Skip training; run inference using the specified checkpoint dir (or model.safetensors file).
  --skip-pretrain    Skip pretrain stage (requires existing checkpoint or SFT_FROM).
  --skip-sft         Skip SFT stage.
  --skip-infer       Skip final inference.
  --run-dpo          Run DPO after SFT using dpo.jsonl.
  --run-r1           Run R1 reasoning SFT after SFT using r1_mix_1024.jsonl.
  --                Forward remaining args to mlx_train.train (pretrain & sft).
  -h, --help         Show this help message and exit.

Environment overrides:
  PY                Python interpreter to use (default: auto-detect 3.10-3.12)
  VENV              Virtualenv directory (default: .venv_mlx)
  UV                Use uv to create/install venv (default: auto)
  OUT               Output root (default: out/mlx; smoke-test: out/mlx_smoke)
  DATA              Dataset cache dir (default: dataset/minimind)
  SMALL             Use smaller SFT dataset; pretrain stays full (default: 1)
  PRE_DATA          Dataset spec for pretrain (default: minimind:auto)
  SFT_DATA          Dataset spec for SFT (default: minimind:sft_mini_512.jsonl)
  R1_DATA           Dataset spec for R1 stage (default: minimind:r1_mix_1024.jsonl)
  AUTO_DL           Auto-download datasets when needed (default: 1; auto-disabled when --skip-pretrain is set)
  MINIMIND_DATA_SOURCE
                    Dataset source for minimind:* specs (modelscope|hf; default: modelscope)
  MINIMIND_DATA_REPO
                    Dataset repo id for minimind:* specs (default: gongjy/minimind_dataset)
  MINIMIND_MS_CACHE ModelScope cache dir (default: ~/.cache/modelscope)
  HF_EP             Optional HuggingFace mirror endpoint (only when MINIMIND_DATA_SOURCE=hf)
  DL_MAX            Per-file download guard in MB (default: 2048; set 0 to disable).
                    When PRE_DATA is large and DL_MAX is not set, defaults to 0.
  DPO_DL            Download DPO dataset too (default: 0).
  KEEP_LAST         Keep last N checkpoints per stage (default: 3)
  SMOKE_CLEAN       Auto-delete smoke-test outputs (default: 1)
  DPO               Enable DPO stage (default: 0)
  DPO_DATA          Dataset spec for DPO stage (default: minimind:dpo.jsonl)
  DPO_OUT           Output dir for DPO stage (default: OUT/dpo)
  DPO_FROM          Checkpoint to init DPO policy (default: latest SFT checkpoint)
  DPO_REF           Reference checkpoint for DPO (default: DPO_FROM)
  DPO_LEN           Sequence length for DPO stage (default: 512; smoke: 256)
  DPO_BS            Batch size for DPO pairs (default: 1; smoke: 2)
  DPO_ACCUM         Grad accumulation steps for DPO stage (default: 8; smoke: 1)
  DPO_EPOCH         Epochs for DPO stage (default: 1)
  DPO_MAX           Optional max steps for DPO stage
  DPO_BETA          DPO beta (default: 0.1)
  R1                Enable R1 reasoning stage (default: 0)
  R1_OUT            Output dir for R1 stage (default: OUT/r1)
  R1_FROM           Checkpoint to init R1 stage (default: latest SFT checkpoint)
  R1_LEN            Sequence length for R1 stage (default: 1024; smoke: 256)
  R1_BS             Batch size for R1 stage (default: 1; smoke: 2)
  R1_ACCUM          Grad accumulation steps for R1 stage (default: 8; smoke: 1)
  R1_EPOCH          Epochs for R1 stage (default: 1)
  R1_MAX            Optional max steps for R1 stage
  R1_TOKEN_WEIGHT   Loss weight multiplier for reasoning special tokens (default: 10)
  R1_TOKENS         Comma-separated R1 special tokens (default: <think>,</think>,<answer>,</answer>)

Model/training overrides:
  PRESET             Model preset: 200mb|tiny|custom (default: custom)
  DTYPE              float16|bfloat16|float32 (default: bfloat16)

  PRE_LEN, PRE_BS, PRE_ACCUM, PRE_EPOCH, PRE_MAX
  SFT_LEN, SFT_BS, SFT_ACCUM, SFT_EPOCH, SFT_MAX

Advanced:
  SFT_FROM           Checkpoint dir or model.safetensors to init SFT from (overrides auto-detect).
  INF_PROMPT         Prompt used by non-demo inference (default: hi)
  INF_MAX_NEW        Max new tokens for inference (default: 512; smoke-test: 64)
  INF_MIN_NEW        Force at least N new tokens (default: 1)
  INF_TEMP           0 for greedy; >0 for sampling (default: 0)
  INF_TOP_P          Nucleus sampling threshold (default: 1.0)
  INF_MAX_SEQ        Max total tokens (prompt + generation) for infer (default: use checkpoint seq_len if available)
  TF_DIR             TensorBoard log root for MLX training (default: out/logs/<out_dir_basename>; set empty to disable)
  TB_AUTO            Auto-start TensorBoard when training (default: 1; set 0 to disable)
  TB_HOST            TensorBoard host (default: 127.0.0.1)
  TB_PORT            TensorBoard port (default: 6006)
  INF_MODE           Demo mode for --infer-only (default: knowledge; other: bench)
  INF_SUITES         [bench mode] Suites (default: copy,json,sort,math_mcq,logic,qa,knowledge)
  INF_N              [bench mode] Examples per suite (default: 2)
  INF_NO_CHAT        [bench mode] Set to 1 to skip the open-ended chat prompt (default: 0)
  GATE               Enable gated attention in training (1=on,0=off; default: unset/preset).
  GATE_INIT          Gate init logit for training (default: 4.0; sigmoid(init) is multiplier).
USAGE
}

SMOKE_TEST=0
DOWNLOAD_ONLY=0
INFER_ONLY=0
INFER_DEMO=0
INFER_CHECKPOINT=""
SKIP_PRETRAIN=0
SKIP_SFT=0
SKIP_INFER=0
RUN_R1=${R1:-0}
RUN_DPO=${DPO:-0}
OUT_DIR_WAS_SET=0
if [ -n "${OUT+x}" ]; then
  OUT_DIR_WAS_SET=1
fi
TF_DIR_WAS_SET=0
if [ -n "${TF_DIR+x}" ]; then
  TF_DIR_WAS_SET=1
fi

TRAIN_EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --smoke-test) SMOKE_TEST=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --infer-only) INFER_ONLY=1; INFER_DEMO=1; SKIP_PRETRAIN=1; SKIP_SFT=1; shift ;;
    --infer-demo) INFER_ONLY=1; INFER_DEMO=1; SKIP_PRETRAIN=1; SKIP_SFT=1; shift ;;
    --infer-checkpoint)
      if [ $# -lt 2 ]; then
        echo "[error] --infer-checkpoint requires a path" >&2
        exit 2
      fi
      INFER_ONLY=1
      INFER_CHECKPOINT=$2
      SKIP_PRETRAIN=1
      SKIP_SFT=1
      shift 2
      ;;
    --skip-pretrain) SKIP_PRETRAIN=1; shift ;;
    --skip-sft) SKIP_SFT=1; shift ;;
    --skip-infer) SKIP_INFER=1; shift ;;
    --run-dpo) RUN_DPO=1; shift ;;
    --run-r1) RUN_R1=1; shift ;;
    --) shift; TRAIN_EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) TRAIN_EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Mirrors (consistent with scripts/run.sh)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY || true
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export UV_INDEX_URL=${UV_INDEX_URL:-$PIP_INDEX_URL}

# Silence transformers advisory warning in MLX-only envs (no torch/tf/flax).
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

VENV_DIR=${VENV:-.venv_mlx}
DATA_DIR=${DATA:-dataset/minimind}
MINIMIND_DATA_SOURCE=${MINIMIND_DATA_SOURCE:-modelscope}
MINIMIND_DATA_REPO=${MINIMIND_DATA_REPO:-gongjy/minimind_dataset}
MINIMIND_MS_CACHE=${MINIMIND_MS_CACHE:-$HOME/.cache/modelscope}
export MINIMIND_DATA_SOURCE MINIMIND_DATA_REPO MINIMIND_MS_CACHE
MLX_SMALL_DATA=${SMALL:-1}
AUTO_DOWNLOAD_SET=0
if [ -n "${AUTO_DL+x}" ]; then
  AUTO_DOWNLOAD_SET=1
fi
AUTO_DOWNLOAD=${AUTO_DL:-1}
MAX_DOWNLOAD_MB_SET=0
if [ -n "${DL_MAX+x}" ]; then
  MAX_DOWNLOAD_MB_SET=1
fi
MAX_DOWNLOAD_MB=${DL_MAX:-2048}
DOWNLOAD_DPO=${DPO_DL:-0}
KEEP_LAST_CHECKPOINTS=${KEEP_LAST:-3}
HF_ENDPOINT=${HF_EP:-}
if [ -n "$HF_ENDPOINT" ]; then
  HF_ENDPOINT=${HF_ENDPOINT%/}
  if [[ "$HF_ENDPOINT" != http://* && "$HF_ENDPOINT" != https://* ]]; then
    HF_ENDPOINT="https://$HF_ENDPOINT"
  fi
else
  unset HF_ENDPOINT
fi
R1_OUT_DIR=${R1_OUT:-}
R1_INIT_FROM=${R1_FROM:-}
DPO_OUT_DIR=${DPO_OUT:-}
DPO_INIT_FROM=${DPO_FROM:-}
DPO_REF_FROM=${DPO_REF:-}
if [ "$SMOKE_TEST" -eq 1 ]; then
  OUT_DIR=${OUT:-out/mlx_smoke}
  PRESET=${PRESET:-tiny}
  DTYPE=${DTYPE:-float32}
  CLEANUP_SMOKE=${SMOKE_CLEAN:-1}
else
  OUT_DIR=${OUT:-out/mlx}
  PRESET=${PRESET:-custom}
  DTYPE=${DTYPE:-bfloat16}
  CLEANUP_SMOKE=${SMOKE_CLEAN:-0}
fi

if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
  AUTO_DOWNLOAD=1
elif [ "$AUTO_DOWNLOAD_SET" -eq 0 ] && [ "$SKIP_PRETRAIN" -eq 1 ] && [ "$INFER_ONLY" -eq 0 ]; then
  AUTO_DOWNLOAD=0
fi

PYTHON_CMD=${PY:-}
DEFAULT_PYTHON_VERSION=3.11

USE_UV=${UV:-auto}
if [ "$USE_UV" = "auto" ]; then
  if command -v uv >/dev/null 2>&1; then
    USE_UV=1
  else
    USE_UV=0
  fi
fi
if [ "$USE_UV" = "1" ] && ! command -v uv >/dev/null 2>&1; then
  echo "[env] UV=1 but uv not found; falling back to venv" >&2
  USE_UV=0
fi

check_python_version() {
  local python_cmd=$1
  if [ -z "$python_cmd" ]; then
    return 1
  fi
  if [ -x "$python_cmd" ]; then
    :
  elif ! command -v "$python_cmd" >/dev/null 2>&1; then
    return 1
  fi

  local py_version
  py_version=$("$python_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
  local major=${py_version%%.*}
  local minor=${py_version#*.}

  if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
    echo "$py_version"
    return 0
  fi
  return 1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  local bootstrap_py=""
  for py_candidate in python3 python; do
    if command -v "$py_candidate" >/dev/null 2>&1; then
      bootstrap_py=$py_candidate
      break
    fi
  done
  if [ -z "$bootstrap_py" ]; then
    return 1
  fi

  echo "[env] uv not found; bootstrapping via $bootstrap_py"
  "$bootstrap_py" -m ensurepip --upgrade >/dev/null 2>&1 || true
  if ! "$bootstrap_py" -m pip install --user -U uv; then
    return 1
  fi
  local user_base
  user_base=$("$bootstrap_py" -c 'import site; print(site.getuserbase())')
  export PATH="$user_base/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

PY_VERSION=""
PYTHON_CMD_IS_VERSION=0
if [ -n "$PYTHON_CMD" ]; then
  if PY_VERSION=$(check_python_version "$PYTHON_CMD"); then
    :
  else
    echo "[error] PY must be Python 3.10-3.12 (got: $PYTHON_CMD)" >&2
    exit 1
  fi
else
  for py_candidate in python3.11 python3.12 python3.10 python3; do
    if PY_VERSION=$(check_python_version "$py_candidate"); then
      PYTHON_CMD=$py_candidate
      break
    fi
  done
  if [ -z "$PYTHON_CMD" ]; then
    if ensure_uv; then
      USE_UV=1
      PYTHON_CMD=$DEFAULT_PYTHON_VERSION
      PY_VERSION=$DEFAULT_PYTHON_VERSION
      PYTHON_CMD_IS_VERSION=1
      echo "[env] No compatible Python found; uv will download Python $DEFAULT_PYTHON_VERSION"
    else
      echo "[error] No compatible Python version found (requires 3.10-3.12)" >&2
      echo "[error] Detected Python versions:" >&2
      for py_test in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$py_test" >/dev/null 2>&1; then
          "$py_test" --version >&2 || true
        fi
      done
      echo "[error] Install Python 3.11+ or install uv to auto-download a compatible Python." >&2
      exit 1
    fi
  fi
fi

if [ "$PYTHON_CMD_IS_VERSION" -eq 1 ]; then
  echo "[env] Using Python $PY_VERSION via uv"
else
  echo "[env] Using Python $PY_VERSION at $PYTHON_CMD"
fi

validate_venv() {
  local venv_path=$1
  if [ ! -d "$venv_path" ]; then
    return 1
  fi
  if [ ! -x "$venv_path/bin/python" ]; then
    return 1
  fi
  if ! "$venv_path/bin/python" -m pip --version >/dev/null 2>&1; then
    return 1
  fi

  local venv_py_version
  venv_py_version=$("$venv_path/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
  local major=${venv_py_version%%.*}
  local minor=${venv_py_version#*.}
  if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
    return 0
  fi
  return 1
}

if validate_venv "$VENV_DIR"; then
  VENV_PY_VERSION=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  echo "[env] Using existing virtual environment at $VENV_DIR (Python $VENV_PY_VERSION)"
else
  if [ -d "$VENV_DIR" ]; then
    echo "[env] Virtual environment at $VENV_DIR is broken or incompatible, removing it"
    rm -rf "$VENV_DIR"
  fi

  if [ "$USE_UV" = "1" ]; then
    echo "[env] Creating venv with uv at $VENV_DIR"
    if ! uv venv "$VENV_DIR" --python "$PYTHON_CMD" --seed; then
      if [ "$PYTHON_CMD_IS_VERSION" -eq 1 ]; then
        echo "[error] uv venv failed while downloading Python $PYTHON_CMD" >&2
        exit 1
      fi
      echo "[env] uv venv failed, falling back to venv" >&2
      rm -rf "$VENV_DIR"
      "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
  else
    echo "[env] Creating venv at $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
  fi
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[error] Python interpreter not found in $VENV_DIR after setup" >&2
  exit 1
fi

PY="$VENV_DIR/bin/python"
echo "[env] Using $PY"
if [ "$USE_UV" = "1" ]; then
  if ! uv pip install -r mlx_train/requirements.txt -p "$PY"; then
    echo "[env] uv pip install failed, falling back to pip" >&2
    "$PY" -m pip -q install --upgrade pip
    "$PY" -m pip -q install -r mlx_train/requirements.txt
  fi
else
  "$PY" -m pip -q install --upgrade pip
  "$PY" -m pip -q install -r mlx_train/requirements.txt
fi

if ! "$PY" -c "import mlx, transformers, huggingface_hub, requests, jinja2" >/dev/null 2>&1; then
  echo "[error] MLX deps not available after install" >&2
  exit 1
fi

abs_path() {
  local py_bin=${PY:-$PYTHON_CMD}
  "$py_bin" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

safe_rm_rf() {
  local target=$1
  if [ -z "$target" ]; then
    echo "[cleanup] Refusing to remove empty path" >&2
    exit 1
  fi
  local abs
  abs=$(abs_path "$target")
  local root_abs
  root_abs=$(abs_path "$ROOT_DIR")

  case "$abs" in
    "$root_abs"/*) ;;
    *) echo "[cleanup] Refusing to remove path outside repo: $abs" >&2; exit 1 ;;
  esac
  if [ "$abs" = "$root_abs" ] || [ "$abs" = "/" ]; then
    echo "[cleanup] Refusing to remove unsafe path: $abs" >&2
    exit 1
  fi
  rm -rf "$abs"
}

download_minimind() {
  local spec=$1
  local task=$2
  "$PY" - <<PY
import os
from mlx_train.download import resolve_data_path_spec

print(
    resolve_data_path_spec(
        "${spec}",
        task="${task}",
        data_dir=os.environ.get("DATA_DIR", "dataset/minimind"),
        hf_repo_id=os.environ.get("MINIMIND_DATA_REPO", "gongjy/minimind_dataset"),
        hf_endpoint=os.environ.get("HF_ENDPOINT"),
        force_download=False,
        max_download_mb=int(os.environ.get("MAX_DOWNLOAD_MB", "2048")),
        data_source=os.environ.get("MINIMIND_DATA_SOURCE", "modelscope"),
        ms_cache_dir=os.environ.get("MINIMIND_MS_CACHE"),
    )
)
PY
}

if [ "$SMOKE_TEST" -eq 1 ] && [ "$CLEANUP_SMOKE" = "1" ] && [ "$OUT_DIR_WAS_SET" -eq 0 ]; then
  if [ -d "$OUT_DIR" ]; then
    echo "[cleanup] Removing previous smoke outputs: $OUT_DIR"
    safe_rm_rf "$OUT_DIR"
  fi
  if [ -n "${TF_DIR:-}" ] && [ "$TF_DIR_WAS_SET" -eq 0 ] && [ -d "$TF_DIR" ]; then
    echo "[cleanup] Removing previous smoke logs: $TF_DIR"
    safe_rm_rf "$TF_DIR"
  fi
fi

mkdir -p "$DATA_DIR"

if [ "$SMOKE_TEST" -eq 1 ]; then
  PRETRAIN_DATA_SPEC=${PRE_DATA:-minimind:smoke}
  SFT_DATA_SPEC=${SFT_DATA:-minimind:smoke}
  DPO_DATA_SPEC=${DPO_DATA:-minimind:dpo.jsonl}
  R1_DATA_SPEC=${R1_DATA:-minimind:smoke}
else
  PRETRAIN_DATA_SPEC=${PRE_DATA:-minimind:auto}
  if [ "$MLX_SMALL_DATA" = "1" ]; then
    SFT_DATA_SPEC=${SFT_DATA:-minimind:sft_mini_512.jsonl}
  else
    SFT_DATA_SPEC=${SFT_DATA:-minimind:sft_512.jsonl}
  fi
  DPO_DATA_SPEC=${DPO_DATA:-minimind:dpo.jsonl}
  R1_DATA_SPEC=${R1_DATA:-minimind:r1_mix_1024.jsonl}
fi

if [ "$MAX_DOWNLOAD_MB_SET" -eq 0 ]; then
  case "$PRETRAIN_DATA_SPEC" in
    *minimind:auto*|*minimind:pretrain_hq*|*minimind:pretrain*)
      MAX_DOWNLOAD_MB=0
      ;;
  esac
fi

if [ "$INFER_ONLY" -eq 1 ]; then
  echo "[data] Skipping dataset download (--infer-only)"
elif [ "$AUTO_DOWNLOAD" -eq 0 ]; then
  echo "[data] Auto download disabled (AUTO_DL=0). Ensure datasets already exist."
else
  export DATA_DIR MAX_DOWNLOAD_MB MINIMIND_DATA_SOURCE MINIMIND_DATA_REPO MINIMIND_MS_CACHE
  if [ -n "${HF_ENDPOINT:-}" ]; then
    export HF_ENDPOINT
  else
    unset HF_ENDPOINT
  fi
  echo "[data] Download required datasets"
  if [ "$SKIP_PRETRAIN" -eq 0 ]; then
    download_minimind "$PRETRAIN_DATA_SPEC" "pretrain"
  fi
  if [ "$SKIP_SFT" -eq 0 ]; then
    download_minimind "$SFT_DATA_SPEC" "sft"
  fi
  if [ "$RUN_DPO" -eq 1 ] || [ "$DOWNLOAD_DPO" = "1" ]; then
    download_minimind "$DPO_DATA_SPEC" "dpo"
  fi
  if [ "$RUN_R1" -eq 1 ]; then
    download_minimind "$R1_DATA_SPEC" "r1"
  fi
fi

if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
  echo "[done] Download complete."
  exit 0
fi

mkdir -p "$OUT_DIR"

if [ -z "${TF_DIR+x}" ]; then
  TF_DIR="out/logs/$(basename "$OUT_DIR")"
fi
if [ -n "$TF_DIR" ]; then
  TB_PRETRAIN_DIR="$TF_DIR/pretrain"
  TB_SFT_DIR="$TF_DIR/sft"
  TB_DPO_DIR="$TF_DIR/dpo"
  TB_R1_DIR="$TF_DIR/r1"
  mkdir -p "$TB_PRETRAIN_DIR" "$TB_SFT_DIR" "$TB_DPO_DIR" "$TB_R1_DIR"
fi

# [tb] auto-start
TB_AUTO=${TB_AUTO:-1}
TB_PORT=${TB_PORT:-6006}
TB_HOST=${TB_HOST:-127.0.0.1}

WILL_TRAIN=0
if [ "$SKIP_PRETRAIN" -eq 0 ] || [ "$SKIP_SFT" -eq 0 ] || [ "$RUN_DPO" -eq 1 ] || [ "$RUN_R1" -eq 1 ]; then
  WILL_TRAIN=1
fi

if [ -n "${TF_DIR:-}" ] && [ "$TB_AUTO" != "0" ] && [ "$WILL_TRAIN" -eq 1 ]; then
  if "$PY" -m tensorboard --version >/dev/null 2>&1; then
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
    "$PY" -m tensorboard --logdir "$TF_DIR" --host "$TB_HOST" --port "$TB_PORT" >/dev/null 2>&1 &
    echo "[tensorboard] http://$TB_HOST:$TB_PORT (pid=$!)"
  else
    echo "[warn] TensorBoard not available; install tensorboard to enable TB_AUTO" >&2
  fi
fi

latest_ckpt() {
  local stage_dir=$1
  local ckpt
  while IFS= read -r ckpt; do
    [ -z "$ckpt" ] && continue
    if is_valid_ckpt "$ckpt"; then
      echo "$ckpt"
      return 0
    fi
    echo "[warn] Skipping invalid checkpoint: $ckpt" >&2
  done < <(ls -dt "$stage_dir"/checkpoints/step_* 2>/dev/null || true)
  return 0
}

is_valid_ckpt() {
  local ckpt_path=$1
  [ -f "$ckpt_path/model.safetensors" ] || return 1
  [ -s "$ckpt_path/model.safetensors" ] || return 1
  [ -f "$ckpt_path/config.json" ] || return 1
  [ -s "$ckpt_path/config.json" ] || return 1
  [ -f "$ckpt_path/state.json" ] || return 1
  [ -s "$ckpt_path/state.json" ] || return 1
  return 0
}

ckpt_step_num() {
  local ckpt_path=$1
  local base
  base=$(basename "$ckpt_path")
  if [[ "$base" =~ ^step_([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  echo ""
  return 1
}

smoke_bump_max_steps() {
  local resume_path=$1
  local current_max=$2
  local extra=${3:-5}

  if [ -z "$resume_path" ] || [ -z "$current_max" ]; then
    echo "$current_max"
    return 0
  fi

  local step
  step=$(ckpt_step_num "$resume_path" || true)
  if [ -z "$step" ]; then
    echo "$current_max"
    return 0
  fi

  local step_num
  step_num=$((10#$step))
  if [ "$step_num" -ge "$current_max" ]; then
    echo $((step_num + extra))
    return 0
  fi

  echo "$current_max"
}

run_stage() {
  local stage=$1
  shift
  echo
  echo "[stage] $stage"
  echo "$PY -m mlx_train.train $*"
  "$PY" -m mlx_train.train "$@"
}

PRETRAIN_OUT="$OUT_DIR/pretrain"
SFT_OUT="$OUT_DIR/sft"
DPO_OUT=${DPO_OUT_DIR:-"$OUT_DIR/dpo"}
R1_OUT=${R1_OUT_DIR:-"$OUT_DIR/r1"}

if [ "$SMOKE_TEST" -eq 1 ]; then
  PRETRAIN_SEQ_LEN=${PRE_LEN:-256}
  PRETRAIN_BATCH_SIZE=${PRE_BS:-2}
  PRETRAIN_ACCUM_STEPS=${PRE_ACCUM:-1}
  PRETRAIN_EPOCHS=${PRE_EPOCH:-1}
  PRETRAIN_MAX_STEPS=${PRE_MAX:-5}

  SFT_SEQ_LEN=${SFT_LEN:-256}
  SFT_BATCH_SIZE=${SFT_BS:-2}
  SFT_ACCUM_STEPS=${SFT_ACCUM:-1}
  SFT_EPOCHS=${SFT_EPOCH:-1}
  SFT_MAX_STEPS=${SFT_MAX:-5}

  DPO_SEQ_LEN=${DPO_LEN:-256}
  DPO_BATCH_SIZE=${DPO_BS:-2}
  DPO_ACCUM_STEPS=${DPO_ACCUM:-1}
  DPO_EPOCHS=${DPO_EPOCH:-1}
  DPO_MAX_STEPS=${DPO_MAX:-5}

  LOG_INTERVAL=${LOG_INTERVAL:-1}
  SAVE_INTERVAL=${SAVE_INTERVAL:-2}
else
  PRETRAIN_SEQ_LEN=${PRE_LEN:-1024}
  PRETRAIN_BATCH_SIZE=${PRE_BS:-1}
  PRETRAIN_ACCUM_STEPS=${PRE_ACCUM:-8}
  PRETRAIN_EPOCHS=${PRE_EPOCH:-1}
  PRETRAIN_MAX_STEPS=${PRE_MAX:-}

  SFT_SEQ_LEN=${SFT_LEN:-512}
  SFT_BATCH_SIZE=${SFT_BS:-1}
  SFT_ACCUM_STEPS=${SFT_ACCUM:-8}
  SFT_EPOCHS=${SFT_EPOCH:-1}
  SFT_MAX_STEPS=${SFT_MAX:-}

  DPO_SEQ_LEN=${DPO_LEN:-512}
  DPO_BATCH_SIZE=${DPO_BS:-1}
  DPO_ACCUM_STEPS=${DPO_ACCUM:-8}
  DPO_EPOCHS=${DPO_EPOCH:-1}
  DPO_MAX_STEPS=${DPO_MAX:-}

  LOG_INTERVAL=${LOG_INTERVAL:-10}
  SAVE_INTERVAL=${SAVE_INTERVAL:-200}
fi

ATTN_GATE=${GATE:-}
ATTN_GATE_INIT=${GATE_INIT:-}

if [ "$SKIP_PRETRAIN" -eq 0 ]; then
  PRETRAIN_RESUME=$(latest_ckpt "$PRETRAIN_OUT")
  if [ "$SMOKE_TEST" -eq 1 ] && [ -n "$PRETRAIN_MAX_STEPS" ]; then
    PRETRAIN_MAX_STEPS=$(smoke_bump_max_steps "$PRETRAIN_RESUME" "$PRETRAIN_MAX_STEPS" "${SMOKE_EXTRA_STEPS:-5}")
  fi
  PRETRAIN_ARGS=(
    --task pretrain
    --preset "$PRESET"
    --dtype "$DTYPE"
    --data_path "$PRETRAIN_DATA_SPEC"
    --data_dir "$DATA_DIR"
    --max_download_mb "$MAX_DOWNLOAD_MB"
    --out_dir "$PRETRAIN_OUT"
    --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
    --seq_len "$PRETRAIN_SEQ_LEN"
    --batch_size "$PRETRAIN_BATCH_SIZE"
    --accum_steps "$PRETRAIN_ACCUM_STEPS"
    --epochs "$PRETRAIN_EPOCHS"
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
  )
  if [ -n "${TB_PRETRAIN_DIR:-}" ]; then
    PRETRAIN_ARGS+=(--tensorboard_dir "$TB_PRETRAIN_DIR")
  fi
  if [ -n "${HF_ENDPOINT:-}" ]; then
    PRETRAIN_ARGS+=(--hf_endpoint "$HF_ENDPOINT")
  fi
  if [ -n "${ATTN_GATE:-}" ]; then
    if [ "$ATTN_GATE" = "1" ]; then
      PRETRAIN_ARGS+=(--attn_gate)
    elif [ "$ATTN_GATE" = "0" ]; then
      PRETRAIN_ARGS+=(--no-attn_gate)
    else
      echo "[warn] Unknown ATTN_GATE=$ATTN_GATE (expected 1 or 0); ignoring." >&2
    fi
    if [ -n "${ATTN_GATE_INIT:-}" ]; then
      PRETRAIN_ARGS+=(--attn_gate_init "$ATTN_GATE_INIT")
    fi
  fi
  if [ -n "$PRETRAIN_MAX_STEPS" ]; then
    PRETRAIN_ARGS+=(--max_steps "$PRETRAIN_MAX_STEPS")
  fi
  if [ -n "$PRETRAIN_RESUME" ]; then
    PRETRAIN_ARGS+=(--resume "$PRETRAIN_RESUME")
  fi
  if [ "${#TRAIN_EXTRA_ARGS[@]}" -gt 0 ]; then
    PRETRAIN_ARGS+=("${TRAIN_EXTRA_ARGS[@]}")
  fi
  run_stage "pretrain" "${PRETRAIN_ARGS[@]}"
fi

if [ "$SKIP_SFT" -eq 0 ]; then
  SFT_RESUME=$(latest_ckpt "$SFT_OUT")
  if [ "$SMOKE_TEST" -eq 1 ] && [ -n "$SFT_MAX_STEPS" ]; then
    SFT_MAX_STEPS=$(smoke_bump_max_steps "$SFT_RESUME" "$SFT_MAX_STEPS" "${SMOKE_EXTRA_STEPS:-5}")
  fi
  INIT_FROM=${SFT_FROM:-}
  if [ -z "$INIT_FROM" ]; then
    INIT_FROM=$(latest_ckpt "$PRETRAIN_OUT")
  fi

  SFT_ARGS=(
    --task sft
    --preset "$PRESET"
    --dtype "$DTYPE"
    --data_path "$SFT_DATA_SPEC"
    --data_dir "$DATA_DIR"
    --max_download_mb "$MAX_DOWNLOAD_MB"
    --out_dir "$SFT_OUT"
    --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
    --seq_len "$SFT_SEQ_LEN"
    --batch_size "$SFT_BATCH_SIZE"
    --accum_steps "$SFT_ACCUM_STEPS"
    --epochs "$SFT_EPOCHS"
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
  )
  if [ -n "${TB_SFT_DIR:-}" ]; then
    SFT_ARGS+=(--tensorboard_dir "$TB_SFT_DIR")
  fi
  if [ -n "${HF_ENDPOINT:-}" ]; then
    SFT_ARGS+=(--hf_endpoint "$HF_ENDPOINT")
  fi
  if [ -n "${ATTN_GATE:-}" ]; then
    if [ "$ATTN_GATE" = "1" ]; then
      SFT_ARGS+=(--attn_gate)
    elif [ "$ATTN_GATE" = "0" ]; then
      SFT_ARGS+=(--no-attn_gate)
    else
      echo "[warn] Unknown ATTN_GATE=$ATTN_GATE (expected 1 or 0); ignoring." >&2
    fi
    if [ -n "${ATTN_GATE_INIT:-}" ]; then
      SFT_ARGS+=(--attn_gate_init "$ATTN_GATE_INIT")
    fi
  fi
  if [ -n "$SFT_MAX_STEPS" ]; then
    SFT_ARGS+=(--max_steps "$SFT_MAX_STEPS")
  fi

  if [ -n "$SFT_RESUME" ]; then
    SFT_ARGS+=(--resume "$SFT_RESUME")
  else
    if [ -z "$INIT_FROM" ]; then
      echo "[error] No pretrain checkpoint found for SFT init; run without --skip-pretrain or set SFT_FROM" >&2
      exit 1
    fi
    SFT_ARGS+=(--init_from "$INIT_FROM")
  fi

  if [ "${#TRAIN_EXTRA_ARGS[@]}" -gt 0 ]; then
    SFT_ARGS+=("${TRAIN_EXTRA_ARGS[@]}")
  fi
  run_stage "sft" "${SFT_ARGS[@]}"
fi

if [ "$RUN_DPO" -eq 1 ]; then
  DPO_RESUME=$(latest_ckpt "$DPO_OUT")
  DPO_INIT_FROM=${DPO_INIT_FROM:-}
  if [ -z "$DPO_INIT_FROM" ]; then
    DPO_INIT_FROM=$(latest_ckpt "$SFT_OUT")
  fi
  if [ -z "$DPO_RESUME" ] && [ -z "$DPO_INIT_FROM" ]; then
    echo "[error] DPO requires a SFT checkpoint (set DPO_FROM or run SFT first)" >&2
    exit 1
  fi

  DPO_REF_FROM=${DPO_REF_FROM:-}
  if [ -z "$DPO_REF_FROM" ] && [ -z "$DPO_RESUME" ]; then
    DPO_REF_FROM="$DPO_INIT_FROM"
  fi
  if [ -z "$DPO_REF_FROM" ] && [ -z "$DPO_RESUME" ]; then
    echo "[error] DPO requires a reference checkpoint (set DPO_REF or DPO_FROM)" >&2
    exit 1
  fi

  DPO_ARGS=(
    --task dpo
    --preset "$PRESET"
    --dtype "$DTYPE"
    --data_path "$DPO_DATA_SPEC"
    --data_dir "$DATA_DIR"
    --max_download_mb "$MAX_DOWNLOAD_MB"
    --out_dir "$DPO_OUT"
    --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
    --seq_len "$DPO_SEQ_LEN"
    --batch_size "$DPO_BATCH_SIZE"
    --accum_steps "$DPO_ACCUM_STEPS"
    --epochs "$DPO_EPOCHS"
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
  )
  if [ -n "${TB_DPO_DIR:-}" ]; then
    DPO_ARGS+=(--tensorboard_dir "$TB_DPO_DIR")
  fi
  if [ -n "${HF_ENDPOINT:-}" ]; then
    DPO_ARGS+=(--hf_endpoint "$HF_ENDPOINT")
  fi
  if [ -n "${DPO_BETA:-}" ]; then
    DPO_ARGS+=(--dpo_beta "$DPO_BETA")
  fi
  if [ -n "$DPO_MAX_STEPS" ]; then
    DPO_ARGS+=(--max_steps "$DPO_MAX_STEPS")
  fi
  if [ -n "$DPO_RESUME" ]; then
    DPO_ARGS+=(--resume "$DPO_RESUME")
  else
    DPO_ARGS+=(--init_from "$DPO_INIT_FROM")
  fi
  if [ -n "$DPO_REF_FROM" ]; then
    DPO_ARGS+=(--dpo_ref_from "$DPO_REF_FROM")
  fi

  if [ "${#TRAIN_EXTRA_ARGS[@]}" -gt 0 ]; then
    DPO_ARGS+=("${TRAIN_EXTRA_ARGS[@]}")
  fi
  run_stage "dpo" "${DPO_ARGS[@]}"
fi

if [ "$RUN_R1" -eq 1 ]; then
  if [ "$SMOKE_TEST" -eq 1 ]; then
    R1_SEQ_LEN=${R1_LEN:-256}
    R1_BATCH_SIZE=${R1_BS:-2}
    R1_ACCUM_STEPS=${R1_ACCUM:-1}
    R1_EPOCHS=${R1_EPOCH:-1}
    R1_MAX_STEPS=${R1_MAX:-5}
  else
    R1_SEQ_LEN=${R1_LEN:-1024}
    R1_BATCH_SIZE=${R1_BS:-1}
    R1_ACCUM_STEPS=${R1_ACCUM:-8}
    R1_EPOCHS=${R1_EPOCH:-1}
    R1_MAX_STEPS=${R1_MAX:-}
  fi

  R1_RESUME=$(latest_ckpt "$R1_OUT")
  R1_INIT_FROM=${R1_INIT_FROM:-}
  if [ -z "$R1_INIT_FROM" ]; then
    if [ "$RUN_DPO" -eq 1 ]; then
      R1_INIT_FROM=$(latest_ckpt "$DPO_OUT")
    fi
  fi
  if [ -z "$R1_INIT_FROM" ]; then
    R1_INIT_FROM=$(latest_ckpt "$SFT_OUT")
  fi
  if [ -z "$R1_RESUME" ] && [ -z "$R1_INIT_FROM" ]; then
    echo "[error] R1 requires a SFT/DPO checkpoint (set R1_FROM or run SFT/DPO first)" >&2
    exit 1
  fi

  if [ -z "${R1_DATA_SPEC:-}" ]; then
    if [ "$SMOKE_TEST" -eq 1 ]; then
      R1_DATA_SPEC=minimind:smoke
    else
      R1_DATA_SPEC=minimind:r1_mix_1024.jsonl
    fi
  fi

  R1_ARGS=(
    --task r1
    --preset "$PRESET"
    --dtype "$DTYPE"
    --data_path "$R1_DATA_SPEC"
    --data_dir "$DATA_DIR"
    --max_download_mb "$MAX_DOWNLOAD_MB"
    --out_dir "$R1_OUT"
    --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
    --seq_len "$R1_SEQ_LEN"
    --batch_size "$R1_BATCH_SIZE"
    --accum_steps "$R1_ACCUM_STEPS"
    --epochs "$R1_EPOCHS"
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
  )
  if [ -n "${TB_R1_DIR:-}" ]; then
    R1_ARGS+=(--tensorboard_dir "$TB_R1_DIR")
  fi
  if [ -n "${HF_ENDPOINT:-}" ]; then
    R1_ARGS+=(--hf_endpoint "$HF_ENDPOINT")
  fi
  if [ -n "${R1_TOKEN_WEIGHT:-}" ]; then
    R1_ARGS+=(--r1_token_weight "$R1_TOKEN_WEIGHT")
  fi
  if [ -n "${R1_TOKENS:-}" ]; then
    R1_ARGS+=(--r1_tokens "$R1_TOKENS")
  fi
  if [ -n "$R1_MAX_STEPS" ]; then
    R1_ARGS+=(--max_steps "$R1_MAX_STEPS")
  fi

  if [ -n "$R1_RESUME" ]; then
    R1_ARGS+=(--resume "$R1_RESUME")
  else
    R1_ARGS+=(--init_from "$R1_INIT_FROM")
  fi

  if [ "${#TRAIN_EXTRA_ARGS[@]}" -gt 0 ]; then
    R1_ARGS+=("${TRAIN_EXTRA_ARGS[@]}")
  fi
  run_stage "r1" "${R1_ARGS[@]}"
fi

if [ "$SKIP_INFER" -eq 0 ]; then
  if [ -n "$INFER_CHECKPOINT" ]; then
    INFER_CKPT=$INFER_CHECKPOINT
    if [ -f "$INFER_CKPT" ] && [[ "$INFER_CKPT" == *.safetensors ]]; then
      INFER_CKPT=$(dirname "$INFER_CKPT")
    fi
    if [ ! -d "$INFER_CKPT" ]; then
      echo "[infer] Invalid --infer-checkpoint: $INFER_CHECKPOINT (resolved: $INFER_CKPT)" >&2
      exit 1
    fi
    if [ ! -s "$INFER_CKPT/model.safetensors" ] || [ ! -s "$INFER_CKPT/config.json" ]; then
      echo "[infer] Checkpoint dir must contain model.safetensors + config.json: $INFER_CKPT" >&2
      exit 1
    fi
  else
    INFER_CKPT=""
    # Allow OUT_DIR to directly be a training output dir (OUT_DIR/checkpoints/step_*)
    # or a single checkpoint dir (OUT_DIR/model.safetensors + config.json).
    if [ -s "$OUT_DIR/model.safetensors" ] && [ -s "$OUT_DIR/config.json" ]; then
      INFER_CKPT=$OUT_DIR
    else
      INFER_CKPT=$(latest_ckpt "$OUT_DIR")
      if [ -z "$INFER_CKPT" ] && [ -d "$R1_OUT" ]; then
        INFER_CKPT=$(latest_ckpt "$R1_OUT")
      fi
      if [ -z "$INFER_CKPT" ] && [ -d "$DPO_OUT" ]; then
        INFER_CKPT=$(latest_ckpt "$DPO_OUT")
      fi
      if [ -z "$INFER_CKPT" ]; then
        INFER_CKPT=$(latest_ckpt "$SFT_OUT")
      fi
      if [ -z "$INFER_CKPT" ]; then
        INFER_CKPT=$(latest_ckpt "$PRETRAIN_OUT")
      fi
    fi
  fi

  if [ -n "$INFER_CKPT" ]; then
    INFER_PROMPT=${INF_PROMPT:-hi}
    if [ "$SMOKE_TEST" -eq 1 ]; then
      INFER_MAX_NEW_TOKENS=${INF_MAX_NEW:-64}
    else
      INFER_MAX_NEW_TOKENS=${INF_MAX_NEW:-512}
    fi
    INFER_MIN_NEW_TOKENS=${INF_MIN_NEW:-1}
    INFER_TEMPERATURE=${INF_TEMP:-0}
    INFER_TOP_P=${INF_TOP_P:-1.0}
    INFER_MAX_SEQ_LEN=${INF_MAX_SEQ:-}
    if [ -z "$INFER_MAX_SEQ_LEN" ]; then
      STATE_JSON="$INFER_CKPT/state.json"
      if [ -f "$STATE_JSON" ]; then
        INFER_MAX_SEQ_LEN=$("$PY" - "$STATE_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(0)

args = data.get("args") if isinstance(data, dict) else None
if isinstance(args, dict):
    seq_len = args.get("seq_len")
    try:
        seq_len = int(seq_len)
    except Exception:
        seq_len = None
    if seq_len and seq_len > 0:
        print(seq_len)
PY
)
      fi
    fi
    echo
    echo "[stage] infer"
    if [ "$INFER_DEMO" -eq 1 ]; then
      INFER_DEMO_MODE=${INF_MODE:-knowledge}
      INFER_DEMO_SUITES=${INF_SUITES:-copy,json,sort,math_mcq,logic,qa,knowledge}
      INFER_DEMO_N=${INF_N:-2}
      INFER_DEMO_NO_CHAT=${INF_NO_CHAT:-0}
      DEMO_ARGS=(--checkpoint "$INFER_CKPT" --mode "$INFER_DEMO_MODE" --max_new_tokens "$INFER_MAX_NEW_TOKENS")
      if [ -n "$INFER_MAX_SEQ_LEN" ]; then
        DEMO_ARGS+=(--max_seq_len "$INFER_MAX_SEQ_LEN")
      fi
      if [ "$INFER_DEMO_MODE" = "bench" ]; then
        DEMO_ARGS+=(--suite "$INFER_DEMO_SUITES" --n "$INFER_DEMO_N")
        if [ "$INFER_DEMO_NO_CHAT" = "1" ]; then
          DEMO_ARGS+=(--no_chat)
        fi
      fi
      echo "$PY -m mlx_train.demo ${DEMO_ARGS[*]}"
      "$PY" -m mlx_train.demo \
        "${DEMO_ARGS[@]}"
    else
      INFER_ARGS=(
        --checkpoint "$INFER_CKPT"
        --prompt "$INFER_PROMPT"
        --max_new_tokens "$INFER_MAX_NEW_TOKENS"
        --min_new_tokens "$INFER_MIN_NEW_TOKENS"
        --temperature "$INFER_TEMPERATURE"
        --top_p "$INFER_TOP_P"
      )
      if [ -n "$INFER_MAX_SEQ_LEN" ]; then
        INFER_ARGS+=(--max_seq_len "$INFER_MAX_SEQ_LEN")
      fi
      echo "$PY -m mlx_train.infer ${INFER_ARGS[*]}"
      "$PY" -m mlx_train.infer "${INFER_ARGS[@]}"
    fi
  else
    echo "[infer] No checkpoint found under $OUT_DIR"
  fi
fi

if [ "$SMOKE_TEST" -eq 1 ] && [ "$CLEANUP_SMOKE" = "1" ] && [ "$OUT_DIR_WAS_SET" -eq 0 ]; then
  echo
  echo "[cleanup] Removing smoke outputs: $OUT_DIR"
  safe_rm_rf "$OUT_DIR"
  if [ -n "${TF_DIR:-}" ] && [ "$TF_DIR_WAS_SET" -eq 0 ] && [ -d "$TF_DIR" ]; then
    echo "[cleanup] Removing smoke logs: $TF_DIR"
    safe_rm_rf "$TF_DIR"
  fi
fi

echo
echo "[done] MLX pipeline finished."
