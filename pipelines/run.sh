#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run.sh [OPTIONS]

Options:
  --smoke-test       Run a fast CPU-only smoke test with tiny dataset slices and
                     a handful of optimizer steps per stage. Useful for local
                     verification that the end-to-end pipeline works without
                     requiring a GPU.

  --skip-pretrain    Skip pretrain stage if a good checkpoint exists. The script
                     will automatically check if a pretrain checkpoint exists and
                     has acceptable quality, then skip to SFT training.

  --run-r1          After DPO, train a reasoning (R1) model via distillation and
                    run a lightweight demo generation.

  --force-pretrain   Force pretrain stage even if a checkpoint exists (overrides
                     --skip-pretrain).

  -h, --help         Show this help message and exit.

Environment Variables:
  NO_VENV=1          Skip virtual environment, use system Python (for Colab/cloud)
  PIP_INDEX_URL      PyPI mirror URL (default: https://pypi.org/simple)
  MODEL_HIDDEN_SIZE  Model hidden size (default: 512, use 1024 for 0.2B)
  MODEL_NUM_LAYERS   Number of layers (default: 8, use 16 for 0.2B)
  MODEL_SEQ_LEN      Sequence length (default: 512)
  BATCH_SIZE         Override auto-detected batch size
  PREPROCESS_DATA    0=skip, 1=cache bin2d, 2=force reprocess (default: 1)
  MINIMIND_DATA_SOURCE  Data source: modelscope (default) or huggingface
  MINIMIND_DATA_REPO    Dataset repo (default: gongjy/minimind_dataset)

Examples:
  ./scripts/run.sh                      # Auto-optimized for your GPU
  ./scripts/run.sh --smoke-test         # Quick test on CPU
  NO_VENV=1 ./scripts/run.sh            # Run without venv (Colab)
  MODEL_HIDDEN_SIZE=1024 MODEL_NUM_LAYERS=16 ./scripts/run.sh  # Train 0.2B model
  MINIMIND_DATA_SOURCE=huggingface MINIMIND_DATA_REPO=jingyaogong/minimind_dataset ./scripts/run.sh
USAGE
}

SMOKE_TEST=0
SKIP_PRETRAIN=0
FORCE_PRETRAIN=0
RUN_R1=0

while (($#)); do
  case "$1" in
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --skip-pretrain)
      SKIP_PRETRAIN=1
      shift
      ;;
    --run-r1)
      RUN_R1=1
      shift
      ;;
    --force-pretrain)
      FORCE_PRETRAIN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Unset proxy variables that may interfere with package installation
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy
unset NO_PROXY

# PyPI index URL - defaults to official PyPI
# Set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple for China
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}

VENV_DIR=${VENV_DIR:-.venv}
MINIMIND_DATA_SOURCE=${MINIMIND_DATA_SOURCE:-modelscope}
MINIMIND_DATA_REPO=${MINIMIND_DATA_REPO:-gongjy/minimind_dataset}
MINIMIND_MS_CACHE=${MINIMIND_MS_CACHE:-$HOME/.cache/modelscope}
export MINIMIND_DATA_SOURCE MINIMIND_DATA_REPO MINIMIND_MS_CACHE

# Check Python version compatibility (requires Python 3.9-3.12)
check_python_version() {
  local python_cmd=$1
  if ! command -v "$python_cmd" >/dev/null 2>&1; then
    return 1
  fi

  local py_version=$("$python_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  local major=$(echo "$py_version" | cut -d. -f1)
  local minor=$(echo "$py_version" | cut -d. -f2)

  # Check if version is in supported range (3.9-3.12)
  if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ] && [ "$minor" -le 12 ]; then
    echo "$python_cmd"
    return 0
  fi
  return 1
}

# Find compatible Python interpreter
PYTHON_CMD=""
for py_candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
  if PYTHON_CMD=$(check_python_version "$py_candidate" 2>/dev/null); then
    PY_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "[env] Using Python $PY_VERSION at $PYTHON_CMD"
    break
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  echo "[error] No compatible Python version found (requires 3.9-3.12)" >&2
  echo "[error] Current Python versions detected:" >&2
  for py_test in python3 python; do
    if command -v "$py_test" >/dev/null 2>&1; then
      "$py_test" --version >&2 || true
    fi
  done
  exit 1
fi

# Detect cloud environment and set defaults accordingly
# Check if running in OpenBayes environment
if [ -d "/openbayes/home" ] 2>/dev/null; then
  IS_CLOUD=1
  TF_DIR=${TF_DIR:-/openbayes/home/tf_dir}
  PRETRAIN_DEFAULT_ROOT=${DATA_ROOT:-/openbayes/input/input0}
else
  IS_CLOUD=0
  TF_DIR=${TF_DIR:-./tf_dir}
  PRETRAIN_DEFAULT_ROOT=${DATA_ROOT:-./data}
  echo "[env] Running in local environment (TF_DIR: $TF_DIR)"
fi

OUT_DIR=${OUT_DIR:-out}
DATA_DIR=${DATA_DIR:-data/processed}
MINIMIND_DATA_DIR=${MINIMIND_DATA_DIR:-dataset/minimind}
RESULTS_FILE=${RESULTS_FILE:-"$TF_DIR/eval_results.jsonl"}
MAX_DOWNLOAD_MB=${MAX_DOWNLOAD_MB:-0}
AUTO_DOWNLOAD=${AUTO_DOWNLOAD:-1}
KEEP_LAST=${KEEP_LAST:-3}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}

# ============================================================
# Environment setup - simplified, uv-first approach
# ============================================================
# Set NO_VENV=1 to skip virtual environment (e.g., in Colab)
NO_VENV=${NO_VENV:-0}

# Check for uv
if command -v uv >/dev/null 2>&1; then
  USE_UV=1
  echo "[env] Using uv for package management"
else
  USE_UV=0
  echo "[env] uv not found, using pip"
fi

# Setup environment
if [ "$NO_VENV" -eq 1 ]; then
  echo "[env] NO_VENV=1, using system Python directly"
  DEPS_MARKER=".deps_installed"
  DEPS_HASH_FILE=".deps_hash"
else
  # Create/use virtual environment
  if [ ! -d "$VENV_DIR" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[env] Creating virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR" 2>/dev/null || true
    if [ "$USE_UV" -eq 1 ]; then
      uv venv "$VENV_DIR" --python "$PYTHON_CMD" --seed
    else
      "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  trap 'deactivate >/dev/null 2>&1 || true' EXIT
  echo "[env] Activated $VENV_DIR"
  DEPS_MARKER="$VENV_DIR/.deps_installed"
  DEPS_HASH_FILE="$VENV_DIR/.deps_hash"
fi

# Check if dependencies need installation
REQUIREMENTS_HASH=$(shasum -a 256 requirements.txt 2>/dev/null | cut -d' ' -f1 || echo "unknown")
NEED_INSTALL=0

if [ ! -f "$DEPS_MARKER" ]; then
  NEED_INSTALL=1
elif [ ! -f "$DEPS_HASH_FILE" ] || [ "$REQUIREMENTS_HASH" != "$(cat "$DEPS_HASH_FILE" 2>/dev/null)" ]; then
  NEED_INSTALL=1
  echo "[env] requirements.txt changed, updating"
else
  echo "[env] Dependencies up to date"
fi

# Install dependencies
if [ "$NEED_INSTALL" -eq 1 ]; then
  echo "[env] Installing dependencies..."

  # Determine pip/uv target
  if [ "$NO_VENV" -eq 1 ]; then
    UV_PYTHON_FLAG=""
    PIP_CMD="python -m pip"
  else
    UV_PYTHON_FLAG="--python $VENV_DIR/bin/python"
    PIP_CMD="$VENV_DIR/bin/python -m pip"
  fi

  if [ "$USE_UV" -eq 1 ]; then
    uv pip install $UV_PYTHON_FLAG -r requirements.txt || {
      echo "[env] uv failed, falling back to pip"
      USE_UV=0
    }
  fi

  if [ "$USE_UV" -eq 0 ]; then
    $PIP_CMD install --upgrade pip -q
    $PIP_CMD install -r requirements.txt
  fi

  # Verify torch
  if ! python -c "import torch" 2>/dev/null; then
    echo "[env] Installing torch..."
    if [ "$USE_UV" -eq 1 ]; then
      uv pip install $UV_PYTHON_FLAG torch
    else
      $PIP_CMD install torch
    fi
  fi

  touch "$DEPS_MARKER"
  echo "$REQUIREMENTS_HASH" > "$DEPS_HASH_FILE"
  echo "[env] Dependencies ready"
fi

# ============================================================
# RustBPE tokenizer setup (optional, for faster tokenization)
# ============================================================
RUSTBPE_MARKER="${DEPS_MARKER:-/tmp}/.rustbpe_compiled"
if [ -d "rustbpe" ]; then
  # Check if rustbpe is already importable
  if ! python -c "import rustbpe" 2>/dev/null; then
    if command -v cargo >/dev/null 2>&1; then
      echo "[env] Compiling RustBPE tokenizer..."
      # Install maturin if needed
      if ! python -c "import maturin" 2>/dev/null; then
        if [ "$USE_UV" -eq 1 ]; then
          uv pip install maturin -q 2>/dev/null || pip install maturin -q
        else
          pip install maturin -q
        fi
      fi
      # Compile rustbpe
      if maturin develop --release --manifest-path rustbpe/Cargo.toml 2>/dev/null; then
        echo "[env] RustBPE compiled successfully"
        touch "$RUSTBPE_MARKER"
      else
        echo "[env] RustBPE compilation failed, will use fallback tokenizer"
      fi
    else
      echo "[env] Rust not found, skipping RustBPE (install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)"
    fi
  else
    echo "[env] RustBPE already available"
  fi
fi

mkdir -p "$TF_DIR" || { echo "[warn] Could not create $TF_DIR directory"; }
mkdir -p "$OUT_DIR" || { echo "[error] Could not create $OUT_DIR directory" >&2; exit 1; }
mkdir -p "$DATA_DIR" || { echo "[error] Could not create $DATA_DIR directory" >&2; exit 1; }
mkdir -p "$MINIMIND_DATA_DIR" || { echo "[error] Could not create $MINIMIND_DATA_DIR directory" >&2; exit 1; }

# PRETRAIN_DEFAULT_ROOT was already set in cloud environment detection above
if [ -z "${PRETRAIN_DEFAULT_ROOT:-}" ]; then
  PRETRAIN_DEFAULT_ROOT=${DATA_ROOT:-./data}
fi

# Check /gemini/data-1/ directory first (highest priority)
GEMINI_DATA_DIR="/gemini/data-1"
if [ -d "$GEMINI_DATA_DIR" ]; then
  echo "[data] Found /gemini/data-1/ directory, checking for training data..."
  
  # Check for pretrain data
  if [ -s "$GEMINI_DATA_DIR/pretrain_hq.jsonl" ]; then
    PRETRAIN_DEFAULT_ROOT="$GEMINI_DATA_DIR"
    echo "[data] Using pretrain data from $GEMINI_DATA_DIR/pretrain_hq.jsonl"
  elif [ -s "$GEMINI_DATA_DIR/pretrain.jsonl" ]; then
    PRETRAIN_DEFAULT_ROOT="$GEMINI_DATA_DIR"
    echo "[data] Using pretrain data from $GEMINI_DATA_DIR/pretrain.jsonl"
  fi
  
  # Check for SFT data
  if [ -s "$GEMINI_DATA_DIR/sft_mini_512.jsonl" ]; then
    SFT_JSON="$GEMINI_DATA_DIR/sft_mini_512.jsonl"
    echo "[data] Using SFT data from $SFT_JSON"
  elif [ -s "$GEMINI_DATA_DIR/sft.jsonl" ]; then
    SFT_JSON="$GEMINI_DATA_DIR/sft.jsonl"
    echo "[data] Using SFT data from $SFT_JSON"
  fi
  
  # Check for DPO data
  if [ -s "$GEMINI_DATA_DIR/dpo.jsonl" ]; then
    DPO_JSON="$GEMINI_DATA_DIR/dpo.jsonl"
    echo "[data] Using DPO data from $DPO_JSON"
  elif [ -s "$GEMINI_DATA_DIR/dpo_pairs.jsonl" ]; then
    DPO_JSON="$GEMINI_DATA_DIR/dpo_pairs.jsonl"
    echo "[data] Using DPO data from $DPO_JSON"
  fi
fi

PRETRAIN_JSON=${PRETRAIN_JSON:-"$PRETRAIN_DEFAULT_ROOT/pretrain_hq.jsonl"}
# Use high-quality SFT dataset (cleaned, deduplicated, filtered)
SFT_JSON=${SFT_JSON:-"data/final/sft_high_quality.jsonl"}
DPO_JSON=${DPO_JSON:-"$PRETRAIN_DEFAULT_ROOT/dpo_pairs.jsonl"}
R1_JSON=${R1_JSON:-"dataset/r1_mix_1024.jsonl"}

if [ ! -s "$DPO_JSON" ]; then
  ALT_DPO="$PRETRAIN_DEFAULT_ROOT/dpo.jsonl"
  if [ -s "$ALT_DPO" ]; then
    DPO_JSON="$ALT_DPO"
  fi
fi

if [ ! -s "$R1_JSON" ]; then
  if [ -s "$PRETRAIN_DEFAULT_ROOT/r1_mix_1024.jsonl" ]; then
    R1_JSON="$PRETRAIN_DEFAULT_ROOT/r1_mix_1024.jsonl"
  elif [ -s "$DATA_DIR/r1_mix_1024.jsonl" ]; then
    R1_JSON="$DATA_DIR/r1_mix_1024.jsonl"
  elif [ -s "$MINIMIND_DATA_DIR/r1_mix_1024.jsonl" ]; then
    R1_JSON="$MINIMIND_DATA_DIR/r1_mix_1024.jsonl"
  fi
fi

# Cloud environment: Auto-process data from /input0
if [ "$IS_CLOUD" -eq 1 ]; then
  echo "[cloud] Cloud environment detected, checking for input data..."

  # Check for raw SFT data in input directory
  INPUT_SFT_RAW="$PRETRAIN_DEFAULT_ROOT/sft_mini_512.jsonl"
  INPUT_SFT_CLEANED="$PRETRAIN_DEFAULT_ROOT/final/sft_mini_512.cleaned.jsonl"
  OUTPUT_SFT_HQ="data/final/sft_high_quality.jsonl"

  # Determine source file for processing
  SFT_SOURCE=""
  if [ -s "$INPUT_SFT_CLEANED" ]; then
    SFT_SOURCE="$INPUT_SFT_CLEANED"
    echo "[cloud] Found cleaned SFT data at $INPUT_SFT_CLEANED"
  elif [ -s "$INPUT_SFT_RAW" ]; then
    SFT_SOURCE="$INPUT_SFT_RAW"
    echo "[cloud] Found raw SFT data at $INPUT_SFT_RAW"
  fi

  # Auto-generate high-quality dataset if source exists but output doesn't
  if [ -n "$SFT_SOURCE" ] && [ ! -s "$OUTPUT_SFT_HQ" ]; then
    echo "[cloud] Auto-generating high-quality SFT dataset..."
    mkdir -p "data/final"

    # Create temporary processing script
    python -c "
import json
import hashlib
from pathlib import Path

input_file = '$SFT_SOURCE'
output_file = '$OUTPUT_SFT_HQ'

print(f'[cloud] Processing {input_file} -> {output_file}')

seen_hashes = set()
kept_count = 0
removed_count = 0

Path(output_file).parent.mkdir(parents=True, exist_ok=True)

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    for line_num, line in enumerate(infile, 1):
        if line_num % 100000 == 0:
            print(f'[cloud] Processed {line_num:,} records, kept {kept_count:,}')

        try:
            data = json.loads(line.strip())

            if 'conversations' not in data:
                removed_count += 1
                continue

            # Extract user and assistant content
            user_content = ''
            assistant_content = ''

            for conv in data['conversations']:
                if conv.get('role') == 'user':
                    user_content = conv.get('content', '').strip()
                elif conv.get('role') == 'assistant':
                    assistant_content = conv.get('content', '').strip()

            # Filter: remove empty or very short content
            if not user_content or not assistant_content:
                removed_count += 1
                continue

            if len(user_content) < 5 or len(assistant_content) < 10:
                removed_count += 1
                continue

            # Deduplicate
            content_hash = hashlib.md5(
                (user_content + assistant_content).encode('utf-8')
            ).hexdigest()

            if content_hash in seen_hashes:
                removed_count += 1
                continue

            seen_hashes.add(content_hash)

            # Keep this record
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            kept_count += 1

        except:
            removed_count += 1

print(f'[cloud] Processing complete: kept {kept_count:,}, removed {removed_count:,}')
" || {
      echo "[cloud] Failed to auto-process SFT data" >&2
      # Fall back to using source directly
      SFT_JSON="$SFT_SOURCE"
    }

    if [ -s "$OUTPUT_SFT_HQ" ]; then
      echo "[cloud] High-quality SFT dataset created successfully"
      SFT_JSON="$OUTPUT_SFT_HQ"
    fi
  elif [ -s "$OUTPUT_SFT_HQ" ]; then
    echo "[cloud] Using existing high-quality SFT dataset"
    SFT_JSON="$OUTPUT_SFT_HQ"
  fi

  # Update SFT path if we found data in input directory
  if [ -n "$SFT_SOURCE" ] && [ ! -s "$SFT_JSON" ]; then
    SFT_JSON="$SFT_SOURCE"
    echo "[cloud] Using SFT data from input: $SFT_JSON"
  fi
fi

download_minimind_file() {
  local filename=$1
  python - <<PY
import os
from mlx_train.download import ensure_dataset_file

path = ensure_dataset_file(
    data_source=os.environ.get("MINIMIND_DATA_SOURCE", "modelscope"),
    repo_id=os.environ.get("MINIMIND_DATA_REPO", "gongjy/minimind_dataset"),
    filename="${filename}",
    data_dir=os.environ.get("MINIMIND_DATA_DIR", "dataset/minimind"),
    endpoint=os.environ.get("HF_ENDPOINT"),
    force=False,
    max_download_mb=int(os.environ.get("MAX_DOWNLOAD_MB", "0")),
    ms_cache_dir=os.environ.get("MINIMIND_MS_CACHE"),
)
print(path)
PY
}

if [ "$AUTO_DOWNLOAD" -eq 1 ]; then
  if [ ! -s "$PRETRAIN_JSON" ]; then
    echo "[data] Downloading pretrain_hq.jsonl..."
    PRETRAIN_JSON=$(download_minimind_file "pretrain_hq.jsonl") || {
      echo "[error] Failed to download pretrain_hq.jsonl" >&2
      exit 1
    }
  fi
  if [ ! -s "$SFT_JSON" ]; then
    echo "[data] Downloading sft_512.jsonl..."
    SFT_JSON=$(download_minimind_file "sft_512.jsonl") || {
      echo "[error] Failed to download sft_512.jsonl" >&2
      exit 1
    }
  fi
  if [ ! -s "$DPO_JSON" ]; then
    echo "[data] Downloading dpo.jsonl..."
    DPO_JSON=$(download_minimind_file "dpo.jsonl") || {
      echo "[error] Failed to download dpo.jsonl" >&2
      exit 1
    }
  fi
  if [ "$RUN_R1" -eq 1 ] && [ ! -s "$R1_JSON" ]; then
    echo "[data] Downloading r1_mix_1024.jsonl..."
    R1_JSON=$(download_minimind_file "r1_mix_1024.jsonl") || {
      echo "[error] Failed to download r1_mix_1024.jsonl" >&2
      exit 1
    }
  fi
fi

NEED_LOCAL_DATA=0

if [ ! -s "$PRETRAIN_JSON" ]; then
  echo "[data] Falling back to processed pretrain data"
  PRETRAIN_JSON="$DATA_DIR/pretrain_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ ! -s "$SFT_JSON" ]; then
  echo "[data] Falling back to processed SFT data"
  SFT_JSON="$DATA_DIR/sft_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ ! -s "$DPO_JSON" ]; then
  echo "[data] Falling back to processed DPO data"
  DPO_JSON="$DATA_DIR/dpo_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ "$NEED_LOCAL_DATA" -eq 1 ]; then
  IDENTITY_DATA="data/chinese/identity_conversations.jsonl"
  if [ -f "$IDENTITY_DATA" ]; then
    echo "[data] Using identity data at $IDENTITY_DATA"
  else
    echo "[data] Identity data missing at $IDENTITY_DATA" >&2
    exit 1
  fi
  echo "[data] Building Chinese data mixtures"
  python scripts/build_chinese_mix.py --output-dir "$DATA_DIR"
fi

for path in "$PRETRAIN_JSON" "$SFT_JSON" "$DPO_JSON"; do
  if [ ! -s "$path" ]; then
    echo "[data] Required dataset not found: $path" >&2
    exit 1
  fi
done

if [ "$SMOKE_TEST" -eq 1 ]; then
  echo "[smoke] Enabling CPU smoke test mode"
  SMOKE_DIR="$OUT_DIR/smoke_data"
  mkdir -p "$SMOKE_DIR"

  SMOKE_PRETRAIN_LIMIT=${SMOKE_PRETRAIN_LIMIT:-64}
  SMOKE_SFT_LIMIT=${SMOKE_SFT_LIMIT:-16}
  SMOKE_DPO_LIMIT=${SMOKE_DPO_LIMIT:-8}

  python scripts/create_smoke_subset.py --input "$PRETRAIN_JSON" --output "$SMOKE_DIR/pretrain.jsonl" --limit "$SMOKE_PRETRAIN_LIMIT"
  PRETRAIN_JSON="$SMOKE_DIR/pretrain.jsonl"

  python scripts/create_smoke_subset.py --input "$SFT_JSON" --output "$SMOKE_DIR/sft.jsonl" --limit "$SMOKE_SFT_LIMIT"
  SFT_JSON="$SMOKE_DIR/sft.jsonl"

  python scripts/create_smoke_subset.py --input "$DPO_JSON" --output "$SMOKE_DIR/dpo.jsonl" --limit "$SMOKE_DPO_LIMIT"
  DPO_JSON="$SMOKE_DIR/dpo.jsonl"

  if [ "$RUN_R1" -eq 1 ] && [ -s "$R1_JSON" ]; then
    SMOKE_R1_LIMIT=${SMOKE_R1_LIMIT:-8}
    python scripts/create_smoke_subset.py --input "$R1_JSON" --output "$SMOKE_DIR/r1.jsonl" --limit "$SMOKE_R1_LIMIT"
    R1_JSON="$SMOKE_DIR/r1.jsonl"
  fi
fi

EXTRA_PRETRAIN_ARGS=()
EXTRA_SFT_ARGS=()
EXTRA_DPO_ARGS=()
EXTRA_R1_ARGS=()

if [ -n "${PRETRAIN_ARGS:-}" ]; then
  read -r -a EXTRA_PRETRAIN_ARGS <<<"${PRETRAIN_ARGS}"
fi
if [ -n "${SFT_ARGS:-}" ]; then
  read -r -a EXTRA_SFT_ARGS <<<"${SFT_ARGS}"
fi
if [ -n "${DPO_ARGS:-}" ]; then
  read -r -a EXTRA_DPO_ARGS <<<"${DPO_ARGS}"
fi
if [ -n "${R1_ARGS:-}" ]; then
  read -r -a EXTRA_R1_ARGS <<<"${R1_ARGS}"
fi

MODEL_HIDDEN_SIZE=${MODEL_HIDDEN_SIZE:-512}   # MiniLLM2-Small (~26M params)
MODEL_NUM_LAYERS=${MODEL_NUM_LAYERS:-8}
MODEL_SEQ_LEN=${MODEL_SEQ_LEN:-512}
USE_MOE=${USE_MOE:-false}

# ============================================================
# GPU detection and auto-optimization
# ============================================================
detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' '
  else
    echo "1"
  fi
}

detect_gpu_memory() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
  else
    echo "0"
  fi
}

NUM_GPUS=${NUM_GPUS:-$(detect_gpu_count)}
GPU_MEM=${GPU_MEM:-$(detect_gpu_memory)}

# Auto-configure batch size based on GPU memory
if [ -z "${BATCH_SIZE:-}" ]; then
  if [ "$GPU_MEM" -ge 70000 ]; then
    # A100 80GB or similar
    BATCH_SIZE=256
    ACCUM_STEPS=1
    NUM_WORKERS=8
    echo "[gpu] A100 80GB detected, using optimized settings: batch=$BATCH_SIZE"
  elif [ "$GPU_MEM" -ge 38000 ]; then
    # A100 40GB or A6000
    BATCH_SIZE=128
    ACCUM_STEPS=2
    NUM_WORKERS=8
    echo "[gpu] 40GB+ GPU detected: batch=$BATCH_SIZE, accum=$ACCUM_STEPS"
  elif [ "$GPU_MEM" -ge 20000 ]; then
    # RTX 3090/4090 24GB
    BATCH_SIZE=64
    ACCUM_STEPS=4
    NUM_WORKERS=4
    echo "[gpu] 24GB GPU detected: batch=$BATCH_SIZE, accum=$ACCUM_STEPS"
  else
    # Smaller GPUs or CPU
    BATCH_SIZE=32
    ACCUM_STEPS=8
    NUM_WORKERS=2
    echo "[gpu] Standard settings: batch=$BATCH_SIZE, accum=$ACCUM_STEPS"
  fi
else
  ACCUM_STEPS=${ACCUM_STEPS:-4}
  NUM_WORKERS=${NUM_WORKERS:-4}
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  echo "[gpu] Detected $NUM_GPUS GPUs, using torchrun"
  TRAIN_CMD_PREFIX="torchrun --nproc_per_node=$NUM_GPUS"
else
  echo "[gpu] Single GPU mode"
  TRAIN_CMD_PREFIX="python"
fi

# ============================================================
# Data preprocessing (convert JSONL to fast bin2d format)
# ============================================================
PREPROCESS_DATA=${PREPROCESS_DATA:-1}

preprocess_to_bin2d() {
  local input_path=$1
  local out_prefix=$2
  local task=${3:-pretrain}
  local seq_len=${4:-$MODEL_SEQ_LEN}
  local meta_file="${out_prefix}.meta.json"

  if [ -f "$meta_file" ] && [ "$PREPROCESS_DATA" -ne 2 ]; then
    echo "[preprocess] Using cached: $meta_file"
    echo "$meta_file"
    return 0
  fi

  # Check for RustBPE tokenizer (faster)
  TOKENIZER_TYPE="auto"
  if [ -f "./model/tokenizer.pkl" ] && python -c "import rustbpe" 2>/dev/null; then
    echo "[preprocess] Using RustBPE tokenizer (fast)"
    TOKENIZER_TYPE="rustbpe"
  elif [ -f "./model/tokenizer.pkl" ]; then
    echo "[preprocess] tokenizer.pkl found but rustbpe not available, using HuggingFace tokenizer"
  fi

  echo "[preprocess] Converting $input_path to bin2d format..."
  # Use all available CPU cores for preprocessing
  PREPROCESS_WORKERS=${PREPROCESS_WORKERS:-0}
  python -m mlx_train.cli.packbin2d \
    --data_path "$input_path" \
    --out_prefix "$out_prefix" \
    --seq_len "$seq_len" \
    --task "$task" \
    --tokenizer_path ./model \
    --tokenizer_type "$TOKENIZER_TYPE" \
    --show_progress \
    --num_workers "$PREPROCESS_WORKERS" \
    --chunk_size 500 \
    --log_interval 5000 || {
      echo "[preprocess] Failed, using original JSONL"
      echo "$input_path"
      return 0
    }
  echo "$meta_file"
}

# Common DataLoader args for GPU training
DATALOADER_ARGS="--num_workers $NUM_WORKERS --pin_memory --prefetch_factor 4 --persistent_workers"

MOE_SUFFIX=""
if [ "$USE_MOE" = "true" ]; then
  MOE_SUFFIX="_moe"
fi

PRETRAIN_OUT="$OUT_DIR/pretrain"
SFT_OUT="$OUT_DIR/sft"
DPO_OUT="$OUT_DIR/dpo"
R1_OUT="$OUT_DIR/r1"

CHECKPOINT_PRETRAIN="$PRETRAIN_OUT/pretrain_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"
CHECKPOINT_SFT="$SFT_OUT/full_sft_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"
CHECKPOINT_DPO="$DPO_OUT/rlhf_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"
CHECKPOINT_R1="$R1_OUT/reason_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"

LEGACY_PRETRAIN="$OUT_DIR/pretrain_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"
LEGACY_SFT="$OUT_DIR/full_sft_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"
LEGACY_DPO="$OUT_DIR/rlhf_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"
LEGACY_R1="$OUT_DIR/reason_${MODEL_HIDDEN_SIZE}${MOE_SUFFIX}.pth"

# Auto-load pretrained checkpoints from /openbayes/home/out or environment
PRETRAINED_PATH=""
find_pretrained_checkpoint() {
  local stage=$1
  local model_size=$2
  local moe_suffix="$MOE_SUFFIX"
  local stage_dir=""

  # Check environment variable first
  if [ -n "${MINILLM_PRETRAINED_PATH:-}" ] && [ -f "$MINILLM_PRETRAINED_PATH" ]; then
    echo "$MINILLM_PRETRAINED_PATH"
    return 0
  fi

  # Check /openbayes/home/out (OpenBayes environment)
  local remote_path="/openbayes/home/out/${stage}_${model_size}${moe_suffix}.pth"
  if [ -f "$remote_path" ]; then
    echo "$remote_path"
    return 0
  fi

  case "$stage" in
    pretrain) stage_dir="$PRETRAIN_OUT" ;;
    full_sft) stage_dir="$SFT_OUT" ;;
    rlhf) stage_dir="$DPO_OUT" ;;
    reason) stage_dir="$R1_OUT" ;;
    *) stage_dir="" ;;
  esac

  if [ -n "$stage_dir" ]; then
    local stage_path="$stage_dir/${stage}_${model_size}${moe_suffix}.pth"
    if [ -f "$stage_path" ]; then
      echo "$stage_path"
      return 0
    fi
  fi

  # Check local out directory
  local local_path="$OUT_DIR/${stage}_${model_size}${moe_suffix}.pth"
  if [ -f "$local_path" ]; then
    echo "$local_path"
    return 0
  fi

  return 1
}

is_valid_ckpt() {
  local ckpt_path=$1
  [ -s "$ckpt_path/model.pth" ] || return 1
  [ -s "$ckpt_path/optimizer.pt" ] || return 1
  [ -s "$ckpt_path/state.json" ] || return 1
  [ -s "$ckpt_path/rng_state.pt" ] || return 1
  return 0
}

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

link_legacy_checkpoint() {
  local src=$1
  local dest=$2
  if [ -f "$src" ] && [ "$src" != "$dest" ]; then
    ln -sf "$src" "$dest" || cp "$src" "$dest"
  fi
}

# Initialize PRETRAINED_PATH if available
PRETRAINED_PATH=$(find_pretrained_checkpoint "pretrain" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
if [ -n "$PRETRAINED_PATH" ]; then
  echo "[checkpoint] Found pretrained model at: $PRETRAINED_PATH"
  EXTRA_PRETRAIN_ARGS+=(--pretrained_path "$PRETRAINED_PATH")
fi

TB_PRETRAIN_DIR="$TF_DIR/pretrain"
TB_SFT_DIR="$TF_DIR/sft"
TB_DPO_DIR="$TF_DIR/dpo"
TB_EVAL_DIR="$TF_DIR/eval"

mkdir -p "$TB_PRETRAIN_DIR" "$TB_SFT_DIR" "$TB_DPO_DIR" "$TB_EVAL_DIR"
mkdir -p "$PRETRAIN_OUT" "$SFT_OUT" "$DPO_OUT" "$R1_OUT"

TB_AUTO=${TB_AUTO:-1}
TB_PORT=${TB_PORT:-6006}
TB_HOST=${TB_HOST:-127.0.0.1}
if [ -n "$TF_DIR" ] && [ "$TB_AUTO" != "0" ]; then
  if python -m tensorboard --version >/dev/null 2>&1; then
    TB_PORT_IN_USE=$(python - "$TB_HOST" "$TB_PORT" <<'PY'
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
      TB_PORT=$(python - "$TB_HOST" <<'PY'
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
    python -m tensorboard --logdir "$TF_DIR" --host "$TB_HOST" --port "$TB_PORT" >/dev/null 2>&1 &
    echo "[tensorboard] http://$TB_HOST:$TB_PORT (pid=$!)"
  else
    echo "[warn] TensorBoard not available; install tensorboard to enable TB_AUTO" >&2
  fi
fi

EVAL_CMD_BASE=(python scripts/evaluate_stage.py --hidden-size "$MODEL_HIDDEN_SIZE" --num-hidden-layers "$MODEL_NUM_LAYERS" --results-file "$RESULTS_FILE")

PRETRAIN_EVAL_MAX_SAMPLES=128
PRETRAIN_EVAL_BATCH=8
SFT_EVAL_MAX_SAMPLES=128
SFT_EVAL_BATCH=4
DPO_EVAL_MAX_SAMPLES=64
DPO_EVAL_BATCH=2

if [ "$SMOKE_TEST" -eq 1 ]; then
  EXTRA_PRETRAIN_ARGS+=(--device cpu --dtype float32 --batch_size "${SMOKE_PRETRAIN_BATCH:-2}" --accumulation_steps 1 --max_steps "${SMOKE_PRETRAIN_STEPS:-4}" --num_workers 0 --log_interval 1 --save_interval "${SMOKE_PRETRAIN_STEPS:-4}")
  EXTRA_SFT_ARGS+=(--device cpu --dtype float32 --batch_size "${SMOKE_SFT_BATCH:-2}" --max_steps "${SMOKE_SFT_STEPS:-4}" --num_workers 0 --log_interval 1 --save_interval "${SMOKE_SFT_STEPS:-4}")
  EXTRA_DPO_ARGS+=(--device cpu --dtype float32 --batch_size "${SMOKE_DPO_BATCH:-2}" --max_steps "${SMOKE_DPO_STEPS:-4}" --num_workers 0 --log_interval 1 --save_interval "${SMOKE_DPO_STEPS:-4}")
  EXTRA_R1_ARGS+=(--device cpu --dtype float32 --batch_size "${SMOKE_R1_BATCH:-2}" --epochs "${SMOKE_R1_EPOCHS:-1}" --num_workers 0 --log_interval 1 --save_interval "${SMOKE_R1_SAVE_INTERVAL:-2}" --max_seq_len "${SMOKE_R1_SEQ_LEN:-256}")

  PRETRAIN_EVAL_MAX_SAMPLES=${SMOKE_PRETRAIN_EVAL_SAMPLES:-8}
  PRETRAIN_EVAL_BATCH=${SMOKE_PRETRAIN_EVAL_BATCH:-2}
  SFT_EVAL_MAX_SAMPLES=${SMOKE_SFT_EVAL_SAMPLES:-8}
  SFT_EVAL_BATCH=${SMOKE_SFT_EVAL_BATCH:-2}
  DPO_EVAL_MAX_SAMPLES=${SMOKE_DPO_EVAL_SAMPLES:-4}
  DPO_EVAL_BATCH=${SMOKE_DPO_EVAL_BATCH:-1}

  EVAL_CMD_BASE+=(--device cpu)
  PREPROCESS_DATA=0  # Skip preprocessing for smoke test
else
  # GPU training with optimized settings
  EXTRA_PRETRAIN_ARGS+=(--save_interval "$SAVE_INTERVAL" --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUM_STEPS" --dtype bfloat16 $DATALOADER_ARGS)
  EXTRA_SFT_ARGS+=(--save_interval "$SAVE_INTERVAL" --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUM_STEPS" --dtype bfloat16 $DATALOADER_ARGS)
  EXTRA_DPO_ARGS+=(--save_interval "$SAVE_INTERVAL" --batch_size "$((BATCH_SIZE / 4))" --accumulation_steps "$((ACCUM_STEPS * 2))" --dtype bfloat16 $DATALOADER_ARGS)
  EXTRA_R1_ARGS+=(--save_interval "$SAVE_INTERVAL" --batch_size "$((BATCH_SIZE / 2))" --accumulation_steps "$ACCUM_STEPS" --dtype bfloat16 $DATALOADER_ARGS)
fi

EXTRA_PRETRAIN_ARGS+=(--keep_last_checkpoints "$KEEP_LAST")
EXTRA_SFT_ARGS+=(--keep_last_checkpoints "$KEEP_LAST")
EXTRA_DPO_ARGS+=(--keep_last_checkpoints "$KEEP_LAST")
EXTRA_R1_ARGS+=(--keep_last_checkpoints "$KEEP_LAST")

# Check if we should skip pretrain stage
SHOULD_SKIP_PRETRAIN=0

if [ "$FORCE_PRETRAIN" -eq 1 ]; then
  echo "[stage] Force pretrain mode enabled, will train from scratch"
  SHOULD_SKIP_PRETRAIN=0
elif [ "$SKIP_PRETRAIN" -eq 1 ]; then
  PRETRAIN_RESUME=$(latest_ckpt "$PRETRAIN_OUT")
  if [ -n "$PRETRAIN_RESUME" ]; then
    echo "[stage] Found pretrain step checkpoint: $PRETRAIN_RESUME"
    SHOULD_SKIP_PRETRAIN=1
  else
    # Check if pretrain checkpoint exists
    EXISTING_PRETRAIN=$(find_pretrained_checkpoint "pretrain" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)

    if [ -n "$EXISTING_PRETRAIN" ] && [ -f "$EXISTING_PRETRAIN" ]; then
      echo "[stage] Found existing pretrain checkpoint: $EXISTING_PRETRAIN"
      echo "[stage] Checking checkpoint quality..."

      # Quick quality check: verify checkpoint file is not corrupted and has reasonable size
      CHECKPOINT_SIZE=$(stat -f%z "$EXISTING_PRETRAIN" 2>/dev/null || stat -c%s "$EXISTING_PRETRAIN" 2>/dev/null || echo "0")
      MIN_SIZE=$((1024 * 100))  # At least 100KB

      if [ "$CHECKPOINT_SIZE" -gt "$MIN_SIZE" ]; then
        echo "[stage] Checkpoint looks valid (size: $((CHECKPOINT_SIZE / 1024))KB)"
        echo "[stage] Skipping pretrain stage, will use existing checkpoint for SFT"
        SHOULD_SKIP_PRETRAIN=1
        if [ "$EXISTING_PRETRAIN" != "$CHECKPOINT_PRETRAIN" ] && [ ! -f "$CHECKPOINT_PRETRAIN" ]; then
          echo "[stage] Linking checkpoint to stage output"
          link_legacy_checkpoint "$EXISTING_PRETRAIN" "$CHECKPOINT_PRETRAIN"
        fi
      else
        echo "[stage] Checkpoint appears corrupted or too small, will retrain"
        SHOULD_SKIP_PRETRAIN=0
      fi
    else
      echo "[stage] No pretrain checkpoint found, will train from scratch"
      SHOULD_SKIP_PRETRAIN=0
    fi
  fi
fi

if [ "$SHOULD_SKIP_PRETRAIN" -eq 0 ]; then
  echo "[stage] Starting pretrain (2 epochs)"

  # Preprocess data to bin2d for faster loading
  PRETRAIN_DATA_PATH="$PRETRAIN_JSON"
  PRETRAIN_DATA_FORMAT_ARG=""
  if [ "$PREPROCESS_DATA" -ge 1 ] && [ "$SMOKE_TEST" -eq 0 ]; then
    PRETRAIN_BIN_PREFIX="$PRETRAIN_OUT/data_bin2d"
    PRETRAIN_DATA_PATH=$(preprocess_to_bin2d "$PRETRAIN_JSON" "$PRETRAIN_BIN_PREFIX" "pretrain" "$MODEL_SEQ_LEN")
    if [[ "$PRETRAIN_DATA_PATH" == *.meta.json ]]; then
      PRETRAIN_DATA_FORMAT_ARG="--data_format bin2d"
    fi
  fi

  PRETRAIN_RESUME=$(latest_ckpt "$PRETRAIN_OUT")
  PRETRAIN_ARGS_WITH_RESUME=("${EXTRA_PRETRAIN_ARGS[@]}")
  if [ -n "$PRETRAIN_RESUME" ]; then
    PRETRAIN_ARGS_WITH_RESUME+=(--resume "$PRETRAIN_RESUME")
  fi
  PRETRAIN_ARGS_WITH_RESUME+=(--ckpt_dir "$PRETRAIN_OUT/checkpoints")
  $TRAIN_CMD_PREFIX trainer/train_pretrain.py --data_path "$PRETRAIN_DATA_PATH" $PRETRAIN_DATA_FORMAT_ARG --hidden_size "$MODEL_HIDDEN_SIZE" --num_hidden_layers "$MODEL_NUM_LAYERS" --max_seq_len "$MODEL_SEQ_LEN" --epochs 2 --out_dir "$PRETRAIN_OUT" --tensorboard_dir "$TB_PRETRAIN_DIR" ${PRETRAIN_ARGS_WITH_RESUME[@]+"${PRETRAIN_ARGS_WITH_RESUME[@]}"}

  if [ -f "$CHECKPOINT_PRETRAIN" ]; then
    link_legacy_checkpoint "$CHECKPOINT_PRETRAIN" "$LEGACY_PRETRAIN"
    echo "[eval] Pretrain evaluation"
    "${EVAL_CMD_BASE[@]}" --stage pretrain --checkpoint "$CHECKPOINT_PRETRAIN" --data-path "$PRETRAIN_JSON" --max-seq-len 512 --max-samples "$PRETRAIN_EVAL_MAX_SAMPLES" --batch-size "$PRETRAIN_EVAL_BATCH" --tensorboard-dir "$TB_EVAL_DIR/pretrain"
  else
    echo "[warn] Pretrain checkpoint not found at $CHECKPOINT_PRETRAIN" >&2
  fi
else
  echo "[stage] Pretrain stage skipped"
  if [ -f "$CHECKPOINT_PRETRAIN" ]; then
    link_legacy_checkpoint "$CHECKPOINT_PRETRAIN" "$LEGACY_PRETRAIN"
    echo "[eval] Running quick evaluation on existing pretrain checkpoint"
    "${EVAL_CMD_BASE[@]}" --stage pretrain --checkpoint "$CHECKPOINT_PRETRAIN" --data-path "$PRETRAIN_JSON" --max-seq-len 512 --max-samples "$PRETRAIN_EVAL_MAX_SAMPLES" --batch-size "$PRETRAIN_EVAL_BATCH" --tensorboard-dir "$TB_EVAL_DIR/pretrain" || echo "[warn] Pretrain evaluation failed, but continuing"
  fi
fi

echo "[stage] Starting SFT"
# Auto-load SFT pretrained checkpoint
# Priority: pretrain checkpoint (for initial SFT training) > full_sft checkpoint (for resume)
SFT_PRETRAINED_PATH=$(find_pretrained_checkpoint "pretrain" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
SFT_ARGS_WITH_PRETRAIN=("${EXTRA_SFT_ARGS[@]}")
if [ -z "$SFT_PRETRAINED_PATH" ]; then
  # If no pretrain checkpoint, try full_sft checkpoint (resume scenario)
  SFT_PRETRAINED_PATH=$(find_pretrained_checkpoint "full_sft" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
fi
if [ -n "$SFT_PRETRAINED_PATH" ]; then
  echo "[checkpoint] Using pretrained model for SFT: $SFT_PRETRAINED_PATH"
  SFT_ARGS_WITH_PRETRAIN+=(--pretrained_path "$SFT_PRETRAINED_PATH")
fi
SFT_RESUME=$(latest_ckpt "$SFT_OUT")
if [ -n "$SFT_RESUME" ]; then
  SFT_ARGS_WITH_PRETRAIN+=(--resume "$SFT_RESUME")
fi
SFT_ARGS_WITH_PRETRAIN+=(--ckpt_dir "$SFT_OUT/checkpoints")
$TRAIN_CMD_PREFIX trainer/train_full_sft.py --data_path "$SFT_JSON" --hidden_size "$MODEL_HIDDEN_SIZE" --num_hidden_layers "$MODEL_NUM_LAYERS" --out_dir "$SFT_OUT" --tensorboard_dir "$TB_SFT_DIR" ${SFT_ARGS_WITH_PRETRAIN[@]+"${SFT_ARGS_WITH_PRETRAIN[@]}"}

if [ -f "$CHECKPOINT_SFT" ]; then
  link_legacy_checkpoint "$CHECKPOINT_SFT" "$LEGACY_SFT"
  echo "[eval] SFT evaluation"
  "${EVAL_CMD_BASE[@]}" --stage sft --checkpoint "$CHECKPOINT_SFT" --data-path "$SFT_JSON" --max-seq-len 512 --max-samples "$SFT_EVAL_MAX_SAMPLES" --batch-size "$SFT_EVAL_BATCH" --tensorboard-dir "$TB_EVAL_DIR/sft"
else
  echo "[warn] SFT checkpoint not found at $CHECKPOINT_SFT" >&2
fi

echo "[stage] Starting DPO"
# Auto-load DPO pretrained checkpoint
DPO_PRETRAINED_PATH=$(find_pretrained_checkpoint "full_sft" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
DPO_ARGS_WITH_PRETRAIN=("${EXTRA_DPO_ARGS[@]}")
if [ -z "$DPO_PRETRAINED_PATH" ]; then
  # If no full_sft checkpoint, try rlhf checkpoint
  DPO_PRETRAINED_PATH=$(find_pretrained_checkpoint "rlhf" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
fi
if [ -z "$DPO_PRETRAINED_PATH" ]; then
  # If no rlhf checkpoint, try pretrain checkpoint
  DPO_PRETRAINED_PATH=$(find_pretrained_checkpoint "pretrain" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
fi
if [ -n "$DPO_PRETRAINED_PATH" ]; then
  echo "[checkpoint] Using pretrained model for DPO: $DPO_PRETRAINED_PATH"
  DPO_ARGS_WITH_PRETRAIN+=(--pretrained_path "$DPO_PRETRAINED_PATH")
fi
DPO_RESUME=$(latest_ckpt "$DPO_OUT")
if [ -n "$DPO_RESUME" ]; then
  DPO_ARGS_WITH_PRETRAIN+=(--resume "$DPO_RESUME")
fi
DPO_ARGS_WITH_PRETRAIN+=(--ckpt_dir "$DPO_OUT/checkpoints")
$TRAIN_CMD_PREFIX trainer/train_dpo.py --data_path "$DPO_JSON" --hidden_size "$MODEL_HIDDEN_SIZE" --num_hidden_layers "$MODEL_NUM_LAYERS" --out_dir "$DPO_OUT" --tensorboard_dir "$TB_DPO_DIR" ${DPO_ARGS_WITH_PRETRAIN[@]+"${DPO_ARGS_WITH_PRETRAIN[@]}"}

if [ -f "$CHECKPOINT_DPO" ]; then
  link_legacy_checkpoint "$CHECKPOINT_DPO" "$LEGACY_DPO"
  echo "[eval] DPO evaluation"
  "${EVAL_CMD_BASE[@]}" --stage dpo --checkpoint "$CHECKPOINT_DPO" --data-path "$DPO_JSON" --max-seq-len 1024 --max-samples "$DPO_EVAL_MAX_SAMPLES" --batch-size "$DPO_EVAL_BATCH" --tensorboard-dir "$TB_EVAL_DIR/dpo"
else
  echo "[warn] DPO checkpoint not found at $CHECKPOINT_DPO" >&2
fi

if [ "$RUN_R1" -eq 1 ]; then
  if [ ! -s "$R1_JSON" ]; then
    echo "[error] R1 dataset not found at $R1_JSON" >&2
    exit 1
  fi
  if [ ! -f "$CHECKPOINT_DPO" ] && [ -f "$LEGACY_DPO" ]; then
    link_legacy_checkpoint "$LEGACY_DPO" "$CHECKPOINT_DPO"
  fi
  if [ ! -f "$CHECKPOINT_DPO" ]; then
    echo "[error] R1 requires DPO checkpoint at $CHECKPOINT_DPO" >&2
    exit 1
  fi
  echo "[stage] Starting R1 distillation (reasoning)"
  R1_ARGS_WITH_PRETRAIN=("${EXTRA_R1_ARGS[@]}")
  R1_RESUME=$(latest_ckpt "$R1_OUT")
  if [ -n "$R1_RESUME" ]; then
    R1_ARGS_WITH_PRETRAIN+=(--resume "$R1_RESUME")
  fi
  R1_ARGS_WITH_PRETRAIN+=(--ckpt_dir "$R1_OUT/checkpoints" --pretrained_path "$CHECKPOINT_DPO")
  $TRAIN_CMD_PREFIX trainer/train_distill_reason.py --data_path "$R1_JSON" --hidden_size "$MODEL_HIDDEN_SIZE" --num_hidden_layers "$MODEL_NUM_LAYERS" --out_dir "$R1_OUT" ${R1_ARGS_WITH_PRETRAIN[@]+"${R1_ARGS_WITH_PRETRAIN[@]}"}

  if [ -f "$CHECKPOINT_R1" ]; then
    link_legacy_checkpoint "$CHECKPOINT_R1" "$LEGACY_R1"
    R1_DEMO=${R1_DEMO:-1}
    if [ "$R1_DEMO" -ne 0 ]; then
      R1_DEMO_PROMPT=${R1_DEMO_PROMPT:-"Compute 23 * 17. Show reasoning and give the final answer."}
      R1_DEMO_MAX_NEW_TOKENS=${R1_DEMO_MAX_NEW_TOKENS:-256}
      R1_DEMO_TEMPERATURE=${R1_DEMO_TEMPERATURE:-0.7}
      R1_DEMO_TOP_P=${R1_DEMO_TOP_P:-0.9}
      R1_DEMO_SEED=${R1_DEMO_SEED:-1337}
      if ! R1_DEMO_CHECKPOINT="$CHECKPOINT_R1" \
        R1_DEMO_PROMPT="$R1_DEMO_PROMPT" \
        R1_DEMO_MAX_NEW_TOKENS="$R1_DEMO_MAX_NEW_TOKENS" \
        R1_DEMO_TEMPERATURE="$R1_DEMO_TEMPERATURE" \
        R1_DEMO_TOP_P="$R1_DEMO_TOP_P" \
        R1_DEMO_SEED="$R1_DEMO_SEED" \
        R1_DEMO_HIDDEN_SIZE="$MODEL_HIDDEN_SIZE" \
        R1_DEMO_NUM_LAYERS="$MODEL_NUM_LAYERS" \
        R1_DEMO_USE_MOE="$USE_MOE" \
        python - <<'PY'
import os
import torch
from transformers import AutoTokenizer
from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM

ckpt = os.environ["R1_DEMO_CHECKPOINT"]
prompt = os.environ["R1_DEMO_PROMPT"]
max_new_tokens = int(os.environ.get("R1_DEMO_MAX_NEW_TOKENS", "256"))
temperature = float(os.environ.get("R1_DEMO_TEMPERATURE", "0.7"))
top_p = float(os.environ.get("R1_DEMO_TOP_P", "0.9"))
seed = int(os.environ.get("R1_DEMO_SEED", "1337"))
hidden_size = int(os.environ["R1_DEMO_HIDDEN_SIZE"])
num_layers = int(os.environ["R1_DEMO_NUM_LAYERS"])
use_moe = os.environ.get("R1_DEMO_USE_MOE", "false").lower() == "true"

device = os.environ.get("R1_DEMO_DEVICE")
if not device:
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("./model")
cfg = MiniLLMConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, use_moe=use_moe)
model = MiniLLMForCausalLM(cfg)
state = torch.load(ckpt, map_location=device)
model.load_state_dict(state, strict=False)
model.eval().to(device)

messages = [{"role": "user", "content": prompt}]
try:
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
except TypeError:
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(device)
do_sample = temperature > 0
with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print("[r1-demo] prompt:", prompt)
print("[r1-demo] response:", response or "<empty>")
PY
      then
        echo "[warn] R1 demo failed, but training completed" >&2
      fi
    fi
  else
    echo "[warn] R1 checkpoint not found at $CHECKPOINT_R1" >&2
  fi
fi

echo "[done] Training pipeline completed. Check $OUT_DIR for checkpoints and $RESULTS_FILE for evaluation logs."
