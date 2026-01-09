#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x ".venv_mlx/bin/python" ]; then
    PY=".venv_mlx/bin/python"
  else
    PY="python3"
  fi
fi

export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

exec "$PY" -m mlx_train.bench "$@"

