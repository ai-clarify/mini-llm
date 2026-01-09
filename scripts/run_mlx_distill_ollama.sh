#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$ROOT_DIR/pipelines/run_mlx_distill_ollama.sh" "$@"
