#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-${STAGE:-sft}}
OUT_ROOTS_DEFAULT=(out/mlx out/mlx_smoke out/mlx_smoke_keep out)

if [ "$STAGE" = "auto" ] || [ "$STAGE" = "any" ]; then
  STAGE=""
fi

if [ -n "${OUT_ROOTS:-}" ]; then
  # OUT_ROOTS="out/mlx out/mlx_smoke" (space-separated)
  read -r -a OUT_ROOTS_ARR <<<"${OUT_ROOTS}"
else
  OUT_ROOTS_ARR=("${OUT_ROOTS_DEFAULT[@]}")
  if [ -n "${OUT:-}" ]; then
    OUT_ROOTS_ARR+=("$OUT")
  fi
  if [ -n "${OUT_DIR:-}" ]; then
    OUT_ROOTS_ARR+=("$OUT_DIR")
  fi
fi

is_valid_ckpt() {
  local ckpt=$1
  if [ ! -s "$ckpt/model.safetensors" ] && [ ! -s "$ckpt/weights.npz" ]; then
    return 1
  fi
  [ -s "$ckpt/config.json" ] || return 1
  return 0
}

latest_ckpt_in() {
  local base=$1
  local ckpt
  if [ ! -d "$base" ]; then
    return 1
  fi
  while IFS= read -r ckpt; do
    [ -z "$ckpt" ] && continue
    if is_valid_ckpt "$ckpt"; then
      echo "$ckpt"
      return 0
    fi
  done < <(ls -dt "$base"/step_* 2>/dev/null || true)
  return 1
}

for root in "${OUT_ROOTS_ARR[@]}"; do
  if is_valid_ckpt "$root"; then
    echo "$root"
    exit 0
  fi
  if [ -n "$STAGE" ]; then
    ckpt=$(latest_ckpt_in "$root/$STAGE/checkpoints" || true)
    if [ -n "${ckpt:-}" ]; then
      echo "$ckpt"
      exit 0
    fi
  fi
done

collect_candidates() {
  local root=$1
  local file
  if command -v rg >/dev/null 2>&1; then
    while IFS= read -r file; do
      [ -z "$file" ] && continue
      echo "$(dirname "$file")"
    done < <(rg --files -g 'model.safetensors' -g 'weights.npz' "$root" 2>/dev/null || true)
  else
    while IFS= read -r file; do
      [ -z "$file" ] && continue
      echo "$(dirname "$file")"
    done < <(find "$root" -type f \( -name model.safetensors -o -name weights.npz \) 2>/dev/null || true)
  fi
}

filter_candidates() {
  local stage=$1
  local line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    if is_valid_ckpt "$line"; then
      if [ -n "$stage" ]; then
        case "$line" in
          *"/$stage/"*) echo "$line" ;;
        esac
      else
        echo "$line"
      fi
    fi
  done
}

pick_latest() {
  local lines
  lines=$(cat)
  if [ -z "$lines" ]; then
    return 1
  fi
  # Sort by mtime (newest first). Assumes no spaces in paths.
  ls -dt $lines 2>/dev/null | head -n 1
}

ALL_CANDIDATES=""
for root in "${OUT_ROOTS_ARR[@]}"; do
  if [ -d "$root" ]; then
    ALL_CANDIDATES+=$(collect_candidates "$root")
    ALL_CANDIDATES+=$'\n'
  fi
done

FOUND_ANY=""
FOUND_STAGE=""
if [ -n "$ALL_CANDIDATES" ]; then
  uniq=$(printf "%s\n" "$ALL_CANDIDATES" | awk 'NF' | sort -u)
  if [ -n "$STAGE" ]; then
    FOUND_STAGE=$(printf "%s\n" "$uniq" | filter_candidates "$STAGE" | tr '\n' ' ')
  fi
  FOUND_ANY=$(printf "%s\n" "$uniq" | filter_candidates "" | tr '\n' ' ')
  latest=$(printf "%s\n" "$uniq" | filter_candidates "$STAGE" | pick_latest || true)
  if [ -n "${latest:-}" ]; then
    echo "$latest"
    exit 0
  fi
fi

cat >&2 <<EOF
[error] No valid MLX checkpoint found.
Tried stage '${STAGE:-any}' under: ${OUT_ROOTS_ARR[*]}
Expecting: <root>/{stage}/checkpoints/step_*/model.safetensors + config.json
Weights may also be stored as weights.npz for mlx-lm checkpoints.
$(if [ -n "$STAGE" ] && [ -n "$FOUND_ANY" ]; then
  printf "Found valid checkpoints outside stage '%s': %s\n" "$STAGE" "$FOUND_ANY"
fi)
EOF
exit 1
