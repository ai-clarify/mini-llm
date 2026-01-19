# Run Series TensorBoard Auto-Start Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-start TensorBoard for run-series scripts only when they actually emit TB logs.

**Architecture:** Add a small auto-start block to `pipelines/run_mlx.sh` gated by `TF_DIR` and “will-train” flags, and add conditional auto-start to `pipelines/run_mlx_distill_ollama.sh` when `--tensorboard_dir` is passed. Use the same Python executable as training, and keep defaults aligned with the Torch pipeline.

**Tech Stack:** Bash scripts, Python standard library (for a lightweight static test).

### Task 1: Add a static test for TensorBoard auto-start hooks

**Files:**
- Create: `tests/test_tensorboard_autostart.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def assert_contains(path: str, needle: str) -> None:
    data = read(path)
    assert needle in data, f"{path} missing: {needle}"


def main() -> None:
    assert_contains("pipelines/run_mlx.sh", "# [tb] auto-start")
    assert_contains("pipelines/run_mlx.sh", "TB_AUTO")
    assert_contains("pipelines/run_mlx.sh", "tensorboard --logdir")
    assert_contains("pipelines/run_mlx_distill_ollama.sh", "# [tb] auto-start")
    assert_contains("pipelines/run_mlx_distill_ollama.sh", "--tensorboard_dir")
    assert_contains("pipelines/run_mlx_distill_ollama.sh", "TB_AUTO")


if __name__ == "__main__":
    main()
```

**Step 2: Run test to verify it fails**

Run: `python tests/test_tensorboard_autostart.py`
Expected: FAIL with an assertion like `pipelines/run_mlx.sh missing: # [tb] auto-start`

### Task 2: Auto-start TensorBoard in MLX pipeline

**Files:**
- Modify: `pipelines/run_mlx.sh`
- Modify: `mlx_train/README.md`

**Step 1: Implement MLX auto-start block**

Add to the usage text (env overrides):

```bash
  TB_AUTO           Auto-start TensorBoard when training (default: 1; set 0 to disable)
  TB_HOST           TensorBoard host (default: 127.0.0.1)
  TB_PORT           TensorBoard port (default: 6006)
```

Add the auto-start block after `TF_DIR` is finalized and TB dirs are created:

```bash
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
```

**Step 2: Update MLX README**

Add a short paragraph in `mlx_train/README.md` under “TensorBoard 训练曲线” to mention auto-start and the env vars, e.g.:

```bash
# 自动启动（默认开启）
TB_AUTO=1 bash scripts/run_mlx.sh

# 关闭自动启动
TB_AUTO=0 bash scripts/run_mlx.sh

# 自定义 host/port
TB_HOST=0.0.0.0 TB_PORT=6007 bash scripts/run_mlx.sh
```

**Step 3: Run test to verify progress**

Run: `python tests/test_tensorboard_autostart.py`
Expected: FAIL (distill script not updated yet)

### Task 3: Auto-start TensorBoard in distill script when enabled

**Files:**
- Modify: `pipelines/run_mlx_distill_ollama.sh`

**Step 1: Add arg parsing + auto-start**

Add a helper to extract `--tensorboard_dir` (supports `--tensorboard_dir=...`) and start TensorBoard only when present:

```bash
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
```

**Step 2: Run test to verify it passes**

Run: `python tests/test_tensorboard_autostart.py`
Expected: PASS

### Task 4: Final verification + commit

**Step 1: Optional manual smoke check**

Run (heavy): `TB_AUTO=1 bash scripts/run_mlx.sh --smoke-test`
Expected: prints a `[tensorboard] http://...` line.

**Step 2: Commit**

```bash
git add tests/test_tensorboard_autostart.py pipelines/run_mlx.sh pipelines/run_mlx_distill_ollama.sh mlx_train/README.md docs/plans/2026-01-19-run-series-tensorboard-design.md docs/plans/2026-01-19-run-series-tensorboard-plan.md
git commit -m "feat: auto-start tensorboard for mlx run scripts"
```
