# Run.sh MLX-Style Checkpoints Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add MLX-style step checkpoints with auto-resume and auto-start TensorBoard for the Torch pipeline, while keeping legacy `out/*.pth` outputs.

**Architecture:** Introduce a shared Torch checkpoint manager for step-based saves/resume; update Torch training scripts to use it; align `pipelines/run.sh` stage outputs and resume detection with MLX; add TensorBoard auto-launch with opt-out env vars.

**Tech Stack:** Bash, Python, PyTorch, pytest.

### Task 1: Add failing tests for Torch checkpoint manager

**Files:**
- Create: `tests/test_checkpoint_manager.py`

**Step 1: Write the failing test**

```python
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer import checkpoint_manager


def test_latest_checkpoint_prefers_highest_step(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "step_00000002").mkdir()
    (ckpt_dir / "step_00000005").mkdir()
    (ckpt_dir / "step_00000005" / "model.pth").write_bytes(b"x")
    (ckpt_dir / "step_00000005" / "optimizer.pt").write_bytes(b"x")
    (ckpt_dir / "step_00000005" / "state.json").write_text("{}")

    latest = checkpoint_manager.latest_checkpoint(str(ckpt_dir))
    assert latest is not None
    assert latest.endswith("step_00000005")


def test_prune_checkpoints_keeps_last_n(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for step in (1, 2, 3):
        step_dir = ckpt_dir / f"step_{step:08d}"
        step_dir.mkdir()
        (step_dir / "model.pth").write_bytes(b"x")
        (step_dir / "optimizer.pt").write_bytes(b"x")
        (step_dir / "state.json").write_text("{}")

    checkpoint_manager.prune_checkpoints(str(ckpt_dir), keep_last=1)

    remaining = [p.name for p in ckpt_dir.iterdir() if p.is_dir()]
    assert remaining == ["step_00000003"]


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    state = {"epoch": 1, "step_in_epoch": 2, "global_step": 3}
    path = checkpoint_manager.save_checkpoint(
        ckpt_root=str(ckpt_dir),
        step=3,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        state=state,
    )

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    loaded_state = checkpoint_manager.load_checkpoint(
        ckpt_path=path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device="cpu",
    )

    assert loaded_state["epoch"] == 1
    assert loaded_state["step_in_epoch"] == 2
    assert loaded_state["global_step"] == 3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_checkpoint_manager.py -v`

Expected: FAIL (module `trainer.checkpoint_manager` missing).

### Task 2: Implement Torch checkpoint manager

**Files:**
- Create: `trainer/checkpoint_manager.py`

**Step 1: Write minimal implementation**

```python
import json
import os
import re
from typing import Any, Dict, Optional

import torch

_STEP_RE = re.compile(r"^step_(\d+)$")


def _step_dir(step: int) -> str:
    return f"step_{step:08d}"


def is_valid_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in ("model.pth", "optimizer.pt", "state.json"):
        file_path = os.path.join(path, name)
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False
    return True


def latest_checkpoint(ckpt_root: str) -> Optional[str]:
    if not os.path.isdir(ckpt_root):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(ckpt_root):
        match = _STEP_RE.match(name)
        if not match:
            continue
        step = int(match.group(1))
        path = os.path.join(ckpt_root, name)
        if not is_valid_checkpoint(path):
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_path = path
    return best_path


def prune_checkpoints(ckpt_root: str, keep_last: int) -> None:
    if keep_last <= 0 or not os.path.isdir(ckpt_root):
        return
    items = []
    for name in os.listdir(ckpt_root):
        match = _STEP_RE.match(name)
        if not match:
            continue
        step = int(match.group(1))
        path = os.path.join(ckpt_root, name)
        if not is_valid_checkpoint(path):
            continue
        items.append((step, path))
    items.sort(key=lambda x: x[0])
    for step, path in items[:-keep_last]:
        for root, dirs, files in os.walk(path, topdown=False):
            for fname in files:
                os.remove(os.path.join(root, fname))
            for dname in dirs:
                os.rmdir(os.path.join(root, dname))
        os.rmdir(path)


def save_checkpoint(
    *,
    ckpt_root: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    state: Dict[str, Any],
) -> str:
    step_dir = os.path.join(ckpt_root, _step_dir(step))
    os.makedirs(step_dir, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, os.path.join(step_dir, "model.pth"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
        os.path.join(step_dir, "optimizer.pt"),
    )
    with open(os.path.join(step_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    return step_dir


def load_checkpoint(
    *,
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
) -> Dict[str, Any]:
    model_path = os.path.join(ckpt_path, "model.pth")
    opt_path = os.path.join(ckpt_path, "optimizer.pt")
    state_path = os.path.join(ckpt_path, "state.json")
    model_state = torch.load(model_path, map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(model_state, strict=False)
    opt_state = torch.load(opt_path, map_location=device)
    optimizer.load_state_dict(opt_state.get("optimizer", {}))
    scaler.load_state_dict(opt_state.get("scaler", {}))
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)
```

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_checkpoint_manager.py -v`

Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_checkpoint_manager.py trainer/checkpoint_manager.py
git commit -m "test: add torch checkpoint manager tests"
```

### Task 3: Integrate step checkpoints into Torch training scripts

**Files:**
- Modify: `trainer/train_pretrain.py`
- Modify: `trainer/train_full_sft.py`
- Modify: `trainer/train_dpo.py`
- Modify: `trainer/train_distill_reason.py`

**Step 1: Add new args and ckpt_dir setup (all scripts)**

```python
parser.add_argument("--resume", type=str, default=None, help="Path to step checkpoint dir")
parser.add_argument("--ckpt_dir", type=str, default=None, help="Root dir for step checkpoints")
parser.add_argument("--keep_last_checkpoints", type=int, default=3)
```

```python
ckpt_root = args.ckpt_dir or os.path.join(args.out_dir, "checkpoints")
os.makedirs(ckpt_root, exist_ok=True)
```

**Step 2: Load resume state when provided**

```python
resume_state = None
start_epoch = 0
start_step = 0
if args.resume:
    resume_state = checkpoint_manager.load_checkpoint(
        ckpt_path=args.resume,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=args.device,
    )
    start_epoch = int(resume_state.get("epoch", 0))
    start_step = int(resume_state.get("step_in_epoch", 0))
    training_state["global_step"] = int(resume_state.get("global_step", 0))
```

**Step 3: Save step checkpoints with keep-last pruning**

```python
state = {
    "epoch": epoch,
    "step_in_epoch": step,
    "global_step": training_state["global_step"],
    "args": vars(args),
}
checkpoint_manager.save_checkpoint(
    ckpt_root=ckpt_root,
    step=training_state["global_step"],
    model=model,
    optimizer=optimizer,
    scaler=scaler,
    state=state,
)
checkpoint_manager.prune_checkpoints(ckpt_root, keep_last=args.keep_last_checkpoints)
```

**Step 4: Keep legacy `out/*_{hidden}.pth` saves**

Keep the existing `.pth` saves for compatibility (copy as-is from current logic).

**Step 5: Run tests**

Run: `python -m pytest tests/test_checkpoint_manager.py -v`

Expected: PASS.

**Step 6: Commit**

```bash
git add trainer/train_pretrain.py trainer/train_full_sft.py trainer/train_dpo.py trainer/train_distill_reason.py
git commit -m "feat: add step checkpoints and resume to torch trainers"
```

### Task 4: Align `pipelines/run.sh` with MLX checkpoint flow + TensorBoard auto-start

**Files:**
- Modify: `pipelines/run.sh`
- Modify: `docs/run_script_options.md`

**Step 1: Add stage output dirs and latest_ckpt helpers**

```bash
PRETRAIN_OUT="$OUT_DIR/pretrain"
SFT_OUT="$OUT_DIR/sft"
DPO_OUT="$OUT_DIR/dpo"
R1_OUT="$OUT_DIR/r1"

latest_ckpt() {
  local stage_dir=$1
  local ckpt
  while IFS= read -r ckpt; do
    [ -z "$ckpt" ] && continue
    if is_valid_ckpt "$ckpt"; then
      echo "$ckpt"
      return 0
    fi
  done < <(ls -dt "$stage_dir"/checkpoints/step_* 2>/dev/null || true)
  return 0
}

is_valid_ckpt() {
  local ckpt_path=$1
  [ -s "$ckpt_path/model.pth" ] || return 1
  [ -s "$ckpt_path/optimizer.pt" ] || return 1
  [ -s "$ckpt_path/state.json" ] || return 1
}
```

**Step 2: Wire auto-resume per stage**

```bash
PRETRAIN_RESUME=$(latest_ckpt "$PRETRAIN_OUT")
if [ -n "$PRETRAIN_RESUME" ]; then
  EXTRA_PRETRAIN_ARGS+=(--resume "$PRETRAIN_RESUME" --ckpt_dir "$PRETRAIN_OUT/checkpoints")
fi
```

Apply similar logic for SFT/DPO/R1, and pass `--keep_last_checkpoints "$KEEP_LAST"` and `--save_interval "$SAVE_INTERVAL"`.

**Step 3: Add TensorBoard auto-start**

```bash
TB_AUTO=${TB_AUTO:-1}
TB_PORT=${TB_PORT:-6006}
TB_HOST=${TB_HOST:-127.0.0.1}
if [ -n "$TF_DIR" ] && [ "$TB_AUTO" != "0" ]; then
  TB_LOGDIR="$TF_DIR"
  python -m tensorboard --logdir "$TB_LOGDIR" --host "$TB_HOST" --port "$TB_PORT" &
  echo "[tensorboard] http://$TB_HOST:$TB_PORT (pid=$!)"
fi
```

**Step 4: Update docs**

Add environment variables and new checkpoint layout to `docs/run_script_options.md`.

**Step 5: Manual verification**

Run: `SAVE_INTERVAL=2 KEEP_LAST=2 TB_AUTO=0 scripts/run.sh --smoke-test`

Expected: step checkpoints appear under `out/pretrain/checkpoints/step_*/` and
script logs show `--resume` on re-run.

**Step 6: Commit**

```bash
git add pipelines/run.sh docs/run_script_options.md
git commit -m "feat: align run.sh checkpoints and tensorboard auto-start"
```
