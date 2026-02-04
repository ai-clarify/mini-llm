"""Utilities for step-based checkpoints with optimizer and RNG state."""
from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, Optional

import torch

_STEP_RE = re.compile(r"^step_(\d+)$")


def _step_dir(step: int) -> str:
    return f"step_{step:08d}"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def is_valid_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in ("model.pth", "optimizer.pt", "state.json", "rng_state.pt"):
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
    for _, path in items[:-keep_last]:
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
    model_state = _unwrap_model(model).state_dict()
    torch.save(model_state, os.path.join(step_dir, "model.pth"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
        os.path.join(step_dir, "optimizer.pt"),
    )
    with open(os.path.join(step_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    torch.save(_capture_rng_state(), os.path.join(step_dir, "rng_state.pt"))
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
    rng_path = os.path.join(ckpt_path, "rng_state.pt")

    model_state = torch.load(model_path, map_location=device)
    _unwrap_model(model).load_state_dict(model_state, strict=False)

    opt_state = torch.load(opt_path, map_location=device)
    try:
        optimizer.load_state_dict(opt_state.get("optimizer", {}))
    except ValueError as e:
        # Optimizer type changed (e.g., AdamW -> Muon), skip loading optimizer state
        print(f"[checkpoint] Warning: Optimizer state mismatch, starting fresh optimizer: {e}")
    scaler.load_state_dict(opt_state.get("scaler", {}))

    if os.path.isfile(rng_path):
        rng_state = torch.load(rng_path, map_location="cpu")
        if isinstance(rng_state, dict):
            _restore_rng_state(rng_state)

    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)
