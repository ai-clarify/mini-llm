"""Unit tests for torch checkpoint manager utilities."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer import checkpoint_manager


def _write_ckpt_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.pth").write_bytes(b"x")
    (path / "optimizer.pt").write_bytes(b"x")
    (path / "state.json").write_text("{}", encoding="utf-8")
    (path / "rng_state.pt").write_bytes(b"x")


def test_latest_checkpoint_prefers_highest_step(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    _write_ckpt_files(ckpt_dir / "step_00000002")
    _write_ckpt_files(ckpt_dir / "step_00000005")

    latest = checkpoint_manager.latest_checkpoint(str(ckpt_dir))
    assert latest is not None
    assert latest.endswith("step_00000005")


def test_prune_checkpoints_keeps_last_n(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        _write_ckpt_files(ckpt_dir / f"step_{step:08d}")

    checkpoint_manager.prune_checkpoints(str(ckpt_dir), keep_last=1)

    remaining = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir())
    assert remaining == ["step_00000003"]


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    torch.manual_seed(1234)
    random.seed(1234)
    _ = torch.rand(1)

    state = {"epoch": 1, "step_in_epoch": 2, "global_step": 3}
    path = checkpoint_manager.save_checkpoint(
        ckpt_root=str(ckpt_dir),
        step=3,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        state=state,
    )

    assert (Path(path) / "rng_state.pt").is_file()

    expected_next = torch.rand(1)
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

    actual_next = torch.rand(1)
    assert torch.allclose(actual_next, expected_next)
    assert loaded_state["epoch"] == 1
    assert loaded_state["step_in_epoch"] == 2
    assert loaded_state["global_step"] == 3
