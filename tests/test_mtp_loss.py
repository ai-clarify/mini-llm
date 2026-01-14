"""Unit tests for MTP loss utilities."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.loss_utils import compute_mtp_loss


def _make_logits(*, batch: int = 1, seq_len: int = 5, vocab: int = 5) -> torch.Tensor:
    return torch.zeros((batch, seq_len, vocab), dtype=torch.float32)


def test_compute_mtp_loss_no_logits() -> None:
    labels = torch.tensor([[0, 1, 2, 3, 4]])
    loss = compute_mtp_loss(None, labels, None, weight=0.5)
    assert float(loss) == 0.0


def test_compute_mtp_loss_zero_weight() -> None:
    labels = torch.tensor([[0, 1, 2, 3, 4]])
    logits = [_make_logits()]
    loss = compute_mtp_loss(logits, labels, None, weight=0.0)
    assert float(loss) == 0.0


def test_compute_mtp_loss_uniform_logits() -> None:
    vocab = 5
    labels = torch.tensor([[0, 1, 2, 3, 4]])
    logits = [_make_logits(vocab=vocab), _make_logits(vocab=vocab)]
    loss = compute_mtp_loss(logits, labels, None, weight=0.5)
    expected = math.log(vocab) * 0.5
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_compute_mtp_loss_zero_mask() -> None:
    labels = torch.tensor([[0, 1, 2, 3, 4]])
    logits = [_make_logits()]
    loss_mask = torch.zeros_like(labels, dtype=torch.float32)
    loss = compute_mtp_loss(logits, labels, loss_mask, weight=1.0)
    assert float(loss) == 0.0
