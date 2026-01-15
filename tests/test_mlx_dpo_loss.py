"""MLX-only unit tests for DPO loss helpers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip(
    "mlx.core",
    reason="MLX is not available in this environment.",
    exc_type=ImportError,
)

import mlx.core as mx

from mlx_train.ops.loss import dpo_loss, sequence_logprobs


def test_sequence_logprobs_uniform_logits() -> None:
    vocab = 4
    hidden = mx.zeros((2, 3, 5), dtype=mx.float32)
    lm_head_weight = mx.zeros((vocab, 5), dtype=mx.float32)
    labels = mx.array([[0, 1, 2], [3, 0, 1]], dtype=mx.int32)
    mask = mx.ones((2, 3), dtype=mx.float32)
    logp = sequence_logprobs(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        labels=labels,
        loss_mask=mask,
        chunk_size=0,
    )
    expected = -math.log(vocab)
    diff = mx.max(mx.abs(logp - expected)).item()
    assert diff < 1e-5


def test_sequence_logprobs_masked_tokens() -> None:
    vocab = 5
    hidden = mx.zeros((2, 2, 6), dtype=mx.float32)
    lm_head_weight = mx.zeros((vocab, 6), dtype=mx.float32)
    labels = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
    mask = mx.array([[1, 0], [0, 1]], dtype=mx.float32)
    logp = sequence_logprobs(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        labels=labels,
        loss_mask=mask,
        chunk_size=0,
    )
    expected = -math.log(vocab)
    diff = mx.max(mx.abs(logp - expected)).item()
    assert diff < 1e-5


def test_sequence_logprobs_chunked_matches_full() -> None:
    hidden = mx.arange(2 * 4 * 3).reshape(2, 4, 3).astype(mx.float32)
    lm_head_weight = mx.arange(5 * 3).reshape(5, 3).astype(mx.float32)
    labels = mx.array([[0, 1, 2, 3], [4, 3, 2, 1]], dtype=mx.int32)
    mask = mx.ones((2, 4), dtype=mx.float32)
    full = sequence_logprobs(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        labels=labels,
        loss_mask=mask,
        chunk_size=0,
    )
    chunked = sequence_logprobs(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        labels=labels,
        loss_mask=mask,
        chunk_size=2,
    )
    diff = mx.max(mx.abs(full - chunked)).item()
    assert diff < 1e-5


def test_dpo_loss_equal_policy_ref() -> None:
    policy = mx.array([0.2, 0.2], dtype=mx.float32)
    ref = mx.array([0.2, 0.2], dtype=mx.float32)
    loss = dpo_loss(policy_logp=policy, ref_logp=ref, beta=0.5)
    assert abs(float(loss.item()) - math.log(2.0)) < 1e-6
