"""MLX-only unit tests for MoE and indexer helpers."""
from __future__ import annotations

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

from mlx_train.config import MiniLLMConfig
from mlx_train.models.minillm import MiniLLMIndexer, MoEFeedForward


def test_moe_single_expert_matches_dense() -> None:
    cfg = MiniLLMConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        use_moe=True,
        n_routed_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        dropout=0.0,
    ).finalize()

    moe = MoEFeedForward(cfg)
    x = mx.random.normal(shape=(2, 3, cfg.hidden_size))
    out = moe(x)
    dense = moe.experts[0](x, trace=None, layer_id=None, start_pos=0)
    diff = mx.max(mx.abs(out - dense)).item()
    assert diff < 1e-5


def test_indexer_topk_bounds() -> None:
    cfg = MiniLLMConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=2,
    ).finalize()

    indexer = MiniLLMIndexer(cfg)
    hidden = mx.random.normal(shape=(1, 4, cfg.hidden_size))
    topk_idx = indexer(hidden, attention_mask=None, past_len=0)
    assert topk_idx is not None
    assert topk_idx.shape == (1, 4, 2)
    assert int(mx.max(topk_idx).item()) < hidden.shape[1]
    assert int(mx.min(topk_idx).item()) >= 0


def test_indexer_disabled_when_topk_ge_seq() -> None:
    cfg = MiniLLMConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=4,
    ).finalize()

    indexer = MiniLLMIndexer(cfg)
    hidden = mx.random.normal(shape=(1, 4, cfg.hidden_size))
    assert indexer(hidden, attention_mask=None, past_len=0) is None
    assert indexer(hidden, attention_mask=None, past_len=1) is None
