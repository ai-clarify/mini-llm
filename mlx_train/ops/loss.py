from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def _pad_to_multiple(
    x: mx.array, *, multiple: int, axis: int, pad_value: float | int = 0
) -> mx.array:
    if multiple <= 0:
        return x
    size = int(x.shape[axis])
    pad = (-size) % int(multiple)
    if pad == 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[axis] = pad
    pad_arr = mx.full(tuple(pad_shape), pad_value, dtype=x.dtype)
    return mx.concatenate([x, pad_arr], axis=axis)


def chunked_ce_loss_sum_and_tokens(
    *,
    hidden: mx.array,
    lm_head_weight: mx.array,
    labels: mx.array,
    loss_mask: mx.array,
    chunk_size: int,
) -> Tuple[mx.array, mx.array]:
    """
    Compute masked cross entropy without materializing full [B, T, V] logits.

    Args:
        hidden: [B, T, H] hidden states (post-norm).
        lm_head_weight: [V, H] weight matrix (tied embedding).
        labels: [B, T] int token ids (next-token labels).
        loss_mask: [B, T] 0/1 mask (1 = include token in loss).
        chunk_size: sequence chunk size for computing logits; <=0 means no chunking.

    Returns:
        (loss_sum, token_count) as float32 arrays.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T, H], got shape={hidden.shape}")
    if labels.shape != hidden.shape[:2]:
        raise ValueError(f"labels shape {labels.shape} != hidden[:2] {hidden.shape[:2]}")
    if loss_mask.shape != hidden.shape[:2]:
        raise ValueError(f"loss_mask shape {loss_mask.shape} != hidden[:2] {hidden.shape[:2]}")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != hidden.shape[2]:
        raise ValueError(
            f"lm_head_weight must be [V, H] with H={hidden.shape[2]}, got {lm_head_weight.shape}"
        )

    bsz, seq_len, _ = hidden.shape
    vocab = int(lm_head_weight.shape[0])

    mask = loss_mask.astype(mx.float32)
    tokens = mx.sum(mask)

    if chunk_size <= 0 or int(chunk_size) >= int(seq_len):
        logits = hidden @ lm_head_weight.transpose()  # [B, T, V]
        loss = nn.losses.cross_entropy(
            logits.reshape(bsz * seq_len, vocab),
            labels.reshape(bsz * seq_len),
            reduction="none",
        ).reshape(bsz, seq_len)
        return mx.sum(loss * mask), tokens

    chunk = int(chunk_size)
    hidden_p = _pad_to_multiple(hidden, multiple=chunk, axis=1, pad_value=0)
    labels_p = _pad_to_multiple(labels, multiple=chunk, axis=1, pad_value=0)
    mask_p = _pad_to_multiple(mask, multiple=chunk, axis=1, pad_value=0.0)

    _, padded_len, _ = hidden_p.shape
    loss_sum = mx.array(0.0, dtype=mx.float32)
    for start in range(0, int(padded_len), chunk):
        h = hidden_p[:, start : start + chunk, :]
        y = labels_p[:, start : start + chunk]
        m = mask_p[:, start : start + chunk]

        logits = h @ lm_head_weight.transpose()  # [B, chunk, V]
        loss = nn.losses.cross_entropy(
            logits.reshape(bsz * chunk, vocab),
            y.reshape(bsz * chunk),
            reduction="none",
        ).reshape(bsz, chunk)
        loss_sum = loss_sum + mx.sum(loss * m)

    return loss_sum, tokens


def chunked_ce_loss(
    *,
    hidden: mx.array,
    lm_head_weight: mx.array,
    labels: mx.array,
    loss_mask: mx.array,
    chunk_size: int,
) -> mx.array:
    loss_sum, tokens = chunked_ce_loss_sum_and_tokens(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        labels=labels,
        loss_mask=loss_mask,
        chunk_size=chunk_size,
    )
    denom = mx.maximum(tokens, mx.array(1.0, dtype=mx.float32))
    return loss_sum / denom


def sparse_ce_loss_sum_and_tokens(
    *,
    hidden: mx.array,
    lm_head_weight: mx.array,
    labels: mx.array,
    label_positions: mx.array,
    label_pos_mask: mx.array,
    chunk_size: int,
) -> Tuple[mx.array, mx.array]:
    """
    Compute masked cross entropy on selected token positions only.

    This is most useful for SFT-style training where `loss_mask` is sparse
    (e.g. only assistant tokens contribute to loss).

    Args:
        hidden: [B, T, H] hidden states (post-norm).
        lm_head_weight: [V, H] weight matrix (tied embedding).
        labels: [B, T] int token ids (next-token labels).
        label_positions: [B, L] int positions in [0, T-1] to include in loss (padded allowed).
        label_pos_mask: [B, L] 0/1 mask for `label_positions` (1 = valid position, 0 = padding).
        chunk_size: chunk size along L for computing logits; <=0 means no chunking.

    Returns:
        (loss_sum, token_count) as float32 arrays.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T, H], got shape={hidden.shape}")
    if labels.shape != hidden.shape[:2]:
        raise ValueError(f"labels shape {labels.shape} != hidden[:2] {hidden.shape[:2]}")
    if label_positions.ndim != 2:
        raise ValueError(f"label_positions must be [B, L], got shape={label_positions.shape}")
    if label_pos_mask.shape != label_positions.shape:
        raise ValueError(
            f"label_pos_mask shape {label_pos_mask.shape} != label_positions {label_positions.shape}"
        )
    if label_positions.shape[0] != hidden.shape[0]:
        raise ValueError(
            f"label_positions batch {label_positions.shape[0]} != hidden batch {hidden.shape[0]}"
        )

    pos = label_positions.astype(mx.int32)
    h_sel = mx.take_along_axis(hidden, pos[..., None], axis=1)  # [B, L, H]
    y_sel = mx.take_along_axis(labels, pos, axis=1)  # [B, L]
    m_sel = label_pos_mask.astype(mx.float32)  # [B, L]

    return chunked_ce_loss_sum_and_tokens(
        hidden=h_sel,
        lm_head_weight=lm_head_weight,
        labels=y_sel,
        loss_mask=m_sel,
        chunk_size=int(chunk_size),
    )


def sparse_ce_loss(
    *,
    hidden: mx.array,
    lm_head_weight: mx.array,
    labels: mx.array,
    label_positions: mx.array,
    label_pos_mask: mx.array,
    chunk_size: int,
) -> mx.array:
    loss_sum, tokens = sparse_ce_loss_sum_and_tokens(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        labels=labels,
        label_positions=label_positions,
        label_pos_mask=label_pos_mask,
        chunk_size=chunk_size,
    )
    denom = mx.maximum(tokens, mx.array(1.0, dtype=mx.float32))
    return loss_sum / denom


def sequence_logprobs(
    *,
    hidden: mx.array,
    lm_head_weight: mx.array,
    labels: mx.array,
    loss_mask: mx.array,
    chunk_size: int,
) -> mx.array:
    """
    Compute per-sequence average log-probabilities.

    Complexity: O(B * T * V) time without chunking (or O(B * T * V) in chunks),
    O(B * T) space for temporary logits (or O(B * chunk * V) per chunk),
    where B=batch, T=seq_len, V=vocab size.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T, H], got shape={hidden.shape}")
    if labels.shape != hidden.shape[:2]:
        raise ValueError(f"labels shape {labels.shape} != hidden[:2] {hidden.shape[:2]}")
    if loss_mask.shape != hidden.shape[:2]:
        raise ValueError(f"loss_mask shape {loss_mask.shape} != hidden[:2] {hidden.shape[:2]}")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != hidden.shape[2]:
        raise ValueError(
            f"lm_head_weight must be [V, H] with H={hidden.shape[2]}, got {lm_head_weight.shape}"
        )

    bsz, seq_len, _ = hidden.shape
    mask = loss_mask.astype(mx.float32)
    seq_lengths = mx.maximum(mx.sum(mask, axis=1), mx.array(1.0, dtype=mx.float32))

    if chunk_size <= 0 or int(chunk_size) >= int(seq_len):
        logits = hidden @ lm_head_weight.transpose()
        log_probs = nn.log_softmax(logits, axis=-1)
        token_logp = mx.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        seq_logp = mx.sum(token_logp * mask, axis=1)
        return seq_logp / seq_lengths

    chunk = int(chunk_size)
    logp_sum = mx.zeros((int(bsz),), dtype=mx.float32)
    for start in range(0, int(seq_len), chunk):
        h = hidden[:, start : start + chunk, :]
        y = labels[:, start : start + chunk]
        m = mask[:, start : start + chunk]
        logits = h @ lm_head_weight.transpose()
        log_probs = nn.log_softmax(logits, axis=-1)
        token_logp = mx.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        logp_sum = logp_sum + mx.sum(token_logp * m, axis=1)
    return logp_sum / seq_lengths


def dpo_loss(
    policy_logp: mx.array,
    ref_logp: mx.array,
    *,
    beta: float,
) -> mx.array:
    """
    Compute DPO loss from per-sequence log-probabilities.

    Complexity: O(B) time, O(B) space for batch size B.
    """
    if policy_logp.shape != ref_logp.shape:
        raise ValueError(
            f"policy_logp shape {policy_logp.shape} != ref_logp shape {ref_logp.shape}"
        )
    if int(policy_logp.shape[0]) % 2 != 0:
        raise ValueError("DPO expects an even batch size (chosen/rejected pairs).")

    half = int(policy_logp.shape[0] // 2)
    pi_logratios = policy_logp[:half] - policy_logp[half:]
    ref_logratios = ref_logp[:half] - ref_logp[half:]
    logits = pi_logratios - ref_logratios
    return mx.mean(-nn.log_sigmoid(float(beta) * logits))
