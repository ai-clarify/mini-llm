"""Loss helpers for MiniLLM training (CE + optional MTP)."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


def _masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).reshape(labels.size())
    if loss_mask is None:
        return loss.mean()
    mask = loss_mask.to(dtype=loss.dtype)
    denom = mask.sum().clamp(min=1)
    return (loss * mask).sum() / denom


def compute_mtp_loss(
    mtp_logits: Optional[Iterable[torch.Tensor]],
    labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    *,
    weight: float,
) -> torch.Tensor:
    """Compute the averaged MTP cross-entropy loss.

    Complexity: O(L * B * T * V) time, O(1) extra space beyond inputs, where
    L is the number of MTP layers, B the batch size, T the sequence length,
    and V the vocabulary size.
    """
    if not mtp_logits or weight <= 0.0:
        return torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    total = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    count = 0
    for idx, logits in enumerate(mtp_logits):
        offset = idx + 2
        if logits.size(1) <= offset:
            continue
        shift_logits = logits[:, :-offset, :]
        shift_labels = labels[:, offset:]
        shift_mask = loss_mask[:, offset:] if loss_mask is not None else None
        total = total + _masked_ce_loss(shift_logits, shift_labels, shift_mask)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    return total / float(count) * float(weight)
