"""
Muon optimizer for PyTorch - ported from modded-nanogpt.

Key features:
1. Newton-Schulz orthogonalization for 2D weight matrices
2. Nesterov momentum with orthogonalization applied after
3. Falls back to AdamW for 1D parameters (biases, norms)
4. Cautious weight decay (only when grad and param agree)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to compute orthogonal matrix.
    Approximately computes G @ (G.T @ G)^(-1/2) using 5 iterations.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized coefficients
    X = G.bfloat16()

    # Transpose if needed to ensure tall matrix
    transposed = False
    if X.size(-2) < X.size(-1):
        X = X.mT
        transposed = True

    # Normalize
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X.mT @ X
        B = a * A + b * (A @ A) + c * (A @ A @ A)
        X = X @ B

    if transposed:
        X = X.mT

    return X.to(G.dtype)


def polar_express(G: Tensor, steps: int = 10) -> Tensor:
    """
    Polar decomposition via Newton iteration.
    More stable than Newton-Schulz for some cases.
    """
    assert G.ndim >= 2
    X = G.bfloat16()

    transposed = False
    if X.size(-2) < X.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        X = 1.5 * X - 0.5 * X @ (X.mT @ X)

    if transposed:
        X = X.mT

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer: SGD with momentum + Newton-Schulz orthogonalization.

    For 2D weight matrices, applies orthogonal updates.
    For 1D parameters (bias, norms), falls back to AdamW.

    Args:
        params: Model parameters
        lr: Learning rate for Muon (2D params)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        adamw_lr: Learning rate for AdamW fallback (1D params)
        adamw_betas: AdamW beta parameters
        adamw_eps: AdamW epsilon
        weight_decay: Weight decay factor
        cautious: Use cautious weight decay (default: True)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            weight_decay=weight_decay,
            cautious=cautious,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            cautious = group["cautious"]

            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Check if this is a 2D weight matrix (use Muon) or 1D (use AdamW)
                use_muon = p.ndim >= 2 and p.size(-1) >= 128 and p.size(-2) >= 128

                if use_muon:
                    # Muon update for 2D matrices
                    lr = group["lr"]

                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                        state["step"] = 0

                    state["step"] += 1
                    buf = state["momentum_buffer"]

                    # Momentum update
                    buf.mul_(momentum).add_(grad)

                    # Nesterov momentum
                    if nesterov:
                        grad_for_update = grad + momentum * buf
                    else:
                        grad_for_update = buf

                    # Newton-Schulz orthogonalization
                    grad_orth = zeropower_via_newtonschulz5(grad_for_update, steps=ns_steps)

                    # Scale by sqrt of dimensions
                    scale = math.sqrt(max(p.size(-2), p.size(-1)))

                    # Cautious weight decay: only decay where grad and param agree
                    if weight_decay > 0:
                        if cautious:
                            mask = (grad * p.data).sign()
                            p.data.mul_(1 - lr * weight_decay * mask.clamp(min=0))
                        else:
                            p.data.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.data.add_(grad_orth, alpha=-lr * scale)

                else:
                    # AdamW fallback for 1D params
                    lr = group["adamw_lr"]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # AdamW update
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    step = state["step"]
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    # Weight decay
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_muon_param_groups(
    model: torch.nn.Module,
    lr: float = 0.02,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.01,
    adamw_weight_decay: float = 0.0,
) -> list:
    """
    Create parameter groups for Muon optimizer.

    - 2D weight matrices: Muon with weight_decay
    - 1D params (bias, norms): AdamW without weight decay
    - Embedding and head: AdamW with lower lr

    Returns list of param groups for Muon optimizer.
    """
    muon_params = []
    adamw_params = []
    embed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Embedding and language model head use AdamW with lower lr
        if "embed" in name.lower() or "lm_head" in name.lower():
            embed_params.append(param)
        # 2D matrices use Muon
        elif param.ndim >= 2 and param.size(-1) >= 128 and param.size(-2) >= 128:
            muon_params.append(param)
        # 1D params use AdamW
        else:
            adamw_params.append(param)

    return [
        {"params": muon_params, "lr": lr, "weight_decay": weight_decay},
        {"params": adamw_params, "lr": adamw_lr, "weight_decay": adamw_weight_decay},
        {"params": embed_params, "lr": adamw_lr * 0.5, "weight_decay": adamw_weight_decay},
    ]
