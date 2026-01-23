from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim


class AdamWFP32State(optim.AdamW):
    """
    AdamW that keeps optimizer state (m/v) in float32.

    This is more numerically stable but uses substantially more memory than the
    default MLX AdamW, which stores state in the parameter dtype.
    """

    def init_single(self, parameter: mx.array, state: dict):
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        # MLX's default AdamW keeps state in param dtype; force float32 state.
        if "m" in state and isinstance(state["m"], mx.array) and state["m"].dtype != mx.float32:
            state["m"] = state["m"].astype(mx.float32)
        if "v" in state and isinstance(state["v"], mx.array) and state["v"].dtype != mx.float32:
            state["v"] = state["v"].astype(mx.float32)
        return super().apply_single(gradient, parameter, state)


def _zeropower_ns(mat: mx.array, *, steps: int, eps: float) -> mx.array:
    """Newton-Schulz iteration to approximate the closest orthogonal matrix."""
    x = mat.astype(mx.float32)
    if x.ndim != 2:
        raise ValueError("Muon expects 2D matrices")
    m, n = int(x.shape[0]), int(x.shape[1])
    transposed = False
    if m < n:
        x = x.T
        m, n = n, m
        transposed = True
    norm = mx.sqrt(mx.sum(x * x)) + float(eps)
    x = x / norm
    a = 3.4445
    b = -4.7750
    c = 2.0315
    for _ in range(int(steps)):
        a_mat = x @ x.transpose()
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x


class Muon(optim.Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization for 2D weights.

    Falls back to AdamW-style update for non-matrix parameters.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        momentum: float = 0.9,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        adam_for_1d: bool = True,
        variant: str = "default",
        normalize_update: bool = False,
    ) -> None:
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        self.eps = float(eps)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        self.adam_for_1d = bool(adam_for_1d)
        self.variant = str(variant)
        self.normalize_update = bool(normalize_update)

    def init_single(self, parameter: mx.array, state: dict) -> None:
        if parameter.ndim >= 2:
            state["momentum"] = mx.zeros(parameter.shape, dtype=mx.float32)
        else:
            state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
            state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)
            state["t"] = mx.array(0, dtype=mx.int32)

    @staticmethod
    def _as_scalar(value: Any) -> Any:
        return value if isinstance(value, mx.array) else float(value)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        lr = self._as_scalar(self.learning_rate)
        wd = self._as_scalar(self.weight_decay)
        if parameter.ndim >= 2:
            grad = gradient.astype(mx.float32)
            if "momentum" not in state:
                state["momentum"] = mx.zeros(parameter.shape, dtype=mx.float32)
            mom = state["momentum"]
            mom = mom * float(self.momentum) + grad * (1.0 - float(self.momentum))
            state["momentum"] = mom
            shape = parameter.shape
            cols = int(math.prod(shape[1:])) if len(shape) > 1 else 1
            g2d = mom.reshape((int(shape[0]), cols))
            steps = int(self.ns_steps)
            if self.variant == "polar":
                steps = max(1, steps // 2)
            update = _zeropower_ns(g2d, steps=steps, eps=float(self.eps))
            if self.normalize_update or self.variant == "norm":
                denom = mx.sqrt(mx.sum(update * update)) + float(self.eps)
                update = update / denom
            update = update.reshape(shape).astype(parameter.dtype)
            scale = lr
            if shape[0] > 0 and shape[1] > 0:
                scale = lr * math.sqrt(max(1.0, float(shape[0]) / float(shape[1])))
            out = parameter * (1.0 - lr * wd)
            out = out - update * scale
            return out

        if not self.adam_for_1d:
            return parameter * (1.0 - lr * wd) - gradient * lr

        m = state.get("m")
        v = state.get("v")
        t = state.get("t")
        if m is None or v is None or t is None:
            m = mx.zeros(parameter.shape, dtype=mx.float32)
            v = mx.zeros(parameter.shape, dtype=mx.float32)
            t = mx.array(0, dtype=mx.int32)
        t = t + 1
        grad = gradient.astype(mx.float32)
        beta1 = float(self.adam_beta1)
        beta2 = float(self.adam_beta2)
        m = m * beta1 + grad * (1.0 - beta1)
        v = v * beta2 + (grad * grad) * (1.0 - beta2)
        t_f = t.astype(mx.float32)
        b1 = mx.array(beta1, dtype=mx.float32)
        b2 = mx.array(beta2, dtype=mx.float32)
        m_hat = m / (1.0 - mx.power(b1, t_f))
        v_hat = v / (1.0 - mx.power(b2, t_f))
        update = m_hat / (mx.sqrt(v_hat) + float(self.adam_eps))
        state["m"], state["v"], state["t"] = m, v, t
        out = parameter * (1.0 - lr * wd) - update.astype(parameter.dtype) * lr
        return out


def make_optimizer(
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    state_dtype: str = "float32",
    muon_momentum: float = 0.9,
    muon_ns_steps: int = 5,
    muon_eps: float = 1e-7,
    muon_adam_beta1: float = 0.9,
    muon_adam_beta2: float = 0.999,
    muon_adam_eps: float = 1e-8,
    muon_adam_for_1d: bool = True,
    muon_variant: str = "default",
    muon_normalize_update: bool = False,
) -> optim.Optimizer:
    """
    Create an MLX optimizer.

    Args:
        name: One of: adamw, adafactor, lion.
        state_dtype: For AdamW only: float32 (stable, memory-heavy) or param
          (stores state in parameter dtype; faster/cheaper).
    """
    name = str(name).lower().strip()
    state_dtype = str(state_dtype).lower().strip()

    if name == "adamw":
        if state_dtype == "float32":
            return AdamWFP32State(
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
            )
        if state_dtype in {"param", "params", "parameter"}:
            return optim.AdamW(
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
            )
        raise ValueError(f"Unknown AdamW state_dtype: {state_dtype} (use float32|param)")

    if name == "adafactor":
        # Make it behave more like a traditional optimizer with a fixed LR.
        return optim.Adafactor(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            relative_step=False,
            scale_parameter=False,
        )

    if name == "lion":
        return optim.Lion(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    if name == "muon":
        return Muon(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            momentum=float(muon_momentum),
            ns_steps=int(muon_ns_steps),
            eps=float(muon_eps),
            adam_beta1=float(muon_adam_beta1),
            adam_beta2=float(muon_adam_beta2),
            adam_eps=float(muon_adam_eps),
            adam_for_1d=bool(muon_adam_for_1d),
            variant=str(muon_variant),
            normalize_update=bool(muon_normalize_update),
        )

    raise ValueError(f"Unknown optimizer: {name} (use adamw|adafactor|lion)")
