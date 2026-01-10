from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class LoRAConfig:
    r: int
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.r) if self.r > 0 else 0.0


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        r: int,
        alpha: float,
        dropout: float,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA r must be > 0, got {r}")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features/out_features must be > 0")

        self.base = base
        self.base.freeze(recurse=True)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else None

        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)
        self.merged = False

        # Common LoRA init: A ~ N(0, 0.02), B = 0
        self.lora_B.weight = mx.zeros_like(self.lora_B.weight)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        r: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        bias = linear.bias
        base = nn.Linear(
            int(linear.weight.shape[1]),
            int(linear.weight.shape[0]),
            bias=bias is not None,
        )
        base.weight = linear.weight
        if bias is not None:
            base.bias = bias
        return cls(
            base,
            r=int(r),
            alpha=float(alpha),
            dropout=float(dropout),
            in_features=int(base.weight.shape[1]),
            out_features=int(base.weight.shape[0]),
        )

    def merge(self) -> None:
        if self.merged:
            return
        if not isinstance(self.base, nn.Linear):
            raise TypeError("LoRA merge is only supported when base is nn.Linear.")
        delta = (self.lora_B.weight @ self.lora_A.weight) * mx.array(self.scaling, dtype=mx.float32)
        self.base.weight = self.base.weight + delta.astype(self.base.weight.dtype)
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        if not isinstance(self.base, nn.Linear):
            raise TypeError("LoRA unmerge is only supported when base is nn.Linear.")
        delta = (self.lora_B.weight @ self.lora_A.weight) * mx.array(self.scaling, dtype=mx.float32)
        self.base.weight = self.base.weight - delta.astype(self.base.weight.dtype)
        self.merged = False

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        if self.merged:
            return out
        if self.dropout is not None:
            x = self.dropout(x)
        return out + self.lora_B(self.lora_A(x)) * self.scaling


def iter_lora_linear_modules(model: nn.Module) -> Iterable[LoRALinear]:
    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            yield mod


def merge_lora(model: nn.Module) -> int:
    n = 0
    for mod in iter_lora_linear_modules(model):
        mod.merge()
        n += 1
    return n


def unmerge_lora(model: nn.Module) -> int:
    n = 0
    for mod in iter_lora_linear_modules(model):
        mod.unmerge()
        n += 1
    return n


def _get_parent_by_path(root: nn.Module, path: List[str]) -> Tuple[object, str]:
    cur: object = root
    for part in path[:-1]:
        if part.isdigit():
            cur = cur[int(part)]  # type: ignore[index]
        else:
            cur = cur.__getattribute__(part)
    return cur, path[-1]


def _set_child(parent: object, name: str, value: object) -> None:
    if name.isdigit():
        parent[int(name)] = value  # type: ignore[index]
    else:
        setattr(parent, name, value)


def apply_lora(
    model: nn.Module,
    *,
    cfg: LoRAConfig,
    verbose: bool = True,
) -> int:
    """
    Replace target `nn.Linear` modules with `LoRALinear` modules.

    Matching is done on the last component of the module name (e.g. `q_proj`).

    Returns:
        Number of modules replaced.
    """
    if cfg.r <= 0:
        return 0

    targets: Set[str] = {t.split(".")[-1] for t in cfg.target_modules if t}
    to_replace: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1] if name else ""
        if leaf in targets:
            to_replace.append((name, mod))

    for name, mod in to_replace:
        path = name.split(".")
        parent, leaf = _get_parent_by_path(model, path)
        repl = LoRALinear.from_linear(mod, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
        _set_child(parent, leaf, repl)

    if verbose and to_replace:
        uniq = sorted({n.split(".")[-1] for n, _ in to_replace})
        lora_params = sum(
            int(mod.lora_A.weight.size + mod.lora_B.weight.size) for mod in iter_lora_linear_modules(model)
        )
        print(
            f"[lora] enabled r={cfg.r} alpha={cfg.alpha} dropout={cfg.dropout} targets={','.join(uniq)} "
            f"| trainable_params={lora_params / 1e6:.3f}M"
        )

    return len(to_replace)
