"""
Training schedule for dynamic hyperparameters (modded-nanogpt style).

Supports:
- Batch size warmup
- Attention window warmup
- MTP weight decay
- Embed/LM-head untying
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainingStage:
    """Configuration for a training stage."""
    # Stage duration as fraction of total training (0-1)
    duration: float = 1.0

    # Batch size (tokens per step)
    batch_size: Optional[int] = None

    # Attention window sizes (short, long)
    window_sizes: Optional[Tuple[int, int]] = None

    # MTP loss weights for each head
    mtp_weights: Optional[Tuple[float, ...]] = None

    # Learning rate multiplier
    lr_mul: float = 1.0

    # Whether to untie embed and lm_head
    untie_embed: bool = False

    # Whether to use YaRN for position scaling
    use_yarn: bool = False


# Default 3-stage schedule (modded-nanogpt style)
DEFAULT_STAGES = [
    TrainingStage(
        duration=1/3,
        window_sizes=(128, 384),
        mtp_weights=(1.0, 0.5, 0.25),
        lr_mul=1.0,
        untie_embed=False,
    ),
    TrainingStage(
        duration=1/3,
        window_sizes=(384, 896),
        mtp_weights=(1.0, 0.5, 0.125),
        lr_mul=1.0,
        untie_embed=False,
        use_yarn=True,
    ),
    TrainingStage(
        duration=1/3,
        window_sizes=(640, 1408),
        mtp_weights=(1.0, 0.5, 0.0),
        lr_mul=1.0,
        untie_embed=True,  # Untie at 2/3 training
        use_yarn=True,
    ),
]

# Batch size warmup schedule (3 stages)
BATCH_SIZE_STAGES = [
    TrainingStage(duration=1/3, batch_size=8, lr_mul=1.0),
    TrainingStage(duration=1/3, batch_size=16, lr_mul=(16/8)**0.6),
    TrainingStage(duration=1/3, batch_size=24, lr_mul=(24/8)**0.5),
]


class TrainingSchedule:
    """
    Dynamic training schedule manager.

    Usage:
        schedule = TrainingSchedule(stages, total_steps=10000)
        for step in range(total_steps):
            stage = schedule.get_stage(step)
            # Apply stage.batch_size, stage.window_sizes, etc.
    """

    def __init__(
        self,
        stages: list[TrainingStage] = None,
        total_steps: int = 1000,
    ):
        self.stages = stages or DEFAULT_STAGES
        self.total_steps = total_steps

        # Compute step boundaries
        self._boundaries = []
        cumsum = 0.0
        for stage in self.stages:
            cumsum += stage.duration
            self._boundaries.append(int(cumsum * total_steps))

    def get_stage_index(self, step: int) -> int:
        """Get the current stage index (0-based)."""
        for i, boundary in enumerate(self._boundaries):
            if step < boundary:
                return i
        return len(self.stages) - 1

    def get_stage(self, step: int) -> TrainingStage:
        """Get the current training stage configuration."""
        return self.stages[self.get_stage_index(step)]

    def get_progress_in_stage(self, step: int) -> float:
        """Get progress within current stage (0-1)."""
        idx = self.get_stage_index(step)
        start = 0 if idx == 0 else self._boundaries[idx - 1]
        end = self._boundaries[idx]
        return (step - start) / max(end - start, 1)

    def should_transition(self, step: int) -> bool:
        """Check if this step is a stage transition."""
        return step in self._boundaries


def apply_projection_zero_init(model) -> None:
    """
    Initialize output projection layers to zero (muP-like).

    This delays the contribution of each layer, allowing gradients
    to flow more directly through residual connections initially.
    """
    import torch.nn as nn

    for name, module in model.named_modules():
        # Zero-init MLP output projections
        if "down_proj" in name or "c_proj" in name or "o_proj" in name:
            if hasattr(module, "weight"):
                nn.init.zeros_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init attention output projections
        if "wo" in name and isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)


def apply_back_out_scaling(model, back_out_layers: int = 4, scale: float = 0.5) -> None:
    """
    Apply "back out" scaling to first N layers.

    This reduces the contribution of early layers to the final prediction,
    letting them focus on building useful representations.
    """
    import torch

    layer_count = 0
    for name, module in model.named_modules():
        if "layers." in name and name.endswith(".self_attn"):
            layer_count += 1

    for name, param in model.named_parameters():
        if "layers." in name:
            # Extract layer number
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx < back_out_layers:
                            # Apply scaling to output projections
                            if "down_proj" in name or "o_proj" in name or "wo" in name:
                                with torch.no_grad():
                                    param.mul_(scale ** (back_out_layers - layer_idx))
                    except ValueError:
                        pass
                    break
