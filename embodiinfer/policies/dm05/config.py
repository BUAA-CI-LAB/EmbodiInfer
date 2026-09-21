"""DM0.5-owned action dimensions layered on the stable policy config."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import VLAPolicyConfig


@dataclass
class DM05PolicyConfig(VLAPolicyConfig):
    """Separate OpenDM's padded flow state from its public robot action."""

    internal_action_dim: int = 32
    output_action_dim: int = 7

    def __post_init__(self) -> None:
        if self.internal_action_dim <= 0 or self.output_action_dim <= 0:
            raise ValueError("DM0.5 action dimensions must be positive")
        if self.action_dim != self.output_action_dim:
            raise ValueError("action_dim must equal the public DM0.5 output dimension")


__all__ = ["DM05PolicyConfig"]
