"""MockFlowVLA architecture config + scale presets.

``MockConfig`` extends the engine contract (:class:`VLAPolicyConfig`) with the
synthetic model's architecture fields. Presets let the benchmark sweep scale.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

from ..config import VLAPolicyConfig


@dataclass
class MockConfig(VLAPolicyConfig):
    """Shape/architecture description of the synthetic flow VLA."""

    name: str = "mock_flow_vla"
    # multimodal prefix
    num_cameras: int = 2
    image_size: int = 224
    patch_size: int = 16
    max_lang_len: int = 32
    vocab_size: int = 4096
    state_dim: int = 8
    # transformer sizes
    hidden_dim: int = 512
    num_heads: int = 8
    num_backbone_layers: int = 6
    num_expert_layers: int = 4
    mlp_ratio: float = 4.0

    @property
    def num_patches(self) -> int:
        p = self.image_size // self.patch_size
        return p * p

    @property
    def prefix_len(self) -> int:
        """Static length of the multimodal prefix token sequence."""
        return self.num_cameras * self.num_patches + self.max_lang_len + 1


# Convenience presets so the benchmark can sweep model scale.
PRESETS = {
    "tiny": dict(
        hidden_dim=256,
        num_heads=4,
        num_backbone_layers=4,
        num_expert_layers=2,
        num_cameras=1,
        action_horizon=16,
    ),
    "small": dict(
        hidden_dim=512,
        num_heads=8,
        num_backbone_layers=6,
        num_expert_layers=4,
        num_cameras=2,
        action_horizon=50,
    ),
    "base": dict(
        hidden_dim=1024,
        num_heads=16,
        num_backbone_layers=12,
        num_expert_layers=6,
        num_cameras=2,
        action_horizon=50,
    ),
}


def preset_config(name: str = "small", **overrides: Any) -> MockConfig:
    """Build a :class:`MockConfig` from a named scale preset, with field overrides.

    Args:
        name: one of :data:`PRESETS` (``"tiny"`` / ``"small"`` / ``"base"``).
        **overrides: individual ``MockConfig`` fields to override the preset.
    """
    if name not in PRESETS:
        suggestion = difflib.get_close_matches(name, PRESETS, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise ValueError(f"unknown preset '{name}'.{hint} Available: {sorted(PRESETS)}")
    kwargs = dict(PRESETS[name])
    kwargs.update(overrides)
    return MockConfig(name=f"mock_flow_vla_{name}", **kwargs)
