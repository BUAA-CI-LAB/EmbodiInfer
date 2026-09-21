"""The engine-facing policy config contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLAPolicyConfig:
    """The contract fields the engine core depends on for every flow VLA policy.

    These are the *only* policy fields the engine reads (to size action buffers
    and drive the denoise schedule). Model-specific architecture — camera count,
    hidden sizes, layer counts — lives on each policy's own config subclass
    (e.g. :class:`~embodiinfer.policies.mock.configuration_mock.MockConfig`), never here.
    """

    name: str = "flow_vla"
    action_dim: int = 7
    action_horizon: int = 50
    default_num_steps: int = 10
    dtype: str = "float32"  # "float32" | "float16" | "bfloat16"
