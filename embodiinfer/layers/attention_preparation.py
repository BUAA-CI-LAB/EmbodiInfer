"""RoPE, strict attention and cache preparation contracts."""

from __future__ import annotations

from typing import Protocol

import torch

from .registry import BackendRegistry, OperatorCapabilities


class RotaryAttentionPlan(Protocol):
    """Fixed BTNH attention layout with a read-only cached prefix per request."""

    def load_prefix(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Refresh BTKH cached keys/values once before decoding, including capture."""
        ...

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sine: torch.Tensor,
        cosine: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return BTNH context and BTKH rotated keys/values in ephemeral storage."""
        ...


class RotaryAttentionBackend(Protocol):
    """Prepare attention while retaining FP32 scores and BF16 rounding boundaries."""

    capabilities: OperatorCapabilities

    def plan(
        self, query_shape: tuple[int, int, int, int], *, kv_heads: int, prefix_length: int = 0
    ) -> RotaryAttentionPlan:
        """Allocate before capture; a cached prefix must be loaded before use."""
        ...


rotary_attention_backends = BackendRegistry[RotaryAttentionBackend]("rotary attention")
rotary_attention_backends.register_lazy(
    "cuda_strict", "embodiinfer.backend.cuda.attention", "StrictRotaryAttention"
)
