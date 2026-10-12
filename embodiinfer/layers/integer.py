"""Checkpoint-calibrated integer projections with prepared activation transforms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from .quantization import WorkspaceKey
from .registry import BackendRegistry, OperatorCapabilities


class IntegerProjectionPlan(Protocol):
    """Immutable integer weights and private activation storage per execution."""

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform and encode BF16 inputs, then project with integer accumulation."""
        ...

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release storage after its captured execution has completed."""
        ...


class IntegerProjectionBackend(Protocol):
    """Execute frozen signed INT8 codes without online weight quantization."""

    capabilities: OperatorCapabilities

    def plan(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        divisor: torch.Tensor,
        *,
        workspace_key: Callable[[torch.device], WorkspaceKey],
        fused_gelu: bool = False,
    ) -> IntegerProjectionPlan:
        """Bind per-output scales and an offline channel/Hadamard transform."""
        ...


integer_projection_backends = BackendRegistry[IntegerProjectionBackend]("integer projection")
integer_projection_backends.register_lazy(
    "triton_rotated", "embodiinfer.backend.triton.integer", "RotatedInt8Backend"
)
