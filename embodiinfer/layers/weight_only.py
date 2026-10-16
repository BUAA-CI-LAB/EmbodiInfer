"""Offline weight-only projection contracts and lazy backend routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from .quantization import WorkspaceKey
from .registry import BackendRegistry, OperatorCapabilities


class WeightOnlyPlan(Protocol):
    """Immutable packed weights with private locks per stream or captured execution."""

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project BF16 activations, optionally applying paired exact GELU/product."""
        ...

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release completed execution storage after its graph has been destroyed."""
        ...


class WeightOnlyBackend(Protocol):
    """Pack calibrated natural-column integer codes without requantizing weights."""

    capabilities: OperatorCapabilities

    def plan(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        group_size: int,
        workspace_key: Callable[[torch.device], WorkspaceKey],
        fused_gelu: bool = False,
        large_m: bool = False,
        blocks: int | None = None,
    ) -> WeightOnlyPlan:
        """Prepare symmetric uint4 (signed value +8) and BF16 group scales."""
        ...


weight_only_backends = BackendRegistry[WeightOnlyBackend]("weight-only projection")
weight_only_backends.register_lazy("marlin", "embodiinfer.backend.cuda.marlin", "MarlinBackend")
