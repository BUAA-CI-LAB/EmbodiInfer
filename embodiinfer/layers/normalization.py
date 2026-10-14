"""RMSNorm and residual/norm/encoding contracts and lazy backend selection."""

from __future__ import annotations

from typing import Protocol

import torch

from .quantization import EncodedActivation
from .registry import BackendRegistry, OperatorCapabilities


class NormalizationPlan(Protocol):
    """Last-axis RMSNorm with FP32 statistics/affine and explicit output rounding.

    Nonadaptive scale is [D] with a +1 offset. Adaptive modulation is raw FP32
    [B,3D] scale/shift/gate. Returned CUDA workspace tensors are ephemeral.
    """

    def normalize(
        self,
        inputs: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize and affine, rounding output and any adaptive gate to input dtype."""
        ...

    def residual_normalize(
        self,
        inputs: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor | None,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Round gated product and residual sum separately before normalization."""
        ...


class NormalizationBackend(Protocol):
    """Prepare fixed-layout RMSNorm execution independently of model modules."""

    capabilities: OperatorCapabilities

    def plan(self, inputs: torch.Tensor, *, adaptive: bool, eps: float = 1e-6) -> NormalizationPlan:
        """Allocate a scoped plan before capture and validate supported epsilon."""
        ...


class NormQuantPlan(EncodedActivation, Protocol):
    """Fused residual/RMSNorm/encoding with the quantizer's standard packed layout."""

    def normalize(
        self,
        inputs: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> EncodedActivation:
        """Round normalized values to BF16 before encoding them."""
        ...

    def residual_normalize(
        self,
        inputs: torch.Tensor,
        update: torch.Tensor,
        residual_gate: torch.Tensor | None,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, EncodedActivation, torch.Tensor | None]:
        """Preserve residual rounding and return the next norm's adaptive gate."""
        ...


class NormQuantBackend(Protocol):
    """Prepare composite norm/encoding plans; reject incompatible dependencies."""

    capabilities: OperatorCapabilities

    def plan(
        self, inputs: torch.Tensor, *, bits: int, calibrated_scale: float, adaptive: bool
    ) -> NormQuantPlan:
        """Allocate workspaces with strict FP32 mean and BF16 rounding boundaries."""
        ...


normalization_backends = BackendRegistry[NormalizationBackend]("normalization")
normalization_backends.register_lazy("cuda_strict", "embodiinfer.backend.cuda.normalization", "NormFusion")
normalization_backends.register_lazy("torch", "embodiinfer.backend.torch.normalization", "TorchRMSNorm")

norm_quant_backends = BackendRegistry[NormQuantBackend]("residual/norm/quantization")
norm_quant_backends.register_lazy("cuda_strict", "embodiinfer.backend.cuda.norm_quant", "NormQuantFusion")
