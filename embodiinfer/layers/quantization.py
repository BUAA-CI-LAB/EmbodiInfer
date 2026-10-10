"""Activation encoding and calibrated projection contracts and routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

import torch

from .registry import BackendRegistry, OperatorCapabilities

Precision = Literal["bf16", "fp8", "nvfp4"]
WorkspaceKey = tuple[str, int]


class EncodedActivation(Protocol):
    """Packed activation storage, ephemeral until its plan is invoked again.

    FP8 uses E4M3FN with a tensorwise FP32 scale. NVFP4 uses E2M1x2 bytes,
    E4M3 block16 scales in SWIZZLE_32_4_4 order and a tensorwise FP32 scale.
    BF16 epilogues return ordinary BF16 output and no scale metadata.
    """

    output: torch.Tensor
    scale: torch.Tensor | None
    scales: torch.Tensor | None
    blocked: torch.Tensor | None


class QuantizationPlan(EncodedActivation, Protocol):
    """Fixed-shape activation encoder whose buffers belong to one execution scope."""

    def quantize(self, inputs: torch.Tensor) -> EncodedActivation:
        """Encode new BF16 inputs, updating output and scale metadata in place."""
        ...


class ActivationQuantizer(Protocol):
    """Create encoders with the standard tensorwise/block16 packed layouts."""

    capabilities: OperatorCapabilities

    def plan(self, inputs: torch.Tensor, bits: int, scale: float | None = None) -> QuantizationPlan:
        """Prepare before capture; None scale requests dynamic weight/activation scaling."""
        ...


class ProjectionPlan(Protocol):
    """Prepared weight format and scoped activation encoders for one projection."""

    precision: Precision
    maximum: float

    def encode(self, inputs: torch.Tensor) -> EncodedActivation | None:
        """Encode inputs at the calibrated range; BF16 returns None."""
        ...

    def apply(self, inputs: torch.Tensor, encoded: EncodedActivation | None) -> torch.Tensor:
        """Apply prepared weights, preserving the original leading input dimensions."""
        ...

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release activation workspaces after their graph and CUDA work have completed."""
        ...


class ProjectionBackend(Protocol):
    """Prepare BF16 or calibrated low precision weights independently of a model."""

    capabilities: OperatorCapabilities

    def plan(
        self,
        weight: torch.Tensor,
        maximum: float,
        precision: Precision,
        quantizer: ActivationQuantizer | None,
        workspace_key: Callable[[torch.device], WorkspaceKey],
    ) -> ProjectionPlan:
        """Pack a bias-free [output,input] weight without replacing its Parameter."""
        ...


quantization_backends = BackendRegistry[ActivationQuantizer]("activation quantization")
quantization_backends.register_lazy("cuda", "embodiinfer.backend.cuda.quantization", "CudaQuantizer")
quantization_backends.register_lazy("torch", "embodiinfer.backend.torch.quantization", "TorchQuantizer")

projection_backends = BackendRegistry[ProjectionBackend]("calibrated projection")
projection_backends.register_lazy(
    "torch", "embodiinfer.backend.torch.projection", "CalibratedProjectionBackend"
)
projection_backends.register_lazy(
    "torch_matmul", "embodiinfer.backend.torch.projection", "CalibratedMatmulBackend"
)
