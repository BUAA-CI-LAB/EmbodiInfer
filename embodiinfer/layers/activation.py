"""GELU/product/encoding and paired projection contracts and routing."""

from __future__ import annotations

from typing import Protocol

import torch

from .quantization import EncodedActivation
from .registry import BackendRegistry, OperatorCapabilities


class GeluMulPlan(EncodedActivation, Protocol):
    """Scoped GELU/product/encoding epilogue with explicit BF16 rounding."""

    def encode(self, gate: torch.Tensor, up: torch.Tensor) -> EncodedActivation:
        """Round GELU and its product to BF16 before storing BF16/FP8/NVFP4."""
        ...


class GeluMulBackend(Protocol):
    """Prepare an epilogue with an explicit exact/tanh GELU choice."""

    capabilities: OperatorCapabilities

    def plan(self, gate: torch.Tensor, bits: int, scale: float | None = None) -> GeluMulPlan:
        """Allocate a fixed-shape epilogue; low precision requires calibrated scale."""
        ...


class PairedGeluPlan(Protocol):
    """Prepared paired weights; the GELU/product output belongs to each invocation."""

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply paired projections and the selected rounded GELU/product."""
        ...


class PairedGeluBackend(Protocol):
    """Prepare paired projections independently of model weight attribute names."""

    capabilities: OperatorCapabilities

    def plan(self, gate_weight: torch.Tensor, up_weight: torch.Tensor) -> PairedGeluPlan:
        """Pack original [output,input] weights before warmup and graph capture."""
        ...


gelu_mul_backends = BackendRegistry[GeluMulBackend]("GELU/product/encoding")
gelu_mul_backends.register_lazy("cuda", "embodiinfer.backend.cuda.activation", "GeluMulFusion")
gelu_mul_backends.register_lazy("torch", "embodiinfer.backend.torch.activation", "TorchGeluMul")
gelu_mul_backends.register_lazy("cuda_lookup", "embodiinfer.backend.cuda.gelu_lookup", "LookupGeluMul")

paired_gelu_backends = BackendRegistry[PairedGeluBackend]("paired GELU projection")
paired_gelu_backends.register_lazy("triton_lookup", "embodiinfer.backend.triton.geglu", "PairedGelu")
paired_gelu_backends.register_lazy("torch", "embodiinfer.backend.torch.activation", "TorchPairedGelu")
paired_gelu_backends.register_lazy("triton_exact", "embodiinfer.backend.triton.geglu", "ExactPairedGelu")
