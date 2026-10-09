"""Torch reference gated GELU epilogues and paired projections."""

from typing import Literal

import torch
import torch.nn.functional as F

from ...layers.launch import GemmTile
from ...layers.quantization import ActivationQuantizer
from ...layers.registry import OperatorCapabilities


class TorchGeluMul:
    """Separate reference GELU/product/encoding with the fused epilogue contract."""

    capabilities = OperatorCapabilities(
        ("cpu", "cuda"), (torch.bfloat16,), "torch_gelu_product_bf16_rounding", rank=2, bits=(4, 8, 16)
    )

    def __init__(
        self, backend: ActivationQuantizer, *, approximate: Literal["none", "tanh"] = "none"
    ) -> None:
        """Retain the selected quantizer and explicit GELU semantics."""
        if approximate not in ("none", "tanh"):
            raise ValueError("GELU approximation must be none or tanh")
        self.backend, self.approximate = backend, approximate

    def plan(self, gate: torch.Tensor, bits: int, scale: float | None = None) -> "TorchGeluPlan":
        """Prepare stable BF16 or calibrated low precision output buffers."""
        return TorchGeluPlan(self, gate, bits, scale)


class TorchGeluPlan:
    """Reference plan whose encoded output is overwritten on every invocation."""

    def __init__(self, backend: TorchGeluMul, gate: torch.Tensor, bits: int, scale: float | None) -> None:
        """Prepare encoding before capture; unsupported formats fail explicitly."""
        if gate.ndim != 2 or min(gate.shape) <= 0 or gate.dtype != torch.bfloat16 or not gate.is_contiguous():
            raise ValueError("GELU reference epilogue requires a contiguous BF16 matrix")
        if bits not in (4, 8, 16) or (bits != 16 and scale is None):
            raise ValueError("GELU encoding requires bits 4/8/16 and calibrated low precision scales")
        self.backend, self.shape = backend, gate.shape
        self.quantized = None if bits == 16 else backend.backend.plan(gate, bits, scale)
        self.output = torch.empty_like(gate) if self.quantized is None else self.quantized.output
        self.scale = None if self.quantized is None else self.quantized.scale
        self.scales = None if self.quantized is None else self.quantized.scales
        self.blocked = None if self.quantized is None else self.quantized.blocked

    def encode(self, gate: torch.Tensor, up: torch.Tensor) -> "TorchGeluPlan":
        """Preserve separate BF16 GELU and multiplication rounding."""
        for value in (gate, up):
            if (
                value.shape != self.shape
                or value.dtype != torch.bfloat16
                or value.device != self.output.device
                or not value.is_contiguous()
            ):
                raise ValueError("GELU reference input shape, layout, dtype or device changed")
        hidden = F.gelu(gate, approximate=self.backend.approximate) * up
        if self.quantized is None:
            self.output.copy_(hidden)
        else:
            self.quantized.quantize(hidden)
        return self


class TorchPairedGelu:
    """Native paired projection reference retaining original row-major weights."""

    capabilities = OperatorCapabilities(("cpu", "cuda"), (torch.bfloat16,), "native_gemm_torch_gelu", rank=2)

    def __init__(
        self,
        *,
        approximate: Literal["none", "tanh"] = "tanh",
        tile: GemmTile | None = None,
        tail_m: int | None = None,
        profile: tuple[int, int] | None = None,
        large_m: bool = False,
    ) -> None:
        """Select GELU; launch hints are accepted without changing Torch arithmetic."""
        if approximate not in ("none", "tanh"):
            raise ValueError("GELU approximation must be none or tanh")
        self.approximate = approximate

    def plan(self, gate_weight: torch.Tensor, up_weight: torch.Tensor) -> "TorchPairedPlan":
        """Retain the supplied BF16 [output,input] projection layout."""
        return TorchPairedPlan(gate_weight, up_weight, self.approximate)


class TorchPairedPlan:
    """Immutable paired weights; each invocation returns an independent output."""

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, approximate: Literal["none", "tanh"]) -> None:
        """Validate shapes/placement before preparing a weight-only plan."""
        if (
            gate.ndim != 2
            or gate.shape != up.shape
            or gate.device != up.device
            or any(value.dtype != torch.bfloat16 for value in (gate, up))
        ):
            raise ValueError("Paired projections require matching BF16 [output,input] weights")
        self.gate, self.up, self.approximate = gate.detach(), up.detach(), approximate

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply native GEMMs followed by separate rounded GELU and product."""
        if (
            inputs.shape[-1] != self.gate.shape[1]
            or inputs.device != self.gate.device
            or inputs.dtype != torch.bfloat16
        ):
            raise ValueError("Paired projection input width, dtype or device changed")
        return F.gelu(F.linear(inputs, self.gate), approximate=self.approximate) * F.linear(inputs, self.up)
