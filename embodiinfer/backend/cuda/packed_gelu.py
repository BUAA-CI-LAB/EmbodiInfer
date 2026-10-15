"""Exact BF16 GELU/product over packed projection views without input copies."""

from __future__ import annotations

import ctypes
from typing import Literal

import torch

from ...layers.registry import OperatorCapabilities, OperatorRequest
from .activation import GeluMulFusion


class PackedGeluMul:
    """Retain Torch's BF16 GELU lookup and launch a BF16-only output kernel.

    No activation quantizer is constructed or called. Inputs keep their checkpoint
    dtype; the lookup and product each retain the original BF16 rounding point.
    """

    capabilities = OperatorCapabilities(
        ("cuda",),
        (torch.bfloat16,),
        "torch_gelu_product_bf16_rounding",
        rank=2,
        bits=(16,),
        minimum_sm=(8, 0),
    )

    def __init__(self, approximate: Literal["none", "tanh"] = "tanh") -> None:
        """Build the device-local reference lookup outside CUDA Graph capture."""
        request = OperatorRequest(
            torch.device("cuda", torch.cuda.current_device()), torch.bfloat16, bits=16, cuda_graph=True
        )
        self.capabilities.check(request)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare packed GELU backend before CUDA Graph capture")
        self.reference = GeluMulFusion(None, approximate=approximate)
        self.device = self.reference.device
        self.function = self.reference.library.cc_geglu_bf16_strided
        pointer = ctypes.c_void_p
        self.function.argtypes = [pointer] * 3 + [ctypes.c_int] * 2 + [ctypes.c_int64] + [pointer] * 2
        self.function.restype = ctypes.c_int

    def plan(self, gate: torch.Tensor, up: torch.Tensor) -> PackedGeluPlan:
        """Prepare one fixed layout before capture; output belongs to this plan."""
        self.capabilities.check(
            OperatorRequest(gate.device, gate.dtype, shape=tuple(gate.shape), bits=16, cuda_graph=True)
        )
        return PackedGeluPlan(self, gate, up)


class PackedGeluPlan:
    """Reuse an ephemeral BF16 output on the caller's guarded CUDA stream."""

    def __init__(self, backend: PackedGeluMul, gate: torch.Tensor, up: torch.Tensor) -> None:
        """Allocate output for one matched layout; the caller owns stream isolation."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare packed GELU plans before CUDA Graph capture")
        self.backend = backend
        self.shape, self.stride = gate.shape, gate.stride()
        if (
            gate.ndim != 2
            or up.ndim != 2
            or min(gate.shape) <= 0
            or gate.stride(1) != 1
            or gate.stride(0) < gate.shape[1]
            or gate.dtype != torch.bfloat16
            or gate.device != backend.device
            or up.shape != gate.shape
            or up.stride() != gate.stride()
            or up.dtype != gate.dtype
            or up.device != gate.device
        ):
            raise ValueError("Expected matching row-strided CUDA BF16 matrices")
        self.output = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)

    def run(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Read current inputs without copying and retain two BF16 rounding points."""
        for tensor in (gate, up):
            if (
                tensor.shape != self.shape
                or tensor.stride() != self.stride
                or tensor.dtype != self.output.dtype
                or tensor.device != self.output.device
            ):
                raise ValueError("Packed GELU input layout, dtype or device changed")
        with torch.cuda.device(gate.device):
            status = self.backend.function(
                gate.data_ptr(),
                up.data_ptr(),
                self.output.data_ptr(),
                *self.shape,
                self.stride[0],
                self.backend.reference.table.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            if status:
                raise RuntimeError(f"Packed BF16 GELU launch failed: {status}")
        return self.output
