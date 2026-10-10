"""BF16 GELU and product fused with calibrated activation encoding."""

import ctypes
from typing import Literal

import torch

from ...layers.quantization import ActivationQuantizer, QuantizationPlan
from ...layers.registry import OperatorCapabilities
from .native import build_library


class GeluMulFusion:
    """Compile and launch single-pass GELU × up × quantization kernels."""

    capabilities = OperatorCapabilities(
        ("cuda",),
        (torch.bfloat16,),
        "torch_gelu_product_bf16_rounding",
        rank=2,
        bits=(4, 8, 16),
        minimum_sm_by_bits=((4, (10, 0)),),
        minimum_sm=(8, 0),
    )

    def __init__(
        self, backend: ActivationQuantizer, *, approximate: Literal["none", "tanh"] = "none"
    ) -> None:
        """Load the cached library for the current CUDA device."""
        if approximate not in ("none", "tanh"):
            raise ValueError("GELU approximation must be none or tanh")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.backend, self.approximate = backend, approximate
        codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
        self.table = torch.nn.functional.gelu(codes.view(torch.bfloat16), approximate=approximate).view(
            torch.uint16
        )
        built = build_library(
            "fusion", strict=False, specific=True, native_fp4=torch.cuda.get_device_capability()[0] >= 10
        )
        self.library, self.compiler, self.command = built.library, built.compiler, built.command
        pointer = ctypes.c_void_p
        self.library.cc_geglu.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
            ctypes.c_int,
            pointer,
            pointer,
        ]
        self.library.cc_geglu_fp4.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            pointer,
        ]
        self.library.cc_geglu.restype = ctypes.c_int
        self.library.cc_geglu_fp4.restype = ctypes.c_int

    def plan(self, gate: torch.Tensor, bits: int, scale: float | None = None) -> "FusionPlan":
        """Allocate fixed-shape BF16 or calibrated low precision output buffers."""
        return FusionPlan(self, gate, bits, scale)

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise RuntimeError(f"CUDA GELU epilogue launch failed with error {status}")


class FusionPlan:
    """Reusable graph-compatible output workspace for an exact GELU gate."""

    def __init__(self, backend: GeluMulFusion, gate: torch.Tensor, bits: int, scale: float | None) -> None:
        """Allocate output and retain the existing quantizer's scale/layout contract."""
        if bits not in (4, 8, 16):
            raise ValueError("Expected bits in {4, 8, 16}")
        if gate.device != backend.device or gate.dtype != torch.bfloat16 or not gate.is_contiguous():
            raise ValueError("Fusion requires contiguous CUDA BF16 gate/up matrices")
        if gate.ndim != 2 or min(gate.shape) <= 0:
            raise ValueError("Fusion requires a matrix")
        self.backend, self.bits, self.shape = backend, bits, gate.shape
        self.quantized: QuantizationPlan | None = None
        if bits == 16:
            self.output = torch.empty_like(gate)
            self.scale = self.scales = self.blocked = None
        else:
            if scale is None:
                raise ValueError("A single-pass fused quantizer requires a calibrated global scale")
            self.quantized = backend.backend.plan(gate, bits, scale)
            self.output = self.quantized.output
            self.scale, self.scales, self.blocked = (
                self.quantized.scale,
                self.quantized.scales,
                self.quantized.blocked,
            )

    def encode(self, gate: torch.Tensor, up: torch.Tensor) -> "FusionPlan":
        """Round GELU and product to BF16 in registers, then store the requested format."""
        for tensor in (gate, up):
            if tensor.shape != self.shape or tensor.device != self.output.device:
                raise ValueError("Fusion input shape/device changed")
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError("Fusion requires contiguous BF16 inputs")
        with torch.cuda.device(gate.device):
            stream = torch.cuda.current_stream(gate.device).cuda_stream
            if self.bits == 4:
                status = self.backend.library.cc_geglu_fp4(
                    gate.data_ptr(),
                    up.data_ptr(),
                    self.output.data_ptr(),
                    self.scales.data_ptr(),
                    self.blocked.data_ptr(),
                    self.scale.data_ptr(),
                    *self.shape,
                    self.backend.table.data_ptr(),
                    stream,
                )
            else:
                status = self.backend.library.cc_geglu(
                    gate.data_ptr(),
                    up.data_ptr(),
                    self.output.data_ptr(),
                    self.scale.data_ptr() if self.scale is not None else None,
                    gate.numel(),
                    self.bits,
                    self.backend.table.data_ptr(),
                    stream,
                )
            self.backend._check(status)
        return self
