"""Reusable CUDA FP8 and NVFP4 encoding workspaces."""

import ctypes
import math

import torch

from ...layers.registry import OperatorCapabilities
from .native import build_library


class CudaQuantizer:
    """Quantize contiguous BF16 matrices into preallocated FP8 or NVFP4 buffers."""

    capabilities = OperatorCapabilities(
        ("cuda",),
        (torch.bfloat16,),
        "rounded_bf16_encoding",
        rank=2,
        width_multiple=64,
        bits=(4, 8),
        minimum_sm_by_bits=((4, (10, 0)),),
        minimum_sm=(8, 0),
    )

    def __init__(self) -> None:
        """Load the cached library for the current CUDA device."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        built = build_library(
            "quantize", strict=False, specific=True, native_fp4=torch.cuda.get_device_capability()[0] >= 10
        )
        self.library, self.compiler, self.command = built.library, built.compiler, built.command
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.library.cc_scale.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
            ctypes.c_float,
            pointer,
        ]
        self.library.cc_fp8.argtypes = [pointer, pointer, pointer, ctypes.c_int64, pointer]
        self.library.cc_fp4.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            integer,
            integer,
            pointer,
        ]
        self.library.cc_rescale.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
            pointer,
        ]
        for name in ("cc_scale", "cc_fp8", "cc_fp4", "cc_rescale"):
            getattr(self.library, name).restype = integer

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise RuntimeError(f"CUDA quantizer launch failed with error {status}")

    def plan(self, inputs: torch.Tensor, bits: int, scale: float | None = None) -> "QuantizationPlan":
        """Allocate reusable buffers; omit scale for tensorwise dynamic quantization."""
        return QuantizationPlan(self, inputs, bits, scale)

    def rescale(
        self, output: torch.Tensor, input_scale: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        """Apply NVFP4 global scales in place with the reference BF16 rounding order."""
        if output.device != self.device or any(
            scale.device != self.device for scale in (input_scale, weight_scale)
        ):
            raise ValueError("Quantizer output and scales must share the backend device")
        with torch.cuda.device(output.device):
            self._check(
                self.library.cc_rescale(
                    output.data_ptr(),
                    input_scale.data_ptr(),
                    weight_scale.data_ptr(),
                    output.numel(),
                    torch.cuda.current_stream(output.device).cuda_stream,
                )
            )
        return output


class QuantizationPlan:
    """Fixed-shape quantization workspace safe to reuse inside CUDA Graph capture."""

    def __init__(self, backend: CudaQuantizer, inputs: torch.Tensor, bits: int, scale: float | None) -> None:
        """Allocate buffers and record a dynamic or calibrated global scaling policy."""
        if inputs.device != backend.device or inputs.dtype != torch.bfloat16 or not inputs.is_contiguous():
            raise ValueError("CUDA quantization requires contiguous CUDA BF16 inputs")
        if inputs.ndim != 2 or min(inputs.shape) <= 0 or inputs.shape[1] % 64 or bits not in (4, 8):
            raise ValueError("Expected a matrix with K divisible by 64 and bits in {4, 8}")
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("A calibrated scale must be finite and positive")
        if bits == 4 and torch.cuda.get_device_capability(inputs.device)[0] < 10:
            raise ValueError("Native NVFP4 encoding requires a Blackwell CUDA device")
        self.backend, self.bits, self.shape = backend, bits, inputs.shape
        self.dynamic = scale is None
        self.scale = torch.tensor(scale or 1.0, device=inputs.device, dtype=torch.float32)
        self.partials = torch.empty(
            math.ceil(inputs.numel() / 4096), device=inputs.device, dtype=torch.float32
        )
        rows, columns = inputs.shape
        self.output = torch.empty(
            (rows, columns if bits == 8 else columns // 2),
            device=inputs.device,
            dtype=torch.float8_e4m3fn if bits == 8 else torch.uint8,
        )
        self.scales = torch.empty((rows, columns // 16), device=inputs.device, dtype=torch.float8_e4m3fn)
        # Padding stays zero when a compact encoder launches only valid rows.
        # Initialize once outside capture; valid scale slots are overwritten per call.
        self.blocked = torch.zeros(
            math.ceil(rows / 128) * 128 * (columns // 16),
            device=inputs.device,
            dtype=torch.float8_e4m3fn,
        )

    def quantize(self, inputs: torch.Tensor) -> "QuantizationPlan":
        """Encode an input matrix using this plan's preallocated output and scales."""
        if inputs.shape != self.shape or not inputs.is_contiguous():
            raise ValueError("Quantization input shape or layout changed")
        if inputs.device != self.output.device or inputs.dtype != torch.bfloat16:
            raise ValueError("Quantization input device or dtype changed")
        library = self.backend.library
        with torch.cuda.device(inputs.device):
            stream = torch.cuda.current_stream(inputs.device).cuda_stream
            if self.dynamic:
                self.backend._check(
                    library.cc_scale(
                        inputs.data_ptr(),
                        self.partials.data_ptr(),
                        self.scale.data_ptr(),
                        inputs.numel(),
                        448.0 if self.bits == 8 else 448.0 * 6.0,
                        stream,
                    )
                )
            if self.bits == 8:
                status = library.cc_fp8(
                    inputs.data_ptr(),
                    self.output.data_ptr(),
                    self.scale.data_ptr(),
                    inputs.numel(),
                    stream,
                )
            else:
                status = library.cc_fp4(
                    inputs.data_ptr(),
                    self.output.data_ptr(),
                    self.scales.data_ptr(),
                    self.blocked.data_ptr(),
                    self.scale.data_ptr(),
                    *self.shape,
                    stream,
                )
            self.backend._check(status)
        return self
