"""Strict FP32 rotation with BF16 storage and precomputed Torch factors."""

import ctypes

import torch

from ...layers.registry import OperatorCapabilities
from .native import build_library


class RotaryKernel:
    """Rotate tensors on the current stream without changing any model globals."""

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "fp32_rope_bf16_output", rank=4, minimum_sm=(8, 0)
    )

    def __init__(self) -> None:
        """Load the architecture-specific rotation kernel outside graph capture."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.library = build_library("rotary", strict=True).library
        pointer = ctypes.c_void_p
        self.library.cc_rotary.argtypes = [pointer] * 4 + [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
        ]
        self.library.cc_rotary.restype = ctypes.c_int

    def __call__(self, inputs: torch.Tensor, sine: torch.Tensor, cosine: torch.Tensor) -> torch.Tensor:
        """Apply reference rotation using FP32 sine/cosine and a distinct BF16 output."""
        if inputs.ndim != 4 or inputs.shape[-1] % 2:
            raise ValueError("RoPE requires [batch, length, heads, even width]")
        if inputs.device != self.device or inputs.dtype != torch.bfloat16:
            raise ValueError("RoPE kernel requires CUDA BF16 inputs")
        values = inputs.contiguous()
        sine, cosine = sine.contiguous(), cosine.contiguous()
        half = values.shape[-1] // 2
        expected = (values.shape[0], values.shape[1], 1, half)
        if any(
            factor.shape != expected or factor.dtype != torch.float32 or factor.device != values.device
            for factor in (sine, cosine)
        ):
            raise ValueError("RoPE requires matching FP32 sine/cosine on the input device")
        output = torch.empty_like(values)
        with torch.cuda.device(values.device):
            status = self.library.cc_rotary(
                values.data_ptr(),
                sine.data_ptr(),
                cosine.data_ptr(),
                output.data_ptr(),
                values.numel() // 2,
                values.shape[2],
                half,
                torch.cuda.current_stream(values.device).cuda_stream,
            )
            if status:
                raise RuntimeError(f"RoPE kernel launch failed: {status}")
        return output
