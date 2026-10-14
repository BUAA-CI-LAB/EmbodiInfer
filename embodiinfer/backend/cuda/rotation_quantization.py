"""Native channel scaling, H128 rotation and per-token signed INT8 encoding."""

from __future__ import annotations

import ctypes

import torch

from .native import build_library


class RotatedInt8Quantizer:
    """Device-bound encoding with the calibrated FP32/BF16 rounding order."""

    def __init__(self) -> None:
        """Compile before capture on the registry-selected CUDA device."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.library = build_library("rotation_quantization", specific=False).library
        self.library.ei_rotated_int8.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        self.library.ei_rotated_int8.restype = ctypes.c_int

    def __call__(
        self,
        inputs: torch.Tensor,
        divisor: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        """Encode a validated contiguous BF16 matrix on its current Torch stream."""
        if inputs.ndim < 2 or inputs.shape[-1] not in (2048, 16384):
            raise ValueError("Rotated INT8 encoding requires K=2048 or 16384")
        rows, width = inputs.numel() // inputs.shape[-1], inputs.shape[-1]
        if (
            rows <= 0
            or rows * width >= 2**31
            or any(
                x.device != self.device
                or x.dtype != dtype
                or tuple(x.shape) != shape
                or not x.is_contiguous()
                for x, dtype, shape in (
                    (inputs, torch.bfloat16, tuple(inputs.shape)),
                    (divisor, torch.float32, (width,)),
                    (codes, torch.int8, (rows, width)),
                    (scales, torch.float32, (rows,)),
                )
            )
        ):
            raise ValueError("Rotated INT8 encoding buffers have incompatible shapes, devices or dtypes")
        with torch.cuda.device(self.device):
            status = self.library.ei_rotated_int8(
                inputs.data_ptr(),
                divisor.data_ptr(),
                codes.data_ptr(),
                scales.data_ptr(),
                inputs.numel() // inputs.shape[-1],
                inputs.shape[-1],
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        if status:
            raise RuntimeError(f"Rotated INT8 encoding failed: {status}")
