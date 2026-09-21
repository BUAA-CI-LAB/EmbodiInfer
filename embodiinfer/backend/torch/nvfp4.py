"""Torch NVFP4 weight conversion, capability probes, and linear kernels."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def blocked_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert linear 1x16 scales to the cuBLASLt 32x4x4 layout."""

    rows, columns = scales.shape
    row_blocks = math.ceil(rows / 128)
    column_blocks = math.ceil(columns / 4)
    padded = F.pad(
        scales,
        (0, column_blocks * 4 - columns, 0, row_blocks * 128 - rows),
    )
    blocks = padded.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten().contiguous()


def quantize_weight(
    values: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode E2M1 nibbles with 1x16 E4M3 scales and a global FP32 scale."""
    if values.ndim != 2 or values.shape[1] % 16 != 0:
        raise ValueError("NVFP4 quantization expects [M, K] with K divisible by 16")
    rows, columns = values.shape
    fp32 = values.float()
    tiny = torch.finfo(torch.float32).tiny
    global_scale = fp32.abs().amax().clamp_min(tiny) / (448.0 * 6.0)
    blocks = fp32.reshape(rows, columns // 16, 16)
    block_scale = (blocks.abs().amax(dim=-1) / 6.0 / global_scale).clamp(tiny, 448.0).to(torch.float8_e4m3fn)
    normalized = (blocks / (global_scale * block_scale.float()).unsqueeze(-1)).clamp(-6.0, 6.0)
    normalized = normalized.reshape(rows, columns)
    magnitude = torch.bucketize(normalized.abs(), thresholds).to(torch.uint8)
    codes = magnitude | ((normalized < 0).to(torch.uint8) << 3)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, block_scale.contiguous(), global_scale.float()


def dequantize_weight(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct a packed NVFP4 matrix at the requested compute dtype."""
    rows, packed_columns = packed.shape
    codes = torch.empty((rows, packed_columns * 2), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    table = packed.new_tensor(_E2M1_VALUES, dtype=torch.float32)
    values = table[(codes & 0x07).long()]
    values = torch.where((codes & 0x08) != 0, -values, values)
    values = values.reshape(rows, -1, 16)
    values = values * block_scale.view(torch.float8_e4m3fn).float().unsqueeze(-1) * global_scale
    return values.reshape(rows, packed_columns * 2).to(dtype)


def quantization_thresholds(weight: torch.Tensor) -> torch.Tensor:
    """Materialize the fixed E2M1 decision boundaries on the weight device."""
    return weight.new_tensor(_E2M1_THRESHOLDS, dtype=torch.float32)


def native_capability(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[bool, str | None]:
    """Check Blackwell support and the native block-scaled GEMM API."""
    if not inputs.is_cuda:
        return False, "native NVFP4 requires CUDA"
    major, minor = torch.cuda.get_device_capability(inputs.device)
    capability = major * 10 + minor
    available = (
        capability >= 100
        and hasattr(torch, "float4_e2m1fn_x2")
        and hasattr(F, "scaled_mm")
        and hasattr(F, "ScalingType")
        and (weight.shape[1] * 2) % 16 == 0
    )
    return (
        available,
        None
        if available
        else f"native NVFP4 requires SM100+, PyTorch scaled_mm, and K divisible by 16; got SM{capability}",
    )


def _native_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    weight_global_scale: torch.Tensor,
    weight_scale_blocked: torch.Tensor,
    quantization_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Run native W4A4 with the precomputed blocked weight scales."""
    rows = inputs.reshape(-1, weight.shape[1] * 2)
    quantized, scale, global_scale = quantize_weight(rows, quantization_thresholds)
    output = F.scaled_mm(
        quantized.view(torch.float4_e2m1fn_x2),
        weight.t().view(torch.float4_e2m1fn_x2),
        blocked_scales(scale),
        F.ScalingType.BlockWise1x16,
        weight_scale_blocked.view(torch.float8_e4m3fn),
        F.ScalingType.BlockWise1x16,
        F.SwizzleType.SWIZZLE_32_4_4,
        F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=inputs.dtype,
    )
    output = output * (global_scale * weight_global_scale).to(inputs.dtype)
    if bias is not None:
        output = output + bias
    return output.reshape(*inputs.shape[:-1], weight.shape[0])


def linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    weight_global_scale: torch.Tensor,
    weight_scale_blocked: torch.Tensor,
    quantization_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Run the NVFP4 weight-only reference path at the input dtype."""
    weight = dequantize_weight(
        weight,
        weight_scale,
        weight_global_scale,
        inputs.dtype,
    )
    return F.linear(inputs, weight, bias)


def native_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    weight_global_scale: torch.Tensor,
    weight_scale_blocked: torch.Tensor,
    quantization_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Run native NVFP4 and report conflicting CUDA library errors explicitly."""
    try:
        return _native_linear(
            inputs,
            weight,
            weight_scale,
            bias,
            weight_global_scale=weight_global_scale,
            weight_scale_blocked=weight_scale_blocked,
            quantization_thresholds=quantization_thresholds,
        )
    except RuntimeError as exc:
        if "CUBLAS_STATUS" in str(exc):
            raise RuntimeError(
                "NVFP4 cuBLASLt dispatch failed; ensure the process uses the CUDA libraries bundled "
                "with its PyTorch environment instead of a conflicting global LD_LIBRARY_PATH"
            ) from exc
        raise
