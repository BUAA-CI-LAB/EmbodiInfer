"""Torch INT8 weight conversion, capability probes, and linear kernels."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return transposed INT8 weights and per-output-channel FP32 scales."""
    fp32 = weight.float()
    maximum = fp32.abs().amax(dim=1)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    quantized = (fp32 / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.t().contiguous(), scale.float()


def _quantize_activation_per_row(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = inputs.reshape(-1, inputs.shape[-1])
    maximum = rows.float().abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    quantized = (rows / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def native_capability(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[bool, str | None]:
    """Check native INT8 hardware, API availability, and matrix alignment."""
    if not inputs.is_cuda:
        return False, "native INT8 requires CUDA"
    major, minor = torch.cuda.get_device_capability(inputs.device)
    capability = major * 10 + minor
    available = (
        capability >= 75
        and hasattr(torch, "_int_mm")
        and weight.shape[0] % 8 == 0
        and weight.shape[1] % 8 == 0
    )
    return (
        available,
        None
        if available
        else f"native INT8 requires SM75+, torch._int_mm, and K/N divisible by 8; got SM{capability}",
    )


def native_linear(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Run dynamic INT8 GEMM, including the existing small-row padding."""
    quantized, input_scale = _quantize_activation_per_row(inputs)
    rows = quantized.shape[0]
    if rows <= 16:
        quantized = F.pad(quantized, (0, 0, 0, 17 - rows))
    output = torch._int_mm(quantized, weight)[:rows]
    output = output.to(inputs.dtype)
    output = output * input_scale.to(inputs.dtype)
    output = output * weight_scale.to(inputs.dtype)[None, :]
    if bias is not None:
        output = output + bias
    return output.reshape(*inputs.shape[:-1], weight.shape[1])


def linear(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Run the weight-only reference path at the input dtype."""
    weight = weight.t().to(inputs.dtype)
    weight = weight * weight_scale.to(inputs.dtype)[:, None]
    return F.linear(inputs, weight, bias)
