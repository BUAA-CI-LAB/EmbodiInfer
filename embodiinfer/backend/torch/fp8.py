"""Torch FP8 weight conversion, capability probes, and linear kernels."""

from __future__ import annotations

import torch
from torch.nn import functional as F

# Bound each FP32 activation workspace to 256 MiB, or one unusually wide row.
_ACTIVATION_CHUNK_ELEMENTS = 64 * 1024 * 1024


def _quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("this PyTorch build does not expose torch.float8_e4m3fn")
    fp8_amax = 448.0
    rows, columns = weight.shape
    quantized = torch.empty((rows, columns), dtype=fp8_dtype, device=weight.device)
    scales = torch.empty((rows,), dtype=torch.float32, device=weight.device)
    rows_per_chunk = max(1, min(rows, 1_048_576 // max(1, columns)))
    with torch.no_grad():
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk = weight[start:end].float()
            maximum = chunk.abs().amax(dim=1)
            scale = torch.where(maximum > 0, maximum / fp8_amax, torch.ones_like(maximum))
            quantized[start:end].copy_((chunk / scale[:, None]).clamp(-fp8_amax, fp8_amax).to(fp8_dtype))
            scales[start:end].copy_(scale)
    return quantized, scales


def _quantize_weight_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("this PyTorch build does not expose torch.float8_e4m3fn")
    fp32 = weight.float()
    maximum = fp32.abs().amax()
    scale = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
    quantized = (fp32 / scale).clamp(-448.0, 448.0).to(fp8_dtype)
    return quantized, scale.float()


def _quantize_activation_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = rows.float().abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
    quantized = (rows / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


def _dynamic_fp8_activation(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize independent rows without materializing full-size FP32 temporaries."""
    rows = inputs.reshape(-1, inputs.shape[-1])
    if rows.numel() <= _ACTIVATION_CHUNK_ELEMENTS:
        return _quantize_activation_rows(rows)
    rows_per_chunk = max(1, _ACTIVATION_CHUNK_ELEMENTS // rows.shape[1])
    quantized = torch.empty(rows.shape, dtype=torch.float8_e4m3fn, device=rows.device)
    scales = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=rows.device)
    for start in range(0, rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, rows.shape[0])
        values, scale = _quantize_activation_rows(rows[start:end])
        quantized[start:end].copy_(values)
        scales[start:end].copy_(scale)
    return quantized, scales


def _dynamic_fp8_activation_tensorwise(
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = inputs.reshape(-1, inputs.shape[-1])
    maximum = rows.float().abs().amax()
    scale = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
    quantized = (rows / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale.float()


def quantize_weight(
    weight: torch.Tensor, *, backend: str = "auto", scaling_scheme: str = "auto"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create E4M3 weights, preserving the native Blackwell tensorwise default."""
    if scaling_scheme == "auto":
        scaling_scheme = "channelwise"
        if weight.is_cuda and backend in {"auto", "native"}:
            major, _ = torch.cuda.get_device_capability(weight.device)
            if major >= 10:
                scaling_scheme = "tensorwise"
    if scaling_scheme == "tensorwise":
        return _quantize_weight_per_tensor(weight)
    return _quantize_weight_per_channel(weight)


def dequantize_weight(weight: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert E4M3 weights using scalar or per-output-channel scales."""
    scale = weight_scale.to(dtype)
    if scale.ndim != 0:
        scale = scale[:, None]
    return weight.to(dtype) * scale


def native_capability(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[bool, str | None]:
    """Check native W8A8 hardware, API availability, and matrix alignment."""
    if not inputs.is_cuda:
        return False, "native FP8 W8A8 requires CUDA"
    major, minor = torch.cuda.get_device_capability(inputs.device)
    capability = major * 10 + minor
    if capability < 89:
        return False, f"native FP8 W8A8 requires SM89+, got SM{capability}"
    if not (
        (hasattr(torch, "_scaled_mm") or hasattr(F, "scaled_mm"))
        and weight.shape[1] % 16 == 0
        and weight.shape[0] % 16 == 0
    ):
        return False, "native FP8 W8A8 requires torch._scaled_mm and K/N divisible by 16"
    return True, None


def native_linear(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Run native dynamic W8A8 using the existing tensorwise or row/channelwise scales."""
    major, _ = torch.cuda.get_device_capability(inputs.device)
    if major >= 10:
        if weight_scale.ndim != 0:
            raise RuntimeError(
                "SM100+ FP8 native dispatch requires tensorwise weights; load and quantize on CUDA "
                "or set scaling_scheme='tensorwise'"
            )
        quantized, input_scale = _dynamic_fp8_activation_tensorwise(inputs)
        output = F.scaled_mm(
            quantized,
            weight.t(),
            input_scale,
            F.ScalingType.TensorWise,
            weight_scale,
            F.ScalingType.TensorWise,
            output_dtype=inputs.dtype,
        )
        if bias is not None:
            output = output + bias
        return output.reshape(*inputs.shape[:-1], weight.shape[0])
    quantized, input_scale = _dynamic_fp8_activation(inputs)
    output = torch._scaled_mm(
        quantized,
        weight.t(),
        scale_a=input_scale,
        scale_b=weight_scale.reshape(1, -1),
        bias=bias,
        out_dtype=inputs.dtype,
    )
    return output.reshape(*inputs.shape[:-1], weight.shape[0])


def linear(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Run the weight-only reference path after dequantizing to the input dtype."""
    return F.linear(inputs, dequantize_weight(weight, weight_scale, inputs.dtype), bias)
