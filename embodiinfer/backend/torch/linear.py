"""Inference projections with explicit FP32 output before bias and final rounding."""

from __future__ import annotations

import torch


def linear_fp32_output(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Project CUDA FP16/BF16 inputs, adding bias before the single output cast.

    Weights retain their original dtype/storage. This avoids a low-precision
    intermediate before the bias addition without changing process-wide BLAS
    settings. Different GEMM shapes can still change FP32 accumulation order;
    callers must validate complete model outputs separately. Requires a Torch
    CUDA build supporting ``mm(..., out_dtype=torch.float32)`` and no autograd.
    """
    if torch.is_grad_enabled():
        raise ValueError("FP32-output projection is inference-only")
    if inputs.device.type != "cuda" or inputs.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FP32-output projection requires CUDA FP16/BF16 inputs")
    if (
        inputs.ndim < 2
        or weight.ndim != 2
        or inputs.shape[-1] != weight.shape[1]
        or weight.device != inputs.device
        or weight.dtype != inputs.dtype
    ):
        raise ValueError("incompatible projection input and weight")
    if bias is not None and (
        bias.shape != (weight.shape[0],) or bias.device != inputs.device or bias.dtype != inputs.dtype
    ):
        raise ValueError("incompatible projection bias")
    projected = torch.mm(inputs.reshape(-1, inputs.shape[-1]), weight.T, out_dtype=torch.float32)
    if bias is not None:
        projected = projected + bias.float()
    return projected.to(inputs.dtype).reshape(*inputs.shape[:-1], weight.shape[0])
