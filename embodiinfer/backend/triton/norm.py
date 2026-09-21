"""Inference RMSNorm kernels owned by EmbodiInfer."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rms_norm_kernel(
        inputs,
        weight,
        output,
        width,
        epsilon,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        values = tl.load(inputs + row * width + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / width
        inverse_rms = tl.rsqrt(variance + epsilon)
        scales = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output + row * width + offsets, values * inverse_rms * scales, mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        residual,
        update,
        weight,
        summed,
        normalized,
        width,
        epsilon,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        residual_values = tl.load(
            residual + row * width + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        update_values = tl.load(
            update + row * width + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = residual_values + update_values
        variance = tl.sum(values * values, axis=0) / width
        inverse_rms = tl.rsqrt(variance + epsilon)
        scales = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(summed + row * width + offsets, values, mask=mask)
        tl.store(
            normalized + row * width + offsets,
            values * inverse_rms * scales,
            mask=mask,
        )


def supports_rms_norm(inputs: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        triton is not None
        and inputs.is_cuda
        and weight.is_cuda
        and inputs.dtype in {torch.float16, torch.bfloat16}
        and inputs.dtype == weight.dtype
        and inputs.is_contiguous()
        and weight.is_contiguous()
        and inputs.shape[-1] == weight.numel()
        and inputs.numel() == weight.numel()
        and inputs.shape[-1] <= 65536
    )


def rms_norm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if not supports_rms_norm(inputs, weight):
        raise ValueError("unsupported input for EmbodiInfer Triton RMSNorm")
    output = torch.empty_like(inputs)
    width = inputs.shape[-1]
    rows = inputs.numel() // width
    _rms_norm_kernel[(rows,)](
        inputs,
        weight,
        output,
        width,
        epsilon,
        BLOCK_SIZE=triton.next_power_of_2(width),
        num_warps=8,
    )
    return output


def add_rms_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape != update.shape or not supports_rms_norm(residual, weight):
        raise ValueError("unsupported input for EmbodiInfer Triton residual RMSNorm")
    if not update.is_contiguous() or update.dtype != residual.dtype:
        raise ValueError("residual RMSNorm inputs must share contiguous layout and dtype")
    summed = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    width = residual.shape[-1]
    rows = residual.numel() // width
    _add_rms_norm_kernel[(rows,)](
        residual,
        update,
        weight,
        summed,
        normalized,
        width,
        epsilon,
        BLOCK_SIZE=triton.next_power_of_2(width),
        num_warps=8,
    )
    return summed, normalized


# Adaptive normalization and gated residual fusion.

if triton is not None:

    @triton.jit
    def _ada_rms_norm_kernel(
        x,
        modulation,
        output,
        gate_output,
        x_row_stride,
        modulation_batch_stride,
        sequence_length: tl.constexpr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // sequence_length
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_size

        values = tl.load(x + row * x_row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / hidden_size
        normalized = values * tl.rsqrt(variance + eps)

        modulation_base = modulation + batch * modulation_batch_stride
        scale = tl.load(modulation_base + offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(
            modulation_base + hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            modulation_base + 2 * hidden_size + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            output + row * x_row_stride + offsets,
            normalized * (1.0 + scale) + shift,
            mask=mask,
        )
        # One row per batch item materializes the broadcast residual gate. This
        # also performs the checkpoint's fp32 -> hidden-dtype cast without a
        # second kernel launch.
        tl.store(
            gate_output + batch * hidden_size + offsets,
            gate,
            mask=mask & (row % sequence_length == 0),
        )

else:
    _ada_rms_norm_kernel = None


def ada_rms_norm(
    x: torch.Tensor,
    modulation: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply adaptive RMSNorm with per-batch scale, shift and residual gate.

    ``x`` is [batch, tokens, hidden]; ``modulation`` is [batch, 3 * hidden]
    in scale/shift/gate order. Variance and affine arithmetic use FP32 before
    the output and broadcast gate are cast to the input dtype.
    """

    if x.ndim != 3:
        raise ValueError("AdaRMSNorm input must be [batch, tokens, hidden]")
    if modulation.ndim != 2:
        raise ValueError("AdaRMSNorm modulation must be [batch, 3 * hidden]")
    batch, sequence_length, hidden_size = x.shape
    if modulation.shape != (batch, 3 * hidden_size):
        raise ValueError("AdaRMSNorm modulation shape must equal [batch, 3 * hidden]")
    if modulation.device != x.device:
        raise ValueError("AdaRMSNorm input and modulation must share a device")
    if x.device.type != "cuda" or _ada_rms_norm_kernel is None:
        values = x.float()
        variance = torch.mean(torch.square(values), dim=-1, keepdim=True)
        scale, shift, gate = modulation.unsqueeze(1).chunk(3, dim=-1)
        output = values * torch.rsqrt(variance + eps)
        output = output * (1.0 + scale.float()) + shift.float()
        return output.to(x.dtype), gate.to(x.dtype)

    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty_like(x)
    gate = torch.empty((batch, 1, hidden_size), device=x.device, dtype=x.dtype)
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65536:
        raise ValueError(f"AdaRMSNorm hidden size {hidden_size} is too large")
    _ada_rms_norm_kernel[(batch * sequence_length,)](
        x,
        modulation,
        output,
        gate,
        x.stride(1),
        modulation.stride(0),
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        eps=eps,
        block_size=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output, gate


if triton is not None:

    @triton.jit
    def _gated_residual_kernel(
        residual,
        update,
        gate,
        output,
        residual_stride_batch,
        residual_stride_token,
        update_stride_batch,
        update_stride_token,
        gate_stride_batch,
        gate_stride_token,
        output_stride_batch,
        output_stride_token,
        sequence_length: tl.constexpr,
        hidden_size: tl.constexpr,
        gate_sequence_length: tl.constexpr,
        is_bfloat16: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // sequence_length
        token = row % sequence_length
        gate_token = 0 if gate_sequence_length == 1 else token
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_size
        residual_values = tl.load(
            residual + batch * residual_stride_batch + token * residual_stride_token + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        update_values = tl.load(
            update + batch * update_stride_batch + token * update_stride_token + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate + batch * gate_stride_batch + gate_token * gate_stride_token + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        product = update_values * gate_values
        if is_bfloat16:
            product = product.to(tl.bfloat16).to(tl.float32)
        else:
            product = product.to(tl.float16).to(tl.float32)
        tl.store(
            output + batch * output_stride_batch + token * output_stride_token + offsets,
            residual_values + product,
            mask=mask,
        )

else:
    _gated_residual_kernel = None


def gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Compute ``residual + update * gate`` in one launch."""

    if residual.ndim != 3 or update.shape != residual.shape:
        raise ValueError("gated residual inputs must have the same [batch, tokens, width] shape")
    if residual.device != update.device or residual.dtype != update.dtype:
        raise ValueError("gated residual inputs must share device and dtype")
    if residual.stride(2) != 1 or update.stride(2) != 1:
        raise ValueError("gated residual input widths must be contiguous")
    if gate.ndim != 3 or gate.shape[0] != residual.shape[0]:
        raise ValueError("residual gate must be [batch, 1|tokens, hidden]")
    if gate.shape[1] not in (1, residual.shape[1]) or gate.shape[2] != residual.shape[2]:
        raise ValueError("residual gate shape is not broadcast-compatible")
    if gate.device != residual.device or gate.dtype != residual.dtype:
        raise ValueError("residual gate must share device and dtype")
    if gate.stride(2) != 1:
        raise ValueError("residual gate width must be contiguous")
    if (
        residual.device.type != "cuda"
        or _gated_residual_kernel is None
        or residual.dtype not in (torch.bfloat16, torch.float16)
    ):
        return residual + update * gate

    output = torch.empty_like(residual)
    batch, sequence_length, hidden_size = residual.shape
    block_size = triton.next_power_of_2(hidden_size)
    grid = (batch * sequence_length,)
    _gated_residual_kernel[grid](
        residual,
        update,
        gate,
        output,
        residual.stride(0),
        residual.stride(1),
        update.stride(0),
        update.stride(1),
        gate.stride(0),
        gate.stride(1),
        output.stride(0),
        output.stride(1),
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        gate_sequence_length=gate.shape[1],
        is_bfloat16=residual.dtype == torch.bfloat16,
        block_size=block_size,
        num_warps=8 if block_size >= 2048 else 4,
        enable_fp_fusion=False,  # keep BF16 multiplication rounded before addition
    )
    return output


__all__ = ["ada_rms_norm", "add_rms_norm", "gated_residual", "rms_norm", "supports_rms_norm"]
