"""Inference activation kernels owned by VVLA."""

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
    def _swiglu_kernel(
        packed,
        output,
        intermediate_size,
        packed_row_stride,
        output_row_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < intermediate_size
        gate = tl.load(
            packed + row * packed_row_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            packed + row * packed_row_stride + intermediate_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output + row * output_row_stride + offsets,
            gate * tl.sigmoid(gate) * up,
            mask=mask,
        )


def supports_swiglu(packed: torch.Tensor) -> bool:
    return (
        triton is not None
        and packed.is_cuda
        and packed.dtype in {torch.float16, torch.bfloat16}
        and packed.is_contiguous()
        and packed.shape[-1] % 2 == 0
    )


def swiglu(packed: torch.Tensor) -> torch.Tensor:
    if not supports_swiglu(packed):
        raise ValueError("unsupported input for VVLA Triton SwiGLU")
    intermediate_size = packed.shape[-1] // 2
    rows = packed.numel() // packed.shape[-1]
    output = torch.empty(
        (*packed.shape[:-1], intermediate_size),
        dtype=packed.dtype,
        device=packed.device,
    )
    block_size = 256
    _swiglu_kernel[(rows, triton.cdiv(intermediate_size, block_size))](
        packed,
        output,
        intermediate_size,
        packed.shape[-1],
        intermediate_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


if triton is not None:

    @triton.jit
    def _gated_gelu_kernel(
        gate,
        up,
        output,
        gate_stride_batch,
        gate_stride_token,
        up_stride_batch,
        up_stride_token,
        output_stride_batch,
        output_stride_token,
        sequence_length: tl.constexpr,
        intermediate_size: tl.constexpr,
        is_bfloat16: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // sequence_length
        token = row % sequence_length
        offsets = tl.arange(0, block_size)
        mask = offsets < intermediate_size
        gate_values = tl.load(
            gate + batch * gate_stride_batch + token * gate_stride_token + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up_values = tl.load(
            up + batch * up_stride_batch + token * up_stride_token + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # PyTorch's approximate="tanh" GELU can be written as
        # x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715*x^3)).
        inner = 0.7978845608028654 * (gate_values + 0.044715 * gate_values * gate_values * gate_values)
        activated = gate_values * tl.sigmoid(2.0 * inner)
        if is_bfloat16:
            activated = activated.to(tl.bfloat16).to(tl.float32)
        else:
            activated = activated.to(tl.float16).to(tl.float32)
        tl.store(
            output + batch * output_stride_batch + token * output_stride_token + offsets,
            activated * up_values,
            mask=mask,
        )

else:
    _gated_gelu_kernel = None


def gated_gelu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute tanh-approximate GELU(gate) times up in one launch."""

    if gate.ndim != 3 or up.shape != gate.shape:
        raise ValueError("gated GELU inputs must have the same [batch, tokens, width] shape")
    if gate.device != up.device or gate.dtype != up.dtype:
        raise ValueError("gated GELU inputs must share device and dtype")
    if gate.stride(2) != 1 or up.stride(2) != 1:
        raise ValueError("gated GELU input widths must be contiguous")
    if (
        gate.device.type != "cuda"
        or _gated_gelu_kernel is None
        or gate.dtype not in (torch.bfloat16, torch.float16)
    ):
        return torch.nn.functional.gelu(gate, approximate="tanh") * up

    output = torch.empty_like(gate, memory_format=torch.contiguous_format)
    batch, sequence_length, intermediate_size = gate.shape
    block_size = triton.next_power_of_2(intermediate_size)
    grid = (batch * sequence_length,)
    _gated_gelu_kernel[grid](
        gate,
        up,
        output,
        gate.stride(0),
        gate.stride(1),
        up.stride(0),
        up.stride(1),
        output.stride(0),
        output.stride(1),
        sequence_length=sequence_length,
        intermediate_size=intermediate_size,
        is_bfloat16=gate.dtype == torch.bfloat16,
        block_size=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output


__all__ = ["gated_gelu", "supports_swiglu", "swiglu"]
