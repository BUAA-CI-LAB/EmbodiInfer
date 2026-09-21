"""Weight-only FP8 linear kernel for pre-Ada NVIDIA GPUs."""

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
    def _fp8_weight_only_linear_kernel(
        inputs,
        weight,
        weight_scale,
        bias,
        output,
        rows,
        output_features,
        input_features: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        column_block = tl.program_id(1)
        row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        column_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start_k in range(0, input_features, BLOCK_K):
            k_offsets = start_k + tl.arange(0, BLOCK_K)
            input_values = tl.load(
                inputs + row_offsets[:, None] * input_features + k_offsets[None, :],
                mask=(row_offsets[:, None] < rows) & (k_offsets[None, :] < input_features),
                other=0.0,
            ).to(tl.float16)
            loaded_weight = tl.load(
                weight + column_offsets[:, None] * input_features + k_offsets[None, :],
                mask=(column_offsets[:, None] < output_features) & (k_offsets[None, :] < input_features),
                other=0,
            )
            # Ampere cannot consume e4m3fn directly, but Triton's e4b15 has the
            # same bit layout with an exponent bias eight larger. Reinterpret
            # the byte and compensate by 2**8 in the per-channel scale.
            weight_values = loaded_weight.to(tl.float8e4b15, bitcast=True)
            weight_values = weight_values.to(tl.float16)
            accumulator += tl.dot(input_values, tl.trans(weight_values))
        scales = tl.load(
            weight_scale + column_offsets,
            mask=column_offsets < output_features,
            other=0.0,
        )
        accumulator *= (scales * 256.0)[None, :]
        if HAS_BIAS:
            bias_values = tl.load(
                bias + column_offsets,
                mask=column_offsets < output_features,
                other=0.0,
            )
            accumulator += bias_values[None, :]
        tl.store(
            output + row_offsets[:, None] * output_features + column_offsets[None, :],
            accumulator,
            mask=(row_offsets[:, None] < rows) & (column_offsets[None, :] < output_features),
        )


def supports_fp8_weight_only(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    if (
        triton is None
        or not inputs.is_cuda
        or inputs.dtype not in {torch.float16, torch.bfloat16}
        or weight.dtype != getattr(torch, "float8_e4m3fn", None)
        or weight.ndim != 2
        or weight_scale.ndim != 1
        or weight_scale.shape[0] != weight.shape[0]
        or inputs.shape[-1] != weight.shape[1]
    ):
        return False
    major, minor = torch.cuda.get_device_capability(inputs.device)
    return major * 10 + minor >= 80


def fp8_weight_only_capability(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[bool, str | None]:
    """Report whether the registered FP8 weight-only kernel can run this projection."""
    if not inputs.is_cuda:
        return False, "Triton FP8 weight-only requires CUDA"
    major, minor = torch.cuda.get_device_capability(inputs.device)
    capability = major * 10 + minor
    if capability < 80:
        return False, f"Triton FP8 weight-only requires SM80+, got SM{capability}"
    if not supports_fp8_weight_only(inputs, weight, weight_scale):
        return False, "the requested Triton FP8 weight-only backend is unavailable"
    return True, None


def fp8_weight_only_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not supports_fp8_weight_only(inputs, weight, weight_scale):
        raise ValueError("unsupported input for VVLA Triton FP8 weight-only linear")
    contiguous_inputs = inputs.contiguous()
    input_features = weight.shape[1]
    output_features = weight.shape[0]
    rows = contiguous_inputs.numel() // input_features
    output = torch.empty(
        (*inputs.shape[:-1], output_features),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    block_m = 16
    block_n = 64
    block_k = 32
    bias_pointer = weight_scale if bias is None else bias
    _fp8_weight_only_linear_kernel[(triton.cdiv(rows, block_m), triton.cdiv(output_features, block_n))](
        contiguous_inputs,
        weight.view(torch.uint8),
        weight_scale,
        bias_pointer,
        output,
        rows,
        output_features,
        input_features=input_features,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


__all__ = ["fp8_weight_only_linear", "supports_fp8_weight_only", "fp8_weight_only_capability"]
