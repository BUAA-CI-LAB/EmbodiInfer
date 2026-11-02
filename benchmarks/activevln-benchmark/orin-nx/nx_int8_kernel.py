"""Dynamic W8A8 projections with fused row quantization and FP32 output rescaling."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _quantize_rows(
        X,
        Q,
        S,
        ChannelScale,
        K: tl.constexpr,
        SX: tl.constexpr,
        HAS_CHANNEL_SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        depth = tl.arange(0, BLOCK)
        values = tl.load(X + row * SX + depth, depth < K, 0).to(tl.float32)
        if HAS_CHANNEL_SCALE:
            channel_scale = tl.load(ChannelScale + depth, depth < K, 1)
            # Preserve the eager divide -> input-dtype cast before row quantization.
            values = tl.div_rn(values, channel_scale).to(X.dtype.element_ty).to(tl.float32)
        maximum = tl.max(tl.abs(values), 0)
        scale = tl.where(maximum > 0, maximum * (1.0 / 127.0), 1.0)
        quantized = libdevice.nearbyint(tl.div_rn(values, scale)).to(tl.int8)
        tl.store(Q + row * K + depth, quantized, depth < K)
        tl.store(S + row, scale)

    @triton.jit
    def _matmul(
        X,
        W,
        XS,
        WS,
        Bias,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        SW0: tl.constexpr,
        SW1: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        columns = tl.program_id(1) * BN + tl.arange(0, BN)
        depth = tl.arange(0, BK)
        accumulator = tl.zeros((BM, BN), tl.int32)
        for block in range(tl.cdiv(K, BK)):
            indices = block * BK + depth
            a = tl.load(
                X + rows[:, None] * K + indices[None, :], (rows[:, None] < M) & (indices[None, :] < K), 0
            )
            b = tl.load(
                W + indices[:, None] * SW0 + columns[None, :] * SW1,
                (indices[:, None] < K) & (columns[None, :] < N),
                0,
            )
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        input_scale = tl.load(XS + rows, rows < M, 0)
        weight_scale = tl.load(WS + columns, columns < N, 0)
        output = accumulator.to(tl.float32) * input_scale[:, None] * weight_scale[None, :]
        if HAS_BIAS:
            output += tl.load(Bias + columns, columns < N, 0).to(tl.float32)[None, :]
        tl.store(
            Y + rows[:, None] * N + columns[None, :], output, (rows[:, None] < M) & (columns[None, :] < N)
        )


def int8_capability(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[bool, str | None]:
    """Require a supported Ampere/Ada/Hopper/Thor device and W8A8 operand layouts."""
    if triton is None or not inputs.is_cuda:
        return False, "Triton INT8 requires Triton and CUDA"
    capability = torch.cuda.get_device_capability(inputs.device)
    if capability not in ((8, 0), (8, 6), (8, 7), (8, 9), (9, 0), (11, 0)):
        return False, f"Triton INT8 is not enabled for SM{capability[0]}{capability[1]}"
    if (
        inputs.ndim < 1
        or inputs.dtype not in (torch.float16, torch.bfloat16)
        or weight.ndim != 2
        or weight.dtype != torch.int8
        or weight.device != inputs.device
        or weight_scale.device != inputs.device
        or weight_scale.dtype != torch.float32
        or weight_scale.shape != (weight.shape[1],)
        or not weight_scale.is_contiguous()
        or inputs.shape[-1] != weight.shape[0]
        or not 0 < weight.shape[0] <= 16384
        or weight.shape[1] < 1
    ):
        return False, "Triton INT8 expects FP16/BF16 inputs, INT8 [K,N], FP32 [N] scales and 0<K<=16384"
    return True, None


def int8_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply per-row W8A8 with INT32 accumulation and one final output cast.

    Activations use absmax / 127, round-to-nearest-even and a scale of one for
    all-zero rows. The epilogue multiplies input then weight scales in FP32,
    adds optional bias in FP32, and rounds to the input dtype. Weight layout is
    [K,N], including strided slices of packed projections. Optional ``input_scale``
    supplies positive finite FP32 channel divisors [K]; divide in FP32 and cast
    back to the input dtype before quantization, fused into the row kernel.
    The caller validates scale values before capture. Allocations and all
    kernels support CUDA Graph capture; no full floating weight is materialized.
    """
    available, reason = int8_capability(inputs, weight, weight_scale)
    if not available:
        raise RuntimeError(reason)
    if torch.is_grad_enabled() and (
        inputs.requires_grad
        or (bias is not None and bias.requires_grad)
        or (input_scale is not None and input_scale.requires_grad)
    ):
        raise RuntimeError("Triton INT8 projections support inference only")
    width = weight.shape[1]
    if bias is not None and (
        bias.shape != (width,) or bias.device != inputs.device or not bias.is_contiguous()
    ):
        raise ValueError("INT8 bias must be contiguous [N] on the input device")
    depth = weight.shape[0]
    if input_scale is not None and (
        input_scale.shape != (depth,)
        or input_scale.dtype != torch.float32
        or input_scale.device != inputs.device
        or not input_scale.is_contiguous()
    ):
        raise ValueError("INT8 input_scale must be contiguous FP32 [K] on the input device")
    flat = inputs.reshape(-1, depth)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    rows = flat.shape[0]
    output = torch.empty((*inputs.shape[:-1], width), dtype=inputs.dtype, device=inputs.device)
    if rows == 0:
        return output
    quantized = torch.empty((rows, depth), dtype=torch.int8, device=inputs.device)
    scales = torch.empty(rows, dtype=torch.float32, device=inputs.device)
    _quantize_rows[(rows,)](
        flat,
        quantized,
        scales,
        weight_scale if input_scale is None else input_scale,
        depth,
        flat.stride(0),
        input_scale is not None,
        triton.next_power_of_2(depth),
    )
    # Measured on Orin NX for decode/tree rows and larger prefill projections.
    block_m = 32 if rows <= 32 else 64
    _matmul[(triton.cdiv(rows, block_m), triton.cdiv(width, 128))](
        quantized,
        weight,
        scales,
        weight_scale,
        weight_scale if bias is None else bias,
        output,
        rows,
        width,
        depth,
        weight.stride(0),
        weight.stride(1),
        bias is not None,
        block_m,
        128,
        64,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return output
