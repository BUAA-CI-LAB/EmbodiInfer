"""Fused activation conversion and rounded epilogue around native INT8 GEMM."""

from __future__ import annotations

import torch

from ..torch.int8 import native_capability

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _quantize(
        X,
        Q,
        S,
        M: tl.constexpr,
        K: tl.constexpr,
        SX: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        depth = tl.arange(0, BLOCK)
        values = tl.load(X + row * SX + depth, (row < M) & (depth < K), 0).to(tl.float32)
        nan = values != values
        has_nan = tl.sum(nan.to(tl.int32), 0) > 0
        maximum = tl.max(tl.where(nan, 0.0, tl.abs(values)), 0)
        # Torch's CUDA tensor/scalar division uses a rounded FP32 reciprocal.
        scale = tl.where((maximum > 0) & ~has_nan, maximum * (1.0 / 127.0), 1.0)
        rounded = libdevice.nearbyint(tl.div_rn(values, scale))
        rounded = tl.where(rounded != rounded, 0.0, rounded)
        quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
        tl.store(Q + row * K + depth, quantized, depth < K)
        tl.store(S + row, scale)

    @triton.jit
    def _epilogue(
        A,
        XS,
        WS,
        B,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, column = indices // N, indices % N
        valid = row < M
        dtype = Y.dtype.element_ty
        # Retain every native BF16/FP16 rounding boundary, including the initial
        # INT32 conversion. FP32 rescaling with a single final cast is different.
        accumulator = tl.load(A + indices, valid, 0).to(dtype).to(tl.float32)
        xs = tl.load(XS + row, valid, 1).to(dtype).to(tl.float32)
        ws = tl.load(WS + column, valid, 1).to(dtype).to(tl.float32)
        result = (accumulator * xs).to(dtype).to(tl.float32)
        result = (result * ws).to(dtype).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(B + column, valid, 0).to(tl.float32)
            result = (result + bias).to(dtype).to(tl.float32)
        tl.store(Y + indices, result, valid)

    @triton.jit
    def _matmul_rounded(
        X,
        W,
        XS,
        WS,
        Bias,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
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
                W + indices[:, None] * N + columns[None, :],
                (indices[:, None] < K) & (columns[None, :] < N),
                0,
            )
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        dtype = Y.dtype.element_ty
        xs = tl.load(XS + rows, rows < M, 1).to(dtype).to(tl.float32)
        ws = tl.load(WS + columns, columns < N, 1).to(dtype).to(tl.float32)
        result = (accumulator.to(dtype).to(tl.float32) * xs[:, None]).to(dtype).to(tl.float32)
        result = (result * ws[None, :]).to(dtype).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(Bias + columns, columns < N, 0).to(tl.float32)
            result = (result + bias[None, :]).to(dtype).to(tl.float32)
        tl.store(
            Y + rows[:, None] * N + columns[None, :], result, (rows[:, None] < M) & (columns[None, :] < N)
        )


def capability(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[bool, str | None]:
    """Require native INT8 GEMM plus Triton and supported floating activation rows."""
    if triton is None:
        return False, "fused native INT8 requires Triton"
    if (
        inputs.ndim < 1
        or weight.ndim != 2
        or inputs.dtype not in (torch.float16, torch.bfloat16)
        or weight.dtype != torch.int8
        or weight.device != inputs.device
        or not weight.is_contiguous()
        or weight_scale.device != inputs.device
        or weight_scale.dtype != torch.float32
        or weight_scale.shape != (weight.shape[1],)
        or not weight_scale.is_contiguous()
        or inputs.shape[-1] != weight.shape[0]
        or not 0 < weight.shape[0] <= 32768
        or weight.shape[1] < 1
    ):
        return False, "fused native INT8 expects FP16/BF16 input, contiguous INT8 [K,N], FP32 [N], 0<K<=32768"
    return native_capability(inputs, weight, weight_scale)


def quantize_activation(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert finite or nonfinite rows exactly as native activation quantization.

    Small row counts include the native GEMM's zero padding to 17 rows. The
    caller validates device, dtype and width through ``capability`` first.
    """
    depth = inputs.shape[-1]
    flat = inputs.reshape(-1, depth)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    rows = flat.shape[0]
    padded = 17 if 0 < rows <= 16 else rows
    quantized = torch.empty((padded, depth), device=inputs.device, dtype=torch.int8)
    scales = torch.empty((padded, 1), device=inputs.device, dtype=torch.float32)
    if padded:
        _quantize[(padded,)](
            flat,
            quantized,
            scales,
            rows,
            depth,
            flat.stride(0),
            triton.next_power_of_2(depth),
            num_warps=8 if depth >= 8192 else 4,
            enable_fp_fusion=False,
        )
    return quantized, scales


def linear(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Fuse conversion and output scaling while preserving native GEMM arithmetic.

    This opt-in backend retains the native weight payload, activation scales,
    matrix dimensions, integer GEMM and intermediate output-dtype rounding.
    It owns its output and supports startup CUDA Graph capture and replay.
    """
    available, reason = capability(inputs, weight, weight_scale)
    if not available:
        raise RuntimeError(reason)
    if torch.is_grad_enabled() and (inputs.requires_grad or (bias is not None and bias.requires_grad)):
        raise RuntimeError("fused native INT8 supports inference only")
    width = weight.shape[1]
    if bias is not None and (
        bias.shape != (width,)
        or bias.device != inputs.device
        or bias.dtype != inputs.dtype
        or not bias.is_contiguous()
    ):
        raise ValueError("fused native INT8 bias must be contiguous [N] with the input device and dtype")
    output = torch.empty((*inputs.shape[:-1], width), device=inputs.device, dtype=inputs.dtype)
    if not inputs.numel():
        return output
    quantized, scales = quantize_activation(inputs)
    accumulator = torch._int_mm(quantized, weight)
    rows = inputs.numel() // inputs.shape[-1]
    _epilogue[(triton.cdiv(rows * width, 1024),)](
        accumulator,
        scales,
        weight_scale,
        weight_scale if bias is None else bias,
        output,
        rows,
        width,
        bias is not None,
        1024,
        enable_fp_fusion=False,
    )
    return output


def triton_capability(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[bool, str | None]:
    """Require the validated native operand layouts and Ampere or newer INT8 MMA."""
    available, reason = capability(inputs, weight, weight_scale)
    if not available:
        return available, reason
    if torch.cuda.get_device_capability(inputs.device)[0] < 8:
        return False, "rounded Triton INT8 requires SM80 or newer"
    return True, None


def triton_linear(
    inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Fuse integer GEMM and native-rounded rescaling without an INT32 matrix.

    Integer products accumulate in registers. For K<=32768 and signed payloads
    in [-127,127], their sum cannot overflow INT32, so changed tiling preserves
    the native accumulator exactly. All subsequent dtype boundaries are retained.
    """
    available, reason = triton_capability(inputs, weight, weight_scale)
    if not available:
        raise RuntimeError(reason)
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (inputs, weight_scale, bias) if value is not None
    ):
        raise RuntimeError("rounded Triton INT8 supports inference only")
    width = weight.shape[1]
    if bias is not None and (
        bias.shape != (width,)
        or bias.device != inputs.device
        or bias.dtype != inputs.dtype
        or not bias.is_contiguous()
    ):
        raise ValueError("rounded Triton INT8 bias must be contiguous [N] with the input device and dtype")
    output = torch.empty((*inputs.shape[:-1], width), device=inputs.device, dtype=inputs.dtype)
    if not inputs.numel():
        return output
    quantized, scales = quantize_activation(inputs)
    rows, depth = inputs.numel() // inputs.shape[-1], inputs.shape[-1]
    block_m = 32 if rows <= 32 else 64
    _matmul_rounded[(triton.cdiv(rows, block_m), triton.cdiv(width, 128))](
        quantized,
        weight,
        scales,
        weight_scale,
        weight_scale if bias is None else bias,
        output,
        rows,
        width,
        depth,
        bias is not None,
        block_m,
        128,
        64,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return output
