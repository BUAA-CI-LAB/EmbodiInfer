"""Fused inference operators retaining intermediate FP16/BF16 rounding.

These differ from fusion that keeps all intermediates in FP32: normalization is
rounded before multiplying by its weight, SiLU before multiplying by its gate,
and each rotary product before their sum. RMSNorm retains Torch's FP32 mean
reduction; SwiGLU uses CUDA libdevice arithmetic to preserve its rounding.
"""

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
    def _squares(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
        tl.store(Y + offsets, values * values, offsets < N)

    @triton.jit
    def _norm(X, Mean, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        x = tl.load(X + row * WIDTH + columns, columns < WIDTH, other=0).to(tl.float32)
        weight = tl.load(W + columns, columns < WIDTH, other=0).to(tl.float32)
        variance = tl.load(Mean + row)
        normalized = (x * tl.rsqrt(variance + EPS)).to(X.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * WIDTH + columns, normalized * weight, columns < WIDTH)

    @triton.jit
    def _swiglu(Gate, Up, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        gate = tl.load(Gate + offsets, offsets < N, other=0).to(tl.float32)
        up = tl.load(Up + offsets, offsets < N, other=0).to(tl.float32)
        activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(Gate.dtype.element_ty).to(tl.float32)
        tl.store(Y + offsets, activated * up, offsets < N)

    @triton.jit
    def _rope(
        Q,
        K,
        Cos,
        Sin,
        Qout,
        Kout,
        qh: tl.constexpr,
        qt: tl.constexpr,
        kh: tl.constexpr,
        kt: tl.constexpr,
        QHEADS: tl.constexpr,
        KHEADS: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token, head = tl.program_id(0), tl.program_id(1)
        offsets = tl.arange(0, BLOCK)
        paired = tl.where(offsets < D // 2, offsets + D // 2, offsets - D // 2)
        sign = tl.where(offsets < D // 2, -1.0, 1.0)
        cosine = tl.load(Cos + token * D + offsets, offsets < D, other=0).to(tl.float32)
        sine = tl.load(Sin + token * D + offsets, offsets < D, other=0).to(tl.float32)
        q = tl.load(Q + head * qh + token * qt + offsets, offsets < D, other=0).to(tl.float32)
        qp = tl.load(Q + head * qh + token * qt + paired, offsets < D, other=0).to(tl.float32)
        first = (q * cosine).to(Q.dtype.element_ty).to(tl.float32)
        second = (sign * qp * sine).to(Q.dtype.element_ty).to(tl.float32)
        tl.store(Qout + (head * T + token) * D + offsets, first + second, offsets < D)
        k = tl.load(K + head * kh + token * kt + offsets, (head < KHEADS) & (offsets < D), other=0).to(
            tl.float32
        )
        kp = tl.load(K + head * kh + token * kt + paired, (head < KHEADS) & (offsets < D), other=0).to(
            tl.float32
        )
        first = (k * cosine).to(K.dtype.element_ty).to(tl.float32)
        second = (sign * kp * sine).to(K.dtype.element_ty).to(tl.float32)
        tl.store(Kout + (head * T + token) * D + offsets, first + second, (head < KHEADS) & (offsets < D))


def _require(*values: torch.Tensor) -> None:
    if triton is None or not values or not values[0].is_cuda:
        raise ValueError("rounded fused operators require CUDA and Triton")
    if values[0].dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("rounded fused operators require FP16/BF16 tensors")
    if any(value.device != values[0].device or value.dtype != values[0].dtype for value in values):
        raise ValueError("rounded fused inputs must share device and dtype")
    if torch.is_grad_enabled() and any(value.requires_grad for value in values):
        raise ValueError("rounded fused operators are inference-only")


def rounded_rms_norm(inputs: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Normalize contiguous rows, rounding to input dtype before the weight product."""
    _require(inputs, weight)
    if not inputs.is_contiguous() or not weight.is_contiguous() or inputs.shape[-1] != weight.numel():
        raise ValueError("RMSNorm needs contiguous rows and a matching weight vector")
    width = weight.numel()
    if width < 1 or width > 65536 or epsilon <= 0:
        raise ValueError("unsupported RMSNorm width or epsilon")
    squares = torch.empty(inputs.shape, dtype=torch.float32, device=inputs.device)
    _squares[(triton.cdiv(inputs.numel(), 256),)](
        inputs,
        squares,
        inputs.numel(),
        256,
        enable_fp_fusion=False,
    )
    # Match the reference reduction's shape, dtype and accumulation order.
    mean = squares.mean(-1, keepdim=True)
    output = torch.empty_like(inputs)
    _norm[(inputs.numel() // width,)](
        inputs,
        mean,
        weight,
        output,
        width,
        epsilon,
        triton.next_power_of_2(width),
        num_warps=4 if width <= 4096 else 8,
        enable_fp_fusion=False,
    )
    return output


def rounded_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Apply SiLU, round, multiply by up, and round once more like eager Torch."""
    _require(gate, up)
    if gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("SwiGLU inputs must have identical contiguous layouts")
    output = torch.empty_like(gate)
    _swiglu[(triton.cdiv(gate.numel(), 256),)](
        gate,
        up,
        output,
        gate.numel(),
        256,
        enable_fp_fusion=False,
    )
    return output


def rounded_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate B=1 BHSD Q/K using contiguous SD tables and rounded products."""
    _require(query, key, cosine, sine)
    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("rounded rotary expects B=1 BHSD inputs")
    size, width = query.shape[-2:]
    if key.shape[-2:] != (size, width) or query.shape[1] < key.shape[1] or width % 2:
        raise ValueError("incompatible rounded rotary query/key shapes")
    if width > 256 or query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("rounded rotary needs contiguous heads of width <= 256")
    if cosine.shape != (size, width) or sine.shape != cosine.shape:
        raise ValueError("rotary tables must match the query sequence and head width")
    if not cosine.is_contiguous() or not sine.is_contiguous():
        raise ValueError("rotary tables must be contiguous")
    q_out = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    k_out = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    _rope[(size, query.shape[1])](
        query,
        key,
        cosine,
        sine,
        q_out,
        k_out,
        query.stride(1),
        query.stride(2),
        key.stride(1),
        key.stride(2),
        query.shape[1],
        key.shape[1],
        size,
        width,
        triton.next_power_of_2(width),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, k_out
