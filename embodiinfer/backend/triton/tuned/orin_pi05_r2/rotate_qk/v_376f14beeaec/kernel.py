"""Fused rotate-half Q/K RoPE: fused-buffer input layout, BHSD production output layout.

One launch replaces the reference's per-operation kernels and the extra relayout copy into
the production output layout.  ``Q`` and ``K`` are head slices of one token-major fused
buffer, so a program owns ``BLOCK_T`` token rows of a single head plane, reads its row half
and the rotate-half partner from the other register half-tile (no second global read), and
stores both output halves directly in contiguous BHSD order.

This candidate keeps the incumbent's work decomposition, tile shape, masks and arithmetic
and changes exactly one launch property: the program uses two warps instead of four, so each
thread owns 8 contiguous bf16 elements (16 B) of every ``[BLOCK_T, HALF]`` tile instead of 4
(8 B).  Each of the six loads and both stores therefore becomes a single 128-bit access,
halving the memory instructions per byte and doubling the bytes in flight per warp while the
grid stays at 117 programs (0.7 waves on this 14-SM device).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_plane(
    in_row,
    out_row,
    cos_row,
    sin_row,
    t0,
    tokens,
    IN_T: tl.constexpr,
    OUT_T: tl.constexpr,
    COS_T: tl.constexpr,
    SIN_T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HALF: tl.constexpr,
):
    """Rotate-half RoPE for ``BLOCK_T`` token rows of one head plane.

    ``in_row``/``out_row`` already point at the plane's first element; ``rows`` past
    ``tokens`` are masked so the padded tail block neither reads nor writes out of bounds.
    """
    t = tl.arange(0, BLOCK_T)
    d = tl.arange(0, HALF)
    rows = t0 + t
    valid = (rows < tokens)[:, None]
    src = in_row + rows[:, None] * IN_T + d[None, :]
    dst = out_row + rows[:, None] * OUT_T + d[None, :]
    lo = tl.load(src, mask=valid, other=0.0).to(tl.float32)
    hi = tl.load(src + HALF, mask=valid, other=0.0).to(tl.float32)
    cos = cos_row + rows[:, None] * COS_T + d[None, :]
    sin = sin_row + rows[:, None] * SIN_T + d[None, :]
    cos_lo = tl.load(cos, mask=valid, other=0.0).to(tl.float32)
    cos_hi = tl.load(cos + HALF, mask=valid, other=0.0).to(tl.float32)
    sin_lo = tl.load(sin, mask=valid, other=0.0).to(tl.float32)
    sin_hi = tl.load(sin + HALF, mask=valid, other=0.0).to(tl.float32)
    # rotate_half(x) = cat(-x[HALF:], x[:HALF]) and y = x * cos + rotate_half(x) * sin.
    tl.store(dst, lo * cos_lo - hi * sin_lo, mask=valid)
    tl.store(dst + HALF, hi * cos_hi + lo * sin_hi, mask=valid)


@triton.jit
def _rotate_qk_kernel(
    Q,
    K,
    Cos,
    Sin,
    QOut,
    KOut,
    tokens,
    Q_IN_T: tl.constexpr,
    Q_IN_H: tl.constexpr,
    K_IN_T: tl.constexpr,
    K_IN_H: tl.constexpr,
    Q_OUT_T: tl.constexpr,
    Q_OUT_H: tl.constexpr,
    K_OUT_T: tl.constexpr,
    K_OUT_H: tl.constexpr,
    COS_T: tl.constexpr,
    SIN_T: tl.constexpr,
    HQ: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HALF: tl.constexpr,
):
    """Grid ``(cdiv(tokens, BLOCK_T), HQ + HK)``: token block times head plane."""
    t0 = tl.program_id(0) * BLOCK_T
    plane = tl.program_id(1)
    if plane < HQ:
        _rope_plane(
            Q + plane * Q_IN_H,
            QOut + plane * Q_OUT_H,
            Cos,
            Sin,
            t0,
            tokens,
            Q_IN_T,
            Q_OUT_T,
            COS_T,
            SIN_T,
            BLOCK_T,
            HALF,
        )
    else:
        head = plane - HQ
        _rope_plane(
            K + head * K_IN_H,
            KOut + head * K_OUT_H,
            Cos,
            Sin,
            t0,
            tokens,
            K_IN_T,
            K_OUT_T,
            COS_T,
            SIN_T,
            BLOCK_T,
            HALF,
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _supported(query, key, cos, sin) -> bool:
    """Specialization preconditions: batched-BHSD Q/K, ``[B, S, D]`` tables."""
    return (
        query.is_cuda
        and query.dtype == torch.bfloat16
        and key.dtype == query.dtype
        and cos.dtype == query.dtype
        and sin.dtype == query.dtype
        and query.ndim == 4
        and key.ndim == 4
        and cos.ndim == 3
        and sin.ndim == 3
        and query.shape[0] == 1
        and key.shape[0] == 1
        and cos.shape[0] == 1
        and sin.shape[0] == 1
        and query.shape[1] >= 1
        and key.shape[1] >= 1
        and query.shape[2] == key.shape[2] == cos.shape[1] == sin.shape[1]
        and query.shape[3] == key.shape[3] == cos.shape[2] == sin.shape[2]
        and query.shape[3] >= 2
        and query.shape[3] % 2 == 0
        and query.shape[3] <= 1024
        and query.stride(3) == 1
        and key.stride(3) == 1
        and cos.stride(2) == 1
        and sin.stride(2) == 1
    )


def run(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotate-half Q/K RoPE and return contiguous BHSD ``(query, key)`` results."""
    qout = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    kout = torch.empty(key.shape, dtype=query.dtype, device=key.device)
    if not _supported(query, key, cos, sin):
        # Not reachable for the recorded layout; keeps ordinary rotate-half semantics.
        cos_ = cos.unsqueeze(1)
        sin_ = sin.unsqueeze(1)
        qout.copy_(query * cos_ + _rotate_half(query) * sin_)
        kout.copy_(key * cos_ + _rotate_half(key) * sin_)
        return qout, kout
    tokens = query.shape[2]
    if tokens == 0:
        return qout, kout
    block_t = 4
    _rotate_qk_kernel[(triton.cdiv(tokens, block_t), query.shape[1] + key.shape[1])](
        query,
        key,
        cos,
        sin,
        qout,
        kout,
        tokens,
        query.stride(2),
        query.stride(1),
        key.stride(2),
        key.stride(1),
        qout.stride(2),
        qout.stride(1),
        kout.stride(2),
        kout.stride(1),
        cos.stride(1),
        sin.stride(1),
        HQ=query.shape[1],
        BLOCK_T=block_t,
        HALF=query.shape[3] // 2,
        num_warps=2,
        num_stages=1,
    )
    return qout, kout
