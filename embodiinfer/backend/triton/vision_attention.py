"""Non-causal attention inside packed segments with FP32-faithful probabilities.

For vision windows/images. Probabilities stay FP32 and multiply V through a
``PARTS``-term BF16 decomposition on tensor cores (three terms carry all 24
significand bits), unlike SDPA kernels that round them to BF16 first.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional accelerator
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _segment_attention(
        Q,
        K,
        V,
        Offsets,
        Out,
        stride_t,
        stride_h,
        scale,
        D: tl.constexpr,
        BD: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        PARTS: tl.constexpr,
    ):
        segment, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        start = tl.load(Offsets + segment)
        length = tl.load(Offsets + segment + 1) - start
        rows = block * BM + tl.arange(0, BM)
        dims = tl.arange(0, BD)
        live = rows < length
        in_head = dims[None, :] < D
        base = start * stride_t + head * stride_h
        q = tl.load(Q + base + rows[:, None] * stride_t + dims[None, :], live[:, None] & in_head, 0)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        total = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, BD), tl.float32)
        low_acc = tl.zeros((BM, BD), tl.float32)
        for begin in range(0, tl.where(block * BM < length, length, 0), BN):
            cols = begin + tl.arange(0, BN)
            valid = cols < length
            where = base + cols[:, None] * stride_t + dims[None, :]
            k = tl.load(K + where, valid[:, None] & in_head, 0)
            v = tl.load(V + where, valid[:, None] & in_head, 0)
            # Inductor can lift the Python scale into an FP64 device scalar.
            # Attention accumulators and loop-carried softmax state are FP32.
            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale.to(tl.float32)
            scores = tl.where(live[:, None] & valid[None, :], scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp2((maximum - safe_max) * 1.4426950408889634)
            p = tl.exp2((scores - safe_max[:, None]) * 1.4426950408889634)
            acc = acc * correction[:, None]
            low_acc = low_acc * correction[:, None]
            high = p.to(tl.bfloat16)
            acc += tl.dot(high, v)
            if PARTS >= 2:
                remainder = p - high.to(tl.float32)
                low = remainder.to(tl.bfloat16)
                low_acc += tl.dot(low, v)
            if PARTS >= 3:
                tail = (remainder - low.to(tl.float32)).to(tl.bfloat16)
                low_acc += tl.dot(tail, v)
            total = total * correction + tl.sum(p, 1)
            maximum = next_max
        out = (acc + low_acc) / total[:, None]
        tl.store(
            Out + base + rows[:, None] * stride_t + dims[None, :],
            out.to(Out.dtype.element_ty),
            live[:, None] & in_head,
        )


def segment_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    *,
    scale: float | None = None,
    probability_parts: int = 3,
    block_m: int | None = None,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Attend within each ``offsets`` segment of contiguous BF16 ``[tokens, heads, dim]`` Q/K/V.

    ``max_length`` bounds every segment and must be a Python int so CUDA Graph
    captures never read device offsets on the host.
    """
    if triton is None or not q.is_cuda:
        raise RuntimeError("segment attention requires CUDA and Triton")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("segment attention requires BF16 Q/K/V")
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V must share one [tokens, heads, dim] shape")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous")
    if (
        offsets.ndim != 1
        or offsets.numel() < 2
        or offsets.device != q.device
        or offsets.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("offsets must be a 1-D int32/int64 device tensor of segment boundaries")
    if probability_parts not in (1, 2, 3) or type(max_length) is not int or max_length < 1:
        raise ValueError("unsupported probability decomposition or segment bound")
    tokens, heads, dim = q.shape
    block_m = block_m or min(64, max(16, triton.next_power_of_2(max_length)))
    output = torch.empty_like(q)
    _segment_attention[(offsets.numel() - 1, heads, triton.cdiv(max_length, block_m))](
        q,
        k,
        v,
        offsets,
        output,
        q.stride(0),
        q.stride(1),
        dim**-0.5 if scale is None else scale,
        dim,
        triton.next_power_of_2(dim),
        block_m,
        block_n,
        probability_parts,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
