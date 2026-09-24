"""Inference attention with partitioned KV and FP32 softmax accumulation.

The optional ancestor mask describes only new tokens; cached prefix tokens are
visible to every query. No repeated GQA cache or dense context mask is created.
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
    def _partials(
        Q,
        K,
        V,
        Position,
        Ancestors,
        Partial,
        Maxima,
        Sums,
        sqb: tl.constexpr,
        sqh: tl.constexpr,
        sqm: tl.constexpr,
        skb: tl.constexpr,
        skh: tl.constexpr,
        skn: tl.constexpr,
        svb: tl.constexpr,
        svh: tl.constexpr,
        svn: tl.constexpr,
        NQ: tl.constexpr,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        NS: tl.constexpr,
        CHUNK: tl.constexpr,
        KB: tl.constexpr,
        TREE: tl.constexpr,
        PARTS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        bh = tl.program_id(1)
        split = tl.program_id(2)
        batch, head = bh // HQ, bh % HQ
        kvhead = head // (HQ // HK)
        dim = tl.arange(0, D)
        past = tl.load(Position)
        q = tl.load(Q + batch * sqb + head * sqh + rows[:, None] * sqm + dim[None, :], rows[:, None] < NQ, 0)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        total = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        low_acc = tl.zeros((BM, D), tl.float32)
        end = tl.minimum((split + 1) * CHUNK, tl.minimum(KB, past + NQ))
        for start in range(split * CHUNK, end, BN):
            cols = start + tl.arange(0, BN)
            valid = cols < end
            k = tl.load(
                K + batch * skb + kvhead * skh + cols[:, None] * skn + dim[None, :], valid[:, None], 0
            )
            v = tl.load(
                V + batch * svb + kvhead * svh + cols[:, None] * svn + dim[None, :], valid[:, None], 0
            )
            allowed = (rows[:, None] < NQ) & valid[None, :]
            if TREE:
                local = cols - past
                edges = tl.load(
                    Ancestors + rows[:, None] * NQ + local[None, :],
                    (rows[:, None] < NQ) & (local[None, :] >= 0) & (local[None, :] < NQ),
                    0,
                )
                allowed = allowed & ((cols[None, :] < past) | edges)
            else:
                allowed = allowed & (cols[None, :] <= past + rows[:, None])
            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * (D**-0.5)
            scores = tl.where(allowed, scores, -float("inf"))
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
        acc += low_acc
        base = (bh * NS + split) * NQ + rows
        tl.store(Partial + base[:, None] * D + dim[None, :], acc, rows[:, None] < NQ)
        tl.store(Maxima + base, maximum, rows < NQ)
        tl.store(Sums + base, total, rows < NQ)

    @triton.jit
    def _merge(
        Partial, Maxima, Sums, Output, NQ: tl.constexpr, NS: tl.constexpr, D: tl.constexpr, BS: tl.constexpr
    ):
        row, bh = tl.program_id(0), tl.program_id(1)
        split = tl.arange(0, BS)
        dim = tl.arange(0, D)
        base = (bh * NS + split) * NQ + row
        maximum = tl.load(Maxima + base, split < NS, -float("inf"))
        total = tl.load(Sums + base, split < NS, 0.0)
        global_max = tl.max(maximum, 0)
        weights = tl.exp2((maximum - global_max) * 1.4426950408889634)
        weights = tl.where(split < NS, weights, 0.0)
        numerator = tl.load(Partial + base[:, None] * D + dim[None, :], split[:, None] < NS, 0.0)
        result = tl.sum(numerator * weights[:, None], 0) / tl.sum(total * weights, 0)
        tl.store(Output + (bh * NQ + row) * D + dim, result)


def split_kv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: torch.Tensor,
    key_bucket: int,
    *,
    ancestors: torch.Tensor | None = None,
    probability_parts: int = 3,
) -> torch.Tensor:
    """Attend to a dynamic prefix and causal/new-tree tokens in fixed KV storage.

    CUDA BF16, zero dropout, grouped query heads, and a power-of-two head width
    are required. FP32 probabilities use one to three BF16 tensor-core products;
    two/three components reduce rounding drift, but are not bitwise SDPA math.
    The caller guarantees valid prefix length and complete KV storage. This
    inference-only operation is graph-safe and does not mutate its inputs.
    """
    if triton is None:
        raise RuntimeError("split KV attention requires Triton")
    if query.device.type != "cuda" or query.dtype != torch.bfloat16:
        raise ValueError("split KV attention requires CUDA BF16")
    if torch.is_grad_enabled():
        raise ValueError("split KV attention is inference-only")
    batch, heads, length, width = query.shape
    if probability_parts not in (1, 2, 3) or width < 16 or width & (width - 1):
        raise ValueError("unsupported probability decomposition or head width")
    if (
        key.shape != value.shape
        or key.shape[0] != batch
        or key.shape[-1] != width
        or heads % key.shape[1]
        or not length <= key_bucket <= key.shape[-2]
        or any(t.device != query.device or t.dtype != query.dtype for t in (key, value))
        or any(t.stride(-1) != 1 for t in (query, key, value))
    ):
        raise ValueError("incompatible query and fixed KV storage")
    if position.device != query.device or position.numel() != 1 or position.dtype != torch.long:
        raise ValueError("position must be one device int64 scalar")
    if ancestors is not None and (
        ancestors.shape != (length, length)
        or ancestors.dtype != torch.bool
        or ancestors.device != query.device
        or not ancestors.is_contiguous()
    ):
        raise ValueError("ancestors must be a contiguous device boolean query-square mask")
    block = 16
    splits = min(32, max(1, 512 // (batch * heads * triton.cdiv(length, block))), max(1, key_bucket // 256))
    chunk = triton.cdiv(key_bucket, splits * 64) * 64
    partial = torch.empty((batch * heads, splits, length, width), device=query.device, dtype=torch.float32)
    maxima = torch.empty(partial.shape[:-1], device=query.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    _partials[(triton.cdiv(length, block), batch * heads, splits)](
        query,
        key,
        value,
        position,
        ancestors if ancestors is not None else position,
        partial,
        maxima,
        sums,
        *query.stride()[:3],
        *key.stride()[:3],
        *value.stride()[:3],
        length,
        heads,
        key.shape[1],
        width,
        splits,
        chunk,
        key_bucket,
        ancestors is not None,
        probability_parts,
        block,
        64,
        num_warps=4,
        num_stages=2,
    )
    _merge[(length, batch * heads)](
        partial,
        maxima,
        sums,
        output,
        length,
        splits,
        width,
        triton.next_power_of_2(splits),
        num_warps=4,
    )
    return output
