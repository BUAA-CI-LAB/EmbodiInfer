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
    def _write_kv(
        K,
        V,
        KC,
        VC,
        Position,
        Starts,
        Capacities,
        WriteLengths,
        skb: tl.constexpr,
        skh: tl.constexpr,
        skq: tl.constexpr,
        svb: tl.constexpr,
        svh: tl.constexpr,
        svq: tl.constexpr,
        scb: tl.constexpr,
        sch: tl.constexpr,
        sct: tl.constexpr,
        H: tl.constexpr,
        Q: tl.constexpr,
        D: tl.constexpr,
        CAP: tl.constexpr,
        N: tl.constexpr,
        POOLED: tl.constexpr,
        LIMITED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        dim = index % D
        query = index // D % Q
        head = index // (D * Q) % H
        batch = index // (D * Q * H)
        past = tl.load(Position + batch, index < N, 0)
        base = tl.load(Starts + batch, index < N, 0) if POOLED else 0
        capacity = tl.load(Capacities + batch, index < N, 0) if POOLED else CAP
        write_length = tl.load(WriteLengths + batch, index < N, 0) if LIMITED else Q
        target = past + query
        source_key = batch * skb + head * skh + query * skq + dim
        source_value = batch * svb + head * svh + query * svq + dim
        key = tl.load(K + source_key, index < N, 0)
        value = tl.load(V + source_value, index < N, 0)
        destination = (0 if POOLED else batch * scb) + head * sch + (base + target) * sct + dim
        valid = (index < N) & (target < capacity) & (query < write_length)
        tl.store(KC + destination, key, valid)
        tl.store(VC + destination, value, valid)

    @triton.jit
    def _partials(
        Q,
        K,
        V,
        Position,
        Ancestors,
        Starts,
        Capacities,
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
        TREE_BATCH: tl.constexpr,
        POOLED: tl.constexpr,
        ROW_POSITIONS: tl.constexpr,
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
        past = tl.load(Position + (batch if ROW_POSITIONS else 0))
        cache_row = 0 if POOLED else batch
        cache_base = tl.load(Starts + batch) if POOLED else 0
        cache_capacity = tl.load(Capacities + batch) if POOLED else KB
        q = tl.load(Q + batch * sqb + head * sqh + rows[:, None] * sqm + dim[None, :], rows[:, None] < NQ, 0)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        total = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        low_acc = tl.zeros((BM, D), tl.float32)
        end = tl.minimum((split + 1) * CHUNK, tl.minimum(cache_capacity, tl.minimum(KB, past + NQ)))
        for start in range(split * CHUNK, end, BN):
            cols = start + tl.arange(0, BN)
            valid = cols < end
            k = tl.load(
                K + cache_row * skb + kvhead * skh + (cache_base + cols[:, None]) * skn + dim[None, :],
                valid[:, None],
                0,
            )
            v = tl.load(
                V + cache_row * svb + kvhead * svh + (cache_base + cols[:, None]) * svn + dim[None, :],
                valid[:, None],
                0,
            )
            allowed = (rows[:, None] < NQ) & valid[None, :]
            if TREE:
                local = cols - past
                edges = tl.load(
                    Ancestors + (batch * NQ * NQ if TREE_BATCH else 0) + rows[:, None] * NQ + local[None, :],
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


def write_batched_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    cache_starts: torch.Tensor | None = None,
    cache_capacities: torch.Tensor | None = None,
    write_lengths: torch.Tensor | None = None,
) -> None:
    """Write BHQD chunks at per-row offsets, ignoring padded slots beyond storage.

    Real query lengths must be validated by the caller. Only padding is allowed
    to overrun capacity; this graph-safe inference operation does not clamp an
    out-of-range write onto a valid cached token.
    """
    if triton is None or key.device.type != "cuda" or torch.is_grad_enabled():
        raise ValueError("batched KV writes require CUDA Triton inference mode")
    if (
        key.ndim != 4
        or key_cache.ndim != 4
        or key.shape != value.shape
        or key_cache.shape != value_cache.shape
    ):
        raise ValueError("batched KV writes require matching BHQD chunks and BHTD caches")
    batch, heads, query, width = key.shape
    pooled = cache_starts is not None
    _validate_layout_vectors(batch, key.device, cache_starts, cache_capacities, write_lengths)
    if (
        key_cache.shape[:2] != (1 if pooled else batch, heads)
        or key_cache.shape[-1] != width
        or positions.shape != (batch,)
        or positions.dtype != torch.long
        or not positions.is_contiguous()
        or key.stride(-1) != 1
        or positions.device != key.device
        or key_cache.stride() != value_cache.stride()
        or any(
            x.device != key.device or x.dtype != key.dtype or x.stride(-1) != 1
            for x in (value, key_cache, value_cache)
        )
    ):
        raise ValueError("incompatible batched KV write layout")
    _write_kv[(triton.cdiv(key.numel(), 256),)](
        key,
        value,
        key_cache,
        value_cache,
        positions,
        cache_starts if pooled else positions,
        cache_capacities if pooled else positions,
        write_lengths if write_lengths is not None else positions,
        *key.stride()[:3],
        *value.stride()[:3],
        *key_cache.stride()[:3],
        heads,
        query,
        width,
        key_cache.shape[-2],
        key.numel(),
        pooled,
        write_lengths is not None,
        256,
    )


def split_kv_partitions(batch_size: int, num_heads: int, query_tokens: int, key_bucket: int) -> int:
    """Return the kernel's partition count for positive execution dimensions.

    A single partition reads only the dynamic prefix plus query, so increasing
    its key bound cannot change partition boundaries or reduction order. Graph
    callers may reuse that case across sufficient context bounds.
    """
    return min(
        32,
        max(1, 512 // (batch_size * num_heads * ((query_tokens + 15) // 16))),
        max(1, key_bucket // 256),
    )


def split_kv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: torch.Tensor,
    key_bucket: int,
    *,
    ancestors: torch.Tensor | None = None,
    probability_parts: int = 3,
    cache_starts: torch.Tensor | None = None,
    cache_capacities: torch.Tensor | None = None,
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
    pooled = cache_starts is not None
    _validate_layout_vectors(batch, query.device, cache_starts, cache_capacities, None)
    if probability_parts not in (1, 2, 3) or width < 16 or width & (width - 1):
        raise ValueError("unsupported probability decomposition or head width")
    if (
        key.shape != value.shape
        or key.shape[0] != (1 if pooled else batch)
        or key.shape[-1] != width
        or heads % key.shape[1]
        or not length <= key_bucket <= key.shape[-2]
        or any(t.device != query.device or t.dtype != query.dtype for t in (key, value))
        or any(t.stride(-1) != 1 for t in (query, key, value))
    ):
        raise ValueError("incompatible query and fixed KV storage")
    if (
        position.device != query.device
        or position.numel() not in (1, batch)
        or position.dtype != torch.long
        or not position.is_contiguous()
    ):
        raise ValueError("position must be one device int64 scalar or one offset per batch row")
    if ancestors is not None and (
        ancestors.shape not in ((length, length), (batch, length, length))
        or ancestors.dtype != torch.bool
        or ancestors.device != query.device
        or not ancestors.is_contiguous()
    ):
        raise ValueError("ancestors must be a contiguous device boolean query-square mask")
    block = 16
    splits = split_kv_partitions(batch, heads, length, key_bucket)
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
        cache_starts if pooled else position,
        cache_capacities if pooled else position,
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
        ancestors is not None and ancestors.ndim == 3,
        pooled,
        position.numel() != 1,
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


def _validate_layout_vectors(batch, device, starts, capacities, write_lengths):
    if (starts is None) != (capacities is None):
        raise ValueError("pooled KV requires both start and capacity vectors")
    for value in (starts, capacities, write_lengths):
        if value is not None and (
            value.shape != (batch,)
            or value.dtype != torch.long
            or value.device != device
            or not value.is_contiguous()
        ):
            raise ValueError("KV layout vectors must be contiguous device int64 batch vectors")
