"""Grouped-query split-KV attention with FP32 softmax accumulation.

Same inputs, masks and FP32 probability decomposition as
``split_attention.split_kv_attention``. One program covers every query head of
a KV group, so each K/V tile is read once per group rather than once per query
head, and partitions divide each row's actual extent instead of the static key
bucket. Results differ from the per-head kernel only in reduction order.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .split_attention import _validate_layout_vectors

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional accelerator
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _accumulate(q, k, v, allowed, maximum, total, acc, low_acc, D: tl.constexpr, PARTS: tl.constexpr):
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
        return next_max, total, acc, low_acc

    @triton.jit
    def _gqa_partials(
        Q,
        K,
        V,
        Position,
        Ancestors,
        Starts,
        Capacities,
        Lengths,
        Partial,
        Maxima,
        Sums,
        Output,
        Blocks,
        QueryStarts,
        QueryExtent,
        sqb,
        sqh,
        sqm,
        skb,
        skh,
        skn,
        svb,
        svh,
        svn,
        sob,
        soh,
        som,
        NQ,
        KB,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        NS: tl.constexpr,
        TREE: tl.constexpr,
        TREE_BATCH: tl.constexpr,
        POOLED: tl.constexpr,
        ROW_POSITIONS: tl.constexpr,
        ROW_LENGTHS: tl.constexpr,
        PARTS: tl.constexpr,
        PACKED: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        G: tl.constexpr = HQ // HK
        split = tl.program_id(2)
        if PACKED:
            # Each program takes one (row, query block) entry; -1 entries are idle.
            entry = tl.load(Blocks + tl.program_id(0))
            batch = tl.maximum(entry, 0) // 65536
            first = (tl.maximum(entry, 0) % 65536) * BM
            kvhead = tl.program_id(1)
            nq = tl.load(QueryExtent)
            token = tl.load(QueryStarts + batch)
        else:
            first = tl.program_id(0) * BM
            group = tl.program_id(1)
            batch = group // HK
            kvhead = group % HK
            nq = NQ
        packed = first + tl.arange(0, BM)
        query = packed // G
        head = kvhead * G + packed % G
        stored = query < nq
        # Padded queries are written as zeros; live rows keep the exact
        # partition and reduction order of the unpadded launch.
        rows = tl.load(Lengths + batch) if ROW_LENGTHS else nq
        if PACKED:
            rows = tl.where(entry >= 0, rows, 0)
        live = query < rows
        dim = tl.arange(0, D)
        past = tl.load(Position + (batch if ROW_POSITIONS else 0))
        cache_row = 0 if POOLED else batch
        cache_base = tl.load(Starts + batch) if POOLED else 0
        cache_capacity = tl.load(Capacities + batch) if POOLED else KB
        extent = tl.minimum(cache_capacity, tl.minimum(KB, past + nq))
        chunk = tl.cdiv(tl.cdiv(extent, NS), BN) * BN
        begin = split * chunk
        end = tl.minimum(begin + chunk, extent)
        if ROW_LENGTHS:
            end = tl.where(first // G < rows, end, begin)
        if PACKED:
            q = tl.load(
                Q + (token + query[:, None]) * sqm + head[:, None] * sqh + dim[None, :], live[:, None], 0
            )
        else:
            q = tl.load(
                Q + batch * sqb + head[:, None] * sqh + query[:, None] * sqm + dim[None, :], live[:, None], 0
            )
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        total = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        low_acc = tl.zeros((BM, D), tl.float32)
        keys = K + cache_row * skb + kvhead * skh
        values = V + cache_row * svb + kvhead * svh
        shared = tl.minimum(end, past)
        for start in range(begin, shared, BN):
            cols = start + tl.arange(0, BN)
            valid = cols < shared
            k = tl.load(keys + (cache_base + cols[:, None]) * skn + dim[None, :], valid[:, None], 0)
            v = tl.load(values + (cache_base + cols[:, None]) * svn + dim[None, :], valid[:, None], 0)
            maximum, total, acc, low_acc = _accumulate(
                q, k, v, live[:, None] & valid[None, :], maximum, total, acc, low_acc, D, PARTS
            )
        for start in range(tl.maximum(begin, past), end, BN):
            cols = start + tl.arange(0, BN)
            valid = cols < end
            k = tl.load(keys + (cache_base + cols[:, None]) * skn + dim[None, :], valid[:, None], 0)
            v = tl.load(values + (cache_base + cols[:, None]) * svn + dim[None, :], valid[:, None], 0)
            local = cols - past
            allowed = live[:, None] & valid[None, :]
            if TREE:
                edges = tl.load(
                    Ancestors + (batch * NQ * NQ if TREE_BATCH else 0) + query[:, None] * NQ + local[None, :],
                    allowed,
                    0,
                )
                allowed = allowed & edges
            else:
                allowed = allowed & (local[None, :] <= query[:, None])
            maximum, total, acc, low_acc = _accumulate(
                q, k, v, allowed, maximum, total, acc, low_acc, D, PARTS
            )
        acc += low_acc
        if NS == 1:
            result = acc / total[:, None]
            if ROW_LENGTHS:
                result = tl.where(live[:, None], result, 0.0)
            if PACKED:
                tl.store(
                    Output + (token + query[:, None]) * som + head[:, None] * soh + dim[None, :],
                    result.to(Output.dtype.element_ty),
                    live[:, None],
                )
            else:
                tl.store(
                    Output + batch * sob + head[:, None] * soh + query[:, None] * som + dim[None, :],
                    result.to(Output.dtype.element_ty),
                    stored[:, None],
                )
        else:
            if ROW_LENGTHS:
                maximum = tl.where(live, maximum, 0.0)
                total = tl.where(live, total, 1.0)
            if PACKED:
                # A row's last block must not spill into the next row's tokens.
                base = ((token + query) * HQ + head) * NS + split
                stored = live
            else:
                base = ((batch * HQ + head) * NS + split) * NQ + query
            tl.store(Partial + base[:, None] * D + dim[None, :], acc, stored[:, None])
            tl.store(Maxima + base, maximum, stored)
            tl.store(Sums + base, total, stored)

    @triton.jit
    def _gqa_merge(
        Partial,
        Maxima,
        Sums,
        Output,
        sob,
        soh,
        som,
        NQ,
        HQ: tl.constexpr,
        NS: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        row, bh = tl.program_id(0), tl.program_id(1)
        split = tl.arange(0, BS)
        dim = tl.arange(0, D)
        base = (row * HQ + bh) * NS + split if PACKED else (bh * NS + split) * NQ + row
        maximum = tl.load(Maxima + base, split < NS, -float("inf"))
        total = tl.load(Sums + base, split < NS, 0.0)
        global_max = tl.max(maximum, 0)
        weights = tl.exp2((maximum - global_max) * 1.4426950408889634)
        weights = tl.where(split < NS, weights, 0.0)
        numerator = tl.load(Partial + base[:, None] * D + dim[None, :], split[:, None] < NS, 0.0)
        result = tl.sum(numerator * weights[:, None], 0) / tl.sum(total * weights, 0)
        if PACKED:
            tl.store(Output + row * som + bh * soh + dim, result.to(Output.dtype.element_ty))
        else:
            tl.store(
                Output + (bh // HQ) * sob + (bh % HQ) * soh + row * som + dim,
                result.to(Output.dtype.element_ty),
            )


def gqa_launch_config(rows: int, key_bucket: int) -> tuple[int, int, int, int, int]:
    """(packed rows per program, key block, warps, stages, partitions) for a graph shape.

    ``rows`` is queries x group size. Tuned on RTX 4090 for 16/2 heads of width
    128; only static dimensions may enter, so a graph never changes its plan.
    """
    if rows <= 160:
        block_m, block_n, warps, stages, splits = 16, 32, 4, 2, 16
    elif rows <= 512:
        block_m, block_n, warps, stages, splits = 32, 64, 4, 2, 16
    else:
        block_m, block_n, warps, stages, splits = 128, 32, 8, 3, 4
    return block_m, block_n, warps, stages, max(1, min(splits, key_bucket // 512))


# Packed launches reuse the plan of padded launches with at least this many rows.
PACKED_MIN_ROWS = 513
PACKED_BLOCK_M = gqa_launch_config(PACKED_MIN_ROWS, 512)[0]


def packed_query_blocks(lengths: Sequence[int], group: int, capacity: int) -> list[int]:
    """Program table of a packed launch.

    One entry, row * 65536 + block, per block of ``PACKED_BLOCK_M // group``
    queries of each row, then -1 (idle) up to ``capacity`` entries.
    """
    per_block = PACKED_BLOCK_M // group
    blocks = [
        row * 65536 + block for row, size in enumerate(lengths) for block in range(-(-size // per_block))
    ]
    if len(blocks) > capacity:
        raise ValueError("packed queries need more programs than the table holds")
    return blocks + [-1] * (capacity - len(blocks))


def gqa_split_kv_attention(
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
    query_lengths: torch.Tensor | None = None,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    splits: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Same contract as ``split_attention.split_kv_attention``.

    The result is a BHQD view of BQHD storage, so the caller's transpose and
    head merge need no copy. With ``query_lengths`` (one count per row),
    queries at or past a row's count return zeros and skip work in whole
    blocks; every other output is bitwise unchanged.

    ``packed`` = (blocks, starts, extent) lays the rows' queries back to back
    in one [1, heads, tokens, width] query: ``blocks`` is the program table of
    ``packed_query_blocks``, ``starts`` each row's first token and ``extent``
    one device integer, the padded query length. With pooled KV and
    ``query_lengths``, every live query then equals the padded launch of that
    length bit for bit, provided length x group >= ``PACKED_MIN_ROWS``; other
    tokens are unspecified.
    """
    if triton is None:
        raise RuntimeError("split KV attention requires Triton")
    if query.device.type != "cuda" or query.dtype != torch.bfloat16:
        raise ValueError("split KV attention requires CUDA BF16")
    if torch.is_grad_enabled():
        raise ValueError("split KV attention is inference-only")
    batch, heads, length, width = query.shape
    pooled = cache_starts is not None
    if packed is not None:
        blocks, starts, extent = packed
        if (
            batch != 1
            or not pooled
            or query_lengths is None
            or ancestors is not None
            or blocks.ndim != 1
            or blocks.dtype != torch.long
            or extent.numel() != 1
            or starts.shape != query_lengths.shape
            or any(t.device != query.device or not t.is_contiguous() for t in packed)
        ):
            raise ValueError("packed queries need one pooled causal batch with per-row lengths")
        batch = starts.shape[0]
    _validate_layout_vectors(batch, query.device, cache_starts, cache_capacities, None)
    if probability_parts not in (1, 2, 3) or width < 16 or width & (width - 1):
        raise ValueError("unsupported probability decomposition or head width")
    if (
        key.shape != value.shape
        or key.shape[0] != (1 if pooled else batch)
        or key.shape[-1] != width
        or heads % key.shape[1]
        or key_bucket > key.shape[-2]
        or (packed is None and length > key_bucket)
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
    if query_lengths is not None and (
        query_lengths.device != query.device
        or query_lengths.shape != (batch,)
        or query_lengths.dtype not in (torch.int32, torch.int64)
        or not query_lengths.is_contiguous()
    ):
        raise ValueError("query_lengths must be one contiguous device integer per batch row")
    kv_heads = key.shape[1]
    rows = length * (heads // kv_heads)
    plan = gqa_launch_config(rows if packed is None else PACKED_MIN_ROWS, key_bucket)
    block_m = block_m or plan[0]
    block_n = block_n or plan[1]
    num_warps = num_warps or plan[2]
    num_stages = num_stages or plan[3]
    splits = splits or plan[4]
    if packed is not None and block_m != PACKED_BLOCK_M:
        raise ValueError("packed program tables assume PACKED_BLOCK_M rows per program")
    if packed is None:
        grid = (triton.cdiv(rows, block_m), batch * kv_heads, splits)
        output = torch.empty((batch, length, heads, width), device=query.device, dtype=query.dtype)
        partial_shape = (batch * heads, splits, length, width)
    else:
        grid = (packed[0].numel(), kv_heads, splits)
        output = torch.empty((1, length, heads, width), device=query.device, dtype=query.dtype)
        partial_shape = (length, heads, splits, width)
    if splits > 1:
        partial = query.new_empty(partial_shape, dtype=torch.float32)
        maxima = torch.empty(partial.shape[:-1], device=query.device, dtype=torch.float32)
        sums = torch.empty_like(maxima)
    else:
        partial = maxima = sums = output
    _gqa_partials[grid](
        query,
        key,
        value,
        position,
        ancestors if ancestors is not None else position,
        cache_starts if pooled else position,
        cache_capacities if pooled else position,
        position if query_lengths is None else query_lengths,
        partial,
        maxima,
        sums,
        output,
        *((position,) * 3 if packed is None else packed),
        *query.stride()[:3],
        *key.stride()[:3],
        *value.stride()[:3],
        output.stride(0),
        output.stride(2),
        output.stride(1),
        length,
        key_bucket,
        heads,
        kv_heads,
        width,
        splits,
        ancestors is not None,
        ancestors is not None and ancestors.ndim == 3,
        pooled,
        position.numel() != 1,
        query_lengths is not None,
        probability_parts,
        packed is not None,
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if splits > 1:
        _gqa_merge[(length, heads if packed is not None else batch * heads)](
            partial,
            maxima,
            sums,
            output,
            output.stride(0),
            output.stride(2),
            output.stride(1),
            length,
            heads,
            splits,
            width,
            triton.next_power_of_2(splits),
            packed is not None,
            num_warps=4,
        )
    return output.transpose(1, 2)
