"""Specialized noncausal split-KV attention for the two captured Orin layouts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_attention(
    Q,
    PK,
    PV,
    SK,
    SV,
    Mask,
    Part,
    Maxima,
    Denominators,
    QH: tl.constexpr,
    QS: tl.constexpr,
    PKS: tl.constexpr,
    PVS: tl.constexpr,
    SKS: tl.constexpr,
    SVS: tl.constexpr,
    MS: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    PART_LENGTH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PACK_QUERY_HEADS: tl.constexpr,
):
    partition = tl.program_id(2)
    dims = tl.arange(0, D)
    lanes = tl.arange(0, BN)
    if PACK_QUERY_HEADS:
        # All eight query heads share K/V. Pack their independent query rows
        # into full tiles, mapping back to each row's head for Q and scratch.
        packed_rows = tl.program_id(0) * BM + tl.arange(0, BM)
        head = packed_rows // S
        rows = packed_rows % S
        active_rows = packed_rows < 8 * S
        q = tl.load(
            Q + head[:, None] * QH + rows[:, None] * QS + dims[None, :],
            active_rows[:, None],
            other=0,
        )
    else:
        head = tl.program_id(0)
        rows = tl.program_id(1) * BM + tl.arange(0, BM)
        active_rows = rows < S
        q = tl.load(
            Q + head * QH + rows[:, None] * QS + dims[None, :],
            active_rows[:, None],
            other=0,
        )

    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    numerator = tl.zeros((BM, D), tl.float32)
    first = partition * PART_LENGTH
    end = tl.minimum(first + PART_LENGTH, P + S)

    for tile_start in range(first, end, BN):
        columns = tile_start + lanes
        # P and partition boundaries are multiples of BN. A tile always reads
        # just one K/V source, including the final, partially occupied tile.
        if P == 0:
            k_ptrs = SK + columns[None, :] * SKS + dims[:, None]
            v_ptrs = SV + columns[:, None] * SVS + dims[None, :]
        else:
            if tile_start < P:
                k_ptrs = PK + columns[None, :] * PKS + dims[:, None]
                v_ptrs = PV + columns[:, None] * PVS + dims[None, :]
            else:
                suffix_columns = columns - P
                k_ptrs = SK + suffix_columns[None, :] * SKS + dims[:, None]
                v_ptrs = SV + suffix_columns[:, None] * SVS + dims[None, :]

        in_bounds = columns < P + S
        k = tl.load(k_ptrs, in_bounds[None, :], other=0)
        padding = tl.load(Mask + columns * MS, in_bounds, other=1.0)
        keep = in_bounds & (padding == 0.0)
        scores = tl.dot(q, k, out_dtype=tl.float32)
        scores = scores * (SCALE * 1.4426950408889634)
        scores = tl.where(
            active_rows[:, None] & keep[None, :], scores, -float("inf")
        )

        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(scores - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, axis=1)

        v = tl.load(v_ptrs, in_bounds[:, None], other=0)
        probability_hi = probability.to(tl.bfloat16)
        probability_lo = (probability - probability_hi.to(tl.float32)).to(tl.bfloat16)
        numerator = numerator * correction[:, None]
        numerator = tl.dot(probability_hi, v, acc=numerator, out_dtype=tl.float32)
        numerator = tl.dot(probability_lo, v, acc=numerator, out_dtype=tl.float32)
        maximum = next_maximum

    state_indices = (head * SPLITS + partition) * S + rows
    if SPLITS == 1:
        safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
        result = numerator / safe_denominator[:, None]
        tl.store(
            Part + state_indices[:, None] * D + dims[None, :],
            result,
            active_rows[:, None],
        )
    else:
        tl.store(
            Part + state_indices[:, None] * D + dims[None, :],
            numerator,
            active_rows[:, None],
        )
        tl.store(Maxima + state_indices, maximum, active_rows)
        tl.store(Denominators + state_indices, denominator, active_rows)


@triton.jit
def _merge_attention(
    Part,
    Maxima,
    Denominators,
    Out,
    S: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    head = tl.program_id(1)
    partitions = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    active_rows = rows < S
    state_indices = (head * SPLITS + partitions[None, :]) * S + rows[:, None]

    maxima = tl.load(Maxima + state_indices, active_rows[:, None], other=-float("inf"))
    denominators = tl.load(Denominators + state_indices, active_rows[:, None], other=0.0)
    row_maximum = tl.max(maxima, axis=1)
    safe_maximum = tl.where(row_maximum == -float("inf"), 0.0, row_maximum)
    weights = tl.exp2(maxima - safe_maximum[:, None])
    denominator = tl.sum(weights * denominators, axis=1)

    partial = tl.load(
        Part + state_indices[:, :, None] * D + dims[None, None, :],
        active_rows[:, None, None],
        other=0.0,
    )
    numerator = tl.sum(partial * weights[:, :, None], axis=1)
    safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
    result = numerator / safe_denominator[:, None]
    tl.store(
        Out + (head * S + rows[:, None]) * D + dims[None, :],
        result,
        active_rows[:, None],
    )


def run(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    x5: torch.Tensor,
) -> torch.Tensor:
    """Return BF16 attention for the captured single-batch, eight-query-head calls.

    Arguments are Q, prefix K/V, suffix K/V, and a shared FP32 padding mask.
    A zero mask element keeps its key; any other value removes it. The scale
    is 0.0625, all KV tensors have one head, and fully masked rows return zero.
    Inputs may have the recorded strided layouts and storage aliases; they are
    read without modification. The result is fresh, contiguous BHSD storage.
    The 50-query call packs rows across query heads that share the same K/V.
    """
    if tuple(x0.shape) not in ((1, 8, 672, 256), (1, 8, 50, 256)):
        raise ValueError("Expected a recorded query shape (1, 8, 672 or 50, 256)")
    size = x0.shape[2]
    prefix = 0 if size == 672 else 672
    if x1.shape != (1, 1, prefix, 256) or x2.shape != x1.shape:
        raise ValueError("Prefix K/V shapes do not match the captured attention call")
    if x3.shape != (1, 1, size, 256) or x4.shape != x3.shape:
        raise ValueError("Suffix K/V shapes do not match the captured attention call")
    if x0.device.type != "cuda":
        raise ValueError("This attention implementation requires CUDA tensors")
    for tensor in (x0, x1, x2, x3, x4):
        if tensor.dtype != torch.bfloat16 or tensor.device != x0.device:
            raise ValueError("Q/K/V must be BF16 tensors on the same CUDA device")
        if tensor.stride(-1) != 1:
            raise ValueError("Q/K/V head dimensions must have unit stride")
    if x5.shape != (1, 1, 1, prefix + size):
        raise ValueError("The padding mask must have shape (1, 1, 1, prefix + suffix)")
    if x5.dtype != torch.float32 or x5.device != x0.device:
        raise ValueError("The padding mask must be FP32 on the Q/K/V device")

    # A 32-row packed tile reuses each K/V tile across more short-call queries.
    # active_rows masks the final tile's 16 padded query rows.
    # Two short-call partitions halve scratch traffic and repeated Q loads.
    # Their 384-key boundary and the 672-key prefix both align to BN=32.
    splits, block_m, part_length = (1, 32, 672) if size == 672 else (2, 32, 384)
    pack_query_heads = size == 50
    output = torch.empty((1, 8, size, 256), device=x0.device, dtype=torch.bfloat16)
    if splits == 1:
        # The SPLITS == 1 specialization never accesses the two state pointers.
        partial = output
        maxima = output
        denominators = output
    else:
        partial = torch.empty((8, splits, size, 256), device=x0.device, dtype=torch.float32)
        maxima = torch.empty((8, splits, size), device=x0.device, dtype=torch.float32)
        denominators = torch.empty((8, splits, size), device=x0.device, dtype=torch.float32)

    partial_grid = (
        (triton.cdiv(8 * size, block_m), 1, splits)
        if pack_query_heads
        else (8, triton.cdiv(size, block_m), splits)
    )
    _partial_attention[partial_grid](
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        partial,
        maxima,
        denominators,
        QH=x0.stride(1),
        QS=x0.stride(2),
        PKS=x1.stride(2),
        PVS=x2.stride(2),
        SKS=x3.stride(2),
        SVS=x4.stride(2),
        MS=x5.stride(3),
        P=prefix,
        S=size,
        D=256,
        SCALE=0.0625,
        SPLITS=splits,
        PART_LENGTH=part_length,
        BM=block_m,
        BN=32,
        PACK_QUERY_HEADS=pack_query_heads,
        num_warps=4,
        # Overlap K/V loads with attention compute across two stages for both calls.
        num_stages=2,
    )
    if splits > 1:
        _merge_attention[(triton.cdiv(size, 4), 8)](
            partial,
            maxima,
            denominators,
            output,
            S=size,
            D=256,
            SPLITS=splits,
            ROWS=4,
            num_warps=4,
            num_stages=1,
        )
    return output
