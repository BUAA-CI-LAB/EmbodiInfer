"""Direct split-KV attention for the two recorded pi05 layouts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_attention(
    Q, PK, PV, SK, SV, Mask, Part, Lse,
    qb: tl.constexpr, qh: tl.constexpr, qs: tl.constexpr,
    pkb: tl.constexpr, pkh: tl.constexpr, pks: tl.constexpr,
    pvb: tl.constexpr, pvh: tl.constexpr, pvs: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, sks: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svs: tl.constexpr,
    mb: tl.constexpr, mn: tl.constexpr,
    P: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    HK: tl.constexpr, D: tl.constexpr,
    SPLITS: tl.constexpr, PART_LENGTH: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    bh = tl.program_id(0)
    batch = bh // H
    head = bh % H
    kv_head = head // (H // HK)
    partition = tl.program_id(2)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    q = tl.load(
        Q + batch * qb + head * qh + rows[:, None] * qs + dims[None, :],
        rows[:, None] < S,
        other=0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)

    for offset in range(PART_LENGTH // BN):
        columns = partition * PART_LENGTH + offset * BN + tl.arange(0, BN)
        suffix = (columns >= P) & (columns < P + S)
        # Clamp inactive offsets too; only the corresponding segment is loaded.
        suffix_rows = tl.maximum(columns - P, 0)
        ks = tl.load(
            SK + batch * skb + kv_head * skh
            + suffix_rows[:, None] * sks + dims[None, :],
            suffix[:, None],
            other=0,
        )
        if P > 0:
            prefix = columns < P
            kp = tl.load(
                PK + batch * pkb + kv_head * pkh
                + columns[:, None] * pks + dims[None, :],
                prefix[:, None],
                other=0,
            )
            k = tl.where(prefix[:, None], kp, ks)
        else:
            # Empty prefix views can alias the suffix; never dereference them.
            k = ks

        valid = columns < P + S
        mask_value = tl.load(
            Mask + batch * mb + columns * mn, valid, other=1.0
        )
        valid = valid & (mask_value == 0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32)
        scores = scores * (0.0625 * 1.4426950408889634)
        scores = tl.where(
            valid[None, :] & (rows[:, None] < S), scores, -float("inf")
        )
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        correction = tl.exp2(maximum - safe_max)
        prob = tl.exp2(scores - safe_max[:, None])

        vs = tl.load(
            SV + batch * svb + kv_head * svh
            + suffix_rows[:, None] * svs + dims[None, :],
            suffix[:, None],
            other=0,
        )
        if P > 0:
            vp = tl.load(
                PV + batch * pvb + kv_head * pvh
                + columns[:, None] * pvs + dims[None, :],
                prefix[:, None],
                other=0,
            )
            v = tl.where(prefix[:, None], vp, vs)
        else:
            v = vs

        # Preserve the starting kernel's BF16 dot operands and FP32 accumulation.
        acc = acc * correction[:, None] + tl.dot(
            prob.to(v.dtype), v, input_precision="ieee", out_dtype=tl.float32
        )
        denominator = denominator * correction + tl.sum(prob, 1)
        maximum = next_max

    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    value = acc / safe_denominator[:, None]
    index = (bh * SPLITS + partition) * S + rows
    tl.store(Part + index[:, None] * D + dims[None, :], value, rows[:, None] < S)
    if SPLITS > 1:
        logsum = tl.where(
            denominator > 0,
            maximum + tl.log2(safe_denominator),
            -float("inf"),
        )
        tl.store(Lse + index, logsum, rows < S)


@triton.jit
def _attention_segment(
    q, rows, maximum, denominator, acc,
    K, V, Mask,
    ks: tl.constexpr, vs: tl.constexpr, mn: tl.constexpr,
    LENGTH: tl.constexpr, MASK_OFFSET: tl.constexpr,
    QUERY_ROWS: tl.constexpr, D: tl.constexpr, BN: tl.constexpr,
):
    dims = tl.arange(0, D)
    for offset in tl.range(0, (LENGTH + BN - 1) // BN, num_stages=2):
        columns = offset * BN + tl.arange(0, BN)
        in_segment = columns < LENGTH
        k = tl.load(
            K + columns[:, None] * ks + dims[None, :],
            in_segment[:, None],
            other=0,
        )
        mask_value = tl.load(
            Mask + (MASK_OFFSET + columns) * mn, in_segment, other=1.0
        )
        valid = in_segment & (mask_value == 0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32)
        scores = scores * (0.0625 * 1.4426950408889634)
        scores = tl.where(
            valid[None, :] & (rows[:, None] < QUERY_ROWS), scores, -float("inf")
        )
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        correction = tl.exp2(maximum - safe_max)
        prob = tl.exp2(scores - safe_max[:, None])

        v = tl.load(
            V + columns[:, None] * vs + dims[None, :],
            in_segment[:, None],
            other=0,
        )
        acc = acc * correction[:, None] + tl.dot(
            prob.to(v.dtype), v, input_precision="ieee", out_dtype=tl.float32
        )
        denominator = denominator * correction + tl.sum(prob, 1)
        maximum = next_max
    return maximum, denominator, acc


@triton.jit
def _segmented_attention(
    Q, PK, PV, SK, SV, Mask, Out,
    qb: tl.constexpr, qh: tl.constexpr, qs: tl.constexpr,
    pkb: tl.constexpr, pkh: tl.constexpr, pks: tl.constexpr,
    pvb: tl.constexpr, pvh: tl.constexpr, pvs: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, sks: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svs: tl.constexpr,
    mb: tl.constexpr, mn: tl.constexpr,
    P: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    HK: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    PACK_QUERY_HEADS: tl.constexpr,
):
    # The frozen prefix ends on a tile boundary; both segments share one softmax.
    tl.static_assert(P % BN == 0)
    bh = tl.program_id(0)
    if PACK_QUERY_HEADS:
        # The host restricts this path to the contiguous, single-batch MQA case.
        tl.static_assert(HK == 1)
        tl.static_assert(qh == S * qs)
        QUERY_ROWS: tl.constexpr = H * S
        batch = 0
        head = 0
        kv_head = 0
    else:
        QUERY_ROWS: tl.constexpr = S
        batch = bh // H
        head = bh % H
        kv_head = head // (H // HK)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    q = tl.load(
        Q + batch * qb + head * qh + rows[:, None] * qs + dims[None, :],
        rows[:, None] < QUERY_ROWS,
        other=0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)

    maximum, denominator, acc = _attention_segment(
        q, rows, maximum, denominator, acc,
        PK + batch * pkb + kv_head * pkh,
        PV + batch * pvb + kv_head * pvh,
        Mask + batch * mb,
        pks, pvs, mn, P, 0, QUERY_ROWS, D, BN,
    )
    maximum, denominator, acc = _attention_segment(
        q, rows, maximum, denominator, acc,
        SK + batch * skb + kv_head * skh,
        SV + batch * svb + kv_head * svh,
        Mask + batch * mb,
        sks, svs, mn, S, P, QUERY_ROWS, D, BN,
    )

    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    value = acc / safe_denominator[:, None]
    if PACK_QUERY_HEADS:
        out_rows = rows
    else:
        out_rows = bh * S + rows
    tl.store(
        Out + out_rows[:, None] * D + dims[None, :],
        value,
        rows[:, None] < QUERY_ROWS,
    )


@triton.jit
def _merge_attention(
    Part, Lse, Out,
    S: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    bh = tl.program_id(1)
    partitions = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    indices = (bh * SPLITS + partitions) * S + row
    logsum = tl.load(Lse + indices)
    maximum = tl.max(logsum, 0)
    # All-empty rows have zero weights, rather than exp2(-inf - -inf).
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp2(logsum - safe_max)
    partial = tl.load(Part + indices[:, None] * D + dims[None, :])
    denominator = tl.sum(weight, 0)
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    value = tl.sum(partial * weight[:, None], 0) / safe_denominator
    tl.store(Out + (bh * S + row) * D + dims, value)


def run(x0, x1, x2, x3, x4, x5):
    """Return contiguous BF16 BHSD attention without modifying any input."""
    batch, heads, size, width = x0.shape
    prefix = x1.shape[2]
    if (batch, heads, width) != (1, 8, 256):
        raise ValueError("Unsupported query shape")
    if (prefix, size) == (0, 576):
        splits, block_m, block_n, part_length = 1, 32, 64, 576
    elif (prefix, size) == (576, 50):
        splits, block_m, block_n, part_length = 1, 16, 64, 640
    else:
        raise ValueError("Unsupported prefix/query lengths")

    pack_query_heads = (
        (prefix, size) == (576, 50)
        and x1.shape[1] == 1
        and x3.shape[1] == 1
        and tuple(x0.stride()) == (102400, 12800, 256, 1)
    )

    # Shape allocation deliberately does not preserve the input query strides.
    output = torch.empty(
        (batch, heads, size, width), device=x0.device, dtype=x0.dtype
    )
    if (prefix, size) in ((0, 576), (576, 50)):
        # Pipeline both layouts; the empty prefix performs zero iterations.
        # Shared K/V lets a query tile cross head boundaries in the short case.
        grid = (
            (1, triton.cdiv(heads * size, block_m), 1)
            if pack_query_heads
            else (batch * heads, triton.cdiv(size, block_m), 1)
        )
        _segmented_attention[grid](
            x0, x1, x2, x3, x4, x5, output,
            *x0.stride()[:3],
            *x1.stride()[:3],
            *x2.stride()[:3],
            *x3.stride()[:3],
            *x4.stride()[:3],
            x5.stride(0), x5.stride(-1),
            prefix, size, heads, x1.shape[1], width,
            block_m, block_n, pack_query_heads,
            num_warps=4,
            num_stages=2,
        )
        return output

    if splits == 1:
        partial = output
        logsum = None
    else:
        partial = torch.empty(
            (batch, heads, splits, size, width),
            device=x0.device,
            dtype=torch.float32,
        )
        logsum = torch.empty(
            (batch, heads, splits, size), device=x0.device, dtype=torch.float32
        )

    # Standard Triton launches use the caller's current CUDA stream.
    _partial_attention[(batch * heads, triton.cdiv(size, block_m), splits)](
        x0, x1, x2, x3, x4, x5, partial, logsum,
        *x0.stride()[:3],
        *x1.stride()[:3],
        *x2.stride()[:3],
        *x3.stride()[:3],
        *x4.stride()[:3],
        x5.stride(0), x5.stride(-1),
        prefix, size, heads, x1.shape[1], width,
        splits, part_length, block_m, block_n,
        num_warps=4,
        num_stages=1,
    )
    if splits > 1:
        _merge_attention[(size, batch * heads)](
            partial, logsum, output, size, width, splits,
            num_warps=4,
            num_stages=1,
        )
    return output
