"""Single-launch split-KV attention with MQA-shared cached and current K/V.

Non-causal multi-query attention over a cached prefix and a current suffix that
share one K/V head.  The eight query heads are flattened with the query position
into one ``heads * size`` row tile.  K/V is read directly from the separately
strided cached-prefix and current-suffix tensors, so no concatenation or mask copy
is performed.  Online softmax runs in exponent-2 units with an FP32 accumulator.

The K/V and padding-mask loads carry no load-time predicate: their row index is
clamped into the tensor instead, and padding rejection happens only on the scores
(``-inf`` before the row max, hence zero probability and exact zero contribution
from the PV dot).  A masked load whose predicate is produced by another load in
the same loop body prevents Triton from multi-buffering the K/V tiles through
shared memory, which leaves the whole L2 latency of every tile exposed; with the
predicate gone the loop pipelines and the next tile's fetch overlaps the current
tile's tensor-core work.  The arithmetic is unchanged, so the results are
bit-identical to the masked formulation.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_LOG2E = 1.4426950408889634
_SCALE_LOG2 = 0.0625 * _LOG2E


if triton is not None:

    @triton.jit
    def _split_kv_attention(
        Q,
        PK,
        PV,
        SK,
        SV,
        Mask,
        Out,
        qbs,
        qhs,
        qss,
        pkbs,
        pks,
        pvbs,
        pvs,
        skbs,
        sks,
        svbs,
        svs,
        mbs,
        mns,
        obs,
        ohs,
        oss,
        H: tl.constexpr,
        S: tl.constexpr,
        P: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        HEAD_MINOR: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        b = tl.program_id(1)
        rows = row_block * BM + tl.arange(0, BM)
        live = rows < H * S
        if HEAD_MINOR:
            head = rows % H
            pos = rows // H
        else:
            head = rows // S
            pos = rows % S

        dims = tl.arange(0, D)
        q = tl.load(
            Q + b * qbs + head[:, None] * qhs + pos[:, None] * qss + dims[None, :],
            mask=live[:, None],
            other=0.0,
        )

        m_i = tl.full((BM,), -float("inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)

        if P > 0:
            for start in range(0, P, BN):
                cols = start + tl.arange(0, BN)
                # Out-of-range columns duplicate the last real row, so the loads stay
                # unmasked (and therefore prefetchable) while remaining in bounds.
                cached = tl.minimum(cols, P - 1)
                valid = cols < P
                if HAS_MASK:
                    # Affine predicate: pipelineable on its own and still vectorised.
                    bias = tl.load(Mask + b * mbs + cols * mns, mask=valid, other=1.0)
                    valid = valid & (bias == 0.0)
                k = tl.load(PK + b * pkbs + cached[:, None] * pks + dims[None, :])
                scores = tl.dot(q, tl.trans(k)) * SCALE
                scores = tl.where(valid[None, :], scores, -float("inf"))
                next_max = tl.maximum(m_i, tl.max(scores, 1))
                safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
                correction = tl.exp2(m_i - safe_max)
                prob = tl.exp2(scores - safe_max[:, None])
                v = tl.load(PV + b * pvbs + cached[:, None] * pvs + dims[None, :])
                acc = acc * correction[:, None] + tl.dot(prob.to(v.dtype), v)
                l_i = l_i * correction + tl.sum(prob, 1)
                m_i = next_max

        for start in range(0, S, BN):
            cols = start + tl.arange(0, BN)
            current = tl.minimum(cols, S - 1)
            valid = cols < S
            if HAS_MASK:
                bias = tl.load(Mask + b * mbs + (P + cols) * mns, mask=valid, other=1.0)
                valid = valid & (bias == 0.0)
            k = tl.load(SK + b * skbs + current[:, None] * sks + dims[None, :])
            scores = tl.dot(q, tl.trans(k)) * SCALE
            scores = tl.where(valid[None, :], scores, -float("inf"))
            next_max = tl.maximum(m_i, tl.max(scores, 1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp2(m_i - safe_max)
            prob = tl.exp2(scores - safe_max[:, None])
            v = tl.load(SV + b * svbs + current[:, None] * svs + dims[None, :])
            acc = acc * correction[:, None] + tl.dot(prob.to(v.dtype), v)
            l_i = l_i * correction + tl.sum(prob, 1)
            m_i = next_max

        denominator = tl.where(l_i > 0.0, l_i, 1.0)
        output = acc / denominator[:, None]
        tl.store(
            Out + b * obs + head[:, None] * ohs + pos[:, None] * oss + dims[None, :],
            output,
            mask=live[:, None],
        )


def run(x0, x1, x2, x3, x4, x5):
    """Attend to cached (x1/x2) and current (x3/x4) K/V for pi0.5 MQA."""
    if triton is None or x0.device.type != "cuda":
        raise RuntimeError("split-KV attention requires CUDA and Triton")
    query, prefix_key, prefix_value = x0, x1, x2
    suffix_key, suffix_value, mask = x3, x4, x5
    batch, heads, size, width = query.shape
    prefix = prefix_key.shape[2]
    if (
        query.ndim != 4
        or prefix_key.ndim != 4
        or prefix_value.shape != prefix_key.shape
        or suffix_key.ndim != 4
        or suffix_key.shape != suffix_value.shape
        or prefix_key.shape[:2] != (batch, 1)
        or suffix_key.shape[:2] != (batch, 1)
        or suffix_key.shape[2:] != (size, width)
        or prefix_key.shape[3] != width
    ):
        raise ValueError("split-KV attention received incompatible tensor layouts")
    if mask is not None and mask.shape != (batch, 1, 1, prefix + size):
        raise ValueError("split-KV attention received an incompatible padding mask")

    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if size <= 64:
        # 25-program grid, 2-3 resident blocks per SM: keep the shared-memory
        # footprint that preserves it.
        block_m, block_n, num_warps, num_stages = 16, 32, 4, 2
    else:
        # One resident block per SM by registers and shared memory; extra pipeline
        # depth is therefore free and is the only latency hiding this path has.
        block_m, block_n, num_warps, num_stages = 64, 32, 8, 3
    total_rows = heads * size
    _split_kv_attention[(triton.cdiv(total_rows, block_m), batch)](
        query,
        prefix_key,
        prefix_value,
        suffix_key,
        suffix_value,
        query if mask is None else mask,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        prefix_key.stride(0),
        prefix_key.stride(2),
        prefix_value.stride(0),
        prefix_value.stride(2),
        suffix_key.stride(0),
        suffix_key.stride(2),
        suffix_value.stride(0),
        suffix_value.stride(2),
        0 if mask is None else mask.stride(0),
        1 if mask is None else mask.stride(-1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        heads,
        size,
        prefix,
        width,
        _SCALE_LOG2,
        query.stride(1) < query.stride(2),
        mask is not None,
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
