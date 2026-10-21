"""Triton non-causal attention over separate cached and current K/V blocks.

Uses online softmax with parallel KV partitions and a stable reduction.
The current block contains one KV token per query; the cached
block may be empty. A shared key-padding mask avoids materializing concatenated
K/V or repeated KV heads. No model, checkpoint or session state is owned here.
"""

from __future__ import annotations

import torch

from ..split_kv_plan import plan_split_kv
from .capability import triton_capability

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _partial_attention(
        Q,
        PK,
        PV,
        SK,
        SV,
        Mask,
        Part,
        Lse,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qs: tl.constexpr,
        pkb: tl.constexpr,
        pkh: tl.constexpr,
        pks: tl.constexpr,
        pvb: tl.constexpr,
        pvh: tl.constexpr,
        pvs: tl.constexpr,
        skb: tl.constexpr,
        skh: tl.constexpr,
        sks: tl.constexpr,
        svb: tl.constexpr,
        svh: tl.constexpr,
        svs: tl.constexpr,
        mb: tl.constexpr,
        mn: tl.constexpr,
        P: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        MASKED: tl.constexpr,
        SPLITS: tl.constexpr,
        PART_LENGTH: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b, head = bh // H, bh % H
        kv_head = head // (H // HK)
        partition = tl.program_id(2)
        rows = tl.program_id(1) * BM + tl.arange(0, BM)
        dims = tl.arange(0, D)
        q = tl.load(Q + b * qb + head * qh + rows[:, None] * qs + dims[None, :], rows[:, None] < S, other=0)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        for offset in range(PART_LENGTH // BN):
            columns = partition * PART_LENGTH + offset * BN + tl.arange(0, BN)
            prefix = columns < P
            suffix = (columns >= P) & (columns < P + S)
            kp = tl.load(
                PK + b * pkb + kv_head * pkh + columns[:, None] * pks + dims[None, :],
                prefix[:, None],
                other=0,
            )
            ks = tl.load(
                SK + b * skb + kv_head * skh + (columns[:, None] - P) * sks + dims[None, :],
                suffix[:, None],
                other=0,
            )
            k = tl.where(prefix[:, None], kp, ks)
            valid = columns < P + S
            if MASKED:
                bias = tl.load(Mask + b * mb + columns * mn, valid, other=-float("inf"))
                valid = valid & (bias == 0)
            scores = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
            scores = tl.where(valid[None, :] & (rows[:, None] < S), scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # An entirely padded partition contributes zero mass, never NaNs.
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp2(maximum - safe_max)
            prob = tl.exp2(scores - safe_max[:, None])
            vp = tl.load(
                PV + b * pvb + kv_head * pvh + columns[:, None] * pvs + dims[None, :],
                prefix[:, None],
                other=0,
            )
            vs = tl.load(
                SV + b * svb + kv_head * svh + (columns[:, None] - P) * svs + dims[None, :],
                suffix[:, None],
                other=0,
            )
            v = tl.where(prefix[:, None], vp, vs)
            acc = acc * correction[:, None] + tl.dot(prob.to(v.dtype), v)
            denominator = denominator * correction + tl.sum(prob, 1)
            maximum = next_max
        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        output = acc / safe_denominator[:, None]
        logsum = tl.where(denominator > 0, maximum + tl.log2(safe_denominator), -float("inf"))
        index = (bh * SPLITS + partition) * S + rows
        tl.store(Part + index[:, None] * D + dims[None, :], output, rows[:, None] < S)
        if SPLITS > 1:
            tl.store(Lse + index, logsum, rows < S)

    @triton.jit
    def _merge_attention(Part, Lse, Out, S: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
        row, bh = tl.program_id(0), tl.program_id(1)
        partitions = tl.arange(0, SPLITS)
        dims = tl.arange(0, D)
        indices = (bh * SPLITS + partitions) * S + row
        logsum = tl.load(Lse + indices)
        maximum = tl.max(logsum, 0)
        weight = tl.exp2(logsum - maximum)
        partial = tl.load(Part + indices[:, None] * D + dims[None, :])
        value = tl.sum(partial * weight[:, None], 0) / tl.sum(weight, 0)
        tl.store(Out + (bh * S + row) * D + dims, value)


def split_kv_attention(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    mask: torch.Tensor | None,
    scaling: float,
    *,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Attend to two BHSD K/V blocks without concatenation or KV-head repetition.

    The current block has one KV token per query. ``mask`` is a shared floating
    padding mask [batch, 1, 1, cached + current]: zero keeps a key, nonzero masks
    it. Per-query masks, arbitrary additive biases, causality and dropout are
    outside this kernel's contract. Outputs for entirely masked queries are undefined.
    ``num_splits=None`` preserves the existing launch geometry. Explicit split
    counts are profiling candidates and can change floating-point reduction order.
    """
    if triton is None or query.device.type != "cuda":
        raise RuntimeError("split-KV attention requires CUDA and Triton")
    if any(tensor.ndim != 4 for tensor in (query, prefix_key, prefix_value, suffix_key, suffix_value)):
        raise ValueError("split-KV attention tensors must have BHSD layout")
    batch, heads, size, width = query.shape
    if min(batch, heads, size, prefix_key.shape[1]) < 1:
        raise ValueError("split-KV attention requires nonempty batch, heads and queries")
    if query.dtype not in (torch.bfloat16, torch.float16) or width not in (64, 128, 256):
        raise ValueError("split-KV attention requires FP16/BF16 and head width 64/128/256")
    if prefix_key.shape != prefix_value.shape or suffix_key.shape != suffix_value.shape:
        raise ValueError("split-KV attention K and V shapes must match")
    if prefix_key.shape[:2] != suffix_key.shape[:2] or prefix_key.shape[0] != batch:
        raise ValueError("cached/current KV batch and KV-head counts must agree")
    if suffix_key.shape[2:] != (size, width) or prefix_key.shape[-1] != width:
        raise ValueError("split-KV attention expects one current KV token per query")
    if heads % prefix_key.shape[1]:
        raise ValueError("query heads must be a multiple of KV heads")
    for tensor in (query, prefix_key, prefix_value, suffix_key, suffix_value):
        if tensor.device != query.device or tensor.dtype != query.dtype or tensor.stride(-1) != 1:
            raise ValueError(
                "split-KV attention tensors must share device/dtype and contiguous head dimensions"
            )
    prefix = prefix_key.shape[2]
    if mask is not None and (mask.shape != (batch, 1, 1, prefix + size) or mask.device != query.device):
        raise ValueError("split-KV padding mask must be [batch, 1, 1, cached + current]")
    if mask is not None and not mask.is_floating_point():
        raise TypeError("split-KV padding mask must use floating zero for valid keys")
    if num_splits is None:
        # Keep the legacy launch setup on the default hot path.
        splits, block_m, block_n = (4, 16, 64) if size <= 64 else (1, 32, 64)
        part_length = triton.cdiv(prefix + size, splits * block_n) * block_n
    else:
        plan = plan_split_kv(size, prefix, num_splits=num_splits)
        splits, block_m, block_n = plan.num_splits, plan.block_m, plan.block_n
        part_length = plan.partition_length
    partial = torch.empty(
        (batch, heads, splits, size, width),
        device=query.device,
        dtype=query.dtype if splits == 1 else torch.float32,
    )
    logsum = torch.empty((batch, heads, splits, size), device=query.device, dtype=torch.float32)
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    _partial_attention[(batch * heads, triton.cdiv(size, block_m), splits)](
        query,
        prefix_key,
        prefix_value,
        suffix_key,
        suffix_value,
        mask,
        partial,
        logsum,
        *query.stride()[:3],
        *prefix_key.stride()[:3],
        *prefix_value.stride()[:3],
        *suffix_key.stride()[:3],
        *suffix_value.stride()[:3],
        0 if mask is None else mask.stride(0),
        0 if mask is None else mask.stride(-1),
        prefix,
        size,
        heads,
        prefix_key.shape[1],
        width,
        scaling,
        mask is not None,
        splits,
        part_length,
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    if splits == 1:
        return partial.view(query.shape)
    _merge_attention[(size, batch * heads)](partial, logsum, output, size, width, splits, num_warps=4)
    return output


class TritonSplitKVAttention:
    """Non-causal self-attention with optional cached K/V and a shared padding mask.

    Supports FP16/BF16, head widths 64/128/256 and GQA/MQA. Dropout, causal masks
    and query-dependent masks are unsupported. Padding masks use zero for valid
    keys and a nonzero value for padding; they do not encode arbitrary biases.
    """

    name = "triton_split_kv"

    @staticmethod
    def capability() -> tuple[bool, str | None]:
        """Probe the optional Triton CUDA runtime without launching kernels."""
        capability = triton_capability()
        return capability.available, capability.reason

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Attend to one self-attention block with an optional [B, 1, 1, S] padding mask."""
        return self.attend_split(q, k[:, :, :0], v[:, :, :0], k, v, attn_mask, scaling, dropout_p)

    def attend_split(
        self,
        q: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
        *,
        num_splits: int | None = None,
    ) -> torch.Tensor:
        """Attend to cached/current K/V blocks without copying or concatenating them."""
        if dropout_p != 0.0:
            raise ValueError("triton_split_kv does not support dropout")
        scaling = q.shape[-1] ** -0.5 if scaling is None else scaling
        return split_kv_attention(
            q, cached_k, cached_v, k, v, key_padding_mask, scaling, num_splits=num_splits
        )

    def attend_split_packed(
        self,
        q: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scaling: float | None = None,
        num_splits: int = 16,
    ) -> torch.Tensor:
        """Pack GQA queries per KV head and return [B, Q, H*D] directly."""
        batch, heads, size, width = q.shape
        kv_heads = cached_k.shape[1]
        prefix = cached_k.shape[2]
        scale = width**-0.5 if scaling is None else scaling
        group = heads // kv_heads
        block_m = 16 if size <= 16 else 32
        block_n = 64
        part_length = triton.cdiv(prefix + size, num_splits * block_n) * block_n
        partial = torch.empty((batch, heads, num_splits, size, width), device=q.device, dtype=torch.float32)
        logsum = torch.empty((batch, heads, num_splits, size), device=q.device, dtype=torch.float32)
        _partial_packed[(batch * kv_heads, triton.cdiv(group * size, block_m), num_splits)](
            q,
            cached_k,
            cached_v,
            k,
            v,
            partial,
            logsum,
            *q.stride()[:3],
            *cached_k.stride()[:3],
            *cached_v.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            prefix,
            size,
            heads,
            kv_heads,
            width,
            scale,
            num_splits,
            part_length,
            block_m,
            block_n,
            num_warps=4,
            num_stages=1,
        )
        if getattr(self, "fp8_scale", None) is not None:
            from types import SimpleNamespace

            out_fp8 = torch.empty((batch * size, heads * width), device=q.device, dtype=torch.float8_e4m3fn)
            _merge_bshd_fp8[(size, batch * heads)](
                partial, logsum, out_fp8, 448.0 / 8.0, size, heads, width, num_splits, num_warps=4
            )
            if self.fp8_scale.device != q.device:
                self.fp8_scale = self.fp8_scale.to(q.device)
            return SimpleNamespace(output=out_fp8, scale=self.fp8_scale, residual=q[:, 0, :, :1])
        output = torch.empty((batch, size, heads * width), device=q.device, dtype=q.dtype)
        _merge_bshd[(size, batch * heads)](
            partial, logsum, output, size, heads, width, num_splits, num_warps=4
        )
        return output


if triton is not None:

    @triton.jit
    def _partial_packed(
        Q,
        PK,
        PV,
        SK,
        SV,
        Part,
        Lse,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qs: tl.constexpr,
        pkb: tl.constexpr,
        pkh: tl.constexpr,
        pks: tl.constexpr,
        pvb: tl.constexpr,
        pvh: tl.constexpr,
        pvs: tl.constexpr,
        skb: tl.constexpr,
        skh: tl.constexpr,
        sks: tl.constexpr,
        svb: tl.constexpr,
        svh: tl.constexpr,
        svs: tl.constexpr,
        P_: tl.constexpr,
        S_: tl.constexpr,
        H_: tl.constexpr,
        HK_: tl.constexpr,
        D_: tl.constexpr,
        SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        PART_LENGTH: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        bk = tl.program_id(0)
        b, kv_head = bk // HK_, bk % HK_
        G: tl.constexpr = H_ // HK_
        partition = tl.program_id(2)
        packed = tl.program_id(1) * BM + tl.arange(0, BM)
        live = packed < G * S_
        head = kv_head * G + packed // S_
        rows = packed % S_
        dims = tl.arange(0, D_)
        q = tl.load(
            Q + b * qb + head[:, None] * qh + rows[:, None] * qs + dims[None, :], live[:, None], other=0
        )
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denom = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D_), tl.float32)
        for offset in range(PART_LENGTH // BN):
            columns = partition * PART_LENGTH + offset * BN + tl.arange(0, BN)
            prefix = columns < P_
            suffix = (columns >= P_) & (columns < P_ + S_)
            kp = tl.load(
                PK + b * pkb + kv_head * pkh + columns[:, None] * pks + dims[None, :],
                prefix[:, None],
                other=0,
            )
            ks = tl.load(
                SK + b * skb + kv_head * skh + (columns[:, None] - P_) * sks + dims[None, :],
                suffix[:, None],
                other=0,
            )
            k = tl.where(prefix[:, None], kp, ks)
            valid = columns < P_ + S_
            scores = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
            scores = tl.where(valid[None, :] & live[:, None], scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp2(maximum - safe_max)
            prob = tl.exp2(scores - safe_max[:, None])
            vp = tl.load(
                PV + b * pvb + kv_head * pvh + columns[:, None] * pvs + dims[None, :],
                prefix[:, None],
                other=0,
            )
            vs = tl.load(
                SV + b * svb + kv_head * svh + (columns[:, None] - P_) * svs + dims[None, :],
                suffix[:, None],
                other=0,
            )
            v = tl.where(prefix[:, None], vp, vs)
            acc = acc * correction[:, None] + tl.dot(prob.to(v.dtype), v)
            denom = denom * correction + tl.sum(prob, 1)
            maximum = next_max
        safe_denom = tl.where(denom > 0, denom, 1.0)
        output = acc / safe_denom[:, None]
        logsum = tl.where(denom > 0, maximum + tl.log2(safe_denom), -float("inf"))
        index = ((b * H_ + head) * SPLITS + partition) * S_ + rows
        tl.store(Part + index[:, None] * D_ + dims[None, :], output, live[:, None])
        tl.store(Lse + index, logsum, live)

    @triton.jit
    def _merge_bshd(part, lse, out, M: tl.constexpr, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
        t = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        s = tl.arange(0, S)
        d = tl.arange(0, D)
        indices = (bh * S + s) * M + t
        lg = tl.load(lse + indices)
        mx = tl.max(lg, axis=0)
        w = tl.exp2(lg - mx)
        p = tl.load(part + indices[:, None] * D + d[None, :]).to(tl.float32)
        v = tl.sum(p * w[:, None], axis=0) / tl.sum(w, axis=0)
        tl.store(out + (b * M + t) * (H * D) + h * D + d, v)

    @triton.jit
    def _merge_bshd_fp8(
        part,
        lse,
        out_fp8,
        inv_scale: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        S: tl.constexpr,
    ):
        t = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        s = tl.arange(0, S)
        d = tl.arange(0, D)
        indices = (bh * S + s) * M + t
        lg = tl.load(lse + indices)
        mx = tl.max(lg, axis=0)
        w = tl.exp2(lg - mx)
        p = tl.load(part + indices[:, None] * D + d[None, :]).to(tl.float32)
        v = (tl.sum(p * w[:, None], axis=0) / tl.sum(w, axis=0)).to(tl.bfloat16).to(tl.float32)
        q = tl.clamp(v * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(out_fp8 + (b * M + t) * (H * D) + h * D + d, q)

    @triton.jit
    def _qkv_rope_split_kernel(
        qkv,
        cos_ptr,
        sin_ptr,
        q_out,
        k_out,
        v_out,
        B: tl.constexpr,
        T: tl.constexpr,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        D: tl.constexpr,
        HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        total_heads: tl.constexpr = QH + 2 * KVH
        h_idx = pid % total_heads
        bt = pid // total_heads
        b = bt // T
        t = bt % T
        row_stride: tl.constexpr = total_heads * D
        base = qkv + bt * row_stride + h_idx * D
        d_half = tl.arange(0, HALF)
        if h_idx < QH + KVH:
            x1 = tl.load(base + d_half).to(tl.float32)
            x2 = tl.load(base + HALF + d_half).to(tl.float32)
            cs_base = bt * HALF + d_half
            c = tl.load(cos_ptr + cs_base).to(tl.float32)
            s = tl.load(sin_ptr + cs_base).to(tl.float32)
            y1 = (x1 * c - x2 * s).to(tl.bfloat16)
            y2 = (x2 * c + x1 * s).to(tl.bfloat16)
            if h_idx < QH:
                dst = q_out + ((b * QH + h_idx) * T + t) * D
            else:
                kh = h_idx - QH
                dst = k_out + ((b * KVH + kh) * T + t) * D
            tl.store(dst + d_half, y1)
            tl.store(dst + HALF + d_half, y2)
        else:
            vh = h_idx - (QH + KVH)
            d_full = tl.arange(0, D)
            val = tl.load(base + d_full)
            dst = v_out + ((b * KVH + vh) * T + t) * D
            tl.store(dst + d_full, val)


def split_qkv_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch: int,
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed QKV and apply RoPE into BHSD tensors in one launch."""
    q_out = torch.empty((batch, q_heads, tokens, head_dim), device=qkv.device, dtype=torch.bfloat16)
    k_out = torch.empty((batch, kv_heads, tokens, head_dim), device=qkv.device, dtype=torch.bfloat16)
    v_out = torch.empty((batch, kv_heads, tokens, head_dim), device=qkv.device, dtype=torch.bfloat16)
    total_heads = q_heads + 2 * kv_heads
    _qkv_rope_split_kernel[(batch * tokens * total_heads,)](
        qkv,
        cos,
        sin,
        q_out,
        k_out,
        v_out,
        B=batch,
        T=tokens,
        QH=q_heads,
        KVH=kv_heads,
        D=head_dim,
        HALF=head_dim // 2,
        num_warps=2,
    )
    return q_out, k_out, v_out
