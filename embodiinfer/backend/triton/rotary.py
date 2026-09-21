"""Decode-only fused RoPE and KV-cache update kernels owned by VVLA."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _prefill_rope_cache_graph_kernel(
        queries,
        keys,
        values,
        cosine,
        sine,
        query_output,
        key_cache,
        value_cache,
        position,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_dim,
        stride_k_batch,
        stride_k_head,
        stride_k_token,
        stride_k_dim,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_dim,
        stride_qo_batch,
        stride_qo_head,
        stride_qo_token,
        stride_qo_dim,
        stride_kc_batch,
        stride_kc_head,
        stride_kc_token,
        stride_kc_dim,
        stride_vc_batch,
        stride_vc_head,
        stride_vc_token,
        stride_vc_dim,
        QUERY_LENGTH: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        program = tl.program_id(0)
        token_block = tl.program_id(1)
        batch = program // NUM_QUERY_HEADS
        head = program % NUM_QUERY_HEADS
        token_offsets = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
        feature_offsets = tl.arange(0, HEAD_DIM)
        token_mask = token_offsets < QUERY_LENGTH
        half = HEAD_DIM // 2
        paired_offsets = tl.where(
            feature_offsets < half,
            feature_offsets + half,
            feature_offsets - half,
        )
        signs = tl.where(feature_offsets < half, -1.0, 1.0)
        past_length = tl.load(position)
        absolute_positions = past_length + token_offsets

        query_base = (
            queries + batch * stride_q_batch + head * stride_q_head + token_offsets[:, None] * stride_q_token
        )
        query = tl.load(
            query_base + feature_offsets[None, :] * stride_q_dim,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        query_pair = tl.load(
            query_base + paired_offsets[None, :] * stride_q_dim,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        rope_offsets = absolute_positions[:, None] * HEAD_DIM + feature_offsets[None, :]
        cosine_values = (
            tl.load(
                cosine + rope_offsets,
                mask=token_mask[:, None],
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        sine_values = (
            tl.load(
                sine + rope_offsets,
                mask=token_mask[:, None],
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        rotated_query = query * cosine_values + signs[None, :] * query_pair * sine_values
        tl.store(
            query_output
            + batch * stride_qo_batch
            + head * stride_qo_head
            + token_offsets[:, None] * stride_qo_token
            + feature_offsets[None, :] * stride_qo_dim,
            rotated_query,
            mask=token_mask[:, None],
        )

        kv_mask = token_mask[:, None] & (head < NUM_KV_HEADS)
        key_base = (
            keys + batch * stride_k_batch + head * stride_k_head + token_offsets[:, None] * stride_k_token
        )
        key = tl.load(
            key_base + feature_offsets[None, :] * stride_k_dim,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        key_pair = tl.load(
            key_base + paired_offsets[None, :] * stride_k_dim,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        cache_offsets = absolute_positions[:, None]
        tl.store(
            key_cache
            + batch * stride_kc_batch
            + head * stride_kc_head
            + cache_offsets * stride_kc_token
            + feature_offsets[None, :] * stride_kc_dim,
            key * cosine_values + signs[None, :] * key_pair * sine_values,
            mask=kv_mask,
        )
        value = tl.load(
            values
            + batch * stride_v_batch
            + head * stride_v_head
            + token_offsets[:, None] * stride_v_token
            + feature_offsets[None, :] * stride_v_dim,
            mask=kv_mask,
            other=0.0,
        )
        tl.store(
            value_cache
            + batch * stride_vc_batch
            + head * stride_vc_head
            + cache_offsets * stride_vc_token
            + feature_offsets[None, :] * stride_vc_dim,
            value,
            mask=kv_mask,
        )

    @triton.jit
    def _rope_cache_kernel(
        queries,
        keys,
        values,
        cosine,
        sine,
        query_output,
        key_cache,
        value_cache,
        position,
        stride_q_batch,
        stride_q_head,
        stride_q_dim,
        stride_k_batch,
        stride_k_head,
        stride_k_dim,
        stride_v_batch,
        stride_v_head,
        stride_v_dim,
        stride_qo_batch,
        stride_qo_head,
        stride_qo_dim,
        stride_kc_batch,
        stride_kc_head,
        stride_kc_token,
        stride_kc_dim,
        stride_vc_batch,
        stride_vc_head,
        stride_vc_token,
        stride_vc_dim,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        POSITION_IS_POINTER: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch = program // NUM_QUERY_HEADS
        head = program % NUM_QUERY_HEADS
        current_position = tl.load(position) if POSITION_IS_POINTER else position
        offsets = tl.arange(0, HEAD_DIM)
        half = HEAD_DIM // 2
        paired_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        signs = tl.where(offsets < half, -1.0, 1.0)
        query_base = queries + batch * stride_q_batch + head * stride_q_head
        query = tl.load(query_base + offsets * stride_q_dim).to(tl.float32)
        query_pair = tl.load(query_base + paired_offsets * stride_q_dim).to(tl.float32)
        rope_offset = current_position * HEAD_DIM if POSITION_IS_POINTER else 0
        cos = tl.load(cosine + rope_offset + offsets).to(tl.bfloat16).to(tl.float32)
        sin = tl.load(sine + rope_offset + offsets).to(tl.bfloat16).to(tl.float32)
        tl.store(
            query_output + batch * stride_qo_batch + head * stride_qo_head + offsets * stride_qo_dim,
            query * cos + signs * query_pair * sin,
        )

        kv_mask = head < NUM_KV_HEADS
        key_base = keys + batch * stride_k_batch + head * stride_k_head
        key = tl.load(key_base + offsets * stride_k_dim, mask=kv_mask, other=0.0).to(tl.float32)
        key_pair = tl.load(
            key_base + paired_offsets * stride_k_dim,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        key_cache_base = (
            key_cache + batch * stride_kc_batch + head * stride_kc_head + current_position * stride_kc_token
        )
        value_cache_base = (
            value_cache + batch * stride_vc_batch + head * stride_vc_head + current_position * stride_vc_token
        )
        tl.store(
            key_cache_base + offsets * stride_kc_dim,
            key * cos + signs * key_pair * sin,
            mask=kv_mask,
        )
        value = tl.load(
            values + batch * stride_v_batch + head * stride_v_head + offsets * stride_v_dim,
            mask=kv_mask,
            other=0.0,
        )
        tl.store(
            value_cache_base + offsets * stride_vc_dim,
            value,
            mask=kv_mask,
        )


def supports_fused_rope_cache(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    return (
        triton is not None
        and queries.is_cuda
        and queries.dtype == torch.bfloat16
        and queries.dtype == keys.dtype == values.dtype
        and key_cache.dtype == value_cache.dtype == queries.dtype
        and queries.ndim == keys.ndim == values.ndim == 4
        and key_cache.ndim == value_cache.ndim == 4
        and queries.shape[-2] == keys.shape[-2] == values.shape[-2] == 1
        and queries.shape[-1] == keys.shape[-1] == values.shape[-1] == 128
        and queries.shape[1] % keys.shape[1] == 0
    )


def fused_rope_cache(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position: int,
) -> torch.Tensor:
    if not supports_fused_rope_cache(queries, keys, values, key_cache, value_cache):
        raise ValueError("unsupported input for VVLA Triton RoPE cache update")
    output = torch.empty(queries.shape, dtype=queries.dtype, device=queries.device)
    batch_size, num_query_heads, _, head_dim = queries.shape
    _rope_cache_kernel[(batch_size * num_query_heads,)](
        queries,
        keys,
        values,
        cosine,
        sine,
        output,
        key_cache,
        value_cache,
        position,
        queries.stride(0),
        queries.stride(1),
        queries.stride(3),
        keys.stride(0),
        keys.stride(1),
        keys.stride(3),
        values.stride(0),
        values.stride(1),
        values.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=keys.shape[1],
        HEAD_DIM=head_dim,
        POSITION_IS_POINTER=False,
        num_warps=4,
    )
    return output


def fused_rope_cache_graph(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    if (
        not supports_fused_rope_cache(queries, keys, values, key_cache, value_cache)
        or position.numel() != 1
        or not position.is_cuda
    ):
        raise ValueError("unsupported input for VVLA graph RoPE cache update")
    output = torch.empty(queries.shape, dtype=queries.dtype, device=queries.device)
    batch_size, num_query_heads, _, head_dim = queries.shape
    _rope_cache_kernel[(batch_size * num_query_heads,)](
        queries,
        keys,
        values,
        cosine,
        sine,
        output,
        key_cache,
        value_cache,
        position,
        queries.stride(0),
        queries.stride(1),
        queries.stride(3),
        keys.stride(0),
        keys.stride(1),
        keys.stride(3),
        values.stride(0),
        values.stride(1),
        values.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=keys.shape[1],
        HEAD_DIM=head_dim,
        POSITION_IS_POINTER=True,
        num_warps=4,
    )
    return output


def fused_prefill_rope_cache_graph(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    if (
        triton is None
        or not queries.is_cuda
        or queries.dtype != torch.bfloat16
        or queries.dtype != keys.dtype
        or queries.dtype != values.dtype
        or queries.ndim != 4
        or keys.ndim != 4
        or values.ndim != 4
        or queries.shape[-2] <= 1
        or queries.shape[-2] != keys.shape[-2]
        or keys.shape != values.shape
        or queries.shape[-1] != 128
        or position.numel() != 1
        or not position.is_cuda
    ):
        raise ValueError("unsupported input for VVLA graph prefill RoPE update")
    output = torch.empty_like(queries)
    batch_size, num_query_heads, query_length, head_dim = queries.shape
    block_t = 8
    _prefill_rope_cache_graph_kernel[(batch_size * num_query_heads, triton.cdiv(query_length, block_t))](
        queries,
        keys,
        values,
        cosine,
        sine,
        output,
        key_cache,
        value_cache,
        position,
        *queries.stride(),
        *keys.stride(),
        *values.stride(),
        *output.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        QUERY_LENGTH=query_length,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=keys.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return output


if triton is not None:

    @triton.jit
    def _rotate_qk_kernel(
        Q,
        K,
        Cos,
        Sin,
        QOut,
        KOut,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qs: tl.constexpr,
        kb: tl.constexpr,
        kh: tl.constexpr,
        ks: tl.constexpr,
        cb: tl.constexpr,
        cs: tl.constexpr,
        sb: tl.constexpr,
        ss: tl.constexpr,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        S: tl.constexpr,
        D: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = row % (HQ + HK)
        token = (row // (HQ + HK)) % S
        batch = row // ((HQ + HK) * S)
        dims = tl.arange(0, D)
        half = (dims + D // 2) % D
        is_query = head < HQ
        q = tl.load(Q + batch * qb + head * qh + token * qs + dims, is_query, other=0)
        qr = tl.load(Q + batch * qb + head * qh + token * qs + half, is_query, other=0)
        k = tl.load(K + batch * kb + (head - HQ) * kh + token * ks + dims, ~is_query, other=0)
        kr = tl.load(K + batch * kb + (head - HQ) * kh + token * ks + half, ~is_query, other=0)
        x, rotated = tl.where(is_query, q, k).to(tl.float32), tl.where(is_query, qr, kr).to(tl.float32)
        rotated = tl.where(dims < D // 2, -rotated, rotated)
        cos = tl.load(Cos + batch * cb + token * cs + dims).to(tl.float32)
        sin = tl.load(Sin + batch * sb + token * ss + dims).to(tl.float32)
        first = (x * cos).to(q.dtype).to(tl.float32)
        second = (rotated * sin).to(q.dtype).to(tl.float32)
        result = first + second
        tl.store(QOut + ((batch * HQ + head) * S + token) * D + dims, result, is_query)
        tl.store(KOut + ((batch * HK + head - HQ) * S + token) * D + dims, result, ~is_query)

else:
    _rotate_qk_kernel = None


def rotate_qk(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotate-half Q/K RoPE in BHSD layout, preserving intermediate rounding."""
    if triton is None or query.device.type != "cuda":
        raise RuntimeError("fused RoPE requires CUDA and Triton")
    batch, heads, size, width = query.shape
    if width not in (64, 128, 256) or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("fused RoPE requires FP16/BF16 and head width 64/128/256")
    if (
        key.shape[0] != batch
        or key.shape[2:] != (size, width)
        or cos.shape != (batch, size, width)
        or sin.shape != cos.shape
    ):
        raise ValueError("fused RoPE tensor shapes are incompatible")
    for tensor in (query, key, cos, sin):
        if tensor.device != query.device or tensor.dtype != query.dtype or tensor.stride(-1) != 1:
            raise ValueError("fused RoPE tensors must share device/dtype and contiguous final dimensions")
    qout = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    kout = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    _rotate_qk_kernel[(batch * size * (heads + key.shape[1]),)](
        query,
        key,
        cos,
        sin,
        qout,
        kout,
        *query.stride()[:3],
        *key.stride()[:3],
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        heads,
        key.shape[1],
        size,
        width,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return qout, kout


__all__ = [
    "rotate_qk",
    "fused_prefill_rope_cache_graph",
    "fused_rope_cache",
    "fused_rope_cache_graph",
    "supports_fused_rope_cache",
]
