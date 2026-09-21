from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional backend
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _flash_prefill_attention_graph_kernel(
        query,
        key_cache,
        value_cache,
        position,
        output,
        scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        QUERY_LENGTH: tl.constexpr,
        KEY_BUCKET: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_QUERY_HEADS
        query_head = batch_head % NUM_QUERY_HEADS
        kv_head = query_head // (NUM_QUERY_HEADS // NUM_KV_HEADS)
        past_length = tl.load(position)
        context_length = past_length + QUERY_LENGTH
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        feature_offsets = tl.arange(0, BLOCK_D)
        query_valid = query_offsets < QUERY_LENGTH
        query_values = tl.load(
            query
            + batch * stride_qb
            + query_head * stride_qh
            + query_offsets[:, None] * stride_qm
            + feature_offsets[None, :] * stride_qd,
            mask=query_valid[:, None] & (feature_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, KEY_BUCKET, BLOCK_N):
            key_offsets = key_start + tl.arange(0, BLOCK_N)
            token_valid = key_offsets < context_length
            key_values = tl.load(
                key_cache
                + batch * stride_kb
                + kv_head * stride_kh
                + key_offsets[:, None] * stride_kn
                + feature_offsets[None, :] * stride_kd,
                mask=token_valid[:, None] & (feature_offsets[None, :] < HEAD_DIM),
                other=0.0,
            )
            value_values = tl.load(
                value_cache
                + batch * stride_vb
                + kv_head * stride_vh
                + key_offsets[:, None] * stride_vn
                + feature_offsets[None, :] * stride_vd,
                mask=token_valid[:, None] & (feature_offsets[None, :] < HEAD_DIM),
                other=0.0,
            )
            scores = tl.dot(query_values, tl.trans(key_values)).to(tl.float32)
            scores = scores * scale
            valid_scores = (
                query_valid[:, None]
                & token_valid[None, :]
                & (key_offsets[None, :] <= past_length + query_offsets[:, None])
            )
            scores = tl.where(valid_scores, scores, -float("inf"))
            scores = tl.where(query_valid[:, None], scores, 0.0)
            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(row_max, block_max)
            correction = tl.exp2((row_max - next_max) * 1.4426950408889634)
            probabilities = tl.exp2((scores - next_max[:, None]) * 1.4426950408889634)
            accumulator = accumulator * correction[:, None]
            accumulator += tl.dot(probabilities.to(query_values.dtype), value_values)
            row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
            row_max = next_max

        output_values = accumulator / row_sum[:, None]
        tl.store(
            output
            + batch * stride_ob
            + query_head * stride_oh
            + query_offsets[:, None] * stride_om
            + feature_offsets[None, :] * stride_od,
            output_values,
            mask=query_valid[:, None] & (feature_offsets[None, :] < HEAD_DIM),
        )

    @triton.jit
    def _flash_prefill_attention_kernel(
        query,
        key,
        value,
        mask,
        output,
        scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_mn,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        QUERY_LENGTH,
        KEY_LENGTH,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        MASK_IS_BOOL: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_QUERY_HEADS
        query_head = batch_head % NUM_QUERY_HEADS
        kv_head = query_head // (NUM_QUERY_HEADS // NUM_KV_HEADS)

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        feature_offsets = tl.arange(0, BLOCK_D)
        query_ptrs = (
            query
            + batch * stride_qb
            + query_head * stride_qh
            + query_offsets[:, None] * stride_qm
            + feature_offsets[None, :] * stride_qd
        )
        query_values = tl.load(
            query_ptrs,
            mask=(query_offsets[:, None] < QUERY_LENGTH) & (feature_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        query_is_valid = query_offsets < QUERY_LENGTH
        causal_offset = KEY_LENGTH - QUERY_LENGTH

        for key_start in tl.range(0, KEY_LENGTH, BLOCK_N):
            key_offsets = key_start + tl.arange(0, BLOCK_N)
            key_ptrs = (
                key
                + batch * stride_kb
                + kv_head * stride_kh
                + key_offsets[:, None] * stride_kn
                + feature_offsets[None, :] * stride_kd
            )
            value_ptrs = (
                value
                + batch * stride_vb
                + kv_head * stride_vh
                + key_offsets[:, None] * stride_vn
                + feature_offsets[None, :] * stride_vd
            )
            key_values = tl.load(
                key_ptrs,
                mask=(key_offsets[:, None] < KEY_LENGTH) & (feature_offsets[None, :] < HEAD_DIM),
                other=0.0,
            )
            value_values = tl.load(
                value_ptrs,
                mask=(key_offsets[:, None] < KEY_LENGTH) & (feature_offsets[None, :] < HEAD_DIM),
                other=0.0,
            )

            scores = tl.dot(query_values, tl.trans(key_values)).to(tl.float32)
            scores = scores * scale
            valid_scores = query_is_valid[:, None] & (key_offsets[None, :] < KEY_LENGTH)
            if IS_CAUSAL:
                valid_scores = valid_scores & (key_offsets[None, :] <= causal_offset + query_offsets[:, None])

            if HAS_MASK:
                mask_ptrs = (
                    mask
                    + batch * stride_mb
                    + query_head * stride_mh
                    + query_offsets[:, None] * stride_mm
                    + key_offsets[None, :] * stride_mn
                )
                mask_values = tl.load(
                    mask_ptrs,
                    mask=query_is_valid[:, None] & (key_offsets[None, :] < KEY_LENGTH),
                    other=0,
                )
                if MASK_IS_BOOL:
                    valid_scores = valid_scores & (mask_values != 0)
                else:
                    scores = scores + mask_values.to(tl.float32)

            scores = tl.where(valid_scores, scores, -float("inf"))
            scores = tl.where(query_is_valid[:, None], scores, 0.0)

            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(row_max, block_max)
            correction = tl.exp2((row_max - next_max) * 1.4426950408889634)
            probabilities = tl.exp2((scores - next_max[:, None]) * 1.4426950408889634)
            block_sum = tl.sum(probabilities, axis=1)
            accumulator = accumulator * correction[:, None]
            accumulator += tl.dot(probabilities.to(query_values.dtype), value_values)
            row_sum = row_sum * correction + block_sum
            row_max = next_max

        output_values = accumulator / row_sum[:, None]
        output_ptrs = (
            output
            + batch * stride_ob
            + query_head * stride_oh
            + query_offsets[:, None] * stride_om
            + feature_offsets[None, :] * stride_od
        )
        tl.store(
            output_ptrs,
            output_values,
            mask=(query_offsets[:, None] < QUERY_LENGTH) & (feature_offsets[None, :] < HEAD_DIM),
        )


def _normalize_attention_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
) -> torch.Tensor | None:
    if mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    elif mask.ndim != 4:
        return None

    expected = (batch_size, num_heads, query_length, key_length)
    if any(actual not in (1, wanted) for actual, wanted in zip(mask.shape, expected)):
        return None
    if mask.dtype != torch.bool and not mask.dtype.is_floating_point:
        return None
    return mask.expand(expected)


def supports_flash_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    enable_gqa: bool,
) -> bool:
    if triton is None or not query.is_cuda or dropout_p != 0.0:
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.shape[2] <= 1 or query.shape[-1] > 256:
        return False
    if query.shape[-1] != key.shape[-1] or key.shape != value.shape:
        return False
    if query.shape[0] != key.shape[0]:
        return False
    if query.shape[1] % key.shape[1] != 0:
        return False
    if query.shape[1] != key.shape[1] and not enable_gqa:
        return False
    if query.shape[2] > key.shape[2]:
        return False
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value)):
        return False
    return attn_mask is None or attn_mask.device == query.device


def flash_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Run EmbodiInfer Triton FlashAttention for prefill, with an SDPA fallback."""

    if not supports_flash_prefill_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        enable_gqa=enable_gqa,
    ):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    batch_size, num_query_heads, query_length, head_dim = query.shape
    _, num_kv_heads, key_length, _ = key.shape
    normalized_mask = None
    if attn_mask is not None:
        normalized_mask = _normalize_attention_mask(
            attn_mask,
            batch_size=batch_size,
            num_heads=num_query_heads,
            query_length=query_length,
            key_length=key_length,
        )
        if normalized_mask is None:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

    output = torch.empty_like(query)
    block_m = 32
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    mask_arg = normalized_mask if normalized_mask is not None else query
    mask_strides = (0, 0, 0, 0)
    if normalized_mask is not None:
        mask_strides = normalized_mask.stride()
    grid = (triton.cdiv(query_length, block_m), batch_size * num_query_heads)
    _flash_prefill_attention_kernel[grid](
        query,
        key,
        value,
        mask_arg,
        output,
        float(scale if scale is not None else 1.0 / math.sqrt(head_dim)),
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *mask_strides,
        *output.stride(),
        QUERY_LENGTH=query_length,
        KEY_LENGTH=key_length,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        IS_CAUSAL=is_causal,
        HAS_MASK=normalized_mask is not None,
        MASK_IS_BOOL=normalized_mask is not None and normalized_mask.dtype == torch.bool,
        num_warps=4,
        num_stages=3,
    )
    return output


def flash_prefill_attention_graph(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position: torch.Tensor,
    key_bucket: int,
) -> torch.Tensor:
    if (
        triton is None
        or not query.is_cuda
        or query.dtype != torch.bfloat16
        or query.ndim != 4
        or query.shape[2] <= 1
        or query.shape[-1] != 128
        or key_cache.shape != value_cache.shape
        or position.numel() != 1
        or key_bucket > key_cache.shape[2]
    ):
        raise ValueError("unsupported input for EmbodiInfer graph prefill attention")
    output = torch.empty_like(query)
    batch_size, num_query_heads, query_length, head_dim = query.shape
    block_m = 32
    block_n = 64
    _flash_prefill_attention_graph_kernel[(triton.cdiv(query_length, block_m), batch_size * num_query_heads)](
        query,
        key_cache,
        value_cache,
        position,
        output,
        1.0 / math.sqrt(head_dim),
        *query.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        *output.stride(),
        QUERY_LENGTH=query_length,
        KEY_BUCKET=key_bucket,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=key_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=128,
        num_warps=4,
        num_stages=3,
    )
    return output
