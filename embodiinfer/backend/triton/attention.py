"""Graph-safe grouped-query decode attention owned by EmbodiInfer."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _gqa_decode_graph_kernel(
        queries,
        key_cache,
        value_cache,
        position,
        output,
        scale,
        stride_q_batch,
        stride_q_head,
        stride_q_dim,
        stride_k_batch,
        stride_k_head,
        stride_k_token,
        stride_k_dim,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_dim,
        stride_o_batch,
        stride_o_head,
        stride_o_dim,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch = program // NUM_QUERY_HEADS
        query_head = program % NUM_QUERY_HEADS
        kv_head = query_head // (NUM_QUERY_HEADS // NUM_KV_HEADS)
        context_length = tl.load(position) + 1
        offsets_d = tl.arange(0, HEAD_DIM)
        query = tl.load(
            queries + batch * stride_q_batch + query_head * stride_q_head + offsets_d * stride_q_dim
        ).to(tl.float32)
        maximum = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        for start_n in tl.range(0, context_length, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            token_mask = offsets_n < context_length
            key = tl.load(
                key_cache
                + batch * stride_k_batch
                + kv_head * stride_k_head
                + offsets_n[:, None] * stride_k_token
                + offsets_d[None, :] * stride_k_dim,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * query[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            correction = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum)
            value = tl.load(
                value_cache
                + batch * stride_v_batch
                + kv_head * stride_v_head
                + offsets_n[:, None] * stride_v_token
                + offsets_d[None, :] * stride_v_dim,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(
                probabilities[:, None] * value,
                axis=0,
            )
            denominator = denominator * correction + tl.sum(probabilities, axis=0)
            maximum = next_maximum

        tl.store(
            output + batch * stride_o_batch + query_head * stride_o_head + offsets_d * stride_o_dim,
            accumulator / denominator,
        )


def graph_gqa_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def gqa_decode_graph(
    queries: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    if (
        not graph_gqa_available()
        or queries.dtype != torch.bfloat16
        or queries.ndim != 4
        or queries.shape[-2:] != (1, 128)
        or key_cache.shape != value_cache.shape
        or position.numel() != 1
    ):
        raise ValueError("unsupported input for EmbodiInfer graph GQA decode")
    output = torch.empty(queries.shape, dtype=queries.dtype, device=queries.device)
    batch_size, num_query_heads, _, head_dim = queries.shape
    _gqa_decode_graph_kernel[(batch_size * num_query_heads,)](
        queries,
        key_cache,
        value_cache,
        position,
        output,
        1.0 / math.sqrt(head_dim),
        queries.stride(0),
        queries.stride(1),
        queries.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(3),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=key_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_N=64,
        num_warps=4,
    )
    return output


__all__ = ["gqa_decode_graph", "graph_gqa_available"]
