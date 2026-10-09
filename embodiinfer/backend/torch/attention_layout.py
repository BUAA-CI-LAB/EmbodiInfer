"""Mask plans and query-major attention tensor operations."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_BACKENDS = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}


@dataclass(frozen=True)
class MaskPlan:
    """Read-only gather/scatter indices for a fixed shared-key mask layout.

    Valid queries must share the same key set within each batch member. Fully
    masked queries are handled separately with uniform attention to every key.
    Preparing/checking the plan synchronizes tensors and belongs outside capture.
    """

    mask: torch.Tensor
    indices: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    dense: bool


def prepare_mask_plan(mask: torch.Tensor) -> MaskPlan:
    """Validate shared-key mask structure and precompute device-local indices."""
    if mask.dtype != torch.bool or mask.ndim != 3:
        raise ValueError("Compact attention requires a Boolean [batch, queries, keys] mask")
    indices = []
    dense = True
    for member in mask:
        valid = member.any(dim=-1).nonzero().flatten()
        invalid = (~member.any(dim=-1)).nonzero().flatten()
        if valid.numel():
            common = member[valid[0]]
            if not torch.equal(member.index_select(0, valid), common.expand(valid.numel(), -1)):
                raise ValueError("Compact attention requires a shared key set for valid queries")
            keys = common.nonzero().flatten()
        else:
            keys = torch.empty(0, dtype=torch.long, device=mask.device)
        dense &= valid.numel() == member.shape[0] and keys.numel() == member.shape[1]
        indices.append((valid, keys, invalid))
    return MaskPlan(mask.clone(), tuple(indices), dense)


def compact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: MaskPlan,
    backend: str = "flash",
) -> torch.Tensor:
    """Gather valid tokens and run mask-free SDPA, returning the original query layout.

    Fully masked queries use zero Q and all K/V, preserving uniform attention
    semantics. SDPA still changes the original probability BF16 rounding and
    reduction arithmetic. Math is available for independent CPU validation.
    """
    expected = (query.shape[0], query.shape[1], key.shape[1])
    if plan.mask.shape != expected:
        raise ValueError("Compact plan does not match query/key shape")
    if plan.dense:
        return _unmasked_attention(query, key, value, backend)
    output = torch.empty_like(query)
    for batch, (valid, keys, invalid) in enumerate(plan.indices):
        current = query[batch : batch + 1]
        current_key, current_value = key[batch : batch + 1], value[batch : batch + 1]
        if valid.numel():
            encoded = _unmasked_attention(
                current.index_select(1, valid),
                current_key.index_select(1, keys),
                current_value.index_select(1, keys),
                backend,
            )
            output[batch : batch + 1].index_copy_(1, valid, encoded)
        if invalid.numel():
            zero_query = torch.zeros(
                (1, invalid.numel(), *query.shape[2:]), dtype=query.dtype, device=query.device
            )
            uniform = _unmasked_attention(zero_query, current_key, current_value, backend)
            output[batch : batch + 1].index_copy_(1, invalid, uniform)
    return output


def _unmasked_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, backend: str
) -> torch.Tensor:
    dtype = query.dtype
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    if backend == "math":
        query, key, value = (tensor.float() for tensor in (query, key, value))
    with sdpa_kernel(backends=[_BACKENDS[backend]]):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
            enable_gqa=query.shape[1] != key.shape[1],
        )
    return output.to(dtype).transpose(1, 2)


def query_major_qk(query: torch.Tensor, key: torch.Tensor, expert: int | None = None) -> torch.Tensor:
    """Flatten T,G rows directly, retaining BF16 inputs and FP32 accumulation.

    Query already follows [B,T,KV_heads,groups,width]. With one KV head this
    avoids the full query copy in the group-major reference BF16 QK adapter.
    Returned logits retain the logical [B,KV_heads,groups,T,S] contract.
    """
    batch, queries, heads, groups, width = query.shape
    keys = key.shape[1]
    q = query.permute(0, 2, 1, 3, 4).reshape(batch * heads, queries * groups, width)
    k = key.permute(0, 2, 3, 1).reshape(batch * heads, width, keys)
    # Honor the caller's matmul settings without mutating process-global state.
    logits = torch.bmm(q, k, out_dtype=torch.float32)
    return logits.reshape(batch, heads, queries, groups, keys).transpose(2, 3)
