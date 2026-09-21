"""Concrete attention implementations backed by PyTorch operators."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def _prepare_mask(
    attn_mask: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor | None:
    if attn_mask is None:
        return None
    if attn_mask.shape[-1] != k.shape[-2]:
        attn_mask = attn_mask[..., : k.shape[-2]]
    if attn_mask.is_floating_point() and attn_mask.dtype != q.dtype:
        attn_mask = attn_mask.to(q.dtype)
    return attn_mask


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand KV heads for grouped-query or multi-query attention."""
    if repeats == 1:
        return x
    batch, kv_heads, sequence, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, kv_heads, repeats, sequence, head_dim)
        .reshape(batch, kv_heads * repeats, sequence, head_dim)
    )


class EagerAttention:
    """Reference attention matching Hugging Face eager attention semantics."""

    name = "eager"

    def attend(self, q, k, v, attn_mask=None, scaling=None, dropout_p=0.0):
        scaling = q.shape[-1] ** -0.5 if scaling is None else scaling
        repeats = q.shape[1] // k.shape[1]
        k, v = _repeat_kv(k, repeats), _repeat_kv(v, repeats)
        attn_mask = _prepare_mask(attn_mask, q, k)
        attention = torch.matmul(q, k.transpose(-2, -1)) * scaling
        if attn_mask is not None:
            attention = attention + attn_mask
        attention = F.softmax(attention, dim=-1, dtype=torch.float32).to(q.dtype)
        if dropout_p > 0.0:
            attention = F.dropout(attention, p=dropout_p)
        return torch.matmul(attention, v)


class EagerBroadcastAttention:
    """Eager attention with broadcast GQA instead of materialized KV copies."""

    name = "eager_bc"

    def attend(self, q, k, v, attn_mask=None, scaling=None, dropout_p=0.0):
        scaling = q.shape[-1] ** -0.5 if scaling is None else scaling
        repeats = q.shape[1] // k.shape[1]
        attn_mask = _prepare_mask(attn_mask, q, k)
        if repeats == 1:
            attention = torch.matmul(q, k.transpose(-2, -1)) * scaling
            if attn_mask is not None:
                attention = attention + attn_mask
            attention = F.softmax(attention, dim=-1, dtype=torch.float32).to(q.dtype)
            if dropout_p > 0.0:
                attention = F.dropout(attention, p=dropout_p)
            return torch.matmul(attention, v)
        batch, heads, query_length, head_dim = q.shape
        kv_heads = k.shape[1]
        grouped_q = q.reshape(batch, kv_heads, repeats, query_length, head_dim)
        attention = torch.matmul(grouped_q, k.unsqueeze(2).transpose(-2, -1)) * scaling
        if attn_mask is not None:
            attention = attention + attn_mask.unsqueeze(1)
        attention = F.softmax(attention, dim=-1, dtype=torch.float32).to(q.dtype)
        if dropout_p > 0.0:
            attention = F.dropout(attention, p=dropout_p)
        output = torch.matmul(attention, v.unsqueeze(2))
        return output.reshape(batch, heads, query_length, head_dim)


def _supports_gqa(sdpa: Callable[..., Any]) -> bool:
    """Inspect static metadata without probing a kernel during graph capture."""
    try:
        return "enable_gqa" in inspect.signature(sdpa).parameters
    except (TypeError, ValueError):
        # PyTorch built-ins often expose their signature only in the docstring.
        return "enable_gqa" in (getattr(sdpa, "__doc__", "") or "")


class SDPAAttention:
    """Attention backed by ``torch.nn.functional.scaled_dot_product_attention``."""

    name = "sdpa"

    def __init__(self) -> None:
        self._sdpa = F.scaled_dot_product_attention
        self._supports_gqa = _supports_gqa(self._sdpa)

    def attend(self, q, k, v, attn_mask=None, scaling=None, dropout_p=0.0):
        attn_mask = _prepare_mask(attn_mask, q, k)
        if not self._supports_gqa:
            repeats = q.shape[1] // k.shape[1]
            k, v = _repeat_kv(k, repeats), _repeat_kv(v, repeats)
            return self._sdpa(q, k, v, attn_mask=attn_mask, scale=scaling, dropout_p=dropout_p)
        return self._sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            scale=scaling,
            dropout_p=dropout_p,
            enable_gqa=q.shape[1] != k.shape[1],
        )


class FlexAttention:
    """Reserved PyTorch FlexAttention backend."""

    name = "flex"

    def attend(self, q, k, v, attn_mask=None, scaling=None, dropout_p=0.0):
        raise NotImplementedError(
            "the 'flex' attention backend is not implemented yet; use 'sdpa', 'eager', or 'eager_bc'"
        )


__all__ = [
    "EagerAttention",
    "EagerBroadcastAttention",
    "FlexAttention",
    "SDPAAttention",
]
