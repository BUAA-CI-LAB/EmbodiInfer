"""Torch reference implementations of registry operators.

Each function is the default implementation of the operator with the same name
in :mod:`embodiinfer.layers.ops` and defines that operator's mathematics. The
arithmetic order is part of the contract: policies compile these functions with
Inductor, and changing an expression changes their rounding.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def gated_residual(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return ``residual + update * gate``."""
    return residual + update * gate


def ada_rms_norm(x: torch.Tensor, modulation: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive RMSNorm over ``[batch, tokens, hidden]`` with ``[batch, 3 * hidden]`` modulation.

    The modulation holds scale, shift and residual gate. Variance and the affine
    step are FP32; the normalized output and the gate return in ``x.dtype``.
    """
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    if x.ndim == 3:
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    normed = normed * (1 + scale.float()) + shift.float()
    return normed.to(x.dtype), gate.to(x.dtype)


def gated_gelu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return tanh-approximate ``GELU(gate) * up``."""
    return F.gelu(gate, approximate="tanh") * up


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_qk(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half RoPE for BHSD query/key with ``[batch, tokens, head_dim]`` tables."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def split_kv_attention(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Non-causal BHSD attention over cached (prefix) and current (suffix) K/V.

    ``mask`` is a shared floating padding mask ``[batch, 1, 1, prefix + suffix]``:
    zero keeps a key, any other value removes it. Query heads may be a multiple
    of the KV heads (grouped-query attention).
    """
    key = torch.cat((prefix_key, suffix_key), dim=2)
    value = torch.cat((prefix_value, suffix_value), dim=2)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None if mask is None else mask == 0,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
