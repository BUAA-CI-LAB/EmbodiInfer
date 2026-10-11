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
