"""Lossless eager baseline with an explicit intermediate dtype rounding."""

import torch


def run(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Match the separate multiply/add semantics used by the gated residual path."""
    product = (update * gate).to(update.dtype)
    return (residual + product).to(residual.dtype)
