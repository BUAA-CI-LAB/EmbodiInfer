"""Torch reference rotation with FP32 factors and explicit BF16 output."""

import torch

from ...layers.registry import OperatorCapabilities


class TorchRotary:
    """Apply the original separate multiply/add rotary arithmetic."""

    capabilities = OperatorCapabilities(("cpu", "cuda"), (torch.bfloat16,), "fp32_rope_bf16_output", rank=4)

    def __call__(self, inputs: torch.Tensor, sine: torch.Tensor, cosine: torch.Tensor) -> torch.Tensor:
        """Return a distinct BTNH tensor; factors have shape [B,T,1,H/2]."""
        if inputs.ndim != 4 or inputs.dtype != torch.bfloat16:
            raise ValueError("Rotary requires rank-four BF16 inputs")
        expected = (*inputs.shape[:2], 1, inputs.shape[-1] // 2)
        if inputs.shape[-1] % 2 or any(
            value.shape != expected or value.dtype != torch.float32 or value.device != inputs.device
            for value in (sine, cosine)
        ):
            raise ValueError("Rotary requires matching FP32 half-width factors")
        first, second = inputs.chunk(2, dim=-1)
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1).to(
            inputs.dtype
        )
