"""FP32 rotary-factor contracts and lazy backend selection."""

from typing import Protocol

import torch

from .registry import BackendRegistry, OperatorCapabilities


class RotaryBackend(Protocol):
    """Rotate BTNH inputs with FP32 half-width factors and BF16 output."""

    capabilities: OperatorCapabilities

    def __call__(self, inputs: torch.Tensor, sine: torch.Tensor, cosine: torch.Tensor) -> torch.Tensor:
        """Preserve separate FP32 products and the final BF16 conversion."""
        ...


rotary_backends = BackendRegistry[RotaryBackend]("rotary")
rotary_backends.register_lazy("cuda", "embodiinfer.backend.cuda.rotary", "RotaryKernel")
rotary_backends.register_lazy("torch", "embodiinfer.backend.torch.rotary", "TorchRotary")
