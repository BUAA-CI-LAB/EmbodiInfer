"""Torch reference tensorwise FP8 activation encoder."""

import math

import torch

from ...layers.registry import OperatorCapabilities


class TorchQuantizer:
    """Prepare tensorwise E4M3 encoders with the native quantizer's scale contract."""

    capabilities = OperatorCapabilities(
        ("cpu", "cuda"), (torch.bfloat16,), "rounded_bf16_encoding", rank=2, bits=(8,)
    )

    def plan(self, inputs: torch.Tensor, bits: int, scale: float | None = None) -> "TorchFP8Plan":
        """Prepare an FP8 reference plan; NVFP4 has no Torch reference encoder here."""
        if bits != 8:
            raise ValueError("Torch reference quantization supports 8-bit encoding only")
        return TorchFP8Plan(inputs, scale)


class TorchFP8Plan:
    """Fixed output/scale storage, overwritten when new activations are encoded."""

    def __init__(self, inputs: torch.Tensor, scale: float | None) -> None:
        """Allocate tensorwise FP8 output without a dependency on native CUDA."""
        if (
            inputs.ndim != 2
            or min(inputs.shape) <= 0
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
        ):
            raise ValueError("FP8 reference encoding requires a contiguous BF16 matrix")
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("Calibrated scale must be finite and positive")
        self.shape, self.dynamic = inputs.shape, scale is None
        self.output = torch.empty_like(inputs, dtype=torch.float8_e4m3fn)
        self.scale = torch.tensor(1.0 if scale is None else scale, device=inputs.device, dtype=torch.float32)
        self.scales = self.blocked = None

    def quantize(self, inputs: torch.Tensor) -> "TorchFP8Plan":
        """Encode rounded BF16 inputs and use saturation to match CUDA conversions."""
        if (
            inputs.shape != self.shape
            or inputs.device != self.output.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
        ):
            raise ValueError("FP8 reference input shape, layout, dtype or device changed")
        values = inputs.float()
        if self.dynamic:
            maximum = values.abs().amax()
            self.scale.copy_(torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum)))
        self.output.copy_((values / self.scale).clamp(-448, 448).to(torch.float8_e4m3fn))
        return self
