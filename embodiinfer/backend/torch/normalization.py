"""Torch reference RMSNorm with explicit affine and residual rounding."""

import math

import torch

from ...layers.registry import OperatorCapabilities


class TorchRMSNorm:
    """Prepare model-independent RMSNorm plans using Torch's FP32 mean."""

    capabilities = OperatorCapabilities(
        ("cpu", "cuda"), (torch.bfloat16, torch.float16, torch.float32), "torch_rmsnorm_bf16_rounding", rank=3
    )

    def plan(self, inputs: torch.Tensor, *, adaptive: bool, eps: float = 1e-6) -> "TorchNormPlan":
        """Allocate a plan with the same offset-scale/AdaRMS contract as CUDA."""
        return TorchNormPlan(inputs, adaptive, eps)


class TorchNormPlan:
    """Scoped reference buffers overwritten by each normalize/residual call."""

    def __init__(self, inputs: torch.Tensor, adaptive: bool, eps: float) -> None:
        """Record input layout and allocate stable output addresses before capture."""
        if inputs.ndim != 3 or min(inputs.shape) <= 0 or not inputs.is_contiguous():
            raise ValueError("RMSNorm requires nonempty contiguous [B,T,D] inputs")
        if inputs.dtype not in TorchRMSNorm.capabilities.dtypes or not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMSNorm requires floating inputs and a positive finite epsilon")
        self.shape, self.adaptive, self.eps = inputs.shape, adaptive, eps
        self.output = torch.empty_like(inputs)
        self.residual = torch.empty_like(inputs)
        self.gate = inputs.new_empty((inputs.shape[0], 1, inputs.shape[2])) if adaptive else None

    def _validate(self, inputs: torch.Tensor) -> None:
        if (
            inputs.shape != self.shape
            or inputs.dtype != self.output.dtype
            or inputs.device != self.output.device
            or not inputs.is_contiguous()
        ):
            raise ValueError("RMSNorm input shape, layout, dtype or device changed")

    def normalize(
        self,
        inputs: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply raw FP32 affine parameters and preserve Torch rounding boundaries."""
        self._validate(inputs)
        affine = modulation if self.adaptive else scale
        expected = (self.shape[0], 3 * self.shape[2]) if self.adaptive else (self.shape[2],)
        if (
            affine is None
            or affine.shape != expected
            or affine.dtype != torch.float32
            or affine.device != inputs.device
            or not affine.is_contiguous()
        ):
            raise ValueError("RMSNorm requires matching contiguous FP32 scale or raw modulation")
        mean = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(mean + self.eps)
        if self.adaptive:
            scale_value, shift, gate = modulation[:, None].chunk(3, dim=-1)
            normalized = normalized * (1 + scale_value) + shift
            self.gate.copy_(gate.to(inputs.dtype))
        else:
            normalized = normalized * (1 + scale)
        self.output.copy_(normalized.to(inputs.dtype))
        return self.output, self.gate

    def residual_normalize(
        self,
        inputs: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor | None,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Round the gated update, then its sum with the residual, then normalize."""
        self._validate(inputs)
        self._validate(update)
        if gate is not None and (
            gate.shape != (self.shape[0], 1, self.shape[2])
            or gate.dtype != inputs.dtype
            or gate.device != inputs.device
        ):
            raise ValueError("Residual gate must share [B,1,D], dtype and device")
        self.residual.copy_(inputs + (update if gate is None else update * gate))
        normalized, next_gate = self.normalize(self.residual, scale=scale, modulation=modulation)
        return self.residual, normalized, next_gate
