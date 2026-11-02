"""Gemma offset RMSNorm with lossless preparation of immutable affine weights."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ...layers.registry import OperatorCapabilities, OperatorRequest
from ..torch.normalization import TorchNormPlan
from .runtime import load_module


class FlashInferGemmaRMSNorm:
    """Nonadaptive BF16 Gemma RMSNorm; AdaRMS stays with the caller's reference."""

    supports_adaptive = False
    capabilities = OperatorCapabilities(
        ("cuda",),
        (torch.bfloat16,),
        "gemma_rmsnorm_fp32_reduction",
        rank=3,
        width_multiple=8,
        minimum_sm=(8, 0),
    )

    def __init__(self) -> None:
        """Resolve the optional public norm API without building any shape plan."""
        self._normalize = load_module("flashinfer.norm").gemma_rmsnorm

    def plan(self, inputs: torch.Tensor, *, adaptive: bool, eps: float = 1e-6) -> FlashInferNormPlan:
        """Prepare contiguous BF16 [B,T,D] storage; reject adaptive modulation."""
        if adaptive:
            raise ValueError("FlashInfer Gemma RMSNorm does not implement adaptive normalization")
        self.capabilities.check(OperatorRequest(inputs.device, inputs.dtype, tuple(inputs.shape)))
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm FlashInfer normalization before CUDA Graph capture")
        return FlashInferNormPlan(inputs, eps, self._normalize)


class FlashInferNormPlan(TorchNormPlan):
    """Scoped norm storage retaining separately rounded residual multiplication/addition.

    Scale parameters must be immutable between warmup and graph replay, as with
    projection weight packs. FP32 scales are accepted only when exactly BF16
    representable; no lossy parameter conversion is implicit in this backend.
    """

    def __init__(self, inputs: torch.Tensor, eps: float, normalize: Callable[..., torch.Tensor]) -> None:
        """Allocate outputs before capture; prepare a scale on first eager use."""
        super().__init__(inputs, False, eps)
        self._normalize = normalize
        self._scale: torch.Tensor | None = None
        self._source: torch.Tensor | None = None
        self._version: int | None = None

    def normalize(
        self,
        inputs: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Apply FP32 statistics and Gemma's (1 + scale) with BF16 output."""
        self._validate(inputs)
        if modulation is not None:
            raise ValueError("FlashInfer Gemma RMSNorm does not accept modulation")
        if (
            scale is None
            or scale.shape != (self.shape[-1],)
            or scale.dtype != torch.float32
            or scale.device != inputs.device
            or not scale.is_contiguous()
        ):
            raise ValueError("Gemma RMSNorm requires matching contiguous FP32 scale [D]")
        version = None if scale.is_inference() else scale._version
        capturing = torch.cuda.is_current_stream_capturing()
        changed = self._source is not scale or self._version != version
        if capturing and (changed or self._scale is None):
            raise RuntimeError("Warm the current Gemma RMSNorm scale before capture")
        if not capturing and (changed or version is None):
            converted = scale.detach().to(inputs.dtype)
            if not torch.equal(converted.float(), scale):
                raise ValueError(
                    "FlashInfer Gemma RMSNorm requires FP32 scales exactly representable in BF16; "
                    "select normalization='torch' or 'cuda_strict' for this checkpoint"
                )
            self._scale, self._source, self._version = converted, scale, version
        self._normalize(
            inputs.view(-1, self.shape[-1]),
            self._scale,
            eps=self.eps,
            out=self.output.view(-1, self.shape[-1]),
            enable_pdl=False,
        )
        return self.output, None
