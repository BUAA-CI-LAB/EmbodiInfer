"""Strict RMSNorm affine, BF16 rounding and calibrated format encoding."""

import ctypes

import torch

from ...layers.normalization import NormalizationBackend
from ...layers.quantization import ActivationQuantizer
from ...layers.registry import OperatorCapabilities
from .native import build_library
from .normalization import NormFusion
from .quantization import CudaQuantizer


class NormQuantFusion:
    """Compile single-pass norm epilogues that encode the rounded BF16 values."""

    capabilities = OperatorCapabilities(
        ("cuda",),
        (torch.bfloat16,),
        "torch_rmsnorm_bf16_rounding",
        rank=3,
        width_multiple=64,
        bits=(4, 8),
        minimum_sm_by_bits=((4, (10, 0)),),
        minimum_sm=(8, 0),
    )

    def __init__(self, quantizer: ActivationQuantizer, norm: NormalizationBackend) -> None:
        """Load the cached library for the current CUDA device."""
        if not isinstance(quantizer, CudaQuantizer) or not isinstance(norm, NormFusion):
            raise ValueError("CUDA norm/encoding requires CUDA quantization and strict CUDA normalization")
        if quantizer.device != norm.device:
            raise ValueError("CUDA norm/encoding backends must share a device")
        self.quantizer, self.norm, self.device = quantizer, norm, norm.device
        with torch.cuda.device(self.device):
            built = build_library(
                "norm_quant",
                strict=True,
                specific=True,
                native_fp4=torch.cuda.get_device_capability(self.device)[0] >= 10,
            )
        self.library, self.compiler, self.command = built.library, built.compiler, built.command
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.library.cc_norm_quant.argtypes = [pointer] * 9 + [integer] * 4 + [pointer]
        self.library.cc_norm_quant.restype = integer

    def plan(
        self,
        inputs: torch.Tensor,
        *,
        bits: int,
        calibrated_scale: float,
        adaptive: bool,
    ) -> "NormQuantPlan":
        """Allocate a fixed-shape calibrated encoder for BF16 [B,T,D] activations."""
        return NormQuantPlan(self, inputs, bits=bits, calibrated_scale=calibrated_scale, adaptive=adaptive)


class NormQuantPlan:
    """Strict Torch statistics with fused affine, BF16 rounding, and format encoding."""

    def __init__(
        self,
        backend: NormQuantFusion,
        inputs: torch.Tensor,
        *,
        bits: int,
        calibrated_scale: float,
        adaptive: bool,
    ) -> None:
        """Allocate packed outputs, gate, residual, and FP32 statistics buffers."""
        if (
            inputs.device != backend.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
            or inputs.ndim != 3
            or min(inputs.shape) <= 0
            or inputs.shape[2] % 64
            or bits not in (4, 8)
        ):
            raise ValueError("Norm quantization requires CUDA BF16 [B,T,D], D % 64 == 0, bits 4/8")
        self.backend, self.shape, self.bits, self.adaptive = backend, inputs.shape, bits, adaptive
        self.quantized = backend.quantizer.plan(inputs.reshape(-1, inputs.shape[-1]), bits, calibrated_scale)
        self.output, self.scale, self.scales, self.blocked = (
            self.quantized.output,
            self.quantized.scale,
            self.quantized.scales,
            self.quantized.blocked,
        )
        self.residual = torch.empty_like(inputs)
        self.square = torch.empty_like(inputs, dtype=torch.float32)
        self.mean = torch.empty((*inputs.shape[:2], 1), device=inputs.device, dtype=torch.float32)
        self.gate = (
            torch.empty((inputs.shape[0], 1, inputs.shape[2]), device=inputs.device, dtype=inputs.dtype)
            if adaptive
            else None
        )

    def _validate(self, inputs: torch.Tensor) -> None:
        if (
            inputs.shape != self.shape
            or inputs.device != self.output.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
        ):
            raise ValueError("Norm quantization input shape, layout, dtype, or device changed")

    def normalize(
        self,
        inputs: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
    ) -> "NormQuantPlan":
        """Round the normalized affine result to BF16 in registers, then encode it.

        Args:
            inputs: Contiguous CUDA BF16 [B,T,D] residual activations.
            scale: Contiguous FP32 [D] ordinary RMSNorm scale, without the +1.
            modulation: Contiguous FP32 [B,3D] adaptive scale/shift/gate projection.
            mean: Optional original FP32 mean of squared BF16 input values.

        Returns:
            This ephemeral workspace, matching QuantizationPlan's packed format.
        """
        self._validate(inputs)
        affine = modulation if self.adaptive else scale
        affine_shape = (self.shape[0], self.shape[2] * 3) if self.adaptive else (self.shape[2],)
        if (
            affine is None
            or affine.shape != affine_shape
            or affine.dtype != torch.float32
            or affine.device != inputs.device
            or not affine.is_contiguous()
        ):
            raise ValueError("Norm quantization requires contiguous FP32 scale or raw modulation")
        if mean is None:
            torch.square(inputs.float(), out=self.square)
            torch.mean(self.square, dim=-1, keepdim=True, out=self.mean)
            mean = self.mean
        if (
            mean.shape != (*self.shape[:2], 1)
            or mean.dtype != torch.float32
            or mean.device != inputs.device
            or not mean.is_contiguous()
        ):
            raise ValueError("Norm quantization requires original FP32 [B,T,1] mean statistics")
        batch, tokens, columns = self.shape
        with torch.cuda.device(inputs.device):
            self.backend.norm._check(
                self.backend.library.cc_norm_quant(
                    inputs.data_ptr(),
                    mean.data_ptr(),
                    scale.data_ptr() if scale is not None else None,
                    modulation.data_ptr() if modulation is not None else None,
                    self.output.data_ptr(),
                    self.scales.data_ptr(),
                    self.blocked.data_ptr(),
                    self.gate.data_ptr() if self.gate is not None else None,
                    self.scale.data_ptr(),
                    batch * tokens,
                    columns,
                    tokens,
                    self.bits,
                    torch.cuda.current_stream(inputs.device).cuda_stream,
                )
            )
        return self

    def residual_normalize(
        self,
        inputs: torch.Tensor,
        update: torch.Tensor,
        residual_gate: torch.Tensor | None,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, "NormQuantPlan", torch.Tensor | None]:
        """Preserve both residual BF16 rounding boundaries before strict norm encoding."""
        self._validate(inputs)
        self._validate(update)
        if residual_gate is not None and (
            residual_gate.shape != (self.shape[0], 1, self.shape[2])
            or residual_gate.dtype != torch.bfloat16
            or residual_gate.device != inputs.device
            or not residual_gate.is_contiguous()
        ):
            raise ValueError("Residual gate must be contiguous CUDA BF16 [B,1,D]")
        batch, tokens, columns = self.shape
        with torch.cuda.device(inputs.device):
            self.backend.norm._check(
                self.backend.norm.library.cc_residual_square(
                    inputs.data_ptr(),
                    update.data_ptr(),
                    residual_gate.data_ptr() if residual_gate is not None else None,
                    self.residual.data_ptr(),
                    self.square.data_ptr(),
                    batch * tokens,
                    columns,
                    tokens,
                    torch.cuda.current_stream(inputs.device).cuda_stream,
                )
            )
        torch.mean(self.square, dim=-1, keepdim=True, out=self.mean)
        self.normalize(self.residual, scale=scale, modulation=modulation, mean=self.mean)
        return self.residual, self, self.gate
