"""Model-independent calibrated projections using Torch native GEMMs."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from ...layers.quantization import ActivationQuantizer, EncodedActivation, Precision, WorkspaceKey
from ...layers.registry import OperatorCapabilities


class Projection:
    """Pack one selected weight format and reuse calibrated activation workspaces."""

    def __init__(
        self,
        weight: torch.Tensor,
        maximum: float,
        precision: Precision,
        backend: ActivationQuantizer | None,
        workspace_key: Callable[[torch.device], WorkspaceKey] | None = None,
    ) -> None:
        """Preserve the validated weight quantizer and native GEMM layouts."""
        if precision not in ("bf16", "fp8", "nvfp4"):
            raise ValueError("Projection precision must be bf16, fp8 or nvfp4")
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("Calibrated projections require a BF16 [output,input] weight")
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Calibrated projection maximum must be finite and positive")
        if precision != "bf16" and (backend is None or weight.device.type != "cuda"):
            raise ValueError("Low precision projections require CUDA and an activation quantizer")
        if precision != "bf16" and not all(hasattr(F, name) for name in ("scaled_mm", "ScalingType")):
            raise RuntimeError("Calibrated projections require the scaled_mm API tested in Torch 2.13")
        if precision != "bf16":
            minimum_sm = (10, 0) if precision == "nvfp4" else (8, 9)
            if torch.cuda.get_device_capability(weight.device) < minimum_sm:
                raise ValueError(f"{precision} native GEMM requires SM {minimum_sm} or newer")
        self.precision, self.maximum, self.backend = precision, maximum, backend
        self.workspace_key = workspace_key or (
            lambda device: ("stream", torch.cuda.current_stream(device).cuda_stream)
        )
        self.method = "bf16" if precision == "bf16" else f"calibrated_{precision}"
        self.output_dim = weight.shape[0]
        self.input_dim, self.device = weight.shape[1], weight.device
        self.plans = {}
        if precision == "bf16":
            self.bf16_weight = weight.detach()
        elif precision == "fp8":
            values = weight.detach().contiguous().float()
            maximum_weight = values.abs().amax()
            self.weight_scale = torch.where(
                maximum_weight > 0, maximum_weight / 448.0, torch.ones_like(maximum_weight)
            )
            self.weight = (values / self.weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        else:
            plan = backend.plan(weight.detach().contiguous(), 4).quantize(weight.detach().contiguous())
            self.weight, self.weight_scale, self.weight_blocks = (
                plan.output,
                plan.scale,
                plan.blocked,
            )

    def encode(self, inputs: torch.Tensor) -> EncodedActivation | None:
        """Encode activations at the fixed calibrated scale, sharing gate/up inputs."""
        if self.precision == "bf16":
            return None
        matrix = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        key = (tuple(matrix.shape), matrix.device, self.workspace_key(matrix.device))
        if key not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm projection encoders before graph capture")
            bits = 8 if self.precision == "fp8" else 4
            self.plans[key] = self.backend.plan(matrix, bits, self.maximum / (448 * (6 if bits == 4 else 1)))
        return self.plans[key].quantize(matrix)

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Drop only the completed graph's activation encoding workspaces."""
        for key in tuple(self.plans):
            if key[-1] == scope:
                del self.plans[key]

    def apply(self, inputs: torch.Tensor, encoded: EncodedActivation | None) -> torch.Tensor:
        """Run the selected native GEMM with BF16 outputs and fused FP4 global scales."""
        if (
            inputs.shape[-1] != self.input_dim
            or inputs.device != self.device
            or inputs.dtype != torch.bfloat16
        ):
            raise ValueError("Projection input width, dtype or device changed")
        if self.precision != "bf16" and (encoded is None or encoded.scale is None):
            raise ValueError("Low precision projections require encoded activations and a global scale")
        shape = (*inputs.shape[:-1], self.output_dim)
        matrix = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        if self.precision == "bf16":
            output = F.linear(matrix, self.bf16_weight)
        elif self.precision == "fp8":
            output = F.scaled_mm(
                encoded.output,
                self.weight.t(),
                encoded.scale,
                F.ScalingType.TensorWise,
                self.weight_scale,
                F.ScalingType.TensorWise,
                output_dtype=torch.bfloat16,
            )
        else:
            output = F.scaled_mm(
                encoded.output.view(torch.float4_e2m1fn_x2),
                self.weight.t().view(torch.float4_e2m1fn_x2),
                [encoded.blocked, encoded.scale],
                [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                [self.weight_blocks, self.weight_scale],
                [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                output_dtype=torch.bfloat16,
            )
        return output.reshape(shape)


class CalibratedProjectionBackend:
    """Prepare bias-free BF16/FP8/NVFP4 projection weights without model knowledge."""

    capabilities = OperatorCapabilities(("cpu", "cuda"), (torch.bfloat16,), "native_gemm_bf16_output")

    def plan(
        self,
        weight: torch.Tensor,
        maximum: float,
        precision: Precision,
        quantizer: ActivationQuantizer | None,
        workspace_key: Callable[[torch.device], WorkspaceKey],
    ) -> Projection:
        """Create weight packs and execution-scoped encoders before capture."""
        return Projection(weight, maximum, precision, quantizer, workspace_key)
