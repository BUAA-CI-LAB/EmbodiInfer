"""Calibrated projection adapters retaining the original checkpoint parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from ...layers.quantization import EncodedActivation, WorkspaceKey
from .optimization_config import ActionLayerPrecision

if TYPE_CHECKING:
    from .optimization import Pi05Optimizations


class MlpPlan:
    """Keep original projection parameters and stream-specific activation buffers."""

    def __init__(
        self, runtime: Pi05Optimizations, module: Any, layer: ActionLayerPrecision, *, prefix: bool
    ) -> None:
        """Pack only selected formats and preserve the adapter's tanh GELU."""
        self.runtime, self.module, self.layer, self.prefix = runtime, module, layer, prefix
        projections = (module.gate_proj, module.up_proj, module.down_proj)
        if any(
            projection.weight.dtype != torch.bfloat16 or projection.bias is not None
            for projection in projections
        ):
            raise ValueError("Optimized Pi05 MLPs require bias-free BF16 projection weights")
        if module.gate_proj.weight.shape != module.up_proj.weight.shape or (
            module.down_proj.weight.shape != tuple(reversed(module.gate_proj.weight.shape))
        ):
            raise ValueError("Optimized Pi05 MLP projection dimensions do not match")
        self.gate = runtime.projection.plan(
            module.gate_proj.weight,
            layer.gate_up_max,
            layer.gate_up,
            runtime.quantizer,
            runtime.workspace_key,
        )
        self.up = runtime.projection.plan(
            module.up_proj.weight, layer.gate_up_max, layer.gate_up, runtime.quantizer, runtime.workspace_key
        )
        self.down = runtime.projection.plan(
            module.down_proj.weight, layer.down_max, layer.down, runtime.quantizer, runtime.workspace_key
        )
        self.paired = None
        if runtime.config.fused_mlp and layer.gate_up == "bf16":
            self.paired = runtime.paired[prefix].plan(module.gate_proj.weight, module.up_proj.weight)
        self.plans = {}

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release only completed graph workspaces through the projection contract."""
        for projection in (self.gate, self.up, self.down):
            projection.release_scope(scope)
        for key in tuple(self.plans):
            if key[-1] == scope:
                del self.plans[key]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Fuse paired BF16 projections or share one calibrated gate/up encoding."""
        if self.paired is not None:
            matrix = inputs.reshape(-1, inputs.shape[-1]).contiguous()
            hidden = self.paired(matrix).reshape(*inputs.shape[:-1], -1)
            return self.down.apply(hidden, self.down.encode(hidden))
        return self.project(inputs, self.gate.encode(inputs))

    def project(self, inputs: torch.Tensor, encoded: EncodedActivation | None) -> torch.Tensor:
        """Fuse tanh GELU/product/down encoding after native selected-precision GEMMs."""
        gate, up = self.gate.apply(inputs, encoded), self.up.apply(inputs, encoded)
        if self.runtime.fusion is None:
            hidden = F.gelu(gate, approximate="tanh") * up
            return self.down.apply(hidden, self.down.encode(hidden))
        matrix = gate.reshape(-1, gate.shape[-1]).contiguous()
        key = (tuple(matrix.shape), matrix.device, self.runtime.workspace_key(matrix.device))
        if key not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm GELU/product/quantization shapes before graph capture")
            bits = {"bf16": 16, "fp8": 8, "nvfp4": 4}[self.down.precision]
            scale = None if bits == 16 else self.down.maximum / (448 * (6 if bits == 4 else 1))
            self.plans[key] = self.runtime.fusion.plan(matrix, bits, scale)
        result = self.plans[key].encode(matrix, up.reshape_as(matrix).contiguous())
        if self.down.precision == "bf16":
            return self.down.apply(result.output.reshape_as(gate), None)
        return self.down.apply(gate, result)
