"""Calibrated MLP execution from checkpoint parameters or offline weight archives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from ....layers.quantization import EncodedActivation, WorkspaceKey
from .config import MlpLayerPrecision

if TYPE_CHECKING:
    from .operators import Pi05OperatorPlans


@dataclass
class _Fp8Activation:
    output: torch.Tensor
    scale: torch.Tensor
    scales: None = None
    blocked: None = None


class CalibratedMlp(torch.nn.Module):
    """MLP dimensions backed by the checkpoint-bound external weight archive.

    No original BF16 MLP tensors are retained. The policy controller prepares
    compressed execution weights on the final device. This module deliberately
    has no eager/training fallback or independently loadable state dictionary.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Retain validated projection dimensions without materializing weights."""
        super().__init__()
        self.shape = (intermediate_size, hidden_size)

    @classmethod
    def from_module(cls, module: Any) -> CalibratedMlp:
        """Validate the original checkpoint layout before releasing BF16 parameters."""
        if isinstance(module, cls):
            return module
        shape = tuple(module.gate_proj.weight.shape)
        if module.up_proj.weight.shape != shape or module.down_proj.weight.shape != tuple(reversed(shape)):
            raise ValueError("Calibrated MLPs require matching gate/up/down dimensions")
        if any(p.bias is not None for p in (module.gate_proj, module.up_proj, module.down_proj)):
            raise ValueError("Calibrated MLP projections must be bias-free")
        return cls(shape[1], shape[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reject execution outside the prepared checkpoint-bound policy plans."""
        raise RuntimeError("Calibrated MLPs require their policy's prepared execution plans")


class MlpPlan:
    """Keep original projection parameters and stream-specific activation buffers."""

    def __init__(
        self, runtime: Pi05OperatorPlans, module: Any, layer: MlpLayerPrecision, *, prefix: bool
    ) -> None:
        """Pack only selected formats and preserve the declared GELU contract."""
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
        self.packed_fp8 = None
        if runtime.config.fused_mlp and layer.gate_up == "bf16":
            self.paired = runtime.paired[prefix].plan(module.gate_proj.weight, module.up_proj.weight)
        elif runtime.config.hardware == "4090" and layer.gate_up == "fp8":
            packed_w = torch.cat(
                (module.gate_proj.weight.detach(), module.up_proj.weight.detach()), dim=0
            ).contiguous()
            self.packed_fp8 = runtime.projection.plan(
                packed_w, layer.gate_up_max, "fp8", runtime.quantizer, runtime.workspace_key
            )
            self.inter_size = module.gate_proj.weight.shape[0]
            if not prefix and layer.down == "fp8":
                self.g_nk_fp8 = self.gate.weight
                self.u_nk_fp8 = self.up.weight
                self.down_scale = torch.tensor(
                    [layer.down_max / 448.0], device=module.gate_proj.weight.device, dtype=torch.float32
                )
                self.down_inv_scale = torch.tensor(
                    [448.0 / layer.down_max], device=module.gate_proj.weight.device, dtype=torch.float32
                )
        self.plans = {}

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release only completed graph workspaces through the projection contract."""
        for projection in (self.gate, self.up, self.down):
            projection.release_scope(scope)
        if self.packed_fp8 is not None:
            self.packed_fp8.release_scope(scope)
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
        """Fuse the selected GELU/product/down encoding after native GEMMs."""
        if getattr(self, "g_nk_fp8", None) is not None and inputs.shape[0] * inputs.shape[1] <= 64:
            from ....backend.triton.geglu import dual_geglu_fp8

            rows = inputs.shape[0] * inputs.shape[1]
            key = (
                "triton_geglu_fp8",
                rows,
                self.inter_size,
                inputs.device,
                self.runtime.workspace_key(inputs.device),
            )
            if key not in self.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm GELU/product/quantization shapes before graph capture")
                out_fp8 = torch.empty(
                    (rows, self.inter_size), device=inputs.device, dtype=torch.float8_e4m3fn
                )
                self.plans[key] = _Fp8Activation(output=out_fp8, scale=self.down_scale)
            entry = self.plans[key]
            dual_geglu_fp8(
                encoded.output,
                self.g_nk_fp8,
                self.u_nk_fp8,
                encoded.scale,
                self.gate.weight_scale_1d,
                self.up.weight_scale_1d,
                self.down_inv_scale,
                entry.output,
                rows,
            )
            return self.down.apply(inputs, entry)
        if self.packed_fp8 is not None and self.runtime.fusion is not None:
            packed = self.packed_fp8.apply(inputs, encoded).reshape(-1, 2 * self.inter_size)
            shape = (packed.shape[0], self.inter_size)
            key = (shape, packed.device, self.runtime.workspace_key(packed.device))
            if key not in self.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm GELU/product/quantization shapes before graph capture")
                bits = {"bf16": 16, "fp8": 8, "nvfp4": 4}[self.down.precision]
                scale = None if bits == 16 else self.down.maximum / (448 * (6 if bits == 4 else 1))
                template = packed[:, : self.inter_size].contiguous()
                self.plans[key] = self.runtime.fusion.plan(template, bits, scale)
            result = self.plans[key].encode_packed(packed)
            out_shape = (*inputs.shape[:-1], self.inter_size)
            if self.down.precision == "bf16":
                return self.down.apply(result.output.reshape(out_shape), None)
            return self.down.apply(inputs, result)
        gate, up = self.gate.apply(inputs, encoded), self.up.apply(inputs, encoded)
        if self.runtime.fusion is None:
            hidden = F.gelu(gate, approximate=self.runtime.approximate) * up
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


class WeightOnlyMlpPlan:
    """Frozen calibrated W4 weights with BF16 activations and exact rounded GELU."""

    def __init__(self, runtime: Pi05OperatorPlans, module: Any, index: int, *, expert: bool) -> None:
        """Load two checkpoint-bound shards and pack without requantizing them."""
        gate_shape = module.shape
        down_shape = tuple(reversed(gate_shape))
        self.runtime, self.expert = runtime, expert
        codes, scales = runtime.weight_archive.load(
            index, expert=expert, projection="gate_up", shape=(2 * gate_shape[0], gate_shape[1])
        )
        self.gate_up = runtime.weight_only.plan(
            codes,
            scales,
            group_size=64,
            workspace_key=runtime.workspace_key,
            fused_gelu=True,
            large_m=not expert,
            blocks=16 if not expert else None,
        )
        codes, scales = runtime.weight_archive.load(index, expert=expert, projection="down", shape=down_shape)
        self.down = runtime.weight_only.plan(
            codes,
            scales,
            group_size=64,
            workspace_key=runtime.workspace_key,
            blocks=4 if expert else None,
        )

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply calibrated gate/up, rounded GeGLU and the down projection."""
        return self.down(self.gate_up(inputs))

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Drop private projection locks after their graph completes."""
        self.gate_up.release_scope(scope)
        self.down.release_scope(scope)


class IntegerMlpPlan:
    """Frozen prefix W8A8 projections with the original rounded exact GeGLU."""

    def __init__(self, runtime: Pi05OperatorPlans, module: Any, index: int) -> None:
        """Load calibrated codes and transforms through the integer backend contract."""
        shape = module.shape
        archive = runtime.weight_archive.integer_prefix
        self.gate_up = runtime.integer_projection.plan(
            *archive.load(index, projection="gate_up", shape=(2 * shape[0], shape[1])),
            workspace_key=runtime.workspace_key,
            fused_gelu=True,
        )
        self.down = runtime.integer_projection.plan(
            *archive.load(index, projection="down", shape=tuple(reversed(shape))),
            workspace_key=runtime.workspace_key,
        )

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run both native integer projections including their activation transforms."""
        return self.down(self.gate_up(inputs))

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release completed graph workspaces in both projection plans."""
        self.gate_up.release_scope(scope)
        self.down.release_scope(scope)
