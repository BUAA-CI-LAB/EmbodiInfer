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


@dataclass
class _Fp4Activation:
    output: torch.Tensor
    blocked: torch.Tensor
    scale: torch.Tensor
    scales: None = None
    packed_bf16: torch.Tensor | None = None


@dataclass
class _Bf16Activation:
    output: torch.Tensor


def smoothed_norm_scale(weight: torch.Tensor, smooth: torch.Tensor) -> torch.Tensor:
    """FP32 RMSNorm scale (applied as ``1 + w``) whose output is divided by the SmoothQuant scales."""
    return ((1.0 + weight.float()) * (1.0 / smooth) - 1.0).contiguous()


def smoothed_adarms(
    weight: torch.Tensor, bias: torch.Tensor, smooth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """AdaRMS ``[scale, shift, gate]`` projection whose normalized output is divided by the scales.

    ``x * (1 + scale) + shift`` becomes ``x * (1 + scale') + shift'`` with ``1 + scale' = (1 + scale) / s``
    and ``shift' = shift / s``; the residual gate rows are unchanged.
    """
    inverse = 1.0 / smooth
    width = inverse.numel()
    folded_weight, folded_bias = weight.float().clone(), bias.float().clone()
    folded_weight[:width] *= inverse[:, None]
    folded_weight[width : 2 * width] *= inverse[:, None]
    folded_bias[:width] = folded_bias[:width] * inverse + (inverse - 1.0)
    folded_bias[width : 2 * width] *= inverse
    return folded_weight.to(weight.dtype), folded_bias.to(bias.dtype)


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
        gate_weight, up_weight, down_weight = (projection.weight.detach() for projection in projections)
        if layer.smooth is not None:
            smooth = torch.tensor(layer.smooth, dtype=torch.float32, device=gate_weight.device)
            if smooth.numel() != gate_weight.shape[1]:
                raise ValueError("SmoothQuant scales must match the MLP input width")
            gate_weight = (gate_weight.float() * smooth).to(gate_weight.dtype)
            up_weight = (up_weight.float() * smooth).to(up_weight.dtype)
        self.side_gate_up = self.side_down = None
        if layer.outliers is not None:
            if not prefix or runtime.config.hardware != "thor":
                raise ValueError("BF16 outlier channels require a Thor prefix NVFP4 MLP")
            index = torch.tensor(layer.outliers, dtype=torch.long, device=gate_weight.device)
            if int(index.max()) >= gate_weight.shape[0]:
                raise ValueError("Outlier channels exceed the MLP width")
            # Kept channels of the NVFP4 path no longer share block scales with the outliers.
            self.side_gate_up = torch.cat((gate_weight[index], up_weight[index]), dim=0).t().contiguous()
            self.side_down = down_weight[:, index].t().contiguous()
            gate_weight, up_weight, down_weight = (w.clone() for w in (gate_weight, up_weight, down_weight))
            gate_weight[index] = 0
            up_weight[index] = 0
            down_weight[:, index] = 0
        self.gate = runtime.projection.plan(
            gate_weight,
            layer.gate_up_max,
            layer.gate_up,
            runtime.quantizer,
            runtime.workspace_key,
        )
        self.up = runtime.projection.plan(
            up_weight, layer.gate_up_max, layer.gate_up, runtime.quantizer, runtime.workspace_key
        )
        self.down = runtime.projection.plan(
            down_weight, layer.down_max, layer.down, runtime.quantizer, runtime.workspace_key
        )
        self.paired = None
        self.packed_fp8 = None
        if runtime.config.fused_mlp and layer.gate_up == "bf16":
            self.paired = runtime.paired[prefix].plan(gate_weight, up_weight)
        elif runtime.config.hardware in ("thor", "4090") and layer.gate_up == "fp8":
            packed_w = torch.cat((gate_weight, up_weight), dim=0).contiguous()
            self.packed_fp8 = runtime.projection.plan(
                packed_w, layer.gate_up_max, "fp8", runtime.quantizer, runtime.workspace_key
            )
            self.inter_size = gate_weight.shape[0]
            if layer.down == "fp8":
                self.down_scale = torch.tensor(
                    [layer.down_max / 448.0], device=gate_weight.device, dtype=torch.float32
                )
                self.down_inv_scale = torch.tensor(
                    [448.0 / layer.down_max], device=gate_weight.device, dtype=torch.float32
                )
                if not prefix:
                    self.g_nk_fp8 = self.gate.weight
                    self.u_nk_fp8 = self.up.weight
                elif runtime.config.hardware == "thor":
                    self.g_kn_fp8 = self.gate.weight.t().contiguous()
                    self.u_kn_fp8 = self.up.weight.t().contiguous()
        self.cudnn_nvfp4 = False
        if runtime.config.hardware == "thor" and layer.gate_up == "nvfp4":
            self.cudnn_nvfp4 = True
            self.inter_size = self.gate.weight.shape[0]
            device = self.gate.weight.device
            a_global = layer.gate_up_max / (448.0 * 6.0)
            self.cudnn_g_scale = (self.gate.weight_scale.reshape(1).float() * a_global).contiguous()
            self.cudnn_u_scale = (self.up.weight_scale.reshape(1).float() * a_global).contiguous()
            self.cudnn_down_global_fp4 = torch.tensor(
                [layer.down_max / (448.0 * 6.0)], device=device, dtype=torch.float32
            )
            self.cudnn_down_inv_fp4 = torch.tensor(
                [(448.0 * 6.0) / layer.down_max], device=device, dtype=torch.float32
            )
            self.cudnn_down_scale_fp8 = torch.tensor(
                [layer.down_max / 448.0], device=device, dtype=torch.float32
            )
            self.cudnn_down_inv_fp8 = torch.tensor(
                [448.0 / layer.down_max], device=device, dtype=torch.float32
            )
            self.cudnn_w_packed = None
            if not prefix and layer.down == "nvfp4" and self.inter_size % 128 == 0:
                self.cudnn_w_packed = torch.cat((self.gate.weight, self.up.weight), dim=0).contiguous()
                self.cudnn_wb_packed = torch.cat(
                    (self.gate.weight_blocks, self.up.weight_blocks), dim=0
                ).contiguous()
                self.cudnn_u_ratio = float(self.up.weight_scale.item() / self.gate.weight_scale.item())
            self.cutlass_weights = None
            if prefix and layer.down == "nvfp4":
                from ....backend.cuda import cutlass_geglu

                if cutlass_geglu.available(device) and self.inter_size % 128 == 0:
                    # The fused epilogue reads each feature's gate and up from accumulator rows 8 apart.
                    self.cutlass_weights = cutlass_geglu.interleave_gate_up(
                        self.gate.weight, self.gate.weight_blocks, self.up.weight, self.up.weight_blocks
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

    def project(
        self, inputs: torch.Tensor, encoded: EncodedActivation | None, *, norm: Any = None
    ) -> torch.Tensor:
        """Fuse the selected GELU/product/down encoding after native GEMMs.

        ``inputs`` is the residual that ``norm`` normalized into ``encoded``; layers with BF16
        outlier channels need both to add those channels' contribution.
        """
        output = self._project(inputs, encoded)
        if self.side_down is not None:
            self._add_outliers(inputs, encoded, norm, output)
        return output

    def _add_outliers(self, inputs: torch.Tensor, encoded: Any, norm: Any, output: torch.Tensor) -> None:
        """Accumulate the outlier channels' BF16 GeGLU and down projection into ``output``."""
        if norm is None or encoded is None:
            raise RuntimeError("BF16 outlier channels need the residual and norm of the fused NVFP4 input")
        rows = encoded.output.shape[0]
        width = self.side_down.shape[0]
        key = ("outliers", rows, inputs.device, self.runtime.workspace_key(inputs.device))
        if key not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm the BF16 outlier channels before graph capture")
            self.plans[key] = (
                torch.empty((rows, 2 * width), device=inputs.device, dtype=torch.bfloat16),
                torch.empty((rows, width), device=inputs.device, dtype=torch.bfloat16),
            )
        gate_up, hidden = self.plans[key]
        normalized = self.runtime.smoothed_normalize(norm, inputs)
        torch.matmul(normalized.reshape(-1, normalized.shape[-1])[:rows], self.side_gate_up, out=gate_up)
        torch.mul(F.gelu(gate_up[:, :width]), gate_up[:, width:], out=hidden)
        output.view(-1, output.shape[-1]).addmm_(hidden, self.side_down)

    def _project(self, inputs: torch.Tensor, encoded: EncodedActivation | None) -> torch.Tensor:
        if getattr(self, "g_kn_fp8", None) is not None and self.runtime.fusion is not None:
            from ....backend.triton.geglu import prefix_geglu_fp8

            rows = inputs.shape[0] * inputs.shape[1]
            key = (
                "triton_prefix_geglu_fp8",
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
            prefix_geglu_fp8(
                encoded.output,
                self.g_kn_fp8,
                self.u_kn_fp8,
                self.runtime.fusion.table,
                encoded.scale,
                self.gate.weight_scale_1d,
                self.up.weight_scale_1d,
                self.down_inv_scale,
                entry.output,
                rows,
            )
            return self.down.apply(inputs, entry)
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
        if (
            self.cudnn_nvfp4
            and encoded is not None
            and encoded.blocked is not None
            and self.runtime.fusion is not None
        ):
            from ....backend.cuda.cudnn_mlp import cudnn_dual_geglu

            rows = encoded.output.shape[0]
            key = (
                "cudnn_nvfp4_geglu",
                rows,
                self.inter_size,
                inputs.device,
                self.runtime.workspace_key(inputs.device),
            )
            if key not in self.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm GELU/product/quantization shapes before graph capture")
                if self.down.precision == "nvfp4":
                    if self.cutlass_weights is not None and rows > 16:
                        output, blocked = self.runtime.prefix_geglu_buffers(
                            rows, self.inter_size, inputs.device
                        )
                    else:
                        output = torch.empty(
                            (rows, self.inter_size // 2), device=inputs.device, dtype=torch.uint8
                        )
                        blocked = torch.zeros(
                            -(-rows // 128) * 128 * (self.inter_size // 16),
                            device=inputs.device,
                            dtype=torch.float8_e4m3fn,
                        )
                    entry = _Fp4Activation(
                        output=output,
                        blocked=blocked,
                        scale=self.cudnn_down_global_fp4,
                        packed_bf16=(
                            torch.empty(
                                (rows, 2 * self.inter_size), device=inputs.device, dtype=torch.bfloat16
                            )
                            if self.cudnn_w_packed is not None
                            else None
                        ),
                    )
                elif self.down.precision == "fp8":
                    entry = _Fp8Activation(
                        output=torch.empty(
                            (rows, self.inter_size), device=inputs.device, dtype=torch.float8_e4m3fn
                        ),
                        scale=self.cudnn_down_scale_fp8,
                    )
                else:
                    entry = _Bf16Activation(
                        output=torch.empty(
                            (rows, self.inter_size), device=inputs.device, dtype=torch.bfloat16
                        )
                    )
                self.plans[key] = entry
            entry = self.plans[key]
            if self.down.precision == "nvfp4":
                if self.cudnn_w_packed is not None and entry.packed_bf16 is not None and rows <= 16:
                    from ....backend.cuda.cudnn_mlp import cudnn_packed_geglu_fp4

                    cudnn_packed_geglu_fp4(
                        encoded.output,
                        encoded.blocked,
                        self.cudnn_w_packed,
                        self.cudnn_wb_packed,
                        self.cudnn_g_scale,
                        self.cudnn_u_ratio,
                        entry.packed_bf16,
                        entry.output,
                        entry.blocked,
                        entry.scale,
                        self.runtime.fusion.table,
                    )
                elif self.cutlass_weights is not None and rows > 16:
                    from ....backend.cuda.cutlass_geglu import cutlass_geglu_fp4

                    cutlass_geglu_fp4(
                        encoded.output,
                        encoded.blocked,
                        *self.cutlass_weights,
                        self.cudnn_g_scale,
                        self.cudnn_u_scale,
                        self.cudnn_down_inv_fp4,
                        entry.output,
                        entry.blocked,
                    )
                else:
                    cudnn_dual_geglu(
                        encoded.output,
                        encoded.blocked,
                        self.gate.weight,
                        self.gate.weight_blocks,
                        self.up.weight,
                        self.up.weight_blocks,
                        self.cudnn_g_scale,
                        self.cudnn_u_scale,
                        self.cudnn_down_inv_fp4,
                        entry.output,
                        entry.blocked,
                        4,
                    )
                return self.down.apply(inputs, entry)
            if self.down.precision == "fp8":
                cudnn_dual_geglu(
                    encoded.output,
                    encoded.blocked,
                    self.gate.weight,
                    self.gate.weight_blocks,
                    self.up.weight,
                    self.up.weight_blocks,
                    self.cudnn_g_scale,
                    self.cudnn_u_scale,
                    self.cudnn_down_inv_fp8,
                    entry.output,
                    None,
                    8,
                )
                return self.down.apply(inputs, entry)
            cudnn_dual_geglu(
                encoded.output,
                encoded.blocked,
                self.gate.weight,
                self.gate.weight_blocks,
                self.up.weight,
                self.up.weight_blocks,
                self.cudnn_g_scale,
                self.cudnn_u_scale,
                None,
                entry.output,
                None,
                16,
            )
            return self.down.apply(entry.output.reshape(*inputs.shape[:-1], self.inter_size), None)
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
