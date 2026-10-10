"""Instance-local Pi05 execution plans for fused operators."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ....layers import (
    OperatorRequest,
    gelu_mul_backends,
    get_attention_backend,
    norm_quant_backends,
    normalization_backends,
    paired_gelu_backends,
    projection_backends,
    quantization_backends,
    rotary_backends,
)
from ....layers.activation import GeluMulBackend, PairedGeluBackend
from ....layers.normalization import NormalizationBackend, NormalizationPlan, NormQuantBackend, NormQuantPlan
from ....layers.quantization import ActivationQuantizer, ProjectionBackend
from ..embeddings import rlinf_rope_tables, rope_tables
from .config import MlpLayerPrecision, Pi05OptimizationConfig

if TYPE_CHECKING:
    from ..modeling_pi05 import Pi05Policy, Pi05Prefix


def _stream_key(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


class Pi05OperatorPlans:
    """Own weight packs and stream-specific tensor storage without replacing modules.

    Construct outside capture after the engine has placed the model. Storage is
    reused within one stream; the policy runtime serializes native encode/decode
    calls. Destroy graphs before clearing these plans after a weight/device change.
    """

    @torch.no_grad()
    def __init__(self, policy: Pi05Policy, config: Pi05OptimizationConfig) -> None:
        """Validate precision boundaries and prepare only selected CUDA operators."""
        self.policy, self.config = policy, config
        self.approximate = "none" if config.activation == "gelu_pytorch_exact" else "tanh"
        self.action_context = None
        self.vision = None
        device = policy._m.action_in_proj.weight.device
        if device.type != "cuda":
            raise ValueError("Fused Pi05 inference requires CUDA")
        if config.capability is not None and torch.cuda.get_device_capability(device) != config.capability:
            raise ValueError("GPU capability does not match the selected Pi05 launch profile")
        if policy.execution_dtype != torch.float32:
            raise ValueError("Pi05 fused inference preserves FP32 action/time projections")
        for tower in (policy._prefix_tower, policy._expert_tower):
            if any(layer.self_attn.q_proj.weight.dtype != torch.bfloat16 for layer in tower.layers):
                raise ValueError("Fused Pi05 operators require BF16 transformer weights")
            for norm in [tower.norm] + [
                n for layer in tower.layers for n in (layer.input_layernorm, layer.post_attention_layernorm)
            ]:
                if norm.eps != 1.0e-6 or any(
                    parameter.dtype != torch.float32 for parameter in norm.parameters()
                ):
                    raise ValueError("Strict fused RMSNorm requires eps=1e-6 and FP32 norm parameters")
        for name, tower in (("action", policy._expert_tower), ("prefix", policy._prefix_tower)):
            layers = getattr(config, f"{name}_layers")
            if layers and len(layers) != len(tower.layers):
                raise ValueError(f"Specify exactly one {name} precision pair per transformer layer")
        precision_layers = (*config.action_layers, *config.prefix_layers)
        if any("nvfp4" in (layer.gate_up, layer.down) for layer in precision_layers) and (
            torch.cuda.get_device_capability(device)[0] < 10
        ):
            raise ValueError("NVFP4 MLP projections require a Blackwell CUDA device")
        if config.fused_mlp and config.hardware in ("thor", "spark"):
            for tower, width, intermediate in (
                (policy._prefix_tower, 2048, 16384),
                (policy._expert_tower, 1024, 4096),
            ):
                if len(tower.layers) != 18 or any(
                    layer.mlp.gate_proj.weight.shape != (intermediate, width) for layer in tower.layers
                ):
                    raise ValueError("Thor/Spark launch profiles require standard Pi05 tower dimensions")
        if config.checkpoint_sha256 is not None:
            checkpoint = Path(policy.checkpoint or "")
            checkpoint = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
            if not checkpoint.is_file():
                raise ValueError("Checkpoint-bound recipes require a local model.safetensors file")
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != config.checkpoint_sha256:
                raise ValueError("Activation calibration belongs to a different checkpoint")
        if config.mixed_precision and not all(hasattr(F, name) for name in ("scaled_mm", "ScalingType")):
            raise RuntimeError("Calibrated mixed precision requires the scaled_mm API tested in PyTorch 2.13")
        selected = config.operators
        request = OperatorRequest(device, torch.bfloat16, cuda_graph=True)
        self.rotary = rotary_backends.get(selected.rotary, request) if config.numerics == "rlinf" else None
        self.norm: NormalizationBackend | None = (
            normalization_backends.get(selected.normalization, request) if config.norm_fusion else None
        )
        self.quantizer: ActivationQuantizer | None = None
        self.fusion: GeluMulBackend | None = None
        self.norm_quant: NormQuantBackend | None = None
        self.projection: ProjectionBackend | None = None
        self.paired: dict[bool, PairedGeluBackend] = {}
        if self.norm is not None and self.norm.capabilities.arithmetic != "torch_rmsnorm_bf16_rounding":
            raise ValueError("Pi05 requires the strict Torch RMSNorm rounding contract")
        if config.fused_mlp or config.mixed_precision:
            self.projection = projection_backends.get(selected.projection, request)
        if config.mixed_precision:
            bits = 4 if any("nvfp4" in (layer.gate_up, layer.down) for layer in precision_layers) else 8
            encoding_request = OperatorRequest(device, torch.bfloat16, bits=bits, cuda_graph=True)
            self.quantizer = quantization_backends.get(selected.quantization, encoding_request)
            self.fusion = gelu_mul_backends.get(
                selected.gelu_mul, request, backend=self.quantizer, approximate=self.approximate
            )
            if (
                self.quantizer.capabilities.arithmetic != "rounded_bf16_encoding"
                or self.fusion.capabilities.arithmetic != "torch_gelu_product_bf16_rounding"
            ):
                raise ValueError("Pi05 requires the rounded BF16 activation encoding and GELU contracts")
            if config.norm_fusion and selected.norm_quant is not None:
                self.norm_quant = norm_quant_backends.get(
                    selected.norm_quant, request, quantizer=self.quantizer, norm=self.norm
                )
                if self.norm_quant.capabilities.arithmetic != "torch_rmsnorm_bf16_rounding":
                    raise ValueError("Pi05 requires strict RMSNorm/encoding rounding")
        if config.fused_mlp:
            for prefix in (False, True):
                self.paired[prefix] = paired_gelu_backends.get(
                    selected.paired_gelu,
                    request,
                    approximate=self.approximate,
                    profile=config.capability,
                    large_m=prefix,
                )
        self.attention = None
        if config.attention != "reference":
            name = config.attention
            if name == "query_major" and config.hardware == "spark":
                name = "query_major_cuda"
            self.attention = get_attention_backend(name)
        self.norm_plans: dict[tuple, NormalizationPlan] = {}
        self.quant_plans: dict[tuple, NormQuantPlan] = {}
        self.mlp_plans: dict[int, Any] = {}
        self.kv_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.active_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.active_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._scope: int | None = None

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Run the RLinf vision contract over controller-owned derived weights."""
        if self.vision is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm vision weights before graph capture")
            from .vision import SiglipPlan

            self.vision = SiglipPlan(self.policy._m.paligemma_with_expert.paligemma.model)
        return self.vision(image)

    def rotary_tables(self, positions: torch.Tensor, width: int) -> tuple:
        """Prepare the original half-width FP32 RLinf factors once per context."""
        return rlinf_rope_tables(positions, width)

    def rotate(self, inputs: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
        """Adapt BHSD storage to the reusable BTNH rotary contract."""
        return self.rotary(inputs.transpose(1, 2), sine, cosine).transpose(1, 2)

    @staticmethod
    def reference_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Preserve RLinf's BF16 query scaling, FP32 scores and BF16 probabilities."""
        batch, heads, length, width = query.shape
        query = (
            (query * width**-0.5)
            .transpose(1, 2)
            .reshape(batch, length, key.shape[1], heads // key.shape[1], width)
        )
        scores = torch.einsum("BTKGH,BSKH->BKGTS", query.float(), key.transpose(1, 2).float())
        if mask is not None:
            allowed = mask if mask.dtype == torch.bool else mask == 0
            scores = torch.where(allowed[:, :, None], scores, -2.3819763e38)
        probabilities = F.softmax(scores, dim=-1).to(value.dtype)
        output = torch.einsum("BKGTS,BSKH->BTKGH", probabilities, value.transpose(1, 2))
        return output.reshape(batch, length, heads, width).transpose(1, 2)

    def workspace_key(self, device: torch.device) -> tuple[str, int]:
        """Give captured executions independent storage and eager streams reusable storage."""
        return ("graph", self._scope) if self._scope is not None else ("stream", _stream_key(device))

    @contextmanager
    def execution_scope(self, owner: object) -> Iterator[None]:
        """Bind warmup/capture workspaces to a graph-owned lifetime token."""
        previous = self._scope
        self._scope = id(owner)
        try:
            yield
        finally:
            self._scope = previous

    def release_scope(self, owner: object) -> None:
        """Release an execution's storage after its graph and CUDA work are completed."""
        scope = ("graph", id(owner))
        for plans in (self.norm_plans, self.quant_plans, self.kv_buffers):
            for key in tuple(plans):
                if key[-1] == scope:
                    del plans[key]
        for mlp in self.mlp_plans.values():
            mlp.release_scope(scope)

    def validate_request(self, batch_size: int, steps: int | None = None) -> None:
        """Reject shapes outside measured profiles before graph creation."""
        if (
            self.config.fused_mlp
            and self.config.hardware in ("thor", "spark")
            and (batch_size != 1 or self.policy.config.action_horizon != 10 or steps not in (None, 10))
        ):
            raise ValueError("Thor/Spark Pi05 launch profiles require B1, horizon10 and 10 denoise steps")
        if self.config.attention == "folded_flash" and batch_size != 1:
            raise ValueError("Pi05 folded FlashAttention currently requires B1 prefix compaction")

    def _norm_plan(self, norm: Any, inputs: torch.Tensor) -> Any:
        key = (id(norm), tuple(inputs.shape), inputs.device, self.workspace_key(inputs.device))
        if key not in self.norm_plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm fused normalization shapes on the capture stream first")
            self.norm_plans[key] = self.norm.plan(inputs, adaptive=norm.dense is not None, eps=norm.eps)
        return self.norm_plans[key]

    def normalize(
        self, norm: Any, inputs: torch.Tensor, modulation: torch.Tensor | None, *, native: bool = False
    ) -> tuple:
        """Apply the original FP32 mean and affine operations with BF16 outputs."""
        if self.norm is None:
            from ..modeling_pi05 import _rmsnorm

            return _rmsnorm(norm, inputs, None, False, modulation, native)
        return self._norm_plan(norm, inputs).normalize(
            inputs, scale=getattr(norm, "weight", None) if modulation is None else None, modulation=modulation
        )

    def residual_normalize(
        self,
        norm: Any,
        inputs: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor | None,
        modulation: torch.Tensor | None,
        *,
        native: bool = False,
    ) -> tuple:
        """Fuse rounded attention residual, squared values and the following norm."""
        if self.norm is None:
            from ..modeling_pi05 import _gated_residual

            residual = _gated_residual(inputs, update, gate, False, native)
            normalized, next_gate = self.normalize(norm, residual, modulation, native=native)
            return residual, normalized, next_gate
        return self._norm_plan(norm, inputs).residual_normalize(
            inputs,
            update,
            gate,
            scale=getattr(norm, "weight", None) if modulation is None else None,
            modulation=modulation,
        )

    def _mlp_plan(self, module: Any, *, expert: bool) -> Any:
        from .mlp import MlpPlan

        if id(module) not in self.mlp_plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare MLP weights before CUDA Graph capture")
            layer = MlpLayerPrecision()
            layers = self.config.action_layers if expert else self.config.prefix_layers
            if layers:
                tower = self.policy._expert_tower if expert else self.policy._prefix_tower
                index = next(i for i, block in enumerate(tower.layers) if block.mlp is module)
                layer = layers[index]
            self.mlp_plans[id(module)] = MlpPlan(self, module, layer, prefix=not expert)
        return self.mlp_plans[id(module)]

    def mlp(self, module: Any, inputs: torch.Tensor, *, expert: bool) -> torch.Tensor:
        """Run paired BF16 GEMMs or calibrated projections with the selected GELU."""
        return self._mlp_plan(module, expert=expert)(inputs)

    def quantized_residual_mlp(
        self,
        norm: Any,
        inputs: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor | None,
        modulation: torch.Tensor | None,
        module: Any,
        *,
        expert: bool = True,
    ) -> tuple:
        """Fuse residual/RMSNorm/encoding for a low precision gate/up in either tower."""
        # Resolve the immutable weight pack before selecting its activation encoder.
        mlp = self._mlp_plan(module, expert=expert)
        bits = 8 if mlp.gate.precision == "fp8" else 4
        scale = mlp.gate.maximum / (448 * (6 if bits == 4 else 1))
        key = (id(norm), tuple(inputs.shape), bits, scale, self.workspace_key(inputs.device))
        if key not in self.quant_plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm fused norm/quantization shapes before capture")
            self.quant_plans[key] = self.norm_quant.plan(
                inputs, bits=bits, calibrated_scale=scale, adaptive=modulation is not None
            )
        residual, encoded, next_gate = self.quant_plans[key].residual_normalize(
            inputs,
            update,
            gate,
            scale=getattr(norm, "weight", None) if modulation is None else None,
            modulation=modulation,
        )
        return residual, mlp.project(residual, encoded), next_gate

    def tower_forward(
        self,
        tower: Any,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        condition: torch.Tensor | None,
        prefix_kv: list | None,
        collect: bool,
        modulations: tuple | None,
        native_prefix: bool,
        *,
        cache_only: bool = False,
    ) -> tuple[torch.Tensor, list | None]:
        """Run the existing tower equations through selected operator plans."""
        from ..modeling_pi05 import _gated_residual, _mlp

        expert = tower is self.policy._expert_tower
        collected = [] if collect else None
        fused = modulations is not None

        def affine(norm, index):
            if modulations is not None:
                return modulations[index].float().contiguous()
            return (
                None
                if condition is None or norm.dense is None
                else norm.dense(condition).float().contiguous()
            )

        for index, layer in enumerate(tower.layers):
            normalized, gate = self.normalize(
                layer.input_layernorm, hidden, affine(layer.input_layernorm, 2 * index), native=fused
            )
            if cache_only and collect and not expert and index == len(tower.layers) - 1:
                from ..modeling_pi05 import _apply_rope, _fused_qkv

                attn = layer.self_attn
                if self.config.numerics == "rlinf" or self.policy.attention == "eager":
                    key, value = attn.k_proj(normalized), attn.v_proj(normalized)
                else:
                    _, key, value = _fused_qkv(attn, normalized)
                batch, length = normalized.shape[:2]
                key, value = (x.view(batch, length, -1, attn.head_dim).transpose(1, 2) for x in (key, value))
                key = (
                    self.rotate(key, cos, sin)
                    if self.config.numerics == "rlinf"
                    else _apply_rope(key, key, cos, sin, False)[0]
                )
                collected.append((key, value))
                return hidden, collected
            attended = self.policy._attn_sublayer(
                layer.self_attn,
                normalized,
                cos,
                sin,
                mask,
                None if prefix_kv is None else prefix_kv[index],
                collected,
                native_prefix,
                fused,
                operators=self,
            )
            modulation = affine(layer.post_attention_layernorm, 2 * index + 1)
            layers = self.config.action_layers if expert else self.config.prefix_layers
            precision = layers[index] if layers else MlpLayerPrecision()
            if self.norm_quant is not None and precision.gate_up != "bf16":
                hidden, update, gate = self.quantized_residual_mlp(
                    layer.post_attention_layernorm,
                    hidden,
                    attended,
                    gate,
                    modulation,
                    layer.mlp,
                    expert=expert,
                )
            else:
                hidden, normalized, gate = self.residual_normalize(
                    layer.post_attention_layernorm, hidden, attended, gate, modulation, native=fused
                )
                if self.config.fused_mlp or precision.gate_up != "bf16" or precision.down != "bf16":
                    update = self.mlp(layer.mlp, normalized, expert=expert)
                else:
                    update = _mlp(
                        layer.mlp,
                        normalized,
                        False,
                        fused,
                        fuse_projections=self.policy.attention != "eager" and self.config.numerics != "rlinf",
                        approximate=self.approximate,
                    )
            hidden = _gated_residual(hidden, update, gate, False, fused)
        hidden, _ = self.normalize(tower.norm, hidden, affine(tower.norm, -1), native=fused)
        return hidden, collected

    @contextmanager
    def decode_context(self, prefix: Pi05Prefix) -> Iterator[None]:
        """Hoist prefix-dependent arithmetic and bind K/V storage for all steps."""
        if getattr(self, "action_context", None) is not None:
            raise RuntimeError("Pi05 action contexts cannot be nested")
        if self.config.reuse_action_context:
            batch, length = prefix.batch_size, self.policy.config.action_horizon
            pad = torch.ones(batch, length, dtype=torch.bool, device=prefix.prefix_pad_masks.device)
            groups = self.policy._suffix_att_masks(length, self.policy.execution_dtype, pad.device).expand(
                batch, length
            )
            positions, mask = self.policy._action_context(prefix, pad, groups)
            tower = self.policy._expert_tower
            if self.config.numerics == "rlinf":
                cosine, sine = self.rotary_tables(positions, tower.layers[0].self_attn.head_dim)
            else:
                cosine, sine = rope_tables(tower.rotary_emb, prefix.kv[0][0], positions)
            self.action_context = positions, mask, cosine, sine
        try:
            with self._kv_context(prefix):
                yield
        finally:
            self.action_context = None

    @contextmanager
    def _kv_context(self, prefix: Pi05Prefix) -> Iterator[None]:
        """Copy each read-only prefix once, then overwrite only suffix K/V slots."""
        if not self.config.kv_workspace:
            yield
            return
        if self.active_kv is not None:
            raise RuntimeError("Pi05 K/V workspace decode contexts cannot be nested")
        active = {}
        sources = {}
        for layer, (key, value) in zip(self.policy._expert_tower.layers, prefix.kv, strict=True):
            if (
                key.shape != value.shape
                or key.ndim != 4
                or key.dtype != value.dtype
                or key.device != value.device
            ):
                raise ValueError("Prefix K/V must have identical BHSD shapes")
            shape = (*key.shape[:2], key.shape[2] + self.policy.config.action_horizon, key.shape[3])
            identifier = (id(layer.self_attn), shape, key.dtype, key.device, self.workspace_key(key.device))
            full = self.kv_buffers.get(identifier)
            if full is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm this K/V workspace shape before graph capture")
                full = (key.new_empty(shape), value.new_empty(shape))
                self.kv_buffers[identifier] = full
            for destination, source in zip(full, (key, value), strict=True):
                destination[:, :, : key.shape[2]].copy_(source)
            active[id(layer.self_attn)] = full
            sources[id(layer.self_attn)] = (key, value)
        self.active_kv = active
        self.active_sources = sources
        try:
            yield
        finally:
            self.active_kv = None
            self.active_sources = {}

    def combine_kv(self, module: Any, key: torch.Tensor, value: torch.Tensor, prefix: tuple) -> tuple:
        """Return the active full workspace or concatenate for arbitrary single steps."""
        full = None if self.active_kv is None else self.active_kv.get(id(module))
        if full is None:
            return torch.cat((prefix[0], key), dim=2), torch.cat((prefix[1], value), dim=2)
        if any(
            original is not supplied
            for original, supplied in zip(self.active_sources[id(module)], prefix, strict=True)
        ):
            raise ValueError("K/V workspace requires the active decode prefix")
        for destination, source in zip(full, (key, value), strict=True):
            suffix = destination[:, :, prefix[0].shape[2] :]
            if suffix.shape != source.shape or suffix.dtype != source.dtype or suffix.device != source.device:
                raise ValueError("Action K/V differs from the warmed workspace layout")
            suffix.copy_(source)
        return full

    def clear(self) -> None:
        """Release transient storage after all policy graphs have been discarded."""
        self.norm_plans.clear()
        self.quant_plans.clear()
        self.mlp_plans.clear()
        self.kv_buffers.clear()
        self.active_kv = None
        self.active_sources.clear()
        self.action_context = self.vision = None
