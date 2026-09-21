"""pi0.5 adapter: LeRobot ``PI05Policy`` weights, **embodiinfer-owned** forward.

pi0.5 is a PaliGemma-2B VLM backbone plus a Gemma-300M flow-matching action
expert — the exact two-stage shape the engine schedules. This adapter uses
LeRobot only to *build and load* the checkpoint (its safetensors key remapping +
the adarms/PiGemma module definitions); **the transformer forward — the loop the
engine optimizes — is ours**. We do not call LeRobot's ``sample_actions`` /
``paligemma_with_expert.forward``; we re-run the Gemma decoder stack ourselves,
so attention goes through embodiinfer's :class:`AttentionBackend` and the KV cache is
ours.

What we own (touching only the loaded weight modules):
  * the per-tower decoder loop (RMSNorm + adarms, gated residual, MLP),
  * the attention sub-layer: q/k/v projection, RoPE, our own prefix-KV cache,
    MQA ``repeat_kv``, and the pluggable ``AttentionBackend`` (eager / sdpa),
  * the split into ``encode_prefix`` (build the prefix KV once) and
    ``denoise_step`` (expert attends cached prefix KV + its own suffix KV).

With ``native_embeddings=True``, EmbodiInfer also owns per-camera SigLIP, token/time
embeddings, default RoPE and masks. The default keeps the existing embedding
path and optimized batched-camera/compiled routes. Native embeddings preserve
per-camera execution order, including for SDPA comparisons.

What the default embedding path reuses from the loaded model:
  * ``embed_prefix`` (SigLIP image features + token embedding + masks) and
    ``embed_suffix`` (action + time embedding), ``action_out_proj``, the
    rotary-embedding tables, and the additive attention-mask construction.

Owning the KV cache removes the per-step ``copy.deepcopy`` LeRobot needs: the
prefix K/V are read-only tensors, concatenated fresh with the suffix K/V each
step (never mutated), so there is no HF ``DynamicCache`` to defend against.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ...engine.parallel import TensorParallelContext
from ...layers import get_attention_backend, get_split_kv_attention_backend
from ...layers.linear import QuantizationConfig, parse_quantization_config
from ...models.linear import QuantizedLinear, pack_quantized_linears, quantize_linear
from ...types import Observation
from ..base import FlowVLAPolicy, VLAPolicy
from ..config import VLAPolicyConfig
from ..factory import register_policy
from .embeddings import attention_mask_4d, make_attention_mask, rope_tables, time_embedding
from .processor_pi05 import Pi05Batch

if TYPE_CHECKING:
    from ...engine.core import EngineCore

_INSTALL = "pi0.5 requires lerobot: run `uv sync --frozen --no-dev --group pi05`"
_COMPILED_IMAGE_ENCODERS: dict[int, Any] = {}
_COMPILED_PREFIX_ENCODERS: dict[tuple[int, bool], Any] = {}
_COMPILED_INFERENCE_HELPERS: dict[Any, Any] = {}
_QUANTIZED_WEIGHT_LOCK = Lock()


def _inference_helper(fn, use_inductor: bool):
    """Compile a helper on first use, never while importing the adapter."""
    if not use_inductor:
        return fn
    compiled = _COMPILED_INFERENCE_HELPERS.get(fn)
    if compiled is None:
        compiled = torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={
                "triton.cudagraphs": False,
                "emulate_precision_casts": True,
            },
        )
        _COMPILED_INFERENCE_HELPERS[fn] = compiled
    return compiled


# ---- attention math (reimplemented to match HF Gemma exactly) ---------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_impl(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # [B, 1, seq, hd] broadcast over heads
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_inductor: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = _inference_helper(_apply_rope_impl, use_inductor)
    return fn(q, k, cos, sin)


def _gated_residual_impl(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    return x + y if gate is None else x + y * gate


def _gated_residual(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor | None,
    use_inductor: bool,
    use_triton: bool = False,
) -> torch.Tensor:
    if use_triton and gate is not None:
        from ...backend.triton.norm import gated_residual

        return gated_residual(x, y, gate)
    fn = _inference_helper(_gated_residual_impl, use_inductor)
    return fn(x, y, gate)


def _rmsnorm_math(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    modulation: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    if modulation is None:
        return (normed * (1.0 + weight.float())).type_as(x), None
    if x.ndim == 3:
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    normed = normed * (1 + scale.float()) + shift.float()
    return normed.to(x.dtype), gate.to(x.dtype)


def _rmsnorm(
    norm,
    x: torch.Tensor,
    cond: torch.Tensor | None,
    use_inductor: bool,
    modulation: torch.Tensor | None = None,
    use_triton: bool = False,
):
    """PiGemmaRMSNorm forward (fp32 variance); returns (output, adarms gate)."""
    fn = _inference_helper(_rmsnorm_math, use_inductor)
    if modulation is None:
        if cond is None or norm.dense is None:
            return fn(x, norm.weight, norm.eps, None)
        modulation = _adarms_projection(norm, cond)
    if use_triton:
        from ...backend.triton.norm import ada_rms_norm

        return ada_rms_norm(x, modulation, norm.eps)
    return fn(x, None, norm.eps, modulation)


def _adarms_projection(norm: Any, cond: torch.Tensor) -> torch.Tensor:
    """Project the current AdaRMS condition without pointer-based reuse.

    CUDA graph capture may warm up and capture with the same input address while
    the tensor contents change between denoise steps.  Reusing a projection by
    ``data_ptr`` would therefore freeze the modulation at the warmup value.  The
    native PI0.5 runtime precomputes its fixed schedule explicitly and passes the
    resulting modulations to ``_tower_forward``; generic denoise calls must keep
    this operation input-dependent.
    """
    return norm.dense(cond)


def _packed_quantized_projection(
    projections: tuple[QuantizedLinear, ...],
    cache_owner: Any,
    cache_name: str,
) -> QuantizedLinear:
    """Return one cached projection for quantized operators with a shared input."""

    first = projections[0]
    expected_output = sum(projection.out_features for projection in projections)
    packed = getattr(cache_owner, cache_name, None)
    if (
        isinstance(packed, QuantizedLinear)
        and packed.weight.device == first.weight.device
        and packed.compute_dtype == first.compute_dtype
        and packed.out_features == expected_output
    ):
        return packed
    with _QUANTIZED_WEIGHT_LOCK:
        packed = getattr(cache_owner, cache_name, None)
        if not (
            isinstance(packed, QuantizedLinear)
            and packed.weight.device == first.weight.device
            and packed.compute_dtype == first.compute_dtype
            and packed.out_features == expected_output
        ):
            packed = pack_quantized_linears(projections)
            object.__setattr__(cache_owner, cache_name, packed)
    return packed


def _fused_linear_pair(x: torch.Tensor, first: Any, second: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate two bias-free projections as one GEMM and split the result.

    The fused weight and the quantization marker are runtime-only caches kept
    in ``module.__dict__`` (never registered buffers or parameters), so they do
    not appear in state dicts and are dropped wholesale by
    ``_clear_inference_caches`` on refit or device moves.
    """
    if torch.is_grad_enabled() or first.training or second.training:
        return first(x), second(x)
    if isinstance(first, QuantizedLinear) or isinstance(second, QuantizedLinear):
        if not isinstance(first, QuantizedLinear) or not isinstance(second, QuantizedLinear):
            return first(x), second(x)
        packed = _packed_quantized_projection(
            (first, second),
            first,
            "_embodiinfer_fused_pair_quantized",
        )
        fused = packed(x)
        return fused.split((first.out_features, second.out_features), dim=-1)
    weight = getattr(first, "_embodiinfer_fused_pair_weight", None)
    if weight is None or weight.device != first.weight.device or weight.dtype != first.weight.dtype:
        weight = torch.cat((first.weight, second.weight), dim=0).contiguous()
        first._embodiinfer_fused_pair_weight = weight
    fused = F.linear(x, weight)
    return fused.split((first.out_features, second.out_features), dim=-1)


def _fused_qkv(attn: Any, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Evaluate Gemma's bias-free Q/K/V projections as one GEMM."""
    if torch.is_grad_enabled() or attn.training:
        return attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    projections = (attn.q_proj, attn.k_proj, attn.v_proj)
    if any(isinstance(projection, QuantizedLinear) for projection in projections):
        if not all(isinstance(projection, QuantizedLinear) for projection in projections):
            return tuple(projection(x) for projection in projections)
        packed = _packed_quantized_projection(
            projections,
            attn.q_proj,
            "_embodiinfer_fused_qkv_quantized",
        )
        fused = packed(x)
        return fused.split(tuple(projection.out_features for projection in projections), dim=-1)
    weight = getattr(attn.q_proj, "_embodiinfer_fused_qkv_weight", None)
    if (
        weight is None
        or weight.device != attn.q_proj.weight.device
        or weight.dtype != attn.q_proj.weight.dtype
    ):
        weight = torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=0).contiguous()
        attn.q_proj._embodiinfer_fused_qkv_weight = weight
    fused = F.linear(x, weight)
    return fused.split(
        (
            attn.q_proj.out_features,
            attn.k_proj.out_features,
            attn.v_proj.out_features,
        ),
        dim=-1,
    )


def _prepare_fused_projection_weights(tower: Any) -> None:
    for layer in tower.layers:
        attn = layer.self_attn
        attention_projections = (attn.q_proj, attn.k_proj, attn.v_proj)
        if all(isinstance(projection, QuantizedLinear) for projection in attention_projections):
            _packed_quantized_projection(
                attention_projections,
                attn.q_proj,
                "_embodiinfer_fused_qkv_quantized",
            )
        elif not any(isinstance(projection, QuantizedLinear) for projection in attention_projections) and (
            getattr(attn.q_proj, "_embodiinfer_fused_qkv_weight", None) is None
        ):
            attn.q_proj._embodiinfer_fused_qkv_weight = torch.cat(
                (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=0
            ).contiguous()
        mlp = layer.mlp
        mlp_projections = (mlp.gate_proj, mlp.up_proj)
        if all(isinstance(projection, QuantizedLinear) for projection in mlp_projections):
            _packed_quantized_projection(
                mlp_projections,
                mlp.gate_proj,
                "_embodiinfer_fused_pair_quantized",
            )
        elif not any(isinstance(projection, QuantizedLinear) for projection in mlp_projections) and (
            getattr(mlp.gate_proj, "_embodiinfer_fused_pair_weight", None) is None
        ):
            mlp.gate_proj._embodiinfer_fused_pair_weight = torch.cat(
                (mlp.gate_proj.weight, mlp.up_proj.weight), dim=0
            ).contiguous()


def _mlp(
    mlp,
    x: torch.Tensor,
    use_inductor: bool,
    use_triton: bool = False,
    *,
    fuse_projections: bool = True,
) -> torch.Tensor:
    if fuse_projections:
        gate, up = _fused_linear_pair(x, mlp.gate_proj, mlp.up_proj)
    else:
        # The eager reference keeps the two native GEMM shapes.  Besides matching
        # LeRobot's eager path, this preserves its rounding on selectively cast
        # (e.g. bf16) checkpoints.
        gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    if use_triton:
        from ...backend.triton.activation import gated_gelu

        activated = gated_gelu(gate, up)
    else:
        fn = _inference_helper(_gated_gelu_impl, use_inductor)
        activated = fn(gate, up)
    return mlp.down_proj(activated)


def _gated_gelu_impl(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.gelu(gate, approximate="tanh") * up


@dataclass
class Pi05Prefix:
    """pi0.5 prefix state: our own per-layer (K, V) cache + the pad mask.

    ``kv[i]`` is the PaliGemma tower's ``(key, value)`` at layer ``i``, shape
    ``[B, num_kv_heads, prefix_len, head_dim]`` (post-RoPE, pre-repeat). Read-only
    across denoise steps — the expert concatenates it with its own suffix K/V.
    """

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    prefix_pad_masks: torch.Tensor
    # Final-layer hidden states of the prefix tower, ``[B, prefix_len, width]``.
    # Populated by ``encode_prefix(..., return_hidden=True)``; an RL trainer's
    # value head reads pooled prefix features from here.
    last_hidden: torch.Tensor | None = None
    # Dense-attention specialization used by CUDA-graph capture.
    all_valid: bool = False

    @property
    def batch_size(self) -> int:
        return self.prefix_pad_masks.shape[0]

    def to(self, device: torch.device | str) -> Pi05Prefix:
        return Pi05Prefix(
            [(k.to(device), v.to(device)) for k, v in self.kv],
            self.prefix_pad_masks.to(device),
            self.last_hidden.to(device) if self.last_hidden is not None else None,
            self.all_valid,
        )

    def expand(self, num_samples: int) -> Pi05Prefix:
        if num_samples == 1:
            return self
        kv = [
            (k.repeat_interleave(num_samples, dim=0), v.repeat_interleave(num_samples, dim=0))
            for k, v in self.kv
        ]
        return Pi05Prefix(
            kv,
            self.prefix_pad_masks.repeat_interleave(num_samples, dim=0),
            self.last_hidden.repeat_interleave(num_samples, dim=0) if self.last_hidden is not None else None,
            self.all_valid,
        )


def _embed_prefix_batched(
    model: Any,
    images: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    token_masks: torch.Tensor,
    use_inductor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed every camera view in one SigLIP call, preserving token order."""
    batch_size = images[0].shape[0]
    image_encoder = model.paligemma_with_expert.embed_image
    if use_inductor:
        model_key = id(model)
        compiled = _COMPILED_IMAGE_ENCODERS.get(model_key)
        if compiled is None:
            compiled = torch.compile(
                image_encoder,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                options={
                    "triton.cudagraphs": False,
                    "emulate_precision_casts": True,
                },
            )
            _COMPILED_IMAGE_ENCODERS[model_key] = compiled
        image_encoder = compiled
    image_embeddings = image_encoder(torch.cat(images, dim=0)).split(batch_size, dim=0)

    embeddings: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []
    for image_embedding, image_mask in zip(image_embeddings, image_masks, strict=True):
        num_image_tokens = image_embedding.shape[1]
        embeddings.append(image_embedding)
        pad_masks.append(image_mask[:, None].expand(batch_size, num_image_tokens))

    language_embedding = model.paligemma_with_expert.embed_language_tokens(tokens)
    language_embedding = language_embedding * language_embedding.shape[-1] ** 0.5
    embeddings.append(language_embedding)
    pad_masks.append(token_masks)

    embeddings_tensor = torch.cat(embeddings, dim=1)
    pad_masks_tensor = torch.cat(pad_masks, dim=1)
    attention_masks = torch.zeros_like(pad_masks_tensor, dtype=torch.bool)
    return embeddings_tensor, pad_masks_tensor, attention_masks


class Pi05Policy(FlowVLAPolicy):
    """First-class pi0.5 policy: checkpoint adapters and embodiinfer-owned forward."""

    def __init__(
        self,
        config: VLAPolicyConfig,
        checkpoint,
        attention: str = "sdpa",
        compile_backend: str = "none",
        load_device: str | None = None,
        low_cpu_mem_usage: bool = False,
        quantization: str | Mapping[str, Any] | QuantizationConfig | None = None,
        *,
        native_inference: bool = False,
        native_embeddings: bool = False,
        vision_attention: str = "sdpa",
        prefix_cuda_graph: bool = False,
        denoise_attention: str = "sdpa",
        prefix_attention: str = "sdpa",
        checkpoint_config: Mapping[str, Any] | None = None,
        tensor_parallel_size: int = 1,
        tensor_parallel_group=None,
    ):
        super().__init__(config)
        if compile_backend not in {"none", "inductor"}:
            raise ValueError(f"pi05 compile_backend must be 'none' or 'inductor'; got {compile_backend!r}")
        self.compile_backend = compile_backend
        if denoise_attention not in {"sdpa", "triton"} or prefix_attention not in {"sdpa", "triton"}:
            raise ValueError("PI0.5 denoise_attention must be 'sdpa' or 'triton'")
        if (
            prefix_cuda_graph or denoise_attention != "sdpa" or prefix_attention != "sdpa"
        ) and not native_inference:
            raise ValueError("PI0.5 specialized graphs/attention require native_inference=True")
        if native_inference and attention != "sdpa":
            raise ValueError("PI0.5 native inference requires the SDPA prefix reference")
        self.native_embeddings = native_embeddings
        self._vision_attn = get_attention_backend(vision_attention)
        self.native_inference = native_inference
        self.prefix_cuda_graph = prefix_cuda_graph
        self.denoise_attention = denoise_attention
        self.prefix_attention = prefix_attention
        try:
            from lerobot.policies.pi05.modeling_pi05 import (
                create_sinusoidal_pos_embedding,
                make_att_2d_masks,
            )
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise ImportError(_INSTALL) from exc

        self.attention = attention
        self._attn = get_attention_backend(attention)
        self._native_attention = None
        self._make_att_2d_masks = make_attention_mask if native_embeddings else make_att_2d_masks
        self._sinusoidal = time_embedding if native_embeddings else create_sinusoidal_pos_embedding
        # cached static suffix att-mask ([1, 0, ..., 0], input-independent) so the
        # denoise step avoids lerobot's ``torch.tensor(pylist)`` (a host->device op
        # that CUDA-graph capture forbids); see ``_embed_suffix``.
        self._cached_suffix_att: torch.Tensor | None = None
        # ``checkpoint`` is a path / HF id to load, or an already-loaded
        # PI05Policy to wrap (sharing weights — used by parity tests).
        self.openpi_config = None
        if isinstance(checkpoint, str):
            from .checkpoints import load_checkpoint

            self._lerobot, self.openpi_config = load_checkpoint(
                checkpoint,
                load_device=load_device,
                checkpoint_config=checkpoint_config,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        else:
            self._lerobot = checkpoint.eval()
        self.checkpoint = checkpoint if isinstance(checkpoint, str) else None
        self._m = self._lerobot.model  # PI05Pytorch: embeddings, projections, masks
        pwe = self._m.paligemma_with_expert
        self._prefix_tower = pwe.paligemma.model.language_model  # PiGemmaModel (VLM text)
        self._expert_tower = pwe.gemma_expert.model  # PiGemmaModel (action expert)
        self.tensor_parallel = TensorParallelContext.from_distributed(
            tensor_parallel_size,
            tensor_parallel_group,
        )
        if self.tensor_parallel.enabled:
            from ...engine.parallel import parallelize_pi05_towers

            parallelize_pi05_towers(self, self.tensor_parallel)
        quantization_config = parse_quantization_config(quantization)
        if quantization_config is not None:
            if self.tensor_parallel.enabled:
                raise NotImplementedError("PI0.5 quantization currently supports tensor_parallel_size=1")
            if self.compile_backend != "none":
                raise NotImplementedError(
                    "PI0.5 quantization has not been validated with compile_backend='inductor'"
                )
            self._quantize_towers(quantization_config)
        # (prefix_len, pad_dtype, kv_dtype) of the most recent encode_prefix — lets
        # ``allocate_static_prefix`` size the CUDA-graph buffer without hardcoding
        # the tokenizer's padded language length. Set on the first encode_prefix.
        self._cached_prefix_meta: tuple[int, torch.dtype, torch.dtype] | None = None
        from .runtime import Pi05FlowDecoder, Pi05Runtime

        self._runtime = Pi05Runtime(self)
        if native_inference:
            self._decoder = Pi05FlowDecoder(self)

    @property
    def execution_dtype(self) -> torch.dtype:
        """Keep noise and time in the action projection dtype."""
        # Noise and time must stay fp32 for the usual bf16/fp32 checkpoint.
        return self._m.action_in_proj.weight.dtype

    def _embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """EmbodiInfer SigLIP forward over loaded weights, without HF model forwards."""
        pg = self._m.paligemma_with_expert.paligemma.model
        vision = pg.vision_tower.vision_model
        emb = vision.embeddings
        h = emb.patch_embedding(image.to(emb.patch_embedding.weight.dtype))
        h = h.flatten(2).transpose(1, 2)
        h = h + emb.position_embedding(emb.position_ids)
        for layer in vision.encoder.layers:
            x = layer.layer_norm1(h)
            a = layer.self_attn
            b, s, _ = x.shape
            q, k, v = [
                proj(x).view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                for proj in (a.q_proj, a.k_proj, a.v_proj)
            ]
            y = self._vision_attn.attend(q, k, v, scaling=a.scale)
            h = h + a.out_proj(y.transpose(1, 2).reshape(b, s, -1).contiguous())
            x = layer.layer_norm2(h)
            h = h + layer.mlp.fc2(layer.mlp.activation_fn(layer.mlp.fc1(x)))
        h = vision.post_layernorm(h)
        h = pg.multi_modal_projector.linear(h)
        # Preserve both operations in LeRobot's embedding path: eliminating
        # this division/multiplication changes rounding before the bf16 tower.
        scale = pg.config.text_config.hidden_size**0.5
        return ((h / scale) * scale).to(image.dtype)

    def _embed_prefix(self, batch: Pi05Batch):
        embs, pads = [], []
        for image, mask in zip(batch.images, batch.img_masks, strict=True):
            h = self._embed_image(image)
            embs.append(h)
            pads.append(mask[:, None].expand(h.shape[:2]))
        embedding = self._prefix_tower.embed_tokens
        lang = embedding(batch.tokens.to(embedding.weight.device)).to(batch.tokens.device)
        embs.append(lang * lang.shape[-1] ** 0.5)
        pads.append(batch.masks)
        pad = torch.cat(pads, dim=1)
        return torch.cat(embs, dim=1), pad, torch.zeros_like(pad)

    def _native_enabled(self) -> bool:
        return (
            self.native_inference
            and not self.training
            and not torch.is_grad_enabled()
            and not self.tensor_parallel.enabled
            and self._m.action_in_proj.weight.device.type == "cuda"
        )

    def _prepare_native_attention(self) -> None:
        """Resolve optional kernels before native execution or graph capture."""
        if self._native_attention is None and "triton" in (self.denoise_attention, self.prefix_attention):
            self._native_attention = get_split_kv_attention_backend("triton_split_kv")

    def _clear_inference_caches(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.clear()
        for key in tuple(_COMPILED_PREFIX_ENCODERS):
            if key[0] == id(self):
                del _COMPILED_PREFIX_ENCODERS[key]
        model = getattr(self, "_m", None)
        if model is not None:
            _COMPILED_IMAGE_ENCODERS.pop(id(model), None)
        for module in self.modules():
            for name in (
                "_embodiinfer_fused_qkv_weight",
                "_embodiinfer_fused_pair_weight",
                "_embodiinfer_fused_qkv_quantized",
                "_embodiinfer_fused_pair_quantized",
            ):
                module.__dict__.pop(name, None)

    def on_refit(self, version: int) -> None:
        """Rebuild packed weights, modulation schedules and graphs after a refit."""
        super().on_refit(version)
        self._clear_inference_caches()

    def _apply(self, fn, recurse: bool = True):
        self._clear_inference_caches()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True):
        """Discard inference-only derived tensors when entering or leaving training."""
        if mode != self.training:
            self._clear_inference_caches()
        return super().train(mode)

    def _quantize_towers(self, config: QuantizationConfig) -> None:
        quantized: list[str] = []
        for tower_name, tower in (("prefix", self._prefix_tower), ("expert", self._expert_tower)):
            for index, layer in enumerate(tower.layers):
                candidates = (
                    (layer.self_attn, "q_proj", f"{tower_name}.layers.{index}.self_attn.q_proj"),
                    (layer.self_attn, "k_proj", f"{tower_name}.layers.{index}.self_attn.k_proj"),
                    (layer.self_attn, "v_proj", f"{tower_name}.layers.{index}.self_attn.v_proj"),
                    (layer.self_attn, "o_proj", f"{tower_name}.layers.{index}.self_attn.o_proj"),
                    (layer.mlp, "gate_proj", f"{tower_name}.layers.{index}.mlp.gate_proj"),
                    (layer.mlp, "up_proj", f"{tower_name}.layers.{index}.mlp.up_proj"),
                    (layer.mlp, "down_proj", f"{tower_name}.layers.{index}.mlp.down_proj"),
                )
                for owner, attribute, name in candidates:
                    if config.is_ignored(name):
                        continue
                    setattr(owner, attribute, quantize_linear(getattr(owner, attribute), config))
                    quantized.append(name)
        self.quantized_layers = tuple(quantized)

    def build_serving_adapter(
        self,
        *,
        core: EngineCore,
        checkpoint: str | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        from .serving import Pi05ServingAdapter

        resolved = checkpoint or self.checkpoint
        if resolved is None:
            raise ValueError("pi05 serving adapter requires a checkpoint path")
        return Pi05ServingAdapter(core=core, checkpoint=resolved, config=config)

    # ---- one Gemma decoder stack (our forward over the loaded weights) -------
    def _attn_sublayer(
        self, attn, h, cos, sin, mask, prefix_kv, collected, native_prefix=False, native_expert=False
    ):
        B, S = h.shape[0], h.shape[1]
        hd = attn.head_dim
        # Keep the eager reference's individual GEMMs.  Fusing Q/K/V changes
        # the reduction shape and therefore the rounding of the eager parity
        # path for low-precision checkpoints.
        if self.attention == "eager":
            q, k, v = attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)
        else:
            q, k, v = _fused_qkv(attn, h)
        q = q.view(B, S, -1, hd).transpose(1, 2)  # [B, n_head, S, hd]
        k = k.view(B, S, -1, hd).transpose(1, 2)  # [B, n_kv, S, hd]
        v = v.view(B, S, -1, hd).transpose(1, 2)
        if native_expert and q.dtype in (torch.bfloat16, torch.float16):
            from ...backend.triton.rotary import rotate_qk

            q, k = rotate_qk(q, k, cos, sin)
        else:
            q, k = _apply_rope(q, k, cos, sin, self.compile_backend == "inductor")
        if collected is not None:  # prefix pass: cache this layer's K/V
            collected.append((k, v))
        if native_expert and self.denoise_attention == "triton":
            # All action queries share the same prefix/suffix key validity.
            key_padding_mask = None if mask is None else mask[:, :, :1, :]
            out = self._native_attention.attend_split(
                q, *prefix_kv, k, v, key_padding_mask=key_padding_mask, scaling=attn.scaling
            )
            return attn.o_proj(out.transpose(1, 2).reshape(B, S, -1))
        if prefix_kv is not None:  # denoise pass: attend [prefix ++ suffix]
            pk, pv = prefix_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        if native_prefix:
            # Prefix preparation supplies a shared key mask; masked query outputs are unused.
            key_padding_mask = None if mask is None else mask[:, :, :1, :]
            out = self._native_attention.attend(q, k, v, attn_mask=key_padding_mask, scaling=attn.scaling)
            return attn.o_proj(out.transpose(1, 2).reshape(B, S, -1))
        # K/V stay at num_kv_heads; the attention backend does the GQA repeat
        # (materialized for 'eager', broadcast for 'eager_bc').
        out = self._attn.attend(q, k, v, attn_mask=mask, scaling=attn.scaling)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return attn.o_proj(out)

    def _tower_forward(
        self,
        tower,
        hidden,
        position_ids,
        mask,
        adarms_cond,
        prefix_kv=None,
        collect=False,
        modulations=None,
        native_prefix=False,
    ):
        # Run the tower in its parameters' dtype. Under a selectively-cast
        # backbone (e.g. openpi's bf16-with-fp32-norms regime) the embeddings /
        # input projections outside the tower stay fp32, so the entry hidden
        # may arrive fp32; openpi casts prefix/suffix embeddings to the tower
        # dtype at exactly this boundary. No-op when dtypes already agree.
        entry_projection = tower.layers[0].self_attn.q_proj
        entry_dtype = (
            entry_projection.compute_dtype
            if isinstance(entry_projection, QuantizedLinear)
            else entry_projection.weight.dtype
        )
        hidden = hidden.to(entry_dtype)
        cos, sin = (
            rope_tables(tower.rotary_emb, hidden, position_ids)
            if self.native_embeddings
            else tower.rotary_emb(hidden, position_ids)
        )
        collected: list | None = [] if collect else None
        fused = modulations is not None and hidden.dtype in (torch.bfloat16, torch.float16)
        for i, layer in enumerate(tower.layers):
            residual = hidden
            h, gate = _rmsnorm(
                layer.input_layernorm,
                hidden,
                adarms_cond,
                self.compile_backend == "inductor",
                None if modulations is None else modulations[2 * i],
                fused,
            )
            attn = self._attn_sublayer(
                layer.self_attn,
                h,
                cos,
                sin,
                mask,
                None if prefix_kv is None else prefix_kv[i],
                collected,
                native_prefix,
                modulations is not None,
            )
            hidden = _gated_residual(
                residual,
                attn,
                gate,
                self.compile_backend == "inductor",
                fused,
            )
            residual = hidden
            h, gate = _rmsnorm(
                layer.post_attention_layernorm,
                hidden,
                adarms_cond,
                self.compile_backend == "inductor",
                None if modulations is None else modulations[2 * i + 1],
                fused,
            )
            hidden = _gated_residual(
                residual,
                _mlp(
                    layer.mlp,
                    h,
                    self.compile_backend == "inductor",
                    fused,
                    fuse_projections=self.attention != "eager",
                ),
                gate,
                self.compile_backend == "inductor",
                fused,
            )
        hidden, _ = _rmsnorm(
            tower.norm,
            hidden,
            adarms_cond,
            self.compile_backend == "inductor",
            None if modulations is None else modulations[-1],
            fused,
        )
        return hidden, collected

    # ---- pi0.5 integrates t: 1 -> 0 with dt = -1/N ---------------------------
    def flow_schedule(self, num_steps: int) -> list[tuple[float, float]]:
        dt = -1.0 / num_steps
        return [(1.0 + i * dt, dt) for i in range(num_steps)]

    # ---- stage 1: encode prefix (once) --------------------------------------
    def encode_prefix(self, batch: Pi05Batch, return_hidden: bool = False) -> Pi05Prefix:
        """Encode one observation, using the optional PI0.5 compact-layout runtime."""
        if self._native_enabled() and not return_hidden:
            self._prepare_native_attention()
            return self._runtime.encode(batch, return_hidden)
        return self._encode_prefix_impl(batch, return_hidden)

    def _encode_prefix_impl(
        self, batch: Pi05Batch, return_hidden: bool = False, *, all_valid: bool | None = None
    ) -> Pi05Prefix:
        m = self._m
        # Batching camera views changes the vision GEMM shape and its rounding.
        # The eager reference therefore keeps model.embed_prefix's per-view
        # execution; the compact batched path is reserved for fused backends.
        if self.native_embeddings:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(batch)
        elif self.attention == "eager":
            prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
                batch.images,
                batch.img_masks,
                batch.tokens,
                batch.masks,
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = _embed_prefix_batched(
                m,
                batch.images,
                batch.img_masks,
                batch.tokens,
                batch.masks,
                self.compile_backend == "inductor",
            )
        prefix_att_2d = self._make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if all_valid is None:
            all_valid = bool(prefix_pad_masks.all().item())
        mask4d = None if all_valid else self._attention_mask_4d(prefix_att_2d)
        native_prefix = self.prefix_attention == "triton" and self._native_enabled() and not return_hidden
        if native_prefix and not all_valid:
            # Every valid query sees the same bidirectional key set. Invalid
            # queries stay masked as keys, so their hidden values are unused.
            length = prefix_pad_masks.shape[1]
            key_mask = prefix_pad_masks[:, None, :].expand(-1, length, -1)
            mask4d = self._attention_mask_4d(key_mask)
        if self.compile_backend == "inductor":
            policy_key = (id(self), native_prefix)
            prefix_encoder = _COMPILED_PREFIX_ENCODERS.get(policy_key)
            if prefix_encoder is None:
                _prepare_fused_projection_weights(self._prefix_tower)

                def encode_tower(hidden, positions, attention_mask):
                    return self._tower_forward(
                        self._prefix_tower,
                        hidden,
                        positions,
                        attention_mask,
                        adarms_cond=None,
                        collect=True,
                        native_prefix=native_prefix,
                    )

                prefix_encoder = torch.compile(
                    encode_tower,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    options={
                        "triton.cudagraphs": False,
                        "emulate_precision_casts": True,
                    },
                )
                _COMPILED_PREFIX_ENCODERS[policy_key] = prefix_encoder
            hidden, kv = prefix_encoder(prefix_embs, position_ids, mask4d)
        else:
            hidden, kv = self._tower_forward(
                self._prefix_tower,
                prefix_embs,
                position_ids,
                mask4d,
                adarms_cond=None,
                collect=True,
                native_prefix=native_prefix,
            )
        self._cached_prefix_meta = (prefix_pad_masks.shape[1], prefix_pad_masks.dtype, kv[0][0].dtype)
        return Pi05Prefix(
            kv,
            prefix_pad_masks,
            hidden if return_hidden else None,
            all_valid,
        )

    def _attention_mask_4d(self, allowed: torch.Tensor) -> torch.Tensor:
        if self.native_embeddings:
            return attention_mask_4d(allowed)
        return self._m._prepare_attention_masks_4d(allowed)

    # ---- suffix embedding (embodiinfer-owned, CUDA-graph-safe) ---------------------
    def _suffix_att_masks(self, suffix_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Static suffix attention pattern ``[1, 0, ..., 0]`` (action block).

        Built once as a device tensor and reused, replacing lerobot's per-call
        ``torch.tensor([...])`` (a host->device op forbidden during graph capture).
        """
        cached = self._cached_suffix_att
        if (
            cached is None
            or cached.shape[0] != suffix_len
            or cached.dtype != dtype
            or cached.device != device
        ):
            cached = torch.zeros(suffix_len, dtype=dtype, device=device)
            cached[0] = 1
            self._cached_suffix_att = cached
        return cached[None, :]  # [1, suffix_len]; caller expands over batch

    def _embed_suffix(self, x_t: torch.Tensor, t: torch.Tensor, *, skip_time: bool = False):
        """Re-implements lerobot ``embed_suffix`` exactly (same weight modules),
        but builds the static attention mask graph-safely. ``skip_time`` is only
        used after the native runtime has explicitly prepared a fixed schedule.
        Generic denoise calls always compute the condition from their current
        ``t`` input."""
        m = self._m
        action_emb = m.action_in_proj(x_t)
        adarms_cond = None
        if not skip_time:
            time_emb = self._sinusoidal(
                t,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=t.device,
            ).type(dtype=t.dtype)
            h = m.time_mlp_in(time_emb)
            h = F.silu(h)
            h = m.time_mlp_out(h)
            adarms_cond = F.silu(h)
        embs = action_emb  # suffix = action block only (state lives in the prefix)
        B, suffix_len = embs.shape[:2]
        pad_masks = torch.ones(B, suffix_len, dtype=torch.bool, device=x_t.device)
        att_masks = self._suffix_att_masks(suffix_len, embs.dtype, x_t.device).expand(B, suffix_len)
        return embs, pad_masks, att_masks, adarms_cond

    def _time_condition(self, t: torch.Tensor) -> torch.Tensor:
        m = self._m
        embedding = self._sinusoidal(
            t,
            m.action_in_proj.out_features,
            min_period=m.config.min_period,
            max_period=m.config.max_period,
            device=t.device,
        ).to(t.dtype)
        return F.silu(m.time_mlp_out(F.silu(m.time_mlp_in(embedding))))

    # ---- stage 2: one denoising step (N times) ------------------------------
    def denoise_step(self, x_t: torch.Tensor, t: torch.Tensor, prefix: Pi05Prefix) -> torch.Tensor:
        """Evaluate arbitrary per-request times without fixed-schedule caching."""
        return self._denoise_step_impl(x_t, t, prefix)

    def _denoise_step_impl(
        self, x_t: torch.Tensor, t: torch.Tensor, prefix: Pi05Prefix, modulations=None
    ) -> torch.Tensor:
        m = self._m
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix(
            x_t, t, skip_time=modulations is not None
        )

        prefix_pad_masks = prefix.prefix_pad_masks
        batch_size, prefix_len = prefix_pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d = self._make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        mask4d = None if prefix.all_valid else self._attention_mask_4d(full_att_2d)

        hidden, _ = self._tower_forward(
            self._expert_tower,
            suffix_embs,
            position_ids,
            mask4d,
            adarms_cond=adarms_cond,
            prefix_kv=prefix.kv,
            modulations=modulations,
        )
        # LeRobot casts the suffix to fp32 before the action projection; match the
        # projection weight dtype instead so a bf16 execution stays consistent
        # (fp32 checkpoint -> no-op, so parity is unchanged).
        suffix_out = hidden[:, -m.config.chunk_size :].to(dtype=m.action_out_proj.weight.dtype)
        return m.action_out_proj(suffix_out)

    # ---- CUDA-graph capability: static-shape denoise loop -------------------
    @property
    def supports_cuda_graph(self) -> bool:
        """Only single-rank execution is supported by the current GraphManager."""
        # Graph capture/replay and NCCL collectives must be coordinated across
        # every TP rank; the current GraphManager owns only one device.
        return not self.tensor_parallel.enabled

    def allocate_static_prefix(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Pi05Prefix:
        """Zero-filled static-shape :class:`Pi05Prefix` for graph capture.

        Sizes the per-layer K/V from the prefix tower (num_kv_heads, head_dim,
        num layers) and the prefix length from the most recent ``encode_prefix``.
        A downstream ``copy_prefix_into`` fills it with the live prefix each step.
        """
        if self._cached_prefix_meta is None:
            raise RuntimeError("allocate_static_prefix needs a prior encode_prefix to know the prefix length")
        prefix_len, pad_dtype, kv_dtype = self._cached_prefix_meta
        attn0 = self._prefix_tower.layers[0].self_attn
        head_dim = attn0.head_dim
        num_kv = attn0.k_proj.out_features // head_dim
        num_layers = len(self._prefix_tower.layers)
        kv = [
            (
                torch.zeros(batch_size, num_kv, prefix_len, head_dim, device=device, dtype=kv_dtype),
                torch.zeros(batch_size, num_kv, prefix_len, head_dim, device=device, dtype=kv_dtype),
            )
            for _ in range(num_layers)
        ]
        # All-valid mask for the capture warmup (an all-masked row would make the
        # prefix softmax NaN); overwritten by copy_prefix_into on every replay.
        pad = torch.ones(batch_size, prefix_len, device=device, dtype=pad_dtype)
        return Pi05Prefix(kv, pad)

    def cuda_graph_variant(self, prefix: Pi05Prefix) -> object | None:
        if self.native_inference:
            return prefix.prefix_pad_masks.shape[1], prefix.all_valid
        return prefix.all_valid

    def allocate_static_prefix_from_live(
        self,
        prefix: Pi05Prefix,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        variant: object | None = None,
    ) -> Pi05Prefix:
        """Size generic graph buffers from this prefix, including compact layouts."""
        if not self.native_inference:
            return super().allocate_static_prefix_from_live(prefix, batch_size, device, dtype, variant)
        if prefix.batch_size != batch_size:
            raise ValueError("PI0.5 live prefix batch must match its graph bucket")
        return Pi05Prefix(
            [(k.to(device=device).clone(), v.to(device=device).clone()) for k, v in prefix.kv],
            prefix.prefix_pad_masks.to(device).clone(),
            all_valid=prefix.all_valid,
        )

    def allocate_static_prefix_for_variant(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        variant: object | None,
    ) -> Pi05Prefix:
        if self.native_inference and isinstance(variant, tuple):
            length, all_valid = variant
            if self._cached_prefix_meta is None or length != self._cached_prefix_meta[0]:
                raise ValueError("PI0.5 graph layout requires its live prefix")
            variant = all_valid
        prefix = self.allocate_static_prefix(batch_size, device, dtype)
        prefix.all_valid = bool(variant)
        return prefix

    def copy_prefix_into(self, dst: Pi05Prefix, src: Pi05Prefix) -> None:
        """In-place copy the live prefix K/V + pad mask into the static buffer."""
        for (dk, dv), (sk, sv) in zip(dst.kv, src.kv):
            dk.copy_(sk)
            dv.copy_(sv)
        dst.prefix_pad_masks.copy_(src.prefix_pad_masks)
        dst.all_valid = src.all_valid

    # ---- batch construction (pi0.5's per-camera layout) ---------------------
    def collate(self, observations: list[Observation], request_ids: list[str]) -> Pi05Batch:
        imgs = torch.stack([o.images for o in observations])  # [B, ncam, 3, H, W]
        keys = list(self._lerobot.config.image_features)
        # Reuse LeRobot's SigLIP preprocessing (resize/pad + [-1,1] norm + empty
        # cameras). ``instruction_tokens`` is the token stream (state injected
        # upstream by the tokenizer processor); mask is all-valid.
        batch_dict = {keys[i]: imgs[:, i] for i in range(min(imgs.shape[1], len(keys)))}
        images, img_masks = self._lerobot._preprocess_images(batch_dict)
        tokens = torch.stack([o.instruction_tokens for o in observations])
        masks = torch.ones_like(tokens, dtype=torch.bool)
        return Pi05Batch(images, img_masks, tokens, masks, list(request_ids))

    def pad(self, batch: Pi05Batch, target_batch_size: int) -> Pi05Batch:
        b = batch.batch_size
        if target_batch_size == b:
            return batch
        pad = target_batch_size - b

        def rep(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0)

        return Pi05Batch(
            [rep(img) for img in batch.images],
            [rep(m) for m in batch.img_masks],
            rep(batch.tokens),
            rep(batch.masks),
            batch.request_ids + [f"__pad_{i}" for i in range(pad)],
        )


@register_policy("pi05")
def _build_pi05(
    checkpoint: str | None = None,
    attention: str = "sdpa",
    compile_backend: str = "none",
    load_device: str | None = None,
    native_inference: bool = False,
    native_embeddings: bool = False,
    vision_attention: str = "sdpa",
    prefix_cuda_graph: bool = False,
    denoise_attention: str = "sdpa",
    prefix_attention: str = "sdpa",
    low_cpu_mem_usage: bool = False,
    checkpoint_config: Mapping[str, Any] | None = None,
    quantization: str | Mapping[str, Any] | QuantizationConfig | None = None,
    **overrides,
) -> VLAPolicy:
    if checkpoint is None:
        raise ValueError("pi0.5 needs a checkpoint, e.g. make_policy('pi05', checkpoint='lerobot/pi05_base')")
    tensor_parallel_size = overrides.pop("tensor_parallel_size", 1)
    tensor_parallel_group = overrides.pop("tensor_parallel_group", None)
    cfg = VLAPolicyConfig(**{"name": "pi0.5", **overrides})
    policy = Pi05Policy(
        cfg,
        checkpoint=checkpoint,
        attention=attention,
        compile_backend=compile_backend,
        load_device=load_device,
        native_inference=native_inference,
        native_embeddings=native_embeddings,
        vision_attention=vision_attention,
        prefix_cuda_graph=prefix_cuda_graph,
        denoise_attention=denoise_attention,
        prefix_attention=prefix_attention,
        low_cpu_mem_usage=low_cpu_mem_usage,
        checkpoint_config=checkpoint_config,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_group=tensor_parallel_group,
    )
    native = policy._lerobot.config
    for key, value in (
        ("action_dim", native.max_action_dim),
        ("action_horizon", native.chunk_size),
        ("default_num_steps", native.num_inference_steps),
    ):
        if key in overrides and overrides[key] != value and key != "default_num_steps":
            raise ValueError(f"pi05 {key} override conflicts with checkpoint value {value}")
        setattr(cfg, key, overrides.get(key, value))
    return policy
