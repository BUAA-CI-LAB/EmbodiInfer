"""VVLA-owned Qwen2 forward, KV cache, vision pooling, and token splice."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from embodiinfer.backend.triton import (
    add_rms_norm,
    fused_rope_cache,
    fused_rope_cache_graph,
    gqa_decode_graph,
    rms_norm,
    supports_fused_rope_cache,
    supports_rms_norm,
    supports_swiglu,
    swiglu,
)
from embodiinfer.backend.triton.prefill_attention import (
    flash_prefill_attention,
    flash_prefill_attention_graph,
)
from embodiinfer.backend.triton.rotary import fused_prefill_rope_cache_graph
from embodiinfer.models.linear import QuantizedLinear

IMAGE_TOKEN_INDEX = -200
MEMORY_TOKEN_INDEX = -300


@dataclass(frozen=True)
class StreamVLNCache:
    """Append-only Qwen2 state backed by fixed-address key/value storage."""

    key_storage: torch.Tensor
    value_storage: torch.Tensor
    length: int

    @property
    def seq_len(self) -> int:
        return self.length

    @property
    def capacity(self) -> int:
        return int(self.key_storage.shape[-2])

    def advance(self, length: int) -> StreamVLNCache:
        if length > self.capacity:
            raise ValueError(f"StreamVLN cache capacity exceeded: {length} > {self.capacity}")
        return StreamVLNCache(self.key_storage, self.value_storage, length)

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> StreamVLNCache:
        return StreamVLNCache(
            self.key_storage.to(
                device=device,
                dtype=dtype or self.key_storage.dtype,
            ),
            self.value_storage.to(
                device=device,
                dtype=dtype or self.value_storage.dtype,
            ),
            self.length,
        )


class StreamVLNProjector(nn.Sequential):
    """Published two-layer GELU projector from SigLIP to Qwen2 width."""

    def __init__(self, vision_hidden_size: int = 1152, text_hidden_size: int = 3584) -> None:
        super().__init__(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )


class _PackedLinear(nn.Module):
    """Store compatible projections contiguously without duplicating weights."""

    def __init__(self, *projections: nn.Module) -> None:
        super().__init__()
        self.split_sizes = tuple(int(projection.weight.shape[0]) for projection in projections)
        with torch.inference_mode(False), torch.no_grad():
            weight = torch.cat([projection.weight.detach() for projection in projections], dim=0)
            biases = [getattr(projection, "bias", None) for projection in projections]
            bias = None
            if any(value is not None for value in biases):
                bias = torch.cat(
                    [
                        value.detach()
                        if value is not None
                        else projection.weight.new_zeros(projection.weight.shape[0])
                        for projection, value in zip(projections, biases, strict=True)
                    ],
                    dim=0,
                )
        requires_grad = any(projection.weight.requires_grad for projection in projections)
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)
        self.bias = nn.Parameter(bias, requires_grad=requires_grad) if bias is not None else None


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(module, QuantizedLinear):
        return module(inputs)
    if isinstance(module, _PackedLinear):
        weights = module.weight.split(module.split_sizes, dim=0)
        biases = (
            (None,) * len(weights) if module.bias is None else module.bias.split(module.split_sizes, dim=0)
        )
        return torch.cat(
            [F.linear(inputs, weight, bias) for weight, bias in zip(weights, biases, strict=True)],
            dim=-1,
        )
    return F.linear(inputs, module.weight, getattr(module, "bias", None))


def _rms_norm(inputs: torch.Tensor, module: nn.Module, default_eps: float) -> torch.Tensor:
    if supports_rms_norm(inputs, module.weight):
        return rms_norm(
            inputs,
            module.weight,
            float(getattr(module, "variance_epsilon", default_eps)),
        )
    dtype = inputs.dtype
    eps = float(getattr(module, "variance_epsilon", default_eps))
    normalized = inputs.float() * torch.rsqrt(inputs.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized.to(dtype=dtype) * module.weight


def _add_rms_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    module: nn.Module,
    default_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon = float(getattr(module, "variance_epsilon", default_eps))
    if supports_rms_norm(residual, module.weight) and update.is_contiguous():
        return add_rms_norm(residual, update, module.weight, epsilon)
    hidden = residual + update
    return hidden, _rms_norm(hidden, module, default_eps)


def _compiled_prefill_block_tail(
    residual: torch.Tensor,
    attended: torch.Tensor,
    post_attention_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    next_norm_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = residual + attended
    normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    normalized = normalized.to(dtype=hidden.dtype) * post_attention_weight
    gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
    gate = F.linear(normalized, gate_weight)
    up = F.linear(normalized, up_weight)
    down = F.linear(F.silu(gate) * up, down_weight)
    hidden = hidden + down
    normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return hidden, normalized.to(dtype=hidden.dtype) * next_norm_weight


def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class StreamVLNBackbone(nn.Module):
    """Execute the published StreamVLN weights without importing its runtime."""

    def __init__(
        self,
        llm: nn.Module,
        vision_tower: nn.Module,
        projector: StreamVLNProjector,
        *,
        max_context: int = 32768,
        vision_feature_layer: int = -2,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.vision_tower = vision_tower
        self.projector = projector
        self.max_context = max_context
        self.vision_feature_layer = vision_feature_layer

        config = llm.config
        if getattr(config, "hidden_act", "silu") not in {"silu", "swish"}:
            raise ValueError("StreamVLN's self-hosted Qwen2 forward requires SwiGLU")
        if bool(getattr(config, "use_sliding_window", False)):
            raise ValueError("StreamVLN uses explicit 32-step resets, not sliding-window attention")
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = float(config.rope_theta)
        self.rms_norm_eps = float(config.rms_norm_eps)
        frequencies = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        angles = torch.outer(
            torch.arange(self.max_context, dtype=torch.float32),
            frequencies,
        )
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("_rope_cosine", angles.cos(), persistent=False)
        self.register_buffer("_rope_sine", angles.sin(), persistent=False)
        self._projections_fused = False
        self._quantization_enabled = False
        self._causal_mask_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}
        self._prefill_optimizations = False
        self._language_prefill_optimizations = False
        self._vision_runner = None
        self._prefill_runner = None
        self._startup_capture_active = False
        self._startup_capture_frozen = False
        self._compiled_prefill_tail = None
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("invalid StreamVLN Qwen2 hidden/head dimensions")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("Qwen2 query heads must be divisible by KV heads")

    def _rope(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        *,
        start_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end_position = start_position + queries.shape[-2]
        cosine = self._rope_cosine[start_position:end_position].to(
            device=queries.device,
            dtype=queries.dtype,
        )[None, None]
        sine = self._rope_sine[start_position:end_position].to(
            device=queries.device,
            dtype=queries.dtype,
        )[None, None]
        return (
            queries * cosine + _rotate_half(queries) * sine,
            keys * cosine + _rotate_half(keys) * sine,
        )

    def fuse_projections(self) -> None:
        """Pack QKV and gate/up weights once before optimized or quantized execution."""
        if self._projections_fused:
            return
        for layer in self.llm.model.layers:
            attention = layer.self_attn
            attention.qkv_proj = _PackedLinear(
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
            )
            del attention.q_proj
            del attention.k_proj
            del attention.v_proj

            mlp = layer.mlp
            mlp.gate_up_proj = _PackedLinear(mlp.gate_proj, mlp.up_proj)
            del mlp.gate_proj
            del mlp.up_proj
        self._projections_fused = True

    def configure_quantized_projections(self, names: tuple[str, ...]) -> None:
        """Record policy-selected quantized projections and invalidate compiled prefill."""
        self.quantized_layers = names
        self._quantization_enabled = True
        self._language_prefill_optimizations = False
        self._prefill_runner = None
        self._compiled_prefill_tail = None

    def _suffix_causal_mask(
        self,
        query_length: int,
        past_length: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if past_length == 0 or query_length == 1:
            return None
        key = (device, query_length, past_length)
        cached = self._causal_mask_cache.get(key)
        if cached is not None:
            return cached
        cached = torch.ones(
            (query_length, past_length + query_length),
            dtype=torch.bool,
            device=device,
        ).tril(diagonal=past_length)[None, None]
        self._causal_mask_cache[key] = cached
        return cached

    def configure_prefill_optimizations(
        self,
        enabled: bool,
        *,
        language_prefill: bool = True,
    ) -> None:
        self._prefill_optimizations = bool(enabled)
        self._language_prefill_optimizations = bool(
            enabled and language_prefill and not self._quantization_enabled
        )
        if not self._language_prefill_optimizations:
            self._prefill_runner = None
        if not enabled:
            self._vision_runner = None
            self._prefill_runner = None
            self._startup_capture_active = False
            self._startup_capture_frozen = False

    def begin_startup_graph_capture(self) -> None:
        """Enter the only phase in which native StreamVLN graphs may be created."""
        if not self._prefill_optimizations:
            raise RuntimeError("StreamVLN prefill graph optimizations are disabled")
        if self._startup_capture_active:
            raise RuntimeError("StreamVLN startup graph capture is already active")
        if self._startup_capture_frozen:
            raise RuntimeError("StreamVLN startup graph capture is already frozen")
        self._startup_capture_active = True
        for runner in (self._vision_runner, self._prefill_runner):
            if runner is not None:
                runner.begin_startup_capture()

    def finish_startup_graph_capture(self) -> None:
        """Freeze the startup graph set so inference can only replay it."""
        if not self._startup_capture_active:
            raise RuntimeError("StreamVLN startup graph capture is not active")
        for runner in (self._vision_runner, self._prefill_runner):
            if runner is not None:
                runner.finish_startup_capture()
        self._startup_capture_active = False
        self._startup_capture_frozen = True

    def abort_startup_graph_capture(self) -> None:
        """Leave a failed startup capture without allowing the partial set at runtime."""
        for runner in (self._vision_runner, self._prefill_runner):
            if runner is not None:
                runner.abort_startup_capture()
        self._vision_runner = None
        self._prefill_runner = None
        self._startup_capture_active = False
        self._startup_capture_frozen = False

    def graph_capture_stats(self) -> dict[str, int | bool]:
        return {
            "active": self._startup_capture_active,
            "frozen": self._startup_capture_frozen,
            "vision": 0 if self._vision_runner is None else len(self._vision_runner.graphs),
            "prefill": 0 if self._prefill_runner is None else len(self._prefill_runner.graphs),
        }

    def graph_runtime_stats(self) -> dict[str, int]:
        return {
            "vision_replays": (0 if self._vision_runner is None else self._vision_runner.graph_replays),
            "vision_eager_fallbacks": (
                0 if self._vision_runner is None else self._vision_runner.eager_fallbacks
            ),
            "prefill_replays": (0 if self._prefill_runner is None else self._prefill_runner.graph_replays),
            "prefill_eager_fallbacks": (
                0 if self._prefill_runner is None else self._prefill_runner.eager_fallbacks
            ),
        }

    def reset_graph_runtime_stats(self) -> None:
        for runner in (self._vision_runner, self._prefill_runner):
            if runner is not None:
                runner.reset_runtime_stats()

    def _attention(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        cache: StreamVLNCache,
        layer_index: int,
        past_length: int,
    ) -> torch.Tensor:
        batch, query_length, _ = hidden.shape
        queries, keys, values = _linear(attention.qkv_proj, hidden).split(
            attention.qkv_proj.split_sizes,
            dim=-1,
        )
        queries = queries.view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cache_end = past_length + query_length
        key_cache = cache.key_storage[layer_index]
        value_cache = cache.value_storage[layer_index]
        if supports_fused_rope_cache(queries, keys, values, key_cache, value_cache):
            queries = fused_rope_cache(
                queries,
                keys,
                values,
                self._rope_cosine[past_length],
                self._rope_sine[past_length],
                key_cache,
                value_cache,
                past_length,
            )
        else:
            queries, keys = self._rope(queries, keys, start_position=past_length)
            key_cache[..., past_length:cache_end, :].copy_(keys)
            value_cache[..., past_length:cache_end, :].copy_(values)
        keys = cache.key_storage[layer_index, ..., :cache_end, :]
        values = cache.value_storage[layer_index, ..., :cache_end, :]
        attention_mask = self._suffix_causal_mask(
            query_length,
            past_length,
            device=hidden.device,
        )
        if query_length == 1:
            keys = keys.repeat_interleave(self.num_kv_groups, dim=1)
            values = values.repeat_interleave(self.num_kv_groups, dim=1)
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attended = flash_prefill_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=past_length == 0,
                enable_gqa=True,
            )
        attended = attended.transpose(1, 2).reshape(batch, query_length, self.hidden_size)
        return _linear(attention.o_proj, attended)

    def _allocate_cache(self, inputs: torch.Tensor) -> StreamVLNCache:
        shape = (
            len(self.llm.model.layers),
            inputs.shape[0],
            self.num_kv_heads,
            self.max_context,
            self.head_dim,
        )
        return StreamVLNCache(
            torch.empty(shape, dtype=inputs.dtype, device=inputs.device),
            torch.empty(shape, dtype=inputs.dtype, device=inputs.device),
            0,
        )

    def _attention_graph(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        cache: StreamVLNCache,
        layer_index: int,
        position: torch.Tensor,
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        queries, keys, values = _linear(attention.qkv_proj, hidden).split(
            attention.qkv_proj.split_sizes,
            dim=-1,
        )
        queries = queries.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        key_cache = cache.key_storage[layer_index]
        value_cache = cache.value_storage[layer_index]
        queries = fused_rope_cache_graph(
            queries,
            keys,
            values,
            self._rope_cosine,
            self._rope_sine,
            key_cache,
            value_cache,
            position,
        )
        attended = gqa_decode_graph(queries, key_cache, value_cache, position)
        attended = attended.transpose(1, 2).reshape(batch, 1, self.hidden_size)
        return _linear(attention.o_proj, attended)

    def decode_token_graph(
        self,
        cache: StreamVLNCache,
        token: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        self.fuse_projections()
        hidden = self.llm.model.embed_tokens(token.reshape(1, 1))
        normalized = _rms_norm(
            hidden,
            self.llm.model.layers[0].input_layernorm,
            self.rms_norm_eps,
        )
        for index, layer in enumerate(self.llm.model.layers):
            attended = self._attention_graph(
                layer.self_attn,
                normalized,
                cache,
                index,
                position,
            )
            hidden, normalized = _add_rms_norm(
                hidden,
                attended,
                layer.post_attention_layernorm,
                self.rms_norm_eps,
            )
            gate_up = _linear(layer.mlp.gate_up_proj, normalized)
            gated = swiglu(gate_up)
            down = _linear(layer.mlp.down_proj, gated)
            next_norm = (
                self.llm.model.layers[index + 1].input_layernorm
                if index + 1 < len(self.llm.model.layers)
                else self.llm.model.norm
            )
            hidden, normalized = _add_rms_norm(
                hidden,
                down,
                next_norm,
                self.rms_norm_eps,
            )
        return normalized[:, -1]

    def _prefill_attention_graph(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        cache: StreamVLNCache,
        layer_index: int,
        position: torch.Tensor,
        key_bucket: int,
    ) -> torch.Tensor:
        batch, query_length, _ = hidden.shape
        queries, keys, values = _linear(attention.qkv_proj, hidden).split(
            attention.qkv_proj.split_sizes,
            dim=-1,
        )
        queries = queries.view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        key_cache = cache.key_storage[layer_index]
        value_cache = cache.value_storage[layer_index]
        queries = fused_prefill_rope_cache_graph(
            queries,
            keys,
            values,
            self._rope_cosine,
            self._rope_sine,
            key_cache,
            value_cache,
            position,
        )
        attended = flash_prefill_attention_graph(
            queries,
            key_cache,
            value_cache,
            position,
            key_bucket,
        )
        attended = attended.transpose(1, 2).reshape(
            batch,
            query_length,
            self.hidden_size,
        )
        return _linear(attention.o_proj, attended)

    def prefill_embeddings_graph(
        self,
        inputs_embeds: torch.Tensor,
        cache: StreamVLNCache,
        position: torch.Tensor,
        key_bucket: int,
    ) -> torch.Tensor:
        self.fuse_projections()
        hidden = inputs_embeds
        normalized = _rms_norm(
            hidden,
            self.llm.model.layers[0].input_layernorm,
            self.rms_norm_eps,
        )
        for index, layer in enumerate(self.llm.model.layers):
            attended = self._prefill_attention_graph(
                layer.self_attn,
                normalized,
                cache,
                index,
                position,
                key_bucket,
            )
            mlp = layer.mlp
            next_norm = (
                self.llm.model.layers[index + 1].input_layernorm
                if index + 1 < len(self.llm.model.layers)
                else self.llm.model.norm
            )
            if self._compiled_prefill_tail is None:
                self._compiled_prefill_tail = torch.compile(
                    _compiled_prefill_block_tail,
                    dynamic=True,
                    fullgraph=True,
                    mode="max-autotune-no-cudagraphs",
                )
            hidden, normalized = self._compiled_prefill_tail(
                hidden,
                attended,
                layer.post_attention_layernorm.weight,
                mlp.gate_up_proj.weight,
                mlp.down_proj.weight,
                next_norm.weight,
                self.rms_norm_eps,
            )
        return normalized

    def _forward_embeddings_eager(
        self,
        inputs_embeds: torch.Tensor,
        cache: StreamVLNCache | None = None,
        *,
        project_logits: bool = True,
        return_hidden_states: bool = False,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise ValueError("StreamVLN requires embeddings shaped [1, T, D]")
        past_length = 0 if cache is None else cache.seq_len
        if past_length + inputs_embeds.shape[1] > self.max_context:
            raise RuntimeError("StreamVLN fast window exceeded the Qwen2 context limit")
        if cache is not None:
            expected_prefix = (
                len(self.llm.model.layers),
                inputs_embeds.shape[0],
                self.num_kv_heads,
            )
            if tuple(cache.key_storage.shape[:3]) != expected_prefix:
                raise ValueError("KV cache shape does not match the Qwen2 backbone")
            if cache.key_storage.shape != cache.value_storage.shape:
                raise ValueError("KV key/value cache shapes do not match")

        self.fuse_projections()
        hidden = inputs_embeds
        working_cache = self._allocate_cache(hidden) if cache is None else cache
        normalized = _rms_norm(
            hidden,
            self.llm.model.layers[0].input_layernorm,
            self.rms_norm_eps,
        )
        for index, layer in enumerate(self.llm.model.layers):
            attended = self._attention(
                layer.self_attn,
                normalized,
                working_cache,
                index,
                past_length,
            )
            mlp = layer.mlp
            next_norm = (
                self.llm.model.layers[index + 1].input_layernorm
                if index + 1 < len(self.llm.model.layers)
                else self.llm.model.norm
            )
            if self._language_prefill_optimizations and inputs_embeds.shape[1] > 1:
                if self._compiled_prefill_tail is None:
                    self._compiled_prefill_tail = torch.compile(
                        _compiled_prefill_block_tail,
                        dynamic=True,
                        fullgraph=True,
                        mode="max-autotune-no-cudagraphs",
                    )
                hidden, normalized = self._compiled_prefill_tail(
                    hidden,
                    attended,
                    layer.post_attention_layernorm.weight,
                    mlp.gate_up_proj.weight,
                    mlp.down_proj.weight,
                    next_norm.weight,
                    self.rms_norm_eps,
                )
                continue
            hidden, normalized = _add_rms_norm(
                hidden,
                attended,
                layer.post_attention_layernorm,
                self.rms_norm_eps,
            )
            gate_up = _linear(mlp.gate_up_proj, normalized)
            if supports_swiglu(gate_up):
                gated = swiglu(gate_up)
            else:
                gate, up = gate_up.split(mlp.gate_up_proj.split_sizes, dim=-1)
                gated = F.silu(gate) * up
            down = _linear(mlp.down_proj, gated)
            hidden, normalized = _add_rms_norm(
                hidden,
                down,
                next_norm,
                self.rms_norm_eps,
            )

        if project_logits and return_hidden_states:
            raise ValueError("cannot project logits while returning all hidden states")
        if return_hidden_states:
            output = normalized
        else:
            output = _linear(self.llm.lm_head, normalized[:, -1]) if project_logits else normalized[:, -1]
        return working_cache.advance(past_length + inputs_embeds.shape[1]), output

    def forward_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        cache: StreamVLNCache | None = None,
        *,
        project_logits: bool = True,
        return_hidden_states: bool = False,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        if (
            self._language_prefill_optimizations
            and not project_logits
            and inputs_embeds.is_cuda
            and inputs_embeds.shape[1] > 1
        ):
            if self._prefill_runner is None:
                from .prefill import StreamVLNPrefillRunner

                self.fuse_projections()
                self._prefill_runner = StreamVLNPrefillRunner(self)
                if self._startup_capture_active:
                    self._prefill_runner.begin_startup_capture()
            working_cache, hidden_states = self._prefill_runner.run(inputs_embeds, cache)
            hidden = hidden_states if return_hidden_states else hidden_states[:, -1]
            return working_cache, hidden
        return self._forward_embeddings_eager(
            inputs_embeds,
            cache,
            project_logits=project_logits,
            return_hidden_states=return_hidden_states,
        )

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return _linear(self.llm.lm_head, hidden)

    def _encode_frames_eager(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_parameter = next(self.vision_tower.parameters())
        pixel_values = pixel_values.to(
            device=vision_parameter.device,
            dtype=vision_parameter.dtype,
        )
        output = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        selected = output.hidden_states[self.vision_feature_layer]
        projected = self.projector(selected)
        frame_count, token_count, hidden_size = projected.shape
        side = math.isqrt(token_count)
        if side * side != token_count:
            raise ValueError(f"StreamVLN vision tokens must form a square grid, got {token_count}")
        grid = projected.reshape(frame_count, side, side, hidden_size).permute(0, 3, 1, 2)
        pooled_side = math.ceil(side / 2)
        pooled = F.interpolate(
            grid.float(),
            size=(pooled_side, pooled_side),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=projected.dtype)
        return pooled.permute(0, 2, 3, 1).reshape(
            frame_count,
            pooled_side * pooled_side,
            hidden_size,
        )

    def encode_frames(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_parameter = next(self.vision_tower.parameters())
        pixel_values = pixel_values.to(
            device=vision_parameter.device,
            dtype=vision_parameter.dtype,
        )
        if self._prefill_optimizations and pixel_values.is_cuda:
            if self._vision_runner is None:
                from .prefill import StreamVLNVisionRunner

                self._vision_runner = StreamVLNVisionRunner(self)
                if self._startup_capture_active:
                    self._vision_runner.begin_startup_capture()
            return self._vision_runner.run(pixel_values)
        return self._encode_frames_eager(pixel_values)

    def prepare_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        current_pixel_values: torch.Tensor,
        memory_pixel_values: torch.Tensor | None,
        *,
        image_position: int | None = None,
        memory_position: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("StreamVLN requires input_ids shaped [1, T]")
        image_positions = (
            [image_position]
            if image_position is not None
            else torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
        )
        memory_positions = (
            [memory_position]
            if memory_position is not None
            else torch.where(input_ids[0] == MEMORY_TOKEN_INDEX)[0].tolist()
        )
        if len(image_positions) != 1 or len(memory_positions) > 1:
            raise ValueError("StreamVLN prompt must contain one image and at most one memory sentinel")
        if bool(memory_positions) != (memory_pixel_values is not None):
            raise ValueError("StreamVLN memory sentinel and slow-frame tensor disagree")

        all_pixels = current_pixel_values
        memory_count = 0
        if memory_pixel_values is not None:
            memory_count = int(memory_pixel_values.shape[0])
            all_pixels = torch.cat((memory_pixel_values, current_pixel_values), dim=0)
        frame_features = self.encode_frames(all_pixels)
        memory_features = (
            frame_features[:memory_count].reshape(-1, self.hidden_size) if memory_count else None
        )
        current_features = frame_features[-1]

        return self.prepare_multimodal_feature_embeddings(
            input_ids,
            current_features,
            memory_features,
            image_position=image_position,
            memory_position=memory_position,
        )

    def prepare_multimodal_feature_embeddings(
        self,
        input_ids: torch.Tensor,
        current_features: torch.Tensor,
        memory_features: torch.Tensor | None,
        *,
        image_position: int | None = None,
        memory_position: int | None = None,
    ) -> torch.Tensor:
        """Splice pre-encoded current and slow-memory frame features into text."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("StreamVLN requires input_ids shaped [1, T]")
        if current_features.ndim != 2 or current_features.shape[-1] != self.hidden_size:
            raise ValueError("StreamVLN current frame features must be shaped [T, D]")
        if memory_features is not None:
            if memory_features.ndim == 3:
                memory_features = memory_features.reshape(-1, self.hidden_size)
            if memory_features.ndim != 2 or memory_features.shape[-1] != self.hidden_size:
                raise ValueError("StreamVLN memory features must be shaped [F, T, D] or [T, D]")

        image_positions = (
            [image_position]
            if image_position is not None
            else torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
        )
        memory_positions = (
            [memory_position]
            if memory_position is not None
            else torch.where(input_ids[0] == MEMORY_TOKEN_INDEX)[0].tolist()
        )
        if len(image_positions) != 1 or len(memory_positions) > 1:
            raise ValueError("StreamVLN prompt must contain one image and at most one memory sentinel")
        if bool(memory_positions) != (memory_features is not None):
            raise ValueError("StreamVLN memory sentinel and slow-frame features disagree")

        safe_ids = input_ids.masked_fill(input_ids < 0, 0)
        text_embeddings = self.llm.model.embed_tokens(safe_ids)
        current_features = current_features.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        if memory_features is not None:
            memory_features = memory_features.to(
                device=text_embeddings.device,
                dtype=text_embeddings.dtype,
            )

        pieces = []
        start = 0
        for position in sorted(image_positions + memory_positions):
            pieces.append(text_embeddings[:, start:position])
            if position in memory_positions:
                pieces.append(memory_features.unsqueeze(0))
            else:
                pieces.append(current_features.unsqueeze(0))
            start = position + 1
        pieces.append(text_embeddings[:, start:])
        return torch.cat(pieces, dim=1)

    def prefill_turn(
        self,
        input_ids: torch.Tensor,
        current_pixel_values: torch.Tensor,
        memory_pixel_values: torch.Tensor | None,
        cache: StreamVLNCache | None,
        *,
        project_logits: bool = True,
        image_position: int | None = None,
        memory_position: int | None = None,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        embeddings = self.prepare_multimodal_embeddings(
            input_ids,
            current_pixel_values,
            memory_pixel_values,
            image_position=image_position,
            memory_position=memory_position,
        )
        return self.forward_embeddings(embeddings, cache, project_logits=project_logits)

    def append_token(
        self,
        cache: StreamVLNCache,
        token: torch.Tensor,
        *,
        project_logits: bool = True,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        embeddings = self.llm.model.embed_tokens(token.reshape(1, 1))
        return self.forward_embeddings(embeddings, cache, project_logits=project_logits)


__all__ = [
    "IMAGE_TOKEN_INDEX",
    "MEMORY_TOKEN_INDEX",
    "StreamVLNBackbone",
    "StreamVLNCache",
    "StreamVLNProjector",
]
