"""Batched Qwen2 execution using the reusable ActiveVLN inference operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from ...backend.torch.linear import linear_fp32_output
from ...backend.triton.decode_linear import decode_linear, decode_qkv, decode_swiglu
from ...backend.triton.gqa_split_attention import gqa_split_kv_attention
from ...backend.triton.rounded_ops import rounded_rms_norm, rounded_rope, rounded_swiglu
from ...backend.triton.split_attention import split_kv_attention, write_batched_kv
from ..linear import QuantizedLinear

if TYPE_CHECKING:
    from .backbone import StreamVLNBackbone, StreamVLNCache


@dataclass(frozen=True)
class StreamVLNTextOptions:
    """Independent experimental kernels; all-disabled preserves the existing path."""

    enabled: bool = False
    fused_ops: bool = False
    split_attention: bool = False
    attention_kernel: str = "gqa"
    decode_linear: bool = False
    tree_fp32_projection: bool = False

    def __post_init__(self) -> None:
        if self.attention_kernel not in ("gqa", "split"):
            raise ValueError("attention_kernel must be gqa or split")
        if self.decode_linear and self.tree_fp32_projection:
            raise ValueError("decode_linear and tree_fp32_projection are alternative projections")


def _parts(module: nn.Module) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor | None, ...]]:
    sizes = module.split_sizes
    weights = module.weight.split(sizes, dim=0)
    biases = (None,) * len(sizes) if module.bias is None else module.bias.split(sizes, dim=0)
    return weights, biases


def project(
    module: nn.Module, inputs: torch.Tensor, options: StreamVLNTextOptions, *, tree: bool = False
) -> torch.Tensor:
    """Project every candidate row together; never fall back to per-token GEMMs."""
    if isinstance(module, QuantizedLinear):
        if tree and options.tree_fp32_projection:
            raise ValueError("FP32-output tree projection requires floating-point weights")
        return module(inputs)
    operation = F.linear
    if tree and options.tree_fp32_projection:
        operation = linear_fp32_output
    elif options.decode_linear:
        operation = decode_linear
    if hasattr(module, "split_sizes"):
        weights, biases = _parts(module)
        return torch.cat([operation(inputs, w, b) for w, b in zip(weights, biases, strict=True)], dim=-1)
    return operation(inputs, module.weight, getattr(module, "bias", None))


def _norm(module: nn.Module, hidden: torch.Tensor, options: StreamVLNTextOptions) -> torch.Tensor:
    epsilon = float(module.variance_epsilon)
    if options.fused_ops:
        return rounded_rms_norm(hidden.contiguous(), module.weight, epsilon)
    values = hidden.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)).to(
        hidden.dtype
    ) * module.weight


def forward_text(
    backbone: StreamVLNBackbone,
    hidden: torch.Tensor,
    cache: StreamVLNCache,
    position: torch.Tensor,
    *,
    key_bucket: int,
    attention_plan_bucket: int | None = None,
    depths: torch.Tensor | None = None,
    ancestors: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
    cache_starts: torch.Tensor | None = None,
    cache_capacities: torch.Tensor | None = None,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    rotary_positions: torch.Tensor | None = None,
    packed_live_tokens: torch.Tensor | None = None,
    packed_padding: int = 0,
) -> torch.Tensor:
    """Append causal/tree K/V physically; the caller commits the logical prefix.

    ``position`` gives each row's prefix length. Tree nodes are written in node
    order but use ancestor masks and depth-based rotary coordinates. Returns
    all normalized states. No hidden state, graph or session memory is retained.
    """
    options = backbone.text_options
    if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
        raise ValueError("optimized StreamVLN text execution requires CUDA BF16")
    backbone.fuse_projections()
    batch, query, _ = hidden.shape
    if depths is None:
        depths = torch.arange(query, device=hidden.device)[None].expand(batch, -1)
    positions = position.reshape(-1, 1) + depths if rotary_positions is None else rotary_positions
    if (packed is not None or cache_starts is not None) and not (
        options.split_attention and options.attention_kernel == "gqa"
    ):
        raise ValueError("pooled/packed StreamVLN execution requires grouped-query split attention")
    cosine = backbone._rope_cosine[positions].to(hidden.dtype).contiguous()
    sine = backbone._rope_sine[positions].to(hidden.dtype).contiguous()
    tree = ancestors is not None
    for layer_index, layer in enumerate(backbone.llm.model.layers):
        normalized = _norm(layer.input_layernorm, hidden, options)
        attention = layer.self_attn
        qkv = attention.qkv_proj
        if options.decode_linear and not isinstance(qkv, QuantizedLinear):
            weights, biases = _parts(qkv)
            queries, keys, values = decode_qkv(normalized, weights, biases)
        else:
            queries, keys, values = project(qkv, normalized, options, tree=tree).split(
                qkv.split_sizes, dim=-1
            )
        queries = queries.view(batch, query, backbone.num_heads, backbone.head_dim).transpose(1, 2)
        keys = keys.view(batch, query, backbone.num_kv_heads, backbone.head_dim).transpose(1, 2)
        values = values.view(batch, query, backbone.num_kv_heads, backbone.head_dim).transpose(1, 2)
        if options.fused_ops:
            queries, keys = rounded_rope(queries, keys, cosine, sine)
        else:
            cos, sin = cosine[:, None], sine[:, None]
            qfirst, qsecond = queries.chunk(2, dim=-1)
            kfirst, ksecond = keys.chunk(2, dim=-1)
            queries = queries * cos + torch.cat((-qsecond, qfirst), dim=-1) * sin
            keys = keys * cos + torch.cat((-ksecond, kfirst), dim=-1) * sin
        key_cache, value_cache = cache.key_storage[layer_index], cache.value_storage[layer_index]
        offsets = position.expand(batch).contiguous() if packed is None else position
        layout = dict(cache_starts=cache_starts, cache_capacities=cache_capacities)
        if packed is None:
            write_batched_kv(
                keys, values, key_cache, value_cache, offsets, write_lengths=query_lengths, **layout
            )
        else:
            # Live slots belong to private rows. Optional trailing padding
            # uses unique slots in a separate region owned by the caller.
            key_cache.index_copy_(2, packed[3], keys)
            value_cache.index_copy_(2, packed[3], values)
        if options.split_attention:
            kernel = gqa_split_kv_attention if options.attention_kernel == "gqa" else split_kv_attention
            extra = {"query_lengths": query_lengths, **layout} if options.attention_kernel == "gqa" else {}
            if attention_plan_bucket is not None:
                from ...backend.triton.gqa_split_attention import gqa_launch_config

                if options.attention_kernel != "gqa" or packed is not None:
                    raise ValueError("independent attention plans require unpacked grouped-query attention")
                # Growing physical KV must not change the split/reduction order.
                extra["splits"] = gqa_launch_config(query * backbone.num_kv_groups, attention_plan_bucket)[4]
            if packed is not None:
                extra["packed"] = packed[:3]
                if packed_live_tokens is not None:
                    extra.update(packed_live_tokens=packed_live_tokens, packed_padding=packed_padding)
            attended = kernel(
                queries, key_cache, value_cache, position, key_bucket, ancestors=ancestors, **extra
            )
        else:
            columns = torch.arange(key_bucket, device=hidden.device)
            local = columns[None, :] - position.reshape(-1, 1)
            if ancestors is None:
                allowed = local[:, None, :] <= torch.arange(query, device=hidden.device)[None, :, None]
            else:
                edges = ancestors.expand(batch, -1, -1) if ancestors.ndim == 2 else ancestors
                visible = edges.gather(2, local.clamp(0, query - 1)[:, None].expand(-1, query, -1))
                allowed = (local[:, None] < 0) | ((local[:, None] < query) & visible)
            if query_lengths is not None:
                allowed = allowed & (local[:, None] < query_lengths[:, None, None])
            attended = F.scaled_dot_product_attention(
                queries,
                key_cache[:, :, :key_bucket],
                value_cache[:, :, :key_bucket],
                attn_mask=allowed[:, None],
                dropout_p=0.0,
                enable_gqa=True,
            )
        attended = attended.transpose(1, 2).reshape(batch, query, backbone.hidden_size)
        hidden = hidden + project(attention.o_proj, attended, options, tree=tree)
        normalized = _norm(layer.post_attention_layernorm, hidden, options)
        gate_up = layer.mlp.gate_up_proj
        if options.decode_linear and options.fused_ops and not isinstance(gate_up, QuantizedLinear):
            weights, biases = _parts(gate_up)
            if any(b is not None for b in biases):
                raise ValueError("fused StreamVLN gated projection requires bias-free MLP")
            gated = decode_swiglu(normalized, *weights)
        else:
            gate, up = project(gate_up, normalized, options, tree=tree).chunk(2, dim=-1)
            gated = rounded_swiglu(gate, up) if options.fused_ops else F.silu(gate) * up
            # These views otherwise retain the entire fused projection into the
            # next layer, including during large slow-memory prefills.
            del gate, up
        hidden = hidden + project(layer.mlp.down_proj, gated, options, tree=tree)
        del gated, normalized
    return _norm(backbone.llm.model.norm, hidden, options)


def encode_siglip(backbone: StreamVLNBackbone, pixel_values: torch.Tensor, *, fused: bool) -> torch.Tensor:
    """Extract the configured SigLIP layer using batched device attention.

    Stops at the selected hidden layer, avoiding unused final encoder/head work.
    The shared segmented-attention kernel supports SigLIP's non-power-of-two
    head width by padding only internal tiles.
    """
    from ...backend.triton.vision_attention import segment_attention

    vision = backbone.vision_tower.vision_model
    layers = vision.encoder.layers
    selected = backbone.vision_feature_layer
    selected = selected if selected >= 0 else len(layers) + 1 + selected
    if not 0 <= selected <= len(layers):
        raise ValueError("invalid SigLIP hidden-state selection")
    hidden = vision.embeddings(pixel_values)
    batch, tokens, _ = hidden.shape
    offsets = torch.arange(batch + 1, device=hidden.device, dtype=torch.int32) * tokens
    for layer in layers[:selected]:
        normalized = layer.layer_norm1(hidden)
        attention = layer.self_attn
        heads, dim = attention.num_heads, attention.head_dim
        q, k, v = [
            p(normalized).view(batch, tokens, heads, dim)
            for p in (attention.q_proj, attention.k_proj, attention.v_proj)
        ]
        if fused:
            output = segment_attention(
                q.reshape(-1, heads, dim).contiguous(),
                k.reshape(-1, heads, dim).contiguous(),
                v.reshape(-1, heads, dim).contiguous(),
                offsets,
                tokens,
                scale=attention.scale,
            ).reshape(batch, tokens, -1)
        else:
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0, scale=attention.scale
            )
            output = output.transpose(1, 2).reshape(batch, tokens, -1)
        hidden = hidden + attention.out_proj(output)
        hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
    return hidden
