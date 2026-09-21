"""Inference-only tensor parallelism for Qwen2.5-VL navigation models.

The vision and language transformer blocks keep replicated residual streams.
Expansion projections are column-sharded; contraction projections are
row-sharded and all-reduced. Embeddings, norms, patch embedding, patch merger,
and LM head stay replicated, so block boundaries and final logits retain their
single-GPU shapes.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .tensor_parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelContext


class FusedQKVColumnParallelLinear(nn.Module):
    """Shard a vision fused Q/K/V projection by heads in each segment."""

    def __init__(self, linear: nn.Linear, context: TensorParallelContext) -> None:
        super().__init__()
        if linear.out_features % 3:
            raise ValueError("fused QKV output width must be divisible by three")
        segment_width = linear.out_features // 3
        if segment_width % context.world_size:
            raise ValueError(
                f"vision QKV width {segment_width} is not divisible by TP size {context.world_size}"
            )
        start, end = context.shard_bounds(segment_width, name="vision QKV segment")
        local_width = end - start
        weight = linear.weight.detach().reshape(3, segment_width, linear.in_features)
        self.weight = nn.Parameter(
            weight[:, start:end].reshape(3 * local_width, linear.in_features).contiguous(),
            requires_grad=linear.weight.requires_grad,
        )
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            bias = linear.bias.detach().reshape(3, segment_width)
            self.bias = nn.Parameter(
                bias[:, start:end].reshape(3 * local_width).contiguous(),
                requires_grad=linear.bias.requires_grad,
            )
        self.in_features = linear.in_features
        self.out_features = 3 * local_width
        self.segment_width = local_width
        self.context = context

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


def _parallelize_mlp(mlp: nn.Module, context: TensorParallelContext, *, name: str) -> None:
    if all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
        mlp.gate_proj = ColumnParallelLinear.from_linear(mlp.gate_proj, context, name=f"{name}.gate_proj")
        mlp.up_proj = ColumnParallelLinear.from_linear(mlp.up_proj, context, name=f"{name}.up_proj")
        mlp.down_proj = RowParallelLinear.from_linear(mlp.down_proj, context, name=f"{name}.down_proj")
        return
    if all(hasattr(mlp, attr) for attr in ("fc1", "fc2")):
        mlp.fc1 = ColumnParallelLinear.from_linear(mlp.fc1, context, name=f"{name}.fc1")
        mlp.fc2 = RowParallelLinear.from_linear(mlp.fc2, context, name=f"{name}.fc2")
        return
    raise TypeError(f"unsupported {name} MLP layout: {type(mlp).__name__}")


def _parallelize_vision(model: nn.Module, context: TensorParallelContext) -> None:
    visual = model.model.visual
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        raise TypeError("Qwen2.5-VL visual tower does not expose transformer blocks")
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None or not hasattr(attention, "qkv") or not hasattr(attention, "proj"):
            raise TypeError(f"unsupported Qwen2.5-VL vision attention at block {index}")
        global_heads = int(attention.num_heads)
        if global_heads % context.world_size:
            raise ValueError(f"vision heads {global_heads} are not divisible by TP size {context.world_size}")
        attention.qkv = FusedQKVColumnParallelLinear(attention.qkv, context)
        attention.proj = RowParallelLinear.from_linear(
            attention.proj, context, name=f"vision block {index}.attention.proj"
        )
        attention.num_heads = global_heads // context.world_size
        _parallelize_mlp(block.mlp, context, name=f"vision block {index}")


def _parallelize_language(model: nn.Module, context: TensorParallelContext) -> None:
    language_model = model.model.language_model
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise TypeError("Qwen2.5-VL language tower does not expose decoder layers")
    text_config = model.config.get_text_config(decoder=True)
    global_query_heads = int(text_config.num_attention_heads)
    global_kv_heads = int(text_config.num_key_value_heads)
    if global_query_heads % context.world_size or global_kv_heads % context.world_size:
        raise ValueError(
            "Qwen2.5-VL query/KV heads must both be divisible by TP size "
            f"{context.world_size} (got {global_query_heads}/{global_kv_heads})"
        )
    local_query_heads = global_query_heads // context.world_size
    local_kv_heads = global_kv_heads // context.world_size
    head_dim = int(getattr(text_config, "head_dim", text_config.hidden_size // global_query_heads))
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        attention.q_proj = ColumnParallelLinear.from_linear(
            attention.q_proj, context, name=f"language layer {index}.q_proj"
        )
        attention.k_proj = ColumnParallelLinear.from_linear(
            attention.k_proj, context, name=f"language layer {index}.k_proj"
        )
        attention.v_proj = ColumnParallelLinear.from_linear(
            attention.v_proj, context, name=f"language layer {index}.v_proj"
        )
        attention.o_proj = RowParallelLinear.from_linear(
            attention.o_proj, context, name=f"language layer {index}.o_proj"
        )
        if hasattr(attention, "num_heads"):
            attention.num_heads = local_query_heads
        if hasattr(attention, "num_key_value_heads"):
            attention.num_key_value_heads = local_kv_heads
        if hasattr(attention, "num_key_value_groups"):
            attention.num_key_value_groups = local_query_heads // local_kv_heads
        _parallelize_mlp(layer.mlp, context, name=f"language layer {index}")

    # StaticCache reads these values after model construction. Pin head_dim
    # before changing the head counts so it cannot be re-derived incorrectly.
    text_config.head_dim = head_dim
    text_config.num_attention_heads = local_query_heads
    text_config.num_key_value_heads = local_kv_heads


def parallelize_qwen25_vl(model: nn.Module, context: TensorParallelContext) -> nn.Module:
    """Apply in-place Qwen2.5-VL tensor parallelism and return the model."""
    if not context.enabled:
        return model
    if getattr(model, "_embodiinfer_tensor_parallel", None) is not None:
        raise RuntimeError("Qwen2.5-VL model is already tensor-parallel")
    _parallelize_vision(model, context)
    _parallelize_language(model, context)
    model._embodiinfer_tensor_parallel = context
    return model


__all__ = ["FusedQKVColumnParallelLinear", "parallelize_qwen25_vl"]
