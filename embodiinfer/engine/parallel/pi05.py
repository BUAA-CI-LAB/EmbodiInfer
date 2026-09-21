"""Tensor-parallel sharding plan for the two pi0.5 Gemma towers."""

from __future__ import annotations

from torch import nn

from .tensor_parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelContext


def _column(linear: nn.Linear, context: TensorParallelContext, name: str) -> nn.Module:
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"{name} must be nn.Linear before tensor parallelism, got {type(linear).__name__}")
    return ColumnParallelLinear.from_linear(linear, context, name=name)


def _row(linear: nn.Linear, context: TensorParallelContext, name: str) -> nn.Module:
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"{name} must be nn.Linear before tensor parallelism, got {type(linear).__name__}")
    return RowParallelLinear.from_linear(linear, context, name=name)


def parallelize_pi05_towers(policy, context: TensorParallelContext) -> None:
    """Shard Q/O and MLP projections while replicating the single MQA K/V head."""
    if not context.enabled:
        return

    tower_layers = {
        "prefix": policy._prefix_tower.layers,
        "expert": policy._expert_tower.layers,
    }
    for tower_name, layers in tower_layers.items():
        for layer_index, layer in enumerate(layers):
            prefix = f"{tower_name}.layers.{layer_index}"
            attention = layer.self_attn
            head_dim = int(attention.head_dim)
            q_heads = attention.q_proj.out_features // head_dim
            kv_heads = attention.k_proj.out_features // head_dim
            if q_heads % context.world_size:
                raise ValueError(
                    f"{prefix} has {q_heads} query heads, not divisible by TP size {context.world_size}"
                )
            if kv_heads != 1:
                raise NotImplementedError(
                    f"pi0.5 TP currently expects one replicated MQA KV head, got {kv_heads} in {prefix}"
                )

            attention.q_proj = _column(attention.q_proj, context, f"{prefix}.self_attn.q_proj")
            attention.o_proj = _row(attention.o_proj, context, f"{prefix}.self_attn.o_proj")
            mlp = layer.mlp
            mlp.gate_proj = _column(mlp.gate_proj, context, f"{prefix}.mlp.gate_proj")
            mlp.up_proj = _column(mlp.up_proj, context, f"{prefix}.mlp.up_proj")
            mlp.down_proj = _row(mlp.down_proj, context, f"{prefix}.mlp.down_proj")
