"""Small inference-only tensor-parallel building blocks.

The wrappers deliberately keep hidden states replicated at transformer block
boundaries. Column-parallel projections produce a local feature shard; the
matching row-parallel projection consumes that shard and sums partial outputs
across the tensor-parallel process group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TensorParallelContext:
    """Rank metadata for one tensor-parallel process group."""

    world_size: int = 1
    rank: int = 0
    process_group: object | None = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @classmethod
    def from_distributed(
        cls,
        expected_size: int = 1,
        process_group: object | None = None,
    ) -> TensorParallelContext:
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
            raise ValueError("tensor_parallel_size must be a positive integer")
        if expected_size == 1:
            return cls()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "tensor_parallel_size > 1 requires an initialized torch.distributed process group"
            )
        world_size = dist.get_world_size(process_group)
        if world_size != expected_size:
            raise ValueError(
                f"tensor_parallel_size={expected_size} does not match process-group size {world_size}"
            )
        return cls(
            world_size=world_size,
            rank=dist.get_rank(process_group),
            process_group=process_group,
        )

    def shard_bounds(self, size: int, *, name: str) -> tuple[int, int]:
        if size % self.world_size:
            raise ValueError(f"{name}={size} must be divisible by TP size {self.world_size}")
        shard = size // self.world_size
        return self.rank * shard, (self.rank + 1) * shard


def _parameter(value: torch.Tensor, *, requires_grad: bool) -> nn.Parameter:
    return nn.Parameter(value.detach().clone(), requires_grad=requires_grad)


class ColumnParallelLinear(nn.Module):
    """Shard a linear projection along its output-feature dimension."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        requires_grad: bool,
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = _parameter(weight, requires_grad=requires_grad)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = _parameter(bias, requires_grad=requires_grad)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        context: TensorParallelContext,
        *,
        name: str,
    ) -> ColumnParallelLinear:
        start, end = context.shard_bounds(linear.out_features, name=f"{name}.out_features")
        bias = None if linear.bias is None else linear.bias[start:end]
        module = cls(
            linear.weight[start:end],
            bias,
            requires_grad=linear.weight.requires_grad,
        )
        return module.train(linear.training)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Shard a linear projection along input features and reduce its output."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        context: TensorParallelContext,
        requires_grad: bool,
    ) -> None:
        super().__init__()
        self.context = context
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = _parameter(weight, requires_grad=requires_grad)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = _parameter(bias, requires_grad=requires_grad)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        context: TensorParallelContext,
        *,
        name: str,
    ) -> RowParallelLinear:
        start, end = context.shard_bounds(linear.in_features, name=f"{name}.in_features")
        module = cls(
            linear.weight[:, start:end],
            linear.bias,
            context=context,
            requires_grad=linear.weight.requires_grad,
        )
        return module.train(linear.training)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.training:
            raise RuntimeError("RowParallelLinear currently supports inference only")
        output = F.linear(value, self.weight, None)
        if self.context.enabled:
            dist.all_reduce(output, group=self.context.process_group)
        if self.bias is not None:
            output = output + self.bias
        return output
