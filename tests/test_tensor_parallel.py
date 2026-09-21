from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from embodiinfer.engine.parallel import (
    ColumnParallelLinear,
    FusedQKVColumnParallelLinear,
    RowParallelLinear,
    TensorParallelContext,
)
from embodiinfer.policies.navida.generation import NaViDAGenerationRuntime


def test_tensor_parallel_context_validates_size():
    with pytest.raises(ValueError, match="positive integer"):
        TensorParallelContext.from_distributed(0)


def test_tensor_parallel_context_size_one_is_disabled():
    context = TensorParallelContext.from_distributed(1)
    assert not context.enabled
    assert context.rank == 0
    assert context.world_size == 1


def test_tensor_parallel_shard_bounds():
    context = TensorParallelContext(world_size=2, rank=1)
    assert context.shard_bounds(16, name="width") == (8, 16)
    with pytest.raises(ValueError, match="divisible"):
        context.shard_bounds(15, name="width")


def test_tensor_parallel_context_requires_process_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized"):
        TensorParallelContext.from_distributed(2)


def test_navida_tp_sampling_uses_context_process_group(monkeypatch):
    runtime = object.__new__(NaViDAGenerationRuntime)
    runtime.tensor_parallel = TensorParallelContext(
        world_size=2,
        rank=1,
        process_group="tp-group",
    )
    runtime._process_navida_scores = lambda sequences, logits: logits
    broadcasts = []

    def get_global_rank(group, rank):
        assert group == "tp-group"
        assert rank == 0
        return 7

    def broadcast(token, src, group):
        broadcasts.append((src, group))
        token.fill_(3)

    monkeypatch.setattr(torch.distributed, "get_global_rank", get_global_rank)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    token = runtime._sample_navida_token(
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 8)),
        generator=None,
    )

    assert token.item() == 3
    assert broadcasts == [(7, "tp-group")]


def test_column_parallel_linear_matches_full_output_shard():
    torch.manual_seed(0)
    linear = nn.Linear(6, 8)
    value = torch.randn(3, 6)
    context = TensorParallelContext(world_size=2, rank=1)

    shard = ColumnParallelLinear.from_linear(linear, context, name="projection")

    torch.testing.assert_close(shard(value), linear(value)[:, 4:])


def test_fused_qkv_column_parallel_reconstructs_full_projection():
    torch.manual_seed(2)
    linear = nn.Linear(4, 12)
    value = torch.randn(2, 4)
    full = linear(value).reshape(2, 3, 4)
    shards = [
        FusedQKVColumnParallelLinear(
            linear,
            TensorParallelContext(world_size=2, rank=rank),
        )(value).reshape(2, 3, 2)
        for rank in range(2)
    ]

    torch.testing.assert_close(torch.cat(shards, dim=2), full)


def test_row_parallel_linear_reduces_partial_outputs(monkeypatch):
    torch.manual_seed(1)
    linear = nn.Linear(8, 5)
    value = torch.randn(3, 8)
    context = TensorParallelContext(world_size=2, rank=1)
    shard = RowParallelLinear.from_linear(linear, context, name="projection").eval()
    other_rank_partial = F.linear(value[:, :4], linear.weight[:, :4], None)

    def fake_all_reduce(output, group=None):
        assert group is None
        output.add_(other_rank_partial)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    torch.testing.assert_close(shard(value[:, 4:]), linear(value))


def test_row_parallel_linear_size_one_skips_collective(monkeypatch):
    linear = nn.Linear(4, 3).eval()
    value = torch.randn(2, 4)
    shard = RowParallelLinear.from_linear(
        linear,
        TensorParallelContext(),
        name="projection",
    ).eval()

    def unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("size-one tensor parallelism must not use a collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_all_reduce)

    torch.testing.assert_close(shard(value), linear(value))
