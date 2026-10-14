"""Split-count tuning keeps the legacy default and never selects a winner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from embodiinfer.backend.split_kv_plan import SUPPORTED_SPLIT_COUNTS, plan_split_kv
from embodiinfer.backend.triton import split_kv_attention as split_module


@pytest.mark.parametrize("queries", [1, 7, 50, 64, 65, 128, 777])
@pytest.mark.parametrize("cached", [0, 1, 63, 256, 1023, 10000])
def test_default_plan_matches_original_geometry(queries: int, cached: int) -> None:
    splits, block_m, block_n = (4, 16, 64) if queries <= 64 else (1, 32, 64)
    old_partition_length = ((cached + queries + splits * block_n - 1) // (splits * block_n)) * block_n
    plan = plan_split_kv(queries, cached)
    assert (plan.num_splits, plan.block_m, plan.block_n, plan.partition_length) == (
        splits,
        block_m,
        block_n,
        old_partition_length,
    )
    assert plan == plan_split_kv(queries, cached, num_splits=splits)


@pytest.mark.parametrize("splits", SUPPORTED_SPLIT_COUNTS)
@pytest.mark.parametrize(("queries", "cached"), [(1, 0), (64, 129), (65, 1000)])
def test_explicit_plan_changes_only_partition_geometry(queries: int, cached: int, splits: int) -> None:
    default = plan_split_kv(queries, cached)
    actual = plan_split_kv(queries, cached, num_splits=splits)
    assert actual.num_splits == splits
    assert (actual.block_m, actual.block_n) == (default.block_m, default.block_n)
    assert actual.partition_length % actual.block_n == 0
    assert actual.partition_length * splits >= queries + cached
    assert (actual.partition_length - actual.block_n) * splits < queries + cached
    with pytest.raises(FrozenInstanceError):
        actual.num_splits = 8


@pytest.mark.parametrize("splits", [0, -1, 3, 5, 32, 4.0, "4", True, False])
def test_unsupported_splits_fail_explicitly(splits: object) -> None:
    with pytest.raises(ValueError, match="num_splits"):
        plan_split_kv(50, 256, num_splits=splits)


@pytest.mark.parametrize(
    ("queries", "cached"), [(0, 0), (-1, 5), (1, -1), (1.0, 1), (1, 0.0), (True, 0), (1, False)]
)
def test_invalid_lengths_fail_explicitly(queries: object, cached: object) -> None:
    with pytest.raises(ValueError, match="length"):
        plan_split_kv(queries, cached)


@pytest.mark.parametrize("splits", [None, *SUPPORTED_SPLIT_COUNTS])
def test_backend_forwards_split_choice_without_changing_input(monkeypatch, splits: int | None) -> None:
    q = torch.zeros(1, 4, 7, 64)
    k = torch.zeros(1, 1, 7, 64)
    cached = k[:, :, :0]
    mask = torch.zeros(1, 1, 1, 7)
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return q

    monkeypatch.setattr(split_module, "split_kv_attention", record)
    backend = split_module.TritonSplitKVAttention()
    assert backend.attend_split(q, cached, cached, k, k, mask, num_splits=splits) is q
    args, kwargs = calls.pop()
    assert all(actual is expected for actual, expected in zip(args[:6], (q, cached, cached, k, k, mask)))
    assert args[6] == 64**-0.5
    assert kwargs == {"num_splits": splits}
    assert backend.attend(q, k, k) is q
    assert calls[-1][1] == {"num_splits": None}


def test_backend_still_rejects_dropout() -> None:
    q = torch.zeros(1, 1, 7, 64)
    with pytest.raises(ValueError, match="dropout"):
        split_module.TritonSplitKVAttention().attend_split(q, q, q, q, q, dropout_p=0.1)


@pytest.mark.parametrize("queries", [7, 64, 65])
@pytest.mark.parametrize("splits", [None, *SUPPORTED_SPLIT_COUNTS])
def test_operator_allocations_and_launches_match_plan(monkeypatch, queries: int, splits: int | None) -> None:
    """Exercise the real Python launch path with metadata-only CUDA tensors."""
    allocations, launches = [], []
    cuda_device = torch.device("cuda", 0)

    class Tensor:
        def __init__(self, shape, dtype=torch.bfloat16, device=cuda_device):
            self.shape = tuple(shape)
            self.ndim = len(shape)
            self.dtype = dtype
            self.device = device

        def stride(self, index=None):
            product, strides = 1, []
            for length in reversed(self.shape):
                strides.insert(0, product)
                product *= length
            return tuple(strides) if index is None else strides[index]

        def view(self, shape):
            return Tensor(shape, self.dtype, self.device)

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, grid, args, kwargs))

            return launch

    def allocate(shape, *, device, dtype):
        allocations.append((tuple(shape), dtype, device))
        return Tensor(shape, dtype, device)

    monkeypatch.setattr(split_module, "triton", SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y))
    monkeypatch.setattr(split_module, "_partial_attention", Kernel("partial"), raising=False)
    monkeypatch.setattr(split_module, "_merge_attention", Kernel("merge"), raising=False)
    monkeypatch.setattr(split_module.torch, "empty", allocate)
    q, prefix, suffix = Tensor((2, 4, queries, 64)), Tensor((2, 1, 257, 64)), Tensor((2, 1, queries, 64))
    output = split_module.split_kv_attention(
        q, prefix, prefix, suffix, suffix, None, 0.125, num_splits=splits
    )
    plan = plan_split_kv(queries, 257, num_splits=splits)
    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert allocations == [
        ((2, 4, plan.num_splits, queries, 64), q.dtype if plan.num_splits == 1 else torch.float32, q.device),
        ((2, 4, plan.num_splits, queries), torch.float32, q.device),
        (q.shape, q.dtype, q.device),
    ]
    assert launches[0][0:2] == ("partial", (8, (queries + plan.block_m - 1) // plan.block_m, plan.num_splits))
    assert launches[0][2][-4:] == (plan.num_splits, plan.partition_length, plan.block_m, plan.block_n)
    assert launches[0][3] == {"num_warps": 4, "num_stages": 1}
    if plan.num_splits == 1:
        assert len(launches) == 1
    else:
        assert len(launches) == 2
        assert launches[1][0:2] == ("merge", (queries, 8))
        assert launches[1][2][-3:] == (queries, 64, plan.num_splits)
        assert launches[1][3] == {"num_warps": 4}
