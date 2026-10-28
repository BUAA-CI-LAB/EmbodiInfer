"""Independent grammar helpers and device-upload layouts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodiinfer.policies.activevln import grammar_tree_activevln as grammar
from embodiinfer.policies.activevln.batching_activevln import ActiveVLNBatchedRuntime


def test_stop_reasons_are_memoized_within_a_bounded_table(monkeypatch):
    calls = []

    def reference(policy, ids):
        calls.append(tuple(ids))
        return "stop" if ids[-1] == 5 else None

    monkeypatch.setattr(grammar, "_stop_reason", reference)
    stops = grammar.StopReasons(object(), capacity=2)
    assert [stops([1, 5]), stops([1, 2]), stops([1, 5])] == ["stop", None, "stop"]
    assert calls == [(1, 5), (1, 2)]
    stops([3])
    assert stops([1, 5]) == "stop"
    assert calls == [(1, 5), (1, 2), (3,), (1, 5)]


def test_history_sets_match_unique_tokens():
    generator = torch.Generator().manual_seed(3)
    histories = [torch.randint(0, 50, (size,), generator=generator) for size in (1, 7, 40)]
    assert grammar._history_sets(histories, 64) == [set(torch.unique(h).tolist()) for h in histories]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_upload_keeps_shapes_dtypes_and_values(device):
    owner = SimpleNamespace(device=torch.device(device))
    ints = np.arange(6, dtype=np.int64).reshape(2, 3)
    broadcast = np.broadcast_to(np.arange(3, dtype=np.int64), (2, 2, 3))
    first, second = ActiveVLNBatchedRuntime._upload(owner, ints, broadcast)
    ints[:] = -1
    (mask,) = ActiveVLNBatchedRuntime._upload(owner, np.eye(3, dtype=bool))
    assert first.device.type == device and first.dtype == torch.int64 and mask.dtype == torch.bool
    assert first.cpu().tolist() == [[0, 1, 2], [3, 4, 5]]
    assert torch.equal(second.cpu(), torch.arange(3).expand(2, 2, 3))
    assert torch.equal(mask.cpu(), torch.eye(3, dtype=torch.bool))
