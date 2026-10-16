"""CPU tests for the model-level numerical gate."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from scripts.kernel_tuning.contracts import ContractError
from scripts.kernel_tuning.verify import flatten_outputs, gate


@dataclass
class _Action:
    kind: str
    values: dict


@dataclass
class _Result:
    action_space: str
    actions: tuple
    timing: dict


def _out(values, token=3):
    return {".actions": ("continuous", list(values)), ".token": ("discrete", [token])}


def test_flatten_outputs_splits_continuous_and_discrete_leaves():
    result = _Result("joint", (_Action("chunk", {"q": [[0.5, 1.0], [2.0, 3.0]], "id": 7}),), {"ms": 1.0})
    leaves = flatten_outputs(result)
    assert leaves[".actions[0].values.q[0]"] == ("continuous", [0.5, 1.0])
    assert leaves[".actions[0].values.id"] == ("discrete", [7])
    assert leaves[".action_space"] == ("discrete", ["joint"])
    assert not any("timing" in name for name in leaves)


def test_gate_compares_mean_and_worst_errors_over_all_inputs():
    reference = [_out([1.0, 2.0]), _out([1.0, 2.0])]
    baseline = [_out([1.001, 2.0]), _out([1.003, 2.0])]  # mean max-abs 2e-3, worst 3e-3
    # The first input exceeds twice its own baseline error, but the means stay within 25%.
    report = gate(reference, baseline, [_out([1.0024, 2.0]), _out([1.0, 2.0])])
    assert report["passed"] and report["failed"] == []
    report = gate(reference, baseline, [_out([1.003, 2.0]), _out([1.003, 2.0])])  # mean 3e-3 > 2.5e-3
    assert not report["passed"] and report["failed"] == [".actions"]
    report = gate(reference, baseline, [_out([1.0, 2.0]), _out([1.0065, 2.0])])  # worst 6.5e-3 > 6e-3
    worst = report["outputs"][".actions"]["worst_max_abs"]
    assert not report["passed"] and worst["candidate"] > worst["limit"]
    exact = [_out([1.0, 2.0])]
    assert gate(exact, exact, [_out([1.00001, 2.0])])["passed"]  # within the 1e-4 / 1e-5 floors
    assert not gate(exact, exact, [_out([1.0002, 2.0])])["passed"]


def test_gate_requires_identical_discrete_outputs_and_comparable_reference():
    reference, baseline = [_out([1.0])], [_out([1.0])]
    assert not gate(reference, baseline, [_out([1.0], token=4)])["passed"]
    with pytest.raises(ContractError, match="did not share inputs/noise"):
        gate(reference, [_out([5.0])], [_out([5.0])])
    with pytest.raises(ContractError, match="one output per input"):
        gate(reference, baseline, [])
