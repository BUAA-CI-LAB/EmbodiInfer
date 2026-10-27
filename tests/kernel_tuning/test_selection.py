"""Selection accounts for all GPU regions without claiming model speedups."""

from __future__ import annotations

import copy

import pytest
from scripts.kernel_tuning.contracts import ContractError
from scripts.kernel_tuning.selection import plan, selected_ops


def report():
    return {
        "kernel_us_per_step": 100,
        "regions": [
            {"region": "stage:prefix/op:linear", "us_per_step": 40},
            {"region": "stage:denoise/op:linear", "us_per_step": 30},
            {"region": "stage:denoise/op:rotate_qk", "us_per_step": 10},
            {"region": "stage:prefix/module:vision", "us_per_step": 20},
        ],
    }


def test_plan_aggregates_stages_and_preserves_unregistered_work():
    value = plan(report(), limit=1)
    assert selected_ops(value) == {"linear"}
    assert value["selected"][0]["us_per_step"] == 70
    assert value["deferred"][0]["operator"] == "rotate_qk"
    assert value["investigations"][0]["share"] == 0.2
    assert "not end-to-end" in value["caveats"][0]


def test_measured_floor_changes_priority_and_requires_source():
    value = plan(
        report(),
        limit=1,
        floors={"linear": {"lower_bound_us_per_step": 69, "source": "target measured bound.json"}},
    )
    assert selected_ops(value) == {"rotate_qk"}
    for floors in (
        {"linear": {"lower_bound_us_per_step": 69}},
        {"missing_op": {"lower_bound_us_per_step": 1, "source": "x"}},
        {"linear": {"lower_bound_us_per_step": 71, "source": "x"}},
    ):
        with pytest.raises(ContractError):
            plan(report(), floors=floors)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_timings_are_rejected(value):
    raw = copy.deepcopy(report())
    raw["regions"][0]["us_per_step"] = value
    with pytest.raises(ContractError):
        plan(raw)


def test_incomplete_report_and_empty_capture_selection_fail():
    raw = report()
    raw["regions"].pop()
    with pytest.raises(ContractError, match="account"):
        plan(raw)
    value = plan(report(), min_share=0.9)
    with pytest.raises(ContractError, match="distinct"):
        selected_ops(value)
