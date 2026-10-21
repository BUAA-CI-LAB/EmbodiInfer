"""Replay admission and the boundary between output parity and navigation accuracy."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

_compare = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/activevln-benchmark/compare.py")
)["compare"]


def report():
    """A synthetic complete-size report for testing the admission gate only."""
    rows = [
        {
            "sample_id": f"R2R/{i % 48 + 1}/{i}.jpg",
            "repeat": 0,
            "token_ids": [1, 2],
            "output_sha256": "same-action",
            "action_mask": [True, False, False],
            "parsed_valid": True,
            "stop_reason": "eos",
            "cache_length": i + 3,
            "generated_tokens": 2,
            "action_slots": 3,
            "latency_ms": 30.0,
        }
        for i in range(2997)
    ]
    return {
        "model": "activevln",
        "batch_size": 1,
        "dataset": {"name": "R2R"},
        "episodes": list(range(1, 49)),
        "selection_sha256": "selection",
        "tokenizer_sha256": {"tokenizer.json": "tokenizer"},
        "timing_boundary": "CPU RGB to CPU action chunk",
        "memory_protocol": "generated history; reset at episode boundary",
        "metrics": {"calls": len(rows)},
        "config": {
            "checkpoint_revision": "checkpoint",
            "dtype": "bfloat16",
            "seed": 42,
            "warmup_calls": 33,
            "repeats": 1,
            "max_new_tokens": 512,
            "max_context": 128000,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "max_steps_per_episode": None,
            "cpu_threads": 4,
        },
        "rows": rows,
    }


def test_admission_requires_behavior_parity_even_when_actions_match():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    assert _compare(baseline, candidate)["admitted"]
    candidate["rows"][5]["token_ids"] = [3, 2]
    result = _compare(baseline, candidate)
    assert not result["admitted"]
    assert result["matching_calls"] == 2996
    assert result["mismatches_by_field"]["token_ids"] == 1
    assert result["mismatches_by_field"]["output_sha256"] == 0


def test_admission_checks_raw_latency_and_strict_target():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    for row in candidate["rows"]:
        row["latency_ms"] = 40.0
    result = _compare(baseline, candidate)
    assert result["exact_behavior_parity"]
    assert not result["admitted"]


def test_task_success_contract_keeps_differences_diagnostic_and_accuracy_pending():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    candidate["rows"][5]["token_ids"] = [3, 2]
    candidate["rows"][5]["output_sha256"] = "different-action"
    result = _compare(baseline, candidate, accuracy_contract="task-success")
    assert result["e2e_mean_below_40ms"]
    assert result["mismatches_by_field"]["output_sha256"] == 1
    assert result["admitted"] is None
    assert result["admission_status"] == "pending_navigation_evaluation"
    assert result["navigation_success_evaluated"] is False
    candidate["rows"].pop()
    with pytest.raises(ValueError, match="every frame"):
        _compare(baseline, candidate, accuracy_contract="task-success")


def test_comparison_rejects_unknown_accuracy_contract():
    with pytest.raises(ValueError, match="accuracy_contract"):
        _compare(report(), report(), accuracy_contract="ignore-errors")


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_admission_rejects_invalid_latency(latency):
    baseline = report()
    candidate = copy.deepcopy(baseline)
    candidate["rows"][0]["latency_ms"] = latency
    with pytest.raises(ValueError, match="finite and positive"):
        _compare(baseline, candidate)


@pytest.mark.parametrize("failure", ["truncated", "missing", "duplicate", "conditions"])
def test_admission_rejects_incomparable_measurements(failure):
    baseline = report()
    candidate = copy.deepcopy(baseline)
    if failure == "truncated":
        candidate["config"]["max_steps_per_episode"] = 64
    elif failure == "missing":
        candidate["rows"].pop()
    elif failure == "duplicate":
        candidate["rows"][1] = candidate["rows"][0]
    else:
        candidate["config"]["max_new_tokens"] = 1
    with pytest.raises(ValueError):
        _compare(baseline, candidate)
