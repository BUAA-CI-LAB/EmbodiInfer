"""Task validation and promotion rules without agents, CUDA, or source execution."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.kernel_tuning.contracts import (
    Acceptance,
    ContractError,
    Precision,
    Settings,
    TaskPackage,
    Timing,
    parse_json,
    promotion,
    read_json,
    relative_path,
    validate_measurement,
    validate_solution,
)


def measurement(task: TaskPackage, latency: float = 8) -> dict[str, Any]:
    """Generate complete synthetic evidence across the task's declared cases."""
    roles = ("baseline", "incumbent", "candidate")
    return {
        "checks": [
            {
                "workload": w,
                "role": role,
                "passed": True,
                "seeds": list(task.settings.seeds),
                "precision": asdict(task.settings.precision),
                "max_abs_error": 0,
                "max_rel_error": 0,
                "graph_replay": True,
            }
            for w in task.workload_ids
            for role in roles
        ],
        "rounds": [
            {
                w: {r: [latency if r == "candidate" else 10] * task.settings.timing.trials for r in roles}
                for w in task.workload_ids
            }
            for _ in range(task.settings.timing.paired_rounds)
        ],
    }


def test_task_validation_does_not_execute_reference(task_dir: Path) -> None:
    definition = read_json(task_dir / "definition.json")
    definition["reference"] = "raise RuntimeError('not imported')\n" + definition["reference"]
    from scripts.kernel_tuning.artifacts import atomic_json

    atomic_json(task_dir / "definition.json", definition)
    task = TaskPackage.load(task_dir)
    assert len(task.workload_ids) == 3


@pytest.mark.parametrize(
    "path", ["../a", "/tmp/x", "C:/x", "a\\b", "a/../b", "a//b", "NUL", "a/CON.txt", "a."]
)
def test_unsafe_or_nonportable_paths_fail(path: str) -> None:
    with pytest.raises(ContractError):
        relative_path(path)


@pytest.mark.parametrize("source", ['{"axes":{},"axes":{}}', '{"value":NaN}', '{"value":Infinity}'])
def test_all_json_records_reject_duplicate_keys_and_nonfinite_values(source: str) -> None:
    with pytest.raises(ContractError):
        parse_json(source)


def test_task_changes_are_detected(store) -> None:
    with (store.task.root / "baseline.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    with pytest.raises(ContractError, match="changed"):
        store.verify()


def test_duplicate_workloads_and_unknown_settings_fail(task_dir: Path) -> None:
    workloads = task_dir / "workloads.jsonl"
    text = workloads.read_text(encoding="utf-8")
    workloads.write_text(text + text.splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="distinct"):
        TaskPackage.load(task_dir)
    workloads.write_text(text, encoding="utf-8")
    with (task_dir / "tuning.yaml").open("a", encoding="utf-8") as handle:
        handle.write("typo: true\n")
    with pytest.raises(ContractError, match="Unknown"):
        TaskPackage.load(task_dir)


@pytest.mark.parametrize(
    "config",
    [
        {"language": "python"},
        {"language": "triton", "seeds": ()},
        {"language": "cuda", "weights": {"one": 0}},
    ],
)
def test_invalid_settings_fail(config: dict) -> None:
    with pytest.raises(ContractError):
        Settings(**config)


def test_precision_and_remeasurement_cannot_be_silently_relaxed() -> None:
    with pytest.raises(ContractError):
        Precision(atol=1e-2)
    with pytest.raises(ContractError):
        Precision(mode="tolerance", atol=1e-2)
    assert Precision(mode="tolerance", atol=1e-4, reason="fixed reduction ordering allowance")
    with pytest.raises(ContractError):
        Timing(paired_rounds=1)
    with pytest.raises(ContractError):
        Acceptance(min_improvement=0)


def test_solution_multi_file_and_frozen_language(store, candidate: dict) -> None:
    candidate["sources"].append({"path": "include/helpers.h", "content": "// helper"})
    assert validate_solution(candidate, store.task) == candidate
    candidate["spec"]["language"] = "cuda"
    with pytest.raises(ContractError, match="language"):
        validate_solution(candidate, store.task)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["checks"].pop(),
        lambda r: r["checks"].append(r["checks"][0]),
        lambda r: r["checks"][0].update(passed="true"),
        lambda r: r["checks"][0].update(max_abs_error=0.001),
        lambda r: r["checks"][0].update(seeds=[99]),
        lambda r: r["rounds"].pop(),
        lambda r: r["rounds"][0].pop("broadcast-small"),
        lambda r: r["rounds"][0]["broadcast-small"]["candidate"].__setitem__(0, float("nan")),
        lambda r: r["rounds"][0]["broadcast-small"]["candidate"].__setitem__(0, 0),
    ],
)
def test_incomplete_or_malformed_evidence_never_promotes(store, mutation) -> None:
    report = measurement(store.task)
    mutation(report)
    with pytest.raises(ContractError):
        promotion(store.task, report)


def test_gain_must_repeat_and_each_workload_must_pass(store) -> None:
    report = measurement(store.task)
    assert promotion(store.task, report)["promoted"]
    report["rounds"][1]["broadcast-small"]["candidate"] = [10.6] * 3
    decision = promotion(store.task, report)
    assert not decision["promoted"]
    assert any("regresses" in r for r in decision["reasons"])
    report = measurement(store.task, 9.8)
    assert not promotion(store.task, report)["promoted"]


def test_incumbent_and_initial_baseline_both_constrain_promotion(store) -> None:
    report = measurement(store.task, 8)
    for row in report["rounds"][1].values():
        row["incumbent"] = [8.1] * 3
    assert not promotion(store.task, report)["promoted"]


def test_explicit_weights_change_the_mean_not_the_regression_guard(task_dir: Path) -> None:
    path = task_dir / "tuning.yaml"
    config = yaml.safe_load(path.read_text())
    config["weights"] = {"broadcast-small": 100, "broadcast-wide": 1, "token-gate-tail": 1}
    path.write_text(yaml.safe_dump(config))
    task = TaskPackage.load(task_dir)
    report = measurement(task)
    for round_ in report["rounds"]:
        round_["token-gate-tail"]["candidate"] = [10.4] * 3
    assert promotion(task, report)["promoted"]
    bad = copy.deepcopy(report)
    for round_ in bad["rounds"]:
        round_["token-gate-tail"]["candidate"] = [10.6] * 3
    assert not promotion(task, bad)["promoted"]


def test_graph_evidence_must_include_changed_inputs(task_dir: Path) -> None:
    path = task_dir / "tuning.yaml"
    config = yaml.safe_load(path.read_text())
    config["timing"]["mode"] = "cuda_graph"
    path.write_text(yaml.safe_dump(config))
    task = TaskPackage.load(task_dir)
    report = measurement(task)
    report["checks"][0]["graph_replay"] = False
    with pytest.raises(ContractError, match="replay"):
        validate_measurement(task, report)
