"""Workflow receipts block stale inputs, partial stages and failed numerical gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.kernel_tuning.artifacts import atomic_json
from scripts.kernel_tuning.contracts import ContractError, read_json
from scripts.kernel_tuning.workflow import KINDS, create, execute, fingerprint, inspect, validate


@pytest.fixture
def configured(tmp_path):
    source = tmp_path / "observations.pt"
    source.write_bytes(b"fixed evaluation inputs")
    config = {
        "schema": 1,
        "profile": "target_pi05",
        "inputs": {"observations": str(source)},
        "verification_input": "observations",
        "profile_artifact": "{run}/installed_profile",
        "stages": [],
    }
    for kind in KINDS[1:]:
        stage = {
            "name": kind,
            "kind": kind,
            "argv": ["trusted-program", kind, "{run}/" + kind + ".json"],
            "artifacts": [kind + ".json"],
        }
        if kind == "verify":
            stage["gate"] = "verify.json"
        if kind == "install":
            stage["artifacts"].append("installed_profile")
        config["stages"].append(stage)
    path = tmp_path / "config.json"
    atomic_json(path, config)
    root = tmp_path / "workflow"
    create(path, root)
    return config, path, root, source


def launcher(called, *, fail=None, bad_gate=False):
    def run(argv, log, timeout):
        kind = argv[1]
        called.append(kind)
        log.write_text("trusted fake stage", encoding="utf-8")
        if kind == fail:
            return 1
        report = {"stage": kind}
        root = Path(argv[2]).parent
        if kind == "install":
            atomic_json(root / "installed_profile/profile.json", {"op": "verified source hash"})
        if kind == "verify":
            inputs = read_json(root / "workflow.json")["inputs"]["observations"]
            report.update(
                passed=not bad_gate,
                inputs=3,
                profile="target_pi05",
                input_fingerprint=inputs["fingerprint"],
                profile_fingerprint=fingerprint(root / "installed_profile"),
            )
        atomic_json(Path(argv[2]), report)
        return 0

    return run


def test_successful_resume_does_not_repeat_stages(configured):
    _, _, root, _ = configured
    called = []
    result = execute(root, launcher=launcher(called))
    assert result["status"] == "completed"
    assert called == list(KINDS[1:])
    execute(root, launcher=launcher(called))
    assert called == list(KINDS[1:])
    assert result["performance_claim"] is False


def test_failed_gate_blocks_benchmark_and_requires_fresh_report_on_retry(configured):
    _, _, root, _ = configured
    called = []
    with pytest.raises(ContractError, match="verification failed"):
        execute(root, launcher=launcher(called, bad_gate=True))
    assert "benchmark" not in called
    with pytest.raises(ContractError, match="explicit --retry-stage"):
        execute(root, launcher=launcher(called))
    with pytest.raises(ContractError, match="fresh"):
        execute(root, retry_stage="verify", launcher=launcher(called))
    (root / "verify.json").rename(root / "failed-verify.json")
    execute(root, retry_stage="verify", launcher=launcher(called))
    assert called[-2:] == ["verify", "benchmark"]
    assert (root / "logs/verify/1/commands.json").exists()


def test_changed_input_and_changed_completed_evidence_are_refused(configured):
    _, _, root, source = configured
    called = []
    execute(root, launcher=launcher(called))
    source.write_bytes(b"different input")
    with pytest.raises(ContractError, match="input changed"):
        execute(root, launcher=launcher(called))
    source.write_bytes(b"fixed evaluation inputs")
    atomic_json(root / "profile.json", {"forged": True})
    with pytest.raises(ContractError, match="evidence changed"):
        execute(root, launcher=launcher(called))
    assert len(called) == len(KINDS) - 1


def test_nonzero_stage_never_advances(configured):
    _, _, root, _ = configured
    called = []
    with pytest.raises(ContractError, match="exited 1"):
        execute(root, launcher=launcher(called, fail="capture"))
    assert called == ["profile", "select", "capture"]
    execute(root, retry_stage="capture", launcher=launcher(called))
    assert called.count("profile") == 1
    assert called.count("capture") == 2


def test_existing_gate_cannot_be_reused(configured):
    _, _, root, _ = configured
    atomic_json(root / "verify.json", {"passed": True, "profile": "target_pi05", "inputs": 3})
    called = []
    with pytest.raises(ContractError, match="fresh"):
        execute(root, launcher=launcher(called))
    assert "verify" not in called and "benchmark" not in called


@pytest.mark.parametrize("field", ["input_fingerprint", "profile_fingerprint"])
def test_gate_must_match_actual_inputs_and_installed_sources(configured, field):
    _, _, root, _ = configured
    called = []
    base = launcher(called)

    def altered(argv, log, timeout):
        code = base(argv, log, timeout)
        if argv[1] == "verify":
            report = read_json(Path(argv[2]))
            report[field] = {"forged": True}
            atomic_json(Path(argv[2]), report)
        return code

    with pytest.raises(ContractError, match="evidence does not match"):
        execute(root, launcher=altered)
    assert "benchmark" not in called


def test_unknown_placeholders_are_rejected_during_planning(configured):
    config, _, _, _ = configured
    config["stages"][-1]["argv"].append("{misspelled_binding}")
    with pytest.raises(ContractError, match="placeholder"):
        validate(config)


def test_readonly_inspection_rejects_changed_inputs_and_invalid_ledger(configured):
    _, _, root, source = configured
    assert inspect(root)["status"] == "pending"
    source.write_bytes(b"changed")
    with pytest.raises(ContractError, match="input changed"):
        inspect(root)
    source.write_bytes(b"fixed evaluation inputs")
    manifest = read_json(root / "workflow.json")
    manifest["stages"]["verify"] = {"status": "passed", "attempts": 1}
    atomic_json(root / "workflow.json", manifest)
    with pytest.raises(ContractError, match="requires output receipts"):
        inspect(root)


def test_invalid_order_missing_gate_and_changed_plan_fail(configured):
    config, _, root, _ = configured
    config["stages"][-1], config["stages"][-2] = config["stages"][-2], config["stages"][-1]
    with pytest.raises(ContractError, match="order"):
        validate(config)
    manifest = read_json(root / "workflow.json")
    manifest["config"]["profile"] = "another_profile"
    atomic_json(root / "workflow.json", manifest)
    with pytest.raises(ContractError, match="configuration/tooling changed"):
        execute(root, launcher=launcher([]))


def test_single_match_bindings_and_foreach_exports(configured):
    config, path, root, _ = configured
    # Build another plan; completed stage receipts refer to files that later
    # stages only read, while the binding selects the unique batch directory.
    stage = next(stage for stage in config["stages"] if stage["kind"] == "tune")
    stage["bindings"] = {"batch": "{run}/batches/*"}
    stage["artifacts"] += ["{batch}/exports"]
    stage = next(stage for stage in config["stages"] if stage["kind"] == "install")
    stage["foreach"] = "{batch}/exports/*"
    stage["argv"].append("{item}")
    atomic_json(path, config)
    root = root.parent / "foreach-workflow"
    create(path, root)
    called = []
    base = launcher(called)
    installed = []

    def run(argv, log, timeout):
        if argv[1] == "tune":
            for name in ("a", "b"):
                atomic_json(root / "batches/one/exports" / name / "solution.json", {"name": name})
        if argv[1] == "install":
            installed.append(Path(argv[3]).name)
        return base(argv, log, timeout)

    execute(root, launcher=run)
    assert installed == ["a", "b"]
    execute(root, launcher=run)
    assert installed == ["a", "b"]


def test_retry_uses_frozen_resume_command_and_existing_batch(configured):
    config, path, root, _ = configured
    stage = next(stage for stage in config["stages"] if stage["kind"] == "tune")
    stage["bindings"] = {"batch": "{run}/batches/*"}
    stage["resume_bindings"] = {"batch": "{run}/batches/*"}
    stage["resume_argv"] = ["trusted-program", "tune", "{run}/tune.json", "--resume", "{batch}"]
    atomic_json(path, config)
    root = root.parent / "resume-workflow"
    create(path, root)
    called = []
    base = launcher(called)

    def interrupted(argv, log, timeout):
        if argv[1] == "tune":
            (root / "batches/existing").mkdir(parents=True)
            return 130
        return base(argv, log, timeout)

    with pytest.raises(ContractError, match="exited 130"):
        execute(root, launcher=interrupted)
    resumed = []

    def continuing(argv, log, timeout):
        if argv[1] == "tune":
            resumed.append(argv[3:])
        return base(argv, log, timeout)

    execute(root, retry_stage="tune", launcher=continuing)
    assert resumed == [["--resume", str(root / "batches/existing")]]
