"""Numerical policy, independent calibration, and frozen all-element replay gates."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from benchmarks.kernel_tuning.calibration import (  # noqa: E402
    calibrate,
    captured_settings,
    compare_calibrated,
    derive_bound,
    environment,
    high_precision,
    production,
    validation_inputs,
)
from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter  # noqa: E402
from scripts.kernel_tuning.artifacts import atomic_json  # noqa: E402
from scripts.kernel_tuning.contracts import ContractError, TaskPackage, read_json  # noqa: E402
from scripts.kernel_tuning.discovery.contracts import ModelCapture  # noqa: E402
from scripts.kernel_tuning.discovery.numerics import PROFILES, policy  # noqa: E402
from scripts.kernel_tuning.discovery.tasks import render_model  # noqa: E402


@pytest.fixture
def calibrated(tmp_path: Path, registry_capture) -> ModelCapture:
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(3, 16, generator=generator)
    b = torch.randn(16, 5, generator=generator)
    c = torch.randn(4, 16, generator=generator) * 64
    raw = registry_capture(tmp_path, [("linear", (a, b.T), {}), ("linear", (c, b.T), {})])
    return calibrate(raw, tmp_path / "calibrated")


def test_registry_semantics_keep_mutation_integer_and_fp64_exact() -> None:
    spec = {
        "kind": "registry",
        "name": "registry.linear",
        "input_dtypes": ["float32"],
        "output_dtypes": ["float32"],
        "mutates": [],
    }
    assert policy(spec) == "registry"
    for extra in (
        {"kind": "aten", "name": "aten.mm.default"},
        {"kind": "backend"},
        {"mutates": [0]},
        {"rng": {"kind": "torch_default"}},
        {"input_dtypes": ["float64"]},
        {"output_dtypes": ["int64"]},
    ):
        assert policy({**spec, **extra}) is None


def test_generation_requires_calibration_and_capture_keeps_only_execution_switches(
    tmp_path: Path, registry_capture
) -> None:
    x = torch.ones(3, 3)
    capture = registry_capture(tmp_path, [("linear", (x, x), {})])
    assert "mode" not in capture.manifest["precision"]
    with pytest.raises(ContractError, match="Missing numerical calibration"):
        render_model(capture, tmp_path / "tasks")
    assert not (tmp_path / "tasks").exists()


def test_calibration_is_per_workload_and_covers_all_profiles(
    calibrated: ModelCapture, tmp_path: Path
) -> None:
    contracts = read_json(calibrated.root / "calibration.json")
    contract = next(iter(contracts.values()))
    assert len(contract["cases"]) == 2
    recorded_bounds = [case["recorded:0"][0]["atol"] for case in contract["cases"].values()]
    assert max(recorded_bounds) > 32 * min(recorded_bounds)
    for case in contract["cases"].values():
        assert set(case) == {f"{name}:{seed}" for name, seed in PROFILES}
    tasks = render_model(calibrated, tmp_path / "tasks")
    task = next(task for task in tasks if "numerics.json" in task.hashes)
    assert task.settings.precision.mode == "tolerance"
    assert read_json(task.root / "numerics.json") == contract
    with pytest.raises(ContractError, match="new destination"):
        calibrate(calibrated, calibrated.root)


@pytest.mark.parametrize("override", [{"precision.atol": 1.0}, {"seeds": [0, 9]}])
def test_generation_rejects_calibration_overrides(
    calibrated: ModelCapture, tmp_path: Path, override: dict
) -> None:
    with pytest.raises(ContractError, match="cannot.*override|cannot be overridden"):
        render_model(calibrated, tmp_path / "tasks", overrides=override)


@pytest.mark.parametrize("edit", ["threshold", "profile", "dtype", "flags"])
def test_task_loader_rejects_calibration_edits(calibrated: ModelCapture, tmp_path: Path, edit: str) -> None:
    task = next(t for t in render_model(calibrated, tmp_path / "tasks") if "numerics.json" in t.hashes)
    contract = read_json(task.root / "numerics.json")
    samples = next(iter(contract["cases"].values()))
    if edit == "threshold":
        samples["recorded:0"][0]["atol"] *= 100
    elif edit == "profile":
        del samples["zeros:0"]
    elif edit == "dtype":
        samples["recorded:0"][0]["dtype"] = "bfloat16"
    else:
        contract["flags"]["fp16_reduced_reduction"] = not contract["flags"]["fp16_reduced_reduction"]
    atomic_json(task.root / "numerics.json", contract)
    with pytest.raises(ContractError, match="Calibration|calibration"):
        TaskPackage.load(task.root)


def test_real_inputs_cannot_silently_fall_back_to_random_when_budget_exhausted(
    tmp_path: Path, registry_capture
) -> None:
    x = torch.ones(8, 8)
    capture = registry_capture(
        tmp_path,
        [("gelu_tanh", (torch.ones(2, dtype=torch.float64),), {}), ("linear", (x, x), {})],
        budget=32,
    )
    assert any("fixture" in error for error in capture.manifest["errors"])
    with pytest.raises(ContractError, match="incomplete"):
        calibrate(capture, tmp_path / "calibrated")


def test_every_element_must_pass_and_nan_or_wrong_dtype_fail() -> None:
    ref = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    baseline = ref.float()
    bound = derive_bound(baseline, ref)
    allowed = baseline.clone()
    allowed[0] += torch.finfo(torch.float32).eps
    assert not torch.equal(allowed, baseline)
    compare_calibrated([allowed], [ref], [bound])
    for value in (torch.tensor([1.0, 1e-3, -1.0]), baseline + float("nan"), baseline.double()):
        with pytest.raises(ContractError):
            compare_calibrated([value], [ref], [bound])
    with pytest.raises(ContractError, match="Nonfinite"):
        derive_bound(baseline + float("inf"), ref)


def test_calibration_restores_global_precision_on_failure() -> None:
    before = environment("cpu")
    flags = {
        name: before[name]
        for name in (
            "cudnn_allow_tf32",
            "fp16_reduced_reduction",
            "bf16_reduced_reduction",
            "deterministic_algorithms",
        )
    }
    with (
        pytest.raises(RuntimeError),
        captured_settings({"matmul_precision": "high", "allow_tf32": True}, flags),
    ):
        raise RuntimeError("broken baseline")
    assert environment("cpu") == before


def test_replay_uses_calibrated_bounds_and_all_profiles(
    calibrated: ModelCapture, tmp_path: Path, monkeypatch
) -> None:
    task = next(t for t in render_model(calibrated, tmp_path / "tasks") if "numerics.json" in t.hashes)
    adapter = object.__new__(ReplayAdapter)
    adapter.task, adapter.cfg, adapter.device = task, task.settings, "cpu"
    adapter.definition = SimpleNamespace(
        outputs=task.definition["outputs"], torch_output_dtypes=[torch.float32]
    )
    adapter.replay = read_json(task.root / "replay.json")
    adapter.numerical = read_json(task.root / "numerics.json")
    adapter._check_flags = lambda: None
    adapter._seed = lambda seed: None
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    operator = adapter.replay["numerical_identity"]

    class Runnable:
        metadata = SimpleNamespace(destination_passing_style=False)

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(list(args))[0]

    reference = Runnable(lambda args: production(operator, args))
    more_accurate = Runnable(lambda args: [high_precision(operator, args)[0].float()])
    workload = SimpleNamespace(uuid=task.workload_ids[0])
    for seed in task.settings.seeds:
        inputs = adapter._inputs(workload, seed)
        adapter._verify_call(more_accurate, reference, inputs, seed)

    def capture_graph(runnable, inputs):
        output = runnable(*inputs).clone()

        def replay():
            output.copy_(runnable(*inputs))

        return SimpleNamespace(replay=replay), inputs, [output]

    adapter._capture = capture_graph
    for seed in task.settings.seeds:
        adapter._verify_graph(more_accurate, reference, workload, seed)
    saved = production(operator, adapter._inputs(workload, 0))
    # The failure names where and by how much, since it becomes the next agent's feedback.
    with pytest.raises(
        ContractError,
        match=rf"frozen calibrated tolerance: workload {workload.uuid}, input profile \w+:\d; "
        r"output 0 has \d+/\d+ elements over the bound, max \|error\| .*worst error/bound",
    ):
        adapter._verify_call(Runnable(lambda args: saved), reference, adapter._inputs(workload, 0), 0)
    with pytest.raises(ContractError, match="frozen calibrated"):
        adapter._verify_graph(Runnable(lambda args: saved), reference, workload, 0)
    assert adapter._check_metadata()["numerical_contract"] == task.hashes["numerics.json"]
    assert adapter._check_metadata()["validation_profiles"] == [list(pair) for pair in PROFILES]
    assert asdict(task.settings.precision)["mode"] == "tolerance"


def test_random_profiles_preserve_masks_and_change_real_values(tmp_path: Path, registry_capture) -> None:
    q = torch.ones(1, 1, 3, 4)
    prefix = torch.ones(1, 1, 2, 4)
    suffix = torch.ones(1, 1, 1, 4)
    mask = torch.tensor([[[[0.0, -1.0, 0.0]]]])
    capture = registry_capture(
        tmp_path, [("split_kv_attention", (q, prefix, prefix, suffix, suffix, mask, 0.5), {})]
    )
    (op,) = capture.operators
    case = op["workloads"][0]
    real = validation_inputs(op["identity"], case, capture.root, "recorded", 0)
    random = validation_inputs(op["identity"], case, capture.root, "random", 1)
    assert not torch.equal(real[0], random[0])
    assert torch.equal(real[-1], random[-1])
    expected = production(op["identity"], [value.double() for value in real])
    torch.testing.assert_close(high_precision(op["identity"], real)[0], expected[0], atol=0, rtol=0)


def test_calibrate_cli_writes_a_new_capture_before_generation(
    tmp_path: Path, capsys, registry_capture
) -> None:
    from scripts.kernel_tuning.__main__ import main

    value = torch.ones(2, 3)
    weight = torch.ones(3, 4)
    raw = registry_capture(tmp_path, [("linear", (value, weight.T), {})])
    destination = tmp_path / "cli-calibrated"
    assert main(["calibrate", "--captured", str(raw.root), "--output", str(destination)]) == 0
    assert "identity" in capsys.readouterr().out
    (task,) = render_model(ModelCapture.load(destination), tmp_path / "tasks")
    assert task.settings.precision.mode == "tolerance"


def test_promotion_requires_all_profile_and_contract_evidence(
    calibrated: ModelCapture, tmp_path: Path
) -> None:
    from scripts.kernel_tuning.contracts import validate_measurement
    from test_batch import measurement

    task = next(t for t in render_model(calibrated, tmp_path / "tasks") if "numerics.json" in t.hashes)
    report = measurement(task, 8.0)
    with pytest.raises(ContractError, match="calibration/profile"):
        validate_measurement(task, report)
    for check in report["checks"]:
        check.update(
            numerical_contract=task.hashes["numerics.json"],
            validation_profiles=[list(pair) for pair in PROFILES],
        )
    validate_measurement(task, report)
    report["checks"][0]["validation_profiles"].pop()
    with pytest.raises(ContractError, match="calibration/profile"):
        validate_measurement(task, report)


def test_changed_oracle_source_invalidates_calibration(
    calibrated: ModelCapture, tmp_path: Path, monkeypatch
) -> None:
    from scripts.kernel_tuning.discovery import numerics

    task = next(t for t in render_model(calibrated, tmp_path / "tasks") if "numerics.json" in t.hashes)
    monkeypatch.setattr(numerics, "implementation_digest", lambda: "changed")
    with pytest.raises(ContractError, match="recalibrate"):
        TaskPackage.load(task.root)
