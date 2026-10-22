"""Registry capture artifacts, replay contracts and the all-task preflight barrier."""

from __future__ import annotations

import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter  # noqa: E402
from scripts.kernel_tuning import batch  # noqa: E402
from scripts.kernel_tuning.__main__ import main  # noqa: E402
from scripts.kernel_tuning.artifacts import atomic_json  # noqa: E402
from scripts.kernel_tuning.contracts import (  # noqa: E402
    ContractError,
    TaskPackage,
    digest,
    read_json,
    tree_hashes,
)
from scripts.kernel_tuning.discovery.contracts import ModelCapture  # noqa: E402
from scripts.kernel_tuning.discovery.tasks import render_model  # noqa: E402
from scripts.kernel_tuning.discovery.tensors import clone_inputs, materialize, storage_key  # noqa: E402


def _capture(root, registry_capture):
    value = torch.ones(4, 4, dtype=torch.float64)
    return registry_capture(root, [("gelu_tanh", (value,), {}), ("linear", (value, value), {})])


def test_capture_retains_every_layout_count_and_input_alias(tmp_path, registry_capture):
    value = torch.arange(16, dtype=torch.float64).reshape(4, 4).T
    other = torch.ones(2, 4, dtype=torch.float64)
    weight = value.clone()
    capture = registry_capture(
        tmp_path, [("linear", (value, weight), {})] * 3 + [("linear", (other, weight), {})]
    )
    (op,) = capture.operators
    assert sorted(case["count"] for case in op["workloads"]) == [1, 3]
    (task,) = render_model(capture, tmp_path / "tasks")
    replay = read_json(task.root / "replay.json")
    case = next(case for case in replay["cases"].values() if case["count"] == 3)
    inputs = materialize(case["inputs"], task.root, 0)
    assert len(inputs) == 2
    assert torch.equal(inputs[0], value) and inputs[0].stride() == value.stride()
    views = [value, value[:, :2]]
    cloned = clone_inputs(views)
    assert storage_key(cloned[0]) == storage_key(cloned[1])
    assert storage_key(cloned[0]) != storage_key(value)
    assert cloned[1].stride() == views[1].stride()
    assert replay["kind"] == "registry"
    assert task.settings.timing.mode == "cuda_graph"


def test_capture_model_and_fixture_integrity_remain_required(tmp_path, registry_capture):
    capture = _capture(tmp_path, registry_capture)
    with pytest.raises(ContractError, match="differs from requested model"):
        ModelCapture.load(capture.root, "pi05")
    fixture = next((capture.root / "fixtures").iterdir())
    fixture.write_bytes(fixture.read_bytes() + b"changed")
    with pytest.raises(ContractError, match="content changed"):
        render_model(capture, tmp_path / "tasks")
    assert not (tmp_path / "tasks").exists()


@pytest.mark.parametrize("legacy", ["manifest", "aten", "backend"])
def test_removed_discovery_artifacts_fail_with_migration_guidance(tmp_path, registry_capture, legacy):
    capture = _capture(tmp_path, registry_capture)
    manifest = read_json(capture.root / "manifest.json")
    if legacy == "manifest":
        manifest.pop("capture")
    else:
        operators = read_json(capture.root / "operators.json")
        operators[0]["identity"]["kind"] = legacy
        operators[0]["id"] = digest(operators[0]["identity"])[:20]
        atomic_json(capture.root / "operators.json", operators)
        manifest["files"] = {k: v for k, v in tree_hashes(capture.root).items() if k != "manifest.json"}
    atomic_json(capture.root / "manifest.json", manifest)
    with pytest.raises(ContractError, match="recapture with capture-ops"):
        ModelCapture.load(capture.root, require_ready=False)


def test_all_baselines_must_pass_before_any_model_task_runs(tmp_path, registry_capture, monkeypatch):
    from test_batch import measurement

    capture = _capture(tmp_path, registry_capture)
    root = batch.create_model(
        capture, tmp_path / "batches", agent="fake/model", overrides={}, hardware_notes=None
    )
    commands = []

    def launcher(*args):
        commands.append(args)

    with pytest.raises(ContractError, match="every task"):
        batch.execute(root, launcher=launcher)

    async def evaluator(store, output, **kwargs):
        if read_json(store.task.root / "replay.json")["operator"] == "registry.linear":
            raise ContractError("missing evaluator support")
        return {
            "measurement": measurement(store.task, 10.0),
            "environment": {"identity": {"hardware": "fake"}},
        }

    monkeypatch.setattr(batch, "evaluate", evaluator)
    results = batch.preflight(root)
    assert any(error is None for error in results.values()) and any(results.values())
    with pytest.raises(ContractError, match="every task"):
        batch.execute(root, launcher=launcher)
    assert commands == []
    assert not read_json(root / "batch.json")["preflight_complete"]

    async def success(store, output, **kwargs):
        return {
            "measurement": measurement(store.task, 10.0),
            "environment": {"identity": {"hardware": "fake"}},
        }

    monkeypatch.setattr(batch, "evaluate", success)
    assert all(error is None for error in batch.preflight(root).values())
    record = read_json(root / "batch.json")
    assert record["preflight_complete"]
    for item in record["items"].values():
        assert batch.verify_model_task(TaskPackage.load(root / item["task"])) == root
    task_root = root / next(iter(record["items"].values()))["task"]
    (task_root / "README.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ContractError, match="changed after generation"):
        batch.execute(root, launcher=launcher)
    assert commands == []


def test_registry_cli_keeps_full_capture_and_preflight_requirements(tmp_path, registry_capture, capsys):
    capture = _capture(tmp_path, registry_capture)
    args = ["--model", "mock_flow_vla", "--captured", str(capture.root)]
    assert main(["generate", *args, "--list"]) == 0
    assert "registry.linear" in capsys.readouterr().out
    assert main(["tune-all", *args, "--dry-run", "--output", str(tmp_path / "batches")]) == 0
    assert main(["generate", *args, "linear"]) == 2
    assert main(["tune-all", *args, "--agent", "fake/model", "--skip-preflight"]) == 2
    assert main(["generate", "--model", "pi05", "--model", "streamvln"]) == 2
    task = render_model(capture, tmp_path / "tasks")[0]
    assert main(["run", str(task.root), "--agent", "fake/model", "--output", str(tmp_path / "runs")]) == 2
    assert not (tmp_path / "runs").exists()


def test_replay_rejects_input_mutation_and_wrong_output_aliases(tmp_path, registry_capture):
    capture = _capture(tmp_path, registry_capture)
    task = render_model(capture, tmp_path / "tasks")[0]
    replay = read_json(task.root / "replay.json")
    case = next(iter(replay["cases"].values()))
    adapter = ReplayAdapter.__new__(ReplayAdapter)
    adapter.case, adapter.replay, adapter.cfg = case, replay, task.settings
    inputs = materialize(case["inputs"], task.root, 0)
    local = clone_inputs(inputs)
    assert adapter._check_state(local, local, inputs) == (0.0, 0.0)
    local[0].add_(1)
    with pytest.raises(ContractError, match="bytes"):
        adapter._check_state(local, local, inputs)
    with pytest.raises(ContractError, match="aliases"):
        adapter._check_outputs(inputs, [inputs[0]])


def test_replay_requires_every_workload(tmp_path, registry_capture):
    capture = _capture(tmp_path, registry_capture)
    task = render_model(capture, tmp_path / "tasks")[0]
    replay = read_json(task.root / "replay.json")
    replay["cases"] = {}
    atomic_json(task.root / "replay.json", replay)
    with pytest.raises(ContractError, match="every workload"):
        TaskPackage.load(task.root)


def test_registry_float64_baseline_and_definition_keep_exact_values(tmp_path, registry_capture):
    value = torch.tensor([[1.0 + 2**-30, 1.0 - 2**-30]], dtype=torch.float64)
    weight = torch.eye(2, dtype=torch.float64)
    capture = registry_capture(tmp_path, [("linear", (value, weight), {})])
    (task,) = render_model(capture, tmp_path / "tasks")
    assert task.definition["inputs"]["x0"]["dtype"] == "float64"
    assert task.definition["outputs"]["y0"]["dtype"] == "float64"
    spec = importlib.util.spec_from_file_location(
        "registry_float64_baseline", task.root / "baseline.py", submodule_search_locations=[str(task.root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = module.run(value, weight)
        assert torch.equal(result, value)
        assert not torch.equal(result, value.float().double())
    finally:
        for name in list(sys.modules):
            if name == spec.name or name.startswith(spec.name + "."):
                del sys.modules[name]
    replay = read_json(task.root / "replay.json")
    del replay["definition_schema"]
    atomic_json(task.root / "replay.json", replay)
    with pytest.raises(ContractError, match="explicit replay definition schema"):
        TaskPackage.load(task.root)


def test_float64_replay_schema_retains_upstream_validation(tmp_path, registry_capture):
    pytest.importorskip("flashinfer_bench")
    from benchmarks.kernel_tuning.replay_schema import ReplayDefinition
    from flashinfer_bench.data import Definition
    from pydantic import ValidationError

    task = render_model(_capture(tmp_path, registry_capture), tmp_path / "tasks")[0]
    definition = ReplayDefinition.model_validate(task.definition)
    assert isinstance(definition, Definition)
    assert definition.torch_output_dtypes == [torch.float64]
    with pytest.raises(ValidationError):
        Definition.model_validate(task.definition)
    wrong = {**task.definition, "outputs": {"y0": {"shape": ["unknown_axis"], "dtype": "float64"}}}
    with pytest.raises(ValidationError):
        ReplayDefinition.model_validate(wrong)


@pytest.mark.gpu
@pytest.mark.parametrize("timing", ["eager", "cuda_graph"])
def test_registry_baseline_passes_real_cuda_replay(tmp_path, registry_capture, timing):
    """Exercise registry calibration, real builders, layout checks and changed-input replay."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("flashinfer_bench")
    pytest.importorskip("triton")
    from benchmarks.kernel_tuning.calibration import calibrate
    from scripts.kernel_tuning.contracts import validate_measurement

    value = torch.ones(2, 16, device="cuda:0", dtype=torch.bfloat16)
    capture = registry_capture(tmp_path, [("gelu_tanh", (value,), {})], device_type="cuda")
    (task,) = render_model(
        calibrate(capture, tmp_path / "calibrated"),
        tmp_path / "tasks",
        overrides={"timing.mode": timing, "timing.warmup": 1, "timing.iterations": 2, "timing.trials": 1},
    )
    adapter = ReplayAdapter(task)
    report = adapter.evaluate({role: task.baseline() for role in ("baseline", "incumbent", "candidate")})
    validate_measurement(task, report)
