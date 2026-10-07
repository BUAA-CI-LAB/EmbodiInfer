"""Model discovery, replay contracts and the all-task barrier, using CPU tensors."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.kernel_tuning import batch  # noqa: E402
from scripts.kernel_tuning.__main__ import main  # noqa: E402
from scripts.kernel_tuning.artifacts import atomic_json  # noqa: E402
from scripts.kernel_tuning.contracts import ContractError, TaskPackage, read_json  # noqa: E402
from scripts.kernel_tuning.discovery import ModelCapture  # noqa: E402
from scripts.kernel_tuning.discovery.recorder import CaptureSession  # noqa: E402
from scripts.kernel_tuning.discovery.tasks import render_model  # noqa: E402
from scripts.kernel_tuning.discovery.tensors import clone_inputs, materialize, storage_key  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_capture(monkeypatch):
    """Keep observer tests independent of CUDA initialization and numerical globals."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(old)


def capture_calls(tmp_path: Path, fn, *, fixture_bytes: int = 1024 * 1024) -> ModelCapture:
    session = CaptureSession(
        "mock_flow_vla", tmp_path / "capture", device_type="cpu", fixture_bytes=fixture_bytes
    )
    try:
        with session.observe(object(), checkpoint="test-revision"):
            fn()
    finally:
        session.close(complete=True)
    return ModelCapture.load(session.root, require_ready=False)


def generated_function(task: TaskPackage):
    scope = {}
    exec((task.root / "baseline.py").read_text(encoding="utf-8"), scope)
    return scope["run"]


def test_discovers_uncatalogued_ops_all_shapes_and_call_counts(tmp_path: Path) -> None:
    from benchmarks.kernel_tuning.calibration import calibrate

    values = [torch.ones(n, 4) for n in range(1, 12)]
    weight = torch.ones(4, 3)

    def forward():
        for value in [*values, values[0]]:
            (value @ weight).relu()

    capture = capture_calls(tmp_path, forward)
    ready = {op["name"]: op for op in capture.operators if op["status"] == "ready"}
    assert set(ready) == {"aten.mm.default", "aten.relu.default"}
    assert len(ready["aten.mm.default"]["workloads"]) == 11
    assert sum(case["count"] for case in ready["aten.mm.default"]["workloads"]) == 12
    tasks = render_model(calibrate(capture, tmp_path / "calibrated"), tmp_path / "tasks")
    assert len(tasks) == 2
    assert all(len(task.workloads) == 11 for task in tasks)
    for task in tasks:
        replay = read_json(task.root / "replay.json")
        assert task.settings.precision.mode == ("tolerance" if "mm." in replay["operator"] else "bit_exact")
        for case in replay["cases"].values():
            inputs = materialize(case["inputs"], task.root, 0)
            result = generated_function(task)(*inputs)
            assert list(result.shape) == case["outputs"][0]["shape"]


def test_noncontiguous_aliases_mutations_and_metadata_are_preserved(tmp_path: Path) -> None:
    backing = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    left, right = backing[:, 1:3], backing[:, 3:5]
    capture = capture_calls(tmp_path, lambda: (left.add_(right), left.transpose(0, 1)))
    assert any(op["status"] == "metadata_only" for op in capture.operators)
    (task,) = render_model(capture, tmp_path / "tasks")
    replay = read_json(task.root / "replay.json")
    (case,) = replay["cases"].values()
    assert replay["mutates"] == [0]
    inputs = materialize(case["inputs"], task.root, 0)
    assert storage_key(inputs[0]) == storage_key(inputs[1])
    assert inputs[0].stride() == (6, 1)
    assert inputs[0].storage_offset() == 1
    cloned = clone_inputs(inputs)
    assert storage_key(cloned[0]) == storage_key(cloned[1])
    assert storage_key(cloned[0]) != storage_key(inputs[0])
    expected = inputs[0] + inputs[1]
    output, mutated = generated_function(task)(*cloned)
    assert output is mutated and output is cloned[0]
    assert torch.equal(output, expected)


def test_required_integer_fixtures_are_valid_and_never_treated_as_quantized(tmp_path: Path) -> None:
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    indices = torch.tensor([4, 1, 0], dtype=torch.int64)
    capture = capture_calls(tmp_path, lambda: torch.nn.functional.embedding(indices, weight))
    assert not any(op["status"] == "excluded_quantized" for op in capture.operators)
    (task,) = render_model(capture, tmp_path / "tasks")
    (case,) = read_json(task.root / "replay.json")["cases"].values()
    args = materialize(case["inputs"], task.root, 0)
    assert torch.equal(generated_function(task)(*args), weight[indices])


def test_tensor_factories_without_tensor_inputs_get_tasks(tmp_path: Path) -> None:
    capture = capture_calls(tmp_path, lambda: torch.arange(9, device="cpu"))
    (task,) = render_model(capture, tmp_path / "tasks")
    assert task.definition["inputs"] == {}
    assert task.settings.device == "cpu"
    assert torch.equal(generated_function(task)(), torch.arange(9))


def test_scalar_reads_are_host_work_not_coverage_gaps(tmp_path: Path) -> None:
    from scripts.kernel_tuning.discovery.recorder import _Observer

    value = torch.arange(4, dtype=torch.float32).sum()
    session = CaptureSession("mock_flow_vla", tmp_path / "capture", device_type="cpu")
    try:
        with session.observe(object(), checkpoint="test-revision"):
            # Torch 2.12 delivers Tensor.item() to the observer as aten.item; older
            # releases decompose it first, so deliver the overload directly.
            result = _Observer(session).__torch_dispatch__(torch.ops.aten.item.default, (), (value,))
            assert result == 6.0
    finally:
        session.close(complete=True)
    capture = ModelCapture.load(session.root, require_ready=False)
    statuses = {op["name"]: op["status"] for op in capture.operators}
    assert statuses["aten.item.default"] == "host_only"
    assert "blocked" not in statuses.values()


def test_host_to_device_copies_keep_required_input_fixtures(tmp_path: Path) -> None:
    # "meta" stands in for the target device: host inputs, target-device output.
    session = CaptureSession("mock_flow_vla", tmp_path / "capture", device_type="meta")
    indices = torch.tensor([4, 1, 0], dtype=torch.int64)
    try:
        with session.observe(object(), checkpoint="test-revision"):
            indices.to("meta")
    finally:
        session.close(complete=True)
    capture = ModelCapture.load(session.root, require_ready=False)
    (op,) = [op for op in capture.operators if op["status"] == "ready"]
    (case,) = op["workloads"]
    assert case["inputs"][0]["device"] == "cpu"
    assert case["inputs"][0].get("fixture")


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_unrepresentable_host_preprocessing_is_not_a_device_gap(tmp_path: Path, device: str) -> None:
    # "meta" stands in for the target device; serving adapters check requests on the host.
    session = CaptureSession("mock_flow_vla", tmp_path / "capture", device_type="meta")
    try:
        with session.observe(object(), checkpoint="test-revision"):
            torch.ones(2, device=device).ne(float("inf"))  # torch.isfinite's decomposition
    finally:
        session.close(complete=True)
    capture = ModelCapture.load(session.root, require_ready=False)
    (op,) = [op for op in capture.operators if op["name"] == "aten.ne.Scalar"]
    # A device call with the same argument still needs a replay recipe.
    assert op["status"] == ("host_only" if device == "cpu" else "blocked")


def test_quantized_scope_includes_dynamic_unregistered_module_and_float_fallback(tmp_path: Path) -> None:
    from embodiinfer.layers.linear import INT8Config
    from embodiinfer.models import linear

    dense = torch.nn.Linear(4, 3, bias=False)
    quantized = linear.INT8Linear.from_linear(dense, INT8Config(backend="torch"))
    owner = torch.nn.Module()
    object.__setattr__(owner, "projection", quantized)
    assert list(owner.children()) == []
    value = torch.ones(2, 4)
    original = linear.QuantizedLinear.forward
    session = CaptureSession("mock_flow_vla", tmp_path / "capture", device_type="cpu")
    try:
        session.install()
        with session.observe(owner):
            output = owner.projection(value)
            output.relu()
    finally:
        session.close(complete=True)
    assert linear.QuantizedLinear.forward is original
    capture = ModelCapture.load(session.root)
    assert [op["name"] for op in capture.operators if op["status"] == "ready"] == ["aten.relu.default"]
    assert sum(op["status"] == "excluded_quantized" for op in capture.operators) == 1
    assert not any("mm." in op["name"] for op in capture.operators)


@pytest.mark.parametrize("failure", ["random", "fixture", "external"])
def test_any_coverage_gap_blocks_all_task_generation(tmp_path: Path, failure: str) -> None:
    value = torch.ones(4)
    index = torch.tensor([1, 0])

    def forward():
        value.relu()
        if failure == "random":
            torch.rand_like(value)
        elif failure == "fixture":
            value[index]
        else:
            custom(value)

    library = torch.library.Library("discovery_test", "FRAGMENT")
    if failure == "external":
        library.define("opaque(Tensor x) -> Tensor")
        library.impl("opaque", lambda x: x + 1, "CPU")
        custom = torch.ops.discovery_test.opaque.default
    capture = capture_calls(tmp_path, forward, fixture_bytes=0)
    assert any(op["status"] == "blocked" for op in capture.operators)
    with pytest.raises(ContractError, match="incomplete"):
        render_model(capture, tmp_path / "tasks")
    assert not (tmp_path / "tasks").exists()


def test_capture_model_and_immutable_fixtures_are_verified(tmp_path: Path) -> None:
    value = torch.ones(4)
    capture = capture_calls(tmp_path, value.relu)
    with pytest.raises(ContractError, match="differs from requested"):
        ModelCapture.load(capture.root, "pi05")
    fixture = next((capture.root / "fixtures").iterdir())
    fixture.write_bytes(b"changed")
    with pytest.raises(ContractError, match="content changed"):
        render_model(capture, tmp_path / "tasks")


def test_single_actual_target_and_uncompiled_execution_are_required(tmp_path: Path) -> None:
    session = CaptureSession("mock_flow_vla", tmp_path / "capture", device_type="cpu")
    try:
        session.install()
        from embodiinfer.policies.factory import make_policy

        with pytest.raises(ContractError, match="expected"):
            make_policy("pi05")
        with pytest.raises(ContractError, match="compilation"):
            torch.compile(lambda x: x)
        session.bind(object(), {})
        with pytest.raises(ContractError, match="one target"):
            session.bind(object(), {})
    finally:
        session.close(complete=False)


def test_model_batch_never_starts_agent_until_every_baseline_passes(tmp_path: Path, monkeypatch) -> None:
    from test_batch import measurement

    value = torch.ones(4)
    capture = capture_calls(tmp_path, lambda: value.relu().square())
    root = batch.create_model(
        capture, tmp_path / "batches", agent="fake/model", overrides={}, hardware_notes=None
    )
    commands = []

    def launcher(*args):
        commands.append(args)

    with pytest.raises(ContractError, match="every task"):
        batch.execute(root, launcher=launcher)

    async def evaluator(store, output, **kwargs):
        if "pow" in read_json(store.task.root / "replay.json")["operator"]:
            raise ContractError("missing evaluator support")
        return {
            "measurement": measurement(store.task, 10.0),
            "environment": {"identity": {"hardware": "fake"}},
        }

    monkeypatch.setattr(batch, "evaluate", evaluator)
    results = batch.preflight(root)
    assert any(error is None for error in results.values())
    assert any(error for error in results.values())
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
    # A subsequently changed task invalidates the whole model, including resume.
    task_root = root / next(iter(record["items"].values()))["task"]
    (task_root / "README.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ContractError, match="changed after generation"):
        batch.execute(root, launcher=launcher)
    assert commands == []


def test_model_cli_rejects_partial_selection_and_legacy_capture(tmp_path: Path, capsys) -> None:
    assert main(["generate", "--output", str(tmp_path / "tasks")]) == 2
    assert "exactly one --model" in capsys.readouterr().err
    value = torch.ones(4)
    capture = capture_calls(tmp_path, value.relu)
    args = ["--model", "mock_flow_vla", "--captured", str(capture.root)]
    assert main(["generate", *args, "--list"]) == 0
    assert "aten.relu.default" in capsys.readouterr().out
    assert main(["tune-all", *args, "--dry-run", "--output", str(tmp_path / "batches")]) == 0
    assert main(["generate", *args, "relu"]) == 2
    assert main(["tune-all", *args, "--agent", "fake/model", "--skip-preflight"]) == 2
    assert main(["generate", "--model", "pi05", "--model", "streamvln"]) == 2
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text("{}", encoding="utf-8")
    assert main(["generate", "--model", "pi05", "--captured", str(legacy)]) == 2
    (task,) = render_model(capture, tmp_path / "tasks")
    assert main(["run", str(task.root), "--agent", "fake/model", "--output", str(tmp_path / "runs")]) == 2
    assert not (tmp_path / "runs").exists()


_ENGINE_DRIVER = """
from concurrent.futures import ThreadPoolExecutor

import torch

from embodiinfer.engine import EngineConfig, EngineCore, GenerationBackend
from embodiinfer.policies.factory import make_policy
from embodiinfer.types import Observation

torch.manual_seed(0)
policy = make_policy("mock_flow_vla", preset="tiny")
cfg = policy.config
observation = Observation(
    images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
    state=torch.rand(cfg.state_dim),
    instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
)
x = torch.ones(3)
x.digamma()  # driver arithmetic outside the model is not a model operator
core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False, num_steps=2))
MODE
"""
_ENGINE_MODES = {
    # The async engine and batched serving execute the engine on a worker thread.
    "thread": "with ThreadPoolExecutor(1) as pool:\n"
    "    pool.submit(core.execute, policy.collate([observation], ['r0'])).result()\n",
    # The RL rollout path samples through decoder methods the engine never calls.
    "rollout": "GenerationBackend(core).generate_with_logprob([observation], num_steps=2)\n",
    # Benchmark-style drivers that call policy internals bypass the engine.
    "bypass": "batch = policy.collate([observation], ['r0']).to('cpu', torch.float32)\n"
    "policy.encode_prefix(batch)\n",
}


def _engine_capture(tmp_path: Path, monkeypatch, mode: str) -> ModelCapture:
    from scripts.kernel_tuning.discovery import recorder

    # The public CLI always targets CUDA; this test substitutes the observer's
    # device classification while exercising the real driver and engine hooks.
    monkeypatch.setattr(
        recorder, "CaptureSession", lambda *args, **kwargs: CaptureSession(*args, device_type="cpu", **kwargs)
    )
    script = tmp_path / f"{mode}.py"
    script.write_text(_ENGINE_DRIVER.replace("MODE", _ENGINE_MODES[mode]), encoding="utf-8")
    with contextlib.suppress(ContractError):  # Gaps are asserted from the published capture.
        recorder.run([str(script)], tmp_path / mode, "mock_flow_vla")
    return ModelCapture.load(tmp_path / mode, require_ready=False)


def _reference_operators(mode: str) -> set[str]:
    """Every dispatcher operator of the same inference, collected without the recorder."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from embodiinfer.engine import EngineConfig, EngineCore, GenerationBackend
    from embodiinfer.policies.factory import make_policy
    from embodiinfer.types import Observation

    names: set[str] = set()

    class Collect(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            names.add(str(func))
            return func(*args, **(kwargs or {}))

    torch.manual_seed(0)
    policy = make_policy("mock_flow_vla", preset="tiny")
    cfg = policy.config
    observation = Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
    )
    with Collect():
        core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False, num_steps=2))
        if mode == "thread":
            core.execute(policy.collate([observation], ["r0"]))
        else:
            GenerationBackend(core).generate_with_logprob([observation], num_steps=2)
    return names


@pytest.mark.parametrize("mode", ["thread", "rollout"])
def test_engine_inference_is_captured_end_to_end(tmp_path: Path, monkeypatch, mode: str) -> None:
    capture = _engine_capture(tmp_path, monkeypatch, mode)
    capture.require_ready()
    assert capture.manifest["target"]["config"] == {"preset": "tiny"}
    captured = {op["name"] for op in capture.operators}
    # No policy method list: the whole inference matches an independent collector.
    assert captured == _reference_operators(mode)
    assert "aten.digamma.default" not in captured


def test_drive_replays_recorded_observations_through_the_engine(tmp_path: Path, monkeypatch) -> None:
    import json

    from scripts.kernel_tuning.discovery import recorder
    from scripts.kernel_tuning.discovery.drive import save_inputs

    from embodiinfer.policies.mock import preset_config
    from embodiinfer.types import Observation

    cfg = preset_config("tiny")

    def observation() -> Observation:
        return Observation(
            images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
            state=torch.rand(cfg.state_dim),
            instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
        )

    save_inputs(tmp_path / "inputs.pt", "observations", [[observation(), observation()], [observation()]])
    (tmp_path / "serving.json").write_text(
        json.dumps({"policy_kwargs": {"preset": "tiny"}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        recorder, "CaptureSession", lambda *args, **kwargs: CaptureSession(*args, device_type="cpu", **kwargs)
    )
    command = ["-m", "scripts.kernel_tuning.discovery.drive", "--policy", "mock_flow_vla", "--device", "cpu"]
    command += ["--adapter-config", str(tmp_path / "serving.json"), "--inputs", str(tmp_path / "inputs.pt")]
    recorder.run(command, tmp_path / "capture", "mock_flow_vla")
    capture = ModelCapture.load(tmp_path / "capture", "mock_flow_vla")
    assert capture.manifest["target"]["config"] == {"preset": "tiny"}
    addmm = next(op for op in capture.operators if op["name"] == "aten.addmm.default")
    # Three requests, each with a prefix and the default denoise loop, were observed.
    assert sum(case["count"] for case in addmm["workloads"]) > 3


def test_model_arithmetic_outside_the_engine_is_a_coverage_gap(tmp_path: Path, monkeypatch) -> None:
    capture = _engine_capture(tmp_path, monkeypatch, "bypass")
    errors = capture.manifest["errors"]
    assert any("read model weights outside an inference entry" in error for error in errors)
    with pytest.raises(ContractError, match="outside an inference entry"):
        capture.require_ready()


def test_capture_without_inference_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from scripts.kernel_tuning.discovery import recorder

    from embodiinfer.policies import factory

    monkeypatch.setitem(factory._REGISTRY, "mock_flow_vla", lambda **kwargs: torch.nn.Linear(2, 2))
    monkeypatch.setattr(
        recorder, "CaptureSession", lambda *args, **kwargs: CaptureSession(*args, device_type="cpu", **kwargs)
    )
    script = tmp_path / "prepare.py"
    script.write_text(
        "from embodiinfer.policies.factory import make_policy\nmake_policy('mock_flow_vla')\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError):
        recorder.run([str(script)], tmp_path / "capture", "mock_flow_vla")
    manifest = read_json(tmp_path / "capture" / "manifest.json")
    assert any("No end-to-end inference ran" in error for error in manifest["errors"])


def test_replay_adapter_checks_output_alias_and_undeclared_mutation_on_cpu(tmp_path: Path) -> None:
    from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter

    value = torch.ones(4)
    capture = capture_calls(tmp_path, lambda: value.add_(2))
    (task,) = render_model(capture, tmp_path / "tasks")
    replay = read_json(task.root / "replay.json")
    (case,) = replay["cases"].values()
    # Exercise the contract logic without constructing the GPU/compiler adapter.
    adapter = ReplayAdapter.__new__(ReplayAdapter)
    adapter.case, adapter.replay, adapter.cfg = case, replay, task.settings
    inputs = materialize(case["inputs"], task.root, 0)
    expected = clone_inputs(inputs)
    outputs = generated_function(task)(*expected)
    adapter._check_outputs(expected, list(outputs))
    assert adapter._check_state(expected, expected, inputs) == (0.0, 0.0)
    with pytest.raises(ContractError, match="aliases"):
        adapter._check_outputs(expected, [x.clone() for x in outputs])
    adapter.replay = {**replay, "mutates": []}
    with pytest.raises(ContractError, match="bytes"):
        adapter._check_state(expected, expected, inputs)


def test_missing_workload_and_unbound_output_axes_are_rejected(tmp_path: Path) -> None:
    from benchmarks.kernel_tuning.calibration import calibrate

    value = torch.ones(4, 2)
    capture = capture_calls(tmp_path, lambda: value.sum(1))
    (task,) = render_model(calibrate(capture, tmp_path / "calibrated"), tmp_path / "tasks")
    replay = read_json(task.root / "replay.json")
    replay["cases"] = {}
    atomic_json(task.root / "replay.json", replay)
    with pytest.raises(ContractError, match="every workload"):
        TaskPackage.load(task.root)


@pytest.mark.parametrize("known", [True, False])
def test_existing_backend_is_atomic_and_requires_a_replay_contract(
    tmp_path: Path, monkeypatch, known: bool
) -> None:
    from scripts.kernel_tuning.discovery import recorder

    from embodiinfer.backend.triton import norm

    if not known:
        monkeypatch.setattr(recorder, "_PURE_KERNELS", recorder._PURE_KERNELS - {"gated_residual"})

    values = [torch.ones(1, 2, 4) for _ in range(3)]
    session = CaptureSession("mock_flow_vla", tmp_path / "capture", device_type="cpu")
    original = norm.gated_residual
    try:
        session.install()
        with session.observe(object()):
            result = norm.gated_residual(*values)
    finally:
        session.close(complete=True)
    assert norm.gated_residual is original
    assert torch.equal(result, torch.full_like(result, 2))
    capture = ModelCapture.load(session.root, require_ready=False)
    (op,) = capture.operators
    assert op["identity"]["kind"] == "backend"
    assert op["name"].endswith(".gated_residual")
    if not known:
        with pytest.raises(ContractError, match="explicit mutation/reference contract"):
            capture.require_ready()
        return
    (task,) = render_model(capture, tmp_path / "tasks")
    assert "vendor/embodiinfer/backend/triton/norm.py" in task.hashes
    assert "from .vendor.embodiinfer.backend.triton.norm" in task.definition["reference"]


def test_changed_numerical_settings_block_capture(tmp_path: Path) -> None:
    value = torch.ones(4)

    def forward():
        value.relu()
        torch.set_float32_matmul_precision("high")
        value.square()

    capture = capture_calls(tmp_path, forward)
    with pytest.raises(ContractError, match="Numerical settings changed"):
        capture.require_ready()


def test_greedy_workspace_has_an_explicit_tensor_state_recipe() -> None:
    from scripts.kernel_tuning.discovery.tasks import _expression
    from scripts.kernel_tuning.discovery.tensors import flatten, restore

    from embodiinfer.backend.triton.sampling import GreedyWorkspace

    workspace = GreedyWorkspace.allocate(16, torch.device("cpu"))
    tensors = []
    recipe = flatten(workspace, tensors)
    cloned = clone_inputs(tensors)
    restored = restore(recipe, cloned)
    assert restored.vocab_size == 16
    assert restored.token is cloned[recipe["greedy_workspace"]["token"]["tensor"]]
    scope = {"GreedyWorkspace": GreedyWorkspace, **{f"x{i}": tensor for i, tensor in enumerate(cloned)}}
    constructed = eval(_expression(recipe), scope)
    assert constructed.penalized is restored.penalized
    assert constructed.token is restored.token


@pytest.mark.gpu
@pytest.mark.parametrize("timing", ["eager", "cuda_graph"])
def test_discovered_mutable_baseline_passes_cuda_replay(tmp_path: Path, timing: str) -> None:
    """Target-runtime smoke test: real builders, alias checks, resets and graph A/B/A."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("flashinfer_bench")
    pytest.importorskip("triton")
    from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter
    from scripts.kernel_tuning.contracts import validate_measurement

    value = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(2, 8)[:, 1:5]
    session = CaptureSession("mock_flow_vla", tmp_path / "capture")
    try:
        with session.observe(object()):
            value.add_(1)
    finally:
        session.close(complete=True)
    (task,) = render_model(
        ModelCapture.load(session.root),
        tmp_path / "tasks",
        overrides={"timing.mode": timing, "timing.warmup": 2, "timing.iterations": 5, "timing.trials": 1},
    )
    adapter = ReplayAdapter(task)
    validate_measurement(
        task, adapter.evaluate({role: task.baseline() for role in ("baseline", "incumbent", "candidate")})
    )
