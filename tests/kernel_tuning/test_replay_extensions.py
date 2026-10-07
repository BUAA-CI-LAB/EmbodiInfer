"""Lossless random-state and float64 contracts for discovered operator tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.kernel_tuning.contracts import ContractError, TaskPackage, read_json  # noqa: E402
from scripts.kernel_tuning.discovery.contracts import ModelCapture  # noqa: E402
from scripts.kernel_tuning.discovery.recorder import CaptureSession  # noqa: E402
from scripts.kernel_tuning.discovery.tasks import render_model  # noqa: E402
from scripts.kernel_tuning.discovery.tensors import materialize  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_numerics(monkeypatch):
    """CPU tests must neither initialize CUDA nor leak numerical/RNG settings."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    precision, state = torch.get_float32_matmul_precision(), torch.get_rng_state()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)
    torch.set_rng_state(state)


def record(tmp_path: Path, fn, *, fixture_bytes: int = 1024 * 1024) -> ModelCapture:
    """Run one CPU operation through the production observer and immutable loader."""
    session = CaptureSession(
        "mock_flow_vla", tmp_path / "capture", device_type="cpu", fixture_bytes=fixture_bytes
    )
    try:
        with session.observe(object()):
            fn()
    finally:
        session.close(complete=True)
    return ModelCapture.load(session.root, require_ready=False)


def baseline(task):
    """Load the generated single-file ATen baseline without a GPU builder."""
    scope = {}
    exec((task.root / "baseline.py").read_text(encoding="utf-8"), scope)
    return scope["run"]


def test_default_randn_records_real_state_and_generates_a_task(tmp_path: Path) -> None:
    torch.manual_seed(123)
    torch.randn(7)  # Include a non-initial generator position/cache.
    before = torch.get_rng_state().clone()
    observed = []
    capture = record(tmp_path, lambda: observed.append(torch.randn(3, 5, generator=None)))
    after = torch.get_rng_state().clone()
    capture.require_ready()
    (task,) = render_model(capture, tmp_path / "tasks")
    replay = read_json(task.root / "replay.json")
    assert replay["rng"] == {"kind": "torch_default", "devices": ["cpu"]}
    (case,) = replay["cases"].values()
    for phase, state in (("before", before), ("after", after)):
        assert (task.root / case["rng"][phase]["cpu"]).read_bytes() == bytes(state.tolist())
    torch.set_rng_state(before)
    assert torch.equal(baseline(task)(), observed[0])
    assert torch.equal(torch.get_rng_state(), after)


def test_float64_task_preserves_dtype_and_values(tmp_path: Path) -> None:
    value = torch.tensor([1.0 + 2**-30, 1.0 - 2**-30], dtype=torch.float64)
    capture = record(tmp_path, value.sin)
    (task,) = render_model(capture, tmp_path / "tasks")
    assert task.definition["inputs"]["x0"]["dtype"] == "float64"
    assert task.definition["outputs"]["y0"]["dtype"] == "float64"
    replay = read_json(task.root / "replay.json")
    assert replay["definition_schema"] == "embodiinfer-replay-v1"
    (case,) = replay["cases"].values()
    inputs = materialize(case["inputs"], task.root, 0)
    actual = baseline(task)(*inputs)
    assert actual.dtype == torch.float64
    assert torch.equal(actual, value.sin())
    assert not torch.equal(actual, value.float().sin().double())


def test_rng_state_is_required_bounded_and_immutable(tmp_path: Path) -> None:
    capture = record(tmp_path / "bounded", lambda: torch.randn(2), fixture_bytes=0)
    with pytest.raises(ContractError, match="RNG fixture.*budget"):
        capture.require_ready()
    assert len(capture.operators) == 1  # Instrumentation does not capture its own state operations.
    capture = record(
        tmp_path / "after_bounded", lambda: torch.randn(2), fixture_bytes=torch.get_rng_state().numel()
    )
    with pytest.raises(ContractError, match="RNG fixture.*budget"):
        capture.require_ready()
    capture = record(tmp_path / "valid", lambda: torch.randn(2))
    (task,) = render_model(capture, tmp_path / "tasks")
    replay = read_json(task.root / "replay.json")
    (case,) = replay["cases"].values()
    (task.root / case["rng"]["after"]["cpu"]).unlink()
    with pytest.raises(ContractError, match="RNG fixture missing"):
        TaskPackage.load(task.root)
    with pytest.raises(ContractError, match="eager timing"):
        render_model(capture, tmp_path / "graphs", overrides={"timing.mode": "cuda_graph"})


def test_unsupported_rng_and_explicit_generators_still_block(tmp_path: Path) -> None:
    for name, fn in (
        ("explicit", lambda: torch.randn(2, generator=torch.Generator().manual_seed(7))),
        ("uniform", lambda: torch.rand(2)),
    ):
        capture = record(tmp_path / name, fn)
        with pytest.raises(ContractError, match="incomplete"):
            capture.require_ready()


@pytest.mark.parametrize("style", ["omitted", "positional", "keyword"])
def test_zero_dropout_sdpa_is_not_a_random_call(tmp_path: Path, style: str) -> None:
    from benchmarks.kernel_tuning.calibration import calibrate

    q = torch.randn(1, 2, 8, 16)
    state = torch.get_rng_state().clone()
    outputs = []

    def forward():
        op = torch.ops.aten.scaled_dot_product_attention.default
        for _ in range(2):
            if style == "omitted":
                outputs.append(op(q, q, q))
            elif style == "positional":
                outputs.append(op(q, q, q, None, 0.0))
            else:
                outputs.append(op(q, q, q, dropout_p=0.0))

    with torch.inference_mode():
        capture = record(tmp_path, forward)
    capture.require_ready()
    assert len(capture.operators) == 1
    assert capture.operators[0]["name"] == "aten.scaled_dot_product_attention.default"
    assert sum(c["count"] for c in capture.operators[0]["workloads"]) == 2
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(outputs[0], outputs[1])
    (task,) = render_model(calibrate(capture, tmp_path / "calibrated"), tmp_path / "tasks")
    assert "rng" not in read_json(task.root / "replay.json")


def test_zero_dropout_sdpa_backend_overload_is_not_a_random_call(tmp_path: Path) -> None:
    # Under the engine's no_grad, the functional API reaches the observer as a backend overload.
    q = torch.randn(1, 2, 8, 16)
    with torch.no_grad():
        capture = record(tmp_path, lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q))
    assert [op["name"] for op in capture.operators] == [
        "aten._scaled_dot_product_flash_attention_for_cpu.default"
    ]
    capture.require_ready()


def test_unused_attention_dropout_state_is_not_compared(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter
    from scripts.kernel_tuning.discovery.rng import undefined_outputs

    op = torch.ops.aten._scaled_dot_product_efficient_attention.default
    q = torch.empty(1, 2, 8, 16)
    # CUDA leaves philox seed/offset uninitialized when dropout is zero.
    assert undefined_outputs(op, (q, q, q, None, False), {}) == [2, 3]
    assert undefined_outputs(op, (q, q, q, None, False, 0.1), {}) == []
    assert undefined_outputs(torch.ops.aten.mm.default, (q, q), {}) == []

    adapter = ReplayAdapter.__new__(ReplayAdapter)
    adapter.cfg = SimpleNamespace(precision=SimpleNamespace(mode="bit_exact"))
    adapter.replay = {"undefined_outputs": [1]}
    value = torch.ones(3)
    assert adapter._compare([value, torch.tensor(16)], [value, torch.tensor(0)], []) == (0.0, 0.0)
    with pytest.raises(ContractError, match="bit-exact"):
        adapter._compare([value + 1, torch.tensor(0)], [value, torch.tensor(0)], [])


@pytest.mark.parametrize("style", ["positional", "keyword"])
def test_nonzero_dropout_sdpa_remains_a_coverage_gap(tmp_path: Path, style: str) -> None:
    q = torch.randn(1, 2, 8, 16)

    def forward():
        op = torch.ops.aten.scaled_dot_product_attention.default
        return op(q, q, q, None, 0.25) if style == "positional" else op(q, q, q, dropout_p=0.25)

    with torch.inference_mode():
        capture = record(tmp_path, forward)
    with pytest.raises(ContractError, match="RNG replay adapter"):
        capture.require_ready()


def test_rng_outputs_and_advancement_are_checked_and_caller_state_restored(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter

    capture = record(tmp_path, lambda: torch.randn(15, generator=None))
    (task,) = render_model(capture, tmp_path / "tasks")
    adapter = ReplayAdapter.__new__(ReplayAdapter)
    adapter.task, adapter.cfg = task, task.settings
    adapter.definition = SimpleNamespace(
        outputs=task.definition["outputs"], torch_output_dtypes=[torch.float32]
    )
    adapter.replay = read_json(task.root / "replay.json")
    adapter.device = "cpu"
    adapter._check_flags = lambda: None
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

    def unexpected_global_seed(*args, **kwargs):
        pytest.fail("RNG replay must not seed unrelated generators")

    monkeypatch.setattr(torch, "manual_seed", unexpected_global_seed)

    class Runnable:
        metadata = SimpleNamespace(destination_passing_style=False)

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    fn = baseline(task)
    reference = Runnable(fn)

    def extra_draw():
        output = fn()
        torch.randn(1)
        return output

    def rewind():
        state = torch.get_rng_state()
        output = fn()
        torch.set_rng_state(state)
        return output

    def crash():
        torch.randn(1)
        raise RuntimeError("candidate failed")

    for seed in (0, 1, 2):
        inputs = adapter._inputs(SimpleNamespace(uuid=task.workload_ids[0]), seed)
        saved = torch.get_rng_state().clone()
        assert adapter._verify_call(reference, reference, inputs, seed) == (0.0, 0.0)
        assert torch.equal(torch.get_rng_state(), saved)
        for bad in (extra_draw, rewind):
            with pytest.raises(ContractError, match="RNG state"):
                adapter._verify_call(Runnable(bad), reference, inputs, seed)
            assert torch.equal(torch.get_rng_state(), saved)
        with pytest.raises(RuntimeError, match="candidate failed"):
            adapter._verify_call(Runnable(crash), reference, inputs, seed)
        assert torch.equal(torch.get_rng_state(), saved)


def test_float64_extension_requires_replay_and_retains_upstream_validation(tmp_path: Path) -> None:
    pytest.importorskip("flashinfer_bench")
    from benchmarks.kernel_tuning.replay_schema import ReplayDefinition
    from flashinfer_bench.data import Definition
    from pydantic import ValidationError

    capture = record(tmp_path, lambda: torch.linspace(0, 1, 8, dtype=torch.float64))
    (task,) = render_model(capture, tmp_path / "tasks")
    definition = ReplayDefinition.model_validate(task.definition)
    assert isinstance(definition, Definition)
    assert definition.torch_output_dtypes == [torch.float64]
    assert definition.outputs["y0"].dtype == "float64"
    assert definition.model_dump(mode="json")["outputs"]["y0"]["dtype"] == "float64"
    with pytest.raises(ValidationError):
        Definition.model_validate(task.definition)  # No process-wide dtype monkeypatch.
    wrong = {**task.definition, "outputs": {"y0": {"shape": ["unknown_axis"], "dtype": "float64"}}}
    with pytest.raises(ValidationError):
        ReplayDefinition.model_validate(wrong)
    replay = read_json(task.root / "replay.json")
    del replay["definition_schema"]
    import json

    (task.root / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    with pytest.raises(ContractError, match="explicit replay definition schema"):
        TaskPackage.load(task.root)


@pytest.mark.gpu
@pytest.mark.parametrize("operation", ["randn", "float64", "sdpa"])
def test_cuda_rng_and_float64_baselines_pass_real_evaluator(tmp_path: Path, operation: str) -> None:
    """Exercise real Python builders, byte comparison, RNG transitions and eager resets."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("flashinfer_bench")
    pytest.importorskip("triton")
    from benchmarks.kernel_tuning.calibration import calibrate
    from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter
    from scripts.kernel_tuning.contracts import validate_measurement

    state = torch.cuda.get_rng_state().clone()
    query = torch.ones(1, 2, 8, 16, device="cuda:0", dtype=torch.bfloat16)
    session = CaptureSession("mock_flow_vla", tmp_path / "capture")
    try:
        with torch.inference_mode(), session.observe(object()):
            if operation == "randn":
                torch.randn(1, 50, 32, device="cuda:0", dtype=torch.bfloat16, generator=None)
            elif operation == "sdpa":
                torch.nn.functional.scaled_dot_product_attention(query, query, query, dropout_p=0.0)
            else:
                times = torch.linspace(0, 1, 16, device="cuda:0", dtype=torch.float64)
                (1000.0**times).reciprocal().sin().to(torch.bfloat16)
    finally:
        session.close(complete=True)
        torch.cuda.set_rng_state(state)
    tasks = render_model(
        calibrate(ModelCapture.load(session.root), tmp_path / "calibrated"),
        tmp_path / "tasks",
        overrides={"timing.warmup": 1, "timing.iterations": 2, "timing.trials": 1},
    )
    for task in tasks:
        adapter = ReplayAdapter(task)
        before = torch.cuda.get_rng_state().clone()
        report = adapter.evaluate({role: task.baseline() for role in ("baseline", "incumbent", "candidate")})
        validate_measurement(task, report)
        if operation == "randn":
            assert torch.equal(before, torch.cuda.get_rng_state())
