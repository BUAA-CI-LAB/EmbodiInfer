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
from scripts.kernel_tuning.discovery.recorder import CaptureSession  # noqa: E402
from scripts.kernel_tuning.discovery.tasks import render_model  # noqa: E402
from scripts.kernel_tuning.discovery.tensors import flatten  # noqa: E402


def record(root: Path, fn, *, budget: int = 1024 * 1024) -> ModelCapture:
    session = CaptureSession("test", root / "raw", device_type="cpu", fixture_bytes=budget)
    try:
        with torch.inference_mode(), session.observe(object()):
            fn()
    finally:
        session.close(complete=True)
    return ModelCapture.load(session.root, require_ready=False)


def identity(name: str, *args, **kwargs) -> tuple[dict, list]:
    tensors = []
    recipe = flatten((args, kwargs), tensors)
    return {
        "kind": "aten",
        "name": name,
        "arguments": recipe,
        "input_dtypes": [str(t.dtype).removeprefix("torch.") for t in tensors],
        "output_dtypes": ["float32"],
        "mutates": [],
    }, tensors


@pytest.fixture
def calibrated(tmp_path: Path) -> ModelCapture:
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(3, 16, generator=generator)
    b = torch.randn(16, 5, generator=generator)
    c = torch.randn(4, 16, generator=generator) * 64
    raw = record(tmp_path, lambda: (a @ b, c @ b))
    return calibrate(raw, tmp_path / "calibrated")


def test_semantics_do_not_relax_copies_rng_mutation_integer_or_fp64() -> None:
    x = torch.ones(3, 3)
    for name in (
        "aten.mm.default",
        "aten.linear.default",
        "aten.sum.default",
        "aten.scaled_dot_product_attention.default",
    ):
        spec, _ = identity(name, x)
        assert policy(spec) == "aten"
    for name in ("aten.clone.default", "aten.copy_.default", "aten.relu.default", "aten.index.Tensor"):
        spec, _ = identity(name, x)
        assert policy(spec) is None
    spec, _ = identity("aten.mm.default", x)
    for extra in (
        {"mutates": [0]},
        {"rng": {"kind": "torch_default"}},
        {"input_dtypes": ["float64"]},
        {"output_dtypes": ["int64"]},
    ):
        assert policy({**spec, **extra}) is None


def test_generation_requires_calibration_and_capture_keeps_only_execution_switches(tmp_path: Path) -> None:
    x = torch.ones(3, 3)
    capture = record(tmp_path, lambda: x @ x)
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


def test_real_inputs_cannot_silently_fall_back_to_random_when_budget_exhausted(tmp_path: Path) -> None:
    x = torch.ones(8, 8)
    capture = record(tmp_path, lambda: x @ x, budget=0)
    assert capture.operators[0]["status"] == "blocked"
    assert "fixture" in capture.operators[0]["reason"]
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


@pytest.mark.parametrize(
    "causal,gqa,masked", [(False, False, True), (True, False, False), (False, True, True)]
)
def test_fp64_attention_matches_math_reference(causal: bool, gqa: bool, masked: bool) -> None:
    generator = torch.Generator().manual_seed(2)
    q = torch.randn(1, 4, 7, 8, generator=generator)
    k = torch.randn(1, 2 if gqa else 4, 9, 8, generator=generator)
    v = torch.randn(1, 2 if gqa else 4, 9, 8, generator=generator)
    mask = torch.ones(7, 9, dtype=torch.bool) if masked else None
    if mask is not None:
        mask[0] = False
        mask[1:, -2:] = False
    spec, tensors = identity(
        "aten.scaled_dot_product_attention.default",
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=0.25,
        enable_gqa=gqa,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), attn_mask=mask, is_causal=causal, scale=0.25, enable_gqa=gqa
    )
    torch.testing.assert_close(high_precision(spec, tensors)[0], expected, atol=1e-12, rtol=1e-12)


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
    with pytest.raises(ContractError, match="frozen calibrated"):
        adapter._verify_call(Runnable(lambda args: saved), reference, adapter._inputs(workload, 0), 0)
    with pytest.raises(ContractError, match="frozen calibrated"):
        adapter._verify_graph(Runnable(lambda args: saved), reference, workload, 0)
    assert adapter._check_metadata()["numerical_contract"] == task.hashes["numerics.json"]
    assert adapter._check_metadata()["validation_profiles"] == [list(pair) for pair in PROFILES]
    assert asdict(task.settings.precision)["mode"] == "tolerance"


def test_random_profiles_preserve_masks_and_change_real_values(tmp_path: Path) -> None:
    q = torch.ones(1, 1, 3, 4)
    mask = torch.tensor([[True, False, True]]).expand(3, 3)
    capture = record(tmp_path, lambda: torch.ops.aten.scaled_dot_product_attention.default(q, q, q, mask))
    # Current Torch observes the public SDPA overload atomically.
    op = next(op for op in capture.operators if op["name"] == "aten.scaled_dot_product_attention.default")
    case = op["workloads"][0]
    real = validation_inputs(op["identity"], case, capture.root, "recorded", 0)
    random = validation_inputs(op["identity"], case, capture.root, "random", 1)
    assert not torch.equal(real[0], random[0])
    assert torch.equal(real[-1], random[-1])


def test_rms_norm_default_epsilon_preserves_input_dtype_semantics() -> None:
    value = torch.full((2, 4), 0.001, dtype=torch.bfloat16)
    spec, tensors = identity("aten.rms_norm.default", value, [4])
    actual = high_precision(spec, tensors)[0]
    expected = torch.nn.functional.rms_norm(value.double(), [4], eps=torch.finfo(value.dtype).eps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_calibrate_cli_writes_a_new_capture_before_generation(tmp_path: Path, capsys) -> None:
    from scripts.kernel_tuning.__main__ import main

    value = torch.ones(2, 3)
    weight = torch.ones(3, 4)
    raw = record(tmp_path, lambda: value @ weight)
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
