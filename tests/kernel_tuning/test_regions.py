"""CPU tests: capture-ops records the output contract of the call the model made."""

from __future__ import annotations

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("embodiinfer")

from scripts.kernel_tuning.discovery import tasks  # noqa: E402
from scripts.kernel_tuning.regions import OpCapture  # noqa: E402

from embodiinfer.backend.torch import ops as torch_ops  # noqa: E402


def _rope_inputs():
    # BHSD views of BSHD projections, as the policy passes them to rotate_qk.
    query = torch.randn(1, 3, 2, 4).transpose(1, 2)
    key = torch.randn(1, 3, 1, 4).transpose(1, 2)
    cos, sin = torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    return query, key, cos, sin


class Production:
    """Stand-in for a backend that allocates fresh contiguous outputs, like the Triton RoPE."""

    def __init__(self, contiguous: bool = True) -> None:
        self.calls, self.contiguous = 0, contiguous

    def __call__(self, *args):
        self.calls += 1
        result = torch_ops.rotate_qk(*args)
        return tuple(value.contiguous() for value in result) if self.contiguous else result


def _capture(tmp_path):
    return OpCapture("pi05", tmp_path / "capture", ops={"rotate_qk"})


def test_output_contract_follows_the_production_call(tmp_path):
    capture, production = _capture(tmp_path), Production()
    args = _rope_inputs()
    reference = torch_ops.rotate_qk(*args)
    assert not reference[0].is_contiguous(), "the reference must keep the input view layout here"

    result = capture.record("rotate_qk", torch_ops.rotate_qk, production, args, {})

    assert production.calls == 1
    assert all(value.is_contiguous() for value in result)
    assert not capture.errors
    (op,) = capture.operators.values()
    (case,) = op["workloads"]
    assert [spec["stride"] for spec in case["outputs"]] == [list(value.stride()) for value in result]
    assert [spec["stride"] for spec in case["reference_outputs"]] == [list(v.stride()) for v in reference]
    assert case["output_aliases"] == [[], []] and case["output_groups"] == [[0], [1]]


def test_matching_layouts_keep_the_previous_case_format(tmp_path):
    capture = _capture(tmp_path)
    capture.record("rotate_qk", torch_ops.rotate_qk, Production(contiguous=False), _rope_inputs(), {})
    (op,) = capture.operators.values()
    assert "reference_outputs" not in op["workloads"][0]


def test_disagreeing_structure_is_an_error_and_the_model_result_is_kept(tmp_path):
    capture = _capture(tmp_path)
    calls = []

    def production(*args):
        calls.append(1)
        return torch_ops.rotate_qk(*args)[0]  # one output instead of two

    result = capture.record("rotate_qk", torch_ops.rotate_qk, production, _rope_inputs(), {})
    assert len(calls) == 1 and isinstance(result, torch.Tensor)
    assert not capture.operators
    assert any("another structure" in error for error in capture.errors)


def test_generated_baseline_returns_the_production_layout(tmp_path):
    capture = _capture(tmp_path)
    args = _rope_inputs()
    capture.record("rotate_qk", torch_ops.rotate_qk, Production(), args, {})
    (op,) = capture.operators.values()
    code, files = tasks._sources(op)
    assert "_LAYOUTS" in code
    package = tmp_path / "pkg" / "relayout_baseline"
    for path, text in {**files, "baseline.py": code}.items():
        target = package / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    sys.path.insert(0, str(tmp_path / "pkg"))
    try:
        run = importlib.import_module("relayout_baseline.baseline").run
    finally:
        sys.path.remove(str(tmp_path / "pkg"))

    expected = torch_ops.rotate_qk(*args)
    outputs = run(*args)
    for value, reference, spec in zip(outputs, expected, op["workloads"][0]["outputs"]):
        assert list(value.stride()) == spec["stride"]
        assert torch.equal(value, reference)
    # Unrecorded input layouts are returned as the reference computes them.
    other = tuple(value.contiguous() for value in args)
    for value, reference in zip(run(*other), torch_ops.rotate_qk(*other)):
        assert value.stride() == reference.stride()


def test_baseline_code_is_unchanged_without_layout_differences(tmp_path):
    capture = _capture(tmp_path)
    capture.record("rotate_qk", torch_ops.rotate_qk, Production(contiguous=False), _rope_inputs(), {})
    (op,) = capture.operators.values()
    code, _ = tasks._sources(op)
    assert "_LAYOUTS" not in code and "_relayout" not in code
