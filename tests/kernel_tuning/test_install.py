"""CPU tests for installing a registry solution into a kernel profile."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("embodiinfer")

from scripts.kernel_tuning import install as installer  # noqa: E402
from scripts.kernel_tuning.contracts import ContractError  # noqa: E402

import embodiinfer.backend.triton.tuned as tuned  # noqa: E402
from embodiinfer.layers import KernelSelection  # noqa: E402

KERNEL = """CALLS = []


def fast(residual, update, gate):
    CALLS.append(tuple(residual.shape))
    return residual + update * gate
"""


def _spec(shape, dtype="float32"):
    stride, step = [], 1
    for size in reversed(shape):
        stride.insert(0, step)
        step *= size
    return {
        "shape": list(shape),
        "stride": stride,
        "offset": 0,
        "dtype": dtype,
        "device": "cpu",
        "storage": 0,
    }


@pytest.fixture
def export(tmp_path, monkeypatch):
    root = tmp_path / "export"
    (root / "task").mkdir(parents=True)
    (root / "sources").mkdir()
    (root / "sources" / "kernel.py").write_text(KERNEL, encoding="utf-8")
    recipe = {"tuple": [{"tuple": [{"tensor": 0}, {"tensor": 1}, {"tensor": 2}]}, {"dict": {}}]}
    shape, gate = [1, 2, 4], [1, 1, 4]
    inputs = [_spec(shape), {**_spec(shape), "storage": 1}, {**_spec(gate), "storage": 2}]
    files = {
        "task/generated.json": {
            "generator": "scripts.kernel_tuning.regions",
            "registry_op": "gated_residual",
            "capture_identity": "capture",
        },
        "task/replay.json": {
            "torch_version": str(torch.__version__),
            "numerical_identity": {
                "op": "gated_residual",
                "arguments": recipe,
                "input_dtypes": ["float32"] * 3,
            },
            "cases": {"case": {"inputs": inputs}},
        },
        "solution.json": {"spec": {"language": "triton", "entry_point": "kernel.py::fast"}},
        "manifest.json": {
            "run_id": "run-1",
            "best": "c1",
            "attempts": [{"id": "c1", "solution_digest": "sol", "decision": {"promoted": True}}],
        },
        "checksums.json": {"manifest.json": "abc"},
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    environment = {
        "device_name": "Test GPU",
        "capability": [9, 9],
        "torch": str(torch.__version__),
        "triton": "x",
    }
    monkeypatch.setattr(
        installer.TaskPackage, "load", lambda path: SimpleNamespace(root=path, identity="task")
    )
    monkeypatch.setattr(tuned, "current_environment", lambda: dict(environment))
    return root, environment


def test_installed_kernel_applies_only_to_verified_layouts(export, tmp_path):
    root, _ = export
    profiles = tmp_path / "tuned"
    result = installer.install(root, "testgpu", root=profiles)
    assert result["verified_layouts"] == 1
    profile = tuned.load_profile(profiles / "testgpu")
    assert profile.ops["gated_residual"]["evidence"]["run_id"] == "run-1"
    op = KernelSelection(profile).resolve("gated_residual")
    calls = profile.kernel("gated_residual").fn.__globals__["run"].__globals__["CALLS"]
    x, gate = torch.ones(1, 2, 4), torch.full((1, 1, 4), 2.0)
    torch.testing.assert_close(op(x, x, gate), x + x * gate)
    assert calls == [(1, 2, 4)]
    y = torch.ones(1, 3, 4)
    torch.testing.assert_close(op(y, y, gate), y + y * gate)
    assert calls == [(1, 2, 4)]  # unverified layout fell back
    z = torch.ones(2, 4).t().contiguous().t().unsqueeze(0)  # (1, 2, 4) with transposed strides
    torch.testing.assert_close(op(z, z, gate), z + z * gate)
    assert calls == [(1, 2, 4)]  # a different layout of the same shape also falls back
    odd = torch.ones(5, 2, 4)[2:3]  # size-one batch dimension with another stride
    torch.testing.assert_close(op(odd, odd, gate), odd + odd * gate)
    assert calls == [(1, 2, 4), (1, 2, 4)]  # its stride addresses nothing, so it matches
    assert op.__closure__ is not None  # a dispatching wrapper, not the raw kernel


def test_reinstall_requires_replace_and_matching_environment(export, tmp_path, monkeypatch):
    root, environment = export
    profiles = tmp_path / "tuned"
    installer.install(root, "testgpu", root=profiles)
    with pytest.raises(ContractError, match="already installed"):
        installer.install(root, "testgpu", root=profiles)
    installer.install(root, "testgpu", root=profiles, replace=True)
    monkeypatch.setattr(tuned, "current_environment", lambda: {**environment, "device_name": "Other"})
    with pytest.raises(ContractError, match="was verified for"):
        installer.install(root, "testgpu", root=profiles, replace=True)


def test_edited_installed_kernel_is_rejected(export, tmp_path):
    root, _ = export
    profiles = tmp_path / "tuned"
    installer.install(root, "testgpu", root=profiles)
    (profiles / "testgpu" / "gated_residual" / "kernel.py").write_text(
        KERNEL + "\n# edit\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sha256"):
        tuned.load_profile(profiles / "testgpu").kernel("gated_residual")


def test_only_registry_tasks_install(export, tmp_path):
    root, _ = export
    (root / "task" / "generated.json").write_text(json.dumps({"generator": "other"}), encoding="utf-8")
    with pytest.raises(ContractError, match="Only registry-operator tasks"):
        installer.install(root, "testgpu", root=tmp_path / "tuned")


def test_installed_kernel_receives_only_tensor_arguments(export, tmp_path):
    root, _ = export
    replay = json.loads((root / "task" / "replay.json").read_text(encoding="utf-8"))
    arguments = replay["numerical_identity"]["arguments"]["tuple"][0]["tuple"]
    arguments.append({"constant": 0.5})  # e.g. an attention scale baked into the task
    (root / "task" / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    profiles = tmp_path / "tuned"
    installer.install(root, "testgpu", root=profiles)
    kernel = tuned.load_profile(profiles / "testgpu").kernel("gated_residual")
    x, gate = torch.ones(1, 2, 4), torch.full((1, 1, 4), 2.0)
    assert kernel.applies(x, x, gate, 0.5) and not kernel.applies(x, x, gate, 0.25)
    torch.testing.assert_close(kernel.fn(x, x, gate, 0.5), x + x * gate)
