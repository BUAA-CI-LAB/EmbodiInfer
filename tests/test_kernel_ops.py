"""CPU tests for the operator registry and kernel-profile selection."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import embodiinfer.backend.triton.tuned as tuned
from embodiinfer.backend.torch import ops as torch_ops
from embodiinfer.layers import KernelSelection, available_ops, get_op
from embodiinfer.policies.pi05 import modeling_pi05

_TUNED_MODULE = """
CALLS = []


def gated_residual(residual, update, gate):
    CALLS.append(residual.shape)
    return residual + update * gate


def applies(residual, update, gate):
    return residual.shape[-1] == 4
"""

_ENVIRONMENT = {"device_name": "Test GPU", "capability": [9, 9], "torch": "t", "triton": "x"}


def _write_profile(root, *, source=_TUNED_MODULE, applies="applies", digest=None):
    root.mkdir()
    (root / "gated_residual.py").write_text(source, encoding="utf-8")
    entry = {
        "module": "gated_residual.py",
        "function": "gated_residual",
        "sha256": digest or hashlib.sha256(source.encode()).hexdigest(),
    }
    if applies:
        entry["applies"] = applies
    manifest = {"schema": 1, "match": _ENVIRONMENT, "ops": {"gated_residual": entry}}
    (root / "profile.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tuned.load_profile(root)


def test_core_operators_have_torch_and_triton_implementations():
    ops = available_ops()
    for op in ("gated_residual", "ada_rms_norm", "gated_gelu", "rotate_qk"):
        assert {"torch", "triton"} <= set(ops[op])
    assert get_op("gated_gelu", "torch") is torch_ops.gated_gelu


def test_unknown_operator_or_implementation_is_explicit():
    with pytest.raises(KeyError, match="unknown operator"):
        get_op("not_an_op")
    with pytest.raises(KeyError, match="no 'cuda' implementation"):
        get_op("gated_residual", "cuda")


def test_none_keeps_preference_and_wraps_only_torch():
    selection = KernelSelection("none")
    wrapped = []

    def wrapper(fn):
        wrapped.append(fn)
        return fn

    assert selection.resolve("gated_residual", "torch", wrapper) is torch_ops.gated_residual
    assert selection.resolve("gated_residual", "triton", wrapper) is get_op("gated_residual", "triton")
    assert wrapped == [torch_ops.gated_residual]
    assert selection.resolve("gated_residual", "torch", wrapper) is torch_ops.gated_residual
    assert len(wrapped) == 1  # cached


def test_torch_mode_forces_reference_implementations():
    selection = KernelSelection("torch")
    assert selection.resolve("rotate_qk", "triton") is torch_ops.rotate_qk


def test_profile_kernel_replaces_only_applicable_calls(tmp_path):
    profile = _write_profile(tmp_path / "fake")
    selection = KernelSelection(profile)
    op = selection.resolve("gated_residual", "torch")
    module = profile.kernel("gated_residual")
    calls = module.fn.__globals__["CALLS"]
    x = torch.ones(1, 2, 4)
    torch.testing.assert_close(op(x, x, x), x + x * x)
    assert calls == [x.shape]
    y = torch.ones(1, 2, 3)
    torch.testing.assert_close(op(y, y, y), y + y * y)
    assert calls == [x.shape]  # the 3-wide call fell back
    assert selection.resolve("gated_gelu", "torch") is torch_ops.gated_gelu


def test_profile_without_applies_always_uses_tuned_kernel(tmp_path):
    profile = _write_profile(tmp_path / "fake", applies=None)
    op = KernelSelection(profile).resolve("gated_residual", "triton")
    assert op is profile.kernel("gated_residual").fn


def test_modified_tuned_kernel_is_rejected(tmp_path):
    profile = _write_profile(tmp_path / "fake", digest="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        KernelSelection(profile).resolve("gated_residual")


def test_profile_selection_by_name_and_auto(tmp_path, monkeypatch):
    profile = _write_profile(tmp_path / "fake")
    monkeypatch.setattr(tuned, "available_profiles", lambda: {"fake": profile})
    monkeypatch.setattr(tuned, "current_environment", lambda: dict(_ENVIRONMENT))
    assert KernelSelection("fake").profile is profile
    assert KernelSelection("auto").profile is profile
    with pytest.raises(ValueError, match="unknown kernel_profile"):
        KernelSelection("missing")
    monkeypatch.setattr(tuned, "current_environment", lambda: {**_ENVIRONMENT, "triton": "other"})
    assert KernelSelection("auto").profile is None
    with pytest.raises(RuntimeError, match="not this environment"):
        KernelSelection("fake")


def test_profile_manifest_requires_match_keys(tmp_path):
    root = tmp_path / "broken"
    root.mkdir()
    (root / "profile.json").write_text(json.dumps({"schema": 1, "match": {}, "ops": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="match is missing"):
        tuned.load_profile(root)


def test_committed_profiles_are_valid():
    for profile in tuned.available_profiles().values():
        for op in profile.ops:
            assert op in available_ops()


def test_pi05_helpers_use_the_registry_reference(tmp_path):
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    modulation = torch.randn(2, 12, dtype=torch.bfloat16)
    norm = SimpleNamespace(eps=1e-6, dense=object())
    out, gate = modeling_pi05._rmsnorm(norm, x, None, False, modulation)
    expected = torch_ops.ada_rms_norm(x, modulation, 1e-6)
    assert torch.equal(out, expected[0]) and torch.equal(gate, expected[1])

    profile = _write_profile(tmp_path / "fake")
    calls = profile.kernel("gated_residual").fn.__globals__["CALLS"]
    gate = torch.randn(2, 1, 4)
    result = modeling_pi05._gated_residual(
        x.float(), x.float(), gate, False, kernels=KernelSelection(profile)
    )
    torch.testing.assert_close(result, x.float() + x.float() * gate)
    assert calls == [x.shape]


def test_prepare_resolves_everything_before_graph_tracing(monkeypatch):
    import embodiinfer.layers.ops as ops_module

    selection = KernelSelection("none")
    selection.prepare((None, torch_ops_wrapper := (lambda fn: fn)))
    monkeypatch.setattr(ops_module, "get_op", lambda *args: pytest.fail("resolve imported after prepare"))
    for op in ("gated_residual", "ada_rms_norm", "gated_gelu", "rotate_qk"):
        for prefer in ("torch", "triton"):
            selection.resolve(op, prefer)
            selection.resolve(op, prefer, torch_ops_wrapper)
