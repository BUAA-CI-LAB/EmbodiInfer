from __future__ import annotations

import pytest
import torch

from embodiinfer import make_policy
from embodiinfer.models.qwen25_vl.compile import (
    QWEN25_VL_COMPILE_ABI,
    Qwen25VLCompileRuntime,
    normalize_qwen25_vl_compile_backend,
)


class _Double(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2


def test_normalize_qwen25_vl_compile_backend() -> None:
    assert normalize_qwen25_vl_compile_backend("none") == "none"
    assert normalize_qwen25_vl_compile_backend("inductor") == "inductor"
    with pytest.raises(ValueError, match="compile_backend"):
        normalize_qwen25_vl_compile_backend("silent-fallback")


def test_compile_runtime_caches_exact_execution_key(monkeypatch) -> None:
    compile_calls: list[dict[str, object]] = []

    def fake_compile(module, **kwargs):
        compile_calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    runtime = Qwen25VLCompileRuntime("inductor", max_entries=2, target="unit_test_next_token")
    module = _Double()
    inputs = (torch.tensor([3.0]),)
    key = ("batch-1", "shape-1")

    first = runtime.get_callable(
        module,
        key,
        inputs,
        resolved_attention_backend="torch_sdpa",
    )
    second = runtime.get_callable(
        module,
        key,
        inputs,
        resolved_attention_backend="torch_sdpa",
    )

    assert first is second
    torch.testing.assert_close(first(*inputs), torch.tensor([6.0]))
    assert compile_calls == [
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "triton.cudagraphs": False,
                "emulate_precision_casts": True,
            },
        }
    ]
    stats = runtime.stats()
    assert stats["requested"] == "inductor"
    assert stats["resolved"] == "inductor"
    assert stats["fallback_reason"] is None
    assert QWEN25_VL_COMPILE_ABI == "qwen25_vl_next_token_compile_v4"
    assert stats["compile_abi"] == "qwen25_vl_next_token_compile_v4"
    assert stats["emulate_precision_casts"] is True
    assert stats["inductor_options"] == {
        "triton.cudagraphs": False,
        "emulate_precision_casts": True,
    }
    assert stats["target"] == "unit_test_next_token"
    assert stats["cache_entries"] == 1
    assert stats["attempts"] == 1
    assert stats["failures"] == 0
    assert stats["entries"][0]["warmup_calls"] == 3
    assert stats["entries"][0]["first_call_wall_ms"] >= 0.0


def test_compile_runtime_counts_compile_failure_without_fallback(monkeypatch) -> None:
    def fail_compile(*args, **kwargs):
        raise RuntimeError("inductor failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    runtime = Qwen25VLCompileRuntime("inductor")
    inputs = (torch.tensor([1.0]),)

    with pytest.raises(RuntimeError, match="inductor failed"):
        runtime.get_callable(
            _Double(),
            ("failure",),
            inputs,
            resolved_attention_backend="torch_sdpa",
        )

    stats = runtime.stats()
    assert stats["fallback_reason"] is None
    assert stats["attempts"] == 1
    assert stats["failures"] == 1
    assert stats["cache_entries"] == 0


def test_compile_runtime_rejects_non_torch_attention_without_compile(monkeypatch) -> None:
    compile_called = False

    def fake_compile(*args, **kwargs):
        nonlocal compile_called
        compile_called = True
        return args[0]

    monkeypatch.setattr(torch, "compile", fake_compile)
    runtime = Qwen25VLCompileRuntime("inductor")
    with pytest.raises(RuntimeError, match="only resolved torch_sdpa"):
        runtime.get_callable(
            _Double(),
            ("triton",),
            (torch.tensor([1.0]),),
            resolved_attention_backend="triton_hybrid",
        )

    assert compile_called is False
    stats = runtime.stats()
    assert stats["fallback_reason"] is None
    assert stats["attempts"] == 0
    assert stats["failures"] == 1
    assert stats["cache_entries"] == 0


def test_navida_builder_rejects_inductor_before_runner_construction(
    monkeypatch,
) -> None:
    def runner_must_not_be_constructed(*args, **kwargs):
        raise AssertionError("NaViDA runner/checkpoint construction was reached")

    monkeypatch.setattr(
        "embodiinfer.policies.navida.runner.NaViDARunner.__init__",
        runner_must_not_be_constructed,
    )
    with pytest.raises(
        ValueError,
        match=r"(?i)(navida.*(compile|inductor)|(compile|inductor).*navida)",
    ):
        make_policy(
            "navida",
            checkpoint="must-not-be-loaded",
            compile_backend="inductor",
        )
