"""Shared test configuration.

Auto-skip tests marked ``gpu`` or ``pi05`` when no CUDA device is present, so the
CPU/CI run collects the whole suite and skips (rather than errors on) GPU-only
tests. GPU tests still run when a CUDA device is available; ``pi05`` additionally
needs ``VVLA_PI05_CKPT`` (guarded in the test itself).
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def activevln_benchmark_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    """Load sibling harness imports without leaking names into other benchmark tests."""
    pytest.importorskip("transformers")
    folder = Path(__file__).parents[1] / "benchmarks/activevln-benchmark"
    modules = {}
    for name in ("benchmark", "benchmark_vllm", "benchmark_vllm_modern", "benchmark_vllm_batch"):
        spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_no_cuda = pytest.mark.skip(reason="requires CUDA (run on a GPU host)")
    for item in items:
        if "gpu" in item.keywords or "pi05" in item.keywords:
            item.add_marker(skip_no_cuda)
