"""CPU tests for end-to-end hotspot attribution and the profiling session."""

from __future__ import annotations

import sys
import types

import pytest
from scripts.kernel_tuning import hotspots
from scripts.kernel_tuning.contracts import ContractError
from scripts.kernel_tuning.hotspots import (
    DeviceKernel,
    Launch,
    Range,
    attribute,
    fold_layers,
    render,
    summarize,
)


def test_fold_layers_merges_repeated_layer_indices():
    assert fold_layers("module:model.layers.12.mlp.gate_proj") == "module:model.layers.*.mlp.gate_proj"
    assert fold_layers("module:blocks.3") == "module:blocks.*"
    assert fold_layers("op:rotate_qk") == "op:rotate_qk"


def test_kernels_follow_their_launch_into_enclosing_ranges():
    ranges = [
        Range("kt/step:0", thread=1, start=0, end=100),
        Range("kt/stage:encode_prefix", thread=1, start=5, end=50),
        Range("kt/module:layers.0.mlp", thread=1, start=10, end=20),
        Range("kt/module:layers.1", thread=1, start=28, end=42),
        Range("kt/module:layers.1.mlp", thread=1, start=30, end=40),
        Range("kt/stage:integrate", thread=1, start=60, end=95),
        Range("kt/step:1", thread=1, start=200, end=300),
        Range("kt/stage:other-thread", thread=2, start=0, end=1000),
    ]
    launches = [
        Launch("cudaLaunchKernel", thread=1, start=12, correlation=1),
        Launch("cudaLaunchKernel", thread=1, start=35, correlation=2),
        Launch("cudaLaunchKernel", thread=1, start=45, correlation=3),
        Launch("cudaGraphLaunch", thread=1, start=70, correlation=4),
        Launch("cudaLaunchKernel", thread=1, start=250, correlation=5),
        Launch("cudaLaunchKernel", thread=1, start=150, correlation=6),
    ]
    kernels = [
        DeviceKernel("gemm", 0, 3000, 1),
        DeviceKernel("gemm", 0, 3000, 2),
        DeviceKernel("norm", 0, 1000, 3),
        DeviceKernel("graph_a", 0, 2000, 4),
        DeviceKernel("graph_b", 0, 2000, 4),
        DeviceKernel("late", 0, 500, 5),
        DeviceKernel("between_steps", 0, 500, 6),
        DeviceKernel("orphan", 0, 500, 99),
    ]
    result = attribute(ranges, launches, kernels)
    assert [(k.step, k.path, k.graph) for k in result] == [
        (0, ("stage:encode_prefix", "module:layers.*.mlp"), False),
        (0, ("stage:encode_prefix", "module:layers.*.mlp"), False),
        (0, ("stage:encode_prefix",), False),
        (0, ("stage:integrate",), True),
        (0, ("stage:integrate",), True),
        (1, (), False),
        (None, (), False),
        (None, ("<unlinked>",), False),
    ]
    assert result[0].duration_us == 3.0


def test_region_keeps_stages_and_innermost_range_only():
    ranges = [
        Range("kt/step:0", 1, 0, 100),
        Range("kt/stage:integrate", 1, 1, 99),
        Range("kt/stage:integrate", 1, 2, 98),
        Range("kt/attention:Split", 1, 3, 90),
        Range("kt/attention:Split", 1, 4, 80),
        Range("kt/module:layers.4.self_attn", 1, 5, 70),
    ]
    kernel = attribute(ranges, [Launch("cudaLaunchKernel", 1, 10, 1)], [DeviceKernel("k", 0, 1000, 1)])[0]
    assert kernel.path == ("stage:integrate", "module:layers.*.self_attn")


def test_summary_ranks_regions_per_step():
    ranges = [Range(f"kt/step:{i}", 1, i * 100, i * 100 + 99) for i in range(2)] + [
        Range(f"kt/op:{name}", 1, i * 100 + offset, i * 100 + offset + 10)
        for i in range(2)
        for name, offset in (("gated_gelu", 10), ("rotate_qk", 50))
    ]
    launches, kernels = [], []
    for i in range(2):
        for correlation, (time, name, ns) in enumerate(
            ((12, "gelu", 2000), (14, "gelu_tail", 1000), (52, "rope", 30000)), start=i * 10
        ):
            launches.append(Launch("cudaLaunchKernel", 1, i * 100 + time, correlation))
            kernels.append(DeviceKernel(name, 0, ns, correlation))
    summary = summarize(attribute(ranges, launches, kernels), steps=2)
    assert summary["kernel_us_per_step"] == pytest.approx(33.0)
    first, second = summary["regions"]
    assert first["region"] == "op:rotate_qk" and first["us_per_step"] == pytest.approx(30.0)
    assert second["region"] == "op:gated_gelu"
    assert second["kernels_per_step"] == 2 and second["small_kernel_fraction"] == 1.0
    assert second["top_kernels"][0] == {"name": "gelu", "calls_per_step": 1.0, "us_per_step": 2.0}
    assert "`op:rotate_qk`" in render(summary)


def test_summary_without_measured_kernels_is_an_error():
    with pytest.raises(ContractError, match="No kernels"):
        summarize([], steps=3)


@pytest.fixture
def fake_engine(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("embodiinfer")
    module = types.ModuleType("kt_fake_engine")

    class Engine:
        def __init__(self) -> None:
            from embodiinfer.layers import KernelSelection

            self.op = KernelSelection("none")

        def execute(self, nested: bool = False) -> None:
            residual = torch.ones(1, 2, 4)
            self.op.resolve("gated_residual")(residual, residual, residual)
            if not nested:
                self.execute(nested=True)

    module.Engine = Engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(hotspots, "_ENTRIES", ((module.__name__, "Engine", ("execute",)),))
    return Engine


def test_session_profiles_the_window_and_restores_patches(fake_engine):
    original = fake_engine.execute
    session = hotspots.ProfileSession(warmup=1, steps=2, modules=False)
    session.install()
    try:
        engine = fake_engine()
        for _ in range(4):
            engine.execute()
    finally:
        session.close()
    assert session.started == 4  # nested calls are not steps
    assert fake_engine.execute is original
    names = [event.name() for event in session.trace.profiler.kineto_results.events()]
    assert names.count("kt/step:0") == 1 and names.count("kt/step:1") == 1
    assert "kt/step:2" not in names
    assert names.count("kt/op:gated_residual") >= 2
    assert session.kernels() == []  # CPU-only run: no device kernels


def test_session_requires_enough_steps(fake_engine):
    session = hotspots.ProfileSession(warmup=2, steps=3, modules=False)
    session.install()
    try:
        fake_engine().execute()
    finally:
        session.close()
    with pytest.raises(ContractError, match="Only 1 inference steps ran"):
        session.kernels()
