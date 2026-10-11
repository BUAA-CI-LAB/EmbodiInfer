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
        (0, ("stage:encode_prefix", "kernels:elementwise"), False),
        (0, ("stage:integrate", "kernels:other"), True),
        (0, ("stage:integrate", "kernels:other"), True),
        (1, ("kernels:other",), False),
        (None, ("kernels:other",), False),
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


def test_graph_replays_take_their_capture_labels_in_order():
    capture_ranges = [  # the capture profiler starts inside the stage: no stage range
        Range("kt/op:ada_rms_norm", 1, 10, 20),
        Range("kt/attention:Split", 1, 30, 40),
    ]
    capture_launches = [
        Launch("cudaLaunchKernel", 1, 35, 2),
        Launch("cudaStreamIsCapturing", 1, 12, 9),
        Launch("cuLaunchKernel", 1, 15, 1),
        Launch("cudaMemsetAsync", 1, 50, 3),
    ]
    labels = hotspots.capture_labels(capture_ranges, capture_launches)
    assert labels == [("op:ada_rms_norm",), ("attention:Split",), ()]

    ranges = [
        Range("kt/step:0", 1, 0, 1000),
        Range("kt/stage:integrate", 1, 1, 999),
        Range("kt/graph:77", 1, 10, 20),
        Range("kt/graph:88", 1, 30, 40),
    ]
    launches = [Launch("cudaGraphLaunch", 1, 12, 5), Launch("cudaGraphLaunch", 1, 32, 6)]
    kernels = [
        DeviceKernel("attn", 200, 4000, 5),
        DeviceKernel("norm", 100, 1000, 5),
        DeviceKernel("memset", 300, 500, 5),
        DeviceKernel("unknown_graph", 400, 500, 6),
    ]
    result = attribute(ranges, launches, kernels, {77: labels, 88: labels})
    assert [(k.name, k.path, k.labelled) for k in result] == [
        ("attn", ("stage:integrate", "attention:Split"), True),
        ("norm", ("stage:integrate", "op:ada_rms_norm"), True),
        ("memset", ("stage:integrate", "kernels:copy"), True),
        ("unknown_graph", ("stage:integrate", "kernels:other"), False),  # 1 kernel vs 3 labels
    ]
    summary = summarize(result, steps=1)
    assert summary["graph_labelled_fraction"] == pytest.approx(5.5 / 6.0)
