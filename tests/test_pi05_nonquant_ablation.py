"""The cumulative matrix must isolate each switch and preserve its controls."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/pi05-benchmark"))
spec = importlib.util.spec_from_file_location(
    "ablate_nonquant", ROOT / "benchmarks/pi05-benchmark/ablate_nonquant.py"
)
ablation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ablation
spec.loader.exec_module(ablation)


def test_external_profile_range_stops_after_request_failure():
    calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            profiler=SimpleNamespace(start=lambda: calls.append("start"), stop=lambda: calls.append("stop"))
        )
    )
    with pytest.raises(RuntimeError, match="request failed"), ablation.cuda_profile_range(torch):
        raise RuntimeError("request failed")
    assert calls == ["start", "stop"]


def test_trace_gate_checks_both_grippers_without_duplicate_metadata():
    from validate_nonquant import Snapshot

    reference = {"actions/physical": Snapshot.from_numpy(np.array([[[0.2, 0.7]]], dtype=np.float32))}
    result = ablation.compare_stage_trace(reference, reference, [0, 1], 0.5, "gt")
    assert result["passed"]
    assert [row["index"] for row in result["grippers"]] == [0, 1]
    changed = {"actions/physical": Snapshot.from_numpy(np.array([[[0.8, 0.7]]], dtype=np.float32))}
    result = ablation.compare_stage_trace(reference, changed, [0, 1], 0.5, "gt")
    assert not result["passed"]
    assert result["grippers"][0]["flips"] == 1


def test_progressive_edges_change_one_switch_and_have_isolated_controls():
    stages = {stage.name: stage for stage in ablation.STAGES}
    assert len(stages) == len(ablation.STAGES)
    assert ablation.stage_fields(stages["baseline"]) is None
    assert ablation.stage_fields(stages["request_only"]) is None
    assert stages["request_only"].request_graph
    for stage in ablation.STAGES:
        if stage.predecessor is None:
            continue
        before = stages[stage.predecessor]
        difference = set(stage.switches) ^ set(before.switches)
        assert len(difference) + int(stage.request_graph != before.request_graph) == 1
    for name in ("kv", "kv_prefix", "kv_prefix_context", "kv_prefix_context_request"):
        assert not ablation.stage_fields(stages[name])["norm_fusion"]
    for name, switch in (
        ("kv_only", "kv_workspace"),
        ("prefix_only", "prefix_kv_only"),
        ("context_only", "reuse_action_context"),
        ("gelu_only", "prefix_gelu_fusion"),
    ):
        fields = ablation.stage_fields(stages[name])
        assert fields[switch]
        assert sum(fields.values()) == 1
        assert not stages[name].request_graph
