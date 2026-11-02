"""Diagnostic collection does not require a GPU to validate its handoff contract."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from benchmarks.kernel_tuning.profiling import collect, parse_metrics, profile_once
from scripts.kernel_tuning.contracts import ContractError, TaskPackage

CSV = """==PROF== Connected
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","fused_kernel","launch__registers_per_thread","register/thread","64"
"0","fused_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","25.0"
"1","tail","gpu__time_duration.sum","nsecond","1,234.5"
"1","tail","unsupported","","N/A"
"""


def test_metrics_keep_launch_units_and_unavailable_values():
    rows = parse_metrics(CSV)
    assert rows[0] == {
        "launch": "0",
        "kernel": "fused_kernel",
        "metric": "launch__registers_per_thread",
        "unit": "register/thread",
        "value": 64.0,
    }
    assert rows[2]["value"] == 1234.5
    assert rows[3]["value"] == "N/A"
    with pytest.raises(ContractError, match="no raw metric"):
        parse_metrics("ERR_NVGPUCTRPERM: profiling permission denied")


def test_raw_wide_csv_keeps_the_unit_row_and_each_launch():
    source = """"ID","Process ID","Kernel Name","gpu__time_duration.sum","launch__registers_per_thread"
"","","","nsecond","register/thread"
"0","12","kernel_a","1234.5","64"
"1","12","kernel_b","456.0","N/A"
"""
    rows = parse_metrics(source)
    assert len(rows) == 4
    assert rows[0] == {
        "launch": "0",
        "kernel": "kernel_a",
        "metric": "gpu__time_duration.sum",
        "unit": "nsecond",
        "value": 1234.5,
    }
    assert rows[-1]["launch"] == "1" and rows[-1]["value"] == "N/A"


def test_collector_profiles_every_role_and_workload(task_dir, candidate, tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.shutil.which", lambda _: "ncu")

    def run(argv, **kwargs):
        commands.append(argv)
        if "--worker" in argv:
            Path(argv[argv.index("--export") + 1]).write_bytes(b"fake profiler report")
        return types.SimpleNamespace(returncode=0, stdout=CSV if "--csv" in argv else "SASS", stderr="")

    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.subprocess.run", run)
    task = TaskPackage.load(task_dir)
    roles = dict.fromkeys(("baseline", "incumbent", "candidate"), candidate)
    result = collect(task, roles, tmp_path / "profiles")
    assert len(result["records"]) == 3 * len(task.workload_ids)
    assert {(r["role"], r["workload"]) for r in result["records"]} == {
        (role, workload) for role in roles for workload in task.workload_ids
    }
    workers = [argv for argv in commands if "--worker" in argv]
    assert all("--profile-from-start" in argv and "off" in argv for argv in workers)
    assert all("--force-overwrite" not in argv for argv in workers)
    assert result["timing_evidence"] is False


def test_profile_worker_uses_replay_adapter_and_fresh_inputs(task_dir, candidate, monkeypatch):
    events = []
    task = TaskPackage.load(task_dir)
    workload = types.SimpleNamespace(uuid="second-layout")
    rng = types.SimpleNamespace(reset=lambda: events.append("reset-rng"))
    adapter = types.SimpleNamespace(
        definition={},
        traces=[None, types.SimpleNamespace(workload=workload)],
        device="cuda:0",
        _rng=rng,
        _seed=lambda seed: events.append(("seed", seed)),
        _inputs=lambda work, seed: [object()],
        _call=lambda runnable, inputs: events.append(("call", inputs[0])),
        environment=lambda: events.append("flags-checked"),
    )
    registry = types.SimpleNamespace(get_instance=lambda: types.SimpleNamespace(build=lambda *args: object()))
    monkeypatch.setitem(
        sys.modules, "flashinfer_bench.compile", types.SimpleNamespace(BuilderRegistry=registry)
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer_bench.data",
        types.SimpleNamespace(Solution=types.SimpleNamespace(model_validate=lambda x: x)),
    )
    monkeypatch.setattr("benchmarks.kernel_tuning.evaluate.load_adapter", lambda task: adapter)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: events.append("start"))
    monkeypatch.setattr(torch.cuda.profiler, "stop", lambda: events.append("stop"))
    profile_once(task, candidate, 1)
    calls = [event for event in events if isinstance(event, tuple) and event[0] == "call"]
    assert len(calls) == 2 and calls[0][1] is not calls[1][1]
    assert events.count("reset-rng") == 2
    assert events.index("start") > events.index(calls[0])
    assert events.index("stop") > events.index(calls[1])
    assert events[-1] == "flags-checked"


def test_missing_profiler_fails_explicitly(task_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.shutil.which", lambda _: None)
    with pytest.raises(ContractError, match="requires ncu"):
        collect(TaskPackage.load(task_dir), {}, tmp_path)


def test_zero_exit_without_report_is_not_accepted(task_dir, candidate, tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.shutil.which", lambda _: "ncu")
    monkeypatch.setattr(
        "benchmarks.kernel_tuning.profiling.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="permission warning", stderr=""),
    )
    with pytest.raises(ContractError, match="no regular report"):
        collect(TaskPackage.load(task_dir), {"candidate": candidate}, tmp_path / "profiles")


@pytest.mark.gpu
def test_real_ncu_replay_profiles_all_captured_layouts(tmp_path):
    """Small real-GPU diagnostic smoke test; no checkpoint or speedup assertion."""
    import shutil

    pytest.importorskip("flashinfer_bench", reason="requires the isolated evaluator dependencies")
    if shutil.which("ncu") is None:
        pytest.skip("requires Nsight Compute on PATH")
    from benchmarks.kernel_tuning.calibration import calibrate
    from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter
    from scripts.kernel_tuning.discovery.contracts import ModelCapture
    from scripts.kernel_tuning.discovery.tasks import render_model
    from scripts.kernel_tuning.regions import OpCapture

    from embodiinfer.backend.torch import ops

    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        capture = OpCapture("mock_flow_vla", tmp_path / "capture", ops={"gated_residual"}, warmup=0, steps=1)
        capture.target = {"model": "mock_flow_vla", "type": "test-only", "config": {}}
        capture.started = 1
        for transposed in (False, True):
            values = [
                torch.arange(128, device="cuda", dtype=torch.float32).reshape(1, 2, 64) / 128
                for _ in range(3)
            ]
            if transposed:
                values = [value.transpose(1, 2) for value in values]
            capture.record("gated_residual", ops.gated_residual, ops.gated_residual, tuple(values), {})
        capture.close(complete=True)
        calibrated = calibrate(ModelCapture.load(capture.root), tmp_path / "calibrated")
        tasks = render_model(
            calibrated,
            tmp_path / "tasks",
            overrides={
                "profile": True,
                "timing.mode": "eager",
                "timing.warmup": 1,
                "timing.iterations": 2,
                "timing.trials": 2,
                "timing.paired_rounds": 2,
            },
        )
        assert len(tasks) == 1
        task = tasks[0]
        solutions = {role: task.baseline() for role in ("baseline", "incumbent", "candidate")}
        measurement = ReplayAdapter(task).evaluate(solutions)
        assert all(check["passed"] for check in measurement["checks"])
        result = collect(task, solutions, tmp_path / "profiles")
        assert len(result["records"]) == 3 * len(task.workload_ids) == 6
        assert all(row["metrics"] for row in result["records"])
    finally:
        torch.set_float32_matmul_precision(old_precision)
