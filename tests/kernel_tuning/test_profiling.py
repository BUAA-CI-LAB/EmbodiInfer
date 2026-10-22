"""CPU coverage of the skill bridge, native-reader seam, and replay profiling scope."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.kernel_tuning.ncu_analysis import analyze
from benchmarks.kernel_tuning.profiling import choose_sections, collect, profile_task
from scripts.kernel_tuning.contracts import ContractError, Settings, TaskPackage
from scripts.kernel_tuning.skill_context import ROOT, skill_instructions


@pytest.mark.parametrize(
    "values",
    [
        {"max_reports": 0},
        {"max_reports": True},
        {"timeout_seconds": -1},
        {"metric_profile": "sm110"},
        {"ncu": ""},
        {"required": "yes"},
    ],
)
def test_invalid_diagnostic_settings(values: dict) -> None:
    with pytest.raises(ContractError):
        Settings.load_mapping({"language": "triton", "profile": True, "diagnostics": values})


def test_required_diagnostics_need_profiling() -> None:
    with pytest.raises(ContractError, match="profile: true"):
        Settings.load_mapping({"language": "triton", "diagnostics": {"required": True}})


@pytest.mark.parametrize("architecture,device", [("sm87", "orin"), ("sm110", "thor")])
def test_wiki_query_from_attempt_directory(tmp_path: Path, architecture: str, device: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-X",
            "utf8",
            str(ROOT / "skills/KernelWiki/scripts/query.py"),
            "--architecture",
            architecture,
            "--compact",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert device in result.stdout.lower()
    instructions = skill_instructions()
    assert str(ROOT / "skills/KernelWiki/SKILL.md") in instructions
    assert str(ROOT / "skills/ncu-report-skill/SKILL.md") in instructions
    assert sys.executable in instructions


def test_section_discovery_never_assumes_full_set() -> None:
    assert choose_sections("Identifier | Name\nLaunchStats | Launch Statistics\nOccupancy") == [
        "LaunchStats",
        "Occupancy",
    ]
    with pytest.raises(ContractError, match="section discovery failed"):
        choose_sections("OldSectionWithoutSupport")


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_collector_retains_zero_exit_permission_warning(
    task_dir: Path, tmp_path: Path, monkeypatch, stream: str
) -> None:
    warning = (
        "==WARNING== Insufficient privileges to launch app for profiling. Launch app with root privileges"
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = {"stdout": "", "stderr": ""}
        output[stream] = warning if command[1] == "--list-sections" else "NCU"
        return SimpleNamespace(returncode=0, **output)

    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.shutil.which", lambda _: "/ncu")
    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.subprocess.run", run)
    output = tmp_path / "profile"
    with pytest.raises(ContractError, match="Insufficient privileges"):
        collect(TaskPackage.load(task_dir), {}, output)
    assert next(output.glob("ncu-*/list-sections.txt")).read_text() == warning
    assert all("--export" not in command for command in calls)


def test_profile_worker_uses_frozen_custom_adapter(task_dir: Path, tmp_path: Path) -> None:
    marker = tmp_path / "worker.json"
    (task_dir / "benchmark.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "class Adapter:\n"
        "    def profile_once(self, solution, workload):\n"
        "        Path(solution['marker']).write_text(json.dumps({'workload': workload}))\n"
        "def create_adapter(task):\n    return Adapter()\n",
        encoding="utf-8",
    )
    solution = tmp_path / "solution.json"
    solution.write_text(json.dumps({"marker": str(marker)}), encoding="utf-8")
    profile_task(task_dir, solution, "broadcast-small")
    assert json.loads(marker.read_text())["workload"] == "broadcast-small"


@pytest.mark.parametrize("mode", ["eager", "cuda_graph"])
def test_profile_scope_preserves_replay_aliases_and_resets_mutations(mode: str, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter

    adapter = object.__new__(ReplayAdapter)
    adapter.cfg = Settings.load_mapping({"language": "triton", "timing": {"mode": mode, "warmup": 2}})
    adapter.device = "cpu"
    buffer = torch.arange(8, dtype=torch.float32)
    inputs = [buffer[::2], buffer[2::2]]
    events = []
    active = False

    def start():
        nonlocal active
        active = True
        events.append("start")

    def stop():
        nonlocal active
        active = False
        events.append("stop")

    def runnable(left, right):
        assert left.stride() == (2,)
        assert right.storage_offset() == 2
        assert left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        assert torch.equal(left, inputs[0])
        assert torch.equal(right, inputs[1])
        if active:
            events.extend(["kernel-1", "kernel-2"])
        left.add_(10)
        return right

    def capture(fn, local):
        runnable(*local)  # Capture mutates storage; worker must restore before replay.
        return SimpleNamespace(replay=lambda: fn(*local)), local, [local[1]]

    monkeypatch.setattr(adapter, "_inputs", lambda workload, seed: inputs)
    monkeypatch.setattr(adapter, "_seed", lambda seed: events.append("seed"))
    monkeypatch.setattr(adapter, "_call", lambda fn, local: (local, [fn(*local)]))
    monkeypatch.setattr(adapter, "_capture", capture)
    monkeypatch.setattr(adapter, "_check_flags", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda.profiler, "start", start)
    monkeypatch.setattr(torch.cuda.profiler, "stop", stop)
    adapter._profile_runnable(runnable, object())
    assert events[events.index("start") :] == ["start", "kernel-1", "kernel-2", "sync", "stop"]
    assert torch.equal(buffer, torch.arange(8, dtype=torch.float32))


def test_all_actions_keep_units_availability_and_source_gaps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "skills/ncu-report-skill/scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "skills/ncu-report-skill/tests"))
    from fake_ncu import capture_of, scalar_metric

    capture, module = capture_of(
        [
            [
                {
                    "name": "first",
                    "metrics": {
                        "gpu__time_duration.sum": scalar_metric(0, "nsecond"),
                        "launch__grid_size": scalar_metric(0, "block", has_value=False),
                    },
                }
            ],
            [{"name": "second", "metrics": {"gpu__time_duration.sum": scalar_metric(12, "nsecond")}}],
        ]
    )
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture), encoding="utf-8")
    summary = analyze(tmp_path / "analysis", capture_path=path)
    assert summary["action_count"] == 2
    assert [a["action"] for a in summary["actions"]] == ["r0a0", "r1a0"]
    first = summary["actions"][0]
    assert first["metrics"]["gpu__time_duration.sum"] == {"value": 0, "unit": "nsecond"}
    assert first["metrics"]["launch__grid_size"]["value"] != 0
    assert first["metrics"]["launch__block_size"]["value"] != 0
    assert first["source_coverage"] is None
    assert "unavailable" in Path(first["pm_timeline"]).read_text().lower()
    assert not module.has_value_arg_calls
    with pytest.raises(FileExistsError):
        analyze(tmp_path / "analysis", capture_path=path)


def test_partial_capture_is_not_success(tmp_path: Path) -> None:
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"schema_version": 3, "status": "partial_native_report"}))
    with pytest.raises(ValueError, match="complete"):
        analyze(tmp_path / "analysis", capture_path=capture)


def test_conflicting_device_profile_preserves_native_capture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "skills/ncu-report-skill/scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "skills/ncu-report-skill/tests"))
    from fake_ncu import capture_of, scalar_metric

    capture, _ = capture_of(
        [
            [
                {
                    "name": "orin",
                    "metrics": {
                        "device__attribute_display_name": scalar_metric("Orin"),
                        "device__attribute_compute_capability_major": scalar_metric(8),
                        "device__attribute_compute_capability_minor": scalar_metric(7),
                    },
                }
            ]
        ]
    )
    source = tmp_path / "capture.json"
    source.write_text(json.dumps(capture), encoding="utf-8")
    target = tmp_path / "analysis"
    with pytest.raises(ValueError, match="conflicts"):
        analyze(target, capture_path=source, profile="thor")
    assert (target / "native-capture.json").exists()
    assert (target / "metrics_all_kernel_r0a0.json").exists()
    assert json.loads((target / "summary.json").read_text())["profile_conflicts"] == ["r0a0"]


def test_collector_uses_discovered_sections_and_separate_reader(
    task_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    import yaml

    config = yaml.safe_load((task_dir / "tuning.yaml").read_text())
    config.update(profile=True, diagnostics={"reader_python": "/reader/python"})
    (task_dir / "tuning.yaml").write_text(yaml.safe_dump(config))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "--list-sections":
            return SimpleNamespace(stdout="LaunchStats SourceCounters", stderr="")
        if command[1] in ("--version", "--help"):
            return SimpleNamespace(stdout="NCU config-file graph-profiling", stderr="")
        if "--export" in command:
            Path(command[command.index("--export") + 1]).write_bytes(b"native report")
        else:
            target = Path(command[command.index("--output") + 1])
            target.mkdir()
            (target / "summary.json").write_text('{"action_count": 2}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.shutil.which", lambda _: "/ncu")
    monkeypatch.setattr("benchmarks.kernel_tuning.profiling.subprocess.run", run)
    result = collect(TaskPackage.load(task_dir), {}, tmp_path / "profile")
    command = result["command"]
    assert command[command.index("--profile-from-start") + 1] == "off"
    assert "--launch-count" not in command and "--force-overwrite" not in command
    assert result["sections"] == ["LaunchStats", "SourceCounters"]
    assert "PmSampling" in result["unavailable_sections"]
    assert calls[-1][0] == "/reader/python"
    assert result["action_count"] == 2
