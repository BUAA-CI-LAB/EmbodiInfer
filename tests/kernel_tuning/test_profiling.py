"""CPU checks of remote-style NCU collection and standalone report analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.kernel_tuning.flashinfer_adapter import FlashInferAdapter
from benchmarks.kernel_tuning.ncu_analysis import ROOT, analyze
from scripts.kernel_tuning.contracts import ContractError, Settings


@pytest.mark.parametrize("command", [[], [""], [True], "ncu"])
def test_invalid_profile_command(command) -> None:
    with pytest.raises(ContractError):
        Settings.load_mapping({"language": "triton", "profile_command": command})


def test_legacy_diagnostic_config_requires_migration() -> None:
    with pytest.raises(ContractError):
        Settings.load_mapping({"language": "triton", "diagnostics": {"max_reports": 2}})


@pytest.mark.parametrize(("architecture", "device"), [("sm87", "orin"), ("sm110", "thor")])
def test_pinned_wiki_remains_queryable(tmp_path: Path, architecture: str, device: str) -> None:
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


@pytest.mark.parametrize("wrapped", [False, True])
def test_profile_command_collects_weighted_workload_and_details(tmp_path, monkeypatch, wrapped) -> None:
    adapter = object.__new__(FlashInferAdapter)
    command = ["sudo", "-n", "/opt/ncu"] if wrapped else ["ncu"]
    adapter.cfg = Settings.load_mapping(
        {
            "language": "triton",
            "profile_command": command,
            "weights": {"small": 1.0, "large": 3.0},
        }
    )
    adapter.task = SimpleNamespace(root=tmp_path / "task")
    adapter.traces = [SimpleNamespace(workload=SimpleNamespace(uuid=x)) for x in ("small", "large")]
    assert adapter.profiled_workload().uuid == "large"
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "--export" in argv:
            Path(argv[argv.index("--export") + 1] + ".ncu-rep").write_bytes(b"report")
        else:
            kwargs["stdout"].write("counter details")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("benchmarks.kernel_tuning.flashinfer_adapter.shutil.which", lambda _: "/ncu")
    monkeypatch.setattr("benchmarks.kernel_tuning.flashinfer_adapter.subprocess.run", run)
    monkeypatch.setattr("benchmarks.kernel_tuning.flashinfer_adapter.os.getuid", lambda: 123, raising=False)
    monkeypatch.setattr(
        "benchmarks.kernel_tuning.flashinfer_adapter.tempfile.gettempdir", lambda: str(tmp_path)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-not-for-command")
    result = adapter.profile({"candidate": {"name": "candidate"}}, tmp_path)
    assert calls[0][: len(command)] == command
    assert calls[0][calls[0].index("--nvtx-include") + 1] == "candidate/"
    assert calls[0][calls[0].index("--set") + 1] == "full"
    assert "--profile-once" in calls[0]
    assert not any("test-secret-not-for-command" in arg for arg in calls[0])
    assert result["workload"] == "large"
    assert Path(result["report"]).read_bytes() == b"report"
    assert Path(result["details"]).read_text() == "counter details"


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
