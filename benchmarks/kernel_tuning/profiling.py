"""Budgeted NCU collection through the same frozen adapter as evaluation."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernel_tuning.artifacts import atomic_json  # noqa: E402
from scripts.kernel_tuning.contracts import ContractError, TaskPackage, read_json  # noqa: E402

SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
    "SourceCounters",
    "PmSampling",
    "PmSampling_WarpStates",
)


def choose_sections(listing: str) -> list[str]:
    """Select only installed section identifiers; retain missing sections as evidence."""
    available = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", listing))
    sections = [name for name in SECTIONS if name in available]
    if not sections:
        # Some Jetson collectors return success with only a privilege warning.
        # Keep that diagnostic in agent feedback instead of claiming absent sections.
        detail = listing.strip()[:2000] or "<empty output>"
        raise ContractError(f"NCU section discovery failed; collector output: {detail}")
    return sections


def collect(task: TaskPackage, solution: dict[str, Any], output: Path) -> dict[str, Any]:
    """Collect and analyze all launches; the parent evaluator enforces the time limit.

    NCU and its native reader may live in different installations/interpreters.
    Both stay descendants of the evaluator for process-tree cancellation.
    """
    cfg = task.settings.diagnostics
    ncu = shutil.which(cfg.ncu)
    if not ncu:
        raise ContractError(f"NCU executable unavailable: {cfg.ncu}")
    workload = cfg.workload or task.workload_ids[0]
    if workload not in task.workload_ids:
        raise ContractError(f"Diagnostic workload not in frozen task: {workload}")
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="ncu-", dir=output)).resolve()
    info = {}
    for option in ("version", "list-sections", "help"):
        result = subprocess.run(
            [ncu, "--" + option], capture_output=True, text=True, errors="replace", check=True
        )
        info[option] = result.stdout + result.stderr
        (directory / f"{option}.txt").write_text(info[option], encoding="utf-8")
    sections = choose_sections(info["list-sections"])
    candidate = directory / "solution.json"
    atomic_json(candidate, solution)
    report = directory / "report.ncu-rep"
    command = [ncu]
    if "config-file" in info["help"]:
        command += ["--config-file", "off"]
    command += [
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--clock-control",
        "none",
        "--cache-control",
        "all",
        "--replay-mode",
        "kernel",
    ]
    if task.settings.timing.mode == "cuda_graph":
        if "graph-profiling" not in info["help"]:
            raise ContractError("Installed NCU does not support CUDA graph profiling")
        command += ["--graph-profiling", "node"]
    for section in sections:
        command += ["--section", section]
    command += [
        "--export",
        str(report),
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--task",
        str(task.root),
        "--solution",
        str(candidate),
        "--workload",
        workload,
    ]
    metadata = {
        "command": command,
        "workload": workload,
        "seed": task.settings.seeds[0],
        "timing_mode": task.settings.timing.mode,
        "sections": sections,
        "unavailable_sections": [s for s in SECTIONS if s not in sections],
        "report": str(report),
        "archive": str(directory),
        "scope": "One workload and seed; all launches; diagnostic replay, not promotion latency",
    }
    atomic_json(directory / "collection.json", metadata)
    with (directory / "ncu.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not report.is_file() or not report.stat().st_size:
        raise ContractError(f"NCU returned without a nonempty report; see {directory / 'ncu.log'}")
    analysis = directory / "analysis"
    reader = cfg.reader_python or sys.executable
    analysis_command = [
        reader,
        "-B",
        "-X",
        "utf8",
        str(Path(__file__).with_name("ncu_analysis.py")),
        "--report",
        str(report),
        "--output",
        str(analysis),
        "--profile",
        cfg.metric_profile,
    ]
    metadata["analysis_command"] = analysis_command
    atomic_json(directory / "collection.json", metadata)
    with (directory / "reader.log").open("w", encoding="utf-8") as log:
        subprocess.run(analysis_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {**metadata, **read_json(analysis / "summary.json")}


def profile_task(task_root: Path, solution_path: Path, workload: str) -> None:
    """Load the trusted task adapter, never substitute the default replay semantics."""
    from benchmarks.kernel_tuning.evaluate import load_adapter

    task = TaskPackage.load(task_root)
    if workload not in task.workload_ids:
        raise ContractError("Unknown diagnostic workload")
    adapter = load_adapter(task)
    method = getattr(adapter, "profile_once", None)
    if method is None:
        raise ContractError("Task adapter must implement profile_once(solution, workload_id)")
    method(read_json(solution_path), workload)
    task.verify()


def main() -> None:
    """Internal worker launched by NCU with collection initially disabled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--solution", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    args = parser.parse_args()
    profile_task(args.task, args.solution, args.workload)


if __name__ == "__main__":
    main()
