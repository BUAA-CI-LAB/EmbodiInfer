"""Evaluator-owned Nsight Compute collection, separate from promotion timing."""

from __future__ import annotations

import csv
import io
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernel_tuning.artifacts import atomic_json
from scripts.kernel_tuning.contracts import ContractError, TaskPackage, read_json

_FEEDBACK_METRICS = {
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__block_size",
    "launch__grid_size",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
}


def parse_metrics(source: str) -> list[dict[str, Any]]:
    """Read NCU long or wide CSV, preserving launch identity, units and missing values.

    NCU diagnostics may precede the CSV header. Numeric values use base units
    (``--csv --print-units base``); unsupported metrics remain text, never zero.
    """
    lines = source.splitlines()

    def header(line: str) -> bool:
        fields = set(next(csv.reader([line])))
        return {"Metric Name", "Metric Value"} <= fields or (
            {"ID", "Kernel Name"} <= fields and any("__" in name for name in fields)
        )

    start = next((i for i, line in enumerate(lines) if header(line)), None)
    if start is None:
        raise ContractError("NCU produced no raw metric CSV header")
    result = []

    def append(launch: str | None, kernel: str | None, name: str, unit: str, raw: Any) -> None:
        try:
            value = float(raw.replace(",", ""))
            value = value if math.isfinite(value) else raw
        except (ValueError, AttributeError):
            value = raw
        result.append({"launch": launch, "kernel": kernel, "metric": name, "unit": unit, "value": value})

    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    if "Metric Name" in (rows.fieldnames or []):
        for row in rows:
            name = row.get("Metric Name")
            if name and name != "Metric Name":
                append(
                    row.get("ID"),
                    row.get("Kernel Name"),
                    name,
                    row.get("Metric Unit", ""),
                    row.get("Metric Value", ""),
                )
    else:
        # --page raw uses one column per metric and an optional unit row.
        metrics = [name for name in rows.fieldnames or [] if "__" in name]
        units: dict[str, str] = {}
        for row in rows:
            if not row.get("ID"):
                if not units:
                    units = {name: row.get(name, "") for name in metrics}
                continue
            if row["ID"] == "ID":
                continue
            for name in metrics:
                append(row["ID"], row.get("Kernel Name"), name, units.get(name, ""), row.get(name, ""))
    if not result:
        raise ContractError("NCU produced no metric rows")
    return result


def collect(task: TaskPackage, solutions: dict[str, dict[str, Any]], output: Path) -> dict[str, Any]:
    """Profile all workloads and roles using the task's actual evaluation adapter.

    Each worker builds and warms its kernel before enabling profiling. Raw CSV,
    SASS and native reports are retained even though they never affect promotion.
    Replay input materialization, mutations and RNG remain adapter-owned.
    """
    ncu = shutil.which("ncu")
    if not ncu:
        raise ContractError("profile: true requires ncu on PATH")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for role, solution in solutions.items():
        solution_path = output / f"{role}.json"
        atomic_json(solution_path, solution)
        for index, workload in enumerate(task.workload_ids):
            folder = output / role / str(index)
            folder.mkdir(parents=True, exist_ok=True)
            report = folder / "ncu.ncu-rep"
            command = [
                ncu,
                "--target-processes",
                "all",
                "--set",
                "full",
                "--profile-from-start",
                "off",
                "--clock-control",
                "none",
                "--export",
                str(report),
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(task.root),
                str(solution_path),
                str(index),
            ]
            atomic_json(folder / "command.json", {"argv": command})
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=task.settings.search.evaluation_timeout_seconds,
                )
            if not report.is_file() or report.is_symlink():
                raise ContractError(f"NCU produced no regular report; see {folder / 'worker.log'}")
            metrics_command = [
                ncu,
                "--import",
                str(report),
                "--page",
                "raw",
                "--csv",
                "--print-units",
                "base",
            ]
            csv_result = subprocess.run(
                metrics_command,
                capture_output=True,
                text=True,
                check=False,
                timeout=task.settings.search.evaluation_timeout_seconds,
            )
            (folder / "metrics.csv").write_text(csv_result.stdout, encoding="utf-8")
            (folder / "metrics.stderr.log").write_text(csv_result.stderr, encoding="utf-8")
            if csv_result.returncode:
                raise ContractError(
                    f"NCU report import failed; see {folder / 'metrics.csv'} and metrics.stderr.log"
                )
            metrics = parse_metrics(csv_result.stdout)
            # Assembly is diagnostic evidence. Missing source correlation must
            # not discard successfully collected counters.
            assembly_command = [ncu, "--import", str(report), "--page", "source", "--print-source", "sass"]
            try:
                assembly = subprocess.run(
                    assembly_command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=task.settings.search.evaluation_timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                assembly = subprocess.CompletedProcess(
                    assembly_command, -1, "", f"{type(exc).__name__}: {exc}"
                )
            (folder / "assembly.txt").write_text(assembly.stdout + assembly.stderr, encoding="utf-8")
            useful = [metric for metric in metrics if metric["metric"] in _FEEDBACK_METRICS]
            records.append(
                {
                    "role": role,
                    "workload": workload,
                    "metrics": useful[:128],
                    "metric_rows": len(metrics),
                    "feedback_truncated": len(useful) > 128,
                    "report": report.relative_to(output).as_posix(),
                    "assembly_available": assembly.returncode == 0
                    and bool(re.search(r"/\*[0-9a-fA-F]{4,}\*/", assembly.stdout)),
                    "assembly_excerpt": assembly.stdout[:6000] if role == "candidate" else "",
                }
            )
    return {
        "collector": "nsight-compute",
        "timing_evidence": False,
        "execution": "one eager call after warmup; not CUDA Graph production timing",
        "records": records,
    }


def profile_once(task: TaskPackage, solution: dict[str, Any], index: int) -> None:
    """Execute one fresh input through the frozen adapter inside a profiler range."""
    import torch
    from benchmarks.kernel_tuning.evaluate import load_adapter
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.data import Solution

    adapter = load_adapter(task)
    runnable = BuilderRegistry.get_instance().build(adapter.definition, Solution.model_validate(solution))
    workload = adapter.traces[index].workload
    seed = task.settings.seeds[0]
    # Re-materialize rather than reuse mutated warmup tensors.
    adapter._seed(seed)
    warm_inputs = adapter._inputs(workload, seed)
    if getattr(adapter, "_rng", None):
        adapter._rng.reset()
    adapter._call(runnable, warm_inputs)
    adapter._seed(seed)
    inputs = adapter._inputs(workload, seed)
    if getattr(adapter, "_rng", None):
        adapter._rng.reset()
    torch.cuda.synchronize(adapter.device)
    torch.cuda.profiler.start()
    try:
        adapter._call(runnable, inputs)
        torch.cuda.synchronize(adapter.device)
    finally:
        torch.cuda.profiler.stop()
    adapter.environment()  # Detect precision/stream changes as in measurement.


if __name__ == "__main__":
    # Script execution must work from an evaluator archive, not just repo cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("Internal NCU worker; use tuning.yaml profile: true")
    profile_once(TaskPackage.load(Path(sys.argv[2])), read_json(Path(sys.argv[3])), int(sys.argv[4]))
