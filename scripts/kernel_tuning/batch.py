"""One-command tuning of every selected catalog operator, serially on one machine.

A batch directory holds freshly generated tasks, one ordinary run archive per
operator, logs, exports of promoted kernels, and a summary. Each operator runs
in its own ``run``/``resume`` subprocess, so one failure or crash does not stop
the others, and an interrupted batch continues where it stopped.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import REPOSITORY, RunStore, atomic_json, run_lock
from .contracts import ContractError, TaskPackage, read_json, validate_measurement
from .generate import render
from .operators import Operator
from .runner import evaluate

Launcher = Callable[[list[str], Path], int]
ARCHIVE = re.compile(r"^Run archive: (.+)$", re.MULTILINE)
DONE = ("completed", "preflight_failed")


def launch(argv: list[str], log: Path) -> int:
    """Run one tuning command from the repository root, appending its output to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        handle.write(f"\n$ {' '.join(argv)}\n".encode())
        handle.flush()
        return subprocess.run(
            argv, cwd=REPOSITORY, stdout=handle, stderr=subprocess.STDOUT, check=False
        ).returncode


def create(
    operators: list[Operator],
    output: Path,
    *,
    agent: str,
    overrides: dict[str, Any],
    hardware_notes: Path | None,
) -> Path:
    """Generate every task into a new batch directory and record the plan."""
    if not operators:
        raise ContractError("No operators selected")
    root = output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-batch")
    root.mkdir(parents=True, exist_ok=False)
    items = {}
    for operator in operators:
        task = render(
            operator, root / "tasks" / operator.name, overrides=overrides, hardware_notes=hardware_notes
        )
        items[operator.name] = {
            "task": task.root.relative_to(root).as_posix(),
            "task_digest": task.identity,
            "status": "pending",
            "run": None,
            "exit_code": None,
            "error": None,
        }
    atomic_json(
        root / "batch.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "overrides": overrides,
            "items": items,
        },
    )
    return root


def preflight(root: Path, names: list[str] | None = None) -> dict[str, str | None]:
    """Measure each production baseline against its own contract before any agent runs.

    This is the run's first evaluator step, done for every operator up front so
    a broken runtime or a contract the production kernel cannot meet is found
    before hours of agent time are spent on earlier operators.
    """
    batch = read_json(root / "batch.json")
    results: dict[str, str | None] = {}
    for name, item in batch["items"].items():
        if names is not None and name not in names or item["status"] != "pending":
            continue
        task = TaskPackage.load(root / item["task"])
        try:
            store = RunStore.create(task, root / "preflight", agent="preflight/none")
            report = asyncio.run(evaluate(store, store.root / "baseline/evaluation"))
            validate_measurement(task, report["measurement"])
            results[name] = None
        except ContractError as exc:
            results[name] = str(exc)
            item.update(status="preflight_failed", error=str(exc))
        atomic_json(root / "batch.json", batch)
    return results


def _state(root: Path, item: dict[str, Any]) -> dict[str, Any] | None:
    if not item["run"]:
        return None
    return read_json(root / item["run"] / "manifest.json")


def execute(root: Path, *, launcher: Launcher = launch) -> dict[str, Any]:
    """Start or resume every unfinished operator in order; return the batch record."""
    batch = read_json(root / "batch.json")
    for name, item in batch["items"].items():
        if item["status"] in DONE:
            continue
        log = root / "logs" / f"{name}.log"
        command = [sys.executable, "-m", "scripts.kernel_tuning"]
        if item["run"]:
            command += ["resume", str(root / item["run"])]
        else:
            command += [
                "run",
                str(root / item["task"]),
                "--agent",
                batch["agent"],
                "--output",
                str(root / "runs"),
            ]
        item["status"] = "running"
        atomic_json(root / "batch.json", batch)
        try:
            code = launcher(command, log)
        except KeyboardInterrupt:
            code = 130
        if not item["run"] and (found := ARCHIVE.findall(log.read_text(encoding="utf-8", errors="replace"))):
            item["run"] = Path(found[-1].strip()).resolve().relative_to(root).as_posix()
        manifest = _state(root, item)
        item["exit_code"] = code
        if manifest and manifest["status"] == "completed":
            item.update(status="completed", error=None)
            if manifest["best"]:
                _export(root, name, item)
        else:
            item["status"] = "interrupted" if code == 130 else "failed"
            item["error"] = (manifest or {}).get("stop_reason") or f"exit {code}; see logs/{name}.log"
        atomic_json(root / "batch.json", batch)
        write_summary(root)
        if code == 130:
            raise KeyboardInterrupt
    write_summary(root)
    return batch


def _export(root: Path, name: str, item: dict[str, Any]) -> None:
    destination = root / "exports" / name
    if destination.exists():
        return
    store = RunStore(root / item["run"])
    with run_lock(store.root):
        store.export(destination)


def _result(root: Path, name: str, item: dict[str, Any]) -> dict[str, Any]:
    manifest = _state(root, item) or {}
    attempts = manifest.get("attempts", [])
    row: dict[str, Any] = {
        "operator": name,
        "status": item["status"],
        "stop_reason": manifest.get("stop_reason") or item.get("error"),
        "attempts": len(attempts),
        "promoted": sum(a["status"] == "promoted" for a in attempts),
        "best": manifest.get("best"),
        "run": item["run"],
        "export": f"exports/{name}" if (root / "exports" / name).is_dir() else None,
    }
    best = next((a for a in attempts if a["id"] == manifest.get("best")), None)
    if best:
        rounds = best["decision"]["rounds"]
        # The weakest paired round is the improvement the evidence supports.
        row["improvement_vs_baseline"] = min(r["improvement"]["baseline"] for r in rounds)
        row["baseline_ms"] = rounds[-1]["mean_latency_ms"]["baseline"]
        row["best_ms"] = rounds[-1]["mean_latency_ms"]["candidate"]
    return row


def write_summary(root: Path) -> list[dict[str, Any]]:
    """Write summary.json and a human-readable summary.md for the whole batch."""
    batch = read_json(root / "batch.json")
    rows = [_result(root, name, item) for name, item in batch["items"].items()]
    atomic_json(root / "summary.json", rows)
    lines = [
        "# Kernel tuning batch",
        "",
        f"Agent: `{batch['agent']}`. Improvements are weighted mean latency against the production",
        "baseline, from the weakest paired round. Exported kernels still need review and",
        "integration before runtime use.",
        "",
        "| operator | status | attempts | promoted | improvement | baseline ms | best ms | detail |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        improvement = f"{row['improvement_vs_baseline']:.1%}" if "improvement_vs_baseline" in row else "-"
        baseline = f"{row['baseline_ms']:.4f}" if "baseline_ms" in row else "-"
        best = f"{row['best_ms']:.4f}" if "best_ms" in row else "-"
        detail = row["export"] or row["stop_reason"] or ""
        lines.append(
            f"| {row['operator']} | {row['status']} | {row['attempts']} | {row['promoted']} | "
            f"{improvement} | {baseline} | {best} | {str(detail).replace('|', '/')} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return rows
