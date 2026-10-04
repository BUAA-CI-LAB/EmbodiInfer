"""One-command tuning of every selected catalog operator, serially on one machine.

A batch directory holds freshly generated tasks, one ordinary run archive per
operator, logs, exports of promoted kernels, and a summary. Each operator runs
in its own ``run``/``resume`` subprocess, so one failure or crash does not stop
the others, and an interrupted batch continues where it stopped.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import REPOSITORY, RunStore, atomic_json, run_lock, tool_identity
from .contracts import ContractError, TaskPackage, digest, read_json, relative_path, validate_measurement
from .discovery.contracts import ModelCapture
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
    _verify_model(root, batch)
    if batch.get("model"):
        if names is not None:
            raise ContractError("Model preflight must include every discovered task")
        batch["preflight_complete"] = False
        atomic_json(root / "batch.json", batch)
    results: dict[str, str | None] = {}
    for name, item in batch["items"].items():
        if (names is not None and name not in names) or (
            not batch.get("model") and item["status"] != "pending"
        ):
            continue
        task = TaskPackage.load(root / item["task"])
        try:
            store = RunStore.create(task, root / "preflight", agent="preflight/none")
            report = asyncio.run(evaluate(store, store.root / "baseline/evaluation"))
            validate_measurement(task, report["measurement"])
            if batch.get("model"):
                if not report.get("environment", {}).get("identity"):
                    raise ContractError("Model preflight requires the evaluator environment identity")
                receipt = {
                    "task_digest": task.identity,
                    "tool_digest": tool_identity(),
                    "environment": report["environment"],
                    "measurement": report["measurement"],
                }
                path = store.root / "baseline/preflight.json"
                atomic_json(path, receipt)
                item.update(
                    preflight_receipt=path.relative_to(root).as_posix(),
                    preflight_receipt_digest=digest(receipt),
                    preflight_environment=report["environment"],
                )
            results[name] = None
            item["preflight_passed"] = True
            if item["status"] == "preflight_failed":
                item.update(status="pending", error=None)
            # Baselines already at the device's launch floor leave an agent nothing to win.
            rounds = report["measurement"]["rounds"]
            item["baseline_us"] = {
                work: 1000
                * sum(sum(r[work]["baseline"]) / len(r[work]["baseline"]) for r in rounds)
                / len(rounds)
                for work in task.workload_ids
            }
        except ContractError as exc:
            results[name] = str(exc)
            item.update(status="preflight_failed", error=str(exc), preflight_passed=False)
        atomic_json(root / "batch.json", batch)
    if batch.get("model"):
        batch["preflight_complete"] = bool(results) and all(error is None for error in results.values())
        batch["preflight_tool_digest"] = tool_identity()
        atomic_json(root / "batch.json", batch)
        write_summary(root)
    return results


def create_model(
    capture: ModelCapture, output: Path, *, agent: str, overrides: dict[str, Any], hardware_notes: Path | None
) -> Path:
    """Freeze one model capture and all tasks in a batch requiring a complete preflight."""
    from .discovery.tasks import render_model

    capture.require_ready()
    root = output.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}-model"
    )
    root.mkdir(parents=True, exist_ok=False)
    # Verify before copying, then verify the copy, to reject a stale capture.
    checked = ModelCapture.load(capture.root, capture.manifest["target"]["model"])
    if checked.identity != capture.identity:
        raise ContractError("Capture changed before batch creation")
    shutil.copytree(capture.root, root / "capture")
    frozen = ModelCapture.load(root / "capture", capture.manifest["target"]["model"])
    if frozen.identity != capture.identity:
        raise ContractError("Capture changed during batch creation")
    tasks = render_model(frozen, root / "tasks", overrides=overrides, hardware_notes=hardware_notes)
    names = {op["id"]: op["name"] for op in capture.operators}
    items = {
        task.root.name: {
            "name": names[task.root.name],
            "task": task.root.relative_to(root).as_posix(),
            "task_digest": task.identity,
            "status": "pending",
            "run": None,
            "exit_code": None,
            "error": None,
            "preflight_passed": False,
        }
        for task in tasks
    }
    atomic_json(
        root / "batch.json",
        {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "overrides": overrides,
            "model": capture.manifest["target"],
            "capture_identity": capture.identity,
            "preflight_complete": False,
            "items": items,
        },
    )
    return root


def _verify_model(root: Path, batch: dict[str, Any], *, require_preflight: bool = False) -> None:
    if batch.get("schema_version") != 2:
        if batch.get("model") or (root / "tasks/model.json").exists():
            raise ContractError("Unsupported model batch schema")
        return
    capture = ModelCapture.load(root / "capture", batch["model"]["model"])
    if capture.identity != batch["capture_identity"] or capture.manifest["target"] != batch["model"]:
        raise ContractError("Batch target/capture identity changed")
    expected = {op["id"] for op in capture.operators if op["status"] == "ready"}
    if set(batch["items"]) != expected:
        raise ContractError("Model batch has task coverage gaps")
    index = read_json(root / "tasks/model.json")
    if index["capture_identity"] != capture.identity or set(index["tasks"]) != expected:
        raise ContractError("Generated task index differs from the model capture")
    for name, item in batch["items"].items():
        if item["task"] != f"tasks/{name}":
            raise ContractError("Model task path changed")
        task = TaskPackage.load(root / item["task"])
        if task.identity != item["task_digest"] or task.identity != index["tasks"][name]:
            raise ContractError("Model task changed after generation; prepare a new batch")
        replay = read_json(task.root / "replay.json")
        if replay["capture_identity"] != capture.identity or replay["target"] != batch["model"]:
            raise ContractError("Task belongs to a different target model")
    if require_preflight and (
        not batch.get("preflight_complete")
        or not all(item.get("preflight_passed") for item in batch["items"].values())
    ):
        raise ContractError(
            "Model batch blocked: every task must pass baseline preflight before any agent starts"
        )
    if require_preflight and batch.get("preflight_tool_digest") != tool_identity():
        raise ContractError("Tuning tools changed after model preflight; rerun the batch preflight")
    if require_preflight:
        for item in batch["items"].values():
            receipt = read_json(root / relative_path(item["preflight_receipt"]))
            if (
                digest(receipt) != item["preflight_receipt_digest"]
                or receipt["task_digest"] != item["task_digest"]
                or receipt["tool_digest"] != batch["preflight_tool_digest"]
                or receipt["environment"] != item["preflight_environment"]
            ):
                raise ContractError("Model preflight evidence changed; rerun the batch preflight")


def verify_model_task(task: TaskPackage) -> Path:
    """Require a generated model task to belong to a fully preflighted model batch."""
    root = task.root.parent.parent
    if not (root / "batch.json").is_file():
        raise ContractError(
            "Start discovered tasks with tune-all; standalone run cannot bypass model preflight"
        )
    record = read_json(root / "batch.json")
    if record.get("schema_version") != 2:
        raise ContractError("Discovered tasks require a model batch")
    _verify_model(root, record, require_preflight=True)
    if record["items"].get(task.root.name, {}).get("task_digest") != task.identity:
        raise ContractError("Task does not belong to this model batch")
    return root


def verify_model_run(store: RunStore) -> None:
    """Revalidate the entire model before an archived operator search resumes."""
    if not store.manifest.get("model_batch"):
        raise ContractError("Model run has no preflighted parent batch")
    root = Path(store.manifest["model_batch"])
    record = read_json(root / "batch.json")
    if record.get("schema_version") != 2:
        raise ContractError("Model run requires a model batch")
    _verify_model(root, record, require_preflight=True)
    if store.task.identity not in {item["task_digest"] for item in record["items"].values()}:
        raise ContractError("Run task no longer belongs to the model batch")
    item = next(item for item in record["items"].values() if item["task_digest"] == store.task.identity)
    if store.manifest["environment"]["identity"] != item["preflight_environment"]["identity"]:
        raise ContractError("Model evaluator environment changed; start a new model batch")


def _state(root: Path, item: dict[str, Any]) -> dict[str, Any] | None:
    if not item["run"]:
        return None
    return read_json(root / item["run"] / "manifest.json")


def execute(root: Path, *, launcher: Launcher = launch) -> dict[str, Any]:
    """Start or resume every unfinished operator in order; return the batch record."""
    batch = read_json(root / "batch.json")
    _verify_model(root, batch, require_preflight=True)
    for name, item in batch["items"].items():
        if item["status"] in DONE:
            continue
        _verify_model(root, batch, require_preflight=True)
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
    if batch.get("model"):
        coverage = read_json(root / "capture/coverage.json")["operators"]
        lines[2:2] = [
            f"Model: `{batch['model']['model']}`. Capture: `{batch['capture_identity']}`.",
            f"Coverage: `{coverage}`. All-baseline preflight passed: `{batch['preflight_complete']}`.",
            "",
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
