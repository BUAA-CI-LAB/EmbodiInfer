"""Batch tuning over generated tasks, with fake run commands and synthetic evidence."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from scripts.kernel_tuning import batch
from scripts.kernel_tuning.__main__ import main
from scripts.kernel_tuning.artifacts import RunStore, atomic_json
from scripts.kernel_tuning.contracts import ContractError, TaskPackage, promotion, read_json
from scripts.kernel_tuning.operators import select


def measurement(task: TaskPackage, latency: float) -> dict[str, Any]:
    """Complete synthetic evidence: the candidate at ``latency``, other roles at 10 ms."""
    roles = ("baseline", "incumbent", "candidate")
    return {
        "checks": [
            {
                "workload": work,
                "role": role,
                "passed": True,
                "seeds": list(task.settings.seeds),
                "precision": asdict(task.settings.precision),
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "graph_replay": True,
            }
            for work in task.workload_ids
            for role in roles
        ],
        "rounds": [
            {
                work: {
                    r: [latency if r == "candidate" else 10.0] * task.settings.timing.trials for r in roles
                }
                for work in task.workload_ids
            }
            for _ in range(task.settings.timing.paired_rounds)
        ],
    }


class FakeRuns:
    """Stands in for ``run``/``resume`` subprocesses, using the real archive code."""

    def __init__(self, outcomes: dict[str, list[str]]) -> None:
        self.outcomes = outcomes
        self.commands: list[list[str]] = []

    def __call__(self, argv: list[str], log: Path) -> int:
        self.commands.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        if argv[3] == "run":
            task = TaskPackage.load(Path(argv[4]))
            store = RunStore.create(
                task, Path(argv[argv.index("--output") + 1]), argv[argv.index("--agent") + 1]
            )
            log.write_text(f"Run archive: {store.root}\n", encoding="utf-8")
        else:
            store = RunStore(Path(argv[4]))
        name = store.task.definition["name"].removeprefix("embodiinfer_")
        outcome = self.outcomes[name].pop(0)
        if outcome == "interrupt":
            store.manifest.update(status="interrupted", stop_reason="KeyboardInterrupt")
            store.save()
            return 130
        if outcome == "fail":
            store.manifest.update(status="failed", stop_reason="ContractError: evaluator unavailable")
            store.save()
            return 2
        if outcome == "promote":
            candidate, _, _ = store.begin_attempt()
            solution = {
                "name": f"{name}_fast",
                "definition": store.task.definition["name"],
                "author": "fake-agent",
                "spec": {
                    "language": "triton",
                    "target_hardware": ["cuda"],
                    "entry_point": "kernel.py::run",
                    "destination_passing_style": False,
                },
                "sources": [{"path": "kernel.py", "content": "def run(*args):\n    return args\n"}],
            }
            decision = promotion(store.task, measurement(store.task, 8.0))
            store.finish_attempt(candidate, status="promoted", decision=decision, solution=solution)
        store.manifest.update(status="completed", stop_reason="max_candidates")
        store.save()
        return 0


def new_batch(tmp_path: Path, names: list[str]) -> Path:
    return batch.create(
        select(names), tmp_path / "batches", agent="claude/fake:low", overrides={}, hardware_notes=None
    )


def test_batch_runs_every_operator_and_exports_promoted_kernels(tmp_path: Path) -> None:
    root = new_batch(tmp_path, ["gated_residual", "swiglu", "rms_norm"])
    runs = FakeRuns({"gated_residual": ["promote"], "swiglu": ["fail", "promote"], "rms_norm": ["complete"]})
    record = batch.execute(root, launcher=runs)
    status = {name: item["status"] for name, item in record["items"].items()}
    assert status == {"gated_residual": "completed", "swiglu": "failed", "rms_norm": "completed"}
    assert record["items"]["swiglu"]["error"] == "ContractError: evaluator unavailable"
    assert (root / "exports/gated_residual/sources/kernel.py").is_file()
    assert not (root / "exports/rms_norm").exists()
    summary = {row["operator"]: row for row in read_json(root / "summary.json")}
    assert summary["gated_residual"]["improvement_vs_baseline"] == pytest.approx(0.2)
    assert summary["gated_residual"]["export"] == "exports/gated_residual"
    assert "| gated_residual | completed | 1 | 1 | 20.0% |" in (root / "summary.md").read_text(
        encoding="utf-8"
    )

    # Continuing the batch resumes only the failed run, inside its existing archive.
    batch.execute(root, launcher=runs)
    assert [command[3] for command in runs.commands] == ["run", "run", "run", "resume"]
    assert runs.commands[-1][4] == str(root / record["items"]["swiglu"]["run"])
    assert read_json(root / "batch.json")["items"]["swiglu"]["status"] == "completed"
    assert (root / "exports/swiglu").is_dir()


def test_interrupt_stops_the_batch_and_resume_continues_it(tmp_path: Path) -> None:
    root = new_batch(tmp_path, ["gated_residual", "rms_norm"])
    runs = FakeRuns({"gated_residual": ["interrupt", "complete"], "rms_norm": ["complete"]})
    with pytest.raises(KeyboardInterrupt):
        batch.execute(root, launcher=runs)
    items = read_json(root / "batch.json")["items"]
    assert items["gated_residual"]["status"] == "interrupted"
    assert items["rms_norm"]["status"] == "pending"
    batch.execute(root, launcher=runs)
    assert [command[3] for command in runs.commands] == ["run", "resume", "run"]


def test_parallel_batch_overlaps_runs_and_leaves_skipped_items(tmp_path: Path) -> None:
    """--parallel starts runs together; an item set to skipped by hand never starts."""
    root = new_batch(tmp_path, ["gated_residual", "swiglu", "rms_norm"])
    record = read_json(root / "batch.json")
    record["items"]["rms_norm"]["status"] = "skipped"
    atomic_json(root / "batch.json", record)
    fake = FakeRuns({"gated_residual": ["promote"], "swiglu": ["promote"]})
    together = threading.Barrier(2, timeout=10)

    def runs(argv: list[str], log: Path) -> int:
        together.wait()  # Fails unless the other run has started too.
        return fake(argv, log)

    record = batch.execute(root, launcher=runs, parallel=2)
    status = {name: item["status"] for name, item in record["items"].items()}
    assert status == {"gated_residual": "completed", "swiglu": "completed", "rms_norm": "skipped"}
    assert (root / "exports/gated_residual").is_dir() and (root / "exports/swiglu").is_dir()
    assert read_json(root / "batch.json")["items"] == record["items"]
    with pytest.raises(ContractError, match="at least 1"):
        batch.execute(root, launcher=runs, parallel=0)


def test_run_that_fails_before_archiving_is_started_again(tmp_path: Path) -> None:
    root = new_batch(tmp_path, ["rms_norm"])
    calls = []

    def broken(argv: list[str], log: Path) -> int:
        calls.append(argv[3])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("kernel_tuning: Install the optional tooling environment first\n", encoding="utf-8")
        return 2

    record = batch.execute(root, launcher=broken)
    assert record["items"]["rms_norm"]["run"] is None
    assert record["items"]["rms_norm"]["error"] == "exit 2; see logs/rms_norm.log"
    batch.execute(root, launcher=broken)
    assert calls == ["run", "run"]


def test_failed_preflight_skips_the_operator(tmp_path: Path, monkeypatch) -> None:
    root = new_batch(tmp_path, ["gated_residual", "rms_norm"])

    async def evaluate(store: RunStore, output: Path, **_: Any) -> dict[str, Any]:
        if store.task.definition["name"].endswith("rms_norm"):
            raise ContractError("Evaluator failed (exit 1)")
        return {"measurement": measurement(store.task, 10.0)}

    monkeypatch.setattr(batch, "evaluate", evaluate)
    assert batch.preflight(root) == {"gated_residual": None, "rms_norm": "Evaluator failed (exit 1)"}
    items = read_json(root / "batch.json")["items"]
    assert items["gated_residual"]["baseline_us"] == {"pi05-expert-b1": 10000.0, "pi05-expert-b8": 10000.0}
    assert "baseline_us" not in items["rms_norm"]
    runs = FakeRuns({"gated_residual": ["complete"]})
    record = batch.execute(root, launcher=runs)
    assert record["items"]["rms_norm"]["status"] == "preflight_failed"
    assert len(runs.commands) == 1


def test_cli_validates_agent_and_resume_arguments(tmp_path: Path, capsys) -> None:
    output = tmp_path / "batches"
    assert main(["tune-all", "rms_norm", "--output", str(output)]) == 2
    assert "explicit Humanize2 agent" in capsys.readouterr().err
    assert not output.exists()
    assert main(["tune-all", "--catalog", "--model", "streamvln", "--dry-run", "--output", str(output)]) == 0
    (root,) = output.iterdir()
    assert set(read_json(root / "batch.json")["items"]) == {"rms_norm", "add_rms_norm", "swiglu"}
    assert "| swiglu | pending |" in capsys.readouterr().out
    assert main(["tune-all", "--resume", str(root), "--set", "device=cuda:1"]) == 2
    assert "--resume continues the recorded batch" in capsys.readouterr().err
    assert main(["tune-all", "--resume", str(root), "--unfenced"]) == 2
    assert "--resume continues the recorded batch" in capsys.readouterr().err


def test_unfenced_batch_starts_every_run_unfenced(tmp_path: Path) -> None:
    root = batch.create(
        select(["rms_norm"]),
        tmp_path / "batches",
        agent="claude/fake:low",
        overrides={},
        hardware_notes=None,
        unfenced=True,
    )
    commands: list[list[str]] = []

    def record(argv: list[str], log: Path) -> int:
        commands.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("stopped before archiving\n", encoding="utf-8")
        return 1

    batch.execute(root, launcher=record)
    assert commands and commands[0][3] == "run" and "--unfenced" in commands[0]
    assert "unfenced" in (root / "summary.md").read_text(encoding="utf-8")
    assert read_json(new_batch(tmp_path / "default", ["rms_norm"]) / "batch.json")["unfenced"] is False


def test_cli_generate_writes_overridden_tasks(tmp_path: Path, capsys) -> None:
    notes = tmp_path / "thor.md"
    notes.write_text("Jetson Thor: sm_110, 128 GB unified memory.\n", encoding="utf-8")
    argv = [
        "generate",
        "--catalog",
        "rms_norm",
        "--output",
        str(tmp_path / "tasks"),
        "--hardware-notes",
        str(notes),
    ]
    assert main([*argv, "--set", "evaluator_python=/opt/runtime/bin/python"]) == 0
    task = TaskPackage.load(tmp_path / "tasks/rms_norm")
    assert task.settings.evaluator_python == "/opt/runtime/bin/python"
    assert "HARDWARE.md" in task.hashes
    assert "HARDWARE.md" in (task.root / "README.md").read_text(encoding="utf-8")
    assert main(argv) == 2
    assert "--force" in capsys.readouterr().err


def test_variable_axes_must_be_inferable_from_inputs(tmp_path: Path) -> None:
    root = new_batch(tmp_path, ["rms_norm"])
    task = root / "tasks/rms_norm"
    definition = read_json(task / "definition.json")
    definition["axes"]["X"] = {"type": "var"}
    definition["outputs"]["output"]["shape"] = ["M", "X"]
    (task / "definition.json").write_text(json.dumps(definition), encoding="utf-8")
    with pytest.raises(ContractError, match="appear in an input shape"):
        TaskPackage.load(task)
