"""CPU fixtures; use --confcutdir here in the isolated, Torch-free tool environment."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.kernel_tuning.artifacts import REPOSITORY, RunStore
from scripts.kernel_tuning.contracts import TaskPackage

yaml = pytest.importorskip("yaml", reason="requires the optional kernel-tuning tooling environment")

FAKE_BENCHMARK = '''"""Deterministic CPU fake: performs no kernel compilation or GPU work."""
from dataclasses import asdict
import time

class Adapter:
    def __init__(self, task):
        self.task = task

    def environment(self):
        return {"identity": {"hardware": "fake-cpu", "version": "1"}, "conditions": {"fake": True}}

    def evaluate(self, solutions):
        if solutions["candidate"].get("description") == "crash":
            raise RuntimeError("simulated compilation failure")
        if solutions["candidate"].get("description") == "timeout":
            time.sleep(30)
        cfg = self.task.settings
        roles = ("baseline", "incumbent", "candidate")
        checks = [{"workload": w, "role": role, "passed": True,
                   "seeds": list(cfg.seeds), "precision": asdict(cfg.precision),
                   "max_abs_error": 0.0, "max_rel_error": 0.0,
                   "graph_replay": cfg.timing.mode == "cuda_graph"}
                  for w in self.task.workload_ids for role in roles]
        rounds = [{w: {r: [float(solutions[r].get("description") or 10)] * cfg.timing.trials
                       for r in roles} for w in self.task.workload_ids}
                  for _ in range(cfg.timing.paired_rounds)]
        return {"checks": checks, "rounds": rounds}

def create_adapter(task):
    return Adapter(task)
'''


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """Make a portable task using a fake measurement adapter and explicit Python."""
    path = tmp_path / "task"
    shutil.copytree(REPOSITORY / "benchmarks/kernel_tuning/tasks/gated_residual", path)
    (path / "benchmark.py").write_text(FAKE_BENCHMARK, encoding="utf-8")
    config = yaml.safe_load((path / "tuning.yaml").read_text(encoding="utf-8"))
    config["evaluator_python"] = sys.executable
    config["search"].update(
        max_candidates=3, patience=2, max_seconds=60, agent_timeout_seconds=5, evaluation_timeout_seconds=10
    )
    (path / "tuning.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def store(task_dir: Path, tmp_path: Path) -> RunStore:
    """Create only the task archive; no agent or evaluator starts here."""
    return RunStore.create(TaskPackage.load(task_dir), tmp_path / "runs", "claude/fake:low")


@pytest.fixture
def candidate(store: RunStore) -> dict[str, Any]:
    """A source-complete Solution; the fake evaluator reads its latency description."""
    return {
        "name": "fake_candidate",
        "definition": store.task.definition["name"],
        "author": "test",
        "description": "8",
        "spec": {
            "language": "triton",
            "target_hardware": ["cuda"],
            "entry_point": "kernel.py::run",
            "destination_passing_style": False,
        },
        "sources": [
            {"path": "kernel.py", "content": "def run(residual, update, gate):\n    return residual\n"}
        ],
    }
