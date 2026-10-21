"""CPU fixtures; use --confcutdir here in the isolated, Torch-free tool environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.kernel_tuning.artifacts import RunStore
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


GATED_RESIDUAL_REFERENCE = """import torch

def run(residual, update, gate):
    product = (update * gate).to(update.dtype)
    return (residual + product).to(residual.dtype)
"""

GATED_RESIDUAL_DEFINITION = {
    "name": "gated_residual_bf16",
    "op_type": "elementwise",
    "description": "Residual plus a gated update, with BF16 product rounding before addition.",
    "axes": {"B": {"type": "var"}, "T": {"type": "var"}, "H": {"type": "var"}, "G": {"type": "var"}},
    "inputs": {
        "residual": {"shape": ["B", "T", "H"], "dtype": "bfloat16"},
        "update": {"shape": ["B", "T", "H"], "dtype": "bfloat16"},
        "gate": {"shape": ["B", "G", "H"], "dtype": "bfloat16"},
    },
    "outputs": {"output": {"shape": ["B", "T", "H"], "dtype": "bfloat16"}},
    "constraints": ["G == 1 or G == T", "B > 0", "T > 0", "H > 0"],
    "reference": GATED_RESIDUAL_REFERENCE,
}

GATED_RESIDUAL_WORKLOADS = {
    "broadcast-small": {"B": 1, "T": 16, "H": 1024, "G": 1},
    "broadcast-wide": {"B": 2, "T": 32, "H": 2048, "G": 1},
    "token-gate-tail": {"B": 1, "T": 17, "H": 1000, "G": 17},
}

GATED_RESIDUAL_TUNING = {
    "schema_version": 1,
    "language": "triton",
    "agent": "",
    "evaluator_python": "python",
    "device": "cuda:0",
    "target_hardware": ["cuda"],
    "seeds": [0, 1, 2],
    "precision": {"mode": "bit_exact", "matmul_precision": "highest", "allow_tf32": False},
    "timing": {"mode": "eager", "warmup": 10, "iterations": 100, "trials": 3, "paired_rounds": 2},
    "acceptance": {"min_improvement": 0.03, "max_regression": 0.05},
    "search": {
        "max_candidates": 20,
        "max_seconds": 7200,
        "patience": 5,
        "agent_timeout_seconds": 600,
        "evaluation_timeout_seconds": 300,
    },
    "profile": False,
}


@pytest.fixture
def example_task(tmp_path: Path) -> Path:
    """Write a hand-written BF16 gated residual task package with synthetic workloads."""
    path = tmp_path / "task"
    path.mkdir()
    (path / "README.md").write_text("# Gated residual BF16 test task\n", encoding="utf-8")
    (path / "definition.json").write_text(json.dumps(GATED_RESIDUAL_DEFINITION, indent=2), encoding="utf-8")
    (path / "baseline.py").write_text(GATED_RESIDUAL_REFERENCE, encoding="utf-8")
    inputs = {name: {"type": "random"} for name in GATED_RESIDUAL_DEFINITION["inputs"]}
    (path / "workloads.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "definition": GATED_RESIDUAL_DEFINITION["name"],
                    "workload": {"uuid": uuid, "axes": axes, "inputs": inputs},
                    "solution": None,
                    "evaluation": None,
                }
            )
            + "\n"
            for uuid, axes in GATED_RESIDUAL_WORKLOADS.items()
        ),
        encoding="utf-8",
    )
    (path / "tuning.yaml").write_text(yaml.safe_dump(GATED_RESIDUAL_TUNING), encoding="utf-8")
    return path


@pytest.fixture
def task_dir(example_task: Path) -> Path:
    """Make a portable task using a fake measurement adapter and explicit Python."""
    path = example_task
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


@pytest.fixture
def registry_capture():
    """Build CPU captures using the same recorder as capture-ops, without a checkpoint."""
    pytest.importorskip("torch")
    from scripts.kernel_tuning.artifacts import atomic_json
    from scripts.kernel_tuning.contracts import read_json
    from scripts.kernel_tuning.discovery.contracts import ModelCapture
    from scripts.kernel_tuning.regions import OpCapture

    from embodiinfer.layers.ops import get_op

    def record(
        root: Path, calls: list[tuple], *, budget: int = 1024 * 1024, device_type: str = "cpu"
    ) -> ModelCapture:
        capture = OpCapture("mock_flow_vla", root / "raw", warmup=0, steps=1, fixture_bytes=budget)
        capture.target = {"model": "mock_flow_vla", "type": "test.Policy", "config": {}}
        capture.started = 1
        try:
            for name, args, kwargs in calls:
                function = get_op(name, "torch")
                capture.record(name, function, function, args, kwargs)
        finally:
            capture.close(complete=True)
        # The CLI targets CUDA; the fixture changes only the declared evaluator
        # device, after recording real CPU tensors through OpCapture.record.
        manifest = read_json(capture.root / "manifest.json")
        manifest["device_type"] = device_type
        atomic_json(capture.root / "manifest.json", manifest)
        return ModelCapture.load(capture.root, require_ready=False)

    return record
