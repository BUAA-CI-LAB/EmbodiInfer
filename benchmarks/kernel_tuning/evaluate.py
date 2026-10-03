"""Trusted, isolated entry point for the kernel-tuning measurement protocol."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, Protocol

# This is a repository measurement program, also usable with an isolated Python.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernel_tuning.artifacts import atomic_json  # noqa: E402
from scripts.kernel_tuning.contracts import TaskPackage, digest, read_json  # noqa: E402


class EvaluationAdapter(Protocol):
    """Task-owned extension for hardware/input semantics outside the default adapter.

    Implement create_adapter(task) in the frozen benchmark.py. Methods return JSON
    data; evaluate uses the checks/rounds contract documented in the proposal.
    environment must separate stable identity from transient measurement conditions.
    """

    def environment(self) -> dict[str, Any]:
        """Report stable hardware/software identity and experimental conditions."""
        ...

    def evaluate(self, solutions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Validate every solution and measure baseline/incumbent/candidate pairs."""
        ...


def load_adapter(task: TaskPackage) -> EvaluationAdapter:
    """Select the frozen task benchmark or the pinned FlashInfer implementation."""
    custom = task.root / "benchmark.py"
    if custom.exists():
        spec = importlib.util.spec_from_file_location(f"kernel_benchmark_{task.identity}", custom)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot import task benchmark.py")
        module = importlib.util.module_from_spec(spec)
        # Helpers can be shipped beside benchmark.py; the entire task is hashed.
        sys.path.insert(0, str(task.root))
        spec.loader.exec_module(module)
        return module.create_adapter(task)
    from flashinfer_adapter import FlashInferAdapter

    return FlashInferAdapter(task)


def execute(request: dict[str, Any], output: Path) -> dict[str, Any]:
    """Bind the adapter's evidence to the exact input sources and request nonce."""
    if request.get("schema_version") != 1 or request.get("operation") not in ("probe", "evaluate", "profile"):
        raise ValueError("Unsupported evaluator request")
    task = TaskPackage.load(Path(request["task_dir"]))
    if task.identity != request["task_digest"]:
        raise ValueError("Task content changed before evaluation")
    solutions = {role: read_json(Path(path)) for role, path in request["solutions"].items()}
    if {role: digest(solution) for role, solution in solutions.items()} != request["solution_digests"]:
        raise ValueError("Solution content changed before evaluation")
    adapter = load_adapter(task)
    before = adapter.environment()
    response = {
        "schema_version": 1,
        "status": "ok",
        "request_digest": digest(request),
        "task_digest": task.identity,
        "solution_digests": request["solution_digests"],
        "environment": before,
    }
    if request["operation"] == "evaluate":
        response["measurement"] = adapter.evaluate(solutions)
    elif request["operation"] == "profile":
        profiler = getattr(adapter, "profile", None)
        if profiler is None:
            raise ValueError("This benchmark adapter does not implement profile(solutions, output_dir)")
        response["profile"] = profiler(solutions, output.parent)
    after = adapter.environment()
    if before["identity"] != after["identity"]:
        raise ValueError("Hardware/software or precision conditions changed during evaluation")
    response["environment_after"] = after
    task.verify()
    for role, path in request["solutions"].items():
        if digest(read_json(Path(path))) != request["solution_digests"][role]:
            raise ValueError("Solution file changed during evaluation")
    return response


def main() -> int:
    """Emit structured results only after successful measurement; errors exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        response = execute(read_json(args.request), args.output)
        atomic_json(args.output, response)
        return 0
    except Exception as exc:
        traceback.print_exc()
        atomic_json(
            args.output, {"schema_version": 1, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
