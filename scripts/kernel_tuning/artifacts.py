"""Durable run snapshots and verified, source-complete operator exports."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ContractError, TaskPackage, digest, read_json, tree_hashes

REPOSITORY = Path(__file__).resolve().parents[2]


def atomic_json(path: Path, value: Any) -> None:
    """Replace a complete JSON document atomically on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def tool_files() -> dict[str, str]:
    """List source files needed to reproduce this controller and evaluator version."""
    hashes = {}
    for folder in (REPOSITORY / "scripts/kernel_tuning", REPOSITORY / "benchmarks/kernel_tuning"):
        for directory, subdirs, names in os.walk(folder):
            subdirs[:] = [
                name for name in subdirs if name not in ("tasks", "__pycache__", ".venv", ".ruff_cache")
            ]
            for name in names:
                path = Path(directory) / name
                if path.suffix not in (".py", ".toml", ".txt", ".md", ".lock"):
                    continue
                hashes[path.relative_to(REPOSITORY).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    hashes["scripts/__init__.py"] = hashlib.sha256(
        (REPOSITORY / "scripts/__init__.py").read_bytes()
    ).hexdigest()
    return hashes


def tool_identity() -> str:
    """Bind resumption to the orchestration, prompts, and evaluator source version."""
    return digest(tool_files())


def runtime_identity() -> dict[str, Any]:
    """Record the installed orchestration runtime, including actual hmz source bytes."""
    packages = {}
    for name in ("hmz", "pydantic", "pyyaml", "psutil"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"version": "absent"}
            continue
        packages[name] = {"version": distribution.version}
        if name == "hmz":
            sources = {}
            for item in distribution.files or []:
                if str(item).startswith("hmz/") and str(item).endswith(".py"):
                    sources[str(item)] = hashlib.sha256(
                        Path(distribution.locate_file(item)).read_bytes()
                    ).hexdigest()
            packages[name]["source_digest"] = digest(sources)
            packages[name]["origin"] = distribution.read_text("direct_url.json")
    return {"python": sys.version, "interpreter": sys.executable, "packages": packages}


@contextlib.contextmanager
def run_lock(root: Path) -> Iterator[None]:
    """Hold an OS lock; process death releases it without stale-PID guessing."""
    path = root / ".run.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ContractError("Another process is using this tuning run") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunStore:
    """Persist a run without modifying the input task or engine source tree."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.manifest = read_json(self.root / "manifest.json")
        self.task = TaskPackage.load(self.root / "task")
        if self.task.identity != self.manifest["task_digest"]:
            raise ContractError("Task snapshot no longer matches its manifest")

    @classmethod
    def create(cls, task: TaskPackage, output: Path, agent: str) -> RunStore:
        """Snapshot a task and baseline under a fresh, collision-resistant run ID."""
        task.verify()
        output = output.resolve()
        if output == task.root or task.root in output.parents:
            raise ContractError("Results must be outside the input task directory")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:10]
        root = output / run_id
        root.mkdir(parents=True, exist_ok=False)
        snapshot = root / "task"
        snapshot.mkdir()
        for name in task.hashes:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(task.root / name, target)
        if tree_hashes(snapshot) != task.hashes:
            raise ContractError("Task changed while being snapshotted")
        (root / "workspace").mkdir()
        baseline = task.baseline()
        atomic_json(root / "baseline/solution.json", baseline)
        source_hashes = tool_files()
        for name in source_hashes:
            destination = root / "tool" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY / name, destination)
        if tree_hashes(root / "tool") != source_hashes:
            raise ContractError("Tool source changed while being snapshotted")
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_source": str(task.root),
            "task_digest": task.identity,
            "task_hashes": task.hashes,
            "tool_digest": digest(source_hashes),
            "tool_runtime": runtime_identity(),
            "agent": agent,
            "baseline_digest": digest(baseline),
            "environment": None,
            "elapsed_seconds": 0.0,
            "attempts": [],
            "best": None,
            "streak": 0,
            "status": "created",
            "stop_reason": None,
            "pending": None,
        }
        atomic_json(root / "manifest.json", manifest)
        return cls(root)

    def verify(self, *, check_tool: bool = True) -> None:
        """Check frozen source, task, baseline, and any incumbent before another turn."""
        self.task.verify()
        if check_tool and tool_identity() != self.manifest["tool_digest"]:
            raise ContractError("Tuning/evaluation source changed; start a new run")
        if digest(tree_hashes(self.root / "tool")) != self.manifest["tool_digest"]:
            raise ContractError("Archived evaluator/controller source changed")
        if digest(read_json(self.root / "baseline/solution.json")) != self.manifest["baseline_digest"]:
            raise ContractError("Archived baseline changed")
        if self.manifest["best"]:
            self.solution_path(self.manifest["best"])
            best = next(a for a in self.manifest["attempts"] if a["id"] == self.manifest["best"])
            actual = {
                k: v
                for k, v in tree_hashes(self.root / "candidates" / best["id"]).items()
                if not k.startswith("profile/")
            }
            if actual != best["evidence_hashes"]:
                raise ContractError("Archived winning evidence changed")

    def save(self) -> None:
        """Publish the state mirror and best pointer after durable candidate evidence."""
        atomic_json(self.root / "manifest.json", self.manifest)
        if self.manifest["best"]:
            best = self.manifest["best"]
            atomic_json(
                self.root / "best.json",
                {
                    "candidate": best,
                    "solution": str(self.solution_path(best).relative_to(self.root)),
                    "task_digest": self.task.identity,
                },
            )

    def solution_path(self, candidate: str | None = None) -> Path:
        """Resolve an archived solution and verify its recorded content digest."""
        if candidate is None:
            return self.root / "baseline/solution.json"
        if not candidate.isascii() or not candidate.isdigit():
            raise ContractError("Candidate IDs must contain only ASCII digits")
        attempt = next((a for a in self.manifest["attempts"] if a["id"] == candidate), None)
        if attempt is None or not attempt.get("solution_digest"):
            raise ContractError(f"No archived solution for {candidate}")
        path = self.root / "candidates" / candidate / "solution.json"
        if digest(read_json(path)) != attempt["solution_digest"]:
            raise ContractError(f"Archived solution changed: {candidate}")
        return path

    def begin_attempt(self, *, persist: Callable[[], None] | None = None) -> tuple[str, Path, Path]:
        """Allocate a counted attempt, with scratch separated from measurement records."""
        self.verify()
        candidate = f"{len(self.manifest['attempts']) + 1:04d}"
        archive = self.root / "candidates" / candidate
        scratch = self.root / "workspace" / candidate
        self.manifest["attempts"].append({"id": candidate, "status": "running", "solution_digest": None})
        self.manifest["pending"] = candidate
        # Reserve the attempt in the framework journal before creating directories;
        # a crash halfway through copying large fixtures must still be resumable.
        (persist or self.save)()
        archive.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        shutil.copytree(self.task.root, scratch / "task")
        shutil.copyfile(self.solution_path(self.manifest["best"]), scratch / "incumbent.json")
        atomic_json(scratch / "feedback.json", self.manifest["attempts"][-5:])
        atomic_json(scratch / "environment.json", self.manifest["environment"])
        return candidate, archive, scratch

    def finish_attempt(
        self, candidate: str, *, status: str, decision: dict[str, Any], solution: dict[str, Any] | None = None
    ) -> None:
        """Write evidence before changing which solution future attempts may inherit."""
        archive = self.root / "candidates" / candidate
        atomic_json(archive / "decision.json", decision)
        attempt = next(a for a in self.manifest["attempts"] if a["id"] == candidate)
        attempt.update(status=status, decision=decision)
        if solution is not None:
            atomic_json(archive / "solution.json", solution)
            from .contracts import validate_solution

            validate_solution(solution, self.task)
            for source in solution["sources"]:
                path = archive / "sources" / source["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source["content"].encode("utf-8"))
            attempt["solution_digest"] = digest(solution)
        attempt["evidence_hashes"] = {
            k: v for k, v in tree_hashes(archive).items() if not k.startswith("profile/")
        }
        if status == "promoted":
            if solution is None:
                raise ContractError("Cannot promote without an archived solution")
            self.manifest["best"] = candidate
            self.manifest["streak"] = 0
        else:
            self.manifest["streak"] += 1
        self.manifest["pending"] = None

    def recover(self) -> None:
        """Keep an interrupted attempt as failed evidence rather than replaying its writes."""
        if candidate := self.manifest["pending"]:
            self.finish_attempt(
                candidate,
                status="interrupted",
                decision={"promoted": False, "reasons": ["Interrupted before durable completion"]},
            )

    def export(self, destination: Path) -> Path:
        """Export the best source, frozen task, and evidence into a new directory."""
        self.verify(check_tool=False)
        best = self.manifest["best"]
        if not best:
            raise ContractError("No candidate has passed promotion; all attempts remain in the run archive")
        destination = destination.resolve()
        if destination.exists():
            raise ContractError("Export destination already exists; choose a new directory")
        if destination == self.root or self.root in destination.parents:
            raise ContractError("Export outside the run directory")
        solution = read_json(self.solution_path(best))
        from .contracts import validate_solution

        validate_solution(solution, self.task)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".kernel-export-", dir=destination.parent))
        try:
            for source in solution["sources"]:
                path = staging / "sources" / source["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source["content"].encode("utf-8"))
            atomic_json(staging / "solution.json", solution)
            atomic_json(staging / "manifest.json", self.manifest)
            shutil.copytree(self.root / "candidates" / best, staging / "evidence")
            shutil.copytree(self.root / "baseline", staging / "baseline")
            shutil.copytree(self.task.root, staging / "task")
            shutil.copytree(self.root / "tool", staging / "tool")
            if (self.root / "humanize").exists():
                shutil.copytree(self.root / "humanize", staging / "humanize")
            if (self.root / "humanize.json").exists():
                shutil.copyfile(self.root / "humanize.json", staging / "humanize.json")
            atomic_json(staging / "checksums.json", tree_hashes(staging))
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination
