"""Isolated Git checkouts, bounded CPU workers and durable registration state."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import difflib
import hashlib
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.kernel_tuning.artifacts import atomic_json, runtime_identity
from scripts.kernel_tuning.contracts import ContractError, digest, read_json
from scripts.kernel_tuning.runner import _kill_tree

from .sites import locate, render, source_path, validate_review
from .spec import RegistrationSpec

REFERENCE = "embodiinfer/backend/torch/ops.py"
REGISTRY = "embodiinfer/layers/ops.py"
ARTIFACTS = {"PLAN.md", "REGISTRATION.json"}


def git(root: Path, *args: str) -> str:
    """Execute a bounded Git command without a shell or ambient Git redirection."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContractError(f"Git command timed out: {args}") from exc
    if result.returncode:
        raise ContractError(f"git {' '.join(args)}: {result.stderr.strip()[:4000]}")
    return result.stdout


def file_hash(path: Path) -> str:
    """Hash artifact bytes, rejecting links and missing files."""
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"Expected a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_hashes(root: Path) -> dict[str, str]:
    """Hash tracked files and link targets without following unrelated repository links."""
    hashes = {}
    for name in git(root, "ls-files", "-z").split("\0"):
        if not name:
            continue
        path = root / name
        # Documentation assets may be tracked links on POSIX. Check their parent
        # for traversal, then bind the link text itself, never the target bytes.
        source_path(root, str(Path(name).parent))
        hashes[name] = (
            "symlink:" + hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
            if path.is_symlink()
            else file_hash(source_path(root, name))
        )
    return hashes


def cpu_environment(root: Path) -> dict[str, str]:
    """Force subprocess imports into this checkout and hide accelerator devices."""
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONOPTIMIZE="0",
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        PYTEST_ADDOPTS="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    return env


async def command(argv: list[str], *, root: Path, output: Path, timeout: float) -> dict[str, Any]:
    """Capture a CPU subprocess and kill its entire process group on cancellation."""
    if timeout <= 0:
        raise TimeoutError("Registration duration exhausted")
    output.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"argv": argv, "cwd": str(root), "timeout_seconds": timeout}
    with output.open("wb") as stream:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=root,
            env=cpu_environment(root),
            stdout=stream,
            stderr=stream,
            start_new_session=os.name != "nt",
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except BaseException:
            await _kill_tree(process)
            raise
    record["returncode"] = process.returncode
    with output.open("rb") as stream:
        stream.seek(max(0, output.stat().st_size - 12000))
        record["output"] = stream.read().decode("utf-8", errors="replace")
    record["log"] = str(output)
    return record


def import_guard(root: Path) -> str:
    """Check the actual package origin inside the same interpreter as the check."""
    return (
        "import pathlib, embodiinfer\n"
        f"assert pathlib.Path(embodiinfer.__file__).resolve().is_relative_to(pathlib.Path({str(root)!r}).resolve()), 'embodiinfer imported outside checkout'\n"
    )


async def check_unregistered(root: Path, op: str, python: str, output: Path) -> None:
    """Probe the public registry in the selected test environment before mutation."""
    tree = ast.parse(source_path(root, REFERENCE).read_text(encoding="utf-8"))
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == op for n in ast.walk(tree)):
        raise ContractError(f"Reference function {op!r} already exists")
    code = (
        import_guard(root)
        + f"from embodiinfer.layers import available_ops\nassert {op!r} not in available_ops(), 'operator already registered'\n"
    )
    result = await command([python, "-c", code], root=root, output=output, timeout=60)
    if result["returncode"]:
        raise ContractError(
            f"Registry preflight failed; select a prepared --set test_python=/abs/python:\n{result['output']}"
        )


class RegistrationRun:
    """Controller-owned ledger plus two checkouts at an immutable starting commit."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.manifest = read_json(self.root / "manifest.json")
        if file_hash(self.root / "spec.json") != self.manifest["spec_hash"]:
            self.manifest.update(status="failed", stop_reason="Frozen specification changed")
            self.save()
            raise ContractError("Frozen specification changed")
        frozen = read_json(self.root / "spec.json")
        self.spec = RegistrationSpec.parse(frozen["spec"])
        self.base_commit = frozen["base_commit"]
        self.base = self.root / "base"
        self.worktree = self.root / "worktree"

    @classmethod
    async def create(
        cls, repository: Path, output: Path, spec: RegistrationSpec, python: str, *, yes: bool = False
    ) -> RegistrationRun:
        """Validate first, then create a detached baseline and a review branch."""
        repository, output = repository.resolve(), output.resolve()
        if git(repository, "status", "--porcelain", "--untracked-files=all").strip():
            raise ContractError("The repository must be clean before register new")
        base_commit = git(repository, "rev-parse", "HEAD").strip()
        review = locate(repository, spec)
        for name in spec.tests:
            if not source_path(repository, name.split("::")[0]).is_file():
                raise ContractError(f"Missing existing CPU test: {name}")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
        root = output / run_id
        root.mkdir(parents=True)
        branch = f"registry/{spec.op}-{run_id}"
        created: list[Path] = []
        try:
            await check_unregistered(repository, spec.op, python, root / "preflight.log")
            if (
                git(repository, "rev-parse", "HEAD").strip() != base_commit
                or git(repository, "status", "--porcelain", "--untracked-files=all").strip()
            ):
                raise ContractError(
                    "Repository changed during preflight; use an ignored/external output directory"
                )
            git(repository, "worktree", "add", "--detach", str(root / "base"), base_commit)
            created.append(root / "base")
            git(repository, "worktree", "add", "-b", branch, str(root / "worktree"), base_commit)
            created.append(root / "worktree")
            atomic_json(
                root / "spec.json", {"schema_version": 1, "base_commit": base_commit, "spec": spec.as_dict()}
            )
            atomic_json(root / "sites.json", review)
            (root / "sites.md").write_text(render(review), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "repository": str(repository),
                "branch": branch,
                "base_commit": base_commit,
                "spec_hash": file_hash(root / "spec.json"),
                "base_hashes": tracked_hashes(root / "base"),
                "status": "awaiting_confirmation",
                "confirmed": yes,
                "elapsed_seconds": 0.0,
                "attempts": [],
                "pending": None,
                "agent": None,
                "test_python": python,
                "tool_runtime": runtime_identity(),
                "sites_hash": None,
                "base_overlay": {},
            }
            atomic_json(root / "manifest.json", manifest)
            return cls(root)
        except BaseException:
            for checkout in reversed(created):
                with contextlib.suppress(ContractError):
                    git(repository, "worktree", "remove", str(checkout))
            if root / "worktree" in created:
                with contextlib.suppress(ContractError):
                    git(repository, "branch", "-D", branch)
            raise

    def save(self) -> None:
        """Atomically checkpoint the external ledger for status and crash recovery."""
        atomic_json(self.root / "manifest.json", self.manifest)

    def verify(self) -> None:
        """Reject changed specifications, commits, review inputs or baseline files."""
        try:
            self._verify()
        except (ContractError, OSError, ValueError) as exc:
            self.manifest.update(status="failed", stop_reason=str(exc))
            self.save()
            raise

    def _verify(self) -> None:
        if (
            file_hash(self.root / "spec.json") != self.manifest["spec_hash"]
            or self.base_commit != self.manifest["base_commit"]
        ):
            raise ContractError("Frozen specification or BASE changed")
        if git(self.base, "rev-parse", "HEAD").strip() != self.base_commit:
            raise ContractError("BASE checkout commit changed")
        if git(self.worktree, "symbolic-ref", "--short", "HEAD").strip() != self.manifest["branch"]:
            raise ContractError("Registration worktree branch changed")
        expected_head = self.manifest.get("commit", self.base_commit)
        if git(self.worktree, "rev-parse", "HEAD").strip() != expected_head:
            raise ContractError("Agent changed worktree history; git commit is forbidden during registration")
        if (
            self.manifest.get("sites_hash")
            and file_hash(self.root / "sites.json") != self.manifest["sites_hash"]
        ):
            raise ContractError("Reviewed sites changed after the run started")
        if self.manifest.get("context_hashes"):
            actual = {
                p.relative_to(self.root / "context").as_posix(): file_hash(p)
                for p in (self.root / "context").rglob("*")
                if p.is_file()
            }
            if actual != self.manifest["context_hashes"]:
                raise ContractError("Read-only agent context changed")
        if self.manifest.get("base_overlay"):
            raise ContractError("Baseline test overlay must be recovered before checking integrity")
        if (
            tracked_hashes(self.base) != self.manifest["base_hashes"]
            or git(self.base, "status", "--porcelain", "--untracked-files=all").strip()
        ):
            raise ContractError("RUN/base was modified")

    def recover_base(self) -> None:
        """Restore only journaled test copies after a killed baseline check."""
        overlay = self.manifest.get("base_overlay", {})
        for name, expected in overlay.items():
            target = source_path(self.base, name)
            original = self.manifest["base_hashes"].get(name)
            actual = file_hash(target) if target.exists() else None
            if actual not in (original, expected):
                raise ContractError(f"Baseline overlay was modified unexpectedly: {name}")
            if original is None:
                target.unlink(missing_ok=True)
            else:
                git(self.base, "restore", f"--source={self.base_commit}", "--worktree", "--", name)
        if overlay:
            self.manifest["base_overlay"] = {}
            self.save()

    def prepare(self, agent: str, python: str | None = None) -> None:
        """Freeze reviewed call sites and build context before the first agent turn."""
        self.recover_base()
        self.verify()
        if self.manifest["sites_hash"]:
            if agent != self.manifest["agent"] or (python and python != self.manifest["test_python"]):
                raise ContractError("Agent/test_python is frozen; resume without overrides")
            return
        review = read_json(self.root / "sites.json")
        validate_review(review, locate(self.base, self.spec))
        context = self.root / "context"
        context.mkdir(exist_ok=True)
        names = {
            "AGENTS.md",
            REFERENCE,
            REGISTRY,
            "tests/test_kernel_ops.py",
            *(s.source for s in self.spec.sites),
        }
        for name in names:
            target = context / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path(self.base, name), target)
        shutil.copyfile(self.root / "sites.json", context / "sites.json")
        from .prompts import RULES

        (context / "RULES.md").write_text(RULES, encoding="utf-8")
        try:
            example = git(self.base, "show", "449ebe2", "--", "embodiinfer/")
        except ContractError:
            example = "The optional migration example 449ebe2 is unavailable in this repository.\n"
        (context / "migration.diff").write_text(example, encoding="utf-8")
        self.manifest.update(
            agent=agent,
            test_python=python or self.manifest["test_python"],
            confirmed=True,
            sites_hash=file_hash(self.root / "sites.json"),
            status="ready",
            context_hashes={
                p.relative_to(context).as_posix(): file_hash(p) for p in context.rglob("*") if p.is_file()
            },
        )
        self.save()

    def changes(self) -> list[str]:
        """Include staged, unstaged, untracked and ignored additions in scope checks."""
        tracked = git(self.worktree, "diff", "--name-only", "-z", self.base_commit, "--").split("\0")
        untracked = git(self.worktree, "ls-files", "--others", "-z").split("\0")
        # Python/test caches are controller-created and never exported.
        return sorted(
            {
                name
                for name in tracked + untracked
                if name
                and not any(
                    part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in Path(name).parts
                )
            }
        )

    def fingerprint(self) -> str:
        """Bind check evidence to exact source bytes, including new test files."""
        return digest(
            {
                name: file_hash(source_path(self.worktree, name)) if (self.worktree / name).exists() else None
                for name in self.changes()
                if name != "PLAN.md"
            }
        )

    def patch(self) -> str:
        """Produce a reviewable patch that includes allowed untracked additions."""
        patch = git(
            self.worktree,
            "diff",
            "--binary",
            self.base_commit,
            "--",
            ".",
            ":(exclude)PLAN.md",
            ":(exclude)REGISTRATION.json",
        )
        tracked = set(git(self.worktree, "ls-files", "-z").split("\0"))
        for name in self.changes():
            if name in tracked or name in ARTIFACTS:
                continue
            path = source_path(self.worktree, name)
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                lines = content.splitlines(keepends=True)
                no_newline = bool(content) and not content.endswith("\n")
                if no_newline:
                    lines[-1] += "\n"
                patch += f"diff --git a/{name} b/{name}\nnew file mode 100644\n"
                patch += "".join(
                    difflib.unified_diff(
                        [],
                        lines,
                        fromfile="/dev/null",
                        tofile=f"b/{name}",
                    )
                )
                if no_newline:
                    patch += "\\ No newline at end of file\n"
        return patch


def check_scope(run: RegistrationRun) -> dict[str, Any]:
    """C1: restrict every changed path and preserve existing test lines and bodies."""
    allowed = {
        REFERENCE,
        REGISTRY,
        "tests/test_kernel_ops.py",
        f"tests/test_registry_{run.spec.op}.py",
        *(s.source for s in run.spec.sites),
    }
    changed = run.changes()
    for name in changed:
        if name in ARTIFACTS:
            source_path(run.worktree, name)
            continue
        if name not in allowed:
            raise ContractError(f"C1: out-of-scope file: {name}")
        current = source_path(run.worktree, name)
        if not current.is_file():
            raise ContractError(f"C1: deleting files is forbidden: {name}")
        if name.startswith("tests/") and (run.base / name).exists():
            before = (run.base / name).read_text(encoding="utf-8")
            after = current.read_text(encoding="utf-8")
            diff = difflib.ndiff(before.splitlines(), after.splitlines())
            if any(line.startswith("- ") for line in diff):
                raise ContractError(f"C1: existing test lines removed or changed: {name}")
            old_tree, new_tree = ast.parse(before), ast.parse(after)
            for old in old_tree.body:
                if ast.dump(old) not in [ast.dump(new) for new in new_tree.body]:
                    raise ContractError(f"C1: existing test statements changed: {name}:{old.lineno}")
    before, after = (
        ast.parse((root / REGISTRY).read_text(encoding="utf-8")) for root in (run.base, run.worktree)
    )
    old_classes = [n for n in before.body if isinstance(n, ast.ClassDef) and n.name == "KernelSelection"]
    new_classes = [n for n in after.body if isinstance(n, ast.ClassDef) and n.name == "KernelSelection"]
    if [ast.dump(n) for n in old_classes] != [ast.dump(n) for n in new_classes]:
        raise ContractError("C1: KernelSelection must not be modified")
    return {
        "files": [name for name in changed if name not in ARTIFACTS],
        "command": ["git", "diff", "--name-status", run.base_commit],
    }
