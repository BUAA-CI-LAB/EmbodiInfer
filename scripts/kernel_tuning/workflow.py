"""Resumable, evidence-bound orchestration of existing kernel tooling programs.

Commands are explicit argv arrays executed without a shell. This module never
measures kernels or selects numerical tolerances; benchmark programs own both.
"""

from __future__ import annotations

import contextlib
import glob
import math
import os
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from string import Formatter
from typing import Any

from .artifacts import REPOSITORY, atomic_json, run_lock, tool_identity
from .artifacts import artifact_fingerprint as fingerprint
from .contracts import ContractError, digest, read_json

KINDS = ("hardware", "profile", "select", "capture", "calibrate", "tune", "install", "verify", "benchmark")
Launcher = Callable[[list[str], Path, float], int]


def _placeholders(value: str, known: set[str]) -> None:
    try:
        for _, field, spec, conversion in Formatter().parse(value):
            if field is not None and (field not in known or spec or conversion):
                raise ContractError(f"Unknown or unsupported workflow placeholder: {field}")
    except ValueError as exc:
        raise ContractError(f"Malformed workflow placeholder in {value}") from exc


def validate(config: dict[str, Any]) -> None:
    """Require ordered stages, explicit evidence and an installed-profile numerical gate."""
    if not isinstance(config, dict) or config.get("schema") != 1:
        raise ContractError("Workflow requires schema: 1")
    if not isinstance(config.get("profile"), str) or not config["profile"].isidentifier():
        raise ContractError("Workflow requires a kernel profile identifier")
    if config["profile"] in ("auto", "none", "torch"):
        raise ContractError("Workflow profile must name an installed candidate, not a reserved dispatch mode")
    if not isinstance(config.get("inputs"), dict) or not config["inputs"]:
        raise ContractError("Workflow requires explicit frozen input files/directories")
    if not all(
        isinstance(name, str) and name.isidentifier() and isinstance(path, str) and path
        for name, path in config["inputs"].items()
    ):
        raise ContractError("Workflow inputs map identifiers to paths")
    if config.get("verification_input") not in config["inputs"] or not isinstance(
        config.get("profile_artifact"), str
    ):
        raise ContractError("Workflow requires verification_input and profile_artifact bindings")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ContractError("Workflow requires stages")
    names = set()
    kinds = []
    known = {"run", "repo", "profile", *(f"input_{name}" for name in config["inputs"])}
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("kind") not in KINDS:
            raise ContractError("Unknown workflow stage kind")
        name = stage.get("name")
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ContractError("Stage names must be distinct identifiers")
        names.add(name)
        kinds.append(stage["kind"])
        for key in ("argv", "artifacts"):
            values = stage.get(key)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(v, str) and v for v in values)
            ):
                raise ContractError(f"Stage {name} requires a nonempty {key} string array")
        timeout = stage.get("timeout_seconds", 3600)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ContractError("Stage timeout must be finite and positive")
        if stage["kind"] == "verify" and not isinstance(stage.get("gate"), str):
            raise ContractError("Verification stage requires a gate report path")
        if "bindings" in stage and (
            not isinstance(stage["bindings"], dict)
            or not all(
                isinstance(k, str) and k.isidentifier() and isinstance(v, str)
                for k, v in stage["bindings"].items()
            )
        ):
            raise ContractError("Stage bindings map identifiers to single-match paths/globs")
        if "foreach" in stage and not isinstance(stage["foreach"], str):
            raise ContractError("Stage foreach must be a path/glob")
        resume_bindings = stage.get("resume_bindings", {})
        if (
            not isinstance(resume_bindings, dict)
            or any(
                not isinstance(key, str) or not key.isidentifier() or not isinstance(value, str)
                for key, value in resume_bindings.items()
            )
            or set(resume_bindings) & (known | {"item"})
        ):
            raise ContractError("Resume bindings must be new identifiers mapped to paths/globs")
        for value in resume_bindings.values():
            _placeholders(value, known)
        if "resume_argv" in stage:
            if (
                not isinstance(stage["resume_argv"], list)
                or not stage["resume_argv"]
                or not all(isinstance(arg, str) and arg for arg in stage["resume_argv"])
            ):
                raise ContractError("resume_argv must be a nonempty string array")
            for arg in stage["resume_argv"]:
                _placeholders(arg, known | set(resume_bindings) | ({"item"} if "foreach" in stage else set()))
        elif resume_bindings:
            raise ContractError("resume_bindings requires resume_argv")
        local = known | ({"item"} if "foreach" in stage else set())
        for arg in stage["argv"]:
            _placeholders(arg, local)
        if "foreach" in stage:
            _placeholders(stage["foreach"], known)
        additions = stage.get("bindings", {})
        if set(additions) & (known | {"item"}):
            raise ContractError("Stage bindings cannot replace existing workflow placeholders")
        for value in additions.values():
            _placeholders(value, known)
        known.update(additions)
        for value in stage["artifacts"]:
            _placeholders(value, known)
        if stage["kind"] == "verify":
            _placeholders(stage["gate"], known)
            _placeholders(config["profile_artifact"], known)
    if set(KINDS[1:]) - set(kinds) or kinds != sorted(kinds, key=KINDS.index):
        raise ContractError(
            "Workflow must order profile/select/capture/calibrate/tune/install/verify/benchmark"
        )


def _expand(value: str, bindings: dict[str, str]) -> str:
    try:
        return value.format_map(bindings)
    except (KeyError, ValueError) as exc:
        raise ContractError(f"Unknown or malformed workflow placeholder: {value}") from exc


def _paths(value: str, bindings: dict[str, str]) -> list[Path]:
    expanded = _expand(value, bindings)
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(bindings["run"]) / path
    matches = sorted(Path(item).absolute() for item in glob.glob(str(path)))
    if not matches:
        raise ContractError(f"Workflow artifact matched no paths: {path}")
    return matches


def create(config_path: Path, output: Path) -> dict[str, Any]:
    """Freeze a workflow and its declared inputs without executing any stage."""
    config = read_json(config_path)
    validate(config)
    if output.exists():
        raise ContractError(f"{output} already exists")
    inputs = {}
    for name, value in config["inputs"].items():
        path = Path(value)
        if not path.is_absolute():
            path = config_path.resolve().parent / path
        path = path.absolute()
        inputs[name] = {"path": str(path), "fingerprint": fingerprint(path)}
    output.mkdir(parents=True)
    manifest = {
        "schema": 1,
        "config": config,
        "config_digest": digest(config),
        "inputs": inputs,
        "tool_digest": tool_identity(),
        "status": "pending",
        "performance_claim": False,
        "stages": {stage["name"]: {"status": "pending", "attempts": 0} for stage in config["stages"]},
    }
    atomic_json(output / "workflow.json", manifest)
    return manifest


def launch(argv: list[str], log: Path, timeout: float) -> int:
    """Run a bounded stage and terminate its process tree on timeout/cancellation."""
    with log.open("wb") as handle:
        process = subprocess.Popen(
            argv,
            cwd=REPOSITORY,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    import psutil

                    with contextlib.suppress(psutil.NoSuchProcess):
                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            with contextlib.suppress(psutil.NoSuchProcess):
                                child.kill()
                        parent.kill()
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def _check_receipts(manifest: dict[str, Any]) -> None:
    for item in manifest["inputs"].values():
        if fingerprint(Path(item["path"])) != item["fingerprint"]:
            raise ContractError("Frozen workflow input changed")
    for stage in manifest["stages"].values():
        if stage["status"] == "passed":
            for path, receipt in stage["outputs"].items():
                if fingerprint(Path(path)) != receipt:
                    raise ContractError(f"Completed workflow evidence changed: {path}")


def inspect(root: Path) -> dict[str, Any]:
    """Validate a frozen plan and completed evidence without executing a stage."""
    manifest = read_json(root / "workflow.json")
    if not isinstance(manifest, dict):
        raise ContractError("Workflow manifest must be an object")
    config = manifest.get("config", {})
    validate(config)
    if digest(config) != manifest.get("config_digest") or tool_identity() != manifest.get("tool_digest"):
        raise ContractError("Workflow configuration/tooling changed; create a new workflow")
    stages = manifest.get("stages")
    if not isinstance(stages, dict) or set(stages) != {stage["name"] for stage in config["stages"]}:
        raise ContractError("Workflow stage ledger does not match the frozen plan")
    for state in stages.values():
        if not isinstance(state, dict) or state.get("status") not in (
            "pending",
            "running",
            "failed",
            "passed",
        ):
            raise ContractError("Invalid workflow stage status")
        if type(state.get("attempts")) is not int or state["attempts"] < 0:
            raise ContractError("Invalid workflow attempt count")
        if state["status"] == "passed" and (
            not isinstance(state.get("outputs"), dict) or not state["outputs"]
        ):
            raise ContractError("Completed workflow stage requires output receipts")
    _check_receipts(manifest)
    return manifest


def execute(root: Path, *, retry_stage: str | None = None, launcher: Launcher = launch) -> dict[str, Any]:
    """Resume only from intact receipts; failed/partial stages need explicit retry.

    Retrying preserves old logs and does not delete or overwrite partial outputs.
    The stage's command must support resumption or the operator must move its
    partial outputs aside. Failed numerical gates always block later stages.
    """
    root = root.resolve(strict=True)
    with run_lock(root):
        manifest = inspect(root)
        config = manifest["config"]
        if retry_stage and (
            retry_stage not in manifest["stages"]
            or manifest["stages"][retry_stage]["status"] not in ("failed", "running")
        ):
            raise ContractError("--retry-stage must name a failed/interrupted stage")
        bindings = {"run": str(root), "repo": str(REPOSITORY), "profile": config["profile"]}
        bindings.update({f"input_{name}": item["path"] for name, item in manifest["inputs"].items()})
        for stage in config["stages"]:
            name = stage["name"]
            state = manifest["stages"][name]
            if state["status"] == "passed":
                bindings.update(state.get("bindings", {}))
                continue
            if state["status"] in ("running", "failed") and retry_stage != name:
                raise ContractError(
                    f"Stage {name} needs explicit --retry-stage {name}; preserve or recover partial artifacts first"
                )
            retrying = state["status"] in ("running", "failed")
            state.update(status="running", attempts=state["attempts"] + 1, error=None)
            manifest["status"] = "running"
            atomic_json(root / "workflow.json", manifest)
            folder = root / "logs" / name / str(state["attempts"])
            folder.mkdir(parents=True)
            try:
                _check_receipts(manifest)
                action_bindings = dict(bindings)
                if retrying and "resume_argv" in stage:
                    for key, pattern in stage.get("resume_bindings", {}).items():
                        matches = _paths(pattern, bindings)
                        if len(matches) != 1:
                            raise ContractError(f"Resume binding {key} must match exactly one path")
                        action_bindings[key] = str(matches[0])
                if stage["kind"] == "verify":
                    gate_path = Path(_expand(stage["gate"], bindings))
                    if not gate_path.is_absolute():
                        gate_path = root / gate_path
                    if gate_path.exists() or glob.has_magic(str(gate_path)):
                        raise ContractError(
                            "Verification requires a fresh, explicit gate report path; recover partial reports before retry"
                        )
                items = _paths(stage["foreach"], action_bindings) if "foreach" in stage else [None]
                commands = []
                for index, item in enumerate(items):
                    local = {**action_bindings, **({"item": str(item)} if item else {})}
                    argv = stage.get("resume_argv", stage["argv"]) if retrying else stage["argv"]
                    command = [_expand(arg, local) for arg in argv]
                    commands.append(command)
                    atomic_json(folder / "commands.json", commands)
                    code = launcher(command, folder / f"{index}.log", stage.get("timeout_seconds", 3600))
                    if code:
                        raise ContractError(f"Stage {name} exited {code}; see {folder}")
                additions = {}
                for key, value in stage.get("bindings", {}).items():
                    if key in bindings or key == "item":
                        raise ContractError(f"Workflow binding cannot replace {key}")
                    matches = _paths(value, bindings)
                    if len(matches) != 1:
                        raise ContractError(f"Workflow binding {key} must match exactly one path")
                    additions[key] = str(matches[0])
                local = {**bindings, **additions}
                paths = [path for pattern in stage["artifacts"] for path in _paths(pattern, local)]
                if stage["kind"] == "verify":
                    gates = _paths(stage["gate"], local)
                    if len(gates) != 1:
                        raise ContractError("Verification requires exactly one gate report")
                    gate = read_json(gates[0])
                    profile_paths = _paths(config["profile_artifact"], local)
                    input_receipt = manifest["inputs"][config["verification_input"]]["fingerprint"]
                    if (
                        gate.get("passed") is not True
                        or gate.get("profile") != config["profile"]
                        or type(gate.get("inputs")) is not int
                        or gate["inputs"] <= 0
                        or gate.get("input_fingerprint") != input_receipt
                        or len(profile_paths) != 1
                        or gate.get("profile_fingerprint") != fingerprint(profile_paths[0])
                    ):
                        raise ContractError(
                            "Whole-model verification failed or input/profile evidence does not match"
                        )
                    paths.extend(gates)
                    paths.extend(profile_paths)
                state.update(
                    status="passed",
                    outputs={str(path): fingerprint(path) for path in paths},
                    bindings=additions,
                )
                bindings.update(additions)
                _check_receipts(manifest)
            except BaseException as exc:
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                manifest["status"] = "failed"
                atomic_json(root / "workflow.json", manifest)
                raise
            atomic_json(root / "workflow.json", manifest)
        manifest["status"] = "completed"
        atomic_json(root / "workflow.json", manifest)
        return manifest
