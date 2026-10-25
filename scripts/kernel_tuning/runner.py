"""Bounded evaluator subprocesses, independent of coding-agent transcripts."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from pathlib import Path
from typing import Any

from .artifacts import REPOSITORY, RunStore, atomic_json
from .contracts import ContractError, digest, read_json


async def _kill_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        # psutil is part of the isolated Humanize2 tooling environment.
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
    await process.wait()


async def evaluate(
    store: RunStore,
    output: Path,
    *,
    operation: str = "evaluate",
    candidate: Path | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run a trusted worker and reject stale, incomplete, or mismatched responses.

    Cancellation and timeout terminate the worker and its compiler children. A
    fresh response filename/nonce prevents a crashed worker reusing old evidence.
    """
    store.verify()
    output.mkdir(parents=True, exist_ok=True)
    cfg = store.task.settings
    baseline = store.solution_path()
    paths = {
        "baseline": baseline,
        "incumbent": store.solution_path(store.manifest["best"]),
        "candidate": candidate or baseline,
    }
    request = {
        "schema_version": 1,
        "nonce": uuid.uuid4().hex,
        "operation": operation,
        "task_dir": str(store.task.root),
        "task_digest": store.task.identity,
        "solutions": {role: str(path.resolve()) for role, path in paths.items()},
        "solution_digests": {role: digest(read_json(path)) for role, path in paths.items()},
    }
    request_path = output / "request.json"
    response_path = output / f"response-{request['nonce']}.json"
    atomic_json(request_path, request)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Both compilers and imported generated Python are confined to this run's cache.
    # This is filesystem organization, not a security sandbox.
    env["TORCH_EXTENSIONS_DIR"] = str(store.root / "cache/torch")
    env["TRITON_CACHE_DIR"] = str(store.root / "cache/triton")
    env["FIB_CACHE_PATH"] = str(store.root / "cache/flashinfer")
    env["FIB_ENABLE_APPLY"] = "0"
    env["FIB_ENABLE_TRACING"] = "0"
    argv = [
        cfg.evaluator_python,
        str(REPOSITORY / "benchmarks/kernel_tuning/evaluate.py"),
        "--request",
        str(request_path),
        "--output",
        str(response_path),
    ]
    atomic_json(
        output / "command.json",
        {"argv": argv, "timeout_seconds": timeout or cfg.search.evaluation_timeout_seconds},
    )
    process = None
    try:
        with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=output,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name != "nt",
            )
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=timeout or cfg.search.evaluation_timeout_seconds
                )
            except BaseException:
                await _kill_tree(process)
                raise
        if process.returncode != 0:
            raise ContractError(f"Evaluator failed (exit {process.returncode}); see {output / 'stderr.log'}")
        if not response_path.is_file():
            raise ContractError("Evaluator produced no structured response")
        response = read_json(response_path)
        if (
            response.get("request_digest") != digest(request)
            or response.get("task_digest") != store.task.identity
        ):
            raise ContractError("Evaluator response identity mismatch")
        if response.get("solution_digests") != request["solution_digests"]:
            raise ContractError("Evaluator measured different solution sources")
        if response.get("status") != "ok":
            raise ContractError(f"Evaluator rejected request: {response.get('error', 'unknown error')}")
        environment = response.get("environment", {})
        if not isinstance(environment.get("identity"), dict) or not environment["identity"]:
            raise ContractError("Evaluator must report hardware/software identity")
        previous = store.manifest["environment"]
        if previous and environment["identity"] != previous["identity"]:
            raise ContractError("Evaluator hardware/software environment changed; start a new run")
        for role, path in paths.items():
            if digest(read_json(path)) != request["solution_digests"][role]:
                raise ContractError("Solution changed during evaluation")
        store.verify()
        atomic_json(output / "result.json", response)
        return response
    except TimeoutError as exc:
        raise ContractError(f"Evaluator timed out; see {output}") from exc
