"""The KDA minimal loop as a resumable Humanize2 Flow.

Only this module needs hmz. Importing it defines a flow; executing the developer
CLI's run/resume command is what opens coding-agent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from hmz.flows import (
    Agent,
    AgentCollection,
    Budget,
    BudgetExceeded,
    EnvCollection,
    FlowContext,
    FlowParams,
    LocalEnv,
    flow,
)

from scripts.kernel_tuning.artifacts import RunStore, atomic_json
from scripts.kernel_tuning.contracts import (
    ContractError,
    promotion,
    read_json,
    tree_hashes,
    validate_measurement,
    validate_solution,
)
from scripts.kernel_tuning.prompts import implementation_prompt, plan_prompt
from scripts.kernel_tuning.runner import evaluate


class Agents(AgentCollection):
    """An existing coding agent, selected by an explicit hmz agent specification."""

    coder: Agent


class Envs(EnvCollection):
    """Humanize2's local workspace, containing only this run's scratch attempts."""

    workspace: LocalEnv


class Params(FlowParams):
    """All tunable conditions are in the snapshotted task; only its run path varies."""

    run_dir: str


@flow(agents=Agents, envs=Envs, params=Params, resumable=True, name="kernel_tuning")
async def kernel_tuning(
    task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext
) -> dict[str, Any]:
    """Plan, generate, verify, measure, and retain one candidate at a time."""
    del task  # The immutable task package supplies the complete objective.
    store = RunStore(Path(params.run_dir))
    state = ctx.state
    if state is None:
        raise ContractError("This flow requires Humanize2 resumable state")
    if "ledger" in state:
        ledger = state["ledger"]
        if ledger["run_id"] != store.manifest["run_id"] or ledger["task_digest"] != store.task.identity:
            raise ContractError("Humanize2 journal belongs to a different task/run")
        store.manifest = ledger
    store.verify()
    store.recover()
    cfg = store.task.settings
    base_elapsed = store.manifest["elapsed_seconds"]
    started = time.monotonic()

    def checkpoint() -> None:
        store.manifest["elapsed_seconds"] = base_elapsed + time.monotonic() - started
        # The framework journal is authoritative after a crash between these writes.
        state["ledger"] = copy.deepcopy(store.manifest)
        store.save()

    def remaining() -> float:
        return max(0.0, cfg.search.max_seconds - (base_elapsed + time.monotonic() - started))

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(1)
            checkpoint()

    async def measure(
        output: Path, *, operation: str = "evaluate", candidate: Path | None = None
    ) -> dict[str, Any]:
        if remaining() <= 0:
            raise TimeoutError("Search duration exhausted")
        return await evaluate(
            store,
            output,
            operation=operation,
            candidate=candidate,
            timeout=min(cfg.search.evaluation_timeout_seconds, remaining()),
        )

    store.manifest.update(status="running", stop_reason=None)
    checkpoint()
    ticker = asyncio.create_task(heartbeat())
    try:
        # Re-probe on every resume before any agent or kernel is allowed to run.
        report = await measure(store.root / "preflight", operation="probe")
        store.manifest["environment"] = report["environment"]
        checkpoint()
        if not store.manifest.get("baseline_verified"):
            report = await measure(store.root / "baseline/evaluation")
            validate_measurement(store.task, report["measurement"])
            store.manifest["baseline_verified"] = True
            checkpoint()
        while True:
            reason = None
            if len(store.manifest["attempts"]) >= cfg.search.max_candidates:
                reason = "max_candidates"
            elif store.manifest["streak"] >= cfg.search.patience:
                reason = "patience"
            elif remaining() <= 0:
                reason = "max_seconds"
            if reason:
                store.manifest.update(status="completed", stop_reason=reason)
                break
            candidate, archive, scratch = store.begin_attempt(persist=checkpoint)
            checkpoint()
            solution = None
            session = None
            try:
                work_env = await envs["workspace"].derive_subdir(subdir=candidate)
                session = await agents["coder"].spawn(env=work_env)
                for stage, prompt in (
                    ("plan", plan_prompt(store.task)),
                    ("implement", implementation_prompt(store.task)),
                ):
                    store.verify()
                    if remaining() <= 0:
                        raise TimeoutError("Search duration exhausted")
                    seconds = min(cfg.search.agent_timeout_seconds, remaining())
                    (archive / f"{stage}-prompt.txt").write_text(prompt, encoding="utf-8")
                    reply = await asyncio.wait_for(
                        agents["coder"].run(
                            prompt,
                            session=session,
                            budget=Budget(duration=timedelta(seconds=seconds), graceful=False),
                        ),
                        timeout=seconds,
                    )
                    (archive / f"{stage}-reply.txt").write_text(reply, encoding="utf-8")
                    plan = scratch / "PLAN.md"
                    if (
                        not plan.is_file()
                        or plan.is_symlink()
                        or not plan.read_text(encoding="utf-8").strip()
                    ):
                        raise ContractError("Agent did not write a nonempty PLAN.md")
                    (archive / "PLAN.md").write_bytes(plan.read_bytes())
                del session
                store.verify()
                if tree_hashes(scratch / "task") != store.task.hashes:
                    raise ContractError("Agent modified its frozen task copy")
                solution_file = scratch / "solution.json"
                if solution_file.is_symlink():
                    raise ContractError("Candidate solution must be a regular file")
                solution = validate_solution(read_json(solution_file), store.task)
                atomic_json(archive / "solution.json", solution)
                report = await measure(archive / "evaluation", candidate=archive / "solution.json")
                decision = promotion(store.task, report["measurement"])
                status = "promoted" if decision["promoted"] else "rejected"
                store.finish_attempt(candidate, status=status, decision=decision, solution=solution)
                checkpoint()
                if cfg.profile:
                    # Profiling is evidence only, outside promotion and performance timing.
                    try:
                        await measure(
                            archive / "profile", operation="profile", candidate=archive / "solution.json"
                        )
                    except Exception as exc:
                        atomic_json(archive / "profile/error.json", {"error": str(exc)})
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                # Per-turn budget exhaustion is an attempt failure. The whole-run
                # duration is checked separately, so resume cannot reset budgets.
                store.finish_attempt(
                    candidate,
                    status="failed",
                    decision={"promoted": False, "reasons": [f"{type(exc).__name__}: {exc}"]},
                    solution=solution,
                )
                checkpoint()
                # A changed reference/tool/baseline is a run error, not another candidate.
                store.verify()
            finally:
                session = None  # hmz closes sessions when the last holder lets go.
        checkpoint()
        return copy.deepcopy(store.manifest)
    except (BudgetExceeded, TimeoutError):
        store.recover()
        store.manifest.update(status="completed", stop_reason="max_seconds")
        raise
    except BaseException as exc:
        store.recover()
        store.manifest.update(
            status="interrupted"
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            else "failed",
            stop_reason=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
        checkpoint()
