"""Resumable Humanize2 plan/implementation/check/repair loop for registration."""

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

from scripts.kernel_tuning.artifacts import atomic_json
from scripts.kernel_tuning.contracts import ContractError
from scripts.kernel_tuning.registration.checks import run_checks
from scripts.kernel_tuning.registration.finalize import write_outputs
from scripts.kernel_tuning.registration.prompts import prompt
from scripts.kernel_tuning.registration.workspace import RegistrationRun, source_path


class Agents(AgentCollection):
    """One explicitly selected coding-agent harness."""

    coder: Agent


class Envs(EnvCollection):
    """Run directory; only its worktree child is given to the coding agent."""

    workspace: LocalEnv


class Params(FlowParams):
    """All mutable progress is in the run; the flow accepts only its identity."""

    run_dir: str


@flow(agents=Agents, envs=Envs, params=Params, resumable=True, name="operator_registration")
async def operator_registration(
    task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext
) -> dict[str, Any]:
    """Keep a session across repairs, restarting planning after two equal failures."""
    del task
    run = RegistrationRun(Path(params.run_dir))
    run.recover_base()  # An interrupted C5 overlay lives in the external ledger.
    state = ctx.state
    if state is None:
        raise ContractError("Registration requires resumable Humanize2 state")
    if "ledger" in state:
        ledger = state["ledger"]
        if any(ledger[k] != run.manifest[k] for k in ("run_id", "spec_hash", "base_commit", "sites_hash")):
            raise ContractError("Humanize2 journal belongs to a different registration")
        ledger["elapsed_seconds"] = max(ledger["elapsed_seconds"], run.manifest["elapsed_seconds"])
        ledger["base_overlay"] = {}
        run.manifest = ledger
    cfg = run.spec.budget
    elapsed, started = run.manifest["elapsed_seconds"], time.monotonic()

    def checkpoint() -> None:
        run.manifest["elapsed_seconds"] = elapsed + time.monotonic() - started
        state["ledger"] = copy.deepcopy(run.manifest)
        run.save()

    def remaining() -> float:
        return max(0.0, cfg.max_seconds - elapsed - (time.monotonic() - started))

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(1)
            checkpoint()

    ticker = None
    session = None
    try:
        run.verify()
        if run.manifest["status"] in {"passed", "finalized", "exhausted", "failed"}:
            return copy.deepcopy(run.manifest)
        if not run.manifest["sites_hash"] or not run.manifest["confirmed"]:
            raise ContractError("Call sites must be confirmed before starting an agent")
        work_env = await envs["workspace"].derive_subdir(subdir="worktree")
        if Path(str(work_env.workdir)).resolve() != run.worktree.resolve():
            raise ContractError("Humanize2 LocalEnv must be rooted at RUN")
        run.manifest["status"] = "running"
        checkpoint()
        ticker = asyncio.create_task(heartbeat())
        while True:
            run.verify()
            if remaining() <= 0 or (
                not run.manifest["pending"] and len(run.manifest["attempts"]) >= cfg.max_attempts
            ):
                run.manifest.update(
                    status="exhausted", stop_reason="max_seconds" if remaining() <= 0 else "max_attempts"
                )
                break
            if run.manifest["pending"]:
                attempt = run.manifest["attempts"][-1]
            else:
                previous = run.manifest["attempts"]
                restart = run.manifest.get("failure_streak", 0) >= 2
                stage = "plan" if not previous or restart else "repair"
                if restart:
                    session = None
                    run.manifest.update(failure_streak=0, last_failure=None)
                attempt = {
                    "id": f"{len(previous) + 1:04d}",
                    "status": "running",
                    "stage": stage,
                    "checks": [],
                }
                previous.append(attempt)
                run.manifest["pending"] = attempt["id"]
                checkpoint()
            archive = run.root / "attempts" / attempt["id"]
            archive.mkdir(parents=True, exist_ok=True)
            feedback = None
            try:
                while attempt["stage"] != "checks":
                    stage = attempt["stage"]
                    run.verify()
                    if remaining() <= 0:
                        raise TimeoutError("Registration duration exhausted")
                    if session is None:
                        session = await agents["coder"].spawn(env=work_env)
                    previous_feedback = run.manifest.get("feedback")
                    instruction = prompt(run, stage, previous_feedback)
                    (archive / f"{stage}-prompt.txt").write_text(instruction, encoding="utf-8")
                    seconds = min(cfg.agent_timeout_seconds, remaining())
                    reply = await asyncio.wait_for(
                        agents["coder"].run(
                            instruction,
                            session=session,
                            budget=Budget(duration=timedelta(seconds=seconds), graceful=False),
                        ),
                        timeout=seconds,
                    )
                    (archive / f"{stage}-reply.txt").write_text(reply, encoding="utf-8")
                    run.verify()
                    plan = source_path(run.worktree, "PLAN.md")
                    if not plan.is_file() or not plan.read_text(encoding="utf-8").strip():
                        raise ContractError("Agent did not write a nonempty PLAN.md")
                    (archive / "PLAN.md").write_bytes(plan.read_bytes())
                    attempt["stage"] = "implement" if stage == "plan" else "checks"
                    checkpoint()
                agent_record = source_path(run.worktree, "REGISTRATION.json")
                if agent_record.is_file():
                    (archive / "REGISTRATION.json").write_bytes(agent_record.read_bytes())
                attempt["checks"], feedback = await run_checks(run, archive, seconds=remaining())
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                run.recover_base()
                run.verify()  # Immutable-input damage is fatal, never agent repair feedback.
                feedback = {
                    "check": "agent",
                    "summary": f"{type(exc).__name__}: {exc}"[:12000],
                    "details": {},
                }
                atomic_json(archive / "feedback.json", feedback)
                session = None
            finally:
                (archive / "patch.diff").write_text(run.patch(), encoding="utf-8")
            attempt.update(
                status="failed" if feedback else "passed",
                failure=feedback["check"] if feedback else None,
                stage="done",
            )
            run.manifest.update(pending=None, feedback=feedback)
            if feedback:
                same = feedback["check"] == run.manifest.get("last_failure")
                run.manifest.update(
                    failure_streak=run.manifest.get("failure_streak", 0) + 1 if same else 1,
                    last_failure=feedback["check"],
                )
            if feedback is None:
                run.verify()
                run.manifest.update(status="passed", verified_fingerprint=run.fingerprint())
                checkpoint()
                write_outputs(run)
                break
            checkpoint()
        checkpoint()
        return copy.deepcopy(run.manifest)
    except (BudgetExceeded, TimeoutError):
        run.manifest.update(status="exhausted", stop_reason="max_seconds")
        raise
    except BaseException as exc:
        run.manifest.update(
            status="interrupted"
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            else "failed",
            stop_reason=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if ticker:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
        session = None
        checkpoint()
