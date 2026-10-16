"""The register command group; inspection never starts a coding agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from scripts.kernel_tuning.artifacts import (
    REPOSITORY,
    UNFENCED_HELP,
    atomic_json,
    run_lock,
    runtime_identity,
    warn_unfenced,
)
from scripts.kernel_tuning.contracts import ContractError

from .finalize import finalize
from .spec import load_spec
from .workspace import RegistrationRun, check_unregistered


def add_parser(commands: Any) -> None:
    """Add a distinct namespace without changing existing tuning commands."""
    parser = commands.add_parser("register", help="Register one operator through a CPU-checked agent loop")
    actions = parser.add_subparsers(dest="registration_command", required=True)
    new = actions.add_parser("new", help="Create worktrees and stop for call-site review")
    new.add_argument("--spec", type=Path)
    new.add_argument("--model")
    new.add_argument("--op")
    new.add_argument("--site", action="append", default=[])
    new.add_argument(
        "--repository", type=Path, default=REPOSITORY, help="Clean model checkout to register into"
    )
    new.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning/register")
    new.add_argument("--set", action="append", default=[], metavar="test_python=/abs/python")
    new.add_argument(
        "--yes", action="store_true", help="Record call-site confirmation without an interactive prompt"
    )
    run = actions.add_parser("run", help="Confirm reviewed sites and start plan/implement/check/repair")
    run.add_argument("run", type=Path)
    run.add_argument("--agent", required=True)
    run.add_argument("--set", action="append", default=[], metavar="test_python=/abs/python")
    run.add_argument("--yes", action="store_true", help="Explicitly confirm the reviewed sites")
    run.add_argument("--unfenced", action="store_true", help=UNFENCED_HELP)
    for name in ("resume", "status"):
        actions.add_parser(name).add_argument("run", type=Path)
    finish = actions.add_parser("finalize", help="Print verified branch/patch; optionally commit or push")
    finish.add_argument("run", type=Path)
    finish.add_argument("--commit", action="store_true")
    finish.add_argument("--push", action="store_true")


def test_python(settings: list[str]) -> str | None:
    """Accept only an existing absolute interpreter path, never shell fragments."""
    if not settings:
        return None
    if len(settings) != 1 or not settings[0].startswith("test_python="):
        raise ContractError("The only register --set override is test_python=/absolute/python")
    value = settings[0].split("=", 1)[1]
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ContractError("test_python must be an existing absolute executable path")
    return str(path)  # Do not resolve a venv's python symlink into its base interpreter.


def execute(run: RegistrationRun, *, resume: bool) -> dict[str, Any]:
    """Start the fixed Humanize2 flow in its POSIX tooling environment."""
    if os.name != "posix":
        raise ContractError("Humanize2 registration run/resume requires POSIX; use the Linux target host")
    if sys.version_info < (3, 12):
        raise ContractError("Humanize2 requires a separate Python 3.12+ tooling environment")
    from hmz.flows import BudgetExceeded
    from hmz.sdk import Hmz

    run.recover_base()
    run.verify()
    if runtime_identity() != run.manifest["tool_runtime"]:
        raise ContractError("Tool runtime changed; start a new registration run")
    remaining = run.spec.budget.max_seconds - run.manifest["elapsed_seconds"]
    if run.manifest["status"] in {"passed", "finalized", "failed", "exhausted"}:
        return run.manifest
    if remaining <= 0:
        run.manifest.update(status="exhausted", stop_reason="max_seconds")
        run.save()
        return run.manifest
    warn_unfenced(run.manifest)
    # LocalEnv cannot derive an absolute path or '..'. Root it at RUN, then give
    # only RUN/worktree to the agent. Archive framework journals in workspace/.
    running = Hmz(workspace=run.root).run(
        flow=Path(__file__).with_name("flow.py"),
        task=f"Register {run.spec.op} and route its call sites through the registry",
        agents={"coder": run.manifest["agent"]},
        params={"run_dir": str(run.root)},
        budget={"duration": remaining, "graceful": False},
        resume=resume,
    )
    try:
        return running.run()
    except BudgetExceeded:
        stopped = RegistrationRun(run.root)
        stopped.recover_base()
        stopped.manifest.update(status="exhausted", stop_reason="max_seconds")
        stopped.save()
        return stopped.manifest
    finally:
        archived = None
        if running.epic:
            archived = run.root / "workspace" / running.epic.name
            archived.mkdir(parents=True, exist_ok=True)
            for path in running.epic.glob("*.jsonl"):
                if path.is_file() and not path.is_symlink():
                    shutil.copyfile(path, archived / path.name)
        atomic_json(
            run.root / "humanize.json",
            {
                "epic": str(running.epic) if running.epic else None,
                "archived": str(archived) if archived else None,
            },
        )


def dispatch(args: argparse.Namespace) -> int:
    """Perform one register action, locking every mutation of an existing run."""
    action = args.registration_command
    if action == "new":
        spec = load_spec(args.spec, model=args.model, op=args.op, sites=args.site)
        python = test_python(args.set) or sys.executable
        run = asyncio.run(RegistrationRun.create(args.repository, args.output, spec, python, yes=args.yes))
        print((run.root / "sites.md").read_text(encoding="utf-8"))
        print(f"Run: {run.root}")
        return 0
    run = RegistrationRun(args.run)
    if action == "status":
        print(json.dumps(run.manifest, indent=2, ensure_ascii=False))
        return 0
    with run_lock(run.root):
        run = RegistrationRun(args.run)  # Reload after acquiring the writer lock.
        if action == "finalize":
            result = finalize(run, commit=args.commit, push=args.push)
        else:
            if action == "run":
                if not args.agent or "/" not in args.agent:
                    raise ContractError("Choose --agent HARNESS/MODEL:EFFORT explicitly")
                if run.manifest["sites_hash"]:
                    raise ContractError("Registration already started; use register resume RUN")
                python = test_python(args.set)
                # Invoking `run` is the documented confirmation after reviewing sites.md.
                run.recover_base()
                run.verify()
                asyncio.run(
                    check_unregistered(
                        run.base,
                        run.spec.op,
                        python or run.manifest["test_python"],
                        run.root / "test-runtime.log",
                    )
                )
                run.prepare(args.agent, python, unfenced=args.unfenced)
            elif not run.manifest["sites_hash"]:
                raise ContractError("Use register run RUN --agent ... to confirm the call sites first")
            try:
                result = execute(run, resume=action == "resume")
            except ContractError as exc:
                # Platform/dependency issues leave a prepared run available on its host.
                if os.name == "posix":
                    stopped = RegistrationRun(run.root)
                    stopped.manifest.update(status="failed", stop_reason=str(exc))
                    stopped.save()
                raise
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if action == "finalize" or result["status"] in {"passed", "finalized"} else 1
