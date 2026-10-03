"""Developer commands for checking, running, resuming, and exporting tuning jobs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from .artifacts import REPOSITORY, RunStore, atomic_json, run_lock, runtime_identity
from .contracts import HMZ_REVISION, ContractError, TaskPackage
from .runner import evaluate


def _execute(store: RunStore, *, resume: bool) -> dict:
    if os.name != "posix":
        raise ContractError(
            "This pinned Humanize2 release requires POSIX; run tuning on the Linux target machine"
        )
    if sys.version_info < (3, 12):
        raise ContractError("Humanize2 requires a separate Python 3.12+ tooling environment")
    try:
        from hmz.flows import BudgetExceeded
        from hmz.sdk import Hmz
    except ImportError as exc:
        raise ContractError("Install the optional scripts/kernel_tuning tooling environment first") from exc
    with run_lock(store.root):
        store.verify()
        if runtime_identity() != store.manifest["tool_runtime"]:
            raise ContractError("Humanize2/Python tooling environment changed; start a new run")
        remaining = store.task.settings.search.max_seconds - store.manifest["elapsed_seconds"]
        if remaining <= 0 or store.manifest["status"] == "completed":
            raise ContractError("Run budget already exhausted; start a new run with a new task contract")
        epic = store.root / "humanize.json"
        running = Hmz(workspace=store.root / "workspace").run(
            flow=Path(__file__).with_name("flow.py"),
            task=f"Optimize {store.task.definition['name']} under its frozen task contract",
            agents={"coder": store.manifest["agent"]},
            params={"run_dir": str(store.root)},
            budget={"duration": remaining, "graceful": False},
            # A killed invocation may be newer than humanize.json. The dedicated
            # workspace lets hmz select its newest journal instead of a stale path.
            resume=resume,
        )
        try:
            return running.run()
        except BudgetExceeded:
            stopped = RunStore(store.root)
            stopped.recover()
            stopped.manifest.update(status="completed", stop_reason="max_seconds")
            stopped.save()
            return stopped.manifest
        finally:
            archived = None
            if running.epic:
                archived = store.root / "humanize" / running.epic.name
                archived.mkdir(parents=True, exist_ok=True)
                # Preserve flow/resume journals, without copying agent account
                # directories. CLI-managed sessions stay at the recorded epic.
                for path in running.epic.glob("*.jsonl"):
                    if path.is_file() and not path.is_symlink():
                        shutil.copyfile(path, archived / path.name)
            atomic_json(
                epic,
                {
                    "epic": str(running.epic) if running.epic else None,
                    "archived": str(archived) if archived else None,
                    "expected_hmz_revision": HMZ_REVISION,
                },
            )


def main(argv: list[str] | None = None) -> int:
    """Run only the requested command; checking/status/export never invoke an agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validate a task without executing its code")
    check.add_argument("task", type=Path)
    check.add_argument(
        "--environment", action="store_true", help="Also probe the selected evaluator environment (no tuning)"
    )
    run = commands.add_parser("run", help="Start real agent-driven tuning on this machine")
    run.add_argument("task", type=Path)
    run.add_argument("--agent", help="Explicit Humanize2 harness/model:effort spec; overrides tuning.yaml")
    run.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning")
    for name in ("resume", "status"):
        commands.add_parser(name).add_argument("run", type=Path)
    export = commands.add_parser("export", help="Export best verified source and evidence; no execution")
    export.add_argument("run", type=Path)
    export.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            task = TaskPackage.load(args.task)
            result = {
                "task": task.definition["name"],
                "digest": task.identity,
                "workloads": len(task.workloads),
                "validation": "structural; no task code executed",
            }
            if args.environment:
                import tempfile

                with tempfile.TemporaryDirectory(prefix="kernel-preflight-") as directory:
                    store = RunStore.create(task, Path(directory), agent="")
                    result["environment"] = asyncio.run(
                        evaluate(store, store.root / "probe", operation="probe")
                    )["environment"]
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "run":
            task = TaskPackage.load(args.task)
            agent = args.agent or task.settings.agent
            if not agent or "/" not in agent:
                raise ContractError("Choose an explicit Humanize2 agent: --agent harness/model:effort")
            store = RunStore.create(task, args.output, agent)
            print(f"Run archive: {store.root}", flush=True)
            print(json.dumps(_execute(store, resume=False), indent=2, ensure_ascii=False))
        else:
            store = RunStore(args.run)
            if args.command == "resume":
                print(json.dumps(_execute(store, resume=True), indent=2, ensure_ascii=False))
            elif args.command == "status":
                print(json.dumps(store.manifest, indent=2, ensure_ascii=False))
            else:
                with run_lock(store.root):
                    print(store.export(args.destination))
        return 0
    except (ContractError, OSError, ValueError, ImportError) as exc:
        print(f"kernel_tuning: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; the run archive can be resumed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
