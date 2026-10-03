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
from .contracts import HMZ_REVISION, ContractError, TaskPackage, read_json
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


def _selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("operators", nargs="*", help="Catalog operator names (default: all)")
    parser.add_argument("--model", action="append", help="Select operators used by a model, e.g. pi05")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override tuning.yaml, e.g. evaluator_python=/abs/python or search.max_candidates=10",
    )
    parser.add_argument("--hardware-notes", type=Path, help="Target-hardware notes copied into each task")


def _generate(args: argparse.Namespace) -> None:
    from .generate import parse_overrides, render
    from .operators import select

    operators = select(args.operators or None, args.model)
    if args.list:
        for operator in operators:
            print(f"{operator.name:22} {','.join(operator.models):10} {operator.summary}")
        return
    overrides = parse_overrides(args.set)
    for operator in operators:
        task = render(
            operator,
            args.output / operator.name,
            overrides=overrides,
            hardware_notes=args.hardware_notes,
            force=args.force,
        )
        print(
            f"{operator.name}: {task.root} ({len(task.workloads)} workloads, {task.settings.precision.mode})"
        )


def _tune_all(args: argparse.Namespace) -> int:
    from . import batch
    from .generate import parse_overrides
    from .operators import select

    if args.resume:
        if args.operators or args.model or args.set or args.hardware_notes or args.agent:
            raise ContractError("--resume continues the recorded batch; omit selection, --set, and --agent")
        root = args.resume.resolve(strict=True)
    else:
        agent = args.agent or ""
        if not (args.dry_run or args.preflight_only) and "/" not in agent:
            raise ContractError("Choose an explicit Humanize2 agent: --agent harness/model:effort")
        root = batch.create(
            select(args.operators or None, args.model),
            args.output,
            agent=agent,
            overrides=parse_overrides(args.set),
            hardware_notes=args.hardware_notes,
        )
    print(f"Batch: {root}", flush=True)
    if args.dry_run:
        batch.write_summary(root)
        print((root / "summary.md").read_text(encoding="utf-8"))
        return 0
    if not args.skip_preflight:
        results = batch.preflight(root)
        items = read_json(root / "batch.json")["items"]
        for name, error in results.items():
            timings = ", ".join(
                f"{work} {us:.1f} us" for work, us in items[name].get("baseline_us", {}).items()
            )
            print(f"preflight {name}: {f'ok ({timings})' if error is None else error}", flush=True)
    if args.preflight_only:
        batch.write_summary(root)
        return (
            0
            if all(
                i["status"] != "preflight_failed" for i in read_json(root / "batch.json")["items"].values()
            )
            else 1
        )
    record = batch.execute(root)
    print((root / "summary.md").read_text(encoding="utf-8"))
    return 0 if all(item["status"] == "completed" for item in record["items"].values()) else 1


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
    generate = commands.add_parser("generate", help="Write task packages for catalog operators; no execution")
    _selection(generate)
    generate.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning/tasks")
    generate.add_argument(
        "--force", action="store_true", help="Replace previously generated task directories"
    )
    generate.add_argument("--list", action="store_true", help="List catalog operators and exit")
    tune_all = commands.add_parser("tune-all", help="Generate, preflight, and tune every selected operator")
    _selection(tune_all)
    tune_all.add_argument("--agent", help="Explicit Humanize2 harness/model:effort spec for every operator")
    tune_all.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning/batches")
    tune_all.add_argument("--resume", type=Path, metavar="BATCH", help="Continue an existing batch directory")
    mode = tune_all.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only generate and validate the tasks")
    mode.add_argument("--preflight-only", action="store_true", help="Stop after measuring the baselines")
    mode.add_argument("--skip-preflight", action="store_true", help="Let each run measure its baseline")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            _generate(args)
        elif args.command == "tune-all":
            return _tune_all(args)
        elif args.command == "check":
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
