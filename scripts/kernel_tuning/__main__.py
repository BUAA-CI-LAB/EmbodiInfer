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
from .discovery.contracts import ModelCapture
from .runner import evaluate


def _execute(store: RunStore, *, resume: bool) -> dict:
    if "replay.json" in store.task.hashes:
        from .batch import verify_model_run

        verify_model_run(store)
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
    parser.add_argument(
        "--catalog", action="store_true", help="Use the legacy hand-maintained operator catalog"
    )
    parser.add_argument("--model", action="append", help="One canonical target policy name, e.g. pi05")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override tuning.yaml, e.g. evaluator_python=/abs/python or search.max_candidates=10",
    )
    parser.add_argument("--hardware-notes", type=Path, help="Target-hardware notes copied into each task")
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument(
        "--captured",
        type=Path,
        help="Completed model capture directory (legacy JSONL only with --catalog)",
    )
    shapes.add_argument("--estimated", action="store_true", help="Use the catalog's estimated shapes")


def _model_capture(args: argparse.Namespace) -> ModelCapture:
    if not args.model or len(args.model) != 1:
        raise ContractError(
            "Model tuning requires exactly one --model; use --catalog for legacy operator selection"
        )
    if (
        args.operators
        or args.estimated
        or getattr(args, "force", False)
        or getattr(args, "skip_preflight", False)
    ):
        raise ContractError(
            "Model tuning requires complete coverage: no operator filter, --estimated, --force, or --skip-preflight"
        )
    if args.captured is None:
        raise ContractError("Model tuning requires --captured MODEL_CAPTURE; run capture --model first")
    return ModelCapture.load(args.captured, args.model[0], require_ready=not getattr(args, "list", False))


def _generate(args: argparse.Namespace) -> None:
    from .capture import apply_file
    from .generate import parse_overrides, render
    from .operators import select

    if not args.catalog:
        from .discovery.tasks import render_model

        capture = _model_capture(args)
        if args.list:
            for op in capture.operators:
                print(
                    f"{op['status']:20} {op['name']} ({len(op['workloads'])} cases) {op.get('reason') or ''}"
                )
            return
        tasks = render_model(
            capture, args.output, overrides=parse_overrides(args.set), hardware_notes=args.hardware_notes
        )
        for task in tasks:
            print(f"{task.definition['description']}: {task.root} ({len(task.workloads)} workloads)")
        return

    operators, notes = apply_file(
        select(args.operators or None, args.model), args.captured, estimated=args.estimated
    )
    for note in notes:
        print(note)
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
    from .capture import apply_file
    from .generate import parse_overrides
    from .operators import select

    if args.resume:
        if (
            args.operators
            or args.model
            or args.set
            or args.hardware_notes
            or args.agent
            or args.captured
            or args.estimated
            or args.catalog
        ):
            raise ContractError("--resume continues the recorded batch; omit selection, --set, and --agent")
        root = args.resume.resolve(strict=True)
    else:
        agent = args.agent or ""
        if not (args.dry_run or args.preflight_only) and "/" not in agent:
            raise ContractError("Choose an explicit Humanize2 agent: --agent harness/model:effort")
        options = {
            "agent": agent,
            "overrides": parse_overrides(args.set),
            "hardware_notes": args.hardware_notes,
        }
        if args.catalog:
            operators, notes = apply_file(
                select(args.operators or None, args.model), args.captured, estimated=args.estimated
            )
            for note in notes:
                print(note, flush=True)
            root = batch.create(operators, args.output, **options)
        else:
            root = batch.create_model(_model_capture(args), args.output, **options)
    if args.skip_preflight and read_json(root / "batch.json").get("model"):
        raise ContractError("Model batches cannot skip the all-baseline preflight")
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
    generate = commands.add_parser("generate", help="Write every discovered model task; no execution")
    _selection(generate)
    generate.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning/tasks")
    generate.add_argument(
        "--force", action="store_true", help="Replace previously generated task directories"
    )
    generate.add_argument("--list", action="store_true", help="List discovered operators and coverage gaps")
    tune_all = commands.add_parser("tune-all", help="Generate, preflight, and tune every selected operator")
    _selection(tune_all)
    tune_all.add_argument("--agent", help="Explicit Humanize2 harness/model:effort spec for every operator")
    tune_all.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning/batches")
    tune_all.add_argument("--resume", type=Path, metavar="BATCH", help="Continue an existing batch directory")
    mode = tune_all.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only generate and validate the tasks")
    mode.add_argument("--preflight-only", action="store_true", help="Stop after measuring the baselines")
    mode.add_argument("--skip-preflight", action="store_true", help="Let each run measure its baseline")
    capture = commands.add_parser(
        "capture", help="Record production kernel shapes while a model script runs (model runtime)"
    )
    capture.add_argument("operators", nargs="*", help="Catalog operators to record (default: all)")
    capture.add_argument("--catalog", action="store_true", help="Record legacy catalog-only JSONL")
    capture.add_argument("--model", help="Canonical policy name constructed by the preparation script")
    capture.add_argument(
        "--fixture-bytes", type=int, default=64 * 1024 * 1024, help="Maximum stored input fixture bytes"
    )
    capture.add_argument(
        "--output", type=Path, required=True, help="New model capture directory (JSONL for --catalog)"
    )
    capture.add_argument("script", nargs=argparse.REMAINDER, help="-- SCRIPT [ARGS] or -- -m MODULE [ARGS]")
    calibration = commands.add_parser(
        "calibrate", help="Freeze baseline/FP64 numerical bounds in the model runtime before generation"
    )
    calibration.add_argument("--captured", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True, help="New calibrated capture directory")
    skeleton = commands.add_parser(
        "skeleton", help="Random-weight copy of a Hugging Face checkpoint for shape capture (model runtime)"
    )
    skeleton.add_argument("repo", help="Model repository, e.g. org/name; HF_ENDPOINT selects a mirror")
    skeleton.add_argument("--revision", required=True, help="Immutable commit to copy")
    skeleton.add_argument("--output", type=Path, required=True)
    raw = sys.argv[1:] if argv is None else argv
    if "--" in raw and raw[0] == "capture":
        split = raw.index("--")
        raw, command = raw[:split], raw[split + 1 :]
    else:
        command = None
    args = parser.parse_args(raw)
    try:
        if args.command == "capture":
            if args.catalog:
                from .capture import run as run_capture
                from .operators import select

                if args.model:
                    raise ContractError("Legacy catalog capture does not bind a model; omit --model")
                rows = run_capture(command or args.script, args.output, select(args.operators or None))
                calls = sum(row["count"] for row in rows if "axes" in row)
                print(
                    f"Captured {calls} calls in {sum('axes' in row for row in rows)} cases -> {args.output}"
                )
                for row in rows:
                    if "skipped" in row:
                        print(f"  skipped {row['count']} {row['operator']} calls: {row['skipped']}")
            else:
                from .discovery.recorder import run as run_model_capture

                if not args.model or args.operators:
                    raise ContractError(
                        "Capture requires one --model and no operator filter; use --catalog for legacy capture"
                    )
                result = run_model_capture(
                    command or args.script, args.output, args.model, fixture_bytes=args.fixture_bytes
                )
                print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "calibrate":
            from benchmarks.kernel_tuning.calibration import calibrate

            result = calibrate(ModelCapture.load(args.captured), args.output)
            print(json.dumps({"capture": str(result.root), "identity": result.identity}, indent=2))
        elif args.command == "skeleton":
            from .skeleton import build

            summary = build(args.repo, args.revision, args.output)
            tensors = sum(summary["synthesized"].values())
            print(
                f"{args.output}: {len(summary['copied'])} files copied, "
                f"{tensors} random tensors in {len(summary['synthesized'])} safetensors files"
            )
        elif args.command == "generate":
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
            model_batch = None
            if "replay.json" in task.hashes:
                from .batch import verify_model_task

                model_batch = verify_model_task(task)
            store = RunStore.create(task, args.output, agent)
            if model_batch is not None:
                store.manifest["model_batch"] = str(model_batch)
                item = read_json(model_batch / "batch.json")["items"][task.root.name]
                store.manifest["environment"] = item["preflight_environment"]
                store.save()
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
