"""Review artifacts and explicitly requested commit/push operations."""

from __future__ import annotations

import json
from typing import Any

from scripts.kernel_tuning.contracts import ContractError, read_json

from .workspace import RegistrationRun, check_scope, git


def write_outputs(run: RegistrationRun) -> None:
    """Export the complete verified patch and CPU evidence without Git mutations."""
    run.verify()
    if run.manifest["status"] not in {"passed", "finalized"} or run.fingerprint() != run.manifest.get(
        "verified_fingerprint"
    ):
        raise ContractError("Only the unchanged, fully checked patch can be finalized")
    (run.root / "patch.diff").write_text(run.patch(), encoding="utf-8")
    review = read_json(run.root / "sites.json")
    lines = [
        f"# Register {run.spec.op}",
        "",
        run.spec.description or f"Register {run.spec.op} for {run.spec.model}.",
        "",
        f"BASE: `{run.base_commit}`",
        f"Branch: `{run.manifest['branch']}`",
        f"Attempts: {len(run.manifest['attempts'])}",
        "",
        "## Call sites",
        "",
    ]
    lines.extend(
        f"- {s['id']}: `{s['source']}::{s['function']}` — `{s['expression']}`" for s in review["sites"]
    )
    lines.extend(["", "## Preserved branches", "", *[f"- {item}" for item in review["preserve"]]])
    lines.extend(
        f"- {s['id']}: `{g['condition']}` ({g['arm']})" for s in review["sites"] for g in s["guards"]
    )
    lines.extend(["", "## Controller checks", ""])
    for attempt in run.manifest["attempts"]:
        lines.append(f"### Attempt {attempt['id']}: {attempt['status']}")
        for check in attempt["checks"]:
            lines.extend(
                [
                    "",
                    f"{check['check']}: {'passed' if check['passed'] else 'failed'}",
                    "",
                    "```json",
                    json.dumps(check, indent=2, ensure_ascii=False),
                    "```",
                ]
            )
    lines.extend(
        [
            "",
            "## Verification still required",
            "",
            "GPU whole-model bit-exact parity has NOT been run. Before merging, compare against BASE on the target GPU under proposal 0009 §6 with kernel_profile: none, identical checkpoint, dtype, attention semantics, seeds/noise, matmul precision, batch shape and decode/denoise schedule. CPU gates do not prove GPU parity.",
            "",
        ]
    )
    (run.root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(run: RegistrationRun, *, commit: bool = False, push: bool = False) -> dict[str, Any]:
    """Commit/push only with explicit flags, never merge or touch the main checkout."""
    run.recover_base()
    write_outputs(run)
    message = f"feat: route {run.spec.model} calls through the {run.spec.op} registry operator"
    if push and not commit and not run.manifest.get("commit"):
        raise ContractError("--push requires --commit or an already finalized commit")
    if commit and not run.manifest.get("commit"):
        files = check_scope(run)["files"]
        # Agent staging must not smuggle PLAN.md or other artifacts into the commit.
        staged = set(filter(None, git(run.worktree, "diff", "--cached", "--name-only", "-z").split("\0")))
        if not staged <= set(files):
            raise ContractError("Index contains files outside the verified patch")
        git(run.worktree, "add", "--", *files)
        git(run.worktree, "commit", "-m", message)
        run.manifest.update(commit=git(run.worktree, "rev-parse", "HEAD").strip(), status="finalized")
        run.save()
    if push:
        # Commit hooks may change files or the index. Never push bytes that were
        # not checked, even when commit and push were requested together.
        write_outputs(run)
        git(run.worktree, "push", "-u", "origin", run.manifest["branch"])
        run.manifest["pushed"] = True
        run.save()
    return {
        "worktree": str(run.worktree),
        "branch": run.manifest["branch"],
        "suggested_commit": message,
        "commit": run.manifest.get("commit"),
        "pushed": run.manifest.get("pushed", False),
        "patch": str(run.root / "patch.diff"),
        "summary": str(run.root / "SUMMARY.md"),
    }
