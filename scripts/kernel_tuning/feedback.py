"""Bounded, evidence-backed context for each independent coding-agent attempt."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .contracts import ContractError, read_json


def log_tail(path: Path, *, limit: int = 12000) -> str:
    """Read at most the last ``limit`` bytes of a compiler/evaluator log."""
    if not path.is_file() or path.is_symlink():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read(limit).decode("utf-8", errors="replace")


def diagnostics(
    archive: Path,
    *,
    measurement: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build controller-owned feedback; profiling never changes the numerical gate."""
    logs = {}
    for folder in ("evaluation", "profile"):
        for name in ("stdout.log", "stderr.log"):
            if tail := log_tail(archive / folder / name):
                logs[f"{folder}/{name}"] = tail

    def recent(path: Path) -> int:
        return path.stat().st_mtime_ns

    for path in sorted((archive / "profile").glob("*/*/worker.log"), key=recent, reverse=True)[:4]:
        if tail := log_tail(path, limit=4000):
            logs[path.relative_to(archive).as_posix()] = tail
    for path in sorted((archive / "profile").glob("*/*/metrics.stderr.log"), key=recent, reverse=True)[:4]:
        if tail := log_tail(path, limit=4000):
            logs[path.relative_to(archive).as_posix()] = tail
    return {"measurement": measurement, "profile": profile, "error": error, "logs": logs}


def verified_diagnostics(archive: Path, attempt: dict[str, Any]) -> dict[str, Any] | None:
    """Refuse modified diagnostic evidence before exposing it to the next agent."""
    name = "diagnostics.json"
    expected = attempt.get("evidence_hashes", {}).get(name)
    if expected is None:  # Existing archives predate diagnostic feedback.
        return None
    path = archive / name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ContractError("Archived diagnostic evidence changed")
    return read_json(path)


def history(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize every past decision, including attempts outside the source window."""
    return {
        "attempts": [
            {key: item.get(key) for key in ("id", "status", "solution_digest", "decision")}
            for item in attempts
            if item["status"] != "running"
        ],
        "guidance": "Use measured failures to choose a distinct hypothesis; unavailable counters are not zero.",
    }
