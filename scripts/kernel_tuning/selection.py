"""Evidence-based operator selection from production GPU attribution reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .contracts import ContractError, digest, read_json


def selected_ops(selection: dict[str, Any]) -> set[str]:
    """Extract a nonempty, unique registry selection for ``capture-ops --plan``."""
    if (
        not isinstance(selection, dict)
        or selection.get("schema") != 1
        or not isinstance(selection.get("selected"), list)
    ):
        raise ContractError("Expected a schema 1 operator selection plan")
    names = [row.get("operator") for row in selection["selected"] if isinstance(row, dict)]
    if (
        len(names) != len(selection["selected"])
        or not names
        or any(not isinstance(name, str) or not name.isidentifier() for name in names)
        or len(names) != len(set(names))
    ):
        raise ContractError("Selection must contain distinct registered operator identifiers")
    return set(names)


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ContractError(f"{name} must be a finite nonnegative number")
    return float(value)


def plan(
    report: dict[str, Any],
    *,
    limit: int = 5,
    min_share: float = 0.01,
    floors: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank registered operators by GPU time minus optional measured lower bounds.

    Shares use summed kernel time, which can overlap across streams. Neither
    shares nor recoverable-time estimates are measured end-to-end savings.
    Unattributed/module regions remain explicit registration investigations.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ContractError("Selection limit must be a positive integer")
    if not 0 <= _number(min_share, "min_share") <= 1:
        raise ContractError("min_share must be between zero and one")
    total = _number(report.get("kernel_us_per_step"), "kernel_us_per_step")
    if total <= 0 or not isinstance(report.get("regions"), list):
        raise ContractError("Selection requires a nonempty production hotspot report")
    grouped: dict[str, dict[str, Any]] = {}
    investigations = []
    observed = 0.0
    for region in report["regions"]:
        path = region.get("region")
        if not isinstance(path, str):
            raise ContractError("Every hotspot region needs a path")
        us = _number(region.get("us_per_step"), "region.us_per_step")
        observed += us
        op = next((part[3:] for part in reversed(path.split("/")) if part.startswith("op:")), None)
        if op:
            row = grouped.setdefault(op, {"operator": op, "us_per_step": 0.0, "regions": []})
            row["us_per_step"] += us
            row["regions"].append(path)
        else:
            investigations.append(
                {
                    "region": path,
                    "us_per_step": us,
                    "share": us / total,
                    "action": "inspect trace/counters and register a reviewed region before tuning",
                }
            )
    if not math.isclose(observed, total, rel_tol=1e-6, abs_tol=1e-6):
        raise ContractError("Hotspot regions do not account for total kernel time")
    floors = floors or {}
    if set(floors) - grouped.keys():
        raise ContractError("Measured floors refer to operators absent from this report")
    for op, row in grouped.items():
        floor = floors.get(op)
        lower = 0.0
        if floor is not None:
            if (
                not isinstance(floor, dict)
                or not isinstance(floor.get("source"), str)
                or not floor["source"].strip()
            ):
                raise ContractError("Measured lower bounds require an evidence source")
            lower = _number(floor.get("lower_bound_us_per_step"), "lower_bound_us_per_step")
            if lower > row["us_per_step"]:
                raise ContractError("Measured lower bound exceeds observed operator time")
        row.update(
            {
                "share": row["us_per_step"] / total,
                "recoverable_us_per_step": row["us_per_step"] - lower,
                "lower_bound": floor,
                "estimate": "measured-floor estimate" if floor else "upper bound with zero floor",
            }
        )
    ranked = sorted(grouped.values(), key=lambda row: (-row["recoverable_us_per_step"], row["operator"]))
    selected = [row for row in ranked if row["share"] >= min_share and row["recoverable_us_per_step"] > 0][
        :limit
    ]
    selected_names = {row["operator"] for row in selected}
    return {
        "schema": 1,
        "report_digest": digest(report),
        "environment": report.get("environment"),
        "command": report.get("command"),
        "kernel_us_per_step": total,
        "selection": {"limit": limit, "min_kernel_share": min_share},
        "selected": selected,
        "deferred": [row for row in ranked if row["operator"] not in selected_names],
        "investigations": sorted(investigations, key=lambda row: -row["us_per_step"]),
        "caveats": [
            "Kernel-time attribution is not end-to-end latency or a speedup prediction.",
            "Fusion requires dependency/layout inspection and reviewed registry integration; no policy rewrite is authorized by this plan.",
            f"Graph attribution labelled fraction: {report.get('graph_labelled_fraction', 'unknown')}",
        ],
    }


def write(
    report: Path, output: Path, *, limit: int = 5, min_share: float = 0.01, floors: Path | None = None
) -> dict[str, Any]:
    """Save a selection artifact without launching an agent or importing a model."""
    if output.exists():
        raise ContractError(f"{output} already exists")
    result = plan(
        read_json(report), limit=limit, min_share=min_share, floors=read_json(floors) if floors else None
    )
    atomic_json(output, result)
    return result
