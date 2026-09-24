"""Compare complete replays; navigation success requires a separate closed-loop run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# Optimization controls may differ; the workload and generation contract may not.
CONDITIONS = (
    "checkpoint_revision",
    "dtype",
    "seed",
    "warmup_calls",
    "repeats",
    "max_new_tokens",
    "max_context",
    "do_sample",
    "repetition_penalty",
    "max_steps_per_episode",
    "cpu_threads",
)
OUTPUT_FIELDS = (
    "token_ids",
    "output_sha256",
    "action_mask",
    "parsed_valid",
    "stop_reason",
    "cache_length",
    "generated_tokens",
    "action_slots",
)
EXPECTED_CALLS = {"R2R": 2997, "RxR": 3879}


def indexed_rows(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Reject duplicate identities instead of silently dropping repeated rows."""
    rows = {}
    for row in report["rows"]:
        if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
            raise ValueError("each measured latency must be finite and positive")
        key = (row["sample_id"], row["repeat"])
        if key in rows:
            raise ValueError(f"duplicate measured call: {key}")
        rows[key] = row
    return rows


def compare(
    reference: dict[str, Any], candidate: dict[str, Any], *, accuracy_contract: str = "exact"
) -> dict[str, Any]:
    """Compare full workloads under exact or externally evaluated task accuracy.

    ``task-success`` retains output differences as diagnostics, but cannot decide
    admission from recorded observations. Its admission result remains unknown
    until the two policies have been evaluated in a closed-loop simulator.
    """
    if accuracy_contract not in ("exact", "task-success"):
        raise ValueError("accuracy_contract must be exact or task-success")
    name = reference["dataset"]["name"]
    if name not in EXPECTED_CALLS or candidate["dataset"]["name"] != name:
        raise ValueError("reports must describe the same supported navigation dataset")
    for report in (reference, candidate):
        if report["model"] != "activevln" or report["batch_size"] != 1:
            raise ValueError("expected B=1 ActiveVLN reports")
        config = report["config"]
        if config["max_steps_per_episode"] is not None or config["repeats"] != 1:
            raise ValueError("admission requires one complete replay, without a frame limit")
        if len(report["episodes"]) != 48 or len(set(report["episodes"])) != 48:
            raise ValueError("admission requires 48 distinct episodes")
        if len(report["rows"]) != EXPECTED_CALLS[name]:
            raise ValueError("report does not contain every frame of the 48 selected trajectories")
        if report["metrics"]["calls"] != len(report["rows"]):
            raise ValueError("summary count does not match raw rows")
    for key in CONDITIONS:
        if reference["config"][key] != candidate["config"][key]:
            raise ValueError(f"experimental condition changed: {key}")
    for key in ("episodes", "selection_sha256", "tokenizer_sha256", "timing_boundary", "memory_protocol"):
        if reference[key] != candidate[key]:
            raise ValueError(f"measurement contract changed: {key}")
    left, right = indexed_rows(reference), indexed_rows(candidate)
    if left.keys() != right.keys():
        raise ValueError("measured call identities differ")
    differences = []
    counts = dict.fromkeys(OUTPUT_FIELDS, 0)
    for key, row in left.items():
        fields = [field for field in OUTPUT_FIELDS if row[field] != right[key][field]]
        for field in fields:
            counts[field] += 1
        if fields:
            differences.append({"sample_id": key[0], "repeat": key[1], "fields": fields})
    # Derive the target and speedup from raw timings, not potentially stale summaries.
    baseline_ms = sum(row["latency_ms"] for row in left.values()) / len(left)
    optimized_ms = sum(row["latency_ms"] for row in right.values()) / len(right)
    return {
        "dataset": name,
        "calls": len(left),
        "matching_calls": len(left) - len(differences),
        "exact_behavior_parity": not differences,
        "mismatches_by_field": counts,
        "first_mismatches": differences[:100],
        "baseline_e2e_mean_ms": baseline_ms,
        "candidate_e2e_mean_ms": optimized_ms,
        "speedup": baseline_ms / optimized_ms,
        "e2e_mean_below_40ms": optimized_ms < 40.0,
        "accuracy_contract": accuracy_contract,
        "navigation_success_evaluated": False,
        "admission_status": (
            "pending_navigation_evaluation"
            if accuracy_contract == "task-success"
            else "passed"
            if not differences and optimized_ms < 40.0
            else "failed"
        ),
        "admitted": (not differences and optimized_ms < 40.0 if accuracy_contract == "exact" else None),
    }


def main() -> None:
    """Write comparison; exit 2 when navigation accuracy still needs evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accuracy-contract",
        choices=("exact", "task-success"),
        default="exact",
        help="task-success reports output differences without treating them as accuracy failures",
    )
    args = parser.parse_args()
    reference, candidate = args.reference.read_bytes(), args.candidate.read_bytes()
    result = compare(json.loads(reference), json.loads(candidate), accuracy_contract=args.accuracy_contract)
    result["reference_sha256"] = hashlib.sha256(reference).hexdigest()
    result["candidate_sha256"] = hashlib.sha256(candidate).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result["admitted"] is None else 0 if result["admitted"] else 1)


if __name__ == "__main__":
    main()
