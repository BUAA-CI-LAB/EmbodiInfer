"""Compare physical action chunks from two completed PI0.5 benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _action_chunks(report: dict) -> dict[tuple[str, int], np.ndarray]:
    """Recover each real observation's chunk from single or batched reports."""
    chunks = {}
    for row in report["rows"]:
        identities = row["sample_ids"] if "sample_ids" in row else [row["sample_id"]]
        actions = np.asarray(row["actions"], dtype=np.float64)
        if (
            not identities
            or len(identities) > report["batch_size"]
            or actions.shape != (50 * len(identities), 7)
        ):
            raise ValueError("invalid action shape or batch membership")
        if not np.isfinite(actions).all():
            raise ValueError("nonfinite actions")
        for index, sample_id in enumerate(identities):
            identity = (sample_id, row["repeat"])
            if identity in chunks:
                raise ValueError("reports must contain unique sample/repeat pairs")
            chunks[identity] = actions[index * 50 : (index + 1) * 50]
    return chunks


def compare(reference: dict, candidate: dict) -> dict:
    """Require identical observations, noise seeds, and complete denoising steps."""
    for key in ("selection_sha256", "batch_size"):
        if reference[key] != candidate[key]:
            raise ValueError(f"incomparable {key}")
    for key in ("seed", "num_steps"):
        if reference["config"][key] != candidate["config"][key]:
            raise ValueError(f"incomparable {key}")
    expected = _action_chunks(reference)
    actual = _action_chunks(candidate)
    if expected.keys() != actual.keys():
        raise ValueError("reports must contain the same unique sample/repeat pairs")
    errors = [actions - expected[identity] for identity, actions in actual.items()]
    delta = np.stack(errors)
    return {
        "observations": len(errors),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "maximum_absolute_error": float(np.abs(delta).max()),
        "per_action_dimension_mae": np.abs(delta).mean(axis=(0, 1)).tolist(),
        "per_action_dimension_rmse": np.sqrt(np.square(delta).mean(axis=(0, 1))).tolist(),
        "note": "Physical action dimensions have different units; this is output drift, not task success.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-absolute-error", type=float)
    args = parser.parse_args()
    if args.max_absolute_error is not None and (
        not np.isfinite(args.max_absolute_error) or args.max_absolute_error < 0
    ):
        parser.error("--max-absolute-error must be finite and nonnegative")
    result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    result.update(reference=str(args.reference), candidate=str(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if args.max_absolute_error is not None and result["maximum_absolute_error"] > args.max_absolute_error:
        raise SystemExit("action error exceeds --max-absolute-error")


if __name__ == "__main__":
    main()
