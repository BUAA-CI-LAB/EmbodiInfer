"""Compare actions over the same recorded StreamVLN trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(reference: dict, candidate: dict) -> dict:
    """Measure replay agreement, including accumulated generated-history differences."""
    for key in ("selection_sha256", "batch_size"):
        if reference[key] != candidate[key]:
            raise ValueError(f"incomparable {key}")
    for key in ("seed", "max_new_tokens", "decode_block_size"):
        if reference["config"][key] != candidate["config"][key]:
            raise ValueError(f"incomparable {key}")
    expected = {(row["sample_id"], row["repeat"]): row for row in reference["rows"]}
    actual = {(row["sample_id"], row["repeat"]): row for row in candidate["rows"]}
    if expected.keys() != actual.keys() or len(expected) != len(reference["rows"]):
        raise ValueError("reports must contain the same unique sample/repeat pairs")
    exact = slots_equal = slots = 0
    for identity, row in actual.items():
        left, right = expected[identity]["actions"], row["actions"]
        if len(left) != len(right):
            raise ValueError(f"action capacity changed for {identity}")
        exact += left == right
        slots_equal += sum(a == b for a, b in zip(left, right, strict=True))
        slots += len(left)
    count = len(actual)
    return {
        "observations": count,
        "action_chunk_agreement": exact / count,
        "action_slot_agreement": slots_equal / slots,
        "reference_invalid_text_rate": sum(bool(r["invalid_action_text"]) for r in expected.values()) / count,
        "candidate_invalid_text_rate": sum(bool(r["invalid_action_text"]) for r in actual.values()) / count,
        "reference_mean_generated_tokens": sum(r["generated_tokens"] for r in expected.values()) / count,
        "candidate_mean_generated_tokens": sum(r["generated_tokens"] for r in actual.values()) / count,
        "note": "Recorded RGB replay with each run's generated history; agreement is not navigation SR/SPL.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    result.update(reference=str(args.reference), candidate=str(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
