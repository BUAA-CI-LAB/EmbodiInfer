#!/usr/bin/env python3
"""Diff two ActiveVLN batch reports by (episode_id, step) token sequences."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(path: Path) -> dict[tuple[int, int], dict]:
    report = json.loads(path.read_text())
    return {(row["episode_id"], row["step"]): row for row in report["rows"]}


def main() -> int:
    left, right = rows(Path(sys.argv[1])), rows(Path(sys.argv[2]))
    common = sorted(set(left) & set(right))
    token_mismatch = [key for key in common if left[key]["token_ids"] != right[key]["token_ids"]]
    reason_mismatch = [
        key for key in common if left[key].get("stop_reason") != right[key].get("stop_reason")
    ]
    print(
        json.dumps(
            {
                "left_rows": len(left),
                "right_rows": len(right),
                "common_rows": len(common),
                "only_left": len(set(left) - set(right)),
                "only_right": len(set(right) - set(left)),
                "token_mismatches": len(token_mismatch),
                "stop_reason_mismatches": len(reason_mismatch),
                "first_token_mismatch": token_mismatch[:5],
            },
            indent=1,
        )
    )
    return 0 if not token_mismatch and not reason_mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
