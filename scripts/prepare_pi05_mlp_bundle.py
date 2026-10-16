"""Bind existing rotated INT8 prefix and GPTQ action archives for deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _reference(path: Path, output: Path) -> dict[str, str]:
    return {
        "manifest": os.path.relpath(path, output.parent),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    """Validate archive metadata and write a portable relative-path bundle.

    This performs no calibration or weight conversion and requires no GPU.
    Keep both referenced archive directories with the bundle when deploying.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True, help="Rotated W8A8 manifest")
    parser.add_argument("--action", type=Path, required=True, help="Group64 GPTQ manifest")
    parser.add_argument("--output", type=Path, required=True, help="New deployment bundle")
    args = parser.parse_args()
    prefix, action, output = (path.expanduser().resolve() for path in (args.prefix, args.action, args.output))
    if output.exists():
        raise FileExistsError(output)
    from embodiinfer.policies.pi05.inference.weights import MlpWeightArchive, RotatedInt8Archive

    checkpoint = json.loads(action.read_text()).get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise ValueError("GPTQ archive must identify its calibrated checkpoint SHA256")
    MlpWeightArchive(str(action), checkpoint, layers=18)
    archive = RotatedInt8Archive(prefix, checkpoint, layers=18)
    action_reference = _reference(action, output)
    if archive.gptq_sha256 != action_reference["sha256"]:
        raise ValueError("Prefix calibration belongs to a different GPTQ archive")
    bundle = {
        "schema_version": 1,
        "method": "w8a8-w4a16",
        "checkpoint_sha256": checkpoint,
        "prefix": _reference(prefix, output),
        "action": action_reference,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
