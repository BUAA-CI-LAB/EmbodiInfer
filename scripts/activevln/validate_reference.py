"""Validate an ActiveVLN parity reference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.activevln.schema import validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    manifest = validate_bundle(args.reference)
    print(f"valid ActiveVLN reference: producer={manifest['producer']} cases={len(manifest['cases'])}")


if __name__ == "__main__":
    main()
