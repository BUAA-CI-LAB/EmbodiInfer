"""Validate the immutable ActiveVLN source/checkpoint pins used by parity tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load_lock(path: Path) -> dict:
    data = json.loads(path.read_text())
    commit = data["source"]["commit"]
    revision = data["checkpoint"]["revision"]
    if not _HEX40.fullmatch(commit):
        raise ValueError(f"source commit must be an immutable 40-hex SHA, got {commit!r}")
    if not _HEX40.fullmatch(revision):
        raise ValueError(f"checkpoint revision must be an immutable 40-hex SHA, got {revision!r}")
    for file_path, blob in data["source"]["files"].items():
        if not _HEX40.fullmatch(blob):
            raise ValueError(f"invalid git blob SHA for {file_path}: {blob!r}")
    for file_path, digest in data["checkpoint"]["files"].items():
        if not _HEX64.fullmatch(digest):
            raise ValueError(f"invalid sha256 for {file_path}: {digest!r}")
    return data


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_source(lock: dict, source_root: Path) -> None:
    head = _git(source_root, "rev-parse", "HEAD")
    if head != lock["source"]["commit"]:
        raise ValueError(f"ActiveVLN HEAD mismatch: expected {lock['source']['commit']}, got {head}")
    dirty = _git(source_root, "status", "--porcelain")
    if dirty:
        raise ValueError("ActiveVLN checkout must be clean before producing a reference")
    for file_path, expected in lock["source"]["files"].items():
        actual = _git(source_root, "rev-parse", f"HEAD:{file_path}")
        if actual != expected:
            raise ValueError(f"ActiveVLN blob mismatch for {file_path}: expected {expected}, got {actual}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(lock: dict, checkpoint_root: Path) -> None:
    for file_path, expected in lock["checkpoint"]["files"].items():
        path = checkpoint_root / file_path
        if not path.is_file():
            raise ValueError(f"checkpoint is missing required file: {file_path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"checkpoint hash mismatch for {file_path}: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).with_name("source_lock.json"),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()

    lock = _load_lock(args.lock)
    if args.source_root is not None:
        verify_source(lock, args.source_root)
    if args.checkpoint_root is not None:
        verify_checkpoint(lock, args.checkpoint_root)
    print("ActiveVLN source lock is valid")


if __name__ == "__main__":
    main()
