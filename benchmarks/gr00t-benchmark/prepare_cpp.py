"""Convert the exact LIBERO-10 N1.7 checkpoint using pinned upstream converters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    """Fingerprint weights without reading them all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Write converted artifacts outside the checkpoint and record their provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vlacpp", "embodied"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cosmos-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"conversion output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scripts = args.source.resolve() / "scripts"
    if args.engine == "vlacpp":
        command = [
            sys.executable,
            str(scripts / "convert_gr00t_n1_7_to_gguf.py"),
            "--ckpt",
            str(args.checkpoint),
            "--out",
            str(output / "gr00t-n17-libero10.gguf"),
        ]
    else:
        command = [
            sys.executable,
            str(scripts / "prepare_groot_n1_backbone.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--cosmos-dir",
            str(args.cosmos_path),
            "--output-dir",
            str(output / "prepared-backbone"),
            "--gguf-dir",
            str(output),
            "--backbone-out",
            str(output / "gr00t-backbone.gguf"),
            "--mmproj-out",
            str(output / "gr00t-mmproj.gguf"),
            "--action-head-out",
            str(output / "gr00t-action-head.gguf"),
            "--embodiment",
            "libero_sim",
            "--outtype",
            "bf16",
        ]
    subprocess.run(command, check=True)
    report = {
        "engine": args.engine,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.source, text=True
        ).strip(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_repository": "nvidia/GR00T-N1.7-LIBERO",
        "checkpoint_variant": "libero_10",
        "command": command,
        "original_weights": {p.name: sha256(p) for p in args.checkpoint.glob("*.safetensors")},
        "artifacts": {p.name: sha256(p) for p in output.glob("*.gguf")},
    }
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
