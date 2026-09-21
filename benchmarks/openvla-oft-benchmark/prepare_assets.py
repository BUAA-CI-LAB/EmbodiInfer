"""Prepare pinned checkpoint assets for this benchmark in a separate local directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.yaml"]


def download(
    repository: str, revision: str, destination: Path, *, subfolder: str = "", weights: bool = True
) -> None:
    """Mirror first; use immutable revisions and avoid other suites/training checkpoints."""
    target = destination.parent if subfolder else destination
    destination.mkdir(parents=True, exist_ok=True)
    patterns = [pattern for pattern in PATTERNS if weights or pattern != "*.safetensors"]
    if subfolder:
        patterns = [f"{subfolder}/{pattern}" for pattern in patterns]
    manifest = destination / ".benchmark-source.json"
    identity = {"repository": repository, "revision": revision, "subfolder": subfolder}
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError(f"{destination}: use another directory for a different source revision")
    for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
        try:
            snapshot_download(
                repo_id=repository,
                revision=revision,
                local_dir=target,
                allow_patterns=patterns,
                max_workers=1,
                endpoint=endpoint,
            )
            break
        except Exception as exc:
            # Exception messages can contain signed download URLs; report only the error type.
            print(f"{endpoint}: {type(exc).__name__}; trying the next source", flush=True)
    else:
        raise RuntimeError(f"could not download {repository} at {revision}")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    print(f"Prepared {repository}@{revision}: {destination}", flush=True)


def main() -> None:
    """Prepare files referenced by config.yaml; no GPU or simulation is started."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    download(
        config["checkpoint_repository"],
        config["checkpoint_revision"],
        Path(config["checkpoint"]),
        subfolder="",
    )


if __name__ == "__main__":
    main()
