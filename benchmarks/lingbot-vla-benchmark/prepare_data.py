"""Download selected public RoboTwin clean demonstrations, without a simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import yaml


def main() -> None:
    """Fetch pinned task ZIPs serially, retaining only selected HDF5 and instruction files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--root", type=Path, help="override the local dataset destination")
    parser.add_argument("--rate", default="10M", help="curl transfer limit, e.g. 10M")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())["dataset"]
    root = (args.root or Path(config["root"])).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    repository, revision = config["repository"], config["revision"]
    task_count, episode_count = config["task_limit"], config["demos_per_task"]
    if any(type(n) is not int or n <= 0 for n in (task_count, episode_count)):
        raise ValueError("task_limit and demos_per_task must be positive integers")
    if episode_count > 50:
        raise ValueError("the clean split contains only 50 demonstrations per task")
    api = f"https://huggingface.co/api/datasets/{repository}/tree/{revision}/dataset?limit=100"
    with urllib.request.urlopen(api, timeout=30) as response:
        tasks = sorted(Path(x["path"]).name for x in json.load(response) if x["type"] == "directory")
    if len(tasks) < task_count:
        raise ValueError("requested more tasks than available in the pinned release")
    selected = tasks[:task_count]
    manifest = {
        "repository": repository,
        "revision": revision,
        "tasks": selected,
        "demos_per_task": episode_count,
        "archives": {},
    }
    manifest_path = root / "selection.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous["repository"], previous["revision"]) != (repository, revision):
            raise ValueError("use a separate dataset directory for another revision")
    for task in selected:
        relative = f"dataset/{task}/aloha-agilex_clean_50.zip"
        archive = root / ".archives" / f"{task}.zip"
        archive.parent.mkdir(exist_ok=True)
        if not archive.is_file():
            partial = archive.with_suffix(".zip.part")
            for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
                result = subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--connect-timeout",
                        "20",
                        "--max-time",
                        "1800",
                        "--limit-rate",
                        args.rate,
                        f"{endpoint}/datasets/{repository}/resolve/{revision}/{relative}",
                        "--output",
                        str(partial),
                    ]
                )
                if result.returncode == 0:
                    partial.rename(archive)
                    break
            else:
                raise RuntimeError(f"download failed for {task}")
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest["archives"][task] = digest.hexdigest()
        with zipfile.ZipFile(archive) as bundle:
            for episode in range(episode_count):
                for subfolder, suffix in (("data", "hdf5"), ("instructions", "json")):
                    member = f"aloha-agilex_clean_50/{subfolder}/episode{episode}.{suffix}"
                    destination = root / task / member
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_suffix(destination.suffix + ".part")
                    # Read only explicitly constructed regular-file names; never unpickle trajectory data.
                    with bundle.open(member) as source, partial.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    partial.replace(destination)
        print(f"Prepared {task}: {episode_count} public demonstrations", flush=True)
    partial_manifest = manifest_path.with_suffix(".json.part")
    partial_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    partial_manifest.replace(manifest_path)
    print(f"Ready: {root}")


if __name__ == "__main__":
    main()
