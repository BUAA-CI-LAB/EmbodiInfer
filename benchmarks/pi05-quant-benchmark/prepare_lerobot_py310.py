"""Backport LeRobot 0.5.1 typing syntax inside the Orin benchmark environment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> None:
    """Keep the PI0.5 model and processor computations intact on Python 3.10."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or Path(sys.prefix).name != ".venv":
        raise RuntimeError("run with the Orin benchmark's isolated Python 3.10 .venv")
    wheel = args.wheel.resolve(strict=True)
    if wheel.name != "lerobot-0.5.1-py3-none-any.whl":
        raise ValueError("expected the original LeRobot 0.5.1 wheel")
    replacements = {
        "lerobot/processor/pipeline.py": (
            "@dataclass\nclass DataProcessorPipeline[TInput, TOutput](HubMixin):",
            "from typing import Generic, TypeVar\n\nTInput = TypeVar('TInput')\n"
            "TOutput = TypeVar('TOutput')\n\n@dataclass\n"
            "class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):",
        ),
        "lerobot/datasets/streaming_dataset.py": (
            "class Backtrackable[T]:",
            "from typing import Generic, TypeVar\n\nT = TypeVar('T')\n\nclass Backtrackable(Generic[T]):",
        ),
        "lerobot/utils/io_utils.py": (
            "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:",
            "from typing import TypeVar\n\nT = TypeVar('T', bound=JsonLike)\n\n"
            "def deserialize_json_into_object(fpath: Path, obj: T) -> T:",
        ),
    }
    changed = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.endswith(".py"):
                continue
            original = archive.read(name).decode()
            text = original
            if name in replacements:
                before, after = replacements[name]
                if text.count(before) != 1:
                    raise ValueError(f"unexpected source layout: {name}")
                text = text.replace(before, after)
            if name == "lerobot/motors/motors_bus.py":
                text = text.replace("type NameOrID =", "NameOrID =").replace("type Value =", "Value =")
            text = "\n".join(
                line.replace("from typing import", "from typing_extensions import")
                if line.startswith("from typing import") and "Unpack" in line
                else line
                for line in text.split("\n")
            )
            ast.parse(text, filename=name)
            if text != original:
                changed[name] = text
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-requires-python",
            str(wheel),
        ],
        check=True,
    )
    distribution = importlib.metadata.distribution("lerobot")
    for name, text in changed.items():
        distribution.locate_file(name).write_text(text)
    report = {
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "python": sys.version,
        "changes": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in changed.items()},
        "scope": "typing syntax only: PEP695 generics/aliases and typing_extensions.Unpack; no model math changes",
    }
    (Path(sys.prefix) / "lerobot-py310-compatibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
