"""Extract the selected public navigation RGB episodes without simulation."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any


def positive(value: Any, name: str) -> int:
    """Reject ambiguous, zero, or negative selection sizes."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def navigation_annotations(path: Path, count: int) -> list[dict[str, Any]]:
    """Choose the first numeric episode IDs before checking local image availability."""
    positive(count, "episode_limit")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("navigation annotations must be a nonempty list")
    records = sorted(records, key=lambda row: (int(row["id"]), row["video"]))
    ids = [int(row["id"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate navigation episode IDs")
    if count > len(records):
        raise ValueError(f"requested {count} episodes, release has {len(records)}")
    return records[:count]


class ArchiveParts(io.RawIOBase):
    """Read explicitly ordered split archive files as one forward-only byte stream."""

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self.paths = iter(paths)
        self.current = next(self.paths).open("rb")
        self.finished = False

    def read(self, size: int = -1) -> bytes:
        """Fill a bounded read across part boundaries without buffering whole archives."""
        if size < 0:
            raise ValueError("archive reader requires bounded reads")
        if self.finished:
            return b""
        blocks = []
        remaining = size
        while remaining:
            block = self.current.read(remaining)
            if block:
                blocks.append(block)
                remaining -= len(block)
            else:
                self.current.close()
                path = next(self.paths, None)
                if path is None:
                    self.finished = True
                    break
                self.current = path.open("rb")
        return b"".join(blocks)

    def close(self) -> None:
        """Close the active part when extraction succeeds or fails."""
        self.current.close()
        super().close()


def main() -> None:
    """Safely extract exact first-N episode RGB files and write selection provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--archives", type=Path, nargs="+", required=True, help="tar.gz or ordered part0 part1 ..."
    )
    parser.add_argument("--output-root", type=Path, required=True, help="directory containing images/")
    parser.add_argument("--episodes", type=int, default=48)
    args = parser.parse_args()
    rows = navigation_annotations(args.annotations, args.episodes)
    wanted = {Path(row["video"]).name: len(row["actions"]) for row in rows}
    selected: set[str] = set()
    for path in args.archives:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_root.resolve()
    with ArchiveParts(args.archives) as parts, tarfile.open(fileobj=parts, mode="r|gz") as archive:
        for member in archive:
            names = Path(member.name).parts
            if not member.isfile() or len(names) < 3 or names[-3] not in wanted or names[-2] != "rgb":
                continue
            filename = Path(names[-1])
            if filename.suffix != ".jpg" or not filename.stem.isdigit():
                continue
            index = int(filename.stem)
            if not 1 <= index <= wanted[names[-3]]:
                raise ValueError(f"out-of-range published frame: {member.name}")
            relative = Path("images", *names[-3:])
            destination = (output / relative).resolve()
            if not destination.is_relative_to(output):
                raise ValueError(f"unsafe extraction target: {relative}")
            if str(relative) in selected:
                raise ValueError(f"duplicate frame in archive: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".jpg.tmp")
            temporary.write_bytes(source.read())
            temporary.replace(destination)
            selected.add(str(relative))
            if len(selected) % 500 == 0:
                print(f"Extracted {len(selected)}/{sum(wanted.values())} frames", flush=True)
            if len(selected) == sum(wanted.values()):
                break
    if len(selected) != sum(wanted.values()):
        raise ValueError(f"incomplete archive selection: {len(selected)}/{sum(wanted.values())} frames")
    (output / "extracted-selection.json").write_text(
        json.dumps(
            {
                "episode_ids": [row["id"] for row in rows],
                "frames": len(selected),
                "annotations": str(args.annotations),
                "archives": [str(path) for path in args.archives],
                "selection": "numeric_episode_id_then_video;first_N;all_frames",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Extracted {len(rows)} complete episodes ({len(selected)} RGB frames)")


if __name__ == "__main__":
    main()
