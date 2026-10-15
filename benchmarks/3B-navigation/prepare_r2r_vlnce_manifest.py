"""Prepare deterministic real-RGB R2R-VLNCE manifests without torch/Habitat."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, ImageOps

SELECTION_ALGORITHM = (
    "exact_episode_instruction_scene_join;"
    "episodes=numeric_id_then_scene;"
    "frame_i_action=actions_i_plus_1;"
    "steps=integer_quantiles_with_history4_and_two_sided_neighbors;v2"
)
PANORAMIC_CLASSIFICATION = "r2r_rgb_shape_compatible_not_official_panorama"
SOURCE_CLASSIFICATION = "structurally_aligned_existing_export_not_official_tar_byte_verified"
ACTION_RESPONSES = {0: "Stop", 1: "Move", 2: "Left", 3: "Right"}
PANORAMIC_ANGLES = (-90.0, -30.0, 30.0, 90.0)
PANORAMIC_DISTANCES = (0.5, 0.25, 0.25, 0.5)
VIDEO_RE = re.compile(r"^(?P<scene>.+)_r2r_(?P<episode>[0-9]+)$")


@dataclass(frozen=True)
class JoinedEpisode:
    episode_id: int
    scene: str
    instruction: str
    frames: tuple[Path, ...]
    actions: tuple[int, ...]
    selected_steps: tuple[int, ...]


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return _sha_bytes(payload)


def _episode_id(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} episode id must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} episode id must be an integer") from exc


def _instruction(episode: dict[str, Any], context: str) -> str:
    value = episode.get("instruction")
    if isinstance(value, dict):
        value = value.get("instruction_text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} has no instruction.instruction_text")
    return value


def _scene(scene_id: Any, context: str) -> str:
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError(f"{context} has no scene_id")
    name = Path(scene_id).name
    return name[:-4] if name.endswith(".glb") else name


def _train_index(payload: Any) -> dict[int, tuple[str, str]]:
    episodes = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(episodes, list):
        raise ValueError("R2R-VLNCE train file must contain an episodes list")
    result: dict[int, tuple[str, str]] = {}
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"train[{index}] must be an object")
        episode_id = _episode_id(episode.get("episode_id"), f"train[{index}]")
        if episode_id in result:
            raise ValueError(f"duplicate R2R-VLNCE episode id {episode_id}")
        result[episode_id] = (
            _scene(episode.get("scene_id"), f"train[{index}]"),
            _instruction(episode, f"train[{index}]"),
        )
    return result


def _annotation_instructions(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = tuple(value)
    else:
        raise ValueError(f"{context}.instructions must be a string or string list")
    if not values or any(not item.strip() for item in values):
        raise ValueError(f"{context}.instructions contains an empty instruction")
    return values


def _actions(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.actions must be a non-empty list")
    result: list[int] = []
    for index, action in enumerate(value):
        valid = {-1} if index == 0 else {0, 1, 2, 3}
        if isinstance(action, bool) or not isinstance(action, int) or action not in valid:
            raise ValueError(f"{context}.actions[{index}] is invalid: {action!r}")
        result.append(action)
    return tuple(result)


def _frame_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (2**63, path.name)


def _quantiles(values: Sequence[int], count: int) -> tuple[int, ...]:
    if count <= 0 or len(values) < count:
        raise ValueError(f"need at least {count} eligible steps; trajectory has {len(values)}")
    if count == 1:
        return (values[(len(values) - 1) // 2],)
    denominator = count - 1
    indexes = tuple((rank * (len(values) - 1) + denominator // 2) // denominator for rank in range(count))
    selected = tuple(values[index] for index in indexes)
    if len(set(selected)) != count:
        raise ValueError("integer quantiles produced duplicate steps")
    return selected


def _resolve_inputs(images_argument: Path, annotations_argument: Path | None) -> tuple[Path, Path]:
    root = images_argument.expanduser().resolve(strict=True)
    if (root / "images").is_dir():
        images_root, dataset_root = (root / "images").resolve(strict=True), root
    else:
        images_root, dataset_root = root, root.parent
    if not images_root.is_dir():
        raise ValueError(f"StreamVLN images root is not a directory: {images_root}")
    if annotations_argument is not None:
        annotations = annotations_argument.expanduser().resolve(strict=True)
    else:
        annotations = next(
            (
                path
                for path in (
                    dataset_root / "annotations_v1-3.json",
                    dataset_root / "annotations.json",
                )
                if path.is_file()
            ),
            None,
        )
        if annotations is None:
            raise ValueError("no annotations_v1-3.json/annotations.json next to images; pass --annotations")
    if not annotations.is_file():
        raise ValueError(f"StreamVLN annotations are not a file: {annotations}")
    return images_root, annotations


def _video(value: Any, context: str) -> tuple[Path, str, int]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.video must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context}.video must be a safe relative path")
    parts = relative.parts[1:] if relative.parts and relative.parts[0] == "images" else relative.parts
    if len(parts) != 1:
        raise ValueError(f"{context}.video must identify one episode directory")
    match = VIDEO_RE.fullmatch(parts[0])
    if match is None:
        raise ValueError(f"{context}.video is not scene_r2r_episode")
    return Path(parts[0]), match.group("scene"), int(match.group("episode"))


def _join(
    train: dict[int, tuple[str, str]],
    annotations: Any,
    images_root: Path,
    episode_count: int,
    steps_per_episode: int,
    history_depth: int,
) -> tuple[JoinedEpisode, ...]:
    if not isinstance(annotations, list):
        raise ValueError("StreamVLN annotations must be a JSON array")
    joined: list[JoinedEpisode] = []
    seen: set[int] = set()
    for index, annotation in enumerate(annotations):
        context = f"annotation[{index}]"
        if not isinstance(annotation, dict):
            raise ValueError(f"{context} must be an object")
        episode_id = _episode_id(annotation.get("id"), context)
        if episode_id not in train:
            continue
        if episode_id in seen:
            raise ValueError(f"duplicate joined StreamVLN episode id {episode_id}")
        relative, video_scene, video_episode = _video(annotation.get("video"), context)
        train_scene, train_instruction = train[episode_id]
        if video_episode != episode_id or video_scene != train_scene:
            raise ValueError(f"{context} id/scene does not match R2R-VLNCE")
        if train_instruction not in _annotation_instructions(annotation.get("instructions"), context):
            raise ValueError(f"{context} lacks the exact R2R-VLNCE instruction")
        episode_candidate = images_root / relative
        if not episode_candidate.is_dir():
            continue
        episode_dir = episode_candidate.resolve(strict=True)
        if not episode_dir.is_relative_to(images_root):
            raise ValueError(f"{context}.video escapes images root")
        rgb = episode_dir / "rgb"
        frames = tuple(
            sorted(
                (
                    path.resolve(strict=True)
                    for path in rgb.iterdir()
                    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")
                ),
                key=_frame_key,
            )
        )
        frame_numbers = [int(path.stem) for path in frames if path.stem.isdigit()]
        if len(frame_numbers) != len(frames) or frame_numbers != list(range(1, len(frames) + 1)):
            raise ValueError(f"{context} JPEG names must be consecutive from 001")
        actions = _actions(annotation.get("actions"), context)
        if len(actions) != len(frames):
            raise ValueError(f"{context} has {len(actions)} actions and {len(frames)} JPEGs")
        eligible = tuple(
            step
            for step in range(history_depth, len(frames) - 2)
            if all(
                actions[action_index] in ACTION_RESPONSES
                for action_index in range(step - history_depth + 1, step + 2)
            )
        )
        if len(eligible) < steps_per_episode:
            continue
        seen.add(episode_id)
        joined.append(
            JoinedEpisode(
                episode_id,
                train_scene,
                train_instruction,
                frames,
                actions,
                _quantiles(eligible, steps_per_episode),
            )
        )
    joined.sort(key=lambda item: (item.episode_id, item.scene))
    if len(joined) < episode_count:
        raise ValueError(
            f"only {len(joined)} exact joined episodes satisfy the contract; requested {episode_count}"
        )
    return tuple(joined[:episode_count])


class Derivatives:
    def __init__(self, images_root: Path, output_dir: Path) -> None:
        self.images_root = images_root
        self.output_dir = output_dir
        self.records: list[dict[str, Any]] = []
        self._paths: dict[tuple[Path, str, tuple[int, int]], str] = {}
        self._sizes: dict[Path, tuple[int, int]] = {}
        self._hashes: dict[Path, str] = {}

    def relative(self, source: Path) -> str:
        return source.relative_to(self.images_root).as_posix()

    def size(self, source: Path) -> tuple[int, int]:
        if source not in self._sizes:
            with Image.open(source) as image:
                if image.format != "JPEG":
                    raise ValueError(f"source is not JPEG: {source}")
                self._sizes[source] = ImageOps.exif_transpose(image).size
        return self._sizes[source]

    def source_hash(self, source: Path) -> str:
        if source not in self._hashes:
            self._hashes[source] = _sha_file(source)
        return self._hashes[source]

    def fit(self, source: Path, category: str, size: tuple[int, int]) -> str:
        key = (source, category, size)
        if key in self._paths:
            return self._paths[key]
        source_relative = source.relative_to(self.images_root)
        destination = (
            self.output_dir
            / "derived"
            / category
            / source_relative.parent
            / f"{source_relative.stem}_{size[0]}x{size[1]}.jpg"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with Image.open(source) as image:
            if image.format != "JPEG":
                raise ValueError(f"source is not JPEG: {source}")
            fitted = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"),
                size,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            fitted.save(
                temporary,
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=False,
            )
        temporary.replace(destination)
        output_relative = destination.relative_to(self.output_dir).as_posix()
        self.records.append(
            {
                "category": category,
                "operation": "ImageOps.fit(LANCZOS,centering=0.5,0.5)",
                "source": self.relative(source),
                "source_sha256": self.source_hash(source),
                "source_size": list(self.size(source)),
                "output": output_relative,
                "output_sha256": _sha_file(destination),
                "output_size": list(size),
            }
        )
        self._paths[key] = output_relative
        return output_relative


def _responses(actions: Sequence[int], indexes: Iterable[int]) -> list[str]:
    return [ACTION_RESPONSES[actions[index + 1]] for index in indexes]


def build_manifests(
    train_json_gz: Path,
    images_argument: Path,
    output_dir: Path,
    *,
    annotations_argument: Path | None = None,
    episode_count: int = 48,
    steps_per_episode: int = 4,
    history_depth: int = 4,
) -> dict[str, Any]:
    if episode_count <= 0 or steps_per_episode <= 0:
        raise ValueError("episode_count and steps_per_episode must be positive")
    if history_depth != 4:
        raise ValueError("this benchmark contract requires history_depth=4")
    train_path = train_json_gz.expanduser().resolve(strict=True)
    if not train_path.is_file() or not train_path.name.endswith(".json.gz"):
        raise ValueError("--train-json-gz must be an existing .json.gz")
    images_root, annotations_path = _resolve_inputs(images_argument, annotations_argument)
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = _join(
        _train_index(_load_json(train_path)),
        _load_json(annotations_path),
        images_root,
        episode_count,
        steps_per_episode,
        history_depth,
    )
    derivative = Derivatives(images_root, output)
    used = {
        episode.frames[index]
        for episode in episodes
        for step in episode.selected_steps
        for index in (
            *range(step - history_depth, step + 1),
            step - 2,
            step - 1,
            step + 1,
            step + 2,
        )
    }
    navida_resize = any(derivative.size(source) != (320, 240) for source in used)
    low: list[dict[str, Any]] = []
    navida: list[dict[str, Any]] = []
    panoramic: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for episode in episodes:
        for quantile, step in enumerate(episode.selected_steps):
            history_indexes = tuple(range(step - history_depth, step))
            candidate_indexes = (step - 2, step - 1, step + 1, step + 2)
            sample_id = f"r2r-{episode.episode_id:06d}-q{quantile}-f{step:06d}"
            current = episode.frames[step]
            history = tuple(episode.frames[index] for index in history_indexes)
            responses = _responses(episode.actions, history_indexes)
            distance = round(
                0.25 * sum(action == 1 for action in episode.actions[1 : step + 1]),
                6,
            )
            low.append(
                {
                    "id": sample_id,
                    "instruction": episode.instruction,
                    "current_image": derivative.relative(current),
                    "history_images": [derivative.relative(frame) for frame in history],
                    "history_responses": responses,
                    "distance_traveled": distance,
                    "move_possible": episode.actions[step + 1] != 0,
                }
            )
            if navida_resize:
                navida_current = derivative.fit(current, "navida_320x240", (320, 240))
                navida_history = [derivative.fit(frame, "navida_320x240", (320, 240)) for frame in history]
            else:
                navida_current = derivative.relative(current)
                navida_history = [derivative.relative(frame) for frame in history]
            navida.append(
                {
                    "id": sample_id,
                    "instruction": episode.instruction,
                    "current_image": navida_current,
                    "history_images": navida_history,
                }
            )
            candidate_paths = [
                derivative.fit(
                    episode.frames[index],
                    "panoramic_candidates_320x240",
                    (320, 240),
                )
                for index in candidate_indexes
            ]
            panoramic.append(
                {
                    "id": sample_id,
                    "instruction": episode.instruction,
                    "panorama_image": derivative.fit(current, "panoramic_960x240", (960, 240)),
                    "history_panoramas": [
                        derivative.fit(frame, "panoramic_960x240", (960, 240)) for frame in history
                    ],
                    "history_responses": responses,
                    "distance_traveled": distance,
                    "candidates": [
                        {
                            "image": path,
                            "relative_angle": angle,
                            "distance": candidate_distance,
                        }
                        for path, angle, candidate_distance in zip(
                            candidate_paths,
                            PANORAMIC_ANGLES,
                            PANORAMIC_DISTANCES,
                            strict=True,
                        )
                    ],
                }
            )
            selected.append(
                {
                    "sample_id": sample_id,
                    "episode_id": episode.episode_id,
                    "scene": episode.scene,
                    "instruction_sha256": _sha_bytes(episode.instruction.encode("utf-8")),
                    "quantile": quantile,
                    "frame_index": step,
                    "history_indexes": list(history_indexes),
                    "candidate_indexes": list(candidate_indexes),
                    "action": episode.actions[step + 1],
                    "history_responses": responses,
                }
            )

    payloads = {
        "low": {"samples": low},
        "navida": {"samples": navida},
        "panoramic": {"samples": panoramic},
    }
    paths = {name: output / f"{name}.json" for name in payloads}
    hashes = {name: _write_json(paths[name], payload) for name, payload in payloads.items()}
    provenance = {
        "schema": "embodiinfer_r2r_vlnce_real_rgb_manifest_v1",
        "classification": SOURCE_CLASSIFICATION,
        "official_tar_byte_verified": False,
        "synthetic_pixels": False,
        "trajectory_success_claim": False,
        "inputs": {
            "r2r_vlnce_train_json_gz": str(train_path),
            "r2r_vlnce_train_sha256": _sha_file(train_path),
            "streamvln_images_root": str(images_root),
            "streamvln_annotations": str(annotations_path),
            "streamvln_annotations_sha256": _sha_file(annotations_path),
        },
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "algorithm_sha256": _sha_bytes(SELECTION_ALGORITHM.encode("utf-8")),
            "selection_sha256": _sha_bytes(_canonical(selected)),
            "episode_count": episode_count,
            "steps_per_episode": steps_per_episode,
            "history_depth": history_depth,
            "samples": selected,
        },
        "contracts": {
            "history_per_sample": history_depth,
            "panoramic_candidates_per_sample": 4,
            "low_action_mapping": {str(key): value for key, value in ACTION_RESPONSES.items()},
            "distance": "0.25m times preceding MoveForward actions",
            "action_alignment": "frame[i] corresponds to actions[i+1]",
            "navida": (
                "ImageOps.fit real-RGB derivatives for strict 320x240"
                if navida_resize
                else "direct source StreamVLN 320x240 JPEG references"
            ),
            "panoramic": PANORAMIC_CLASSIFICATION,
            "panoramic_candidate_metadata": ("fixed-shape temporal neighbors; not Habitat navigation edges"),
        },
        "data_roots": {
            "low": str(images_root),
            "navida": str(output if navida_resize else images_root),
            "panoramic": str(output),
        },
        "manifests": {
            name: {
                "path": str(paths[name]),
                "sha256": hashes[name],
                "sample_count": len(payload["samples"]),
            }
            for name, payload in payloads.items()
        },
        "derivative_encoder": {
            "pillow_version": PIL.__version__,
            "format": "JPEG",
            "quality": 95,
            "subsampling": 0,
            "optimize": False,
        },
        "derivatives": derivative.records,
    }
    provenance_path = output / "provenance.json"
    provenance_sha = _write_json(provenance_path, provenance)
    return {
        **provenance,
        "provenance_path": str(provenance_path),
        "provenance_sha256": provenance_sha,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json-gz", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=48)
    parser.add_argument("--steps-per-episode", type=int, default=4)
    parser.add_argument("--history-depth", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_manifests(
            args.train_json_gz,
            args.images_root,
            args.output_dir,
            annotations_argument=args.annotations,
            episode_count=args.episode_count,
            steps_per_episode=args.steps_per_episode,
            history_depth=args.history_depth,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"manifest preparation failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "pass",
                "provenance": result["provenance_path"],
                "provenance_sha256": result["provenance_sha256"],
                "selection_sha256": result["selection"]["selection_sha256"],
                "manifests": result["manifests"],
                "data_roots": result["data_roots"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
