"""Strict real-image manifest loading for the 3B navigation benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from embodiinfer import Observation

LOW_PROFILE = "qwen2.5-vl-3b-r2r-low-level"
PANORAMIC_PROFILE = "qwen2.5-vl-3b-r2r-panoramic"
NAVIDA_PROFILE = "navida"
PROFILES = (LOW_PROFILE, PANORAMIC_PROFILE, NAVIDA_PROFILE)

_PROFILE_KINDS = {
    LOW_PROFILE: "low",
    PANORAMIC_PROFILE: "panoramic",
    NAVIDA_PROFILE: "navida",
}


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    instruction: str
    observation: Observation
    history_frames: tuple[torch.Tensor, ...]
    history_responses: tuple[str, ...]
    image_paths: dict[str, Any]

    @property
    def memory_source(self) -> str:
        return "manifest" if self.history_frames or self.history_responses else "empty"


@dataclass(frozen=True)
class LoadedManifest:
    path: Path
    data_root: Path
    sha256: str
    samples: tuple[ManifestSample, ...]
    provenance: dict[str, Any]


def profile_kind(profile: str) -> str:
    try:
        return _PROFILE_KINDS[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported profile {profile!r}; expected one of {PROFILES}") from exc


def _manifest_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    manifest_path = path.expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError(f"manifest is not a regular file: {manifest_path}")
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"manifest must be UTF-8: {manifest_path}") from exc

    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {manifest_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{manifest_path}:{line_number} must contain one JSON object")
            records.append(value)
    elif suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON manifest {manifest_path}: {exc.msg}") from exc
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            if set(payload) != {"samples"} or not isinstance(payload["samples"], list):
                raise ValueError("JSON object manifests must contain exactly one list field: samples")
            records = payload["samples"]
        else:
            raise ValueError("JSON manifest must be an array or an object with samples")
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("every manifest sample must be a JSON object")
    else:
        raise ValueError("manifest filename must end in .json or .jsonl")

    if not records:
        raise ValueError(f"manifest contains no samples: {manifest_path}")
    return records, digest


def _sample_id(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{context}.id must be a string or integer")
    sample_id = str(value)
    if not sample_id:
        raise ValueError(f"{context}.id must not be empty")
    return sample_id


def _instruction(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.instruction must be a non-empty string")
    return value


def _check_fields(
    record: dict[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(record))
    unknown = sorted(set(record) - required - optional)
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unsupported fields: {unknown}")


def _number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _relative_image(
    data_root: Path,
    value: Any,
    field: str,
    expected_size: tuple[int, int] | None,
) -> tuple[torch.Tensor, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative file path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must remain relative to --data-root: {value!r}")
    try:
        resolved = (data_root / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} does not exist: {value!r}") from exc
    if not resolved.is_relative_to(data_root) or not resolved.is_file():
        raise ValueError(f"{field} must resolve to a file inside --data-root: {value!r}")

    try:
        with Image.open(resolved) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            size = rgb.size
            array = np.array(rgb, dtype=np.float32, copy=True)
    except Exception as exc:
        raise ValueError(f"{field} is not a readable image: {value!r}") from exc

    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"{field} must be {expected_size[0]}x{expected_size[1]} RGB; got "
            f"{size[0]}x{size[1]} for {value!r}"
        )
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(255.0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"{field} did not decode to CHW RGB: {value!r}")
    return tensor, relative.as_posix()


def _path_list(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _response_list(value: Any, field: str) -> tuple[str, ...]:
    values = _path_list(value, field)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{field} must contain only strings")
    return tuple(values)


def _history_images(
    data_root: Path,
    value: Any,
    field: str,
    expected_size: tuple[int, int] | None,
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    frames: list[torch.Tensor] = []
    paths: list[str] = []
    for index, item in enumerate(_path_list(value, field)):
        frame, path = _relative_image(data_root, item, f"{field}[{index}]", expected_size)
        frames.append(frame)
        paths.append(path)
    return tuple(frames), paths


def _observation(
    image: torch.Tensor,
    instruction: str,
    metadata: dict[str, Any],
) -> Observation:
    return Observation(
        images=image.unsqueeze(0),
        state=torch.empty(0),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction=instruction,
        metadata=metadata,
    )


def _parse_low(record: dict[str, Any], context: str, data_root: Path) -> ManifestSample:
    required = {"id", "instruction", "current_image"}
    optional = {
        "history_images",
        "history_responses",
        "distance_traveled",
        "move_possible",
    }
    _check_fields(record, required, optional, context)
    sample_id = _sample_id(record["id"], context)
    instruction = _instruction(record["instruction"], context)
    current, current_path = _relative_image(
        data_root, record["current_image"], f"{context}.current_image", None
    )
    history, history_paths = _history_images(
        data_root, record.get("history_images"), f"{context}.history_images", None
    )
    responses = _response_list(record.get("history_responses"), f"{context}.history_responses")
    if len(history) != len(responses):
        raise ValueError(f"{context}.history_images and history_responses must have equal length")
    distance = _number(
        record.get("distance_traveled", 0.0),
        f"{context}.distance_traveled",
        nonnegative=True,
    )
    move_possible = _boolean(record.get("move_possible", True), f"{context}.move_possible")
    return ManifestSample(
        sample_id=sample_id,
        instruction=instruction,
        observation=_observation(
            current,
            instruction,
            {
                "distance_traveled": distance,
                "move_possible": move_possible,
            },
        ),
        history_frames=history,
        history_responses=responses,
        image_paths={
            "current_image": current_path,
            "history_images": history_paths,
        },
    )


def _parse_navida(record: dict[str, Any], context: str, data_root: Path) -> ManifestSample:
    required = {"id", "instruction", "current_image"}
    optional = {"history_images"}
    _check_fields(record, required, optional, context)
    sample_id = _sample_id(record["id"], context)
    instruction = _instruction(record["instruction"], context)
    current, current_path = _relative_image(
        data_root,
        record["current_image"],
        f"{context}.current_image",
        (320, 240),
    )
    history, history_paths = _history_images(
        data_root,
        record.get("history_images"),
        f"{context}.history_images",
        (320, 240),
    )
    return ManifestSample(
        sample_id=sample_id,
        instruction=instruction,
        observation=_observation(current, instruction, {}),
        history_frames=history,
        history_responses=(),
        image_paths={
            "current_image": current_path,
            "history_images": history_paths,
        },
    )


def _parse_panoramic(record: dict[str, Any], context: str, data_root: Path) -> ManifestSample:
    required = {"id", "instruction", "panorama_image", "candidates"}
    optional = {
        "history_panoramas",
        "history_responses",
        "distance_traveled",
    }
    _check_fields(record, required, optional, context)
    sample_id = _sample_id(record["id"], context)
    instruction = _instruction(record["instruction"], context)
    panorama, panorama_path = _relative_image(
        data_root,
        record["panorama_image"],
        f"{context}.panorama_image",
        (960, 240),
    )
    history, history_paths = _history_images(
        data_root,
        record.get("history_panoramas"),
        f"{context}.history_panoramas",
        (960, 240),
    )
    responses = _response_list(record.get("history_responses"), f"{context}.history_responses")
    if len(history) != len(responses):
        raise ValueError(f"{context}.history_panoramas and history_responses must have equal length")

    raw_candidates = record["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"{context}.candidates must be a non-empty list")
    candidate_images: list[torch.Tensor] = []
    candidate_metadata: list[dict[str, float]] = []
    candidate_paths: list[str] = []
    for index, candidate in enumerate(raw_candidates):
        candidate_context = f"{context}.candidates[{index}]"
        if not isinstance(candidate, dict):
            raise ValueError(f"{candidate_context} must be an object")
        _check_fields(
            candidate,
            {"image", "relative_angle", "distance"},
            set(),
            candidate_context,
        )
        image, path = _relative_image(
            data_root,
            candidate["image"],
            f"{candidate_context}.image",
            (320, 240),
        )
        candidate_images.append(image)
        candidate_paths.append(path)
        candidate_metadata.append(
            {
                "relative_angle": _number(
                    candidate["relative_angle"],
                    f"{candidate_context}.relative_angle",
                ),
                "distance": _number(
                    candidate["distance"],
                    f"{candidate_context}.distance",
                    nonnegative=True,
                ),
            }
        )

    distance = _number(
        record.get("distance_traveled", 0.0),
        f"{context}.distance_traveled",
        nonnegative=True,
    )
    return ManifestSample(
        sample_id=sample_id,
        instruction=instruction,
        observation=_observation(
            panorama,
            instruction,
            {
                "candidate_images": candidate_images,
                "candidates": candidate_metadata,
                "distance_traveled": distance,
            },
        ),
        history_frames=history,
        history_responses=responses,
        image_paths={
            "panorama_image": panorama_path,
            "candidate_images": candidate_paths,
            "history_panoramas": history_paths,
        },
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _declared_path(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} does not exist: {value!r}") from exc
    return resolved


def _relative_provenance_file(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a safe relative path")
    try:
        resolved = (root / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} does not exist: {value!r}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"{field} must resolve to a file inside {root}")
    return resolved


def _manifest_image_paths(records: list[dict[str, Any]], kind: str) -> set[str]:
    paths: set[str] = set()
    for index, record in enumerate(records):
        context = f"sample[{index}]"
        if kind in ("low", "navida"):
            values = [record.get("current_image"), *record.get("history_images", [])]
        else:
            values = [
                record.get("panorama_image"),
                *record.get("history_panoramas", []),
                *[
                    candidate.get("image")
                    for candidate in record.get("candidates", [])
                    if isinstance(candidate, dict)
                ],
            ]
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{context} contains an invalid image path")
            paths.add(Path(value).as_posix())
    return paths


def validate_manifest_provenance(
    manifest_path: Path,
    data_root: Path,
    profile: str,
    records: list[dict[str, Any]],
    provenance: str | Path | None,
    *,
    require_formal: bool,
) -> dict[str, Any]:
    if provenance is None:
        return {
            "status": "unverified",
            "formal": False,
            "reason": "--provenance was not provided",
            "path": None,
            "sha256": None,
            "synthetic_pixels": None,
        }

    provenance_path = Path(provenance).expanduser().resolve(strict=True)
    if not provenance_path.is_file():
        raise ValueError(f"provenance is not a regular file: {provenance_path}")
    raw = provenance_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON provenance: {provenance_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("provenance must be a JSON object")
    if payload.get("schema") != "embodiinfer_r2r_vlnce_real_rgb_manifest_v1":
        raise ValueError("unsupported R2R provenance schema")
    if payload.get("synthetic_pixels") is not False:
        raise ValueError("formal R2R provenance requires synthetic_pixels=false")
    if payload.get("trajectory_success_claim") is not False:
        raise ValueError("R2R throughput provenance cannot claim trajectory success")
    if payload.get("official_tar_byte_verified") is not False:
        raise ValueError("R2R throughput provenance must declare official_tar_byte_verified=false")

    kind = profile_kind(profile)
    base = provenance_path.parent
    manifests = payload.get("manifests")
    if not isinstance(manifests, dict) or not isinstance(manifests.get(kind), dict):
        raise ValueError(f"provenance has no manifest record for profile {kind}")
    manifest_record = manifests[kind]
    declared_manifest = _declared_path(manifest_record.get("path"), base, f"manifests.{kind}.path")
    if declared_manifest != manifest_path:
        raise ValueError("provenance manifest path does not match --manifest")
    manifest_sha = _file_sha256(manifest_path)
    if manifest_record.get("sha256") != manifest_sha:
        raise ValueError("provenance manifest SHA256 does not match --manifest")
    if manifest_record.get("sample_count") != len(records):
        raise ValueError("provenance manifest sample_count does not match manifest")

    roots = payload.get("data_roots")
    if not isinstance(roots, dict):
        raise ValueError("provenance data_roots must be an object")
    declared_root = _declared_path(roots.get(kind), base, f"data_roots.{kind}")
    if declared_root != data_root:
        raise ValueError("provenance data_root does not match --data-root")

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("provenance inputs must be an object")
    train_path = _declared_path(
        inputs.get("r2r_vlnce_train_json_gz"),
        base,
        "inputs.r2r_vlnce_train_json_gz",
    )
    annotations_path = _declared_path(
        inputs.get("streamvln_annotations"),
        base,
        "inputs.streamvln_annotations",
    )
    source_root = _declared_path(
        inputs.get("streamvln_images_root"),
        base,
        "inputs.streamvln_images_root",
    )
    if not source_root.is_dir():
        raise ValueError("provenance StreamVLN images root is not a directory")
    if _file_sha256(train_path) != inputs.get("r2r_vlnce_train_sha256"):
        raise ValueError("R2R-VLNCE train source SHA256 mismatch")
    if _file_sha256(annotations_path) != inputs.get("streamvln_annotations_sha256"):
        raise ValueError("StreamVLN annotation source SHA256 mismatch")

    derivatives = payload.get("derivatives")
    if not isinstance(derivatives, list):
        raise ValueError("provenance derivatives must be a list")
    source_paths: set[str] = set()
    output_paths: set[str] = set()
    verified_files: dict[Path, str] = {}
    for index, derivative in enumerate(derivatives):
        context = f"derivatives[{index}]"
        if not isinstance(derivative, dict):
            raise ValueError(f"{context} must be an object")
        source_value = derivative.get("source")
        output_value = derivative.get("output")
        source = _relative_provenance_file(source_root, source_value, f"{context}.source")
        output = _relative_provenance_file(base, output_value, f"{context}.output")
        expected_source = derivative.get("source_sha256")
        expected_output = derivative.get("output_sha256")
        if not isinstance(expected_source, str) or len(expected_source) != 64:
            raise ValueError(f"{context}.source_sha256 is invalid")
        if not isinstance(expected_output, str) or len(expected_output) != 64:
            raise ValueError(f"{context}.output_sha256 is invalid")
        for path, expected in (
            (source, expected_source),
            (output, expected_output),
        ):
            previous = verified_files.setdefault(path, expected)
            if previous != expected:
                raise ValueError(f"conflicting SHA256 declarations for {path}")
        source_paths.add(Path(source_value).as_posix())
        output_paths.add(Path(output_value).as_posix())
    for path, expected in verified_files.items():
        if _file_sha256(path) != expected:
            raise ValueError(f"provenance file SHA256 mismatch: {path}")

    manifest_images = _manifest_image_paths(records, kind)
    expected_paths = source_paths if data_root == source_root else output_paths
    uncovered = sorted(manifest_images - expected_paths)
    if uncovered:
        raise ValueError(f"manifest image paths lack source/derivative SHA provenance: {uncovered[:3]}")

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("provenance selection must be an object")
    selected = selection.get("samples")
    if not isinstance(selected, list):
        raise ValueError("provenance selection.samples must be a list")
    if selection.get("selection_sha256") != _canonical_sha256(selected):
        raise ValueError("provenance selection SHA256 mismatch")
    selected_ids = [str(record.get("sample_id")) for record in selected if isinstance(record, dict)]
    manifest_ids = [_sample_id(record.get("id"), f"sample[{index}]") for index, record in enumerate(records)]
    if selected_ids != manifest_ids:
        raise ValueError("provenance selection sample ids do not match manifest")

    if require_formal:
        if (
            selection.get("episode_count") != 48
            or selection.get("steps_per_episode") != 4
            or selection.get("history_depth") != 4
            or len(records) != 192
            or len(selected) != 192
        ):
            raise ValueError(
                "formal R2R admission requires exactly 48 episodes x 4 steps = "
                "192 samples with history_depth=4"
            )
        episode_groups: dict[int, list[dict[str, Any]]] = {}
        for index, selected_record in enumerate(selected):
            if not isinstance(selected_record, dict):
                raise ValueError(f"selection.samples[{index}] must be an object")
            episode_id = selected_record.get("episode_id")
            quantile = selected_record.get("quantile")
            frame_index = selected_record.get("frame_index")
            if isinstance(episode_id, bool) or not isinstance(episode_id, int):
                raise ValueError(f"selection.samples[{index}].episode_id must be an integer")
            if quantile not in (0, 1, 2, 3):
                raise ValueError(f"selection.samples[{index}].quantile must be q0..q3")
            if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                raise ValueError(f"selection.samples[{index}].frame_index must be an integer")
            episode_groups.setdefault(episode_id, []).append(selected_record)
        if len(episode_groups) != 48:
            raise ValueError("formal R2R admission requires 48 unique episodes")
        for episode_id, group in episode_groups.items():
            if (
                len(group) != 4
                or {record["quantile"] for record in group} != {0, 1, 2, 3}
                or len({record["frame_index"] for record in group}) != 4
            ):
                raise ValueError(
                    f"formal R2R episode {episode_id} must contain q0..q3 at four distinct steps"
                )
        for index, (record, selected_record) in enumerate(zip(records, selected)):
            if (
                not isinstance(selected_record, dict)
                or not isinstance(selected_record.get("history_responses"), list)
                or len(selected_record["history_responses"]) != 4
            ):
                raise ValueError(f"selection.samples[{index}] must contain four history responses")
            if kind in ("low", "navida"):
                if len(record.get("history_images", [])) != 4:
                    raise ValueError(f"sample[{index}] must contain four history images")
                if kind == "low" and len(record.get("history_responses", [])) != 4:
                    raise ValueError(f"sample[{index}] must contain four history responses")
            else:
                if (
                    len(record.get("history_panoramas", [])) != 4
                    or len(record.get("history_responses", [])) != 4
                    or len(record.get("candidates", [])) != 4
                ):
                    raise ValueError(
                        f"sample[{index}] must contain four histories, responses, and candidates"
                    )
        if payload.get("classification") != (
            "structurally_aligned_existing_export_not_official_tar_byte_verified"
        ):
            raise ValueError("formal R2R provenance classification is missing")

    return {
        "status": "verified",
        "formal": require_formal,
        "reason": None,
        "path": str(provenance_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "synthetic_pixels": False,
        "classification": payload.get("classification"),
        "official_tar_byte_verified": False,
        "selection_sha256": selection.get("selection_sha256"),
        "verified_source_and_derivative_files": len(verified_files),
    }


def load_manifest(
    manifest: str | Path,
    data_root: str | Path,
    profile: str,
    limit: int,
    provenance: str | Path | None = None,
    *,
    require_formal: bool = False,
) -> LoadedManifest:
    if limit <= 0:
        raise ValueError("--limit must be positive")
    kind = profile_kind(profile)
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"--data-root is not a directory: {root}")
    records, digest = _manifest_records(manifest_path)
    admission = validate_manifest_provenance(
        manifest_path,
        root,
        profile,
        records,
        provenance,
        require_formal=require_formal,
    )
    if limit > len(records):
        raise ValueError(f"--limit={limit} exceeds manifest sample count {len(records)}")
    if require_formal and limit != len(records):
        raise ValueError("formal R2R admission requires --limit=192 with complete manifest coverage")

    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        context = f"sample[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{context} must be an object")
        sample_id = _sample_id(record.get("id"), context)
        if sample_id in seen:
            raise ValueError(f"duplicate sample id {sample_id!r} at indexes {seen[sample_id]} and {index}")
        seen[sample_id] = index

    parser = {
        "low": _parse_low,
        "panoramic": _parse_panoramic,
        "navida": _parse_navida,
    }[kind]
    samples = tuple(parser(record, f"sample[{index}]", root) for index, record in enumerate(records[:limit]))
    return LoadedManifest(
        path=manifest_path,
        data_root=root,
        sha256=digest,
        samples=samples,
        provenance=admission,
    )
