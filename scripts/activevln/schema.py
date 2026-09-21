"""Versioned manifest and tensor-bundle schema for ActiveVLN parity references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_NAME = "vvla.activevln.reference"
SCHEMA_VERSION = 1
PRODUCERS = {"hf_full", "hf_incremental", "official_vllm"}
LOCK_PATH = Path(__file__).with_name("source_lock.json")


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_index(tensors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        for key, value in sorted(tensors.items())
    }


def validate_manifest(manifest: dict[str, Any], *, lock: dict[str, Any] | None = None) -> None:
    lock = lock or load_lock()
    if manifest.get("schema") != SCHEMA_NAME:
        raise ValueError(f"unexpected schema: {manifest.get('schema')!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    if manifest.get("producer") not in PRODUCERS:
        raise ValueError(f"unknown producer: {manifest.get('producer')!r}")

    source = manifest.get("source", {})
    if source.get("repo") != lock["source"]["repo"]:
        raise ValueError("reference source repository does not match source_lock.json")
    if source.get("commit") != lock["source"]["commit"]:
        raise ValueError("reference source commit does not match source_lock.json")
    if source.get("files") != lock["source"]["files"]:
        raise ValueError("reference source blob map does not match source_lock.json")

    checkpoint = manifest.get("checkpoint", {})
    if checkpoint.get("repo_id") != lock["checkpoint"]["repo_id"]:
        raise ValueError("reference checkpoint repo does not match source_lock.json")
    if checkpoint.get("revision") != lock["checkpoint"]["revision"]:
        raise ValueError("reference checkpoint revision does not match source_lock.json")
    if checkpoint.get("files") != lock["checkpoint"]["files"]:
        raise ValueError("reference checkpoint file hashes do not match source_lock.json")

    profile = manifest.get("runner_profile")
    if profile not in lock["runner_profiles"]:
        raise ValueError(f"unknown runner profile: {profile!r}")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    for field in ("requested", "generation_config", "effective"):
        if not isinstance(sampling.get(field), dict):
            raise ValueError(f"sampling.{field} must be an object")
    if manifest["producer"] == "official_vllm" and not sampling["effective"]:
        raise ValueError("official_vllm references must record effective sampling parameters")

    if not isinstance(manifest.get("software"), dict):
        raise ValueError("software must be an object")
    if not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("cases must contain at least one parity case")
    if not isinstance(manifest.get("tensor_index"), dict):
        raise ValueError("tensor_index must be an object")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("files must be an object")


def validate_bundle(root: Path, *, lock: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    tensors_path = root / "tensors.safetensors"
    if not manifest_path.is_file() or not tensors_path.is_file():
        raise ValueError("reference bundle needs manifest.json and tensors.safetensors")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, lock=lock)
    expected_hash = manifest["files"].get("tensors.safetensors")
    actual_hash = sha256_file(tensors_path)
    if expected_hash != actual_hash:
        raise ValueError(f"tensors.safetensors hash mismatch: expected {expected_hash}, got {actual_hash}")

    from safetensors import safe_open

    with safe_open(tensors_path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        expected_keys = set(manifest["tensor_index"])
        if actual_keys != expected_keys:
            raise ValueError(
                f"tensor keys differ: missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        for key in actual_keys:
            tensor = handle.get_tensor(key)
            expected = manifest["tensor_index"][key]
            if list(tensor.shape) != expected["shape"]:
                raise ValueError(f"tensor shape mismatch for {key}")
            if str(tensor.dtype).removeprefix("torch.") != expected["dtype"]:
                raise ValueError(f"tensor dtype mismatch for {key}")
    return manifest
