"""Shared helpers for the three ActiveVLN reference producers."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scripts.activevln.schema import SCHEMA_NAME, SCHEMA_VERSION, load_lock, sha256_file, tensor_index


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("input manifest must contain a non-empty cases list")
    for case in cases:
        if not case.get("case_id") or not isinstance(case.get("turns"), list) or not case["turns"]:
            raise ValueError("each case needs case_id and a non-empty turns list")
        for turn in case["turns"]:
            if not turn.get("instruction") or not turn.get("image"):
                raise ValueError("each turn needs instruction and image")
    return cases


def software_versions(*, producer: str) -> dict[str, str | None]:
    import transformers

    software: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vllm": None,
    }
    if producer == "official_vllm":
        try:
            import vllm
        except ImportError:
            pass
        else:
            software["vllm"] = vllm.__version__
    return software


def base_manifest(
    producer: str,
    cases: list[dict[str, Any]],
    *,
    requested_sampling: dict[str, Any],
    generation_config: dict[str, Any],
    effective_sampling: dict[str, Any],
) -> dict[str, Any]:
    lock = load_lock()
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "producer": producer,
        "source": lock["source"],
        "checkpoint": lock["checkpoint"],
        "software": software_versions(producer=producer),
        "runner_profile": "official_eval_r2r",
        "sampling": {
            "requested": requested_sampling,
            "generation_config": generation_config,
            "effective": effective_sampling,
            "seed": 0,
        },
        "cases": cases,
        "tensor_index": {},
        "files": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_bundle(root: Path, manifest: dict[str, Any], tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=False)
    tensor_path = root / "tensors.safetensors"
    cpu_tensors = {key: value.detach().contiguous().cpu() for key, value in tensors.items()}
    save_file(cpu_tensors, tensor_path)
    manifest["tensor_index"] = tensor_index(cpu_tensors)
    manifest["files"]["tensors.safetensors"] = sha256_file(tensor_path)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
