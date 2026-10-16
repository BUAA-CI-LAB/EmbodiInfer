"""Verified, hardware-specific operator kernels grouped into kernel profiles.

Each subdirectory is one profile (for example ``thor`` or ``rtx4090``) holding a
``profile.json`` manifest and one self-contained module or package per tuned
operator (``scripts.kernel_tuning install`` writes packages).
Profiles are produced by the kernel-tuning tooling and selected at runtime through
:class:`embodiinfer.layers.ops.KernelSelection`; nothing here is used by default.

``profile.json``::

    {
      "schema": 1,
      "match": {"device_name": "NVIDIA Thor", "capability": [11, 0],
                "torch": "2.13.0+cu132", "triton": "3.7.1"},
      "ops": {
        "gated_residual": {"module": "gated_residual", "function": "run",
                           "applies": "applies", "sha256": "<source_digest of the package>"}
      }
    }

``applies`` names an optional predicate taking the operator's arguments and
returning whether the kernel supports them; unsupported calls fall back to the
policy's preferred implementation. Modules may import only Torch and Triton.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILE_ROOT = Path(__file__).resolve().parent
_SCHEMA = 1
_MATCH_KEYS = ("device_name", "capability", "torch", "triton")


@dataclass(frozen=True)
class TunedKernel:
    """One tuned operator implementation and its applicability predicate."""

    fn: Callable[..., Any]
    applies: Callable[..., bool] | None


@dataclass(frozen=True)
class KernelProfile:
    """A validated profile manifest; kernels load on first use."""

    name: str
    root: Path
    match: Mapping[str, Any]
    ops: Mapping[str, Mapping[str, Any]]
    _loaded: dict[str, TunedKernel] = field(default_factory=dict, compare=False, repr=False)

    def matches(self, environment: Mapping[str, Any]) -> bool:
        """Return whether every match key equals the current environment."""
        return all(_normalize(self.match[key]) == _normalize(environment.get(key)) for key in _MATCH_KEYS)

    def kernel(self, op: str) -> TunedKernel:
        """Import ``op``'s module after checking its recorded digest."""
        if op in self._loaded:
            return self._loaded[op]
        entry = self.ops[op]
        path = self.root / entry["module"]
        if source_digest(path) != entry["sha256"]:
            raise ValueError(f"kernel profile {self.name!r}: {path.name} does not match its recorded sha256")
        name = f"embodiinfer_tuned_{self.name}_{path.stem}"
        if path.is_dir():
            spec = importlib.util.spec_from_file_location(
                name, path / "__init__.py", submodule_search_locations=[str(path)]
            )
        else:
            spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"kernel profile {self.name!r}: cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        # Packages import their own files relatively, which needs the parent registered.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        applies = entry.get("applies")
        kernel = TunedKernel(
            getattr(module, entry["function"]), getattr(module, applies) if applies else None
        )
        self._loaded[op] = kernel
        return kernel


def source_digest(path: Path) -> str:
    """SHA-256 of a module file, or of every file (by relative path) in a package directory."""
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file() and "__pycache__" not in item.parts
    }
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _normalize(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def load_profile(root: Path) -> KernelProfile:
    """Read and validate one profile directory without importing its kernels."""
    manifest = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != _SCHEMA:
        raise ValueError(f"kernel profile {root.name!r}: unsupported schema {manifest.get('schema')!r}")
    match = manifest["match"]
    missing = [key for key in _MATCH_KEYS if key not in match]
    if missing:
        raise ValueError(f"kernel profile {root.name!r}: match is missing {missing}")
    for op, entry in manifest["ops"].items():
        for key in ("module", "function", "sha256"):
            if key not in entry:
                raise ValueError(f"kernel profile {root.name!r}: operator {op!r} is missing {key!r}")
    return KernelProfile(root.name, root, match, manifest["ops"])


def available_profiles(root: Path = PROFILE_ROOT) -> dict[str, KernelProfile]:
    """Return every profile directory under ``root`` by name."""
    return {
        path.name: load_profile(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "profile.json").is_file()
    }


def current_environment() -> dict[str, Any]:
    """Describe the active CUDA device and Torch/Triton versions for profile matching."""
    import torch

    environment: dict[str, Any] = {"torch": torch.__version__}
    try:
        import triton

        environment["triton"] = triton.__version__
    except ImportError:
        environment["triton"] = None
    if torch.cuda.is_available():
        environment["device_name"] = torch.cuda.get_device_name()
        environment["capability"] = list(torch.cuda.get_device_capability())
    return environment


__all__ = [
    "KernelProfile",
    "PROFILE_ROOT",
    "TunedKernel",
    "available_profiles",
    "current_environment",
    "load_profile",
    "source_digest",
]
