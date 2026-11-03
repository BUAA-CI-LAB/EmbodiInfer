"""Dependency and device checks shared by the pinned FlashInfer adapters."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import ModuleType

import torch

FLASHINFER_VERSION = "0.7.2.post1"


def availability() -> tuple[bool, str | None]:
    """Probe installation and hardware without importing or compiling FlashInfer."""
    if sys.platform != "linux":
        return False, "FlashInfer backends require Linux"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        return False, "FlashInfer backends require a CUDA SM80+ device"
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return False, f"Install the optional flashinfer-python=={FLASHINFER_VERSION} dependency"
    if version != FLASHINFER_VERSION:
        return False, f"FlashInfer adapters require {FLASHINFER_VERSION}, found {version}"
    return True, None


def load_module(name: str) -> ModuleType:
    """Load the selected optional API outside capture with an actionable failure."""
    available, reason = availability()
    if not available:
        raise RuntimeError(reason)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare FlashInfer backends before CUDA Graph capture")
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Cannot load {name}; check the FlashInfer/CUDA installation: {exc}") from exc
