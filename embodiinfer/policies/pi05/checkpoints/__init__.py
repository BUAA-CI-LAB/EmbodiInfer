"""Pi0.5 checkpoint detection and the common loading entry point."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .openpi import OpenPiConfig


def checkpoint_format(root: Path) -> str:
    """Recognize supported layouts, rejecting ambiguous or incomplete exports."""
    orbax = (root / "params" / "_METADATA").is_file()
    torch_weights = (root / "model.safetensors").is_file()
    if orbax and torch_weights:
        raise ValueError("checkpoint contains both Orbax and PyTorch weights; use separate directories")
    if orbax:
        return "openpi-orbax"
    if (root / "openpi_config.json").is_file() and torch_weights:
        return "openpi-pytorch"
    if torch_weights and (root / "config.json").is_file():
        return "lerobot"
    raise ValueError(f"unsupported or incomplete pi05 checkpoint: {root}")


def load_checkpoint(
    checkpoint: str,
    *,
    load_device: str | None = None,
    checkpoint_config: Mapping[str, Any] | None = None,
    low_cpu_mem_usage: bool = False,
) -> tuple[Any, OpenPiConfig | None]:
    """Load model modules and the associated processor recipe.

    LeRobot Hub IDs retain their usual resolution. OpenPI requires a local
    directory with explicit training semantics, never a dataset-name guess.
    """
    root = Path(checkpoint).expanduser()
    if checkpoint_config is not None and not root.is_dir():
        raise ValueError("OpenPI checkpoints require a local directory")
    if checkpoint_config is not None and (root / "model.safetensors").is_file():
        if (root / "params" / "_METADATA").is_file():
            raise ValueError("checkpoint contains both Orbax and PyTorch weights; use separate directories")
        dialect = "openpi-pytorch"
    else:
        dialect = checkpoint_format(root) if root.is_dir() else "lerobot"
    if dialect == "lerobot":
        from .lerobot import load_lerobot_checkpoint

        return load_lerobot_checkpoint(
            checkpoint, load_device=load_device, low_cpu_mem_usage=low_cpu_mem_usage
        ), None

    if low_cpu_mem_usage:
        raise NotImplementedError("low_cpu_mem_usage currently supports LeRobot PI0.5 checkpoints")

    from .openpi import load_openpi_checkpoint

    return load_openpi_checkpoint(
        root,
        load_device=load_device,
        checkpoint_config=checkpoint_config,
        use_orbax=dialect == "openpi-orbax",
    )


__all__ = ["checkpoint_format", "load_checkpoint"]
