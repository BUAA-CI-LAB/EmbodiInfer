"""Argument-aware RNG classification and exact default-generator randn replay."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ..contracts import ContractError
from .contracts import RNG_OVERLOADS


def _zero_dropout_attention(function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    # The composite call decomposes to a backend overload (flash, efficient,
    # cuDNN, CPU) before Python dispatch; none draws random state without dropout.
    name = str(function)
    if name != "aten.scaled_dot_product_attention.default" and not name.startswith(
        "aten._scaled_dot_product_"
    ):
        return False
    for index, argument in enumerate(function._schema.arguments):
        if argument.name == "dropout_p":
            dropout = kwargs.get(argument.name, args[index] if index < len(args) else argument.default_value)
            return type(dropout) in (int, float) and dropout == 0
    return False


def requires_rng_adapter(function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Treat seeded tags as conditional for SDPA with explicitly zero/default dropout."""
    tags = {tag for tag in getattr(function, "tags", ()) if "nondeterministic" in str(tag)}
    if _zero_dropout_attention(function, args, kwargs):
        tags.discard(torch.Tag.nondeterministic_seeded)
    return bool(tags)


def undefined_outputs(function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[int]:
    """Outputs whose bytes are unspecified: dropout RNG state left uninitialized without dropout."""
    if not _zero_dropout_attention(function, args, kwargs):
        return []
    return [
        i
        for i, value in enumerate(function._schema.returns)
        if value.name in {"philox_seed", "philox_offset"}
    ]


def contract_for(name: str, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Recognize randn with the default CPU/CUDA generator; other RNG calls stay gaps."""
    if name not in RNG_OVERLOADS or kwargs.get("generator") is not None:
        return None
    device = torch.device(kwargs.get("device") or torch.get_default_device())
    if device.type not in {"cpu", "cuda"}:
        return None
    devices = ["cpu"]
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        devices.append(f"cuda:{index}")
    return {"kind": "torch_default", "devices": devices}


def snapshot(devices: list[str]) -> dict[str, torch.Tensor]:
    """Read opaque generator states without consuming random numbers."""
    return {
        device: (torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state(device)).clone()
        for device in devices
    }


def restore(states: dict[str, torch.Tensor]) -> None:
    """Restore every declared default generator from its opaque byte state."""
    for device, state in states.items():
        if device == "cpu":
            torch.set_rng_state(state)
        else:
            torch.cuda.set_rng_state(state, device)


@contextmanager
def preserve(devices: list[str]) -> Iterator[None]:
    """Restore the caller's RNG state on both successful and failed evaluation."""
    saved = snapshot(devices)
    try:
        yield
    finally:
        restore(saved)


class RngReplay:
    """Replay the captured transition at seed 0 and independent states at other seeds."""

    def __init__(self, contract: dict[str, Any], case: dict[str, Any], root: Path, seed: int) -> None:
        self.devices = contract["devices"]
        states = {
            phase: {
                device: torch.tensor(list((root / path).read_bytes()), dtype=torch.uint8)
                for device, path in paths.items()
            }
            for phase, paths in case["rng"].items()
        }
        current = snapshot(self.devices)
        if any(
            state.shape != current[device].shape
            for phase in states.values()
            for device, state in phase.items()
        ):
            raise ContractError("RNG fixture state size differs from the current Torch runtime")
        self.before = (
            states["before"]
            if seed == 0
            else {
                device: torch.Generator(device=device).manual_seed(seed).get_state()
                for device in self.devices
            }
        )
        self.captured_after = states["after"] if seed == 0 else None

    def reset(self) -> None:
        """Restore the selected workload state before a call, outside timed events."""
        restore(self.before)

    def check_reference(self, actual: dict[str, torch.Tensor]) -> None:
        """Require the frozen overload to reproduce its observed seed-0 transition."""
        if self.captured_after is not None:
            self.check(actual, self.captured_after)

    @staticmethod
    def check(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
        """Reject changed generator advancement even when output values are correct."""
        if set(actual) != set(expected) or any(not torch.equal(actual[d], expected[d]) for d in expected):
            raise ContractError("RNG state after the call differs from the reference")
