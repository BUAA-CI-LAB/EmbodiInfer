"""Tensor argument recipes and storage-preserving replay in the model runtime."""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from ..contracts import ContractError


def tensor_spec(value: torch.Tensor) -> dict[str, Any]:
    """Describe layout without copying data or synchronizing a device."""
    if value.layout != torch.strided or value.is_quantized or value.is_conj() or value.is_neg():
        raise ContractError("Only ordinary dense strided tensors have a replay recipe")
    if any(stride < 0 for stride in value.stride()):
        raise ContractError("Negative strides require a dedicated replay adapter")
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "offset": value.storage_offset(),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
    }


def flatten(value: Any, tensors: list[torch.Tensor]) -> Any:
    """Replace tensor leaves with positions, preserving supported argument containers."""
    if isinstance(value, torch.Tensor):
        index = next((i for i, tensor in enumerate(tensors) if tensor is value), None)
        if index is None:
            index = len(tensors)
            tensors.append(value)
        return {"tensor": index}
    if (
        type(value).__module__ == "embodiinfer.backend.triton.sampling"
        and type(value).__name__ == "GreedyWorkspace"
    ):
        return {
            "greedy_workspace": {
                field.name: flatten(getattr(value, field.name), tensors) for field in fields(value)
            }
        }
    if isinstance(value, (tuple, list)):
        return {"tuple" if isinstance(value, tuple) else "list": [flatten(x, tensors) for x in value]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {"dict": {key: flatten(item, tensors) for key, item in value.items()}}
    if value is None or isinstance(value, (bool, str, int)):
        return {"constant": value}
    if isinstance(value, float) and math.isfinite(value):
        return {"constant": value}
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return {"torch_value": str(value)}
    raise ContractError(
        f"No replay recipe for argument type {type(value).__module__}.{type(value).__qualname__}"
    )


def restore(recipe: Any, tensors: list[torch.Tensor]) -> Any:
    """Rebuild a captured argument tree with current input tensors."""
    kind, value = next(iter(recipe.items()))
    if kind == "tensor":
        return tensors[value]
    if kind == "constant":
        return value
    if kind == "torch_value":
        return getattr(torch, value[6:]) if value.startswith("torch.") else torch.device(value)
    if kind in ("tuple", "list"):
        result = [restore(x, tensors) for x in value]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict":
        return {key: restore(item, tensors) for key, item in value.items()}
    if kind == "greedy_workspace":
        from embodiinfer.backend.triton.sampling import GreedyWorkspace

        return GreedyWorkspace(**{key: restore(item, tensors) for key, item in value.items()})
    raise ContractError(f"Unknown argument recipe: {kind}")


def storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identify shared storage within one call, including zero-sized storage objects."""
    return str(tensor.device), tensor.untyped_storage()._cdata


def describe_inputs(tensors: list[torch.Tensor]) -> list[dict[str, Any]]:
    """Assign stable per-call alias groups to tensor inputs."""
    groups: dict[tuple[Any, ...], int] = {}
    result = []
    for tensor in tensors:
        key = storage_key(tensor)
        result.append({**tensor_spec(tensor), "storage": groups.setdefault(key, len(groups))})
    for group in groups.values():
        if len({spec["dtype"] for spec in result if spec["storage"] == group}) != 1:
            raise ContractError("Reinterpreted storage with different dtypes needs a dedicated adapter")
    return result


def extent(spec: dict[str, Any]) -> int:
    """Number of storage elements needed for a layout, including its offset."""
    if not all(spec["shape"]):
        return spec["offset"]
    return spec["offset"] + 1 + sum((n - 1) * stride for n, stride in zip(spec["shape"], spec["stride"]))


def materialize(specs: list[dict[str, Any]], root: Path, seed: int) -> list[torch.Tensor]:
    """Restore captured storage and aliasing, or generate unconstrained floating storage."""
    buffers = {}
    for group in {spec["storage"] for spec in specs}:
        members = [spec for spec in specs if spec["storage"] == group]
        spec = members[0]
        dtype = getattr(torch, spec["dtype"])
        size = max(extent(member) for member in members)
        generator = torch.Generator(device=spec["device"]).manual_seed(seed + group)
        if "fixture" in spec:
            data = bytearray((root / spec["fixture"]).read_bytes())
            buffer = (
                torch.frombuffer(data, dtype=dtype).clone().to(spec["device"])
                if data
                else torch.empty(0, dtype=dtype, device=spec["device"])
            )
            if buffer.numel() != size:
                raise ContractError("Captured fixture size differs from its storage layout")
            # Different seeds exercise changed floating values while respecting integer/index fixtures.
            if buffer.is_floating_point() and seed:
                buffer = buffer * (1.0 + (seed % 7) / 128.0)
        elif dtype.is_floating_point:
            buffer = torch.randn(size, dtype=dtype, device=spec["device"], generator=generator)
        else:
            raise ContractError("Non-floating inputs require a valid captured fixture")
        buffers[group] = buffer
    return [
        buffers[spec["storage"]].as_strided(spec["shape"], spec["stride"], spec["offset"]) for spec in specs
    ]


def clone_inputs(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Clone backing storage once per alias group instead of breaking views apart."""
    specs = describe_inputs(inputs)
    buffers = {}
    for tensor, spec in zip(inputs, specs):
        group = spec["storage"]
        if group not in buffers:
            size = max(extent(item) for item in specs if item["storage"] == group)
            buffers[group] = tensor.as_strided((size,), (1,), 0).clone()
    return [
        buffers[spec["storage"]].as_strided(spec["shape"], spec["stride"], spec["offset"]) for spec in specs
    ]


def tensor_outputs(result: Any) -> list[torch.Tensor]:
    """Flatten tensor-only outputs without silently dropping data-dependent scalar results."""
    tensors: list[torch.Tensor] = []
    recipe = flatten(result, tensors)

    def check(node: Any) -> None:
        if "greedy_workspace" in node:
            raise ContractError("Workspace outputs require explicit tensor returns")
        if "constant" in node and node["constant"] is not None:
            raise ContractError("Non-tensor results require a dedicated output contract")
        for key in ("tuple", "list"):
            for child in node.get(key, []):
                check(child)
        for child in node.get("dict", {}).values():
            check(child)

    check(recipe)
    return tensors
