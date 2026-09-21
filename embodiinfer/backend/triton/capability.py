"""Import-safe capability probing shared by VVLA Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TritonCapability:
    available: bool
    reason: str | None
    version: str | None = None
    device: str | None = None
    compute_capability: tuple[int, int] | None = None


def triton_capability(
    device: torch.device | str | int | None = None,
    *,
    minimum_compute_capability: tuple[int, int] = (8, 0),
) -> TritonCapability:
    """Probe Triton CUDA support without compiling or launching a kernel."""
    try:
        import triton
    except Exception as exc:
        return TritonCapability(False, f"Triton import failed: {exc}")
    version = getattr(triton, "__version__", "unknown")
    if not torch.cuda.is_available():
        return TritonCapability(False, "PyTorch reports that CUDA is unavailable", version)
    try:
        if device is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            resolved = torch.device("cuda", device)
        else:
            resolved = torch.device(device)
        if resolved.type != "cuda":
            return TritonCapability(False, f"device {resolved} is not a CUDA device", version)
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        compute_capability = torch.cuda.get_device_capability(index)
        device_name = torch.cuda.get_device_name(index)
        if compute_capability < minimum_compute_capability:
            required = "sm" + "".join(str(part) for part in minimum_compute_capability)
            actual = "sm" + "".join(str(part) for part in compute_capability)
            return TritonCapability(
                False,
                f"Triton kernels require {required} or newer; {device_name} is {actual}",
                version,
                device_name,
                compute_capability,
            )
        target = triton.runtime.driver.active.get_current_target()
        if getattr(target, "backend", None) != "cuda":
            return TritonCapability(
                False,
                f"active Triton target is {target!r}, not the CUDA backend",
                version,
                device_name,
                compute_capability,
            )
    except Exception as exc:
        return TritonCapability(False, f"Triton CUDA target probe failed: {exc}", version)
    return TritonCapability(True, None, version, device_name, compute_capability)


def require_triton(device: torch.device | str | int | None = None) -> TritonCapability:
    capability = triton_capability(device)
    if not capability.available:
        raise RuntimeError(f"Triton backend is unavailable: {capability.reason}")
    return capability


__all__ = ("TritonCapability", "require_triton", "triton_capability")
