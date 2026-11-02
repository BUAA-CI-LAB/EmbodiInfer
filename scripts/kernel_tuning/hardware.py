"""Target-runtime observations and portable hardware guidance for kernel searches."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .contracts import ContractError


def render_notes(environment: dict[str, Any] | None) -> str:
    """Separate observed hardware facts from optimization hypotheses and unknowns."""
    environment = environment or {}
    identity = environment.get("identity", environment)
    capability = identity.get("capability")
    guidance = [
        "Use the observed compute capability and compiler, not a GPU family name, to select instructions.",
        "Treat tile sizes, warps, stages, split counts and fusion as hypotheses requiring target measurements.",
        "For small batches compare launch overhead, weight traffic, occupancy and register spills separately.",
        "Do not infer achievable bandwidth from capacity, theoretical peak rates or results on another board.",
        "Compare every recorded layout; a kernel win does not establish whole-model savings.",
        "Keep precision, seeds, attention masks and graph replay semantics frozen.",
    ]
    if capability == [8, 7]:
        guidance.append(
            "CC 8.7 target: do not use Hopper/Blackwell-only instructions or assume their tuning parameters apply."
        )
    elif capability == [11, 0]:
        guidance.append(
            "CC 11.0 target: do not assume B200/B300 CC 10.x instructions or launch configurations transfer."
        )
    return (
        "# Target hardware context\n\n"
        "Observed facts (probe/evaluator source; no performance calibration):\n\n```json\n"
        + json.dumps(identity, indent=2, ensure_ascii=False)
        + "\n```\n\nMeasurement conditions:\n\n```json\n"
        + json.dumps(environment.get("conditions", {}), indent=2, ensure_ascii=False)
        + "\n```\n\nOptimization guidance (hypotheses, not measured winners):\n\n"
        + "\n".join(f"- {line}" for line in guidance)
        + "\n\nUnknown unless supplied in frozen task/HARDWARE.md: measured bandwidth, launch floor, "
        "thermal stability and prior winning configurations. Record evidence before claiming these.\n"
    )


def _driver_properties(device: str) -> dict[str, Any]:
    """Query the CUDA Driver API when the model's Torch environment is unavailable.

    Attribute numbers are the stable CUdevice_attribute values in cuda.h.
    Driver probing does not establish Torch/Triton/checkpoint readiness.
    """
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ContractError("Hardware device must be cuda:INDEX")
    try:
        library = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise ContractError("CUDA driver is unavailable; use a prepared target machine") from exc

    def call(name: str, arguments: list[Any], types: list[Any]) -> None:
        function = getattr(library, name)
        function.argtypes, function.restype = types, ctypes.c_int
        code = function(*arguments)
        if code != 0:
            raise ContractError(f"CUDA driver {name} failed with error {code}")

    integer = ctypes.c_int
    pointer = ctypes.POINTER(integer)
    call("cuInit", [0], [ctypes.c_uint])
    ordinal = integer()
    call("cuDeviceGet", [ctypes.byref(ordinal), int(device[5:])], [pointer, integer])
    name = ctypes.create_string_buffer(256)
    call("cuDeviceGetName", [name, len(name), ordinal], [ctypes.c_char_p, integer, integer])
    memory = ctypes.c_size_t()
    call("cuDeviceTotalMem_v2", [ctypes.byref(memory), ordinal], [ctypes.POINTER(ctypes.c_size_t), integer])
    attributes = {
        "shared_memory_per_block": 8,
        "warp_size": 10,
        "multi_processor_count": 16,
        "L2_cache_size": 38,
        "max_threads_per_multi_processor": 39,
        "major": 75,
        "minor": 76,
        "shared_memory_per_multiprocessor": 81,
        "regs_per_multiprocessor": 82,
    }
    result = {
        "name": name.value.decode(),
        "total_memory": memory.value,
        "device": device,
        "machine": platform.machine(),
        "torch": "not probed",
        "cuda": "driver API",
    }
    for key, attribute in attributes.items():
        value = integer()
        call("cuDeviceGetAttribute", [ctypes.byref(value), attribute, ordinal], [pointer, integer, integer])
        result[key] = value.value
    result["capability"] = [result.pop("major"), result.pop("minor")]
    return result


def probe(device: str = "cuda:0") -> dict[str, Any]:
    """Query CUDA properties and read-only power state; fall back to the driver API."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not torch.cuda.is_available():
        identity = _driver_properties(device)
        source = "CUDA Driver API (no model-runtime readiness claim)"
    else:
        properties = torch.cuda.get_device_properties(device)
        fields = (
            "name",
            "total_memory",
            "multi_processor_count",
            "warp_size",
            "max_threads_per_multi_processor",
            "regs_per_multiprocessor",
            "shared_memory_per_block",
            "shared_memory_per_multiprocessor",
            "L2_cache_size",
        )
        identity = {name: getattr(properties, name) for name in fields if hasattr(properties, name)}
        identity.update(
            {
                "capability": list(torch.cuda.get_device_capability(device)),
                "device": device,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "machine": platform.machine(),
            }
        )
        source = "Torch CUDA properties"
    conditions = {}
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel:
        result = subprocess.run([nvpmodel, "-q"], capture_output=True, text=True, timeout=10, check=False)
        conditions["nvpmodel"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return {"identity": identity, "conditions": conditions, "source": source}


def write(output: Path, device: str = "cuda:0") -> dict[str, Any]:
    """Write a new target observation bundle usable through ``--hardware-notes``."""
    if output.exists():
        raise ContractError(f"{output} already exists")
    environment = probe(device)
    output.mkdir(parents=True)
    atomic_json(output / "hardware.json", environment)
    (output / "HARDWARE.md").write_text(render_notes(environment), encoding="utf-8")
    return environment
