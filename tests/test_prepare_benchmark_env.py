"""Guard platform and GPU isolation before environment setup mutates the filesystem."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "profile", ["streamvln", "pi05", "qwenvl", "cosmos", "dm05", "gr00t", "openvla-oft", "lingbot-vla"]
)
@pytest.mark.parametrize(
    ("platform", "version", "capability"),
    [("thor", "2.13.0+cu132", [11, 0]), ("4090", "2.13.0+cu129", [8, 9])],
)
def test_platform_requires_its_cuda_build_and_one_visible_gpu(
    profile: str, platform: str, version: str, capability: list[int]
) -> None:
    validate_runtime = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmarks" / f"{profile}-benchmark" / "setup_env.py")
    )["validate_runtime"]
    runtime = {"torch": version, "capability": capability, "device_count": 1}
    validate_runtime(runtime, platform)
    with pytest.raises(ValueError, match="exactly one GPU"):
        validate_runtime({**runtime, "device_count": 2}, platform)
    with pytest.raises(ValueError, match="requires torch"):
        validate_runtime({**runtime, "torch": "2.10.0+cu128"}, platform)
    with pytest.raises(ValueError, match="requires torch"):
        validate_runtime({**runtime, "capability": [8, 0]}, platform)
