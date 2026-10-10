"""Measured Pi05 execution combinations, independent of calibration artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from ....layers.config import OperatorBackends
from .config import MlpLayerPrecision, Pi05OptimizationConfig

_DEVICES = ("thor", "spark", "orin", "4090")
_PRECISIONS = ("bf16", "fp8", "nvfp4", "mixed")


def resolve_preset(
    device: str, preset: str, *, precision: str, calibration: str | Path
) -> Pi05OptimizationConfig:
    """Expand a supported combination and load ranges only for low precision."""
    if device not in _DEVICES:
        raise ValueError(f"Unknown Pi05 device {device!r}; choose from {_DEVICES}")
    if precision not in _PRECISIONS:
        raise ValueError(f"Unknown Pi05 precision {precision!r}; choose from {_PRECISIONS}")
    if preset == "strict":
        if precision != "bf16":
            raise ValueError("The strict preset requires BF16; low precision uses the RLinf preset")
        return Pi05OptimizationConfig(hardware=device)
    if preset != "rlinf":
        raise ValueError("Pi05 preset must be strict or rlinf")
    if device not in ("thor", "spark"):
        raise ValueError("The RLinf preset has launch profiles only for Thor and Spark")

    config = Pi05OptimizationConfig(
        hardware=device,
        numerics="rlinf",
        activation="gelu_pytorch_exact",
        fused_mlp=True,
        batch_cameras=True,
        compact_prefix=True,
        prefix_kv_only=True,
        reuse_action_context=True,
        attention="folded_flash" if device == "thor" else "query_major",
        operators=OperatorBackends(
            paired_gelu="triton_exact",
            projection="torch_matmul",
            rotary="cuda",
            gelu_mul="cuda_lookup" if device == "thor" else "cuda",
        ),
    )
    if precision == "bf16":
        return config

    source = (
        files(__package__).joinpath("calibration/rlinf_libero.json")
        if calibration == "rlinf_libero"
        else Path(calibration)
    )
    data = json.loads(source.read_text())
    if not isinstance(data, dict) or (
        type(data.get("schema_version")) is not int
        or data.get("schema_version") != 1
        or data.get("numerics") != config.numerics
        or data.get("activation") != config.activation
    ):
        raise ValueError("Calibration must identify the matching schema and RLinf exact-GELU contract")
    ranges = data.get("devices", {}).get(device)
    if ranges is None:
        raise ValueError(f"Calibration has no ranges for {device}")
    layers = {}
    for tower, enabled in (
        ("action", precision in ("fp8", "mixed")),
        ("prefix", precision in ("nvfp4", "mixed")),
    ):
        if enabled:
            values = tuple(MlpLayerPrecision(**row) for row in ranges.get(f"{tower}_layers", ()))
            if len(values) != 18:
                raise ValueError(f"Calibration requires 18 {tower} layer ranges")
            expected = "fp8" if tower == "action" else "nvfp4"
            if not any(expected in (layer.gate_up, layer.down) for layer in values) or any(
                value not in ("bf16", expected) for layer in values for value in (layer.gate_up, layer.down)
            ):
                raise ValueError(
                    f"Calibration {tower} formats must select {expected} with optional BF16 protection"
                )
            layers[f"{tower}_layers"] = values
    if data.get("checkpoint_sha256") is None:
        raise ValueError("Calibration requires a checkpoint SHA256")
    return replace(config, **layers, checkpoint_sha256=data["checkpoint_sha256"])
