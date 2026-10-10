"""Device optimization presets and independent numerical/precision selection."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from ....layers.config import OperatorBackends
from ....layers.quantization import Precision
from .config import MlpLayerPrecision, Pi05OptimizationConfig

_DEVICES = ("thor", "spark", "orin", "4090")
_ACTIVATIONS = {"lerobot": "gelu_pytorch_tanh", "openpi_rlinf": "gelu_pytorch_exact"}


def resolve_preset(
    device: str,
    preset: str,
    *,
    numerics: str,
    prefix_mlp: Precision,
    action_mlp: Precision,
    calibration: str | Path | None,
) -> Pi05OptimizationConfig:
    """Compose execution choices, preserving independently selected semantics."""
    if device not in _DEVICES:
        raise ValueError(f"Unknown Pi05 device {device!r}; choose from {_DEVICES}")
    if preset not in ("strict", "optimized"):
        raise ValueError("Pi05 preset must be strict or optimized")
    if numerics not in _ACTIVATIONS:
        raise ValueError("Pi05 numerics must be lerobot or openpi_rlinf")
    formats = {"prefix": prefix_mlp, "action": action_mlp}
    if any(value not in ("bf16", "fp8", "nvfp4") for value in formats.values()):
        raise ValueError("MLP precision must be bf16, fp8 or nvfp4")
    if preset == "optimized" and device not in ("thor", "spark"):
        raise ValueError("The optimized preset has launch profiles only for Thor and Spark")
    if device == "orin" and any(value != "bf16" for value in formats.values()):
        raise ValueError("Orin does not support these native low-precision MLP projections")
    if device == "4090" and "nvfp4" in formats.values():
        raise ValueError("NVFP4 requires Blackwell; RTX 4090 supports BF16/FP8 projections")

    operators = OperatorBackends()
    if numerics == "openpi_rlinf":
        operators = replace(
            operators,
            paired_gelu="triton_exact",
            projection="torch_matmul",
            rotary="cuda",
            gelu_mul="cuda_lookup" if device == "thor" else "cuda",
        )
    config = Pi05OptimizationConfig(
        hardware=device,
        numerics=numerics,
        activation=_ACTIVATIONS[numerics],
        operators=operators,
    )
    if preset == "optimized":
        config = replace(
            config,
            fused_mlp=True,
            batch_cameras=True,
            compact_prefix=True,
            prefix_kv_only=True,
            reuse_action_context=True,
            attention="folded_flash" if device == "thor" else "query_major",
        )
    if all(value == "bf16" for value in formats.values()):
        return config
    return _apply_calibration(config, formats, calibration)


def _apply_calibration(
    config: Pi05OptimizationConfig,
    formats: dict[str, Precision],
    calibration: str | Path | None,
) -> Pi05OptimizationConfig:
    if calibration is None:
        if config.numerics != "openpi_rlinf":
            raise ValueError("LeRobot low precision requires its own calibration data path")
        calibration = "rlinf_libero"
    source = (
        files(__package__).joinpath("calibration/rlinf_libero.json")
        if calibration == "rlinf_libero"
        else Path(calibration)
    )
    data = json.loads(source.read_text())
    if not isinstance(data, dict) or (
        type(data.get("schema_version")) is not int
        or data.get("schema_version") != 2
        or data.get("numerics") != config.numerics
        or data.get("activation") != config.activation
    ):
        raise ValueError("Calibration must identify schema 2 and the matching numerical contract")
    devices = data.get("devices")
    ranges = devices.get(config.hardware) if isinstance(devices, dict) else None
    if not isinstance(ranges, dict):
        raise ValueError(f"Calibration has no ranges for {config.hardware}")
    layers = {}
    for tower, precision in formats.items():
        if precision == "bf16":
            continue
        choices = ranges.get(tower)
        rows = choices.get(precision) if isinstance(choices, dict) else None
        if not isinstance(rows, list) or len(rows) != 18:
            raise ValueError(f"Calibration requires 18 {tower} MLP layer ranges for {precision}")
        values = tuple(MlpLayerPrecision(**row) for row in rows)
        if not any(precision in (layer.gate_up, layer.down) for layer in values) or any(
            value not in ("bf16", precision) for layer in values for value in (layer.gate_up, layer.down)
        ):
            raise ValueError(
                f"Calibration {tower} formats must select {precision} with optional BF16 protection"
            )
        layers[f"{tower}_layers"] = values
    if data.get("checkpoint_sha256") is None:
        raise ValueError("Calibration requires a checkpoint SHA256")
    return replace(config, **layers, checkpoint_sha256=data["checkpoint_sha256"])
