"""Opt-in Pi05 operator plans, separate from engine execution configuration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ....layers.config import OperatorBackends
from ....layers.quantization import Precision


@dataclass(frozen=True)
class MlpLayerPrecision:
    """Calibrated gate/up and down formats for one transformer MLP."""

    gate_up: Precision = "bf16"
    down: Precision = "bf16"
    gate_up_max: float = 1.0
    down_max: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid precision names and activation ranges before packing."""
        if any(value not in ("bf16", "fp8", "nvfp4") for value in (self.gate_up, self.down)):
            raise ValueError("MLP precision must be bf16, fp8 or nvfp4")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (self.gate_up_max, self.down_max)
        ):
            raise ValueError("Activation maxima must be finite and positive")


@dataclass(frozen=True)
class Pi05OptimizationConfig:
    """Immutable selection of opt-in inference operators.

    The default enables strict pointwise RMSNorm/residual fusion and K/V storage
    reuse. Paired GEMMs and alternative attention require explicit selection;
    they change floating-point accumulation. Paired-GEMM profiles cover
    B1/horizon10/10steps; benchmark results apply only to their measured settings.
    Use ``from_preset`` for deployment; individual fields support operator
    ablations and custom backends. The default preserves LeRobot numerics.
    """

    hardware: Literal["thor", "spark", "orin", "4090"] | None = None
    norm_fusion: bool = True
    kv_workspace: bool = True
    fused_mlp: bool = False
    attention: Literal["reference", "query_major", "folded_flash"] = "reference"
    action_layers: tuple[MlpLayerPrecision, ...] = ()
    checkpoint_sha256: str | None = None
    activation: Literal["gelu_pytorch_tanh", "gelu_pytorch_exact"] = "gelu_pytorch_tanh"
    schema_version: int = 1
    operators: OperatorBackends = field(default_factory=OperatorBackends)
    prefix_layers: tuple[MlpLayerPrecision, ...] = ()
    numerics: Literal["lerobot", "rlinf"] = "lerobot"
    batch_cameras: bool = False
    compact_prefix: bool = False
    prefix_kv_only: bool = False
    reuse_action_context: bool = False

    def __post_init__(self) -> None:
        """Validate semantics and require a checkpoint-bound mixed precision recipe."""
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.hardware not in (None, "thor", "spark", "orin", "4090")
        ):
            raise ValueError("Unsupported Pi05 optimization schema or hardware profile")
        expected = {"lerobot": "gelu_pytorch_tanh", "rlinf": "gelu_pytorch_exact"}.get(self.numerics)
        if expected is None or self.activation != expected:
            raise ValueError("Pi05 numerics require the matching exact or tanh GELU contract")
        if not isinstance(self.operators, OperatorBackends):
            raise ValueError("operators must be an OperatorBackends configuration")
        if any(
            type(value) is not bool
            for value in (
                self.norm_fusion,
                self.kv_workspace,
                self.fused_mlp,
                self.batch_cameras,
                self.compact_prefix,
                self.prefix_kv_only,
                self.reuse_action_context,
            )
        ):
            raise ValueError("Pi05 optimization switches must be booleans")
        if self.attention not in ("reference", "query_major", "folded_flash"):
            raise ValueError("Unknown optimized attention implementation")
        for name in ("action_layers", "prefix_layers"):
            layers = getattr(self, name)
            if not isinstance(layers, tuple) or any(
                not isinstance(layer, MlpLayerPrecision) for layer in layers
            ):
                raise ValueError(f"{name} must be a tuple of MlpLayerPrecision values")
        if (
            self.fused_mlp
            and self.hardware not in ("thor", "spark")
            and self.operators.paired_gelu in ("triton_lookup", "triton_exact")
        ):
            raise ValueError("Paired GEMMs require an explicit Thor or Spark launch profile")
        if self.mixed_precision and self.checkpoint_sha256 is None:
            raise ValueError("Mixed precision requires a calibration checkpoint SHA256")
        if self.checkpoint_sha256 is not None and (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must contain 64 lowercase hexadecimal characters")

    @property
    def mixed_precision(self) -> bool:
        """Whether either tower uses a calibrated low precision MLP format."""
        return any(
            layer.gate_up != "bf16" or layer.down != "bf16"
            for layer in (*self.action_layers, *self.prefix_layers)
        )

    @property
    def capability(self) -> tuple[int, int] | None:
        """Return the selected profile's capability, or allow generic BF16 fusion."""
        return {"thor": (11, 0), "spark": (12, 1), "orin": (8, 7), "4090": (8, 9)}.get(self.hardware)

    @classmethod
    def from_json(cls, path: str | Path) -> Pi05OptimizationConfig:
        """Load a standalone recipe with an explicit activation/numerics contract."""
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict) or values.get("activation") not in (
            "gelu_pytorch_tanh",
            "gelu_pytorch_exact",
        ):
            raise ValueError("Recipe must identify its exact or tanh GELU calibration contract")
        for name in ("action_layers", "prefix_layers"):
            if name in values:
                values[name] = tuple(MlpLayerPrecision(**row) for row in values[name])
        if "operators" in values:
            if not isinstance(values["operators"], dict):
                raise ValueError("operators must be a mapping of implementation names")
            values["operators"] = OperatorBackends(**values["operators"])
        return cls(**values)

    @classmethod
    def from_preset(
        cls,
        device: Literal["thor", "spark", "orin", "4090"],
        preset: Literal["strict", "rlinf"] = "strict",
        *,
        precision: Literal["bf16", "fp8", "nvfp4", "mixed"] = "bf16",
        calibration: str | Path = "rlinf_libero",
    ) -> Pi05OptimizationConfig:
        """Resolve a device preset without importing kernels or probing CUDA.

        ``strict`` preserves LeRobot numerics and supports all listed devices.
        ``rlinf`` selects the optimized B1/H10/10-step Thor/Spark profile.
        FP8 selects protected action projections; NVFP4 selects prefix MLPs;
        mixed combines both. Low precision requires matching calibration data;
        the bundled ranges belong exclusively to RLinf-Pi05-LIBERO-SFT.
        Engine graph configuration and request shapes remain caller-owned.
        """
        from .presets import resolve_preset

        return resolve_preset(device, preset, precision=precision, calibration=calibration)

    def to_json(self, path: str | Path) -> None:
        """Export deployment settings without benchmark or calibration dependencies."""
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")
