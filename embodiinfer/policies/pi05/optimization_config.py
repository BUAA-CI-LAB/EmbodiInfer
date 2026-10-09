"""Opt-in Pi05 operator plans, separate from engine execution configuration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ...layers.config import OperatorBackends
from ...layers.quantization import Precision


@dataclass(frozen=True)
class ActionLayerPrecision:
    """Calibrated gate/up and down formats for one action transformer layer."""

    gate_up: Precision = "bf16"
    down: Precision = "bf16"
    gate_up_max: float = 1.0
    down_max: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid precision names and activation ranges before packing."""
        if any(value not in ("bf16", "fp8", "nvfp4") for value in (self.gate_up, self.down)):
            raise ValueError("Action precision must be bf16, fp8 or nvfp4")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (self.gate_up_max, self.down_max)
        ):
            raise ValueError("Activation maxima must be finite and positive")


@dataclass(frozen=True)
class Pi05OptimizationConfig:
    """Immutable selection of migrated inference operators.

    The default enables strict pointwise RMSNorm/residual fusion and K/V storage
    reuse. Paired GEMMs and alternative attention require explicit selection;
    they change floating-point accumulation. Hardware profiles originate from
    ccinfer measurements for B1/horizon10/10steps, not migrated performance claims.
    Mixed recipes must be calibrated against this adapter's tanh GELU contract.
    """

    hardware: Literal["thor", "spark"] | None = None
    norm_fusion: bool = True
    kv_workspace: bool = True
    fused_mlp: bool = False
    attention: Literal["reference", "query_major", "folded_flash"] = "reference"
    action_layers: tuple[ActionLayerPrecision, ...] = ()
    checkpoint_sha256: str | None = None
    activation: Literal["gelu_pytorch_tanh"] = "gelu_pytorch_tanh"
    schema_version: int = 1
    operators: OperatorBackends = field(default_factory=OperatorBackends)

    def __post_init__(self) -> None:
        """Validate semantics and require a checkpoint-bound mixed precision recipe."""
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.hardware not in (None, "thor", "spark")
        ):
            raise ValueError("Unsupported Pi05 optimization schema or hardware profile")
        if self.activation != "gelu_pytorch_tanh":
            raise ValueError("Pi05 optimizations must preserve the adapter's tanh GELU")
        if not isinstance(self.operators, OperatorBackends):
            raise ValueError("operators must be an OperatorBackends configuration")
        if any(type(value) is not bool for value in (self.norm_fusion, self.kv_workspace, self.fused_mlp)):
            raise ValueError("Pi05 optimization switches must be booleans")
        if self.attention not in ("reference", "query_major", "folded_flash"):
            raise ValueError("Unknown optimized attention implementation")
        if not isinstance(self.action_layers, tuple) or any(
            not isinstance(layer, ActionLayerPrecision) for layer in self.action_layers
        ):
            raise ValueError("action_layers must be a tuple of ActionLayerPrecision values")
        if self.fused_mlp and self.hardware is None and self.operators.paired_gelu == "triton_lookup":
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
        """Whether any action projection uses a calibrated low precision format."""
        return any(layer.gate_up != "bf16" or layer.down != "bf16" for layer in self.action_layers)

    @property
    def capability(self) -> tuple[int, int] | None:
        """Return the selected profile's capability, or allow generic BF16 fusion."""
        return {"thor": (11, 0), "spark": (12, 1)}.get(self.hardware)

    @classmethod
    def from_json(cls, path: str | Path) -> Pi05OptimizationConfig:
        """Load a standalone recipe, rejecting ccinfer's different activation contract."""
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict) or values.get("activation") != "gelu_pytorch_tanh":
            raise ValueError("Recipe must identify the EmbodiInfer tanh GELU calibration contract")
        if "action_layers" in values:
            values["action_layers"] = tuple(ActionLayerPrecision(**row) for row in values["action_layers"])
        if "operators" in values:
            if not isinstance(values["operators"], dict):
                raise ValueError("operators must be a mapping of implementation names")
            values["operators"] = OperatorBackends(**values["operators"])
        return cls(**values)

    def to_json(self, path: str | Path) -> None:
        """Export deployment settings without benchmark or calibration dependencies."""
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")
