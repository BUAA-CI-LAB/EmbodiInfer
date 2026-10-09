"""Model-independent, serializable choices of operator implementation names."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class OperatorBackends:
    """Explicit implementations; custom registrations can use their own names.

    None norm_quant selects separate normalization and encoding. Composite
    implementations validate their dependency backends before allocating plans.
    Model-level precision, calibration and fusion placement remain policy-owned.
    """

    normalization: str = "cuda_strict"
    quantization: str = "cuda"
    gelu_mul: str = "cuda"
    norm_quant: str | None = "cuda_strict"
    paired_gelu: str = "triton_lookup"
    projection: str = "torch"

    def __post_init__(self) -> None:
        """Validate names without importing optional backends or resolving devices."""
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "norm_quant" and value is None:
                continue
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field.name} backend must be a nonempty implementation name")
