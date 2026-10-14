"""Serializable launch settings chosen outside inference and graph capture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmTile:
    """Row, column and reduction tile with a fixed warp and pipeline count."""

    m: int
    n: int
    k: int
    warps: int
    stages: int = 3

    def __post_init__(self) -> None:
        """Reject invalid launch dimensions before compiling kernels."""
        for value in (self.m, self.n, self.k):
            if not isinstance(value, int) or value < 16 or value & (value - 1):
                raise ValueError("GEMM tile dimensions must be powers of two, at least 16")
        if self.warps not in (4, 8) or self.stages not in (2, 3, 4, 5):
            raise ValueError("Unsupported GEMM warp or pipeline count")


@dataclass(frozen=True)
class SoftmaxTile:
    """Rows per block, warp count and exponential backend."""

    rows: int = 4
    warps: int = 4
    accurate_exp: bool = False
    backend: str = "triton"

    def __post_init__(self) -> None:
        """Reject unsupported probability kernel settings."""
        if self.rows not in (1, 2, 4, 8, 16) or self.warps not in (1, 2, 4, 8):
            raise ValueError("Unsupported softmax row or warp count")
        if self.backend not in ("triton", "cuda"):
            raise ValueError("Softmax backend must be triton or cuda")
        if self.backend == "cuda" and (self.rows != self.warps or not self.accurate_exp):
            raise ValueError("CUDA softmax uses one row per warp and the accurate exponential")
