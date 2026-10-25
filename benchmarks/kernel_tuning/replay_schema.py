"""Opt-in replay definitions retaining real float64 tensors with FlashInfer builders."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

from flashinfer_bench.data.definition import Definition, DType, TensorSpec
from flashinfer_bench.data.utils import NonEmptyString
from flashinfer_bench.utils import dtype_str_to_torch_dtype

if TYPE_CHECKING:
    import torch


class ReplayTensorSpec(TensorSpec):
    """Keep upstream tensor validation while admitting native double precision."""

    dtype: DType | Literal["float64"]


def _dtype(value: DType | Literal["float64"]) -> torch.dtype:
    import torch

    return torch.float64 if value == "float64" else dtype_str_to_torch_dtype(value)


class ReplayDefinition(Definition):
    """A validated Definition subclass for the local replay adapter, without global patches."""

    inputs: dict[NonEmptyString, ReplayTensorSpec]
    outputs: dict[NonEmptyString, ReplayTensorSpec]

    @cached_property
    def torch_input_dtypes(self) -> list[torch.dtype]:
        """Resolve input dtypes without narrowing doubles to an upstream enum member."""
        return [_dtype(spec.dtype) for spec in self.inputs.values()]

    @cached_property
    def torch_output_dtypes(self) -> list[torch.dtype]:
        """Resolve output dtypes using the same declared precision as replay metadata."""
        return [_dtype(spec.dtype) for spec in self.outputs.values()]
