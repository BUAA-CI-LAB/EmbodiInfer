"""Weight-bearing quantized linear modules shared by model policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch import nn

from embodiinfer.backend.torch.fp8 import dequantize_weight as dequantize_fp8_weight
from embodiinfer.backend.torch.fp8 import quantize_weight as quantize_fp8_weight
from embodiinfer.backend.torch.int8 import quantize_weight as quantize_int8_weight
from embodiinfer.backend.torch.nvfp4 import blocked_scales, quantization_thresholds
from embodiinfer.backend.torch.nvfp4 import dequantize_weight as dequantize_nvfp4_weight
from embodiinfer.backend.torch.nvfp4 import quantize_weight as quantize_nvfp4_weight
from embodiinfer.layers.linear import (
    FP8Config,
    INT8Config,
    NVFP4Config,
    QuantizationConfig,
    get_linear_backend,
    resolve_linear_backend,
)


class QuantizedLinear(nn.Module, ABC):
    """Common interface used by fused VLA projection paths."""

    quantization_method: ClassVar[str]
    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None
    backend: str

    in_features: int
    out_features: int
    compute_dtype: torch.dtype

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> QuantizedLinear:
        """Move quantized modules while retaining the precision of FP32 metadata.

        Engine initialization casts the whole policy to its execution dtype.
        Scales must remain FP32 for native low-precision GEMM and must not be
        rounded before being restored to FP32.
        """
        metadata = {
            name: value
            for name, value in self._buffers.items()
            if value is not None and value.dtype == torch.float32
        }
        compute = torch.empty(0, device=self.weight.device, dtype=self.compute_dtype)
        result = super()._apply(fn, recurse=recurse)
        for name, value in metadata.items():
            converted = self._buffers[name]
            if converted.dtype != torch.float32:
                self._buffers[name] = value.to(device=converted.device)
        self.compute_dtype = fn(compute).dtype
        return result

    @classmethod
    @abstractmethod
    def from_linear(cls, linear: nn.Module, config: Any) -> QuantizedLinear:
        """Create an inference-only quantized projection."""

    @classmethod
    @abstractmethod
    def pack(cls, linears: tuple[QuantizedLinear, ...]) -> QuantizedLinear:
        """Pack projections that share the same input into one projection."""

    @property
    def quantized_weight(self) -> torch.Tensor:
        """Expose the stored payload in the format expected by the selected kernel."""
        return self.weight

    def _resolved_backend(self, inputs: torch.Tensor) -> str:
        return resolve_linear_backend(
            self.quantization_method, self.backend, inputs, self.quantized_weight, self.weight_scale
        )

    def _linear_metadata(self) -> dict[str, torch.Tensor]:
        return {}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the registered linear backend using this module's weight buffers."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"{type(self).__name__} expected K={self.in_features}, got {inputs.shape[-1]}")
        backend = self._resolved_backend(inputs)
        kernel = get_linear_backend(f"{self.quantization_method}_{backend}")
        return kernel(inputs, self.quantized_weight, self.weight_scale, self.bias, **self._linear_metadata())


class FP8Linear(QuantizedLinear):
    """Per-output-channel FP8 weights with capability-aware dispatch."""

    quantization_method = "fp8"

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: nn.Parameter | None,
        *,
        compute_dtype: torch.dtype,
        backend: str = "auto",
        split_sizes: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if weight.ndim != 2 or weight_scale.shape not in ((), (weight.shape[0],)):
            raise ValueError("FP8Linear expects weight [N, K] and scalar or per-channel scale")
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if weight.dtype == fp8_dtype:
            weight = weight.view(torch.uint8)
        elif weight.dtype != torch.uint8:
            raise ValueError("FP8Linear weight must contain E4M3 data")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", weight_scale)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = bias
        if split_sizes is not None:
            self.split_sizes = split_sizes

    @property
    def quantized_weight(self) -> torch.Tensor:
        """Return the uint8-stored payload as E4M3 without a copy."""

        return self.weight.view(torch.float8_e4m3fn)

    @classmethod
    def from_linear(cls, linear: nn.Module, config: FP8Config) -> FP8Linear:
        """Convert one projection while preserving its bias, dtype, and training state."""
        weight = getattr(linear, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError(f"cannot FP8-quantize {type(linear).__name__} without a matrix weight")
        quantized, scale = quantize_fp8_weight(
            weight.detach(), backend=config.backend, scaling_scheme=config.scaling_scheme
        )
        split_sizes = getattr(linear, "split_sizes", None)
        module = cls(
            quantized,
            scale,
            getattr(linear, "bias", None),
            compute_dtype=weight.dtype,
            backend=config.backend,
            split_sizes=None if split_sizes is None else tuple(split_sizes),
        )
        return module.train(linear.training)

    @classmethod
    def pack(cls, linears: tuple[FP8Linear, ...]) -> FP8Linear:
        """Pack compatible projections and make their weights share the packed storage."""
        if not linears:
            raise ValueError("cannot pack an empty FP8 linear sequence")
        first = linears[0]
        if any(
            linear.in_features != first.in_features
            or linear.compute_dtype != first.compute_dtype
            or linear.backend != first.backend
            or linear.weight.dtype != first.weight.dtype
            or linear.bias is not None
            for linear in linears
        ):
            raise ValueError("FP8 fused projections must have matching inputs, dtype, backend, and no bias")
        if first.weight_scale.ndim == 0:
            weight = torch.cat(
                [
                    dequantize_fp8_weight(
                        linear.quantized_weight.detach(), linear.weight_scale, torch.float32
                    )
                    for linear in linears
                ],
                dim=0,
            )
            packed_weight, packed_scale = quantize_fp8_weight(weight, scaling_scheme="tensorwise")
        else:
            packed_weight = torch.cat([linear.weight.detach() for linear in linears], dim=0).contiguous()
            packed_scale = torch.cat([linear.weight_scale.detach() for linear in linears], dim=0).contiguous()
        packed = cls(
            packed_weight,
            packed_scale,
            None,
            compute_dtype=first.compute_dtype,
            backend=first.backend,
            split_sizes=tuple(linear.out_features for linear in linears),
        )
        packed.train(first.training)
        offset = 0
        for linear in linears:
            end = offset + linear.out_features
            linear.weight = nn.Parameter(packed.weight[offset:end], requires_grad=False)
            linear.weight_scale = (
                packed.weight_scale if packed.weight_scale.ndim == 0 else packed.weight_scale[offset:end]
            )
            offset = end
        return packed

    def extra_repr(self) -> str:
        """Describe projection dimensions and the requested backend."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend!r}"
        )


class INT8Linear(QuantizedLinear):
    """Per-channel weights and dynamic per-row activations using CUDA INT8 GEMM."""

    quantization_method = "int8"

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: nn.Parameter | None,
        *,
        compute_dtype: torch.dtype,
        backend: str = "auto",
        split_sizes: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.int8:
            raise ValueError("INT8Linear expects transposed INT8 weight [K, N]")
        if weight_scale.shape != (weight.shape[1],):
            raise ValueError("INT8Linear expects one weight scale per output channel")
        self.in_features = int(weight.shape[0])
        self.out_features = int(weight.shape[1])
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", weight_scale)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = bias
        if split_sizes is not None:
            self.split_sizes = split_sizes

    @classmethod
    def from_linear(cls, linear: nn.Module, config: INT8Config) -> INT8Linear:
        """Convert one projection while preserving its bias, dtype, and training state."""
        weight = getattr(linear, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError(f"cannot INT8-quantize {type(linear).__name__} without a matrix weight")
        quantized, scale = quantize_int8_weight(weight.detach())
        split_sizes = getattr(linear, "split_sizes", None)
        module = cls(
            quantized,
            scale,
            getattr(linear, "bias", None),
            compute_dtype=weight.dtype,
            backend=config.backend,
            split_sizes=None if split_sizes is None else tuple(split_sizes),
        )
        return module.train(linear.training)

    @classmethod
    def pack(cls, linears: tuple[INT8Linear, ...]) -> INT8Linear:
        """Pack compatible projections and make their weights share the packed storage."""
        if not linears:
            raise ValueError("cannot pack an empty INT8 linear sequence")
        first = linears[0]
        if any(
            type(linear) is not cls
            or linear.in_features != first.in_features
            or linear.compute_dtype != first.compute_dtype
            or linear.backend != first.backend
            or linear.bias is not None
            for linear in linears
        ):
            raise ValueError("INT8 fused projections must match inputs, dtype, backend, and bias")
        packed = cls(
            torch.cat([linear.weight.detach() for linear in linears], dim=1).contiguous(),
            torch.cat([linear.weight_scale.detach() for linear in linears]).contiguous(),
            None,
            compute_dtype=first.compute_dtype,
            backend=first.backend,
            split_sizes=tuple(linear.out_features for linear in linears),
        )
        packed.train(first.training)
        offset = 0
        for linear in linears:
            end = offset + linear.out_features
            linear.weight = nn.Parameter(packed.weight[:, offset:end], requires_grad=False)
            linear.weight_scale = packed.weight_scale[offset:end]
            offset = end
        return packed

    def extra_repr(self) -> str:
        """Describe projection dimensions and the requested backend."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend!r}"
        )


class NVFP4Linear(QuantizedLinear):
    """Dynamic W4A4 NVFP4 linear backed by Blackwell cuBLASLt."""

    quantization_method = "nvfp4"

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
        bias: nn.Parameter | None,
        *,
        compute_dtype: torch.dtype,
        backend: str = "auto",
        split_sizes: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("NVFP4Linear expects packed weight [N, K/2]")
        if weight_scale.shape != (weight.shape[0], weight.shape[1] // 8):
            raise ValueError("NVFP4Linear expects one weight scale per 16 logical values")
        if weight_scale.dtype == torch.float8_e4m3fn:
            weight_scale = weight_scale.view(torch.uint8)
        elif weight_scale.dtype != torch.uint8:
            raise ValueError("NVFP4Linear weight_scale must contain E4M3 data")
        self.in_features = int(weight.shape[1] * 2)
        self.out_features = int(weight.shape[0])
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer(
            "weight_scale_blocked",
            blocked_scales(weight_scale.view(torch.float8_e4m3fn)).view(torch.uint8),
        )
        self.register_buffer("weight_global_scale", weight_global_scale)
        self.register_buffer(
            "quantization_thresholds",
            quantization_thresholds(weight),
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = bias
        if split_sizes is not None:
            self.split_sizes = split_sizes

    @classmethod
    def from_linear(cls, linear: nn.Module, config: NVFP4Config) -> NVFP4Linear:
        """Convert one projection while preserving its bias, dtype, and training state."""
        weight = getattr(linear, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError(f"cannot NVFP4-quantize {type(linear).__name__} without a matrix weight")
        thresholds = quantization_thresholds(weight)
        quantized, scale, global_scale = quantize_nvfp4_weight(weight.detach(), thresholds)
        split_sizes = getattr(linear, "split_sizes", None)
        module = cls(
            quantized,
            scale,
            global_scale,
            getattr(linear, "bias", None),
            compute_dtype=weight.dtype,
            backend=config.backend,
            split_sizes=None if split_sizes is None else tuple(split_sizes),
        )
        return module.train(linear.training)

    @classmethod
    def pack(cls, linears: tuple[NVFP4Linear, ...]) -> NVFP4Linear:
        """Pack compatible projections and make their weights share the packed storage."""
        if not linears:
            raise ValueError("cannot pack an empty NVFP4 linear sequence")
        first = linears[0]
        if any(
            type(linear) is not cls
            or linear.in_features != first.in_features
            or linear.compute_dtype != first.compute_dtype
            or linear.backend != first.backend
            or linear.bias is not None
            for linear in linears
        ):
            raise ValueError("NVFP4 fused projections must match inputs, dtype, backend, and bias")
        weight = torch.cat(
            [
                dequantize_nvfp4_weight(
                    linear.weight.detach(),
                    linear.weight_scale,
                    linear.weight_global_scale,
                    torch.float32,
                )
                for linear in linears
            ],
            dim=0,
        )
        quantized, scale, global_scale = quantize_nvfp4_weight(weight, first.quantization_thresholds)
        packed = cls(
            quantized,
            scale,
            global_scale,
            None,
            compute_dtype=first.compute_dtype,
            backend=first.backend,
            split_sizes=tuple(linear.out_features for linear in linears),
        )
        packed.train(first.training)
        offset = 0
        for linear in linears:
            end = offset + linear.out_features
            linear.weight = nn.Parameter(packed.weight[offset:end], requires_grad=False)
            linear.weight_scale = packed.weight_scale[offset:end]
            linear.weight_scale_blocked = blocked_scales(linear.weight_scale.view(torch.float8_e4m3fn)).view(
                torch.uint8
            )
            linear.weight_global_scale = packed.weight_global_scale
            offset = end
        return packed

    def extra_repr(self) -> str:
        """Describe projection dimensions and the requested backend."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend!r}"
        )

    def _linear_metadata(self) -> dict[str, torch.Tensor]:
        return {
            "weight_global_scale": self.weight_global_scale,
            "weight_scale_blocked": self.weight_scale_blocked,
            "quantization_thresholds": self.quantization_thresholds,
        }


def quantize_linear(linear: nn.Module, config: QuantizationConfig) -> QuantizedLinear:
    """Convert a projection using the selected quantization method."""
    if isinstance(config, FP8Config):
        return FP8Linear.from_linear(linear, config)
    if isinstance(config, INT8Config):
        return INT8Linear.from_linear(linear, config)
    if isinstance(config, NVFP4Config):
        return NVFP4Linear.from_linear(linear, config)
    raise TypeError(f"unsupported quantization config {type(config).__name__}")


def pack_quantized_linears(
    linears: tuple[QuantizedLinear, ...],
) -> QuantizedLinear:
    """Fuse projections of one quantization method while preserving shared storage."""
    if not linears:
        raise ValueError("cannot pack an empty quantized linear sequence")
    linear_type = type(linears[0])
    if not all(type(linear) is linear_type for linear in linears):
        raise ValueError("all packed quantized projections must use the same quantization method")
    return linear_type.pack(linears)


__all__ = [
    "QuantizedLinear",
    "FP8Linear",
    "INT8Linear",
    "NVFP4Linear",
    "quantize_linear",
    "pack_quantized_linears",
]
