"""Quantized-linear configuration contracts and lazy backend routing.

Configuration describes the quantization method and scaling policy. Tensor
conversion, capability probes, and GEMM implementations live in embodiinfer.backend.
"""

from __future__ import annotations

import fnmatch
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import torch

_FP8_BACKENDS = {"auto", "native", "triton", "torch"}


_SCALING_SCHEMES = {"auto", "channelwise", "tensorwise"}


@dataclass(frozen=True)
class FP8Config:
    """Runtime FP8 policy shared by VLA model adapters."""

    activation_scheme: str = "dynamic"
    ignored_layers: tuple[str, ...] = ()
    backend: str = "auto"
    scaling_scheme: str = "auto"

    def __post_init__(self) -> None:
        if self.activation_scheme != "dynamic":
            raise ValueError("VVLA online FP8 currently supports activation_scheme='dynamic' only")
        if self.backend not in _FP8_BACKENDS:
            raise ValueError(
                f"unsupported FP8 backend {self.backend!r}; expected one of {sorted(_FP8_BACKENDS)}"
            )
        if self.scaling_scheme not in _SCALING_SCHEMES:
            raise ValueError(
                f"unsupported FP8 scaling_scheme {self.scaling_scheme!r}; "
                f"expected one of {sorted(_SCALING_SCHEMES)}"
            )

    def is_ignored(self, name: str) -> bool:
        """Match a policy-provided projection name against the configured exclusions."""
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.ignored_layers)


def parse_fp8_config(value: str | Mapping[str, Any] | FP8Config | None) -> FP8Config | None:
    """Parse the public ``quantization`` builder option."""

    if value is None:
        return None
    if isinstance(value, FP8Config):
        return value
    if isinstance(value, str):
        if value.lower() != "fp8":
            raise ValueError("quantization must be None, 'fp8', or an FP8 config mapping")
        return FP8Config()
    if not isinstance(value, Mapping):
        raise TypeError("quantization must be None, 'fp8', an FP8Config, or a mapping")
    method = str(value.get("method", value.get("quant_method", "fp8"))).lower()
    if method != "fp8":
        raise ValueError(f"unsupported quantization method {method!r}")
    ignored = value.get("ignored_layers", ())
    if isinstance(ignored, str) or not isinstance(ignored, Sequence):
        raise TypeError("ignored_layers must be a sequence of glob patterns")
    unknown = set(value) - {
        "method",
        "quant_method",
        "activation_scheme",
        "ignored_layers",
        "backend",
        "scaling_scheme",
    }
    if unknown:
        raise TypeError(f"unknown FP8 config fields: {', '.join(sorted(unknown))}")
    return FP8Config(
        activation_scheme=str(value.get("activation_scheme", "dynamic")).lower(),
        ignored_layers=tuple(str(pattern) for pattern in ignored),
        backend=str(value.get("backend", "auto")).lower(),
        scaling_scheme=str(value.get("scaling_scheme", "auto")).lower(),
    )


_INT8_BACKENDS = {"auto", "native", "torch"}


@dataclass(frozen=True)
class INT8Config:
    """Runtime dynamic W8A8 policy for CUDA inference."""

    activation_scheme: str = "dynamic"
    ignored_layers: tuple[str, ...] = ()
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.activation_scheme != "dynamic":
            raise ValueError("VVLA INT8 currently supports activation_scheme='dynamic' only")
        if self.backend not in _INT8_BACKENDS:
            raise ValueError(
                f"unsupported INT8 backend {self.backend!r}; expected one of {sorted(_INT8_BACKENDS)}"
            )

    def is_ignored(self, name: str) -> bool:
        """Match a policy-provided projection name against the configured exclusions."""
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.ignored_layers)


def parse_int8_config(
    value: str | Mapping[str, Any] | INT8Config | None,
) -> INT8Config | None:
    """Parse a method name, mapping, or typed INT8 configuration."""
    if value is None:
        return None
    if isinstance(value, INT8Config):
        return value
    if isinstance(value, str):
        if value.lower() not in {"int8", "w8a8"}:
            raise ValueError("quantization must be None, 'int8', or an INT8 config mapping")
        return INT8Config()
    if not isinstance(value, Mapping):
        raise TypeError("quantization must be None, 'int8', an INT8Config, or a mapping")
    method = str(value.get("method", value.get("quant_method", "int8"))).lower()
    if method not in {"int8", "w8a8"}:
        raise ValueError(f"unsupported quantization method {method!r}")
    ignored = value.get("ignored_layers", ())
    if isinstance(ignored, str) or not isinstance(ignored, Sequence):
        raise TypeError("ignored_layers must be a sequence of glob patterns")
    unknown = set(value) - {
        "method",
        "quant_method",
        "activation_scheme",
        "ignored_layers",
        "backend",
    }
    if unknown:
        raise TypeError(f"unknown INT8 config fields: {', '.join(sorted(unknown))}")
    return INT8Config(
        activation_scheme=str(value.get("activation_scheme", "dynamic")).lower(),
        ignored_layers=tuple(str(pattern) for pattern in ignored),
        backend=str(value.get("backend", "auto")).lower(),
    )


_NVFP4_BACKENDS = {"auto", "native", "torch"}


@dataclass(frozen=True)
class NVFP4Config:
    """Runtime NVFP4 policy for SM100+ VLA inference."""

    activation_scheme: str = "dynamic"
    ignored_layers: tuple[str, ...] = ()
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.activation_scheme != "dynamic":
            raise ValueError("VVLA NVFP4 currently supports activation_scheme='dynamic' only")
        if self.backend not in _NVFP4_BACKENDS:
            raise ValueError(
                f"unsupported NVFP4 backend {self.backend!r}; expected one of {sorted(_NVFP4_BACKENDS)}"
            )

    def is_ignored(self, name: str) -> bool:
        """Match a policy-provided projection name against the configured exclusions."""
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.ignored_layers)


def parse_nvfp4_config(
    value: str | Mapping[str, Any] | NVFP4Config | None,
) -> NVFP4Config | None:
    """Parse a method name, mapping, or typed NVFP4 configuration."""
    if value is None:
        return None
    if isinstance(value, NVFP4Config):
        return value
    if isinstance(value, str):
        if value.lower() != "nvfp4":
            raise ValueError("quantization must be None, 'nvfp4', or an NVFP4 config mapping")
        return NVFP4Config()
    if not isinstance(value, Mapping):
        raise TypeError("quantization must be None, 'nvfp4', an NVFP4Config, or a mapping")
    method = str(value.get("method", value.get("quant_method", "nvfp4"))).lower()
    if method != "nvfp4":
        raise ValueError(f"unsupported quantization method {method!r}")
    ignored = value.get("ignored_layers", ())
    if isinstance(ignored, str) or not isinstance(ignored, Sequence):
        raise TypeError("ignored_layers must be a sequence of glob patterns")
    unknown = set(value) - {
        "method",
        "quant_method",
        "activation_scheme",
        "ignored_layers",
        "backend",
    }
    if unknown:
        raise TypeError(f"unknown NVFP4 config fields: {', '.join(sorted(unknown))}")
    return NVFP4Config(
        activation_scheme=str(value.get("activation_scheme", "dynamic")).lower(),
        ignored_layers=tuple(str(pattern) for pattern in ignored),
        backend=str(value.get("backend", "auto")).lower(),
    )


QuantizationConfig: TypeAlias = FP8Config | INT8Config | NVFP4Config


def parse_quantization_config(
    value: str | Mapping[str, Any] | QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Parse a public model-builder quantization option."""

    if value is None or isinstance(value, (FP8Config, INT8Config, NVFP4Config)):
        return value
    method = value if isinstance(value, str) else value.get("method", value.get("quant_method", "fp8"))
    method = str(method).lower()
    if method == "fp8":
        return parse_fp8_config(value)
    if method in {"int8", "w8a8"}:
        return parse_int8_config(value)
    if method == "nvfp4":
        return parse_nvfp4_config(value)
    raise ValueError(f"unsupported quantization method {method!r}")


class LinearBackend(Protocol):
    """Apply a quantized projection; keyword metadata supplies method-specific scales."""

    def __call__(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        **metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Return [..., N] from [..., K] using the method's stored weight layout."""
        ...


class LinearCapability(Protocol):
    """Check hardware and tensor compatibility without launching a GEMM."""

    def __call__(
        self, inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> tuple[bool, str | None]:
        """Return availability and an actionable reason when unavailable."""
        ...


_REGISTRY: dict[str, tuple[LinearBackend, LinearCapability | None]] = {}
_LAZY_REGISTRY: dict[str, tuple[str, str, str | None]] = {}


def register_linear(name: str, kernel: LinearBackend, capability: LinearCapability | None = None) -> None:
    """Register a method/backend pair such as fp8_native, without model dependencies."""
    if name in _REGISTRY or name in _LAZY_REGISTRY:
        raise ValueError(f"linear backend {name!r} is already registered")
    _REGISTRY[name] = kernel, capability


def register_lazy_linear(name: str, module: str, kernel: str, capability: str | None = None) -> None:
    """Register an optional implementation without importing its dependencies."""
    if name in _REGISTRY or name in _LAZY_REGISTRY:
        raise ValueError(f"linear backend {name!r} is already registered")
    _LAZY_REGISTRY[name] = module, kernel, capability


def _resolve(name: str) -> tuple[LinearBackend, LinearCapability | None]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY_REGISTRY:
        raise ValueError(f"unknown linear backend {name!r}")
    module_name, kernel, capability = _LAZY_REGISTRY[name]
    module = importlib.import_module(module_name)
    entry = getattr(module, kernel), getattr(module, capability) if capability else None
    _REGISTRY[name] = entry
    return entry


def get_linear_backend(name: str) -> LinearBackend:
    """Resolve a registered implementation after selecting a compatible backend."""
    return _resolve(name)[0]


def resolve_linear_backend(
    method: str,
    backend: str,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> str:
    """Select native/Triton/Torch execution with the existing CPU reference fallback.

    Explicit CUDA backends fail when unavailable. Auto preserves the existing
    weight-only fallback when native activation-quantized execution is unavailable.
    """
    if method not in {"fp8", "int8", "nvfp4"}:
        raise ValueError(f"unsupported quantization method {method!r}")
    if backend == "torch" or not inputs.is_cuda:
        return "torch"
    if backend == "auto":
        candidates = ("native", "triton", "torch") if method == "fp8" else ("native", "torch")
    else:
        candidates = (backend,)
    reason = None
    for candidate in candidates:
        _, capability = _resolve(f"{method}_{candidate}")
        available, reason = capability(inputs, weight, weight_scale) if capability else (True, None)
        if available:
            return candidate
    raise RuntimeError(reason or f"the requested {method} {backend} backend is unavailable")


for _method in ("fp8", "int8", "nvfp4"):
    register_lazy_linear(f"{_method}_torch", f"embodiinfer.backend.torch.{_method}", "linear")
    register_lazy_linear(
        f"{_method}_native", f"embodiinfer.backend.torch.{_method}", "native_linear", "native_capability"
    )
register_lazy_linear(
    "fp8_triton", "embodiinfer.backend.triton.fp8", "fp8_weight_only_linear", "fp8_weight_only_capability"
)

__all__ = [
    "FP8Config",
    "INT8Config",
    "NVFP4Config",
    "QuantizationConfig",
    "parse_fp8_config",
    "parse_int8_config",
    "parse_nvfp4_config",
    "parse_quantization_config",
    "LinearBackend",
    "LinearCapability",
    "register_linear",
    "register_lazy_linear",
    "get_linear_backend",
    "resolve_linear_backend",
]
