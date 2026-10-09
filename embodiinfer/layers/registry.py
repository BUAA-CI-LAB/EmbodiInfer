"""Lazy operator registration and explicit execution support checks."""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class OperatorRequest:
    """Execution requirements checked before constructing an operator backend.

    Shape and bits are optional during weight preparation. Individual plans still
    validate their complete tensor layouts, affine inputs and encoding formats.
    """

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...] | None = None
    bits: int | None = None
    cuda_graph: bool = False

    def __post_init__(self) -> None:
        """Normalize devices and reject ambiguous shape requirements."""
        object.__setattr__(self, "device", torch.device(self.device))
        if self.shape is not None and (
            not self.shape or any(type(value) is not int or value <= 0 for value in self.shape)
        ):
            raise ValueError("Operator shapes must contain positive integer dimensions")


@dataclass(frozen=True)
class OperatorCapabilities:
    """Declared support and arithmetic contract for a prepared operator.

    Graph support applies to warmed plans; backend construction is outside
    capture. The arithmetic name describes rounding/encoding rather than promising
    whole-model parity. Registries reject unsupported requests without fallback.
    """

    devices: tuple[str, ...]
    dtypes: tuple[torch.dtype, ...]
    arithmetic: str
    cuda_graph: bool = True
    rank: int | None = None
    width_multiple: int = 1
    bits: tuple[int, ...] = ()
    minimum_sm: tuple[int, int] | None = None
    minimum_sm_by_bits: tuple[tuple[int, tuple[int, int]], ...] = ()

    def check(self, request: OperatorRequest) -> None:
        """Raise before allocation/compilation when a request is unsupported."""
        if request.device.type not in self.devices or request.dtype not in self.dtypes:
            raise ValueError(f"Operator does not support {request.device.type}/{request.dtype}")
        if request.cuda_graph and not self.cuda_graph:
            raise ValueError("Operator does not support CUDA Graph replay")
        if request.bits is not None and request.bits not in self.bits:
            raise ValueError(f"Operator does not support {request.bits}-bit encoding")
        if request.shape is not None:
            if self.rank is not None and len(request.shape) != self.rank:
                raise ValueError(f"Operator requires rank {self.rank} inputs")
            if request.shape[-1] % self.width_multiple:
                raise ValueError(f"Operator width must be divisible by {self.width_multiple}")
        if request.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Operator requires an available CUDA device")
            if (
                self.minimum_sm is not None
                and torch.cuda.get_device_capability(request.device) < self.minimum_sm
            ):
                raise ValueError(f"Operator requires SM {self.minimum_sm} or newer")
            required_sm = dict(self.minimum_sm_by_bits).get(request.bits)
            if required_sm is not None and torch.cuda.get_device_capability(request.device) < required_sm:
                raise ValueError(f"{request.bits}-bit encoding requires SM {required_sm} or newer")


class BackendRegistry(Generic[T]):
    """Registry for one operator contract, with lazy optional implementations."""

    def __init__(self, family: str) -> None:
        """Create a model-independent registry for the named operator family."""
        self.family = family
        self._factories: dict[str, type[T]] = {}
        self._lazy: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    def register(self, name: str, implementation: type[T]) -> None:
        """Register an implementation class without instantiating it."""
        with self._lock:
            self._validate_name(name)
            self._factories[name] = implementation

    def register_lazy(self, name: str, module: str, class_name: str) -> None:
        """Record an import path without importing Torch extensions or Triton."""
        with self._lock:
            self._validate_name(name)
            self._lazy[name] = (module, class_name)

    def _validate_name(self, name: str) -> None:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("Backend names must be nonempty strings without surrounding whitespace")
        if name in self._factories or name in self._lazy:
            raise ValueError(f"{self.family} backend {name!r} is already registered")

    def available(self) -> tuple[str, ...]:
        """List names without resolving optional implementation imports."""
        with self._lock:
            return tuple(sorted(self._factories.keys() | self._lazy.keys()))

    def get(self, name: str, request: OperatorRequest, **options: Any) -> T:
        """Resolve support, then construct on the requested device before capture."""
        with self._lock:
            if name not in self._factories:
                if name not in self._lazy:
                    raise ValueError(f"Unknown {self.family} backend {name!r}; available: {self.available()}")
                module, class_name = self._lazy[name]
                implementation = getattr(importlib.import_module(module), class_name)
                self._factories[name] = implementation
            factory = self._factories[name]
        capabilities = getattr(factory, "capabilities", None)
        if not isinstance(capabilities, OperatorCapabilities):
            raise TypeError(f"{self.family} backend {name!r} must declare OperatorCapabilities")
        capabilities.check(request)
        if request.device.type == "cuda":
            with torch.cuda.device(request.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Prepare operator backends before CUDA Graph capture")
                return factory(**options)
        return factory(**options)
