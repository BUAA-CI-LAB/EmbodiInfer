"""Tensor operator registry and kernel-profile selection.

An operator is named by its mathematics (``gated_residual``, ``ada_rms_norm``),
never by a model. Each operator has a ``torch`` reference implementation and may
have further implementations such as ``triton``. Callers ask for an operator by
name with their preferred implementation instead of importing a backend, so a
verified tuned kernel can replace it without changing model code.

``kernel_profile`` selects tuned kernels (see :mod:`embodiinfer.backend.triton.tuned`):

* ``none`` (default): the caller's preferred implementation, exactly as before.
* ``torch``: every registered operator uses its Torch reference.
* ``auto``: the profile matching this device and Torch/Triton versions, else ``none``.
* a profile name: that profile; a mismatched environment raises.

A tuned kernel replaces only the operators its profile contains, and only for
calls its ``applies`` predicate accepts; other calls use the preferred
implementation.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

OpFunction = Callable[..., Any]
TorchWrapper = Callable[[OpFunction], OpFunction]

_REGISTRY: dict[str, dict[str, OpFunction]] = {}
_LAZY_REGISTRY: dict[str, dict[str, tuple[str, str]]] = {}
_PROFILE_MODES = ("none", "torch", "auto")


def register_op(op: str, impl: str, fn: OpFunction) -> None:
    """Register ``fn`` as implementation ``impl`` of operator ``op``."""
    if impl in _REGISTRY.get(op, {}) or impl in _LAZY_REGISTRY.get(op, {}):
        raise ValueError(f"operator {op!r} implementation {impl!r} is already registered")
    _REGISTRY.setdefault(op, {})[impl] = fn


def register_lazy_op(op: str, impl: str, module: str, function: str) -> None:
    """Register an implementation without importing its backend until first use."""
    if impl in _REGISTRY.get(op, {}) or impl in _LAZY_REGISTRY.get(op, {}):
        raise ValueError(f"operator {op!r} implementation {impl!r} is already registered")
    _LAZY_REGISTRY.setdefault(op, {})[impl] = (module, function)


def available_ops() -> dict[str, list[str]]:
    """Return every registered operator with its implementation names."""
    names = set(_REGISTRY) | set(_LAZY_REGISTRY)
    return {op: sorted(set(_REGISTRY.get(op, {})) | set(_LAZY_REGISTRY.get(op, {}))) for op in sorted(names)}


def get_op(op: str, impl: str = "torch") -> OpFunction:
    """Return implementation ``impl`` of ``op`` without any profile override."""
    if impl in _REGISTRY.get(op, {}):
        return _REGISTRY[op][impl]
    if impl not in _LAZY_REGISTRY.get(op, {}):
        known = available_ops()
        if op not in known:
            raise KeyError(f"unknown operator {op!r}; registered: {sorted(known)}")
        raise KeyError(f"operator {op!r} has no {impl!r} implementation; registered: {known[op]}")
    module, function = _LAZY_REGISTRY[op].pop(impl)
    fn = getattr(importlib.import_module(module), function)
    _REGISTRY.setdefault(op, {})[impl] = fn
    return fn


class KernelSelection:
    """Resolve operators for one policy under one ``kernel_profile`` setting.

    Resolution is cached per ``(op, prefer, torch_wrapper)``, so call sites may
    resolve on every call. The profile is chosen at construction from the current
    CUDA device; rebuild the selection after moving the policy to another device.
    """

    def __init__(self, kernel_profile: Any = "none") -> None:
        from ..backend.triton.tuned import KernelProfile

        self.mode = kernel_profile.name if isinstance(kernel_profile, KernelProfile) else kernel_profile
        self.profile: KernelProfile | None = (
            kernel_profile if isinstance(kernel_profile, KernelProfile) else _select_profile(kernel_profile)
        )
        self._cache: dict[tuple[str, str, TorchWrapper | None], OpFunction] = {}

    def prepare(self, torch_wrappers: tuple[TorchWrapper | None, ...] = (None,)) -> None:
        """Resolve every registered operator and implementation ahead of time.

        Call this outside ``torch.compile`` regions: afterwards ``resolve`` only
        reads its cache, which graph tracing can follow without importing modules.
        """
        for op, impls in available_ops().items():
            for impl in impls:
                for wrapper in torch_wrappers:
                    self.resolve(op, impl, wrapper)

    def resolve(
        self, op: str, prefer: str = "torch", torch_wrapper: TorchWrapper | None = None
    ) -> OpFunction:
        """Return the callable for ``op``.

        ``prefer`` is the implementation the caller would use without a profile;
        ``torch_wrapper`` (for example an Inductor compile helper) is applied to
        the Torch reference only.
        """
        key = (op, prefer, torch_wrapper)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        impl = "torch" if self.mode == "torch" else prefer
        base = get_op(op, impl)
        if impl == "torch" and torch_wrapper is not None:
            base = torch_wrapper(base)
        resolved = base
        if self.profile is not None and op in self.profile.ops:
            tuned = self.profile.kernel(op)
            resolved = tuned.fn if tuned.applies is None else _dispatch(tuned.fn, tuned.applies, base)
        self._cache[key] = resolved
        return resolved


def _dispatch(tuned: OpFunction, applies: Callable[..., bool], fallback: OpFunction) -> OpFunction:
    def call(*args: Any, **kwargs: Any) -> Any:
        return tuned(*args, **kwargs) if applies(*args, **kwargs) else fallback(*args, **kwargs)

    return call


def _select_profile(name: str) -> Any:
    from ..backend.triton.tuned import available_profiles, current_environment

    if not isinstance(name, str):
        raise TypeError(f"kernel_profile must be a string; got {type(name).__name__}")
    if name in ("none", "torch"):
        return None
    profiles = available_profiles()
    environment = current_environment()
    if name == "auto":
        for profile in profiles.values():
            if profile.matches(environment):
                logger.info("kernel_profile=auto selected %r", profile.name)
                return profile
        logger.info("kernel_profile=auto found no profile for %s", environment)
        return None
    if name not in profiles:
        choices = sorted(_PROFILE_MODES) + sorted(profiles)
        raise ValueError(f"unknown kernel_profile {name!r}; expected one of {choices}")
    profile = profiles[name]
    if not profile.matches(environment):
        raise RuntimeError(
            f"kernel_profile {name!r} was verified for {dict(profile.match)}, not this environment {environment}"
        )
    return profile


for _op, _module in (
    ("gated_residual", "norm"),
    ("ada_rms_norm", "norm"),
    ("gated_gelu", "activation"),
    ("rotate_qk", "rotary"),
):
    register_lazy_op(_op, "torch", "embodiinfer.backend.torch.ops", _op)
    register_lazy_op(_op, "triton", f"embodiinfer.backend.triton.{_module}", _op)

__all__ = [
    "KernelSelection",
    "available_ops",
    "get_op",
    "register_lazy_op",
    "register_op",
]
