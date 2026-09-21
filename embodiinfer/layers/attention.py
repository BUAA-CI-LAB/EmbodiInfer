"""Attention contract and backend routing.

A single small contract, :class:`AttentionBackend`, lets a model or policy select
an eager, SDPA, FlexAttention, or Triton implementation without importing the
concrete backend. Implementations live under :mod:`embodiinfer.backend`; this module
contains only the protocol, registry, capability checks, and resolver.

Contract (``attend``):

    q         : ``[B, num_heads, seq_q, head_dim]``
    k, v      : ``[B, num_kv_heads, seq_k, head_dim]`` — ``num_kv_heads`` may be
                smaller than ``num_heads`` (grouped-query / multi-query); the
                backend handles the repeat, so callers pass K/V at their native
                head count and need not materialize the expansion themselves.
    attn_mask : additive float mask broadcastable to ``[B, 1, seq_q, seq_k]``, or
                ``None``
    scaling   : softmax scale; ``None`` -> ``head_dim ** -0.5`` (SDPA default)
    returns   : ``[B, num_heads, seq_q, head_dim]`` (the caller transposes /
                reshapes as it sees fit)

"""

from __future__ import annotations

import difflib
import importlib
from typing import Protocol, runtime_checkable

import torch


class AttentionBackend(Protocol):
    """Swappable scaled-dot-product attention kernel."""

    name: str

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Return the attention output ``[B, num_heads, seq_q, head_dim]``."""
        ...


@runtime_checkable
class SplitKVAttentionBackend(AttentionBackend, Protocol):
    """Attention over separate cached/current K/V with one current key per query."""

    def attend_split(
        self,
        q: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Return BHSD output without concatenation; zero in the shared mask keeps a key.

        The floating padding mask is [B, 1, 1, cached_length + current_length].
        It is shared across queries and does not represent arbitrary additive biases.
        Each batch row must contain at least one valid key.
        """
        ...


_REGISTRY: dict[str, type[AttentionBackend]] = {}
_LAZY_REGISTRY: dict[str, tuple[str, str]] = {}


def register_attention(name: str, cls: type[AttentionBackend]) -> None:
    """Register an attention backend class under ``name``."""
    if name in _LAZY_REGISTRY:
        raise ValueError(f"attention backend {name!r} is already lazily registered")
    _REGISTRY[name] = cls


def register_lazy_attention(name: str, module: str, class_name: str) -> None:
    """Register an optional backend without importing its dependencies."""
    if name in _REGISTRY or name in _LAZY_REGISTRY:
        raise ValueError(f"attention backend {name!r} is already registered")
    _LAZY_REGISTRY[name] = (module, class_name)


def _resolve_attention_backend(name: str) -> type[AttentionBackend]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    module_name, class_name = _LAZY_REGISTRY[name]
    cls = getattr(importlib.import_module(module_name), class_name)
    _REGISTRY[name] = cls
    return cls


def attention_backend_capability(name: str) -> tuple[bool, str | None]:
    """Return whether ``name`` can run in this process and why not."""
    if name == "auto":
        reasons: list[str] = []
        for candidate in ("triton_segmented", "sdpa"):
            available, reason = attention_backend_capability(candidate)
            if available:
                return True, None
            reasons.append(f"{candidate}: {reason}")
        return False, "; ".join(reasons)
    if name not in _REGISTRY and name not in _LAZY_REGISTRY:
        return False, f"unknown attention backend {name!r}"
    try:
        cls = _resolve_attention_backend(name)
    except Exception as exc:
        return False, f"failed to import backend {name!r}: {exc}"
    probe = getattr(cls, "capability", None)
    if probe is None:
        return True, None
    try:
        available, reason = probe()
    except Exception as exc:
        return False, f"capability probe failed: {exc}"
    return bool(available), reason


def get_attention_backend(name: str) -> AttentionBackend:
    """Instantiate a registered attention backend by name."""
    if name == "auto":
        for candidate in ("triton_segmented", "sdpa"):
            compatible, _ = attention_backend_capability(candidate)
            if compatible:
                name = candidate
                break
    if name not in _REGISTRY and name not in _LAZY_REGISTRY:
        registered = available_attention_backends()
        suggestion = difflib.get_close_matches(name, registered, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise KeyError(f"unknown attention backend '{name}'.{hint} Registered: {registered}")
    cls = _resolve_attention_backend(name)
    compatible, reason = attention_backend_capability(name)
    if not compatible:
        raise RuntimeError(f"attention backend {name!r} is unavailable: {reason}")
    return cls()


def get_split_kv_attention_backend(name: str) -> SplitKVAttentionBackend:
    """Resolve a registered backend that supports separate cached/current K/V."""
    backend = get_attention_backend(name)
    if not isinstance(backend, SplitKVAttentionBackend):
        raise TypeError(f"attention backend {name!r} does not support split K/V")
    return backend


def available_attention_backends() -> list[str]:
    """Return the sorted list of registered attention backend names."""
    return sorted(set(_REGISTRY) | set(_LAZY_REGISTRY))


register_lazy_attention("eager", "embodiinfer.backend.torch.attention", "EagerAttention")
register_lazy_attention(
    "eager_bc",
    "embodiinfer.backend.torch.attention",
    "EagerBroadcastAttention",
)
register_lazy_attention("sdpa", "embodiinfer.backend.torch.attention", "SDPAAttention")
register_lazy_attention("flex", "embodiinfer.backend.torch.attention", "FlexAttention")
register_lazy_attention(
    "triton_segmented",
    "embodiinfer.backend.triton.segmented_attention",
    "SegmentedTritonAttention",
)
register_lazy_attention(
    "triton_split_kv",
    "embodiinfer.backend.triton.split_kv_attention",
    "TritonSplitKVAttention",
)
