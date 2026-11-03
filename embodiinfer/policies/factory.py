"""Policy factory: a name -> lazily imported builder registry.

Adapters still register builders through :func:`register_policy`, but optional
model modules are imported only when their policy is constructed.  Keeping the
catalog here lets :func:`available_policies` remain complete without importing
incompatible model dependency stacks into every EmbodiInfer environment.
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Callable

from ..exceptions import PolicyNotFoundError
from .base import VLAPolicy

_REGISTRY: dict[str, Callable[..., VLAPolicy]] = {}
_LAZY_POLICY_MODULES = {
    "activevln": "embodiinfer.policies.activevln",
    "cosmos": "embodiinfer.policies.cosmos",
    "dm05": "embodiinfer.policies.dm05",
    "gr00t": "embodiinfer.policies.gr00t",
    "lingbot_vla": "embodiinfer.policies.lingbot_vla",
    "mock_flow_vla": "embodiinfer.policies.mock",
    "navida": "embodiinfer.policies.navida",
    "openvla_oft": "embodiinfer.policies.openvla_oft.modeling_openvla_oft",
    "pi05": "embodiinfer.policies.pi05",
    "qwen2.5-vl-3b-r2r-low-level": "embodiinfer.policies.qwen_r2r_low",
    "qwen2.5-vl-3b-r2r-panoramic": "embodiinfer.policies.qwen_r2r_panoramic",
    "streamvln": "embodiinfer.policies.streamvln",
}


def register_policy(name: str) -> Callable[[Callable[..., VLAPolicy]], Callable[..., VLAPolicy]]:
    """Decorator registering a builder ``fn(**kwargs) -> VLAPolicy`` under ``name``."""

    def deco(fn: Callable[..., VLAPolicy]) -> Callable[..., VLAPolicy]:
        _REGISTRY[name] = fn
        return fn

    return deco


def make_policy(name: str, **kwargs) -> VLAPolicy:
    """Construct a registered policy by name, forwarding ``**kwargs`` to its builder."""
    module = _LAZY_POLICY_MODULES.get(name)
    if name not in _REGISTRY and module is not None:
        importlib.import_module(module)
    if name not in _REGISTRY:
        known = available_policies()
        suggestion = difflib.get_close_matches(name, known, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise PolicyNotFoundError(f"unknown policy '{name}'.{hint} Registered: {known}")
    return _REGISTRY[name](**kwargs)


def available_policies() -> list[str]:
    """Return registered and lazily available policy names."""
    return sorted(_REGISTRY.keys() | _LAZY_POLICY_MODULES.keys())
