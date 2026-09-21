"""Policy contracts, lazy factory, and compatibility exports.

Optional adapters are intentionally not imported while initializing this
package.  Their dependency profiles conflict, so the factory loads only the
requested adapter and module attributes are resolved lazily for callers that use
the historical ``from embodiinfer.policies import Pi05Policy`` form.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import PrefixState, VLAPolicy
from .config import VLAPolicyConfig
from .factory import available_policies, make_policy, register_policy
from .mock import MockFlowVLA

_LAZY_EXPORTS = {
    "ActiveVLNPolicy": ("embodiinfer.policies.activevln", "ActiveVLNPolicy"),
    "CosmosPolicy": ("embodiinfer.policies.cosmos", "CosmosPolicy"),
    "DM05Batch": ("embodiinfer.policies.dm05", "DM05Batch"),
    "DM05Policy": ("embodiinfer.policies.dm05", "DM05Policy"),
    "Gr00tPolicy": ("embodiinfer.policies.gr00t", "Gr00tPolicy"),
    "LingBotVLAPolicy": ("embodiinfer.policies.lingbot_vla", "LingBotVLAPolicy"),
    "NaViDAPolicy": ("embodiinfer.policies.navida", "NaViDAPolicy"),
    "OpenVLAOFTPolicy": (
        "embodiinfer.policies.openvla_oft.modeling_openvla_oft",
        "OpenVLAOFTPolicy",
    ),
    "Pi05Policy": ("embodiinfer.policies.pi05", "Pi05Policy"),
    "QwenR2RLowPolicy": ("embodiinfer.policies.qwen_r2r_low", "QwenR2RLowPolicy"),
    "QwenR2RPanoramicPolicy": (
        "embodiinfer.policies.qwen_r2r_panoramic",
        "QwenR2RPanoramicPolicy",
    ),
    "StreamVLNPolicy": ("embodiinfer.policies.streamvln", "StreamVLNPolicy"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(globals().keys() | _LAZY_EXPORTS.keys())


__all__ = [
    "VLAPolicy",
    "VLAPolicyConfig",
    "PrefixState",
    "MockFlowVLA",
    "Pi05Policy",
    "CosmosPolicy",
    "DM05Policy",
    "DM05Batch",
    "Gr00tPolicy",
    "OpenVLAOFTPolicy",
    "LingBotVLAPolicy",
    "ActiveVLNPolicy",
    "QwenR2RLowPolicy",
    "QwenR2RPanoramicPolicy",
    "NaViDAPolicy",
    "StreamVLNPolicy",
    "make_policy",
    "register_policy",
    "available_policies",
]
