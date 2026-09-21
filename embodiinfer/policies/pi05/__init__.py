"""pi0.5 adapter (openpi / LeRobot PI05) -> embodiinfer ``VLAPolicy``."""

from .modeling_pi05 import Pi05Policy

PI05_ACTION_SPACE = "pi05.action_chunk.v1"
try:
    from .serving import Pi05ServingAdapter, Pi05ServingConfig
except ImportError:
    Pi05ServingAdapter = None
    Pi05ServingConfig = None

# Preserve the original public spelling while exposing the actual class names.
PI05ServingAdapter = Pi05ServingAdapter
PI05ServingConfig = Pi05ServingConfig

__all__ = [
    "PI05_ACTION_SPACE",
    "Pi05Policy",
    "Pi05ServingAdapter",
    "Pi05ServingConfig",
    "PI05ServingAdapter",
    "PI05ServingConfig",
]
