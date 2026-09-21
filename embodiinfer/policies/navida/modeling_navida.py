"""Compatibility facade for the self-contained NaViDA policy package."""

from .contract import (
    # Explicit re-exports: consumers read the checkpoint provenance from this module.
    NAVIDA_CHECKPOINT_REVISION as NAVIDA_CHECKPOINT_REVISION,
)
from .contract import (
    NAVIDA_SOURCE_REVISION as NAVIDA_SOURCE_REVISION,
)
from .contract import (
    NaViDABatch,
    NaViDAMemory,
    NaViDAPrefix,
    parse_navida_actions,
    select_navida_history,
)
from .cuda_graph import CapturedNaViDADecodeGraph, NaViDAGraphRuntime
from .generation import navida_generation_kwargs
from .policy import NaViDADecoder, NaViDAPolicy, build_navida
from .processing import navida_pil_image
from .runner import NaViDARunner

_pil = navida_pil_image
_parse_navida_actions = parse_navida_actions
_select_navida_history = select_navida_history
_navida_generation_kwargs = navida_generation_kwargs
_CapturedNaViDADecodeGraph = CapturedNaViDADecodeGraph
_NaViDADecoder = NaViDADecoder

__all__ = [
    "NaViDAMemory",
    "NaViDABatch",
    "NaViDAPrefix",
    "NaViDARunner",
    "NaViDAPolicy",
    "build_navida",
    "parse_navida_actions",
    "select_navida_history",
    "navida_generation_kwargs",
    "CapturedNaViDADecodeGraph",
    "NaViDAGraphRuntime",
    "NaViDADecoder",
]
