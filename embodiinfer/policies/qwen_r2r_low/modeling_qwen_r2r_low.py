"""Compatibility facade for the independent Qwen R2R Low policy."""

from .contract import (
    LOW_LEVEL_CHECKPOINT,
    LOW_LEVEL_IMAGE_SIZE,
    LOW_LEVEL_REVISION,
    LOW_LEVEL_SYSTEM_PROMPT_SHA256,
    R2R_PREPROCESSOR_SHA256,
    NavigationOutputError,
    QwenR2RLowBatch,
    QwenR2RLowMemory,
    QwenR2RLowPrefix,
    parse_low_action,
)
from .cuda_graph import CapturedNextTokenGraph, QwenR2RLowGraphRuntime
from .forward import QwenR2RLowNextTokenForward
from .policy import (
    QwenR2RLowDecoder,
    QwenR2RLowPolicy,
    build_qwen_r2r_low,
    create_qwen_r2r_low_policy,
)
from .processing import tensor_to_pil
from .runner import QwenR2RLowRunner

QwenVLNMemory = QwenR2RLowMemory
QwenVLNBatch = QwenR2RLowBatch
QwenVLNPrefix = QwenR2RLowPrefix
_parse_low_level_action = parse_low_action
_pil = tensor_to_pil
_FullNextToken = QwenR2RLowNextTokenForward
_CapturedNextTokenGraph = CapturedNextTokenGraph
_Qwen25VLRunner = QwenR2RLowRunner
_NavigationDecoder = QwenR2RLowDecoder
_build = create_qwen_r2r_low_policy
build_low_level = build_qwen_r2r_low

__all__ = [
    "LOW_LEVEL_CHECKPOINT",
    "LOW_LEVEL_IMAGE_SIZE",
    "LOW_LEVEL_REVISION",
    "LOW_LEVEL_SYSTEM_PROMPT_SHA256",
    "R2R_PREPROCESSOR_SHA256",
    "NavigationOutputError",
    "QwenR2RLowMemory",
    "QwenR2RLowBatch",
    "QwenR2RLowPrefix",
    "parse_low_action",
    "tensor_to_pil",
    "QwenR2RLowNextTokenForward",
    "CapturedNextTokenGraph",
    "QwenR2RLowGraphRuntime",
    "QwenR2RLowRunner",
    "QwenR2RLowDecoder",
    "QwenR2RLowPolicy",
    "create_qwen_r2r_low_policy",
    "build_qwen_r2r_low",
    "QwenVLNMemory",
    "QwenVLNBatch",
    "QwenVLNPrefix",
    "_parse_low_level_action",
    "_pil",
    "_FullNextToken",
    "_CapturedNextTokenGraph",
    "_Qwen25VLRunner",
    "_NavigationDecoder",
    "_build",
    "build_low_level",
]
