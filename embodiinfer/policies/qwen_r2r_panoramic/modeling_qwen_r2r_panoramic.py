"""Public facade for the Qwen R2R panoramic policy."""

from .contract import (
    CANDIDATE_IMAGE_SIZE,
    PANORAMIC_CHECKPOINT,
    PANORAMIC_IMAGE_SIZE,
    PANORAMIC_REVISION,
    PANORAMIC_SYSTEM_PROMPT_SHA256,
    R2R_PREPROCESSOR_SHA256,
    NavigationOutputError,
    QwenR2RPanoramicBatch,
    QwenR2RPanoramicMemory,
    QwenR2RPanoramicPrefix,
    parse_panoramic_action,
)
from .cuda_graph import CapturedNextTokenGraph, QwenR2RPanoramicGraphRuntime
from .forward import QwenR2RPanoramicNextTokenForward
from .policy import (
    QwenR2RPanoramicDecoder,
    QwenR2RPanoramicPolicy,
    _build_qwen_r2r_panoramic,
    build_qwen_r2r_panoramic,
)
from .runner import QwenR2RPanoramicRunner

QwenVLNMemory = QwenR2RPanoramicMemory
QwenVLNBatch = QwenR2RPanoramicBatch
QwenVLNPrefix = QwenR2RPanoramicPrefix
_parse_panoramic_action = parse_panoramic_action
_FullNextToken = QwenR2RPanoramicNextTokenForward
_CapturedNextTokenGraph = CapturedNextTokenGraph
_Qwen25VLRunner = QwenR2RPanoramicRunner
_NavigationDecoder = QwenR2RPanoramicDecoder
_build = _build_qwen_r2r_panoramic
build_panoramic = build_qwen_r2r_panoramic

__all__ = [
    "CANDIDATE_IMAGE_SIZE",
    "PANORAMIC_CHECKPOINT",
    "PANORAMIC_IMAGE_SIZE",
    "PANORAMIC_REVISION",
    "PANORAMIC_SYSTEM_PROMPT_SHA256",
    "R2R_PREPROCESSOR_SHA256",
    "NavigationOutputError",
    "QwenR2RPanoramicBatch",
    "QwenR2RPanoramicDecoder",
    "QwenR2RPanoramicGraphRuntime",
    "QwenR2RPanoramicMemory",
    "QwenR2RPanoramicNextTokenForward",
    "QwenR2RPanoramicPolicy",
    "QwenR2RPanoramicPrefix",
    "QwenR2RPanoramicRunner",
    "CapturedNextTokenGraph",
    "parse_panoramic_action",
    "build_qwen_r2r_panoramic",
]
