"""Reusable Qwen2.5-VL tensor execution primitives."""

from .backend import (
    QWEN25_VL_KERNEL_ABI,
    Qwen25VLAttentionBackend,
    Qwen25VLAttentionSelection,
    normalize_qwen25_vl_attention_backend,
    resolve_qwen25_vl_attention_backend,
)
from .compile import (
    QWEN25_VL_COMPILE_ABI,
    CompiledQwen25VLForward,
    Qwen25VLCompileBackend,
    Qwen25VLCompileRuntime,
    Qwen25VLCompileSelection,
    normalize_qwen25_vl_compile_backend,
)
from .compile_cache import (
    QWEN25_VL_PERSISTENT_CACHE_SCHEMA,
    Qwen25VLPersistentCompileCache,
    qwen25_vl_model_config_sha256,
)
from .history_image_cache import (
    QWEN25_VL_HISTORY_IMAGE_CACHE_ABI,
    QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES,
    HistoryImageCache,
    HistoryImageCacheEntry,
    HistoryImageCacheKey,
)
from .inputs import (
    Qwen25VLPreparedNextTokenInputs,
    prepare_qwen25_vl_next_token_inputs,
)
from .next_token import Qwen25VLNextTokenForward
from .shape_buckets import (
    QWEN25_VL_TEXT_BUCKET_SCHEMA,
    Qwen25VLTextBucket,
    apply_qwen25_vl_text_bucket,
    normalize_qwen25_vl_text_buckets,
    qwen25_vl_bucket_schema,
    qwen25_vl_text_bucket_key,
)
from .vision import VisionBounds

__all__ = [
    "QWEN25_VL_KERNEL_ABI",
    "QWEN25_VL_COMPILE_ABI",
    "QWEN25_VL_PERSISTENT_CACHE_SCHEMA",
    "QWEN25_VL_TEXT_BUCKET_SCHEMA",
    "QWEN25_VL_HISTORY_IMAGE_CACHE_ABI",
    "QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES",
    "CompiledQwen25VLForward",
    "HistoryImageCache",
    "HistoryImageCacheEntry",
    "HistoryImageCacheKey",
    "Qwen25VLAttentionBackend",
    "Qwen25VLAttentionSelection",
    "Qwen25VLCompileBackend",
    "Qwen25VLCompileRuntime",
    "Qwen25VLCompileSelection",
    "Qwen25VLPersistentCompileCache",
    "Qwen25VLTextBucket",
    "Qwen25VLNextTokenForward",
    "Qwen25VLPreparedNextTokenInputs",
    "VisionBounds",
    "apply_qwen25_vl_text_bucket",
    "normalize_qwen25_vl_attention_backend",
    "normalize_qwen25_vl_compile_backend",
    "normalize_qwen25_vl_text_buckets",
    "prepare_qwen25_vl_next_token_inputs",
    "qwen25_vl_bucket_schema",
    "qwen25_vl_model_config_sha256",
    "qwen25_vl_text_bucket_key",
    "resolve_qwen25_vl_attention_backend",
]
