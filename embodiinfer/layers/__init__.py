"""Operator contracts and backend-routing registries."""

from .activation import GeluMulBackend, PairedGeluBackend, gelu_mul_backends, paired_gelu_backends
from .attention import (
    AttentionBackend,
    SplitKVAttentionBackend,
    available_attention_backends,
    get_attention_backend,
    get_split_kv_attention_backend,
    register_attention,
)
from .config import OperatorBackends
from .linear import (
    FP8Config,
    INT8Config,
    LinearBackend,
    NVFP4Config,
    QuantizationConfig,
    get_linear_backend,
    parse_quantization_config,
    register_linear,
    resolve_linear_backend,
)
from .normalization import NormalizationBackend, NormQuantBackend, norm_quant_backends, normalization_backends
from .quantization import (
    ActivationQuantizer,
    EncodedActivation,
    ProjectionBackend,
    projection_backends,
    quantization_backends,
)
from .registry import BackendRegistry, OperatorCapabilities, OperatorRequest

__all__ = [
    "BackendRegistry",
    "OperatorCapabilities",
    "OperatorRequest",
    "OperatorBackends",
    "NormalizationBackend",
    "NormQuantBackend",
    "GeluMulBackend",
    "PairedGeluBackend",
    "ActivationQuantizer",
    "EncodedActivation",
    "ProjectionBackend",
    "normalization_backends",
    "norm_quant_backends",
    "gelu_mul_backends",
    "paired_gelu_backends",
    "quantization_backends",
    "projection_backends",
    "AttentionBackend",
    "SplitKVAttentionBackend",
    "get_attention_backend",
    "get_split_kv_attention_backend",
    "register_attention",
    "available_attention_backends",
    "FP8Config",
    "INT8Config",
    "NVFP4Config",
    "QuantizationConfig",
    "LinearBackend",
    "get_linear_backend",
    "parse_quantization_config",
    "register_linear",
    "resolve_linear_backend",
]
