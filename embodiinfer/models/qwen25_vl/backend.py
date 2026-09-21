"""Attention backend selection for the shared Qwen2.5-VL forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ...backend.triton import triton_capability
from ...layers.attention import attention_backend_capability

Qwen25VLAttentionBackend = Literal["torch_sdpa", "triton", "auto"]
QWEN25_VL_KERNEL_ABI = "qwen25_vl_hybrid_attention_v3"
_SUPPORTED_BACKENDS = ("torch_sdpa", "triton", "auto")
# Auto-selection is a performance admission policy, not merely a capability
# check. Add an SM only after its full-policy CUDA Graph path clears admission.
_AUTO_TRITON_DEVICE_ALLOWLIST: frozenset[tuple[int, int]] = frozenset()
_AUTO_NO_ADMITTED_PLAN = "no admitted Triton plan for device"


@dataclass(frozen=True)
class Qwen25VLAttentionSelection:
    requested: Qwen25VLAttentionBackend
    resolved: Literal["torch_sdpa", "triton_hybrid"]
    fallback_reason: str | None
    kernel_abi: str
    config: tuple[tuple[str, object], ...]
    layers_backend: Literal["sdpa", "triton_segmented"]
    window_attention: Literal["torch_sdpa", "triton_segmented"]
    full_attention: Literal["torch_sdpa"]
    rope: Literal["torch", "triton"]
    triton_version: str | None

    @property
    def cache_key(self) -> tuple[object, ...]:
        return self.resolved, self.kernel_abi, self.config

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "fallback_reason": self.fallback_reason,
            "kernel_abi": self.kernel_abi,
            "config": dict(self.config),
            "layers_backend": self.layers_backend,
            "window_attention": self.window_attention,
            "full_attention": self.full_attention,
            "rope": self.rope,
            "triton_version": self.triton_version,
        }


def normalize_qwen25_vl_attention_backend(value: str) -> Qwen25VLAttentionBackend:
    if value not in _SUPPORTED_BACKENDS:
        choices = ", ".join(_SUPPORTED_BACKENDS)
        raise ValueError(f"attention_backend must be one of {choices}; got {value!r}")
    return value  # type: ignore[return-value]


def resolve_qwen25_vl_attention_backend(
    requested: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    window_max_length: int,
    full_max_length: int,
) -> Qwen25VLAttentionSelection:
    requested = normalize_qwen25_vl_attention_backend(requested)
    fast_window = window_max_length <= 64
    block_d = 1 << (head_dim - 1).bit_length() if head_dim > 0 else 0
    block_n = 64 if fast_window else 32
    precision_path = "accurate_fp32_single_block" if fast_window else "accurate_fp32_ieee"
    common_config: tuple[tuple[str, object], ...] = (
        ("dtype", str(dtype)),
        ("device", str(device)),
        ("num_query_heads", int(num_query_heads)),
        ("num_key_value_heads", int(num_key_value_heads)),
        ("head_dim", int(head_dim)),
        ("window_max_length", int(window_max_length)),
        ("full_max_length", int(full_max_length)),
    )
    torch_config = common_config + (
        ("window_attention", "torch_sdpa"),
        ("full_attention", "torch_sdpa"),
        ("rope", "torch"),
    )
    if requested == "torch_sdpa":
        return Qwen25VLAttentionSelection(
            requested=requested,
            resolved="torch_sdpa",
            fallback_reason=None,
            kernel_abi=QWEN25_VL_KERNEL_ABI,
            config=torch_config,
            layers_backend="sdpa",
            window_attention="torch_sdpa",
            full_attention="torch_sdpa",
            rope="torch",
            triton_version=None,
        )

    auto_device_admitted = False
    if (
        requested == "auto"
        and _AUTO_TRITON_DEVICE_ALLOWLIST
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        with torch.cuda.device(device):
            compute_capability = torch.cuda.get_device_capability(device)
        auto_device_admitted = compute_capability in _AUTO_TRITON_DEVICE_ALLOWLIST
    if requested == "auto" and not auto_device_admitted:
        return Qwen25VLAttentionSelection(
            requested=requested,
            resolved="torch_sdpa",
            fallback_reason=_AUTO_NO_ADMITTED_PLAN,
            kernel_abi=QWEN25_VL_KERNEL_ABI,
            config=torch_config,
            layers_backend="sdpa",
            window_attention="torch_sdpa",
            full_attention="torch_sdpa",
            rope="torch",
            triton_version=None,
        )

    incompatibilities: list[str] = []
    if device.type != "cuda":
        incompatibilities.append(f"device {device} is not CUDA")
    if dtype not in (torch.float16, torch.bfloat16):
        incompatibilities.append(f"dtype {dtype} is not FP16/BF16")
    if num_query_heads <= 0 or num_key_value_heads <= 0:
        incompatibilities.append("attention head counts must be positive")
    elif num_query_heads % num_key_value_heads != 0:
        incompatibilities.append("query heads must be divisible by key/value heads")
    if head_dim <= 0 or head_dim > 256 or head_dim % 2:
        incompatibilities.append(f"head_dim {head_dim} must be positive, even, and <= 256")
    if window_max_length <= 0 or full_max_length <= 0:
        incompatibilities.append("vision bounds must contain non-empty segments")

    triton_version: str | None = None
    if device.type == "cuda":
        capability = triton_capability(device)
        triton_version = capability.version
        if not capability.available:
            incompatibilities.append(capability.reason or "Triton CUDA capability unavailable")
        else:
            with torch.cuda.device(device):
                attention_available, reason = attention_backend_capability("triton_segmented")
            if not attention_available:
                incompatibilities.append(reason or "triton_segmented backend unavailable")

    if not incompatibilities:
        hybrid_config = common_config + (
            ("max_query_length", int(window_max_length)),
            ("max_key_length", int(window_max_length)),
            ("fast_window", fast_window),
            ("precision_path", precision_path),
            ("exp_mode", "exp2_fp32"),
            ("softmax_dtype", "float32"),
            ("probability_dtype", "float32"),
            ("value_dot_dtype", "float32"),
            ("pv_input_precision", "ieee"),
            ("accumulator_dtype", "float32"),
            ("output_dtype", str(dtype)),
            ("block_d", block_d),
            ("block_m", 32),
            ("block_n", block_n),
            ("num_warps", 4),
            ("num_stages", 1),
            ("window_attention", "triton_segmented"),
            ("full_attention", "torch_sdpa"),
            ("rope", "triton"),
        )
        return Qwen25VLAttentionSelection(
            requested=requested,
            resolved="triton_hybrid",
            fallback_reason=None,
            kernel_abi=QWEN25_VL_KERNEL_ABI,
            config=hybrid_config,
            layers_backend="triton_segmented",
            window_attention="triton_segmented",
            full_attention="torch_sdpa",
            rope="triton",
            triton_version=triton_version,
        )

    reason = "; ".join(incompatibilities)
    if requested == "triton":
        raise RuntimeError(f"Qwen2.5-VL Triton attention is unavailable: {reason}")
    return Qwen25VLAttentionSelection(
        requested=requested,
        resolved="torch_sdpa",
        fallback_reason=reason,
        kernel_abi=QWEN25_VL_KERNEL_ABI,
        config=torch_config,
        layers_backend="sdpa",
        window_attention="torch_sdpa",
        full_attention="torch_sdpa",
        rope="torch",
        triton_version=triton_version,
    )


__all__ = [
    "QWEN25_VL_KERNEL_ABI",
    "Qwen25VLAttentionBackend",
    "Qwen25VLAttentionSelection",
    "normalize_qwen25_vl_attention_backend",
    "resolve_qwen25_vl_attention_backend",
]
