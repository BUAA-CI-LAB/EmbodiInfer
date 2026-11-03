"""Optional FlashAttention-4 (flash-attn-4, CuTe DSL) forward for Blackwell prefix attention."""

import os
import sys
from collections.abc import Callable

import torch

_FORWARD: list[Callable | None] = []


def forward() -> Callable | None:
    """flash_attn.cute.flash_attn_func, from the environment or EMBODIINFER_FA4_DIR, else None.

    EMBODIINFER_FA4_DIR names a ``pip install --target`` directory holding flash-attn-4 and the
    CuTe DSL release it needs; it is put first on sys.path, so it must be set before anything in
    the process imports another ``cutlass`` Python package.
    """
    if not _FORWARD:
        _FORWARD.append(_load())
    return _FORWARD[0]


def _load() -> Callable | None:
    root = os.environ.get("EMBODIINFER_FA4_DIR")
    if root:
        loaded = sys.modules.get("cutlass")
        if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(root):
            return None
        for path in (root, os.path.join(root, "nvidia_cutlass_dsl", "dsl_packages")):
            if path not in sys.path:
                sys.path.insert(0, path)
    try:
        from flash_attn.cute import flash_attn_func
    except ImportError:
        return None
    return flash_attn_func


def supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """BF16/FP16 BHSD tensors with 256-wide heads on SM100/SM110 and FA4 available."""
    return (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.dtype == k.dtype == v.dtype
        and q.shape[-1] == k.shape[-1] == v.shape[-1] == 256
        and q.shape[1] % k.shape[1] == 0
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and torch.cuda.get_device_capability(q.device)[0] in (10, 11)
        and forward() is not None
    )


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaling: float | None) -> torch.Tensor:
    """Non-causal attention of BHSD q over BHSD k/v (grouped heads); returns a BHSD view."""
    out = forward()(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scaling, causal=False
    )
    return (out[0] if isinstance(out, tuple) else out).transpose(1, 2)
