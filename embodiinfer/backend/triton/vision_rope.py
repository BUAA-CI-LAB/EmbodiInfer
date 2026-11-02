"""Split packed Qwen2.5-VL vision QKV and apply its FP32 rotate-half RoPE.

Bit-identical to ``apply_rotary_pos_emb_vision`` in transformers 4.51.3: both
products and their sum round in FP32 (no FMA contraction) before one final
rounding to the activation dtype.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional accelerator
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _qkv_rope(
        QKV, Cos, Sin, Q, K, V, H: tl.constexpr, D: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr
    ):
        token = tl.program_id(0)
        heads = tl.arange(0, BH)[:, None]
        dims = tl.arange(0, BD)[None, :]
        live = (heads < H) & (dims < D)
        paired = tl.where(dims < D // 2, dims + D // 2, dims - D // 2)
        sign = tl.where(dims < D // 2, -1.0, 1.0)
        cosine = tl.load(Cos + token * D + dims, dims < D, other=0.0).to(tl.float32)
        sine = tl.load(Sin + token * D + dims, dims < D, other=0.0).to(tl.float32)
        source = QKV + token * 3 * H * D + heads * D
        target = token * H * D + heads * D + dims
        q = tl.load(source + dims, live, other=0.0).to(tl.float32)
        q_pair = tl.load(source + paired, live, other=0.0).to(tl.float32)
        tl.store(Q + target, q * cosine + sign * q_pair * sine, live)
        k = tl.load(source + H * D + dims, live, other=0.0).to(tl.float32)
        k_pair = tl.load(source + H * D + paired, live, other=0.0).to(tl.float32)
        tl.store(K + target, k * cosine + sign * k_pair * sine, live)
        tl.store(V + target, tl.load(source + 2 * H * D + dims, live, other=0.0), live)


def vision_qkv_rope(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contiguous ``[tokens, heads, dim]`` rotated Q, rotated K and V."""
    if triton is None or not qkv.is_cuda:
        raise RuntimeError("vision QKV RoPE requires CUDA and Triton")
    if qkv.dtype not in (torch.float16, torch.bfloat16) or qkv.ndim != 2 or not qkv.is_contiguous():
        raise ValueError("vision QKV must be a contiguous FP16/BF16 [tokens, 3*heads*dim] matrix")
    tokens, width = qkv.shape
    dim = width // (3 * heads)
    if dim < 2 or dim % 2 or width != 3 * heads * dim:
        raise ValueError("vision QKV width must split into three even-width head groups")
    if (
        cos.shape != (tokens, dim)
        or sin.shape != cos.shape
        or cos.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or sin.dtype != cos.dtype
        or not cos.is_contiguous()
        or not sin.is_contiguous()
        or cos.device != qkv.device
        or sin.device != qkv.device
    ):
        raise ValueError("vision rotary tables must be contiguous [tokens, dim] on the QKV device")
    q, k, v = (torch.empty((tokens, heads, dim), device=qkv.device, dtype=qkv.dtype) for _ in range(3))
    _qkv_rope[(tokens,)](
        qkv,
        cos,
        sin,
        q,
        k,
        v,
        heads,
        dim,
        triton.next_power_of_2(heads),
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k, v
