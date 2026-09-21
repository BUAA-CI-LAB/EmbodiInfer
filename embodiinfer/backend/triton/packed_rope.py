"""Fused Triton rotate-half rotary embedding for query and key tensors."""

from __future__ import annotations

import torch

from .capability import require_triton, triton_capability

try:
    import triton
    import triton.language as tl
except Exception as exc:  # CPU-only imports remain valid.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: Exception | None = exc
else:
    _TRITON_IMPORT_ERROR = None


_WARMED_CONFIGS: set[tuple[object, ...]] = set()


if triton is not None:

    @triton.jit
    def _rotate_half_rope_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        q_stride_t,
        q_stride_h,
        k_stride_t,
        k_stride_h,
        cos_stride_t,
        sin_stride_t,
        q_out_stride_t,
        q_out_stride_h,
        k_out_stride_t,
        k_out_stride_h,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        d_index = tl.arange(0, BLOCK_D)
        d_mask = d_index < HEAD_DIM
        half = HEAD_DIM // 2
        paired_index = tl.where(d_index < half, d_index + half, d_index - half)
        sign = tl.where(d_index < half, -1.0, 1.0)
        cosine = tl.load(cos_ptr + token * cos_stride_t + d_index, mask=d_mask, other=0.0).to(tl.float32)
        sine = tl.load(sin_ptr + token * sin_stride_t + d_index, mask=d_mask, other=0.0).to(tl.float32)
        q_head_mask = head < NUM_Q_HEADS
        q = tl.load(
            q_ptr + token * q_stride_t + head * q_stride_h + d_index,
            mask=q_head_mask & d_mask,
            other=0.0,
        ).to(tl.float32)
        q_pair = tl.load(
            q_ptr + token * q_stride_t + head * q_stride_h + paired_index,
            mask=q_head_mask & d_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            q_out_ptr + token * q_out_stride_t + head * q_out_stride_h + d_index,
            q * cosine + sign * q_pair * sine,
            mask=q_head_mask & d_mask,
        )
        k_head_mask = head < NUM_KV_HEADS
        k = tl.load(
            k_ptr + token * k_stride_t + head * k_stride_h + d_index,
            mask=k_head_mask & d_mask,
            other=0.0,
        ).to(tl.float32)
        k_pair = tl.load(
            k_ptr + token * k_stride_t + head * k_stride_h + paired_index,
            mask=k_head_mask & d_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            k_out_ptr + token * k_out_stride_t + head * k_out_stride_h + d_index,
            k * cosine + sign * k_pair * sine,
            mask=k_head_mask & d_mask,
        )

else:
    _rotate_half_rope_kernel = None


def _is_capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def rotate_half_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``x*cos + rotate_half(x)*sin`` to packed Q and K in one launch.

    Q and K are ``[tokens, heads, head_dim]``. Cosine and sine are either
    ``[tokens, head_dim]`` or ``[1, head_dim]``.
    """
    require_triton(q.device)
    if _rotate_half_rope_kernel is None:
        reason = _TRITON_IMPORT_ERROR or "Triton JIT is unavailable"
        raise RuntimeError(f"Triton rotate-half RoPE is unavailable: {reason}")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must be [tokens, heads, head_dim]")
    if q.device.type != "cuda" or k.device != q.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise TypeError("q and k must have the same FP16 or BF16 dtype")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q and k must have equal token counts and head dimensions")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and k must be contiguous")
    head_dim = q.shape[2]
    if head_dim % 2 != 0 or head_dim > 256:
        raise ValueError(f"head_dim must be even and <= 256, got {head_dim}")
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("cos and sin must have matching [tokens|1, head_dim] shapes")
    if cos.shape[0] not in (1, q.shape[0]) or cos.shape[1] != head_dim:
        raise ValueError("cos and sin must provide one row or one row per token")
    if cos.device != q.device or sin.device != q.device:
        raise ValueError("cos and sin must share the q/k CUDA device")
    if cos.dtype not in (torch.float16, torch.bfloat16, torch.float32) or sin.dtype != cos.dtype:
        raise TypeError("cos and sin must have matching FP16, BF16, or FP32 dtype")
    if not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("cos and sin must be contiguous")
    capturing = _is_capturing(q.device)
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    config_key = (
        device_index,
        q.dtype,
        cos.dtype,
        q.shape[1],
        k.shape[1],
        head_dim,
        cos.shape[0] == 1,
    )
    if capturing and config_key not in _WARMED_CONFIGS:
        raise RuntimeError(
            "rotate_half_rope was not JIT-warmed before CUDA Graph capture; "
            "call warmup_rotate_half_rope with identical dimensions first"
        )
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    block_d = triton.next_power_of_2(head_dim)
    cos_stride_t = 0 if cos.shape[0] == 1 else cos.stride(0)
    sin_stride_t = 0 if sin.shape[0] == 1 else sin.stride(0)
    grid = (q.shape[0], max(q.shape[1], k.shape[1]))
    _rotate_half_rope_kernel[grid](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        cos_stride_t,
        sin_stride_t,
        q_out.stride(0),
        q_out.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        NUM_Q_HEADS=q.shape[1],
        NUM_KV_HEADS=k.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    if not capturing:
        _WARMED_CONFIGS.add(config_key)
    return q_out, k_out


def warmup_rotate_half_rope(*args, **kwargs) -> None:
    """JIT-compile and launch a configuration before CUDA Graph capture."""
    rotate_half_rope(*args, **kwargs)


def rope_capability() -> tuple[bool, str | None]:
    capability = triton_capability()
    return capability.available, capability.reason


__all__ = ("rotate_half_rope", "warmup_rotate_half_rope", "rope_capability")
