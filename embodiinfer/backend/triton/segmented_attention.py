"""Triton non-causal attention over packed variable-length segments."""

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
    def _segmented_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_offsets_ptr,
        kv_offsets_ptr,
        out_ptr,
        q_stride_t,
        q_stride_h,
        k_stride_t,
        k_stride_h,
        v_stride_t,
        v_stride_h,
        out_stride_t,
        out_stride_h,
        scale,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_KV_LENGTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        FAST_WINDOW: tl.constexpr,
    ):
        segment = tl.program_id(0)
        q_head = tl.program_id(1)
        q_block = tl.program_id(2)
        q_start = tl.load(q_offsets_ptr + segment)
        q_end = tl.load(q_offsets_ptr + segment + 1)
        kv_start = tl.load(kv_offsets_ptr + segment)
        kv_end = tl.load(kv_offsets_ptr + segment + 1)
        q_length = q_end - q_start
        kv_length = kv_end - kv_start
        q_index = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        d_index = tl.arange(0, BLOCK_D)
        q_mask = q_index < q_length
        d_mask = d_index < HEAD_DIM
        q = tl.load(
            q_ptr + (q_start + q_index[:, None]) * q_stride_t + q_head * q_stride_h + d_index[None, :],
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)
        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for kv_block_start in range(0, MAX_KV_LENGTH, BLOCK_N):
            kv_index = kv_block_start + tl.arange(0, BLOCK_N)
            kv_mask = kv_index < kv_length
            k = tl.load(
                k_ptr + (kv_start + kv_index[:, None]) * k_stride_t + kv_head * k_stride_h + d_index[None, :],
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(q_mask[:, None] & kv_mask[None, :], scores, -float("inf"))
            if FAST_WINDOW:
                block_max = tl.where(q_mask, tl.max(scores, axis=1), 0.0)
                probabilities = tl.exp2((scores - block_max[:, None]) * 1.4426950408889634)
            else:
                block_max = tl.maximum(row_max, tl.max(scores, axis=1))
                block_max = tl.where(q_mask, block_max, 0.0)
                correction = tl.where(
                    q_mask,
                    tl.exp2((row_max - block_max) * 1.4426950408889634),
                    0.0,
                )
                probabilities = tl.exp2((scores - block_max[:, None]) * 1.4426950408889634)
            probabilities = tl.where(q_mask[:, None] & kv_mask[None, :], probabilities, 0.0)
            v = tl.load(
                v_ptr + (kv_start + kv_index[:, None]) * v_stride_t + kv_head * v_stride_h + d_index[None, :],
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if FAST_WINDOW:
                accumulator = tl.dot(
                    probabilities,
                    v.to(tl.float32),
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )
                row_sum = tl.sum(probabilities, axis=1)
            else:
                accumulator = accumulator * correction[:, None] + tl.dot(
                    probabilities,
                    v.to(tl.float32),
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )
                row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
                row_max = block_max
        output = accumulator / row_sum[:, None]
        tl.store(
            out_ptr + (q_start + q_index[:, None]) * out_stride_t + q_head * out_stride_h + d_index[None, :],
            output,
            mask=q_mask[:, None] & d_mask[None, :],
        )

else:
    _segmented_attention_kernel = None


def _is_capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
) -> None:
    require_triton(q.device)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be packed [tokens, heads, head_dim] tensors")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Triton segmented attention supports FP16/BF16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, and v must be contiguous")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes must match, got {k.shape} and {v.shape}")
    if q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1] != 0:
        raise ValueError("head dimensions must match and query heads must divide by KV heads")
    if q.shape[2] > 256:
        raise ValueError(f"head_dim must be <= 256, got {q.shape[2]}")
    for name, offsets in (("q_segment_offsets", q_offsets), ("kv_segment_offsets", kv_offsets)):
        if offsets.ndim != 1 or offsets.numel() < 2:
            raise ValueError(f"{name} must be one-dimensional with at least two entries")
        if offsets.device != q.device or offsets.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be an int32/int64 tensor on {q.device}")
        if not offsets.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q_offsets.numel() != kv_offsets.numel():
        raise ValueError("query and key/value offsets must describe the same segments")


def _resolve_max_length(
    offsets: torch.Tensor,
    supplied: int | None,
    name: str,
    *,
    capturing: bool,
) -> int:
    if supplied is None:
        if capturing:
            raise RuntimeError(f"{name} must be supplied during CUDA Graph capture")
        supplied = int((offsets[1:] - offsets[:-1]).max().item())
    supplied = int(supplied)
    if supplied <= 0:
        raise ValueError(f"{name} must be positive, got {supplied}")
    if not capturing:
        lengths = offsets[1:] - offsets[:-1]
        minimum = int(lengths.min().item())
        actual = int(lengths.max().item())
        if minimum <= 0:
            raise ValueError(f"{name} offsets must define non-empty increasing segments")
        if actual > supplied:
            raise ValueError(f"{name}={supplied} is smaller than actual maximum {actual}")
    return supplied


def segmented_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor | None = None,
    *,
    scaling: float | None = None,
    max_query_length: int | None = None,
    max_key_length: int | None = None,
) -> torch.Tensor:
    """Attend independently inside packed variable-length non-causal segments.

    Inputs are ``q[total_q, q_heads, d]`` and ``k/v[total_kv, kv_heads, d]``.
    Offset tensors hold ``num_segments + 1`` cumulative token positions.
    """
    if kv_segment_offsets is None:
        kv_segment_offsets = q_segment_offsets
    _validate_inputs(q, k, v, q_segment_offsets, kv_segment_offsets)
    if _segmented_attention_kernel is None:
        reason = _TRITON_IMPORT_ERROR or "Triton JIT is unavailable"
        raise RuntimeError(f"Triton segmented attention is unavailable: {reason}")
    capturing = _is_capturing(q.device)
    max_q = _resolve_max_length(q_segment_offsets, max_query_length, "max_query_length", capturing=capturing)
    max_kv = _resolve_max_length(kv_segment_offsets, max_key_length, "max_key_length", capturing=capturing)
    if not capturing:
        if int(q_segment_offsets[0].item()) != 0 or int(q_segment_offsets[-1].item()) != q.shape[0]:
            raise ValueError("query offsets must start at 0 and end at total_q")
        if int(kv_segment_offsets[0].item()) != 0 or int(kv_segment_offsets[-1].item()) != k.shape[0]:
            raise ValueError("key/value offsets must start at 0 and end at total_kv")
    head_dim = q.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    block_m = 32
    fast_window = max_kv <= 64
    if fast_window:
        precision_path = "accurate_fp32_single_block"
        block_n = 64
        num_stages = 1
    else:
        precision_path = "accurate_fp32_ieee"
        block_n = 32
        num_stages = 1
    exp_mode = "exp2_fp32"
    num_warps = 4
    scale = float(head_dim**-0.5 if scaling is None else scaling)
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    config_key = (
        device_index,
        q.dtype,
        q.shape[1],
        k.shape[1],
        head_dim,
        max_q,
        max_kv,
        precision_path,
        exp_mode,
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
    if capturing and config_key not in _WARMED_CONFIGS:
        raise RuntimeError(
            "segmented_attention was not JIT-warmed before CUDA Graph capture; "
            "call warmup_segmented_attention with identical bounds first"
        )
    output = torch.empty_like(q)
    grid = (
        q_segment_offsets.numel() - 1,
        q.shape[1],
        triton.cdiv(max_q, block_m),
    )
    _segmented_attention_kernel[grid](
        q,
        k,
        v,
        q_segment_offsets,
        kv_segment_offsets,
        output,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        output.stride(0),
        output.stride(1),
        scale,
        NUM_Q_HEADS=q.shape[1],
        NUM_KV_HEADS=k.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        MAX_KV_LENGTH=max_kv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        FAST_WINDOW=fast_window,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if not capturing:
        _WARMED_CONFIGS.add(config_key)
    return output


def warmup_segmented_attention(*args, **kwargs) -> None:
    """JIT-compile and launch a configuration before CUDA Graph capture."""
    segmented_attention(*args, **kwargs)


class SegmentedTritonAttention:
    """Expose packed Triton attention through the common attention contract."""

    name = "triton_segmented"

    def __init__(self) -> None:
        self._offsets: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def capability() -> tuple[bool, str | None]:
        capability = triton_capability()
        return capability.available, capability.reason

    def _dense_offsets(self, q, batch, query_length, key_length):
        device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
        cache_key = (device_index, batch, query_length, key_length)
        if cache_key not in self._offsets:
            if _is_capturing(q.device):
                raise RuntimeError(
                    "dense offsets were not prepared before CUDA Graph capture; "
                    "call attend once eagerly with identical dimensions"
                )
            self._offsets[cache_key] = (
                torch.arange(
                    0,
                    (batch + 1) * query_length,
                    query_length,
                    device=q.device,
                    dtype=torch.int32,
                ),
                torch.arange(
                    0,
                    (batch + 1) * key_length,
                    key_length,
                    device=q.device,
                    dtype=torch.int32,
                ),
            )
        return self._offsets[cache_key]

    def attend(self, q, k, v, attn_mask=None, scaling=None, dropout_p=0.0):
        if attn_mask is not None:
            raise ValueError("triton_segmented supports non-causal unmasked attention only")
        if dropout_p != 0.0:
            raise ValueError("triton_segmented does not support dropout")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("dense q, k, and v must be [batch, heads, sequence, head_dim]")
        batch, q_heads, query_length, head_dim = q.shape
        if k.shape[0] != batch or v.shape[0] != batch or k.shape != v.shape:
            raise ValueError("dense k and v must have matching shapes and q batch size")
        if k.shape[3] != head_dim:
            raise ValueError("dense q, k, and v must share head_dim")
        key_length = k.shape[2]
        q_offsets, kv_offsets = self._dense_offsets(q, batch, query_length, key_length)
        packed_q = q.permute(0, 2, 1, 3).contiguous().view(-1, q_heads, head_dim)
        packed_k = k.permute(0, 2, 1, 3).contiguous().view(-1, k.shape[1], head_dim)
        packed_v = v.permute(0, 2, 1, 3).contiguous().view(-1, v.shape[1], head_dim)
        packed_output = segmented_attention(
            packed_q,
            packed_k,
            packed_v,
            q_offsets,
            kv_offsets,
            scaling=scaling,
            max_query_length=query_length,
            max_key_length=key_length,
        )
        return packed_output.view(batch, query_length, q_heads, head_dim).permute(0, 2, 1, 3)

    def attend_segmented(self, *args, **kwargs) -> torch.Tensor:
        return segmented_attention(*args, **kwargs)

    def warmup(self, *args, **kwargs) -> None:
        self.attend(*args, **kwargs)


__all__ = (
    "SegmentedTritonAttention",
    "segmented_attention",
    "warmup_segmented_attention",
)
