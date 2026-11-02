"""Fused ``gated_residual`` for the recorded pi0.5 inference layout.

The registry operator is ``y = residual + update * gate`` with the gate broadcast
over the token axis.  Torch evaluates it as two elementwise kernels plus a
100 KiB bf16 intermediate round trip; on the target (Jetson AGX Orin, sm_87,
14 SMs, cold L2 per sample) both kernels are latency-bound, so the whole call
costs far more than the ~302 KiB of unique traffic it touches.

This module fuses the whole expression into one launch:

* the recorded layout has ``batch == 1`` and row-major ``(51200, 1024, 1)``
  strides, so the call is a single 1-D streaming pass of 51200 bf16 elements,
  ``y[i] = x0[i] + x1[i] * gate[i & 1023]``;
* one program covers two whole rows (2 x 1024 elements = 25 programs, exactly
  covering the tensor, so no mask and no bounds predicate);
* the 1024-element gate vector is loaded once per program and reused for both
  rows from registers, and every stream access is a single 128-bit vector per
  thread (1024 elements / 128 threads = 8 bf16), giving five independent loads
  in flight per thread before the first use.

Numerics are unchanged: the bf16 multiply is performed in fp32 and explicitly
rounded back to bf16 before the fp32 add, exactly like the Torch reference, so
the result is bit-identical to the baseline elementwise path and the explicit
round also prevents mul+add contraction into an FMA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Recorded workload (task/replay.json, case 9aea05ca99896b6d0a0d):
#   x0/x1 [1, 50, 1024] bf16 strides (51200, 1024, 1), x2 [1, 1, 1024] bf16.
_FAST_ROWS = 50
_FAST_HALF = 1024                    # hidden width == one row
_FAST_SHAPE = (1, _FAST_ROWS, _FAST_HALF)
_FAST_GATE_SHAPE = (1, 1, _FAST_HALF)
_FAST_ROW_STRIDE = _FAST_HALF        # rows are packed: flat row-major run
_ROWS_PER_PROGRAM = 2
_FAST_PAIR_BLOCKS = _FAST_ROWS // _ROWS_PER_PROGRAM  # 25; 25*2*1024 == 51200


@triton.jit
def _gated_residual_pair_kernel(x0, x1, gate, out, HALF: tl.constexpr):
    """Two rows per program; ``HALF`` must equal the (power-of-two) hidden width."""
    pid = tl.program_id(0)
    cols = tl.arange(0, HALF)                 # 1024 contiguous elements
    lo = pid * (2 * HALF) + cols              # alignment provable: base % 1024 == 0
    hi = lo + HALF
    g = tl.load(gate + cols).to(tl.float32)   # [1024], shared by both rows
    r0 = tl.load(x0 + lo)
    u0 = tl.load(x1 + lo)                     # all five loads issue before any use
    r1 = tl.load(x0 + hi)
    u1 = tl.load(x1 + hi)
    # fp32 multiply with an explicit round to bf16: identical to Torch's
    # BFloat16 MulFunctor<float> + its separate BFloat16 add kernel.
    p0 = (u0.to(tl.float32) * g).to(tl.bfloat16).to(tl.float32)
    p1 = (u1.to(tl.float32) * g).to(tl.bfloat16).to(tl.float32)
    tl.store(out + lo, (r0.to(tl.float32) + p0).to(tl.bfloat16))
    tl.store(out + hi, (r1.to(tl.float32) + p1).to(tl.bfloat16))


@triton.jit
def _gated_residual_generic_kernel(
    residual,
    update,
    gate,
    output,
    residual_stride_batch,
    residual_stride_token,
    update_stride_batch,
    update_stride_token,
    gate_stride_batch,
    gate_stride_token,
    output_stride_batch,
    output_stride_token,
    sequence_length: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_sequence_length: tl.constexpr,
    is_bfloat16: tl.constexpr,
    block_size: tl.constexpr,
):
    """One row per program, masked: the fallback for any non-recorded layout.

    Semantics and rounding match ``vendor/embodiinfer/backend/triton/norm.py``.
    """
    row = tl.program_id(0)
    batch = row // sequence_length
    token = row % sequence_length
    gate_token = 0 if gate_sequence_length == 1 else token
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    residual_values = tl.load(
        residual + batch * residual_stride_batch + token * residual_stride_token + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    update_values = tl.load(
        update + batch * update_stride_batch + token * update_stride_token + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_values = tl.load(
        gate + batch * gate_stride_batch + gate_token * gate_stride_token + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    product = update_values * gate_values
    if is_bfloat16:
        product = product.to(tl.bfloat16).to(tl.float32)
    else:
        product = product.to(tl.float16).to(tl.float32)
    tl.store(
        output + batch * output_stride_batch + token * output_stride_token + offsets,
        residual_values + product,
        mask=mask,
    )


def _is_recorded_layout(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> bool:
    """True for the recorded argument layout of replay.json (batch == 1).

    The pair kernel addresses all three inputs as one flat row-major run, so the
    live requirements are the widths/row strides below; ``stride(0)`` is never
    used because ``batch == 1``, and the gate's outer strides are never used
    because the gate is indexed by column only.
    """
    return (
        x0.shape == _FAST_SHAPE
        and x1.shape == _FAST_SHAPE
        and x2.shape == _FAST_GATE_SHAPE
        and x0.stride(1) == _FAST_ROW_STRIDE
        and x1.stride(1) == _FAST_ROW_STRIDE
        and x0.stride(2) == 1
        and x1.stride(2) == 1
        and x2.stride(2) == 1
        and x0.dtype == torch.bfloat16
        and x1.dtype == torch.bfloat16
        and x2.dtype == torch.bfloat16
        and x0.device.type == "cuda"
        and x1.device == x0.device
        and x2.device == x0.device
    )


def _generic_supported(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> bool:
    """Contract checks of the vendored Triton wrapper, minus the fast path."""
    return (
        x0.ndim == 3
        and x2.ndim == 3
        and x1.shape == x0.shape
        and x0.shape[0] == x2.shape[0]
        and x2.shape[1] in (1, x0.shape[1])
        and x2.shape[2] == x0.shape[2]
        and x0.stride(2) == 1
        and x1.stride(2) == 1
        and x2.stride(2) == 1
        and x0.dtype in (torch.bfloat16, torch.float16)
        and x1.dtype == x0.dtype
        and x2.dtype == x0.dtype
        and x0.device.type == "cuda"
        and x1.device == x0.device
        and x2.device == x0.device
    )


def gated_residual(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Return ``residual + update * gate`` in one device launch when possible.

    The returned tensor is freshly allocated with the residual's shape, strides
    and dtype; no input is mutated.
    """
    if _is_recorded_layout(residual, update, gate):
        output = torch.empty_like(residual)
        _gated_residual_pair_kernel[(_FAST_PAIR_BLOCKS,)](
            residual,
            update,
            gate,
            output,
            HALF=_FAST_HALF,
            num_warps=4,
        )
        return output

    if not _generic_supported(residual, update, gate):
        return residual + update * gate

    output = torch.empty_like(residual)
    batch, sequence_length, hidden_size = residual.shape
    block_size = triton.next_power_of_2(hidden_size)
    gate_sequence_length = gate.shape[1]
    _gated_residual_generic_kernel[(batch * sequence_length,)](
        residual,
        update,
        gate,
        output,
        residual.stride(0),
        residual.stride(1),
        update.stride(0),
        update.stride(1),
        gate.stride(0),
        gate.stride(1),
        output.stride(0),
        output.stride(1),
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        gate_sequence_length=gate_sequence_length,
        is_bfloat16=residual.dtype == torch.bfloat16,
        block_size=block_size,
        num_warps=8 if block_size >= 2048 else 4,
        enable_fp_fusion=False,  # keep the bf16 multiplication rounded before the add
    )
    return output


def run(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Entry point: ``y = x0 + x1 * x2`` with ``x2`` broadcast over the tokens."""
    return gated_residual(x0, x1, x2)
