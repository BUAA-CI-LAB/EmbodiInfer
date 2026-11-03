"""Fused tanh-approximate GELU gate: ``out = gelu(x0, approximate="tanh") * x1``.

The production path is two Torch elementwise kernels (``F.gelu`` then a
multiply), so it materialises the GELU result and moves five tensor volumes per
call. This single pass moves three: both operands in, the result out.

The recorded call (``replay.json``) is a pair of row-interleaved views of one
buffer: row ``r`` of the gate starts at ``r * GAP`` and row ``r`` of ``up`` at
``r * GAP + ROW`` with ``GAP == 2 * ROW``, both contiguous in the last
dimension. Both operands are therefore walked with one shared column offset, so
every thread issues whole 128-bit accesses on both loads and on the store.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Host-side launch shape only. These are never read from inside a @triton.jit
# body (Triton 3.6 rejects non-constexpr globals there); the kernel inlines its
# constants as literals.
_BLOCK = 2048
_NUM_WARPS = 4


@triton.jit
def _gated_gelu_tiled(
    gate_ptr,
    up_ptr,
    out_ptr,
    ROW: tl.constexpr,
    GAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One aligned tile of one row; requires ``ROW % BLOCK == 0``, hence no mask.

    ``program_id(0)`` selects the tile inside a row and ``program_id(1)`` the
    row, so no integer division or modulo is emitted. ``BLOCK`` is a multiple of
    the 128-bit vector width and ``GAP`` is a multiple of it too, so the base of
    each access run stays 16-byte aligned and Triton merges the lane values into
    ``ld.global.v4.b32`` / ``st.global.v4.b32`` accesses.
    """
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    src = tl.program_id(1) * GAP + col
    gate = tl.load(gate_ptr + src).to(tl.float32)
    up = tl.load(up_ptr + src).to(tl.float32)
    # tanh GELU: 0.5 * (1 + tanh(z)) == sigmoid(2z), z = sqrt(2/pi)*(g + c*g^3).
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    gate_gelu = gate / (1.0 + tl.exp2(-2.8853900817779268 * inner))
    tl.store(
        out_ptr + tl.program_id(1) * ROW + col,
        (gate_gelu * up).to(tl.bfloat16),
    )


@triton.jit
def _gated_gelu_strided(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    ROW,
    GAP,
    BLOCK: tl.constexpr,
):
    """Masked, layout-tolerant path for shapes the row-tiled kernel cannot tile."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elements
    src = off + (off // ROW) * (GAP - ROW)
    gate = tl.load(gate_ptr + src, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + src, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    gate_gelu = gate / (1.0 + tl.exp2(-2.8853900817779268 * inner))
    tl.store(out_ptr + off, (gate_gelu * up).to(tl.bfloat16), mask=mask)


def _row_foldable(gate: torch.Tensor, up: torch.Tensor, row: int, gap: int) -> bool:
    """Whether both operands form one dense row-strided walk that tiles exactly."""
    if gate.dim() < 2 or gate.shape != up.shape or gate.numel() != up.numel():
        return False
    if row % _BLOCK:
        return False
    for tensor in (gate, up):
        if tensor.stride(-1) != 1 or tensor.stride(-2) != gap:
            return False
        for dim in range(tensor.dim() - 2):
            if tensor.stride(dim) != tensor.shape[dim + 1] * tensor.stride(dim + 1):
                return False
    return True


def run(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Return tanh-approximate ``GELU(x0) * x1`` in a fresh contiguous tensor.

    ``x0`` is the gate and ``x1`` the up operand; neither is written, and the
    result never aliases an input. The result is newly allocated contiguous, so
    it reproduces the recorded output shape, strides, dtype and device exactly.
    """
    gate, up = x0, x1
    if gate.dtype is not torch.bfloat16 or up.dtype is not torch.bfloat16:
        # Outside the recorded contract: stay total and exact.
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    if not gate.is_cuda or gate.numel() == 0:
        return out
    row = gate.shape[-1]
    gap = gate.stride(-2) if gate.dim() >= 2 else row
    n_elements = gate.numel()
    rows = n_elements // row
    if _row_foldable(gate, up, row, gap) and rows <= 65535 and gap >= row:
        _gated_gelu_tiled[(row // _BLOCK, rows)](
            gate,
            up,
            out,
            ROW=row,
            GAP=gap,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    else:
        _gated_gelu_strided[(triton.cdiv(n_elements, 1024),)](
            gate,
            up,
            out,
            n_elements,
            row,
            gap,
            BLOCK=1024,
            num_warps=_NUM_WARPS,
        )
    return out
