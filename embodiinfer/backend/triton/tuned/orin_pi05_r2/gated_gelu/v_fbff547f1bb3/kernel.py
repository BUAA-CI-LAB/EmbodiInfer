"""Fused tanh-GELU gate for the recorded pi0.5 ``gated_gelu`` call.

The frozen call is ``y = F.gelu(x0, approximate="tanh") * x1`` for two
``[1, 50, 4096]`` bf16 views with strides ``(409600, 8192, 1)``.  ``x0`` starts
at element offset 0 and ``x1`` at element offset 4096 of each 8192-element token
row.  This module specializes on that recorded layout and falls back to the
Torch reference for every other call.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_gelu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    ROW_WIDTH: tl.constexpr,
    OUT_ROW_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    column = tl.max_contiguous(column, BLOCK)
    row = tl.program_id(1)

    gate_bf16 = tl.load(gate_ptr + row * ROW_WIDTH + column)
    up = tl.load(up_ptr + row * ROW_WIDTH + column)

    gate = gate_bf16.to(tl.float32)
    argument = gate * (1.5957691216057308 + 0.07135481627260025 * (gate * gate))
    activated = gate * tl.sigmoid(argument)
    tl.store(out_ptr + row * OUT_ROW_WIDTH + column, activated * up)


def _supports_recorded_layout(gate: torch.Tensor, up: torch.Tensor) -> bool:
    """Return whether the arguments match the frozen input layout."""
    return (
        gate.is_cuda
        and up.is_cuda
        and gate.device == up.device
        and gate.dtype == torch.bfloat16
        and up.dtype == torch.bfloat16
        and gate.ndim == 3
        and up.ndim == 3
        and tuple(gate.shape) == (1, 50, 4096)
        and tuple(up.shape) == (1, 50, 4096)
        and tuple(gate.stride()) == (409600, 8192, 1)
        and tuple(up.stride()) == (409600, 8192, 1)
        and gate.storage_offset() == 0
        and up.storage_offset() == 4096
    )


def run(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Return ``F.gelu(x0, approximate="tanh") * x1`` for the recorded layout."""
    if not _supports_recorded_layout(x0, x1):
        return torch.nn.functional.gelu(x0, approximate="tanh") * x1

    output = torch.empty((1, 50, 4096), dtype=torch.bfloat16, device=x0.device)
    _gated_gelu_kernel[(4, 50)](
        x0,
        x1,
        output,
        ROW_WIDTH=8192,
        OUT_ROW_WIDTH=4096,
        BLOCK=1024,
        num_warps=2,
    )
    return output


__all__ = ["run"]
