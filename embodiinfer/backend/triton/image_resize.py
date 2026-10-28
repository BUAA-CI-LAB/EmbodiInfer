"""Pillow-compatible integer bicubic RGB resize, adapted from ActiveVLN.

Weights and each uint8 pass retain Pillow's 22-bit fixed-point rounding.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_PRECISION_BITS = 22


def _bicubic(x: float) -> float:
    a = -0.5
    if x < 0.0:
        x = -x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def pillow_bicubic_coefficients(in_size: int, out_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``precompute_coeffs`` + ``normalize_coeffs_8bpc`` for a full-image box.

    Returns source indices and fixed-point weights, both ``[out_size, ksize]``;
    taps past a row's support carry weight zero and a valid index.
    """
    if in_size <= 0 or out_size <= 0:
        raise ValueError("image dimensions must be positive")
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    ss = 1.0 / filterscale
    indices, weights = [], []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        taps = [_bicubic((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        total = 0.0
        for tap in taps:
            total += tap
        if total != 0.0:
            taps = [tap / total for tap in taps]
        fixed = [
            int(-0.5 + tap * (1 << _PRECISION_BITS)) if tap < 0 else int(0.5 + tap * (1 << _PRECISION_BITS))
            for tap in taps
        ]
        indices.append([xmin + x for x in range(xmax)] + [xmin] * (ksize - xmax))
        weights.append(fixed + [0] * (ksize - xmax))
    return torch.tensor(indices, dtype=torch.long), torch.tensor(weights, dtype=torch.int32)


if triton is not None:

    @triton.jit
    def _resize_pass(
        source,
        indices,
        weights,
        output,
        total,
        HEIGHT: tl.constexpr,
        WIDTH: tl.constexpr,
        OUT: tl.constexpr,
        TAPS: tl.constexpr,
        HORIZONTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < total
        channel = offset % 3
        if HORIZONTAL:
            coordinate = offset // 3 % OUT
            base = offset // (3 * OUT) * (3 * WIDTH) + channel
            stride = 3
        else:
            coordinate = offset // (3 * WIDTH) % OUT
            batch = offset // (3 * WIDTH * OUT)
            base = batch * HEIGHT * WIDTH * 3 + offset % (3 * WIDTH)
            stride = 3 * WIDTH
        accum = tl.full((BLOCK,), 1 << 21, tl.int32)
        for tap in range(TAPS):
            index = tl.load(indices + coordinate * TAPS + tap, valid, other=0)
            weight = tl.load(weights + coordinate * TAPS + tap, valid, other=0)
            value = tl.load(source + base + index * stride, valid, other=0).to(tl.int32)
            accum += value * weight
        result = tl.minimum(tl.maximum(accum >> 22, 0), 255)
        tl.store(output + offset, result.to(tl.uint8), valid)


class PillowBicubicResize:
    """Resize contiguous BHWC uint8 RGB with exact Pillow pass rounding.

    CUDA uses two integer kernels. The CPU implementation is useful for
    reference checks. At most 16 dimension/device coefficient tables are kept.
    """

    def __init__(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")
        self.height, self.width = height, width
        self._tables: OrderedDict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )

    def _table(self, source: int, target: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (source, target, device)
        if key not in self._tables:
            index, weight = pillow_bicubic_coefficients(source, target)
            self._tables[key] = index.to(device), weight.to(device)
            if len(self._tables) > 16:
                self._tables.popitem(last=False)
        self._tables.move_to_end(key)
        return self._tables[key]

    def _pass(self, source: torch.Tensor, target: int, *, horizontal: bool) -> torch.Tensor:
        batch, height, width, _ = source.shape
        length = width if horizontal else height
        if length == target:
            return source
        index, weight = self._table(length, target, source.device)
        shape = (batch, height, target, 3) if horizontal else (batch, target, width, 3)
        if source.is_cuda:
            if triton is None:
                raise RuntimeError("CUDA Pillow resize requires Triton")
            output = torch.empty(shape, dtype=torch.uint8, device=source.device)
            _resize_pass[(triton.cdiv(output.numel(), 256),)](
                source,
                index,
                weight,
                output,
                output.numel(),
                HEIGHT=height,
                WIDTH=width,
                OUT=target,
                TAPS=index.shape[1],
                HORIZONTAL=horizontal,
                BLOCK=256,
                num_warps=4,
            )
            return output
        source = source.to(torch.int32)
        total = torch.full(shape, 1 << 21, dtype=torch.int32, device=source.device)
        for tap in range(index.shape[1]):
            if horizontal:
                total += source[:, :, index[:, tap], :] * weight[:, tap, None]
            else:
                total += source[:, index[:, tap], :, :] * weight[:, tap, None, None]
        return (total >> 22).clamp_(0, 255).to(torch.uint8)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return BHWC uint8 RGB at the configured output dimensions."""
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Pillow resize expects BHWC uint8 RGB images")
        if any(size <= 0 for size in images.shape):
            raise ValueError("image dimensions must be positive")
        rows = self._pass(images.contiguous(), self.width, horizontal=True)
        return self._pass(rows, self.height, horizontal=False)
