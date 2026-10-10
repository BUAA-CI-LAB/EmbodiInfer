"""Bit-exact GPU port of the slow Qwen2-VL image processor for uint8 RGB frames.

transformers' ``Qwen2VLImageProcessor`` resizes with Pillow (fixed-point
bicubic: a horizontal then a vertical pass, each rounded and clipped to uint8),
rescales in float64, casts to float32 and normalizes in float32. Every step
here performs the same integer or correctly rounded IEEE operation, so the
patches equal the CPU processor's output bit for bit.
"""

from __future__ import annotations

import math

import torch

# Pillow Resample.c: 32 bits minus 8 for the pixel and 2 for the sign/overflow.
_PRECISION_BITS = 32 - 8 - 2
_BICUBIC = 3


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


class QwenImageTransform:
    """Device-side ``Qwen2VLImageProcessor`` for batches of same-size uint8 RGB frames."""

    def __init__(self, image_processor) -> None:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        ip = image_processor
        supported = (
            type(ip).__name__ == "Qwen2VLImageProcessor"
            and ip.do_resize
            and ip.do_rescale
            and ip.do_normalize
            and int(ip.resample) == _BICUBIC
        )
        if not supported:
            raise ValueError("GPU preprocessing reproduces only the slow bicubic Qwen2VLImageProcessor")
        self._smart_resize = smart_resize
        self.patch_size = int(ip.patch_size)
        self.merge_size = int(ip.merge_size)
        self.temporal_patch_size = int(ip.temporal_patch_size)
        self.min_pixels = ip.min_pixels
        self.max_pixels = ip.max_pixels
        self.rescale_factor = float(ip.rescale_factor)
        self._mean = torch.tensor(ip.image_mean, dtype=torch.float32)
        self._std = torch.tensor(ip.image_std, dtype=torch.float32)
        self._tables: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._constants: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def resized_size(self, height: int, width: int) -> tuple[int, int]:
        return self._smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def _table(self, in_size: int, out_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (in_size, out_size, device)
        table = self._tables.get(key)
        if table is None:
            index, weight = pillow_bicubic_coefficients(in_size, out_size)
            table = self._tables[key] = (index.to(device), weight.to(device))
        return table

    def resize(self, images: torch.Tensor) -> torch.Tensor:
        """Pillow ``Image.resize(..., BICUBIC)`` of ``[B, H, W, 3]`` uint8 frames."""
        batch, height, width, _ = images.shape
        out_height, out_width = self.resized_size(height, width)
        rounding = 1 << (_PRECISION_BITS - 1)
        device = images.device
        index, weight = self._table(width, out_width, device)
        source = images.to(torch.int32)
        total = torch.full((batch, height, out_width, 3), rounding, dtype=torch.int32, device=device)
        for tap in range(index.shape[1]):
            total += source[:, :, index[:, tap], :] * weight[:, tap, None]
        rows = (total >> _PRECISION_BITS).clamp_(0, 255)
        index, weight = self._table(height, out_height, device)
        total = torch.full((batch, out_height, out_width, 3), rounding, dtype=torch.int32, device=device)
        for tap in range(index.shape[1]):
            total += rows[:, index[:, tap], :, :] * weight[:, tap, None, None]
        return (total >> _PRECISION_BITS).clamp_(0, 255).to(torch.uint8)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, H, W, 3]`` uint8 -> ``[B * grid_h * grid_w, C * T * P * P]`` float32 patches."""
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("GPU preprocessing expects [B, H, W, 3] uint8 frames")
        device = images.device
        constants = self._constants.get(device)
        if constants is None:
            constants = self._constants[device] = (self._mean.to(device), self._std.to(device))
        mean, std = constants
        resized = self.resize(images)
        pixels = (resized.to(torch.float64) * self.rescale_factor).to(torch.float32)
        pixels = (pixels - mean) / std
        batch, height, width, channels = pixels.shape
        p, m, t = self.patch_size, self.merge_size, self.temporal_patch_size
        grid_h, grid_w = height // p, width // p
        patches = pixels.permute(0, 3, 1, 2).unsqueeze(1).expand(batch, t, channels, height, width)
        patches = patches.reshape(batch, t, channels, grid_h // m, m, p, grid_w // m, m, p)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return patches.reshape(batch * grid_h * grid_w, channels * t * p * p)
