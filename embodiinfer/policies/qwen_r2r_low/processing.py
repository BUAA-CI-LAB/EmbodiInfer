"""Low-level image conversion owned by this policy."""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_pil(tensor: torch.Tensor, size: tuple[int, int], *, resize: bool):
    from PIL import Image

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"navigation images must be [3,H,W], got {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError("navigation images must be floating point tensors in [0, 1]")
    value = tensor.detach().cpu()
    if not torch.isfinite(value).all() or value.min().item() < 0 or value.max().item() > 1:
        raise ValueError("navigation images must contain finite values in [0, 1]")
    array = (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    if image.size != size:
        if not resize:
            raise ValueError(f"expected image size {size}, got {image.size}")
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


__all__ = ["tensor_to_pil"]
