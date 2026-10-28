"""Device resize and exact normalization for the published SigLIP processor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...backend.triton.image_resize import PillowBicubicResize


class StreamVLNImageTransform:
    """Preserve slow SigLIP float-image semantics while moving resize to CUDA.

    Host validation and float-to-uint8 conversion match Transformers 4.51.3.
    The device performs Pillow integer resize and a float32 normalization lookup.
    Integer-valued float frames follow the reference's no-rescale special case.
    """

    def __init__(self, image_processor: Any) -> None:
        ip = image_processor
        if not (
            type(ip).__name__ == "SiglipImageProcessor"
            and ip.do_resize
            and not ip.do_rescale
            and ip.do_normalize
            and int(ip.resample) == 3
            and list(ip.image_mean) == [0.5] * 3
            and list(ip.image_std) == [0.5] * 3
            and set(ip.size) == {"height", "width"}
        ):
            raise ValueError("GPU preprocessing requires the slow bicubic StreamVLN SiglipImageProcessor")
        self.resize = PillowBicubicResize(int(ip.size["height"]), int(ip.size["width"]))
        self._luts: dict[tuple[torch.device, bool], torch.Tensor] = {}

    def __call__(self, frame: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        """Convert CHW float [0,1] or uint8 RGB, preserving the float-reference result."""
        if frame.ndim != 3 or frame.shape[0] != 3 or min(frame.shape[1:]) <= 0:
            raise ValueError(f"StreamVLN expects CHW RGB frames, got {tuple(frame.shape)}")
        if frame.dtype == torch.uint8:
            array = frame.detach().cpu().numpy()
            # The float reference treats all-{0,255} byte images as binary
            # after /255 and skips its PIL rescale. Preserve that special case.
            rescale = not bool(np.all((array == 0) | (array == 255)))
            if not rescale:
                array = array // 255
        else:
            array = frame.detach().float().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("StreamVLN frame contains non-finite values")
            if array.min() < 0.0 or array.max() > 1.0:
                raise ValueError("StreamVLN observation images must be float RGB in [0, 1]")
            rescale = not np.allclose(array, array.astype(int))
            if rescale:
                array = (array.astype(np.float64) * 255).astype(np.float32)
        image = torch.from_numpy(np.ascontiguousarray(array.astype(np.uint8).transpose(1, 2, 0)))
        pixels = self.resize(image.unsqueeze(0).to(device=device))[0]
        key = (device, rescale)
        if key not in self._luts:
            values = np.arange(256, dtype=np.uint8)
            values = (
                (values.astype(np.float64) * (1 / 255)).astype(np.float32)
                if rescale
                else values.astype(np.float32)
            )
            values = (values - np.float32(0.5)) / np.float32(0.5)
            self._luts[key] = torch.from_numpy(values).to(device=device)
        return self._luts[key][pixels.long()].permute(2, 0, 1).contiguous()
