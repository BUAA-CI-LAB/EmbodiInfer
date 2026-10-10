"""The device image transform reproduces the slow Qwen2-VL image processor bit for bit."""

import numpy as np
import pytest
import torch

from embodiinfer.policies.activevln.image_gpu import QwenImageTransform


def _processor(**kwargs):
    # Imported lazily: registering policies must not pull in transformers.
    module = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    return module.Qwen2VLImageProcessor(**kwargs)


@pytest.mark.parametrize(
    ("height", "width", "max_pixels"),
    [(480, 640, 224 * 308), (480, 640, None), (300, 200, None), (97, 131, None)],
)
def test_device_transform_matches_slow_processor_bitwise(height, width, max_pixels):
    from PIL import Image

    processor = _processor(**({} if max_pixels is None else {"max_pixels": max_pixels}))
    frame = np.random.default_rng(height * width).integers(0, 256, (height, width, 3), dtype=np.uint8)
    expected = processor(images=[Image.fromarray(frame)], return_tensors="pt")["pixel_values"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual = QwenImageTransform(processor)(torch.from_numpy(frame)[None].to(device)).cpu()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_device_transform_rejects_unsupported_processors():
    from PIL import Image

    with pytest.raises(ValueError, match="slow bicubic"):
        QwenImageTransform(_processor(resample=Image.Resampling.BILINEAR))
