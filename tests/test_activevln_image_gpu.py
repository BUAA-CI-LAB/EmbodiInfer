"""The device image transform reproduces the slow Qwen2-VL image processor bit for bit."""

import os

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


def test_token_only_turns_match_the_full_processor():
    checkpoint = os.environ.get("VVLA_ACTIVEVLN_CKPT")
    if not checkpoint:
        pytest.skip("set VVLA_ACTIVEVLN_CKPT to a local ActiveVLN checkpoint")
    from embodiinfer.policies.activevln.processor_activevln import ActiveVLNProcessor
    from embodiinfer.types import Observation

    processor = ActiveVLNProcessor(checkpoint, None)
    rng = np.random.default_rng(7)
    turns = [("Walk past the sofa.", True), ("Walk past the sofa.", False), ("Stop at the stairs.", True)]
    for height, width in ((480, 640), (256, 256)):
        # The first turn of a size runs the full processor; later ones only tokenize.
        for instruction, initial in turns:
            frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            observation = Observation(
                images=torch.from_numpy(frame).permute(2, 0, 1).float().div_(255),
                state=torch.empty(0),
                instruction_tokens=torch.empty(0, dtype=torch.long),
                instruction=instruction,
            )
            observation.metadata["rgb_uint8"] = frame
            expected = processor.process_turn(observation, initial=initial)
            actual = processor.process_turn_tokens(observation, initial=initial)
            for name in ("input_ids", "attention_mask", "image_grid_thw"):
                assert getattr(actual, name).dtype == getattr(expected, name).dtype
                assert torch.equal(getattr(actual, name), getattr(expected, name))
            assert actual.serialized_prompt == expected.serialized_prompt
            assert actual.prompt_sha256 == expected.prompt_sha256
            assert actual.pixel_values.shape == (0, expected.pixel_values.shape[-1])
            assert actual.pixel_values.dtype == expected.pixel_values.dtype
