"""NaViDA prompt construction, image processing, and multimodal encoding."""

from __future__ import annotations

import io

import numpy as np
import torch

from ...types import Observation
from .contract import (
    NAVIDA_MODEL_IMAGE_SIZE,
    NAVIDA_SOURCE_IMAGE_SIZE,
    NaViDAMemory,
    select_navida_history,
)


def navida_pil_image(tensor: torch.Tensor, size: tuple[int, int], *, resize: bool):
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


class NaViDAProcessingRuntime:
    def _navida_prompt(self, observation: Observation, memory: NaViDAMemory):
        current = observation.images[0]
        history = select_navida_history(memory.frames, current)
        frames = [*history, current]
        content = [
            {
                "type": "text",
                "text": (
                    "Imagine you are a robot programmed for navigation tasks. You have been given a "
                    "video of historical observations"
                ),
            }
        ]
        for _ in history:
            content.append({"type": "image"})
        content.extend(
            [
                {"type": "text", "text": "and an image of the current observation"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f". Your assigned task is: '{observation.instruction}'. Analyze this series of "
                        "images to decide your next move, which could involve turning left or right by "
                        "a specific degree or moving forward a certain distance."
                    ),
                },
            ]
        )
        return content, frames

    def _render_image(self, frame: torch.Tensor, kind: str = "navida"):
        if kind != "navida":
            raise ValueError(f"NaViDA cannot render image kind {kind!r}")
        image = navida_pil_image(frame, NAVIDA_SOURCE_IMAGE_SIZE, resize=False)
        image = image.resize(NAVIDA_MODEL_IMAGE_SIZE)
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        encoded.seek(0)
        from PIL import Image

        with Image.open(encoded) as decoded:
            return decoded.convert("RGB").copy()

    def _prepare_batch(
        self, observations: list[Observation], memories: list[NaViDAMemory]
    ) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        image_batches: list[list] = []
        for observation, memory in zip(observations, memories, strict=True):
            content, frames = self._navida_prompt(observation, memory)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": content},
            ]
            texts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            image_batches.append([self._render_image(frame) for frame in frames])
        encoded = self.processor(text=texts, images=image_batches, padding=True, return_tensors="pt")
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = int(getattr(text_config, "max_position_embeddings", 0))
        if context_limit and encoded["input_ids"].shape[1] > context_limit:
            raise ValueError(
                f"navigation prompt has {encoded['input_ids'].shape[1]} tokens, above {context_limit}"
            )
        return dict(encoded)

    def _encode_batch(
        self, observations: list[Observation], memories: list[NaViDAMemory]
    ) -> dict[str, torch.Tensor]:
        encoded = self._prepare_batch(observations, memories)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in encoded.items()
        }
