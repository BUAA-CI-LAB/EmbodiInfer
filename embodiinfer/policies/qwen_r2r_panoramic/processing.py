"""Native Qwen2.5-VL-3B R2R panoramic navigation policy."""

from __future__ import annotations

import math

import numpy as np
import torch

from ...models.qwen25_vl.history_image_cache import (
    HistoryImageCacheEntry,
    prepare_history_images,
)
from ...types import Observation
from .contract import (
    CANDIDATE_IMAGE_SIZE,
    PANORAMIC_IMAGE_SIZE,
    QwenR2RPanoramicMemory,
)


def _pil(
    tensor: torch.Tensor,
    size: tuple[int, int],
    *,
    resize: bool,
):
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


class _QwenR2RPanoramicProcessing:
    def _panoramic_prompt(self, observation: Observation, memory: QwenR2RPanoramicMemory):
        panorama = observation.images[0]
        candidate_images = observation.metadata.get("candidate_images")
        if candidate_images is None:
            candidate_images = list(observation.images[1:])
        else:
            candidate_images = [torch.as_tensor(image, dtype=torch.float32) for image in candidate_images]
        if not candidate_images:
            raise ValueError(
                "panoramic profile needs candidate images in Observation.images[1:] "
                "or metadata['candidate_images']"
            )
        candidates = list(observation.metadata.get("candidates", []))
        if not candidates:
            raise ValueError("panoramic profile requires metadata['candidates']")
        if len(candidates) != len(candidate_images):
            raise ValueError("metadata['candidates'] must match the candidate image count")
        content = [
            {
                "type": "text",
                "text": (
                    f"Route instruction: {observation.instruction}\nCurrent step: {len(memory.frames)}\n"
                    f"Cumulative Distance Traveled: {observation.metadata.get('distance_traveled', 0.0)} meters\n\n"
                    "Panorama Images from Previous Steps:"
                ),
            }
        ]
        for index, _ in enumerate(memory.frames):
            content.extend(
                [
                    {"type": "text", "text": f"\n\tPanorama at step: {index}: "},
                    {"type": "image"},
                ]
            )
        if not memory.frames:
            content[0]["text"] += "[]"
        content.extend(
            [
                {"type": "text", "text": "\n\nCurrent Panorama Image:\n\t"},
                {"type": "image"},
            ]
        )
        content.append({"type": "text", "text": "\n\nCandidate Directions:"})
        for index, candidate in enumerate(candidates):
            try:
                angle = round(float(candidate["relative_angle"]), 0)
                distance = round(float(candidate["distance"]), 2)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "each panoramic candidate requires numeric relative_angle and distance"
                ) from exc
            if not math.isfinite(angle) or not math.isfinite(distance):
                raise ValueError("panoramic candidate values must be finite")
            side = "Left" if angle < 0 else "Right"
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"\n\tCandidate: {index}:\n"
                            f"\t\tRelative angle: {abs(angle)} degrees to the {side}\n"
                            f"\t\tDistance: {distance} meters\n\t\tview: "
                        ),
                    },
                    {"type": "image"},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "\n\tCandidate: Stop\n\nNow, analyze the route instruction, your current position, "
                    "and the available candidate directions. Select the candidate that best matches "
                    "the instruction and helps you continue along the correct path. Answer on the "
                    "format: Candidate: (and then the number)"
                ),
            }
        )
        frames = [*memory.frames, panorama, *candidate_images]
        kinds = ["panorama"] * (len(memory.frames) + 1) + ["candidate"] * len(candidate_images)
        return content, frames, kinds, "Candidate: "

    def _render_image(self, frame: torch.Tensor, kind: str):
        if kind == "panorama":
            return _pil(frame, PANORAMIC_IMAGE_SIZE, resize=False)
        if kind == "candidate":
            return _pil(frame, CANDIDATE_IMAGE_SIZE, resize=False)
        raise ValueError(f"Qwen R2R panoramic cannot render image kind {kind!r}")

    def _prepare_batch_impl(
        self,
        observations: list[Observation],
        memories: list[QwenR2RPanoramicMemory],
        *,
        capture_current_entries: bool,
    ) -> tuple[dict[str, torch.Tensor], list[HistoryImageCacheEntry | None]]:
        texts: list[str] = []
        image_batches: list[list] = []
        current_entries: list[HistoryImageCacheEntry | None] = []
        for observation, memory in zip(observations, memories, strict=True):
            content, frames, kinds, label = self._panoramic_prompt(observation, memory)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": content},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text += f"<|im_start|>assistant\n{label}"
            texts.append(text)
            history_count = len(memory.frames)
            rendered, current_entry, _ = prepare_history_images(
                frames,
                kinds,
                history_count=history_count,
                cache=memory.history_image_cache,
                key=self.history_image_cache_key,
                cache_enabled=self.history_image_cache_enabled,
                capture_current_entry=capture_current_entries,
                render=self._render_image,
            )
            image_batches.append(rendered)
            current_entries.append(current_entry)
        encoded = self.processor(text=texts, images=image_batches, padding=True, return_tensors="pt")
        if len(observations) == 1:
            encoded["attention_mask"].fill_(1)
        encoded = self.bucket_encoded(dict(encoded))
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = int(getattr(text_config, "max_position_embeddings", 0))
        if context_limit and encoded["input_ids"].shape[1] > context_limit:
            raise ValueError(
                f"navigation prompt has {encoded['input_ids'].shape[1]} tokens, above {context_limit}"
            )
        return encoded, current_entries

    def _prepare_batch_with_history_entries(
        self, observations: list[Observation], memories: list[QwenR2RPanoramicMemory]
    ) -> tuple[dict[str, torch.Tensor], list[HistoryImageCacheEntry | None]]:
        return self._prepare_batch_impl(observations, memories, capture_current_entries=True)

    def _prepare_batch(
        self, observations: list[Observation], memories: list[QwenR2RPanoramicMemory]
    ) -> dict[str, torch.Tensor]:
        encoded, _ = self._prepare_batch_impl(observations, memories, capture_current_entries=False)
        return encoded

    def _move_encoded(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in encoded.items()
        }

    def _encode_batch_with_history_entries(
        self, observations: list[Observation], memories: list[QwenR2RPanoramicMemory]
    ) -> tuple[dict[str, torch.Tensor], list[HistoryImageCacheEntry | None]]:
        encoded, current_entries = self._prepare_batch_with_history_entries(observations, memories)
        return self._move_encoded(encoded), current_entries

    def _encode_batch(
        self, observations: list[Observation], memories: list[QwenR2RPanoramicMemory]
    ) -> dict[str, torch.Tensor]:
        return self._move_encoded(self._prepare_batch(observations, memories))
