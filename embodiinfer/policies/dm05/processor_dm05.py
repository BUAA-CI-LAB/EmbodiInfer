"""Convert shared observations into OpenDM-compatible inputs and batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...types import Observation


class RoboChallengeChatTokenization:
    """RoboChallenge prompt layout used by the Table30 v2 checkpoints.

    This is deliberately a small preprocessing adapter around OpenDM's loaded
    processor.  Model code and token ids remain OpenDM-owned.
    """

    def __init__(self, original):
        self.processor = original.processor
        self.tokenizer = original.tokenizer
        self.n_bins = original.n_bins
        self.max_length = original.max_length
        self.image_prompts = original.image_prompts
        self.add_state = original.add_state
        self.is_history = original.is_history
        self.history_placeholder_token_id = original.history_placeholder_token_id

    def action_to_bin_tokens(self, values: np.ndarray) -> list[int]:
        clipped = np.clip(values, -1.0, 1.0)
        bins = np.floor((clipped + 1.0) / 2.0 * (self.n_bins - 1)).astype(int)
        return np.clip(bins, 0, self.n_bins - 1).tolist()

    def _messages(self, data: dict, prompt: str) -> tuple[list[dict], torch.Tensor | None]:
        meta = data.get("meta_data") or {}
        text = ""
        if meta.get("robot_type") is not None:
            text += f"Robot: {meta['robot_type']}\n"
        if meta.get("control_mode") is not None:
            text += f"Control mode: {meta['control_mode']}\n"
        if meta.get("speed") is not None:
            text += f"Overall speed: {meta['speed']}\n"
        if prompt:
            text += f"{prompt}\n"
        content = [{"type": "text", "text": text}]

        history_pixel_values = None
        if self.is_history:
            history_images = data.get("history_images") or []
            placeholder = data.get("history_placeholder_text")
            if placeholder is None:
                placeholder = "<unused0>" * (16 * len(history_images))
            content[-1]["text"] += "History images: " + placeholder
            if history_images:
                history_pixel_values = self.processor.image_processor(
                    images=[image.convert("RGB") for image in history_images],
                    return_tensors="pt",
                )["pixel_values"]

        for label, image in zip(self.image_prompts, data["images"], strict=True):
            image_label = f"{label} image: "
            if content[-1]["type"] == "text":
                content[-1]["text"] += image_label
            else:
                content.append({"type": "text", "text": image_label})
            content.append({"type": "image", "image": image})
        if self.add_state:
            state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
            token_dim = meta.get("state_token_dim")
            if token_dim is not None:
                state = state[: int(token_dim)]
            state_text = " ".join(str(value) for value in self.action_to_bin_tokens(state))
            content.append({"type": "text", "text": f"States: {state_text}\n"})
        return [{"role": "user", "content": content}], history_pixel_values

    def __call__(self, data: dict) -> dict[str, torch.Tensor | None]:
        prompt = data.get("prompt") or ""
        messages, history_pixel_values = self._messages(data, prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if self.max_length is not None and inputs["input_ids"].shape[1] > self.max_length and prompt:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            overflow = inputs["input_ids"].shape[1] - self.max_length
            keep = max(0, len(prompt_ids) - overflow - 16)
            shortened = self.tokenizer.decode(prompt_ids[:keep], skip_special_tokens=False).strip()
            messages, history_pixel_values = self._messages(data, shortened)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        history_mask = None
        token_type_ids = inputs["token_type_ids"]
        if self.is_history:
            history_mask = inputs["input_ids"] == self.history_placeholder_token_id
            token_type_ids = token_type_ids.clone()
            token_type_ids[history_mask] = 1
        return {
            "action": torch.from_numpy(data["action"]) if "action" in data else None,
            "action_mask": (torch.from_numpy(data["action_mask"]) if "action_mask" in data else None),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "token_type_ids": token_type_ids,
            "history_pixel_values": history_pixel_values,
            "history_mask": history_mask,
        }


def install_robochallenge_tokenizer(runtime) -> None:
    """Replace OpenDM's stock chat transform with the Table30 v2 layout."""
    transforms = getattr(getattr(runtime, "input_transform", None), "transforms", None)
    if not transforms:
        raise TypeError("OpenDM runtime does not expose input_transform.transforms")
    for index, transform in enumerate(transforms):
        if type(transform).__name__ == "ChatTokenization":
            transforms[index] = RoboChallengeChatTokenization(transform)
            return
    raise TypeError("OpenDM runtime has no ChatTokenization transform to replace")


@dataclass
class DM05Batch:
    """A collated OpenDM prefix batch plus output-transform context."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    token_type_ids: torch.Tensor
    history_pixel_values: torch.Tensor | None
    history_mask: torch.Tensor | None
    states: list[np.ndarray]
    meta_data: list[dict[str, Any]]
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> DM05Batch:
        pixel_values = self.pixel_values.to(device, non_blocking=True)
        history = (
            self.history_pixel_values.to(device, non_blocking=True)
            if self.history_pixel_values is not None
            else None
        )
        if dtype is not None:
            pixel_values = pixel_values.to(dtype)
            if history is not None:
                history = history.to(dtype)
        return DM05Batch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            pixel_values=pixel_values,
            token_type_ids=self.token_type_ids.to(device, non_blocking=True),
            history_pixel_values=history,
            history_mask=(
                self.history_mask.to(device, non_blocking=True) if self.history_mask is not None else None
            ),
            states=self.states,
            meta_data=self.meta_data,
            request_ids=self.request_ids,
        )


def image_to_pil(image: Any):
    """Convert CHW/HWC tensors or arrays to the RGB PIL input OpenDM expects."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def pack_images(images: Sequence[Any]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Pack RGB views into Observation.images, padding without resizing.

    Store the returned (height, width) pairs in metadata["image_sizes"] so
    OpenDM receives each original view before its own image preprocessing.
    """
    if not images:
        raise ValueError("DM0.5 requires at least one image")
    views = [np.array(image_to_pil(image), copy=True) for image in images]
    sizes = [(view.shape[0], view.shape[1]) for view in views]
    packed = torch.zeros(
        len(views), 3, max(h for h, _ in sizes), max(w for _, w in sizes), dtype=torch.float32
    )
    for index, (view, (height, width)) in enumerate(zip(views, sizes, strict=True)):
        packed[index, :, :height, :width] = torch.from_numpy(view).permute(2, 0, 1).float() / 255.0
    return packed, sizes


def observation_images(observation: Observation) -> list[Any]:
    """Recover RGB views, removing only padding explicitly described in metadata."""
    images = observation.images
    sizes = observation.metadata.get("image_sizes")
    if sizes is None:
        return [image_to_pil(image) for image in images]
    if not isinstance(sizes, Sequence) or len(sizes) != len(images):
        raise ValueError("metadata.image_sizes must contain one (height, width) pair per camera")
    result = []
    for image, size in zip(images, sizes, strict=True):
        if (
            not isinstance(size, Sequence)
            or len(size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in size)
        ):
            raise ValueError("metadata.image_sizes entries must be integer (height, width) pairs")
        height, width = size
        if not 0 < height <= image.shape[-2] or not 0 < width <= image.shape[-1]:
            raise ValueError("metadata.image_sizes must fit within the observation images")
        result.append(image_to_pil(image[:, :height, :width]))
    return result


def pad_prefix_tensors(
    samples: list[dict[str, torch.Tensor | None]],
    *,
    pad_token_id: int,
    padding_side: str,
) -> dict[str, torch.Tensor | None]:
    """Pad variable prompt lengths and concatenate flattened image batches."""
    if not samples:
        raise ValueError("DM0.5 cannot collate an empty observation list")
    max_len = max(int(sample["input_ids"].shape[1]) for sample in samples)

    def pad_2d(value: torch.Tensor, fill: int | bool) -> torch.Tensor:
        width = max_len - int(value.shape[1])
        if width == 0:
            return value
        padding = torch.full((value.shape[0], width), fill, dtype=value.dtype, device=value.device)
        return (
            torch.cat([padding, value], dim=1)
            if padding_side == "left"
            else torch.cat([value, padding], dim=1)
        )

    result: dict[str, torch.Tensor | None] = {
        "input_ids": torch.cat([pad_2d(sample["input_ids"], pad_token_id) for sample in samples], dim=0),
        "attention_mask": torch.cat([pad_2d(sample["attention_mask"], 0) for sample in samples], dim=0),
        "token_type_ids": torch.cat([pad_2d(sample["token_type_ids"], 0) for sample in samples], dim=0),
        # Gemma3 receives all current views flattened across the request batch.
        "pixel_values": torch.cat([sample["pixel_values"] for sample in samples], dim=0),
    }
    history_masks = [sample.get("history_mask") for sample in samples]
    result["history_mask"] = (
        torch.cat(
            [
                pad_2d(
                    mask if mask is not None else torch.zeros_like(sample["input_ids"], dtype=torch.bool),
                    False,
                )
                for sample, mask in zip(samples, history_masks, strict=True)
            ],
            dim=0,
        )
        if any(mask is not None for mask in history_masks)
        else None
    )
    history_pixels = [sample.get("history_pixel_values") for sample in samples]
    present_history = [value for value in history_pixels if value is not None]
    result["history_pixel_values"] = torch.cat(present_history, dim=0) if present_history else None
    return result
