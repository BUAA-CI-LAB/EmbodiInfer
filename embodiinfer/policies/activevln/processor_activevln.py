"""Qwen2.5-VL turn construction for the pinned ActiveVLN R2R profile."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ...types import Observation
from .prompt_activevln import render_turn_text


@dataclass
class ActiveVLNBatch:
    observations: list[Observation]
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return len(self.observations)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> ActiveVLNBatch:
        # Qwen image preprocessing is CPU/PIL based and runs inside encode_prefix.
        del device, dtype
        return self


@dataclass
class ProcessedTurn:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    serialized_prompt: str
    prompt_sha256: str

    def to(self, device: torch.device | str, dtype: torch.dtype) -> ProcessedTurn:
        return ProcessedTurn(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            pixel_values=self.pixel_values.to(device=device, dtype=dtype),
            image_grid_thw=self.image_grid_thw.to(device),
            serialized_prompt=self.serialized_prompt,
            prompt_sha256=self.prompt_sha256,
        )


class ActiveVLNProcessor:
    def __init__(self, checkpoint: str, revision: str, *, allow_download: bool = False) -> None:
        from transformers import AutoProcessor

        path = Path(checkpoint)
        if not path.exists() and not allow_download:
            raise ValueError("activevln requires a local checkpoint snapshot unless allow_download=True")
        self._processor = AutoProcessor.from_pretrained(
            checkpoint,
            revision=revision,
            trust_remote_code=False,
            local_files_only=not allow_download,
        )

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    @staticmethod
    def collate(observations: list[Observation], request_ids: list[str]) -> ActiveVLNBatch:
        if len(observations) != 1:
            raise ValueError("ActiveVLN currently supports one observation per call")
        if len(request_ids) != 1:
            raise ValueError("request_ids must align with the single ActiveVLN observation")
        return ActiveVLNBatch(list(observations), list(request_ids))

    @staticmethod
    def _pil_image(observation: Observation):
        from PIL import Image

        if observation.images.shape[0] != 1:
            raise ValueError("ActiveVLN R2R currently expects one first-person image")
        image = observation.images[0].detach().cpu().clamp(0, 1)
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def process_turn(self, observation: Observation, *, initial: bool) -> ProcessedTurn:
        if not observation.instruction:
            raise ValueError("ActiveVLN requires Observation.instruction")
        text = render_turn_text(self._processor, observation.instruction, initial=initial)
        encoded = self._processor(
            text=[text],
            images=[self._pil_image(observation)],
            padding=False,
            return_tensors="pt",
        )
        required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        missing = [name for name in required if name not in encoded]
        if missing:
            raise RuntimeError(f"Qwen processor omitted required ActiveVLN fields: {missing}")
        return ProcessedTurn(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            pixel_values=encoded.pixel_values,
            image_grid_thw=encoded.image_grid_thw,
            serialized_prompt=text,
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
