"""SlowFast frame selection and preprocessing for StreamVLN."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import torch

from ...types import Observation
from .prompt import (
    VISION_PHRASE,
    render_streamvln_prompt,
    tokenize_multimodal_prompt,
)

_RESPONSE_PREFIX = "<|im_start|>assistant\n"
_ACTION_SYMBOLS = ("STOP", "\u2191", "\u2190", "\u2192")


@dataclass
class StreamVLNBatch:
    observations: list[Observation]
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return len(self.observations)

    def to(self, *args, **kwargs) -> StreamVLNBatch:
        # StreamVLN preprocesses the CPU frame bank inside encode_prefix and
        # transfers only the resulting model inputs.
        del args, kwargs
        return self


@dataclass(frozen=True)
class StreamVLNInputs:
    input_ids: torch.Tensor
    current_pixel_values: torch.Tensor
    memory_pixel_values: torch.Tensor | None
    slow_frame_indices: tuple[int, ...]
    window_start: bool
    prompt: str
    image_position: int
    memory_position: int | None


class StreamVLNProcessor:
    def __init__(
        self,
        tokenizer: Any,
        image_processor: Any,
        *,
        window_size: int = 32,
        num_history: int = 8,
        vision_phrase: str = VISION_PHRASE,
    ) -> None:
        if window_size <= 0 or num_history <= 0 or window_size < num_history:
            raise ValueError("invalid StreamVLN SlowFast profile")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.window_size = window_size
        self.num_history = num_history
        self.vision_phrase = vision_phrase
        self.response_prefix_ids = tuple(
            int(token_id) for token_id in tokenize_multimodal_prompt(_RESPONSE_PREFIX, tokenizer).tolist()
        )
        if not self.response_prefix_ids:
            raise ValueError("StreamVLN response prefix must contain at least one token")
        action_token_ids = []
        for symbol in _ACTION_SYMBOLS:
            token_ids = tokenize_multimodal_prompt(symbol, tokenizer).tolist()
            if len(token_ids) == 1:
                action_token_ids.append(int(token_ids[0]))
        self.action_token_ids = frozenset(action_token_ids)
        self._input_ids_cache: dict[tuple[str, bool, bool], torch.Tensor] = {}
        self._pixel_cache: dict[
            int,
            tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor],
        ] = {}

    def clear_runtime_caches(self) -> None:
        """Drop CPU preprocessing artifacts created while capturing startup graphs."""
        self._input_ids_cache.clear()
        self._pixel_cache.clear()

    def collate(
        self,
        observations: list[Observation],
        request_ids: list[str],
    ) -> StreamVLNBatch:
        if len(observations) != 1 or len(request_ids) != 1:
            raise ValueError("StreamVLN is recurrent and requires batch size 1 per replica")
        return StreamVLNBatch(observations=list(observations), request_ids=list(request_ids))

    def slow_frame_indices(self, step_id: int) -> tuple[int, ...]:
        if step_id <= 0 or step_id % self.window_size:
            return ()
        stride = step_id // self.num_history
        if stride <= 0:
            raise RuntimeError("StreamVLN slow-memory stride must be positive")
        return tuple(range(0, step_id, stride))[: self.num_history]

    def _preprocess(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"StreamVLN expects CHW RGB frames, got {tuple(frame.shape)}")
        frame = frame.detach().float().cpu()
        if not bool(torch.isfinite(frame).all()):
            raise ValueError("StreamVLN frame contains non-finite values")
        if float(frame.min()) < 0.0 or float(frame.max()) > 1.0:
            raise ValueError("StreamVLN observation images must be float RGB in [0, 1]")
        values = self.image_processor.preprocess(frame, return_tensors="pt")
        pixels = values["pixel_values"][0]
        return pixels.pin_memory() if torch.cuda.is_available() else pixels

    def _preprocess_cached(self, frame: torch.Tensor) -> torch.Tensor:
        key = id(frame)
        cached = self._pixel_cache.get(key)
        if cached is not None and cached[0]() is frame:
            return cached[1]

        pixels = self._preprocess(frame)

        def discard(_reference, *, cache_key=key) -> None:
            self._pixel_cache.pop(cache_key, None)

        self._pixel_cache[key] = (weakref.ref(frame, discard), pixels)
        return pixels

    def process(
        self,
        current: torch.Tensor,
        instruction: str,
        *,
        step_id: int,
        frame_bank: tuple[torch.Tensor, ...] | None = None,
    ) -> StreamVLNInputs:
        if frame_bank is not None and len(frame_bank) != step_id + 1:
            raise ValueError("StreamVLN requires one committed frame per streaming step")
        slow_indices = self.slow_frame_indices(step_id)
        window_start = step_id % self.window_size == 0
        prompt = render_streamvln_prompt(
            instruction,
            window_start=window_start,
            include_memory=bool(slow_indices),
            vision_phrase=self.vision_phrase,
        )
        prompt_key = (instruction, window_start, bool(slow_indices))
        input_ids = self._input_ids_cache.get(prompt_key)
        if input_ids is None:
            input_ids = tokenize_multimodal_prompt(prompt, self.tokenizer).unsqueeze(0)
            if torch.cuda.is_available():
                input_ids = input_ids.pin_memory()
            self._input_ids_cache[prompt_key] = input_ids
        image_matches = torch.where(input_ids[0] == -200)[0]
        memory_matches = torch.where(input_ids[0] == -300)[0]
        if image_matches.numel() != 1 or memory_matches.numel() > 1:
            raise ValueError("invalid StreamVLN multimodal prompt sentinels")
        image_position = int(image_matches[0])
        memory_position = int(memory_matches[0]) if memory_matches.numel() else None
        current_pixels = self._preprocess_cached(current).unsqueeze(0)
        memory_pixels = None
        if slow_indices and frame_bank is not None:
            memory_pixels = torch.stack(
                [self._preprocess_cached(frame_bank[index]) for index in slow_indices]
            )
            if torch.cuda.is_available():
                memory_pixels = memory_pixels.pin_memory()
        return StreamVLNInputs(
            input_ids=input_ids,
            current_pixel_values=current_pixels,
            memory_pixel_values=memory_pixels,
            slow_frame_indices=slow_indices,
            window_start=window_start,
            prompt=prompt,
            image_position=image_position,
            memory_position=memory_position,
        )

    def retains_history_feature(self, step_id: int) -> bool:
        """Whether frame ``step_id`` can be selected by a future slow window.

        StreamVLN's locked 32/8 profile selects multiples of four.  For a custom
        profile that is not evenly divisible, retain every feature rather than
        risk dropping a future slow-memory candidate.
        """
        if self.window_size % self.num_history:
            return True
        return step_id % (self.window_size // self.num_history) == 0


__all__ = ["StreamVLNBatch", "StreamVLNInputs", "StreamVLNProcessor"]
