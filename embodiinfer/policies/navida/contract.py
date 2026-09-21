"""NaViDA constants, recurrent state, history selection, and action contract."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from ...types import Observation
from ..base import PrefixState

NAVIDA_SOURCE_IMAGE_SIZE = (320, 240)
NAVIDA_MODEL_IMAGE_SIZE = (308, 252)
NAVIDA_MAX_STORED_FRAMES = 200
NAVIDA_HISTORY_FRAMES = 8
NAVIDA_SYSTEM_PROMPT = "You are a helpful assistant."
NAVIDA_SOURCE_REVISION = "b86316a78a2865c80b445ed5a8fcc08fc0243dfb"
NAVIDA_CHECKPOINT_REVISION = "0ccd8b6b3cd0826c5dd67374415ceedf51adb9e2"
NAVIDA_MAX_ATOMIC_ACTIONS = 6


@dataclass(frozen=True)
class NaViDAMemory:
    frames: tuple[torch.Tensor, ...] = ()
    responses: tuple[str, ...] = ()

    @property
    def seq_len(self) -> int:
        return len(self.frames)

    def to(self, device: torch.device | str) -> NaViDAMemory:
        del device
        return self


@dataclass
class NaViDABatch:
    observations: list[Observation]
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return len(self.observations)

    def to(self, device, dtype=None):
        del device, dtype
        return self


@dataclass
class NaViDAPrefix:
    observation: Observation
    memory: NaViDAMemory
    batch_size: int = 1

    def to(self, device):
        del device
        return self

    def expand(self, num_samples: int) -> PrefixState:
        if num_samples != 1:
            raise NotImplementedError("NaViDA does not support candidate expansion")
        return self


def parse_navida_actions(
    text: str,
    execute_chunks: int = 2,
    *,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Expand the official NaViDA v2 two-chunk response into atomic actions."""
    chooser = rng if rng is not None else random
    rows: list[tuple[float, float]] = []
    for raw_chunk in text.split(", ")[:execute_chunks]:
        tagged = re.search(r"<answer>(.*?)</answer>", raw_chunk, flags=re.IGNORECASE | re.DOTALL)
        chunk = (tagged.group(1).strip() if tagged is not None else raw_chunk.strip()).casefold()
        number = re.search(r"-?\d+", chunk)
        numeric = float(number.group()) if number is not None else None
        if "stop" in chunk:
            action_id, count = 0, 1
        elif "forward" in chunk:
            value = 25.0 if numeric is None else numeric
            action_id, count = 1, min(3, round(value / 25.0))
        elif "left" in chunk:
            value = 15.0 if numeric is None else numeric
            action_id, count = 2, min(3, round(value / 15.0))
        elif "right" in chunk:
            value = 15.0 if numeric is None else numeric
            action_id, count = 3, min(3, round(value / 15.0))
        else:
            action_id, count = None, 0
        if action_id is None or count <= 0:
            action_id, count = chooser.randint(1, 3), 1
        value = 0.25 if action_id == 1 else (15.0 if action_id in (2, 3) else 0.0)
        rows.extend([(float(action_id), value)] * count)
    return torch.tensor(rows, dtype=torch.float32)


def select_navida_history(
    previous_frames: tuple[torch.Tensor, ...], current_frame: torch.Tensor
) -> list[torch.Tensor]:
    if not previous_frames:
        return [current_frame]
    if len(previous_frames) <= NAVIDA_HISTORY_FRAMES:
        return list(previous_frames)
    last = len(previous_frames) - 1
    indices = [round(index * last / (NAVIDA_HISTORY_FRAMES - 1)) for index in range(NAVIDA_HISTORY_FRAMES)]
    return [previous_frames[index] for index in indices]
