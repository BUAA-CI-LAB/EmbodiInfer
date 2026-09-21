"""Native Qwen2.5-VL-3B R2R panoramic navigation policy."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ...models.qwen25_vl.history_image_cache import HistoryImageCache, HistoryImageCacheKey
from ...types import Observation
from ..base import PrefixState

PANORAMIC_CHECKPOINT = "Vebbern/Qwen2.5-VL-3B-R2R-panoramic"
PANORAMIC_REVISION = "f7f4d90aa17cb5168d38981b641066d113a826f8"
PANORAMIC_SYSTEM_PROMPT_SHA256 = "5fe1a4bc58b4d124f15c4b376bfa2374681a10259619abd221926eb74c3a905e"
R2R_PREPROCESSOR_SHA256 = "8d55622ce2c010e1991233a47c138fccca18541721fba78dd43fd03180894e36"
PANORAMIC_IMAGE_SIZE = (960, 240)
CANDIDATE_IMAGE_SIZE = (320, 240)


class NavigationOutputError(ValueError):
    """The model emitted text outside the checkpoint's action grammar."""


@dataclass(frozen=True)
class QwenR2RPanoramicMemory:
    frames: tuple[torch.Tensor, ...] = ()
    responses: tuple[str, ...] = ()
    history_image_cache: HistoryImageCache = field(default_factory=HistoryImageCache)

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.responses):
            raise ValueError("panoramic frames and responses must stay aligned")
        self.history_image_cache.validate_history(
            len(self.frames),
            HistoryImageCacheKey(
                profile="panoramic",
                kind="panorama",
                size=PANORAMIC_IMAGE_SIZE,
                resize=False,
                processor_sha256=R2R_PREPROCESSOR_SHA256,
            ),
        )

    @property
    def seq_len(self) -> int:
        return len(self.frames)

    def to(self, device: torch.device | str) -> QwenR2RPanoramicMemory:
        torch.device(device)
        return self

    def expand(self, num_samples: int) -> QwenR2RPanoramicMemory:
        if type(num_samples) is not int or num_samples != 1:
            raise ValueError("qwen_r2r_panoramic memory only supports B1 expand(1)")
        return self

    def compact(
        self,
        keep: Sequence[int] | torch.Tensor | None = None,
    ) -> QwenR2RPanoramicMemory:
        if keep is None:
            return self
        if isinstance(keep, torch.Tensor):
            if keep.ndim != 1:
                raise ValueError("qwen_r2r_panoramic compact indices must be rank one")
            indices = tuple(int(value) for value in keep.detach().cpu().tolist())
        else:
            indices = tuple(keep)
        if indices != (0,) or any(type(value) is not int for value in indices):
            raise ValueError("qwen_r2r_panoramic memory only supports compact([0])")
        return self


@dataclass
class QwenR2RPanoramicBatch:
    observations: list[Observation]
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return len(self.observations)

    def to(self, device, dtype=None):
        del device, dtype
        return self


@dataclass
class QwenR2RPanoramicPrefix:
    observation: Observation
    memory: QwenR2RPanoramicMemory
    batch_size: int = 1

    def to(self, device):
        del device
        return self

    def expand(self, num_samples: int) -> PrefixState:
        if num_samples != 1:
            raise NotImplementedError("Qwen R2R panoramic does not support candidate expansion")
        return self


def parse_panoramic_action(text: str, candidate_count: int) -> torch.Tensor:
    chosen = text.strip()
    if chosen.casefold() == "stop":
        return torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
    if re.fullmatch(r"\d+", chosen) is None:
        raise NavigationOutputError(f"invalid panoramic action {text!r}; expected Stop or a candidate index")
    candidate = int(chosen)
    if not 0 <= candidate < candidate_count:
        raise NavigationOutputError(f"candidate index {candidate} outside [0, {candidate_count})")
    return torch.tensor([[float(candidate), 0.0]], dtype=torch.float32)
