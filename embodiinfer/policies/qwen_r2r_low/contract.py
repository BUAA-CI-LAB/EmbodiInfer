"""Low-level navigation constants, states, and action contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ...models.qwen25_vl.history_image_cache import HistoryImageCache, HistoryImageCacheKey
from ...types import Observation
from ..base import PrefixState

LOW_LEVEL_CHECKPOINT = "Vebbern/Qwen2.5-VL-3B-R2R-low-level"

LOW_LEVEL_REVISION = "fb14e7e3672c5da5f4694ac170a54e983a50ccf3"

LOW_LEVEL_SYSTEM_PROMPT_SHA256 = "f5a7f75be433db6c6ead1702baf334ee0f0d81191a9a7a22bf02003fbce52559"

R2R_PREPROCESSOR_SHA256 = "8d55622ce2c010e1991233a47c138fccca18541721fba78dd43fd03180894e36"

LOW_LEVEL_IMAGE_SIZE = (320, 240)


class NavigationOutputError(ValueError):
    """The model emitted text outside the checkpoint's action grammar."""


@dataclass(frozen=True)
class QwenR2RLowMemory:
    frames: tuple[torch.Tensor, ...] = ()
    responses: tuple[str, ...] = ()
    history_image_cache: HistoryImageCache = field(default_factory=HistoryImageCache)

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.responses):
            raise ValueError("low-level frames and responses must stay aligned")
        self.history_image_cache.validate_history(
            len(self.frames),
            HistoryImageCacheKey(
                profile="low_level",
                kind="low",
                size=LOW_LEVEL_IMAGE_SIZE,
                resize=True,
                processor_sha256=R2R_PREPROCESSOR_SHA256,
            ),
        )

    @property
    def seq_len(self) -> int:
        return len(self.frames)

    def to(self, device: torch.device | str) -> QwenR2RLowMemory:
        torch.device(device)
        return self

    def expand(self, num_samples: int) -> QwenR2RLowMemory:
        if type(num_samples) is not int or num_samples != 1:
            raise ValueError("qwen_r2r_low memory only supports B1 expand(1)")
        return self

    def compact(
        self,
        keep: Sequence[int] | torch.Tensor | None = None,
    ) -> QwenR2RLowMemory:
        if keep is None:
            return self
        if isinstance(keep, torch.Tensor):
            if keep.ndim != 1:
                raise ValueError("qwen_r2r_low compact indices must be rank one")
            indices = tuple(int(value) for value in keep.detach().cpu().tolist())
        else:
            indices = tuple(keep)
        if indices != (0,) or any(type(value) is not int for value in indices):
            raise ValueError("qwen_r2r_low memory only supports compact([0])")
        return self


@dataclass
class QwenR2RLowBatch:
    observations: list[Observation]
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return len(self.observations)

    def to(self, device, dtype=None):
        del device, dtype
        return self


@dataclass
class QwenR2RLowPrefix:
    observation: Observation
    memory: QwenR2RLowMemory
    batch_size: int = 1

    def to(self, device):
        del device
        return self

    def expand(self, num_samples: int) -> PrefixState:
        if num_samples != 1:
            raise NotImplementedError("Qwen VLN profiles do not support candidate expansion")
        return self


def parse_low_action(text: str) -> torch.Tensor:
    action_ids = {"stop": 0, "move": 1, "left": 2, "right": 3}
    chosen = text.strip().casefold()
    if chosen not in action_ids:
        raise NavigationOutputError(f"invalid low-level action {text!r}; expected Left, Right, Move, or Stop")
    return torch.tensor([[action_ids[chosen], 0.0]], dtype=torch.float32)


__all__ = [
    "LOW_LEVEL_CHECKPOINT",
    "LOW_LEVEL_REVISION",
    "LOW_LEVEL_SYSTEM_PROMPT_SHA256",
    "R2R_PREPROCESSOR_SHA256",
    "LOW_LEVEL_IMAGE_SIZE",
    "NavigationOutputError",
    "QwenR2RLowMemory",
    "QwenR2RLowBatch",
    "QwenR2RLowPrefix",
    "parse_low_action",
]
