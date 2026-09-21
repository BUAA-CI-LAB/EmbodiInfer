"""Cosmos Policy batch construction: raw observations -> the latent-video layout.

Cosmos Policy consumes a *raw image sequence* (VAE-encoded into latent frames), a
proprio vector, and a T5 text embedding — not embodiinfer's tokenized ``Observation``. This
module builds that batch: the LIBERO 33-frame pixel sequence (1 blank + each modality
tiled ``temporal_compression`` times, blanks for the to-generate frames), the proprio
tensor, the per-task T5 embedding, and the fixed latent-frame index layout.

Latent frame layout (LIBERO, ``state_t=9``): 0 blank, 1 current-proprio, 2 current-wrist,
3 current-primary, 4 action, 5 future-proprio, 6 future-wrist, 7 future-primary, 8 value.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import torch

TEMPORAL_COMPRESSION = 4
IMAGE_SIZE = 224


# LIBERO latent-frame index layout.
LIBERO_LATENT_IDX = {
    "current_proprio": 1,
    "current_wrist_image": 2,
    "current_image": 3,
    "action": 4,
    "future_proprio": 5,
    "future_wrist_image": 6,
    "future_image": 7,
    "value": 8,
}
LIBERO_STATE_T = 9
LIBERO_NUM_CONDITIONAL_FRAMES = 4


@dataclass
class CosmosBatch:
    """A collated Cosmos batch: the raw pixel video, proprio, text embedding, and layout.

    ``pixel_video`` is ``[B, 3, T_pix, H, W]`` in ``[-1, 1]`` (the VAE input);
    ``proprio`` ``[B, proprio_dim]``; ``crossattn`` ``[B, N_text, 1024]`` (the T5 embedding).
    ``latent_idx`` maps modality -> latent frame index; ``num_conditional_frames`` marks the
    leading conditioning frames.
    """

    pixel_video: torch.Tensor
    proprio: torch.Tensor
    crossattn: torch.Tensor
    padding_mask: torch.Tensor
    request_ids: list[str]
    latent_idx: dict[str, int] = field(default_factory=lambda: dict(LIBERO_LATENT_IDX))
    state_t: int = LIBERO_STATE_T
    num_conditional_frames: int = LIBERO_NUM_CONDITIONAL_FRAMES

    @property
    def batch_size(self) -> int:
        return self.pixel_video.shape[0]

    def to(self, device, dtype=None) -> CosmosBatch:
        pv = self.pixel_video.to(device)
        pr = self.proprio.to(device)
        ca = self.crossattn.to(device)
        pm = self.padding_mask.to(device)
        if dtype is not None:
            pv, ca, pm = pv.to(dtype), ca.to(dtype), pm.to(dtype)
        return CosmosBatch(
            pixel_video=pv,
            proprio=pr,
            crossattn=ca,
            padding_mask=pm,
            request_ids=self.request_ids,
            latent_idx=self.latent_idx,
            state_t=self.state_t,
            num_conditional_frames=self.num_conditional_frames,
        )


def _tile(img: torch.Tensor, n: int) -> torch.Tensor:
    """Tile a single ``[3,H,W]`` frame ``n`` times along a new time axis -> ``[3,n,H,W]``."""
    return img.unsqueeze(1).repeat(1, n, 1, 1)


def build_libero_pixel_video(primary_image: torch.Tensor, wrist_image: torch.Tensor) -> torch.Tensor:
    """Build the LIBERO 33-frame pixel sequence from the current images (each ``[3,224,224]``, [-1,1]).

    Sequence (matching ``get_action``): 1 blank + proprio-blank×4 + wrist×4 + primary×4 +
    action-blank×4 + future-proprio-blank×4 + future-wrist×4 + future-primary×4 + value-blank×4
    = 33 frames -> 9 latent frames.
    """
    blank = torch.zeros_like(primary_image)
    n = TEMPORAL_COMPRESSION
    seq = [
        blank.unsqueeze(1),  # 0: single blank placeholder
        _tile(blank, n),  # 1: current proprio (blank; injected in latent)
        _tile(wrist_image, n),  # 2: current wrist
        _tile(primary_image, n),  # 3: current primary
        _tile(blank, n),  # 4: action (blank)
        _tile(blank, n),  # 5: future proprio (blank)
        _tile(wrist_image, n),  # 6: future wrist (copy of current)
        _tile(primary_image, n),  # 7: future primary (copy of current)
        _tile(blank, n),  # 8: value (blank)
    ]
    return torch.cat(seq, dim=1)  # [3, 33, H, W]


class CosmosProcessor:
    """Builds :class:`CosmosBatch` from raw current images + proprio + text embedding.

    ``t5_embeddings`` is the per-task cache (dict: instruction string -> ``[1, N, 1024]``);
    supply a task's embedding directly for a single build. Image inputs are current
    ``primary_image`` / ``wrist_image`` as ``[3, 224, 224]`` in ``[-1, 1]``.
    """

    def build(
        self,
        primary_images: list[torch.Tensor],
        wrist_images: list[torch.Tensor],
        proprios: torch.Tensor,
        crossattn: torch.Tensor,
        request_ids: list[str] | None = None,
    ) -> CosmosBatch:
        B = len(primary_images)
        videos = [build_libero_pixel_video(primary_images[i], wrist_images[i]) for i in range(B)]
        pixel_video = torch.stack(videos, dim=0)  # [B, 3, 33, 224, 224]
        padding_mask = torch.zeros((B, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=pixel_video.dtype)
        if request_ids is None:
            request_ids = [f"c{i}" for i in range(B)]
        return CosmosBatch(
            pixel_video=pixel_video,
            proprio=proprios,
            crossattn=crossattn,
            padding_mask=padding_mask,
            request_ids=request_ids,
        )


class CosmosTextEmbedder:
    """Instruction string -> T5 cross-attn embedding ``[B, max_length, 1024]``.

    Serves LIBERO tasks from the precomputed ``t5_embeddings.pkl`` (dict: instruction ->
    ``[1, 512, 1024]``) and lazily loads the :class:`~embodiinfer.models.text_encoders.T5TextEncoder`
    only on a cache miss (an instruction not in the pkl), caching the result.
    """

    def __init__(
        self,
        t5_embeddings_pkl: str | None = None,
        model_name: str = "google-t5/t5-11b",
        device: str = "cuda",
        dtype=torch.bfloat16,
        max_length: int = 512,
    ):
        self.max_length = max_length
        self._cache: dict[str, torch.Tensor] = {}
        if t5_embeddings_pkl:
            with open(t5_embeddings_pkl, "rb") as f:
                self._cache = pickle.load(f)
        self._encoder = None
        self._model_name, self._device, self._dtype = model_name, device, dtype

    def embed(self, instructions: list[str]) -> torch.Tensor:
        embs = []
        for ins in instructions:
            if ins not in self._cache:
                if self._encoder is None:
                    from ...models.text_encoders import T5TextEncoder

                    self._encoder = T5TextEncoder(self._model_name, self._device, self._dtype)
                self._cache[ins] = self._encoder.encode(ins, self.max_length).cpu()
            embs.append(self._cache[ins])
        return torch.cat(embs, dim=0)  # [B, max_length, 1024]


def rescale_proprio(proprio: torch.Tensor, stats: dict) -> torch.Tensor:
    """Normalize proprio to ``[-1, 1]``: ``2*(x-min)/(max-min)-1`` (cosmos ``rescale_proprio``)."""
    pmin = torch.as_tensor(stats["proprio_min"], dtype=proprio.dtype, device=proprio.device)
    pmax = torch.as_tensor(stats["proprio_max"], dtype=proprio.dtype, device=proprio.device)
    return 2 * (proprio - pmin) / (pmax - pmin) - 1


def unnormalize_actions(actions: torch.Tensor, stats: dict) -> torch.Tensor:
    """Un-normalize actions from ``[-1, 1]`` to dataset scale (cosmos ``unnormalize_actions``)."""
    amin = torch.as_tensor(stats["actions_min"], dtype=actions.dtype, device=actions.device)
    amax = torch.as_tensor(stats["actions_max"], dtype=actions.dtype, device=actions.device)
    return 0.5 * (actions + 1) * (amax - amin) + amin
