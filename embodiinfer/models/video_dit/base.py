"""The video-DiT category: base contract + a lightweight family registry.

A **video diffusion transformer** denoises latent video frames conditioned on a
per-frame timestep (noise level) and a cross-attention context. This is a reusable
*model family* category (the HuggingFace Diffusers ``models/transformers/`` analogue) —
each family is one module + one class + one config dataclass, distinguished by its
**architecture lineage**, not by any policy that uses it:

  * ``cosmos_predict2`` — NVIDIA Cosmos-Predict2 (``MiniTrainDIT``): AdaLN-LoRA
    modulation (low-rank, separate self/cross/mlp), head-dim QK-norm, GELU MLP.
  * ``wan`` (future) — Alibaba Wan2.1 (``WanModel``): AdaLN-single (one learned
    ``[1, 6, dim]`` table), full-dim QK-norm, T2V/I2V cross-attention variants.

A new family adds its own ``embodiinfer/models/video_dit/<family>.py`` and
``@register_video_dit("<family>")``. The forward routes attention through
:mod:`embodiinfer.layers.attention` so the engine has a single CUDA-graph seam. Engine-agnostic:
a ``VideoDiT`` never imports ``embodiinfer.policies`` / ``embodiinfer.engine`` and knows nothing about
``Observation`` / ``ActionChunk`` — a policy composes it.
"""

from __future__ import annotations

import abc

import torch
from torch import nn


class VideoDiT(nn.Module, abc.ABC):
    """Denoise latent video frames. The one method the engine/policy depends on.

    ``forward(latent, timesteps, context, cond_mask, padding_mask) -> denoised`` where
    ``latent`` is ``[B, C, T', H', W']``, ``timesteps`` is per-frame ``[B, T']`` (the
    preconditioned noise level), ``context`` is the cross-attention text embedding
    ``[B, N, D_ctx]``, ``cond_mask`` marks conditioning frames (frame-replace), and
    ``padding_mask`` is the spatial validity mask. Concrete families keep their own
    descriptive parameter names."""

    @abc.abstractmethod
    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


_REGISTRY: dict[str, type[VideoDiT]] = {}


def register_video_dit(name: str):
    """Register a video-DiT family class under ``name`` (e.g. ``"cosmos_predict2"``)."""

    def deco(cls: type[VideoDiT]) -> type[VideoDiT]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get_video_dit(name: str) -> type[VideoDiT]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown video-DiT family '{name}'. Registered: {available_video_dit()}")
    return _REGISTRY[name]


def available_video_dit() -> list[str]:
    return sorted(_REGISTRY)
