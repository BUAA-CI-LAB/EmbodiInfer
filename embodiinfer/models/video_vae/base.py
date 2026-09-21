"""The video-VAE category: the pixels<->latent tokenizer contract.

A **video VAE** encodes pixel frames into latent frames (and decodes back) — the
Diffusers ``models/autoencoders/`` analogue and the direct analogue of NVIDIA
cosmos-policy's ``VideoTokenizerInterface``. It is a **vendored perception leaf**: it
runs once per observation inside a policy's ``encode_prefix`` (never in the denoise
loop), so — unlike a :class:`~embodiinfer.models.video_dit.base.VideoDiT` — it does not route
attention through :mod:`embodiinfer.layers` and is not a CUDA-graph target. Per Diffusers'
convention, families do NOT share a common ``AutoencoderKL`` superclass; they share
only this thin behavioural ABC. Engine-agnostic (never import ``embodiinfer.policies`` /
``embodiinfer.engine``).

Families: ``wan`` (Wan2.1 causal-3D VAE, used by the Cosmos policy).
"""

from __future__ import annotations

import abc

import torch
from torch import nn


class VideoVAE(nn.Module, abc.ABC):
    """Encode pixels ``[B, 3, T, H, W]`` in ``[-1, 1]`` to latent frames and back."""

    @abc.abstractmethod
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, 3, T, H, W]`` -> latent ``[B, latent_ch, T', H/s, W/s]``."""

    @abc.abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent ``[B, latent_ch, T', H', W']`` -> pixels ``[B, 3, T, H, W]``."""

    @property
    @abc.abstractmethod
    def spatial_compression(self) -> int: ...

    @property
    @abc.abstractmethod
    def temporal_compression(self) -> int: ...

    @property
    @abc.abstractmethod
    def latent_ch(self) -> int: ...
