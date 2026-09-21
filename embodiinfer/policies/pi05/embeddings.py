"""VVLA-owned positional math for the LeRobot Pi0.5 embedding path."""

from __future__ import annotations

import math
from typing import Any

import torch


def make_attention_mask(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Allow keys in the same or earlier attention group, excluding padding."""
    groups = torch.cumsum(att_masks, dim=1)
    allowed = groups[:, None, :] <= groups[:, :, None]
    return allowed & (pad_masks[:, None, :] * pad_masks[:, :, None])


def attention_mask_4d(allowed: torch.Tensor) -> torch.Tensor:
    """Use the finite OpenPI mask value, including for fully padded query rows."""
    return torch.where(allowed[:, None, :, :], 0.0, -2.3819763e38)


def time_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> torch.Tensor:
    """Match LeRobot's FP64 sinusoidal arithmetic (FP32 on MPS)."""
    if dimension % 2:
        raise ValueError("Pi0.5 time embedding dimension must be even")
    dtype = torch.float32 if device.type == "mps" else torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    angles = (1.0 / period * 2 * math.pi)[None, :] * time[:, None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def rope_tables(
    rotary: Any, x: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build default RoPE tables from initialized frequencies without HF forward."""
    if getattr(rotary, "rope_type", "default") != "default":
        raise ValueError("Pi05 native embeddings currently support default RoPE only")
    inv = rotary.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    with torch.autocast(device_type=x.device.type, enabled=False):
        freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((freqs, freqs), dim=-1)
        cos = angles.cos() * rotary.attention_scaling
        sin = angles.sin() * rotary.attention_scaling
    return cos.to(x.dtype), sin.to(x.dtype)
