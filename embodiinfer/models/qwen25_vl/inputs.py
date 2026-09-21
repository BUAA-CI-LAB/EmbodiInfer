"""Qwen2.5-VL tensor preparation for graph-safe next-token forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .vision import VisionBounds


@dataclass(frozen=True)
class Qwen25VLPreparedNextTokenInputs:
    tensors: tuple[torch.Tensor, ...]
    window_bounds: VisionBounds
    full_bounds: VisionBounds


def prepare_qwen25_vl_next_token_inputs(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    video_grid_thw: torch.Tensor | None = None,
    second_per_grid_ts: object | None = None,
) -> Qwen25VLPreparedNextTokenInputs:
    qwen = model.model
    visual = qwen.visual
    rotary = visual.rot_pos_emb(image_grid_thw)
    window_index, cu_window = visual.get_window_index(image_grid_thw)
    window_index = window_index.to(device=input_ids.device)
    reverse_indices = torch.argsort(window_index)
    cu_window_seqlens = torch.unique_consecutive(
        torch.tensor(cu_window, device=input_ids.device, dtype=torch.int32)
    )
    cu_seqlens = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
    window_values = cu_window_seqlens.detach().cpu().to(torch.int64).tolist()
    full_values = cu_seqlens.detach().cpu().to(torch.int64).tolist()
    window_bounds = tuple(
        (int(start), int(end))
        for start, end in zip(window_values[:-1], window_values[1:], strict=True)
        if end > start
    )
    full_bounds = tuple(
        (int(start), int(end))
        for start, end in zip(full_values[:-1], full_values[1:], strict=True)
        if end > start
    )
    seq_len = pixel_values.shape[0]
    unit = visual.spatial_merge_unit
    rotary = rotary.reshape(seq_len // unit, unit, -1)[window_index]
    rotary = rotary.reshape(seq_len, -1)
    rotary = torch.cat((rotary, rotary), dim=-1)
    rotary_cos, rotary_sin = rotary.cos(), rotary.sin()
    position_ids, _ = qwen.get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
    )
    tensors = (
        input_ids,
        pixel_values,
        attention_mask,
        position_ids,
        window_index,
        reverse_indices,
        cu_window_seqlens,
        cu_seqlens,
        rotary_cos,
        rotary_sin,
    )
    return Qwen25VLPreparedNextTokenInputs(
        tensors=tensors,
        window_bounds=window_bounds,
        full_bounds=full_bounds,
    )


__all__ = [
    "Qwen25VLPreparedNextTokenInputs",
    "prepare_qwen25_vl_next_token_inputs",
]
