"""Semantics-preserving text-shape buckets for Qwen2.5-VL compilation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

QWEN25_VL_TEXT_BUCKET_SCHEMA = "qwen25_vl_left_masked_text_v3"


@dataclass(frozen=True)
class Qwen25VLTextBucket:
    source_length: int
    target_length: int
    left_padding: int

    @property
    def bucket_id(self) -> tuple[str, int]:
        return QWEN25_VL_TEXT_BUCKET_SCHEMA, self.target_length

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": QWEN25_VL_TEXT_BUCKET_SCHEMA,
            "source_length": self.source_length,
            "target_length": self.target_length,
            "left_padding": self.left_padding,
        }


def normalize_qwen25_vl_text_buckets(values: tuple[int, ...]) -> tuple[int, ...]:
    buckets = tuple(values)
    if any(type(value) is not int or value <= 0 for value in buckets):
        raise ValueError("compile_text_buckets must contain positive integers")
    if buckets != tuple(sorted(set(buckets))):
        raise ValueError("compile_text_buckets must be strictly increasing and unique")
    return buckets


def qwen25_vl_bucket_schema(profile: str, buckets: tuple[int, ...]) -> dict[str, object]:
    return {
        "schema": QWEN25_VL_TEXT_BUCKET_SCHEMA,
        "profile": profile,
        "text_buckets": list(buckets),
        "text_padding": "left_attention_mask_zero",
        "truncation": False,
        "pixel_values": "exact",
        "image_grid_thw": "exact",
        "history_count": "exact",
        "candidate_count": "exact",
        "batch": "exact",
    }


def qwen25_vl_text_bucket_key(encoded: dict[str, torch.Tensor], buckets: tuple[int, ...]) -> tuple[str, int]:
    sequence_length = int(encoded["input_ids"].shape[1])
    if buckets:
        return QWEN25_VL_TEXT_BUCKET_SCHEMA, sequence_length
    return "qwen25_vl_exact_text_shape_v1", sequence_length


def apply_qwen25_vl_text_bucket(
    encoded: dict[str, torch.Tensor],
    *,
    pad_token_id: int | None,
    buckets: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], Qwen25VLTextBucket]:
    """Left-pad only text tensors; visual tensors and official prompt stay exact."""
    normalized = normalize_qwen25_vl_text_buckets(buckets)
    if normalized and pad_token_id is None:
        raise ValueError("processor.tokenizer.pad_token_id is required for compile text buckets")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be aligned rank-2 tensors")
    source_length = int(input_ids.shape[1])
    if normalized:
        target_length = next((bucket for bucket in normalized if bucket >= source_length), None)
        if target_length is None:
            raise ValueError(
                f"Qwen2.5-VL prompt length {source_length} exceeds the largest "
                f"compile text bucket {normalized[-1]}; truncation is forbidden"
            )
    else:
        target_length = source_length
    left_padding = target_length - source_length
    bucket = Qwen25VLTextBucket(source_length, target_length, left_padding)
    if left_padding == 0:
        return encoded, bucket
    batch_size = int(input_ids.shape[0])
    padded_ids = torch.full(
        (batch_size, left_padding),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    padded_mask = torch.zeros(
        (batch_size, left_padding),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    result = dict(encoded)
    result["input_ids"] = torch.cat((padded_ids, input_ids), dim=1)
    result["attention_mask"] = torch.cat((padded_mask, attention_mask), dim=1)
    return result, bucket


__all__ = [
    "QWEN25_VL_TEXT_BUCKET_SCHEMA",
    "Qwen25VLTextBucket",
    "apply_qwen25_vl_text_bucket",
    "normalize_qwen25_vl_text_buckets",
    "qwen25_vl_bucket_schema",
    "qwen25_vl_text_bucket_key",
]
