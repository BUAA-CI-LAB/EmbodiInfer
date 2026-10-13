"""CPU checks for π0.5 RTC model-chunk cache alignment."""

from __future__ import annotations

import pytest
import torch

from embodiinfer.engine.async_inference import parse_async_plan
from embodiinfer.policies.pi05.serving import Pi05ServingAdapter


def _plan(*, delay: int, cache_start_index: int | None):
    rtc = {
        "prev_chunk_left_over": None,
        "inference_delay": delay,
        "execution_horizon": 40,
        "prefix_attention_schedule": "exp",
        "max_guidance_weight": 5.0,
        "hard_prefix": False,
    }
    if cache_start_index is not None:
        rtc["cache_start_index"] = cache_start_index
    return parse_async_plan({"async": {"rtc": rtc}})


def _adapter_with_cache(chunk: torch.Tensor) -> Pi05ServingAdapter:
    adapter = object.__new__(Pi05ServingAdapter)
    adapter._last_model_chunk = {"session": chunk}
    return adapter


def test_session_cache_slices_by_replan_offset_not_delay() -> None:
    cached = torch.arange(100, dtype=torch.float32).reshape(50, 2)
    adapter = _adapter_with_cache(cached)

    resolved = adapter._resolve_rtc_plan(
        _plan(delay=3, cache_start_index=10),
        "session",
    )

    assert resolved is not None and resolved.rtc is not None
    assert resolved.rtc.prev_chunk_left_over == tuple(
        tuple(float(value) for value in row) for row in cached[10:].tolist()
    )
    assert len(resolved.rtc.prev_chunk_left_over) == 40


def test_legacy_session_cache_request_still_falls_back_to_delay() -> None:
    cached = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    adapter = _adapter_with_cache(cached)

    resolved = adapter._resolve_rtc_plan(
        _plan(delay=3, cache_start_index=None),
        "session",
    )

    assert resolved is not None and resolved.rtc is not None
    assert resolved.rtc.prev_chunk_left_over[0] == tuple(float(value) for value in cached[3].tolist())
    assert len(resolved.rtc.prev_chunk_left_over) == 7


def test_explicit_nonzero_cache_offset_requires_session_cache() -> None:
    adapter = _adapter_with_cache(torch.zeros(1, 2))
    adapter._last_model_chunk.clear()

    with pytest.raises(ValueError, match="requires a previous model chunk"):
        adapter._resolve_rtc_plan(
            _plan(delay=4, cache_start_index=10),
            "missing-session",
        )


def test_explicit_cache_offset_must_leave_a_prefix() -> None:
    adapter = _adapter_with_cache(torch.zeros(10, 2))

    with pytest.raises(ValueError, match="smaller than the cached model horizon"):
        adapter._resolve_rtc_plan(
            _plan(delay=4, cache_start_index=10),
            "session",
        )
