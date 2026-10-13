"""CPU checks for the transport and numeric async-inference contracts."""

import pytest
import torch

from embodiinfer.engine.async_inference import (
    PrefixAttentionSchedule,
    RTCGuidanceConfig,
    build_rtc_guidance,
    parse_async_plan,
    prefix_weights,
)


def test_rtc_prefix_schedule_and_first_request_are_deterministic():
    assert torch.equal(
        prefix_weights(2, 6, 10, PrefixAttentionSchedule.LINEAR),
        torch.tensor([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]),
    )
    guidance = build_rtc_guidance(None, 2, RTCGuidanceConfig(execution_horizon=4), action_horizon=8)
    assert guidance.prev_chunk_left_over is None
    assert not guidance.enabled


def test_rtc_prefix_is_clipped_to_the_current_decode_shape():
    guidance = build_rtc_guidance(
        torch.arange(20, dtype=torch.float32).reshape(4, 5),
        1,
        RTCGuidanceConfig(execution_horizon=9),
        action_horizon=2,
        action_dim=3,
    )
    assert guidance.prev_chunk_left_over.shape == (1, 2, 3)
    assert torch.equal(guidance.prev_chunk_left_over[0], torch.tensor([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]))
    assert guidance.prefix_weights.shape == (2,)


def test_explicit_rtc_prefix_must_declare_model_space():
    with pytest.raises(ValueError, match="model_space=true"):
        parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[1.0]]}}})
    plan = parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[1.0]], "model_space": True}}})
    assert plan is not None and plan.rtc is not None and plan.rtc.model_space


def test_rtc_cache_start_index_is_independent_of_inference_delay():
    plan = parse_async_plan(
        {
            "async": {
                "rtc": {
                    "prev_chunk_left_over": None,
                    "inference_delay": 3,
                    "cache_start_index": 10,
                    "execution_horizon": 40,
                }
            }
        }
    )
    assert plan is not None and plan.rtc is not None
    assert plan.rtc.inference_delay == 3
    assert plan.rtc.cache_start_index == 10

    with pytest.raises(ValueError, match="cache_start_index"):
        parse_async_plan({"async": {"rtc": {"cache_start_index": -1}}})
    with pytest.raises(ValueError, match="cache_start_index"):
        parse_async_plan({"async": {"rtc": {"cache_start_index": True}}})
