"""Numeric RTC invariants that do not require a model checkpoint."""

import torch

from embodiinfer.engine.async_inference.rtc import (
    RTCGuidanceConfig,
    build_rtc_guidance,
    hard_prefix_mask,
    rtc_guided_velocity,
)


def test_disabled_rtc_is_the_plain_velocity() -> None:
    x = torch.tensor([[[1.0], [2.0]]])
    guidance = build_rtc_guidance(
        None,
        inference_delay=1,
        config=RTCGuidanceConfig(execution_horizon=2),
        action_horizon=2,
        action_dim=1,
    )
    plain = 0.5 * x
    actual = rtc_guided_velocity(lambda value: 0.5 * value, x, 0.5, guidance)
    torch.testing.assert_close(actual, plain)


def test_soft_rtc_uses_the_velocity_jacobian() -> None:
    # v(x) = 0.5x, t = 0.5 in the pi0.5 (noise-at-one) convention.  The
    # reference correction differentiates x1_t = x - (1-t)v(x), so its VJP
    # coefficient is 0.75 here.  A detached/error-only implementation would
    # produce a different result and silently lose RTC's model-space guidance.
    x = torch.tensor([[[1.0]]])
    previous = torch.tensor([[2.0]])
    guidance = build_rtc_guidance(
        previous,
        inference_delay=0,
        config=RTCGuidanceConfig(execution_horizon=1, max_guidance_weight=10.0),
        action_horizon=1,
        action_dim=1,
    )
    actual = rtc_guided_velocity(lambda value: 0.5 * value, x, 0.5, guidance)
    # With a one-row execution horizon the linear schedule gives that row a
    # 0.5 prefix weight (the interior of the reference ramp), hence -0.4375.
    torch.testing.assert_close(actual, torch.tensor([[[-0.4375]]]))


def test_hard_prefix_mask_marks_only_actions_consumed_during_delay() -> None:
    torch.testing.assert_close(
        hard_prefix_mask(2, 5).squeeze(-1),
        torch.tensor([True, True, False, False, False]),
    )
