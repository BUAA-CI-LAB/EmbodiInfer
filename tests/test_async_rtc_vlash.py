"""RTC and VLASH asynchronous-inference algorithms.

The numeric core is pinned three ways:

* against the **reference implementations** — the prefix-weight tables and the
  guidance constants reproduce Physical Intelligence's ``real-time-chunking-kinetix``
  and its LeRobot port exactly;
* against **closed-form answers** — a linear velocity field has an analytic
  Jacobian-vector product, so the soft-guidance correction is checked as a formula
  rather than against another implementation of itself;
* against **shared golden vectors** (``fixtures/async_inference_golden.json``),
  which EmbodiRun asserts with its own execution-side implementation. Both halves
  of the async loop therefore agree on the same arithmetic, across a process and a
  repository boundary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from embodiinfer import EngineConfig, Observation, preset_config
from embodiinfer.engine.async_inference import (
    ASYNC_SCHEMA,
    ActionSpaceSemantics,
    PrefixAttentionSchedule,
    RTCGuidance,
    RTCGuidanceConfig,
    action_ness,
    apply_request_async_state,
    apply_vlash_state_plan,
    batch_rtc_guidance,
    build_rtc_guidance,
    clamp_prefix,
    flow_noise_end,
    guidance_strength,
    hard_prefix_mask,
    parse_async_plan,
    prefix_weights,
    roll_state_forward,
    roll_state_forward_projected,
    rtc_guided_velocity,
)
from embodiinfer.engine.async_inference.vlash import VlashStatePlan
from embodiinfer.engine.core import EngineCore
from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest
from embodiinfer.exceptions import UnsupportedAsyncGuidanceError, UnsupportedRecurrentModeError
from embodiinfer.policies.decoder import AutoregressiveDecoder, FlowDecoder
from embodiinfer.policies.factory import make_policy
from embodiinfer.types import collate

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "async_inference_golden.json"


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def _obs(cfg, env_id=0):
    return Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
        env_id=env_id,
    )


# ============================================================================
# RTC — prefix attention schedules
# ============================================================================
def test_prefix_weights_match_reference_table():
    """The documented reference example: start=2, end=6, total=10 -> 1 1 4/5 3/5 2/5 1/5 0 0 0 0."""
    weights = prefix_weights(2, 6, 10, PrefixAttentionSchedule.LINEAR)
    assert [round(float(value), 6) for value in weights] == [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0]


def test_prefix_weights_golden_vectors():
    """Every schedule agrees with the shared golden table, bit for bit."""
    for case in _golden()["rtc_prefix_weights"]:
        weights = prefix_weights(
            case["start"], case["end"], case["total"], PrefixAttentionSchedule(case["schedule"])
        )
        assert weights.shape == (case["total"],)
        assert torch.equal(weights, torch.tensor(case["expected"], dtype=torch.float32)), case["name"]


def test_prefix_weights_start_is_pushed_down_to_end():
    """`end` takes precedence: a delay beyond the horizon cannot hold more than it."""
    assert torch.equal(
        prefix_weights(9, 4, 10, PrefixAttentionSchedule.ZEROS),
        prefix_weights(4, 4, 10, PrefixAttentionSchedule.ZEROS),
    )


def test_prefix_weights_ones_and_zeros_shapes():
    ones = prefix_weights(0, 4, 6, PrefixAttentionSchedule.ONES)
    assert torch.equal(ones, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))
    zeros = prefix_weights(3, 8, 6, PrefixAttentionSchedule.ZEROS)
    assert torch.equal(zeros, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_prefix_weights_rejects_bad_arguments():
    with pytest.raises(ValueError):
        prefix_weights(0, 1, 0)
    with pytest.raises(ValueError):
        prefix_weights(-1, 1, 4)


# ============================================================================
# RTC — guidance coefficient
# ============================================================================
def test_guidance_strength_saturates_at_noise_end():
    """s -> 0 (pure noise) is singular; the clamp is what makes it usable."""
    assert float(guidance_strength(0.0, 10.0)) == pytest.approx(10.0)
    assert float(guidance_strength(1e-12, 10.0)) == pytest.approx(10.0)


def test_guidance_strength_releases_at_action_end():
    """s -> 1 is a 0 * inf NaN in the closed form; the reference maps it to 0."""
    assert float(guidance_strength(1.0, 10.0)) == 0.0


def test_guidance_strength_matches_closed_form_off_singularities():
    for s in (0.1, 0.25, 0.5, 0.75, 0.9):
        expected = ((1 - s) / s) * ((1 - s) ** 2 + s**2) / (1 - s) ** 2
        assert float(guidance_strength(s, 1e9)) == pytest.approx(expected, rel=1e-5)


def test_guidance_strength_is_symmetric_about_the_midpoint():
    """The closed form is symmetric about s = 1/2, so it is NOT monotone.

    ``strength(s) = ((1-s)^2 + s^2) / (s (1-s))`` peaks at both ends of the
    interval. The two ends are then treated asymmetrically on purpose: at
    ``s -> 0`` it saturates at the clamp, while at ``s -> 1`` the ``0 * inf``
    limit is mapped to 0 so the final clean steps are released.
    """
    for s in (0.05, 0.2, 0.35, 0.5):
        assert float(guidance_strength(s, 1e9)) == pytest.approx(float(guidance_strength(1 - s, 1e9)))


def test_guidance_strength_is_minimal_at_the_midpoint():
    values = [float(guidance_strength(s, 1e9)) for s in (0.05, 0.2, 0.5, 0.8, 0.95)]
    assert values[2] == min(values)
    assert values[2] == pytest.approx(2.0)


# ============================================================================
# RTC — guidance construction (padding / clamping rules)
# ============================================================================
def test_build_guidance_without_previous_chunk_is_disabled():
    config = RTCGuidanceConfig(execution_horizon=4)
    guidance = build_rtc_guidance(None, 3, config, action_horizon=8)
    assert not guidance.enabled
    assert guidance.prev_chunk_left_over is None


def test_build_guidance_right_pads_short_leftover():
    config = RTCGuidanceConfig(execution_horizon=6)
    leftover = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    guidance = build_rtc_guidance(leftover, inference_delay=2, config=config, action_horizon=5)
    assert guidance.prev_chunk_left_over.shape == (1, 5, 2)
    assert torch.equal(guidance.prev_chunk_left_over[0, :3], leftover)
    assert torch.equal(guidance.prev_chunk_left_over[0, 3:], torch.zeros(2, 2))


def test_build_guidance_execution_horizon_clamp_is_a_noop_guard():
    """Clamping the horizon to the leftover length cannot change the weights.

    The reference weight table already treats ``end > total`` as ``end = total``
    (its trailing-zero count clamps at zero), so the explicit clamp in
    :func:`build_rtc_guidance` is a defensive guard rather than a semantic rule.
    Asserting the equivalence stops a future rewrite of the table from silently
    making the clamped and unclamped paths disagree.
    """
    leftover = torch.ones(3, 2)
    for schedule in PrefixAttentionSchedule:
        for delay in (0, 1, 3, 7):
            clamped = build_rtc_guidance(
                leftover, delay, RTCGuidanceConfig(execution_horizon=3, prefix_attention_schedule=schedule)
            )
            unclamped = build_rtc_guidance(
                leftover, delay, RTCGuidanceConfig(execution_horizon=99, prefix_attention_schedule=schedule)
            )
            assert torch.equal(clamped.prefix_weights, unclamped.prefix_weights), (schedule, delay)


def test_build_guidance_batches_independent_leftovers():
    config = RTCGuidanceConfig(execution_horizon=4)
    leftover = torch.stack([torch.full((6, 2), 1.0), torch.full((6, 2), 2.0)])
    guidance = build_rtc_guidance(leftover, inference_delay=2, config=config)
    assert guidance.prev_chunk_left_over.shape == (2, 6, 2)
    assert torch.equal(guidance.prev_chunk_left_over[0], torch.full((6, 2), 1.0))
    assert torch.equal(guidance.prev_chunk_left_over[1], torch.full((6, 2), 2.0))


def test_build_guidance_rejects_negative_delay():
    with pytest.raises(ValueError):
        build_rtc_guidance(torch.ones(4, 2), -1, RTCGuidanceConfig(execution_horizon=2))


# ============================================================================
# RTC — the guided step, checked against a closed-form Jacobian-vector product
# ============================================================================
def _linear_denoiser(weight: torch.Tensor, bias: torch.Tensor):
    """A velocity field affine in x: v(x) = W x + b. Its VJP is analytic."""

    def denoise(x: torch.Tensor) -> torch.Tensor:
        return x @ weight.T + bias

    return denoise


def _expected_guided_velocity(x, weight, bias, prev, weights, time, noise_at, clamp):
    """Closed form of one guided step for the affine denoiser."""
    identity = torch.eye(weight.shape[0])
    base = x @ weight.T + bias
    s = (1 - time) if noise_at == 1.0 else time
    v_s = -base if noise_at == 1.0 else base
    x1 = x + (1 - s) * v_s
    error = (prev - x1) * weights.reshape(1, -1, 1)
    # d x1 / d x  =  I - time * W  in the pi0.5 direction, I + (1-time) * W otherwise
    jacobian = (identity - time * weight) if noise_at == 1.0 else (identity + (1 - time) * weight)
    correction = error @ jacobian
    strength = float(guidance_strength(math.nan if s == 0 else s, clamp))
    return base - strength * correction


def test_guided_velocity_matches_closed_form_true_vjp():
    """The correction is a real VJP through the velocity field, not the residual.

    This is the point where the LeRobot PyTorch port deviates from Physical
    Intelligence's original: it evaluates ``v_t`` before marking ``x_t`` as
    requiring grad, which collapses ``d x1/d x`` to the identity. Asserting the
    analytic Jacobian pins the corrected behaviour.
    """
    torch.manual_seed(0)
    horizon, width = 4, 3
    weight = torch.randn(width, width) * 0.3
    bias = torch.randn(width) * 0.1
    x = torch.randn(2, horizon, width)
    prev = torch.randn(2, horizon, width)
    config = RTCGuidanceConfig(execution_horizon=4, prefix_attention_schedule=PrefixAttentionSchedule.ZEROS)
    guidance = build_rtc_guidance(prev, inference_delay=2, config=config)

    time = 0.4
    got = rtc_guided_velocity(_linear_denoiser(weight, bias), x, time, guidance, noise_at=1.0)
    want = _expected_guided_velocity(
        x, weight, bias, prev, guidance.prefix_weights, time, 1.0, guidance.max_guidance_weight
    )
    assert torch.allclose(got, want, atol=1e-5)


def test_guided_velocity_differs_from_residual_only_correction():
    """Guards the fidelity note: the degenerate `correction = err` is NOT equivalent."""
    torch.manual_seed(1)
    width = 3
    weight = torch.randn(width, width)
    bias = torch.zeros(width)
    x = torch.randn(1, 4, width)
    prev = torch.zeros(1, 4, width)
    guidance = build_rtc_guidance(
        prev,
        2,
        RTCGuidanceConfig(execution_horizon=4, prefix_attention_schedule=PrefixAttentionSchedule.ZEROS),
    )
    guided = rtc_guided_velocity(_linear_denoiser(weight, bias), x, 0.4, guidance, noise_at=1.0)

    base = _linear_denoiser(weight, bias)(x)
    s = 1 - 0.4
    x1 = x - 0.4 * base
    residual = (prev - x1) * guidance.prefix_weights.reshape(1, -1, 1)
    degenerate = base - float(guidance_strength(s, guidance.max_guidance_weight)) * residual
    assert not torch.allclose(guided, degenerate, atol=1e-4)


def test_guided_velocity_honours_both_flow_directions():
    """The same field conditioned in either direction uses its own clean estimate."""
    torch.manual_seed(2)
    width = 3
    weight = torch.randn(width, width) * 0.2
    bias = torch.randn(width) * 0.1
    x = torch.randn(1, 4, width)
    prev = torch.randn(1, 4, width)
    guidance = build_rtc_guidance(prev, 2, RTCGuidanceConfig(execution_horizon=4))

    for noise_at in (1.0, 0.0):
        time = 0.5
        got = rtc_guided_velocity(_linear_denoiser(weight, bias), x, time, guidance, noise_at=noise_at)
        want = _expected_guided_velocity(
            x, weight, bias, prev, guidance.prefix_weights, time, noise_at, guidance.max_guidance_weight
        )
        assert torch.allclose(got, want, atol=1e-5), noise_at


def test_guided_velocity_without_prefix_is_the_plain_velocity():
    """First inference of an episode must be untouched by RTC."""
    denoise = _linear_denoiser(torch.eye(3), torch.ones(3))
    x = torch.randn(1, 4, 3)
    guidance = build_rtc_guidance(None, 2, RTCGuidanceConfig(execution_horizon=4), action_horizon=4)
    assert torch.equal(rtc_guided_velocity(denoise, x, 0.4, guidance), denoise(x))


def test_guided_velocity_rejects_shape_mismatch():
    guidance = build_rtc_guidance(torch.ones(1, 4, 3), 2, RTCGuidanceConfig(execution_horizon=4))
    with pytest.raises(ValueError, match="must match the decode state"):
        rtc_guided_velocity(lambda x: x, torch.ones(1, 5, 3), 0.5, guidance)


def test_guided_velocity_preserves_dtype():
    x = torch.randn(1, 4, 3, dtype=torch.float64)
    guidance = build_rtc_guidance(torch.ones(1, 4, 3), 2, RTCGuidanceConfig(execution_horizon=4))
    out = rtc_guided_velocity(lambda value: value * 0.5, x, 0.4, guidance)
    assert out.dtype == torch.float64


# ============================================================================
# RTC — flow direction and hard-prefix helpers
# ============================================================================
def test_flow_noise_end_reads_the_schedule_direction():
    assert flow_noise_end([(1.0, -0.1), (0.9, -0.1)]) == 1.0  # pi0.5 / openpi
    assert flow_noise_end([(0.0, 0.1), (0.1, 0.1)]) == 0.0  # GR00T / mock
    with pytest.raises(ValueError):
        flow_noise_end([])


def test_action_ness_normalises_both_directions():
    assert float(action_ness(1.0, 1.0)) == 0.0  # noise
    assert float(action_ness(0.0, 1.0)) == 1.0  # action
    assert float(action_ness(0.0, 0.0)) == 0.0  # noise
    assert float(action_ness(1.0, 0.0)) == 1.0  # action


def test_hard_prefix_mask_and_clamp():
    mask = hard_prefix_mask(2, 5)
    assert torch.equal(mask.flatten(), torch.tensor([True, True, False, False, False]))
    x = torch.zeros(1, 5, 2)
    prefix = torch.ones(1, 5, 2) * 7
    out = clamp_prefix(x, prefix, mask)
    assert torch.equal(out[0, :2], torch.full((2, 2), 7.0))
    assert torch.equal(out[0, 2:], torch.zeros(3, 2))
    with pytest.raises(ValueError):
        hard_prefix_mask(0, 0)


# ============================================================================
# RTC — decoder capability declaration
# ============================================================================
def test_flow_decoder_declares_rtc_support():
    policy = make_policy("mock_flow_vla", preset="tiny")
    assert isinstance(policy.decoder, FlowDecoder)
    assert policy.decoder.supports_rtc_guidance is True


def test_non_flow_decoder_rejects_guidance():
    """A decoder without a denoising loop must fail loudly rather than skip the prefix."""

    class _Stub(AutoregressiveDecoder):
        def decode(self, *args, **kwargs):  # pragma: no cover - never reached
            raise AssertionError("decode must not run when guidance is rejected")

        def produce_chunk(self, state, prefix, num_steps, bucket, graphs, *, guidance=None):
            self.reject_rtc_guidance(guidance)
            return torch.zeros(1, 1, 1)

    stub = _Stub()
    assert stub.supports_rtc_guidance is False
    guidance = build_rtc_guidance(torch.ones(1, 4, 3), 2, RTCGuidanceConfig(execution_horizon=4))
    with pytest.raises(UnsupportedAsyncGuidanceError):
        stub.produce_chunk(None, None, 1, 1, None, guidance=guidance)
    # An absent prefix is not "guidance": it must not raise.
    stub.produce_chunk(None, None, 1, 1, None, guidance=RTCGuidance(None, torch.zeros(0)))


# ============================================================================
# RTC — engine integration
# ============================================================================
def _core_and_batch():
    cfg = preset_config("tiny")
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu"))
    batch = collate([_obs(cfg), _obs(cfg, 1)], ["a", "b"])
    return cfg, policy, core, batch


def test_engine_without_guidance_is_bit_identical():
    """The RTC plumbing must be invisible when no async block is present."""
    _, _, core, batch = _core_and_batch()
    plain = core.execute(batch, generator=torch.Generator().manual_seed(0))
    explicit_none = core.execute(batch, generator=torch.Generator().manual_seed(0), rtc_guidance=None)
    disabled = core.execute(
        batch,
        generator=torch.Generator().manual_seed(0),
        rtc_guidance=build_rtc_guidance(None, 0, RTCGuidanceConfig(execution_horizon=0), action_horizon=16),
    )
    for a, b, c in zip(plain, explicit_none, disabled, strict=True):
        assert torch.equal(a.actions, b.actions)
        assert torch.equal(a.actions, c.actions)


def test_engine_rtc_matches_a_hand_rolled_reference_loop():
    """The engine's guided decode equals calling the guided step directly."""
    _, policy, core, batch = _core_and_batch()
    horizon = policy.config.action_horizon
    width = policy.config.action_dim
    prev = torch.randn(2, horizon, width)

    guidance = build_rtc_guidance(
        prev,
        inference_delay=3,
        config=RTCGuidanceConfig(
            execution_horizon=8, prefix_attention_schedule=PrefixAttentionSchedule.LINEAR
        ),
    )

    got = core.execute(batch, generator=torch.Generator().manual_seed(7), rtc_guidance=guidance)

    # Reproduce independently: same prefix, same noise, same guided step.
    staged = core._prefill(batch, None, torch.Generator().manual_seed(7), guidance)
    noise_at = flow_noise_end(policy.flow_schedule(staged.num_steps))
    x = staged.x
    prefix = staged.prefix
    for t_val, dt in policy.flow_schedule(staged.num_steps):
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
        velocity = rtc_guided_velocity(
            lambda value, t=t: policy.denoise_step(value, t, prefix), x, t_val, guidance, noise_at=noise_at
        )
        x = x + velocity * dt
    want = policy.finalize_actions(x, prefix)

    assert torch.allclose(got[0].actions, want[0], atol=1e-5)
    assert torch.allclose(got[1].actions, want[1], atol=1e-5)


def test_engine_rtc_actually_changes_the_chunk():
    """Sanity: conditioning a held prefix must move the produced actions."""
    _, policy, core, batch = _core_and_batch()
    horizon = policy.config.action_horizon
    width = policy.config.action_dim
    prev = torch.zeros(2, horizon, width)
    guidance = build_rtc_guidance(
        prev,
        4,
        RTCGuidanceConfig(execution_horizon=8, prefix_attention_schedule=PrefixAttentionSchedule.ZEROS),
    )
    plain = core.execute(batch, generator=torch.Generator().manual_seed(3))
    guided = core.execute(batch, generator=torch.Generator().manual_seed(3), rtc_guidance=guidance)
    assert not torch.allclose(plain[0].actions, guided[0].actions)


def test_engine_hard_prefix_pins_the_committed_steps_exactly():
    """Fixed-prefix inpainting clamps after every step, so held rows are exact."""
    _, policy, core, batch = _core_and_batch()
    horizon = policy.config.action_horizon
    width = policy.config.action_dim
    prev = torch.randn(2, horizon, width)
    guidance = build_rtc_guidance(prev, 4, RTCGuidanceConfig(execution_horizon=8, hard_prefix=True))
    got = core.execute(batch, generator=torch.Generator().manual_seed(11), rtc_guidance=guidance)
    assert torch.allclose(got[0].actions[:4], prev[0, :4], atol=1e-6)
    assert torch.allclose(got[1].actions[:4], prev[1, :4], atol=1e-6)


def test_engine_accepts_guidance_assembled_from_the_wire():
    """Conditioning built the serving way must need no pre-move by the caller.

    There are two producers of guidance: `build_rtc_guidance` on a caller-owned
    tensor, and `batch_rtc_guidance`, which parses request plans and always yields
    a host-side float32 tensor. The decode runs on the policy's device and dtype.
    A CPU/CUDA mismatch on this path is a real deployment failure — the GPU
    serving run found it — and the dtype half of it is reproducible here.
    """
    _, policy, core, batch = _core_and_batch()
    horizon = policy.config.action_horizon
    width = policy.config.action_dim
    plan = parse_async_plan(
        {
            "async": {
                "rtc": {
                    "prev_chunk_left_over": [[0.5] * width] * horizon,
                    "inference_delay": 4,
                    "execution_horizon": 8,
                    "hard_prefix": True,
                }
            }
        }
    )
    # Two plans for a two-row batch; the second row did not ask for RTC, which
    # also exercises the row-exclusion path that shares these weights.
    guidance = batch_rtc_guidance([plan, None], action_horizon=horizon, action_dim=width)
    assert guidance.prev_chunk_left_over.dtype == torch.float32
    got = core.execute(batch, generator=torch.Generator().manual_seed(13), rtc_guidance=guidance)
    assert torch.allclose(got[0].actions[:4], torch.full((4, width), 0.5), atol=1e-6)


def test_engine_rejects_guidance_for_recurrent_policies():
    policy = make_policy("mock_flow_vla", preset="tiny")

    class _Recurrent(type(policy)):  # type: ignore[misc, valid-type]
        @property
        def is_recurrent(self) -> bool:
            return True

    policy.__class__ = _Recurrent
    stub = EngineCore.__new__(EngineCore)
    stub.policy = policy
    guidance = build_rtc_guidance(torch.ones(1, 4, 2), 1, RTCGuidanceConfig(execution_horizon=2))
    with pytest.raises(UnsupportedRecurrentModeError):
        EngineCore.execute(stub, None, rtc_guidance=guidance)


# ============================================================================
# VLASH — state roll-forward
# ============================================================================
def test_roll_state_forward_absolute_reads_the_target():
    state = torch.tensor([0.0, 0.0])
    actions = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    rolled = roll_state_forward(state, actions, 2, ActionSpaceSemantics.ABSOLUTE)
    assert torch.equal(rolled, torch.tensor([2.0, 20.0]))


def test_roll_state_forward_delta_accumulates():
    state = torch.tensor([0.5, -0.5])
    actions = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    rolled = roll_state_forward(state, actions, 2, ActionSpaceSemantics.DELTA)
    assert torch.equal(rolled, torch.tensor([3.5, 2.5]))


def test_roll_state_forward_zero_delay_is_identity():
    state = torch.randn(4)
    actions = torch.randn(5, 4)
    for semantics in ActionSpaceSemantics:
        assert torch.equal(roll_state_forward(state, actions, 0, semantics), state)


def test_roll_state_forward_consumes_the_whole_buffer():
    state = torch.zeros(2)
    actions = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    assert torch.equal(
        roll_state_forward(state, actions, 2, ActionSpaceSemantics.ABSOLUTE), torch.tensor([2.0, 2.0])
    )
    assert torch.equal(
        roll_state_forward(state, actions, 2, ActionSpaceSemantics.DELTA), torch.tensor([3.0, 3.0])
    )


def test_roll_state_forward_batched_and_rank_preserving():
    state = torch.zeros(2, 3)
    actions = torch.ones(2, 4, 3)
    rolled = roll_state_forward(state, actions, 3, ActionSpaceSemantics.DELTA)
    assert rolled.shape == (2, 3)
    assert torch.equal(rolled, torch.full((2, 3), 3.0))


def test_roll_state_forward_golden_vectors():
    """Both repos assert the same table, so the timing math cannot drift."""
    for case in _golden()["vlash_roll_forward"]:
        rolled = roll_state_forward(
            torch.tensor(case["state"], dtype=torch.float32),
            torch.tensor(case["actions"], dtype=torch.float32),
            case["delay"],
            ActionSpaceSemantics(case["semantics"]),
        )
        assert torch.allclose(rolled, torch.tensor(case["expected"], dtype=torch.float32), atol=1e-6), case[
            "name"
        ]


@pytest.mark.parametrize(
    "state,actions,delay,match",
    [
        (torch.zeros(2), torch.zeros(3, 2), -1, "non-negative"),
        (torch.zeros(2), torch.zeros(3, 2), 4, "exceeds"),
        (torch.zeros(3), torch.zeros(3, 2), 1, "does not match state width"),
        (torch.zeros(2, 2), torch.zeros(3, 3, 2), 1, "batch"),
    ],
)
def test_roll_state_forward_rejects_invalid_input(state, actions, delay, match):
    with pytest.raises(ValueError, match=match):
        roll_state_forward(state, actions, delay)


def test_roll_state_forward_rejects_non_integer_delay():
    with pytest.raises(TypeError):
        roll_state_forward(torch.zeros(2), torch.zeros(3, 2), 1.5)  # type: ignore[arg-type]


def test_roll_state_forward_is_dtype_preserving():
    state = torch.zeros(2, dtype=torch.float64)
    actions = torch.ones(3, 2, dtype=torch.float32)
    assert roll_state_forward(state, actions, 1, ActionSpaceSemantics.DELTA).dtype == torch.float64


# ============================================================================
# VLASH — applying the plan to a request state
# ============================================================================
def test_apply_vlash_state_plan_rewrites_only_named_fields():
    state = {"joint_a": 0.0, "joint_b": 0.0, "gripper": 0.5, "unrelated": "keep"}
    plan = VlashStatePlan(
        state_fields=("joint_a", "joint_b"),
        pending_actions=((1.0, 2.0), (3.0, 4.0)),
        delay=2,
        semantics=ActionSpaceSemantics.DELTA,
    )
    updated = apply_vlash_state_plan(state, plan)
    assert updated["joint_a"] == pytest.approx(4.0)
    assert updated["joint_b"] == pytest.approx(6.0)
    assert updated["gripper"] == 0.5
    assert updated["unrelated"] == "keep"


def test_apply_vlash_state_plan_preserves_integral_fields():
    state = {"joint": 1}
    plan = VlashStatePlan(
        state_fields=("joint",),
        pending_actions=((2.0,), (3.0,)),
        delay=1,
        semantics=ActionSpaceSemantics.ABSOLUTE,
    )
    assert apply_vlash_state_plan(state, plan)["joint"] == 2


def test_apply_vlash_state_plan_rejects_missing_fields():
    plan = VlashStatePlan(state_fields=("absent",), pending_actions=((1.0,),), delay=1)
    with pytest.raises(ValueError, match="missing VLASH field"):
        apply_vlash_state_plan({"present": 1.0}, plan)


def test_apply_vlash_state_plan_handles_vector_valued_fields():
    """The normal deployment shape: one field holding the whole joint vector.

    The pi0.5 adapter reads proprioception as a single ``observation.state`` entry,
    not as named scalars, so a plan must be able to name that one field and supply
    action vectors of the field's full width.
    """
    state = {"observation.state": [0.0, 0.0, 0.0], "gripper": 0.5}
    plan = VlashStatePlan(
        state_fields=("observation.state",),
        pending_actions=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        delay=2,
        semantics=ActionSpaceSemantics.DELTA,
    )
    assert plan.action_width == 3
    updated = apply_vlash_state_plan(state, plan)
    assert updated["observation.state"] == pytest.approx([5.0, 7.0, 9.0])
    assert updated["gripper"] == 0.5


def test_apply_vlash_state_plan_mixes_scalar_and_vector_fields():
    state = {"arm": [1.0, 1.0], "gripper": 1.0}
    plan = VlashStatePlan(
        state_fields=("arm", "gripper"),
        pending_actions=((1.0, 2.0, 3.0),),
        delay=1,
        semantics=ActionSpaceSemantics.DELTA,
    )
    updated = apply_vlash_state_plan(state, plan)
    assert updated["arm"] == pytest.approx([2.0, 3.0])
    assert updated["gripper"] == pytest.approx(4.0)


def test_projected_roll_maps_action_columns_onto_a_wider_state():
    """LIBERO's case: an 8-wide state rolled by a 7-wide OSC delta action.

    The gripper is the mismatch — two finger positions against one command — so the
    projection marks it unmapped and it keeps its current value rather than being
    silently driven by an unrelated column.
    """
    state = torch.zeros(8)
    actions = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    mapping = (0, 1, 2, 3, 4, 5, -1)
    rolled = roll_state_forward_projected(state, actions, 2, ActionSpaceSemantics.DELTA, mapping)
    assert rolled.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0]


def test_projected_roll_absolute_writes_only_mapped_columns():
    state = torch.tensor([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    actions = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    rolled = roll_state_forward_projected(
        state, actions, 1, ActionSpaceSemantics.ABSOLUTE, (0, 1, 2, 3, 4, 5, -1)
    )
    assert rolled.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 9.0]


@pytest.mark.parametrize(
    "mapping,actions,match",
    [
        ((0, 1), torch.zeros(1, 3), "3 entries but the action vector is 3 wide|entries"),
        ((0, 0, 1), torch.zeros(1, 3), "two action columns"),
        ((0, 1, 8), torch.zeros(1, 3), "beyond the"),
        ((0, 1, -2), torch.zeros(1, 3), ">= -1"),
    ],
)
def test_projected_roll_validates_the_mapping(mapping, actions, match):
    with pytest.raises(ValueError):
        roll_state_forward_projected(torch.zeros(8), actions, 1, ActionSpaceSemantics.DELTA, mapping)


def test_apply_vlash_state_plan_uses_the_projection():
    """The end-to-end path: a projection makes an 8-state / 7-action plan legal."""
    state = {"observation.state": [0.0] * 8}
    plan = VlashStatePlan(
        state_fields=("observation.state",),
        pending_actions=((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),),
        delay=1,
        semantics=ActionSpaceSemantics.DELTA,
        action_to_state=(0, 1, 2, 3, 4, 5, -1),
    )
    updated = apply_vlash_state_plan(state, plan)
    assert updated["observation.state"] == pytest.approx([1.0] * 6 + [0.0, 0.0])


def test_apply_vlash_state_plan_rejects_width_disagreement():
    """A state whose width does not match the action vectors is an error.

    Silently reinterpreting the columns would roll the wrong joints, which is
    worse than refusing the request.
    """
    state = {"observation.state": [0.0, 0.0]}
    plan = VlashStatePlan(state_fields=("observation.state",), pending_actions=((1.0, 2.0, 3.0),), delay=1)
    with pytest.raises(ValueError, match="pending_actions rows are 3 wide"):
        apply_vlash_state_plan(state, plan)


def test_apply_vlash_state_plan_rejects_nested_non_numeric():
    plan = VlashStatePlan(state_fields=("a",), pending_actions=((1.0,),), delay=1)
    with pytest.raises(ValueError, match="must hold numbers"):
        apply_vlash_state_plan({"a": "not-a-number"}, plan)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(state_fields=(), pending_actions=(), delay=0), "must not be empty"),
        (dict(state_fields=("a", "a"), pending_actions=((1.0, 1.0),), delay=1), "unique"),
        (dict(state_fields=("a",), pending_actions=((1.0,), (2.0, 3.0)), delay=1), "same width"),
        (dict(state_fields=("a",), pending_actions=((),), delay=1), "must not be empty"),
        (dict(state_fields=("a",), pending_actions=((1.0,),), delay=5), "exceeds"),
        (dict(state_fields=("a",), pending_actions=((1.0,),), delay=-1), "non-negative"),
    ],
)
def test_vlash_plan_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VlashStatePlan(**kwargs)


# ============================================================================
# Wire contracts
# ============================================================================
def test_parse_async_plan_absent_block_returns_none():
    assert parse_async_plan({}) is None
    assert parse_async_plan({"observation_id": "x"}) is None


def test_parse_async_plan_reads_both_halves():
    plan = parse_async_plan(
        {
            "async": {
                "schema": ASYNC_SCHEMA,
                "rtc": {"prev_chunk_left_over": [[1.0, 2.0]], "inference_delay": 1, "execution_horizon": 4},
                "vlash": {
                    "state_fields": ["a", "b"],
                    "pending_actions": [[1.0, 2.0], [3.0, 4.0]],
                    "delay": 1,
                    "action_space": "delta",
                },
            }
        }
    )
    assert plan is not None and plan.requested
    assert plan.rtc is not None and plan.rtc.inference_delay == 1
    assert plan.vlash is not None and plan.vlash.delay == 1
    assert plan.vlash.semantics is ActionSpaceSemantics.DELTA


def test_parse_async_plan_rejects_malformed_blocks():
    with pytest.raises(ValueError, match="must be an object"):
        parse_async_plan({"async": 5})
    with pytest.raises(ValueError, match="unsupported async schema"):
        parse_async_plan({"async": {"schema": "other"}})
    with pytest.raises(ValueError, match="must be a list of lists"):
        parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": 3}}})
    with pytest.raises(ValueError, match="finite"):
        parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[float("nan")]]}}})
    with pytest.raises(ValueError, match="non-empty list"):
        parse_async_plan({"async": {"vlash": {"state_fields": []}}})


def _request(metadata, state):
    return RawPolicyRequest(
        session_id="session-a",
        request_id="req-1",
        step_id=0,
        instruction="pick",
        state=state,
        images=(RawImage("cam", "image/png", b"x"),),
        metadata=metadata,
    )


def test_apply_request_async_state_is_a_noop_without_a_block():
    request = _request({}, {"a": 1.0})
    assert apply_request_async_state(request) is request


def test_apply_request_async_state_rolls_the_state():
    request = _request(
        {
            "async": {
                "vlash": {
                    "state_fields": ["a"],
                    "pending_actions": [[2.0], [3.0]],
                    "delay": 2,
                    "action_space": "delta",
                }
            }
        },
        {"a": 1.0, "b": 9.0},
    )
    updated = apply_request_async_state(request)
    assert updated.state["a"] == pytest.approx(6.0)
    assert updated.state["b"] == 9.0
    # the original is untouched: the plan is applied by replacement
    assert request.state["a"] == 1.0


def test_batch_rtc_guidance_returns_none_without_requests():
    assert batch_rtc_guidance([None, None]) is None


def test_batch_rtc_guidance_stacks_and_zero_fills():
    active = parse_async_plan(
        {
            "async": {
                "rtc": {"prev_chunk_left_over": [[1.0], [2.0]], "inference_delay": 1, "execution_horizon": 2}
            }
        }
    )
    guidance = batch_rtc_guidance([active, None], action_horizon=2, action_dim=1)
    assert guidance is not None
    assert guidance.prev_chunk_left_over.shape == (2, 2, 1)
    assert torch.equal(guidance.prev_chunk_left_over[1], torch.zeros(2, 1))
    assert guidance.row_scale is not None
    assert guidance.row_scale.flatten().tolist() == [1.0, 0.0]


def test_batch_rtc_guidance_excludes_rows_that_did_not_ask():
    """A zero leftover must NOT be guided toward zero actions.

    The weights are shared across a batch, so a row that did not request RTC has
    to be excluded explicitly; otherwise its correction would pull it toward the
    zero prefix. Assert that the excluded row's velocity is exactly the
    unguided one while the participating row's is not.
    """
    torch.manual_seed(5)
    width = 2
    weight = torch.randn(width, width) * 0.4
    bias = torch.randn(width) * 0.1
    denoise = _linear_denoiser(weight, bias)
    x = torch.randn(2, 3, width)

    active = parse_async_plan(
        {"async": {"rtc": {"prev_chunk_left_over": [[1.0, 1.0], [1.0, 1.0]], "inference_delay": 1}}}
    )
    guidance = batch_rtc_guidance([active, None], action_horizon=3, action_dim=width)
    got = rtc_guided_velocity(denoise, x, 0.5, guidance)

    plain = denoise(x)
    assert torch.allclose(got[1], plain[1], atol=1e-6), "the un-asked row must be unconstrained"
    assert not torch.allclose(got[0], plain[0], atol=1e-4), "the asked row must be conditioned"


def test_batch_rtc_guidance_rejects_mixed_delays():
    one = parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[1.0]], "inference_delay": 1}}})
    two = parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[1.0]], "inference_delay": 2}}})
    with pytest.raises(ValueError, match="inference_delay must match"):
        batch_rtc_guidance([one, two])


def test_batch_rtc_guidance_rejects_mismatched_prefix_shapes():
    one = parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[1.0]], "inference_delay": 1}}})
    two = parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[1.0], [2.0]], "inference_delay": 1}}})
    with pytest.raises(ValueError, match="shape must match"):
        batch_rtc_guidance([one, two])


# ============================================================================
# RTC — server-side committed-prefix cache (the pi0.5 adapter)
# ============================================================================
def _prefix_adapter(cached: torch.Tensor | None, session_id: str = "session-a"):
    """A pi0.5 adapter with only the committed-prefix cache wired up."""
    from embodiinfer.policies.pi05.serving import Pi05ServingAdapter

    adapter = object.__new__(Pi05ServingAdapter)
    adapter._last_model_chunk = {} if cached is None else {session_id: cached}
    return adapter


def test_committed_prefix_comes_from_the_last_issued_chunk():
    """With only ``inference_delay`` on the wire, the server slices its own cache."""
    adapter = _prefix_adapter(torch.arange(12, dtype=torch.float32).reshape(6, 2))
    plan = parse_async_plan({"async": {"rtc": {"inference_delay": 2}}})
    resolved = adapter._resolve_rtc_plan(plan, "session-a")
    assert resolved.rtc.prev_chunk_left_over == ((4.0, 5.0), (6.0, 7.0), (8.0, 9.0), (10.0, 11.0))


def test_committed_prefix_is_absent_on_the_first_inference():
    """No cached chunk means no conditioning, not an error."""
    adapter = _prefix_adapter(None)
    plan = parse_async_plan({"async": {"rtc": {"inference_delay": 1}}})
    resolved = adapter._resolve_rtc_plan(plan, "session-a")
    assert resolved.rtc.prev_chunk_left_over is None
    assert not resolved.rtc.enabled


def test_committed_prefix_is_absent_once_the_chunk_is_consumed():
    adapter = _prefix_adapter(torch.zeros(4, 2))
    plan = parse_async_plan({"async": {"rtc": {"inference_delay": 4}}})
    assert adapter._resolve_rtc_plan(plan, "session-a").rtc.prev_chunk_left_over is None


def test_explicit_prefix_takes_precedence_over_the_cache():
    """A client that genuinely holds model-space actions can override the cache."""
    adapter = _prefix_adapter(torch.zeros(4, 2))
    plan = parse_async_plan({"async": {"rtc": {"prev_chunk_left_over": [[7.0, 7.0]], "inference_delay": 1}}})
    resolved = adapter._resolve_rtc_plan(plan, "session-a")
    assert resolved.rtc.prev_chunk_left_over == ((7.0, 7.0),)


def test_reset_drops_the_committed_prefix():
    """A new episode must not be held to the previous episode's chunk."""
    adapter = _prefix_adapter(torch.arange(12, dtype=torch.float32).reshape(6, 2))
    adapter.reset("session-a")
    plan = parse_async_plan({"async": {"rtc": {"inference_delay": 1}}})
    assert adapter._resolve_rtc_plan(plan, "session-a").rtc.prev_chunk_left_over is None


def test_request_without_async_block_needs_no_cache():
    adapter = _prefix_adapter(None)
    assert adapter._resolve_rtc_plan(None, "session-a") is None
