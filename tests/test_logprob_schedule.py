"""Scheduled-sigma and per-step logprob tests for the flow-SDE sampler (CPU).

Cover the two extensions an RL trainer integration relies on: a callable
``sigma(t)`` schedule behaves identically to the equivalent constant, the
``per_step`` form sums to the aggregate, and the ratio-starts-at-1 invariant
(recompute at unchanged parameters equals the behavior logprob) holds under a
non-constant schedule.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from embodiinfer import EmbodiInfer, EngineConfig, preset_config
from embodiinfer.engine.rollout.demo.env import ToyReachEnv
from embodiinfer.engine.rollout.logprob import flow_logprob_recompute, flow_sample_with_logprob

_OVERRIDES = dict(image_size=32, patch_size=16, action_horizon=8)


def _policy_and_prefix(num_envs: int = 2):
    cfg = preset_config("tiny", **_OVERRIDES)
    engine = EmbodiInfer(
        "mock_flow_vla",
        preset="tiny",
        engine_config=EngineConfig(device="cpu", use_cuda_graph=False),
        **_OVERRIDES,
    )
    policy = engine.backend.policy
    obs = ToyReachEnv(num_envs=num_envs, cfg=cfg, seed=0).reset()
    batch = policy.collate(obs, [f"r{i}" for i in range(len(obs))]).to("cpu")
    return policy, policy.encode_prefix(batch)


def _sample(policy, prefix, sigma, per_step=False, seed=0):
    B = prefix.batch_size
    torch.manual_seed(seed)
    x0 = policy.new_noise(B)
    gen = torch.Generator().manual_seed(123)
    return flow_sample_with_logprob(
        policy, prefix, x0, 4, sigma, return_trajectory=True, per_step=per_step, generator=gen
    )


def test_constant_callable_matches_float_sigma():
    policy, prefix = _policy_and_prefix()
    a1, lp1, tr1 = _sample(policy, prefix, 0.1)
    a2, lp2, tr2 = _sample(policy, prefix, lambda t: 0.1)
    assert torch.equal(a1, a2)
    assert torch.equal(lp1, lp2)
    assert torch.equal(tr1, tr2)


def test_per_step_sums_to_aggregate():
    policy, prefix = _policy_and_prefix()
    sched = lambda t: 0.2 * math.sqrt(max(t, 1e-3) / max(1.0 - t, 1e-3))  # noqa: E731
    _, lp_total, _ = _sample(policy, prefix, sched)
    _, lp_steps, _ = _sample(policy, prefix, sched, per_step=True)
    assert lp_steps.shape == (prefix.batch_size, 4)
    assert torch.allclose(lp_steps.sum(dim=1), lp_total, atol=1e-5)


def test_recompute_matches_behavior_under_schedule():
    policy, prefix = _policy_and_prefix()
    sched = lambda t: 0.05 + 0.2 * t  # noqa: E731  (non-constant, well-behaved)
    _, lp_behavior, traj = _sample(policy, prefix, sched, per_step=True)
    lp_re = flow_logprob_recompute(policy, prefix, traj, 4, sched, per_step=True)
    assert torch.allclose(lp_re, lp_behavior, atol=1e-3, rtol=1e-4)
    assert lp_re.requires_grad


def test_generator_reproducibility():
    policy, prefix = _policy_and_prefix()
    a1, lp1, _ = _sample(policy, prefix, 0.1, seed=7)
    a2, lp2, _ = _sample(policy, prefix, 0.1, seed=7)
    assert torch.equal(a1, a2) and torch.equal(lp1, lp2)


def test_flow_decoder_generator_controls_initial_and_transition_noise():
    policy, prefix = _policy_and_prefix()
    before = torch.get_rng_state().clone()
    first = policy.decoder.sample_with_logprob(prefix, 4, 0.1, torch.Generator().manual_seed(7))
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(17)  # Unrelated global RNG consumers must not affect the supplied generator.
    second = policy.decoder.sample_with_logprob(prefix, 4, 0.1, torch.Generator().manual_seed(7))
    for actual, expected in zip(second, first, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_flow_preserves_samples_and_composes_with_schedule_and_mean(dtype):
    theta = torch.tensor(0.25, requires_grad=True)
    policy = SimpleNamespace(
        flow_schedule=lambda n: [(1.0 - k / n, -1.0 / n) for k in range(n)],
        denoise_step=lambda x, t, prefix: x * 0.1 + theta.to(dtype),
    )
    initial = torch.zeros(2, 2, 4, dtype=dtype)
    mask = torch.tensor([True, False, True, False])
    sigma = lambda t: 0.2 if t > 0.5 else 0.0  # noqa: E731
    mean_fn = lambda x, v, t, dt, sig: x + v * dt - x * (sig**2 * abs(dt))  # noqa: E731

    def sample(selection):
        return flow_sample_with_logprob(
            policy,
            None,
            initial,
            4,
            sigma,
            return_trajectory=True,
            per_step=True,
            generator=torch.Generator().manual_seed(19),
            mean_fn=mean_fn,
            return_velocities=True,
            logprob_mask=selection,
        )

    actions, behavior, trajectory, velocities = sample(mask)
    unmasked_actions, _, unmasked_trajectory, unmasked_velocities = sample(None)
    for actual, expected in [
        (actions, unmasked_actions),
        (trajectory, unmasked_trajectory),
        (velocities, unmasked_velocities),
    ]:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(behavior[:, 2:]) == 0

    def recompute(states):
        return flow_logprob_recompute(
            policy,
            None,
            states,
            4,
            sigma,
            per_step=True,
            mean_fn=mean_fn,
            logprob_mask=mask,
        )

    score = recompute(trajectory)
    torch.testing.assert_close(score, behavior, rtol=0, atol=1e-5)
    changed = trajectory.clone()
    changed[..., ~mask] += 100
    torch.testing.assert_close(recompute(changed), score, rtol=0, atol=0)
    score.sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad) and theta.grad.abs() > 0


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize(
    "mask, message",
    [
        (torch.ones(5), "does not match"),
        (torch.zeros(4), "at least one"),
        (torch.tensor([[[1, 0, 0, 0]], [[1, 1, 0, 0]]]), "equal across the batch"),
    ],
)
def test_flow_rejects_invalid_likelihood_masks_before_model_execution(mask, message, recompute):
    # No model methods are needed: mask validation must precede sampling/scoring.
    with pytest.raises(ValueError, match=message):
        if recompute:
            flow_logprob_recompute(None, None, torch.zeros(2, 3, 2, 4), 2, logprob_mask=mask)
        else:
            flow_sample_with_logprob(None, None, torch.zeros(2, 2, 4), 2, logprob_mask=mask)


def test_mean_fn_default_equivalent():
    policy, prefix = _policy_and_prefix()
    a1, lp1, tr1 = _sample(policy, prefix, 0.1)
    B = prefix.batch_size
    torch.manual_seed(0)
    x0 = policy.new_noise(B)
    gen = torch.Generator().manual_seed(123)
    a2, lp2, tr2 = flow_sample_with_logprob(
        policy,
        prefix,
        x0,
        4,
        0.1,
        return_trajectory=True,
        mean_fn=lambda x, v, t, dt, sig: x + v * dt,
        generator=gen,
    )
    assert torch.equal(a1, a2) and torch.equal(lp1, lp2) and torch.equal(tr1, tr2)


def test_zero_sigma_steps_are_deterministic_ode():
    policy, prefix = _policy_and_prefix()
    grid = [t for t, _ in policy.flow_schedule(4)]
    t_sde = grid[2]
    sched = lambda t: 0.3 if t == t_sde else 0.0  # noqa: E731
    _, lp_steps, traj = _sample(policy, prefix, sched, per_step=True)
    zero_cols = [k for k, t in enumerate(grid) if t != t_sde]
    assert torch.all(lp_steps[:, zero_cols] == 0)
    assert torch.all(lp_steps[:, 2] != 0)
    # deterministic steps reproduce regardless of generator state
    _, lp_steps2, traj2 = _sample(policy, prefix, sched, per_step=True, seed=0)
    assert torch.equal(traj, traj2) and torch.equal(lp_steps, lp_steps2)


def test_return_velocities_matches_trajectory_transitions():
    policy, prefix = _policy_and_prefix()
    B = prefix.batch_size
    torch.manual_seed(0)
    x0 = policy.new_noise(B)
    gen = torch.Generator().manual_seed(123)
    a1, lp1, tr1 = _sample(policy, prefix, 0.1)
    a2, lp2, tr2, vel = flow_sample_with_logprob(
        policy,
        prefix,
        x0,
        4,
        0.1,
        return_trajectory=True,
        generator=gen,
        return_velocities=True,
    )
    # velocities are an additive return: everything else is unchanged
    assert torch.equal(a1, a2) and torch.equal(lp1, lp2) and torch.equal(tr1, tr2)
    assert vel.shape == (B, 4, *tr2.shape[2:])
    # each returned v(x_k, t_k) reproduces the recorded transition exactly
    # under a zero-sigma (pure ODE) rerun: x_{k+1} == x_k + v_k * dt
    _, _, tr_ode, vel_ode = flow_sample_with_logprob(
        policy, prefix, x0, 4, 0.0, return_trajectory=True, return_velocities=True
    )
    for k, (_t_val, dt) in enumerate(policy.flow_schedule(4)):
        assert torch.equal(tr_ode[:, k + 1], tr_ode[:, k] + vel_ode[:, k] * dt)
    # and independently: v at (x_k, t_k) equals a direct denoise_step call
    for k, (t_val, _dt) in enumerate(policy.flow_schedule(4)):
        t = torch.full((B,), t_val, dtype=tr2.dtype)
        v_direct = policy.denoise_step(tr2[:, k], t, prefix)
        assert torch.equal(vel[:, k], v_direct)


def test_return_velocities_without_trajectory():
    policy, prefix = _policy_and_prefix()
    B = prefix.batch_size
    torch.manual_seed(0)
    x0 = policy.new_noise(B)
    out = flow_sample_with_logprob(policy, prefix, x0, 4, 0.1, return_velocities=True)
    assert len(out) == 3
    actions, logprob, vel = out
    assert vel.shape[:2] == (B, 4)
    assert actions.shape == x0.shape


def test_recompute_matches_behavior_with_mean_fn_and_mixed_sigma():
    policy, prefix = _policy_and_prefix()
    grid = [t for t, _ in policy.flow_schedule(4)]
    t_sde = grid[1]
    sched = lambda t: 0.25 if t == t_sde else 0.0  # noqa: E731
    # a flow_sde-style score-corrected mean (arbitrary but well-behaved form)
    mean_fn = lambda x, v, t, dt, sig: (  # noqa: E731
        x + v * dt - (sig * sig * abs(dt) / (2 * max(t, 1e-3))) * (x + v * (1 - t))
    )
    B = prefix.batch_size
    torch.manual_seed(0)
    x0 = policy.new_noise(B)
    gen = torch.Generator().manual_seed(123)
    _, lp_b, traj = flow_sample_with_logprob(
        policy,
        prefix,
        x0,
        4,
        sched,
        return_trajectory=True,
        per_step=True,
        generator=gen,
        mean_fn=mean_fn,
    )
    lp_re = flow_logprob_recompute(policy, prefix, traj, 4, sched, per_step=True, mean_fn=mean_fn)
    assert torch.allclose(lp_re, lp_b, atol=1e-3, rtol=1e-4)
    assert lp_re.requires_grad
