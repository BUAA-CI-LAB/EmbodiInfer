"""SDE-loop CUDA-graph capture (RL rollout surface).

CPU tests cover the host-side pieces: the per-step SDE coefficients and the
algebraic identity between the coefficient-form transition mean the graph
computes and the ``mean_fn``-form the eager sampler uses. The capture parity
itself (graph vs an eager loop consuming the same pre-drawn noise, bit-exact)
is CUDA-only (``@pytest.mark.gpu``), run on a GPU host.
"""

import math

import pytest
import torch

from embodiinfer.models.schedulers.flow import sde_coefficients as _sde_coefficients
from embodiinfer.policies.factory import make_policy


def _flow_sde_sigma(t_val: float, num_steps: int, noise_level: float) -> float:
    """GR00T-orientation sigma for the mock policy's forward-time schedule."""
    denom = t_val if t_val > 0.0 else 1.0 / num_steps
    return noise_level * math.sqrt((1.0 - t_val) / denom)


# ---- CPU: coefficient formulas ----------------------------------------------
def test_sde_coefficients_formula():
    policy = make_policy("mock_flow_vla", preset="tiny")
    N = 4
    schedule = policy.flow_schedule(N)
    sigmas = [_flow_sde_sigma(t, N, 0.5) for t, _ in schedule]
    c_noise, c_corr, gamma = _sde_coefficients(schedule, sigmas)
    # mock schedule integrates 0 -> 1: noise at t = 0, data at t = 1
    for (t_val, dt), sig, cn, cc, g in zip(schedule, sigmas, c_noise, c_corr, gamma):
        assert cn == sig * math.sqrt(abs(dt))
        assert g == -t_val
        if sig > 0:
            assert cc == sig * sig * abs(dt) / (2.0 * abs(t_val - 1.0))
        else:
            assert cc == 0.0


def test_sde_coefficients_zero_sigma_is_ode():
    policy = make_policy("mock_flow_vla", preset="tiny")
    N = 4
    schedule = policy.flow_schedule(N)
    sigmas = [0.0] * N
    sigmas[2] = 0.3  # noise on one selected step only (flow-SDE mixed pattern)
    c_noise, c_corr, _gamma = _sde_coefficients(schedule, sigmas)
    for k in range(N):
        if k == 2:
            assert c_noise[k] > 0 and c_corr[k] > 0
        else:
            assert c_noise[k] == 0.0 and c_corr[k] == 0.0


def test_sde_coefficients_length_mismatch():
    policy = make_policy("mock_flow_vla", preset="tiny")
    with pytest.raises(ValueError):
        _sde_coefficients(policy.flow_schedule(4), [0.1] * 3)


# ---- CPU: coefficient-form mean == mean_fn-form mean ------------------------
def test_coefficient_mean_matches_mean_fn_form():
    """The graph's ``x + v*dt - c_corr*(x + gamma*v)`` equals the score-corrected
    flow-SDE ``mean_fn`` the eager sampler is driven with."""
    torch.manual_seed(0)
    N = 4
    policy = make_policy("mock_flow_vla", preset="tiny")
    schedule = policy.flow_schedule(N)
    sigmas = [0.0] * N
    sigmas[2] = 0.4
    c_noise, c_corr, gamma = _sde_coefficients(schedule, sigmas)
    x = torch.randn(2, 8, 4)
    v = torch.randn(2, 8, 4)
    for (t_val, dt), sig, cc, g in zip(schedule, sigmas, c_corr, gamma):
        if sig == 0.0:
            assert cc == 0.0
            continue
        # mock schedule is forward time (t: 0 -> 1, noise at t = 0): the
        # correction acts on x0_pred = x - v*t with denominator 2*(1 - t),
        # i.e. the GR00T-orientation form.
        assert g == -t_val
        mean_fn_form = x + v * dt - (sig * sig * abs(dt) / (2.0 * (1.0 - t_val))) * (x - v * t_val)
        coeff_form = x + v * dt - cc * (x + g * v)
        assert torch.equal(mean_fn_form, coeff_form)


# ---- GPU: capture parity against an eager loop on the same noise ------------
@pytest.mark.gpu
def test_sde_loop_graph_parity_gpu():
    from embodiinfer.engine.graph import SdeLoopGraph

    device = torch.device("cuda")
    policy = make_policy("mock_flow_vla", preset="tiny").to(device)
    cfg = policy.config
    from embodiinfer import Observation

    B, N = 4, 4
    obs = [
        Observation(
            images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
            state=torch.rand(cfg.state_dim),
            instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
            env_id=i,
        )
        for i in range(B)
    ]
    batch = policy.collate(obs, [str(i) for i in range(B)]).to(device)
    prefix = policy.encode_prefix(batch)

    schedule = policy.flow_schedule(N)
    sigmas = [0.0] * N
    sigmas[1] = _flow_sde_sigma(schedule[1][0], N, 0.5)

    torch.manual_seed(0)
    x0 = torch.randn(B, cfg.action_horizon, cfg.action_dim, device=device)
    eps = torch.randn(N, B, cfg.action_horizon, cfg.action_dim, device=device)

    # eager reference consuming the same pre-drawn noise
    from embodiinfer.models.schedulers.flow import sde_coefficients as coeffs

    c_noise, c_corr, gamma = coeffs(schedule, sigmas)
    x = x0
    traj_ref, vel_ref = [x0], []
    with torch.no_grad():
        for k, (t_val, dt) in enumerate(schedule):
            t = torch.full((B,), t_val, device=device, dtype=x.dtype)
            v = policy.denoise_step(x, t, prefix)
            vel_ref.append(v)
            mean = x + v * dt - c_corr[k] * (x + gamma[k] * v)
            x = mean + c_noise[k] * eps[k]
            traj_ref.append(x)
    traj_ref = torch.stack(traj_ref, dim=1)
    vel_ref = torch.stack(vel_ref, dim=1)

    graph = SdeLoopGraph(policy, B, device, torch.float32, N)
    actions, traj, vel = graph.sample(prefix, x0, sigmas, eps=eps)

    assert torch.max(torch.abs(traj - traj_ref)).item() == 0.0
    assert torch.max(torch.abs(vel - vel_ref)).item() == 0.0
    assert torch.max(torch.abs(actions - traj_ref[:, -1])).item() == 0.0

    # second call with a different selected step re-uses the same graph
    sigmas2 = [0.0] * N
    sigmas2[3] = _flow_sde_sigma(schedule[3][0], N, 0.5)
    actions2, traj2, _ = graph.sample(prefix, x0, sigmas2, eps=eps)
    assert not torch.equal(traj2, traj)
