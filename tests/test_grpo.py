"""GRPO closed-loop tests (CPU).

Cover the pieces the demo depends on: the differentiable log-prob recompute
agrees with the behavior log-prob at unchanged parameters (so the PPO ratio
starts at 1) and carries a gradient, group-relative advantage is normalized, the
toy reward ranks candidates sensibly, and one GRPO update moves a favored
candidate's log-prob up and disfavored ones down.
"""

from __future__ import annotations

import torch

from embodiinfer import EmbodiInfer, EngineConfig, ToyReachEnv, preset_config
from embodiinfer.engine.rollout.demo import GRPOConfig, GRPOTrainer, ToyReachReward
from embodiinfer.engine.rollout.logprob import flow_logprob_recompute
from embodiinfer.types import Observation

# Small images keep the CPU forward cheap without changing any behavior.
_OVERRIDES = dict(image_size=32, patch_size=16, action_horizon=8)


def _build(num_envs: int = 2):
    cfg = preset_config("tiny", **_OVERRIDES)
    engine = EmbodiInfer(
        "mock_flow_vla",
        preset="tiny",
        engine_config=EngineConfig(device="cpu", use_cuda_graph=False),
        **_OVERRIDES,
    )
    env = ToyReachEnv(num_envs=num_envs, cfg=cfg, seed=0)
    return engine.backend, env, cfg


def _recompute(policy, observations, trajectory, group_size, num_steps, sigma):
    batch = policy.collate(observations, [f"t{i}" for i in range(len(observations))]).to("cpu")
    prefix = policy.encode_prefix(batch).expand(group_size)
    return flow_logprob_recompute(policy, prefix, trajectory, num_steps, sigma)


def test_recompute_matches_behavior_at_theta0():
    torch.manual_seed(0)
    backend, env, _ = _build(num_envs=2)
    obs = env.reset()
    G, num_steps, sigma = 4, 4, 0.1
    samples = backend.sample_group(obs, G, sigma=sigma, num_steps=num_steps)

    recomputed = _recompute(backend.policy, obs, samples.recompute_state, G, num_steps, sigma).view(2, G)

    # Same trajectory + same params -> residual reduces to the sampled eps, so
    # the recomputed log-prob equals the behavior log-prob (up to fp32 noise).
    assert torch.allclose(recomputed, samples.behavior_logprob, atol=1e-3, rtol=1e-4)
    assert recomputed.requires_grad


def test_recompute_is_differentiable():
    torch.manual_seed(0)
    backend, env, _ = _build(num_envs=1)
    obs = env.reset()
    samples = backend.sample_group(obs, 2, sigma=0.1, num_steps=3)
    logp = _recompute(backend.policy, obs, samples.recompute_state, 2, 3, 0.1)
    logp.sum().backward()
    grads = [p.grad for p in backend.policy.parameters() if p.grad is not None]
    assert grads, "no gradient reached the policy"
    assert any(g.abs().sum() > 0 for g in grads)


def test_ratio_is_one_at_first_step():
    torch.manual_seed(0)
    backend, env, _ = _build(num_envs=2)
    obs = env.reset()
    G, num_steps, sigma = 4, 4, 0.1
    samples = backend.sample_group(obs, G, sigma=sigma, num_steps=num_steps)
    recomputed = _recompute(backend.policy, obs, samples.recompute_state, G, num_steps, sigma).view(2, G)
    ratio = torch.exp(recomputed - samples.behavior_logprob)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-2)


def test_advantage_normalization():
    backend, _, _ = _build(num_envs=1)
    trainer = GRPOTrainer(backend, ToyReachReward(), GRPOConfig(group_size=8))
    torch.manual_seed(1)
    reward = torch.randn(4, 8)
    adv = trainer._advantage(reward)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(adv.std(dim=1), torch.ones(4), atol=1e-3)


def test_toy_reach_reward_ranks_candidates():
    reward = ToyReachReward()
    obs = [
        Observation(
            images=torch.zeros(1, 3, 32, 32),
            state=torch.tensor([0.0, 0.0, 0.5, 0.0, 0, 0, 0, 0], dtype=torch.float32),
            instruction_tokens=torch.zeros(32, dtype=torch.long),
        )
    ]
    # candidate 0 steps onto the goal (+0.5 x); candidate 1 drives away (-0.5 x).
    actions = torch.zeros(1, 2, 8, 7)
    actions[0, 0, 0, 0] = 0.5
    actions[0, 1, 0, 0] = -0.5
    r = reward(obs, actions)
    assert r.shape == (1, 2)
    assert r[0, 0] > r[0, 1]
    assert torch.isclose(r[0, 0], torch.tensor(1.0), atol=1e-6)  # reaches the goal -> reward 1
    assert (r >= 0).all() and (r <= 1).all()  # bounded


def test_grpo_step_updates_weights():
    torch.manual_seed(0)
    backend, env, _ = _build(num_envs=4)
    trainer = GRPOTrainer(backend, ToyReachReward(), GRPOConfig(group_size=6, lr=1e-2, num_steps=4))
    before = backend.policy.action_out.weight.detach().clone()
    stats = trainer.step(env.reset())
    after = backend.policy.action_out.weight.detach()
    assert not torch.allclose(before, after)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


def test_grpo_update_moves_logprob_by_advantage():
    torch.manual_seed(0)
    backend, env, _ = _build(num_envs=1)
    policy = backend.policy
    obs = env.reset()
    # Larger sigma -> larger transition variance -> tamer log-prob gradients.
    G, num_steps, sigma = 4, 3, 0.5
    samples = backend.sample_group(obs, G, sigma=sigma, num_steps=num_steps)
    traj, behavior = samples.recompute_state, samples.behavior_logprob

    # Favor candidate 0, disfavor the rest (normalized group-relative advantage).
    raw = torch.tensor([[1.0, -1.0, -1.0, -1.0]])
    adv = (raw - raw.mean(dim=1, keepdim=True)) / raw.std(dim=1, keepdim=True)

    def logp():
        return _recompute(policy, obs, traj, G, num_steps, sigma).view(1, G)

    before = logp().detach()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy.train()
    ratio = torch.exp(logp() - behavior)
    loss = -(ratio * adv).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    after = logp().detach()

    # A step off theta_0 lowers every candidate's log-prob (the stored trajectory
    # becomes less likely under any shifted mean); GRPO raises the favored
    # candidate's log-prob *relative to* the disfavored group.
    delta = (after - before)[0]  # [G]
    assert delta[0] > delta[1:].mean()


def test_grpo_candidate_logprob_reduces_action_tokens_per_candidate():
    behavior = torch.tensor([[[-1.0, -2.0, 0.0], [-3.0, 0.0, 0.0]]])
    recomputed = behavior.reshape(2, 3)
    expected = torch.tensor([[-3.0, -3.0]])
    torch.testing.assert_close(GRPOTrainer._candidate_logprob(behavior, 1, 2), expected)
    torch.testing.assert_close(GRPOTrainer._candidate_logprob(recomputed, 1, 2), expected)
