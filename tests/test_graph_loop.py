"""Whole-loop CUDA-graph capture (proposal 0001).

CPU tests cover the value-independent pieces — the precomputed time schedule and
the backward-compatible engine switch. The bit-exact capture parity itself is
CUDA-only (``@pytest.mark.gpu`` / ``@pytest.mark.pi05``), run on a GPU host.
"""

import pytest
import torch

from embodiinfer import EngineConfig, Observation, preset_config
from embodiinfer.engine.core import EngineCore
from embodiinfer.engine.graph import _build_time_schedule
from embodiinfer.policies.factory import make_policy
from embodiinfer.types import collate


def _obs(cfg, env_id=0):
    return Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
        env_id=env_id,
    )


# ---- CPU: static schedule precompute ---------------------------------------
def test_time_schedule_matches_stepwise():
    """Precomputed ``t_all`` / ``dts`` equal the per-step ``torch.full`` sequence."""
    policy = make_policy("mock_flow_vla", preset="tiny")
    num_steps, batch = 10, 4
    t_all, dts = _build_time_schedule(policy, num_steps, batch, torch.device("cpu"), torch.float32)
    schedule = policy.flow_schedule(num_steps)
    assert t_all.shape == (num_steps, batch)
    assert len(dts) == num_steps
    for k, (t_val, dt) in enumerate(schedule):
        assert dts[k] == dt
        assert torch.equal(t_all[k], torch.full((batch,), t_val))


# ---- CPU: the switch is a safe no-op without a graph -----------------------
def test_full_loop_flag_is_noop_without_cuda():
    """On CPU (no capture) ``capture_full_loop`` must not change results: both
    configs fall through to the same eager integration path."""
    policy = make_policy("mock_flow_vla", preset="tiny")
    cfg = preset_config("tiny")
    batch = collate([_obs(cfg), _obs(cfg, 1)], ["a", "b"])
    core_off = EngineCore(policy, EngineConfig(device="cpu", capture_full_loop=False))
    core_on = EngineCore(policy, EngineConfig(device="cpu", capture_full_loop=True))
    a = core_off.execute(batch, generator=torch.Generator().manual_seed(0))
    b = core_on.execute(batch, generator=torch.Generator().manual_seed(0))
    for ca, cb in zip(a, b):
        assert torch.equal(ca.actions, cb.actions)


# ---- GPU: whole-loop capture is bit-exact against the eager loop -----------
@pytest.mark.gpu
def test_loop_graph_parity_gpu():
    policy = make_policy("mock_flow_vla", preset="tiny")
    cfg = preset_config("tiny")
    batch = collate([_obs(cfg, i) for i in range(4)], [str(i) for i in range(4)])
    eager = EngineCore(policy, EngineConfig(device="cuda", use_cuda_graph=False))
    loop = EngineCore(policy, EngineConfig(device="cuda", use_cuda_graph=True, capture_full_loop=True))

    def _run(core):
        return core.execute(batch, generator=torch.Generator(device="cuda").manual_seed(0))

    for ce, cl in zip(_run(eager), _run(loop)):
        assert torch.max(torch.abs(ce.actions - cl.actions)).item() == 0.0


# ---- GPU: full-loop matches the existing per-step graph too ----------------
@pytest.mark.gpu
def test_loop_graph_matches_per_step_graph_gpu():
    policy = make_policy("mock_flow_vla", preset="tiny")
    cfg = preset_config("tiny")
    batch = collate([_obs(cfg, i) for i in range(4)], [str(i) for i in range(4)])
    per_step = EngineCore(policy, EngineConfig(device="cuda", use_cuda_graph=True, capture_full_loop=False))
    loop = EngineCore(policy, EngineConfig(device="cuda", use_cuda_graph=True, capture_full_loop=True))

    def _run(core):
        return core.execute(batch, generator=torch.Generator(device="cuda").manual_seed(0))

    for cp, cl in zip(_run(per_step), _run(loop)):
        assert torch.max(torch.abs(cp.actions - cl.actions)).item() == 0.0
