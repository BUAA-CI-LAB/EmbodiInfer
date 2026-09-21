"""``EngineCore.execute_pipelined``: CPU fallback equivalence + (GPU) lossless overlap.

The CPU tests exercise the fallback path (no CUDA graph -> sequential execute),
which must return exactly what calling ``execute`` per batch would. The bit-exact
overlap itself is CUDA-only (``@pytest.mark.gpu``), verified on a GPU host.
"""

import asyncio

import pytest
import torch

from embodiinfer import EngineConfig, Observation, preset_config
from embodiinfer.engine.async_engine import AsyncEngine
from embodiinfer.engine.core import EngineCore
from embodiinfer.policies.factory import make_policy
from embodiinfer.types import collate


def _obs(cfg, i=0):
    return Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
        env_id=i,
    )


def _batches(cfg, sizes):
    out, k = [], 0
    for s in sizes:
        out.append(collate([_obs(cfg, k + j) for j in range(s)], [str(k + j) for j in range(s)]))
        k += s
    return out


def test_pipelined_fallback_matches_sequential_cpu():
    """On CPU (no graph) execute_pipelined falls back to sequential execute and
    returns per-batch actions identical to calling execute on each batch."""
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    cfg = preset_config("tiny")
    batches = _batches(cfg, [2, 3])
    r_pipe = core.execute_pipelined(batches, generator=torch.Generator().manual_seed(0))
    gen = torch.Generator().manual_seed(0)
    r_seq = [core.execute(b, generator=gen) for b in batches]
    assert len(r_pipe) == len(r_seq)
    for rp, rs in zip(r_pipe, r_seq):
        assert [c.request_id for c in rp] == [c.request_id for c in rs]
        for cp, cs in zip(rp, rs):
            assert torch.equal(cp.actions, cs.actions)


def test_pipelined_single_batch_is_execute_cpu():
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    batches = _batches(preset_config("tiny"), [2])
    r = core.execute_pipelined(batches, generator=torch.Generator().manual_seed(0))
    assert len(r) == 1 and len(r[0]) == 2


def test_pipelined_empty_is_empty_cpu():
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    assert core.execute_pipelined([]) == []


@pytest.mark.parametrize("pipeline", [False, True])
def test_async_engine_delivers_all_requests_cpu(pipeline):
    """Both the sync and the pipelined async loop deliver one action chunk per
    request (the pipelined loop must also drain its last staged window)."""
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False, max_batch_size=3))
    cfg = preset_config("tiny")

    async def run():
        eng = AsyncEngine(core, pipeline=pipeline)
        await eng.start()
        results = await asyncio.gather(*[eng.generate(_obs(cfg, i)) for i in range(5)])
        await eng.stop()
        return results

    results = asyncio.wait_for(run(), timeout=10.0)
    results = asyncio.run(results)
    assert len(results) == 5
    for c in results:
        assert c.actions.shape == (cfg.action_horizon, cfg.action_dim)


@pytest.mark.gpu
def test_pipelined_lossless_gpu():
    """On CUDA, overlapping consecutive batches is bit-exact vs sequential execute."""
    policy = make_policy("mock_flow_vla", preset="small")
    core = EngineCore(policy, EngineConfig(device="cuda", use_cuda_graph=True, capture_full_loop=True))
    cfg = policy.config
    batches = _batches(cfg, [4, 4])  # B=4 buckets (mock B=1 has a separate known race)
    r_pipe = core.execute_pipelined(batches, generator=torch.Generator(device="cuda").manual_seed(0))
    gen = torch.Generator(device="cuda").manual_seed(0)
    r_seq = [core.execute(b, generator=gen) for b in batches]
    for rp, rs in zip(r_pipe, r_seq):
        for cp, cs in zip(rp, rs):
            assert torch.max(torch.abs(cp.actions - cs.actions)).item() == 0.0
