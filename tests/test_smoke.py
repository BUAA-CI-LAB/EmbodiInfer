"""CPU-runnable smoke tests (tiny preset, no CUDA graph)."""

import torch

from embodiinfer import EmbodiInfer, EngineConfig, Observation, preset_config
from embodiinfer.engine.rollout.demo import RolloutEngine, ToyReachEnv


def _cfg():
    return preset_config("tiny")


def _obs(cfg, env_id=0):
    return Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
        env_id=env_id,
    )


def _engine():
    return EmbodiInfer(
        "mock_flow_vla",
        preset="tiny",
        engine_config=EngineConfig(device="cpu", use_cuda_graph=False, max_batch_size=8),
    )


def test_single_action_shape():
    cfg = _cfg()
    engine = _engine()
    chunk = engine.act(_obs(cfg))
    assert chunk.actions.shape == (cfg.action_horizon, cfg.action_dim)


def test_batched_count_and_shape():
    cfg = _cfg()
    engine = _engine()
    chunks = engine.act([_obs(cfg, i) for i in range(5)])
    assert len(chunks) == 5
    for c in chunks:
        assert c.actions.shape == (cfg.action_horizon, cfg.action_dim)


def test_determinism_with_seed():
    cfg = _cfg()
    engine = _engine()
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    from embodiinfer.types import collate

    b = collate([_obs(cfg)], ["a"])
    a1 = engine.core.execute(b, generator=g1)[0].actions
    a2 = engine.core.execute(b, generator=g2)[0].actions
    assert torch.allclose(a1, a2, atol=1e-4)


def test_rollout_and_logprob():
    cfg = _cfg()
    engine = _engine()
    env = ToyReachEnv(num_envs=4, cfg=cfg)
    roller = RolloutEngine(engine.backend, env, exec_horizon=2)
    stats = roller.collect(num_chunks=2)
    assert stats["total_env_steps"] == 4 * 2 * 2
    actions, logprob = engine.backend.generate_with_logprob(env.reset(), num_samples=3)
    assert actions.shape == (4, 3, cfg.action_horizon, cfg.action_dim)
    assert logprob.shape == (4, 3)


def test_data_parallel_shards_and_gathers():
    from embodiinfer.engine.parallel.data_parallel import DataParallelEngine

    cfg = _cfg()
    cores = [_engine().core for _ in range(2)]  # 2 CPU replicas
    dp = DataParallelEngine(cores)
    assert dp.num_replicas == 2
    chunks = dp.execute([_obs(cfg, i) for i in range(5)])
    assert len(chunks) == 5
    for c in chunks:
        assert c.actions.shape == (cfg.action_horizon, cfg.action_dim)


def test_best_of_n_and_weight_sync():
    cfg = _cfg()
    engine = _engine()
    env = ToyReachEnv(num_envs=3, cfg=cfg)
    picked = engine.backend.best_of_n(env.reset(), num_samples=4)
    assert len(picked) == 3
    engine.backend.update_weights(engine.policy.state_dict())
