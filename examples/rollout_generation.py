"""Rollout generation: the surface an RL trainer drives, on a toy env.

Exercises the ``GenerationBackend`` contract an RLinf-style split-worker learner
would call for action generation — plain rollout, best-of-N planning, a
surrogate-log-prob path for policy gradients, and in-place weight sync. This is
not an RL training loop: advantage estimation / PPO / GRPO live in the trainer,
which is out of scope for embodiinfer.

    python examples/rollout_generation.py
"""

from __future__ import annotations

import torch

from embodiinfer import EngineConfig, ToyReachEnv, EmbodiInfer, preset_config
from embodiinfer.engine.rollout.demo import RolloutEngine


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = preset_config("small")
    engine = EmbodiInfer(
        "mock_flow_vla",
        preset="small",
        engine_config=EngineConfig(device=device, use_cuda_graph=(device == "cuda"), max_batch_size=32),
    )
    backend = engine.backend
    env = ToyReachEnv(num_envs=16, cfg=cfg)
    print(f"[rollout-generation] device={device}  envs={env.num_envs}")

    # 1) plain rollout with env <-> policy pipelining
    roller = RolloutEngine(backend, env, exec_horizon=4)
    stats = roller.collect(num_chunks=5)
    print(
        "[rollout]   env_steps={total_env_steps}  env_steps/s={env_steps_per_s:.0f}  "
        "gen_frac={gen_frac:.2f}  mean_reward={mean_reward:.3f}".format(**stats)
    )

    # 2) best-of-N planning: N candidates from one shared prefix, keep the best
    obs = env.reset()
    picked = backend.best_of_n(obs, num_samples=4)
    print(
        f"[best-of-N] chose 1 of 4 candidates per env; "
        f"value={picked[0].value:.2f}  action={tuple(picked[0].actions.shape)}"
    )

    # 3) surrogate log-prob (policy-gradient rollout path)
    actions, logprob = backend.generate_with_logprob(obs, num_samples=1, sigma=0.1)
    print(
        f"[logprob]   actions={tuple(actions.shape)}  logprob={tuple(logprob.shape)}  "
        f"mean_logprob={logprob.mean().item():.1f}"
    )

    # 4) framework-neutral refit (in-place after a learner optimizer step)
    result = backend.refit(engine.policy.state_dict())
    print(f"[refit] updated policy weights in place, version={result.version}")


if __name__ == "__main__":
    main()
