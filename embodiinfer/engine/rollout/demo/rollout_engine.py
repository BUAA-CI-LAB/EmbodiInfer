"""RolloutEngine: couples the generation backend to a vectorized env.

Demonstrates the env<->policy loop with action chunking (receding horizon): each
generated chunk of ``exec_horizon`` actions is applied open-loop before the next
generation. This is the loop an RL trainer drives; here it is exercised
standalone so the generation path can be profiled against real environment
stepping. Deep RL analysis (advantages, PPO/GRPO updates) is intentionally left
to the trainer layer.
"""

from __future__ import annotations

import numpy as np
import torch

from ....utils import now_ns
from ..generation_backend import GenerationBackend
from .env import VectorEnv


class RolloutEngine:
    """Couples the generation backend to a vectorised environment.

    It exercises the environment/policy loop with action chunking and a receding horizon:
    each generated chunk of ``exec_horizon`` actions is applied open-loop before the next
    generation. The point is to drive and profile the generation path against real
    environment stepping; advantages and PPO/GRPO updates are intentionally left to the
    trainer layer.
    """

    def __init__(self, backend: GenerationBackend, env: VectorEnv, exec_horizon: int | None = None):
        self.backend = backend
        self.env = env
        H = backend.pcfg.action_horizon
        self.exec_horizon = min(exec_horizon or H, H)

    def collect(
        self, num_chunks: int, num_steps: int | None = None, with_logprob: bool = False, sigma: float = 0.1
    ) -> dict:
        obs = self.env.reset()
        total_env_steps = 0
        gen_ns = 0
        env_ns = 0
        rewards = []
        logprobs = []
        for _ in range(num_chunks):
            t0 = now_ns()
            if with_logprob:
                actions, logprob = self.backend.generate_with_logprob(
                    obs, num_steps=num_steps, sigma=sigma, num_samples=1
                )
                chunk = actions[:, 0]  # [num_envs, H, A]
                logprobs.append(logprob[:, 0].cpu().numpy())
            else:
                results = self.backend.generate(obs, num_steps=num_steps)
                chunk = torch.stack([r.actions for r in results], dim=0)  # [num_envs, H, A]
            gen_ns += now_ns() - t0

            for k in range(self.exec_horizon):
                a_k = chunk[:, k, :].numpy()
                te = now_ns()
                obs, reward, done = self.env.step(a_k)
                env_ns += now_ns() - te
                rewards.append(reward)
                total_env_steps += self.env.num_envs

        rewards = np.stack(rewards, axis=0)  # [T, num_envs]
        total_ns = gen_ns + env_ns
        return {
            "num_envs": self.env.num_envs,
            "num_chunks": num_chunks,
            "exec_horizon": self.exec_horizon,
            "total_env_steps": total_env_steps,
            "gen_time_s": gen_ns / 1e9,
            "env_time_s": env_ns / 1e9,
            "gen_frac": gen_ns / max(total_ns, 1),
            "env_steps_per_s": total_env_steps / max(total_ns / 1e9, 1e-9),
            "mean_reward": float(rewards.mean()),
            "logprobs": np.concatenate(logprobs) if logprobs else None,
        }
