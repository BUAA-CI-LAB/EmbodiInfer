"""Vectorized environment interface + a toy reaching task.

The interface mirrors what a VLA rollout needs: batched observations shaped like
real camera + proprio inputs, chunked action application, and per-env reward /
done. The toy env is deliberately cheap on the CPU side so that benchmarks
measure the *policy generation* path, while still emitting correctly shaped
images so the vision encoder does real work.
"""

from __future__ import annotations

import abc

import numpy as np
import torch

from ....policies.mock.configuration_mock import MockConfig
from ....types import Observation


class VectorEnv(abc.ABC):
    num_envs: int

    @abc.abstractmethod
    def reset(self) -> list[Observation]: ...

    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> tuple[list[Observation], np.ndarray, np.ndarray]:
        """Advance one step. actions: [num_envs, action_dim].
        Returns (observations, reward[num_envs], done[num_envs])."""


class ToyReachEnv(VectorEnv):
    """Batched 2D point-reach. Cheap dynamics, real-shaped observations."""

    def __init__(
        self,
        num_envs: int,
        cfg: MockConfig,
        seed: int = 0,
        fixed_goal: tuple[float, float] | None = None,
    ):
        self.num_envs = num_envs
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        # A shared, fixed goal turns the task into a small set of learnable
        # state->action pairs (used by the GRPO demo so learning is visible in a
        # short run); None randomizes a distinct goal per env, as in real rollout.
        self.fixed_goal = None if fixed_goal is None else np.asarray(fixed_goal, dtype=np.float32)
        self._instr = "reach the goal"
        self._tokens = torch.zeros(cfg.max_lang_len, dtype=torch.long)
        self._tokens[: min(4, cfg.max_lang_len)] = torch.tensor([1, 2, 3, 4][: cfg.max_lang_len])
        self.pos = None
        self.goal = None
        self.reset()

    def _obs(self) -> list[Observation]:
        c = self.cfg
        obs = []
        for i in range(self.num_envs):
            img = torch.from_numpy(
                self.rng.rand(c.num_cameras, 3, c.image_size, c.image_size).astype(np.float32)
            )
            state = np.zeros(c.state_dim, dtype=np.float32)
            state[:2] = self.pos[i]
            state[2:4] = self.goal[i]
            obs.append(
                Observation(
                    images=img,
                    state=torch.from_numpy(state),
                    instruction_tokens=self._tokens.clone(),
                    instruction=self._instr,
                    env_id=i,
                )
            )
        return obs

    def _sample_goal(self, n: int) -> np.ndarray:
        if self.fixed_goal is not None:
            return np.tile(self.fixed_goal, (n, 1))
        return self.rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)

    def reset(self) -> list[Observation]:
        self.pos = self.rng.uniform(-1, 1, size=(self.num_envs, 2)).astype(np.float32)
        self.goal = self._sample_goal(self.num_envs)
        return self._obs()

    def step(self, actions: np.ndarray):
        # use first 2 action dims as a velocity command
        act = np.asarray(actions, dtype=np.float32)[:, :2]
        self.pos = self.pos + 0.1 * np.tanh(act)
        dist = np.linalg.norm(self.pos - self.goal, axis=1)
        reward = -dist
        done = dist < 0.1
        # auto-reset finished envs
        for i in np.where(done)[0]:
            self.pos[i] = self.rng.uniform(-1, 1, size=2)
            self.goal[i] = self.fixed_goal if self.fixed_goal is not None else self.rng.uniform(-1, 1, size=2)
        return self._obs(), reward.astype(np.float32), done.astype(np.bool_)
