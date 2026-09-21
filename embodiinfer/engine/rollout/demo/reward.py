"""Reward functions for the GRPO demo.

A reward scores each candidate action chunk a group-sampling rollout produced.
The contract is deliberately small — ``(observations, actions[B, G, H, A]) ->
reward[B, G]`` — so the RL trainer stays agnostic to where the signal comes
from (a toy scorer here; a LIBERO task success / dense reward later).

Keeping the scorer a pure function of the observation (rather than stepping a
live env) keeps the demo's learning signal deterministic and cheap, so the
measured cost is the *generation* path, which is what vvla optimizes.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ....types import Observation


class Reward(Protocol):
    """Scores group-sampled candidates. ``actions``: ``[B, G, H, A]`` ->
    ``reward``: ``[B, G]`` (higher is better)."""

    def __call__(self, observations: list[Observation], actions: torch.Tensor) -> torch.Tensor: ...


class ToyReachReward:
    """Bounded dense reach reward: Gaussian proximity to the goal after one step.

    The env encodes ``pos = state[:2]`` and ``goal = state[2:4]``. This scorer
    reads the first action of each candidate chunk as a velocity command, applies
    one linear step ``new_pos = pos + step_scale * action[:2]``, and returns

        reward = exp( -0.5 * (dist / width)^2 )   in (0, 1], = 1 at the goal.

    Bandit-style: one action chunk, one scalar reward. Two deliberate choices
    make the demo's single-epoch policy gradient behave:

      * *Bounded* (unlike raw ``-distance``): a stray large-action candidate
        saturates to reward ~ 0 instead of a huge negative that would destabilize
        the update.
      * *Linear step* (unlike the env's ``tanh``-squashed dynamics): no
        saturation, so the action keeps a usable gradient toward the goal. The
        scorer is decoupled from the live env (the demo does not step it).
    """

    def __init__(self, step_scale: float = 1.0, width: float = 1.0):
        self.step_scale = step_scale
        self.width = width

    def __call__(self, observations: list[Observation], actions: torch.Tensor) -> torch.Tensor:
        # actions: [B, G, H, A]; use the first action's first two dims as velocity.
        states = torch.stack([o.state for o in observations]).to(torch.float32)  # [B, state_dim]
        pos = states[:, :2].unsqueeze(1)  # [B, 1, 2]
        goal = states[:, 2:4].unsqueeze(1)  # [B, 1, 2]
        vel = actions[:, :, 0, :2].to(torch.float32).cpu()  # [B, G, 2]
        dist = torch.linalg.norm(pos + self.step_scale * vel - goal, dim=-1)  # [B, G]
        return torch.exp(-0.5 * (dist / self.width) ** 2)
