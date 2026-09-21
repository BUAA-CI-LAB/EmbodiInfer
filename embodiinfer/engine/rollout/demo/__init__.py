"""Demo / toy scaffolding for the rollout surface — NOT the product.

``engine/rollout/`` proper is the drop-in generation backend an RL trainer consumes
(``GenerationBackend`` + flow-SDE log-prob + weight sync). This subpackage holds
the demonstration pieces that close a loop on top of it so the rollout path can be
exercised end-to-end without a real trainer/env:

  * :class:`GRPOTrainer` / :class:`GRPOConfig` — a thin GRPO demo trainer;
  * :class:`ToyReachReward` — a bounded toy reward;
  * :class:`ToyReachEnv` / :class:`VectorEnv` — a toy vectorized env;
  * :class:`RolloutEngine` — couples the backend to a vectorized env.

Kept separate from the product surface so "what a trainer imports" stays small
(cf. verl's narrow ``BaseRollout``; the algorithm/env is the framework's concern).
"""

from .env import ToyReachEnv, VectorEnv
from .grpo import GRPOConfig, GRPOTrainer
from .reward import Reward, ToyReachReward
from .rollout_engine import RolloutEngine

__all__ = [
    "GRPOTrainer",
    "GRPOConfig",
    "Reward",
    "ToyReachReward",
    "ToyReachEnv",
    "VectorEnv",
    "RolloutEngine",
]
