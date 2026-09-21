"""Rollout / generation surface — the drop-in backend for an RL trainer.

This is the *product* surface an RLinf/verl-style trainer consumes: generate
actions + log-probability, best-of-N, and in-place weight sync. Kept deliberately
narrow (cf. verl's ``BaseRollout``); the trainer, algorithm and env are out of
scope for vvla and live under :mod:`embodiinfer.engine.rollout.demo` (toy GRPO loop + env) for
demonstration only.
"""

from .generation_backend import GenerationBackend, RolloutSamples
from .logprob import flow_logprob_recompute, flow_sample_with_logprob
from .refit import (
    RefitResult,
    WeightNameMap,
    commit_refit,
    policy_version,
    refit_module,
    refit_state_dict,
)
from .weight_sync import LocalWeightSync, NCCLBroadcastSync, NCCLWeightSync

__all__ = [
    "GenerationBackend",
    "RolloutSamples",
    "flow_sample_with_logprob",
    "flow_logprob_recompute",
    "RefitResult",
    "WeightNameMap",
    "refit_module",
    "refit_state_dict",
    "commit_refit",
    "policy_version",
    "LocalWeightSync",
    "NCCLBroadcastSync",
    "NCCLWeightSync",
]
