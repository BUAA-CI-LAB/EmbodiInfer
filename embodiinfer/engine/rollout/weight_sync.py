"""Weight synchronization between a learner and rollout policy replicas.

On-policy VLA RL requires pushing freshly updated weights into the rollout
policy after each optimizer step, ideally without a full reload. This module
provides the interface a split-worker trainer calls. Two backends are sketched:

  * ``LocalWeightSync``  — same-process ``load_state_dict`` (used by the demo and
    single-node rollout). Working, in-place, no reallocation.
  * ``NCCLBroadcastSync`` — cross-process broadcast of parameter tensors from the
    learner (rank 0) to rollout workers, the datacenter path. Sketched with the
    call structure; a real deployment fills in the process-group setup.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from ...policies.base import VLAPolicy
from .refit import (
    RefitResult,
    WeightNameMap,
    commit_refit,
    policy_version,
    refit_module,
)


class LocalWeightSync:
    def __init__(self, policy: VLAPolicy):
        self.policy = policy

    @property
    def policy_version(self) -> int:
        """Monotonic version of weights currently visible to rollout inference."""
        return policy_version(self.policy)

    @torch.no_grad()
    def update(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        *,
        name_map: WeightNameMap | None = None,
        version: int | None = None,
    ) -> RefitResult:
        """Copy new weights in place (preserves dtype/device of the live model)."""
        return refit_module(
            self.policy,
            state_dict,
            strict=strict,
            name_map=name_map,
            version=version,
        )


class NCCLBroadcastSync:
    """Cross-process weight broadcast (datacenter rollout). Sketch."""

    def __init__(self, policy: VLAPolicy, src_rank: int = 0, group=None):
        self.policy = policy
        self.src_rank = src_rank
        self.group = group

    @property
    def policy_version(self) -> int:
        """Monotonic version advanced after every complete broadcast."""
        return policy_version(self.policy)

    @torch.no_grad()
    def update(self, param_names: Iterable[str] | None = None) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed not initialized")
        params = dict(self.policy.named_parameters())
        for name in param_names or params.keys():
            dist.broadcast(params[name].data, src=self.src_rank, group=self.group)
        commit_refit(self.policy)


# Proposal 0004 uses this shorter name; retain the original public spelling for
# compatibility while exposing the canonical contract name too.
NCCLWeightSync = NCCLBroadcastSync
