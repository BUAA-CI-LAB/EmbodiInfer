"""The action-decode strategy: how a policy turns an encoded prefix into actions.

embodiinfer factors every VLA into ``encode_prefix`` (shared, model-agnostic prefill) plus
an :class:`ActionDecoder` that produces the action chunk from that prefix. The engine
depends only on this interface — never on *how* the chunk is produced.

Two capability tiers keep the abstraction honest across paradigms:

  * :class:`ActionDecoder` — the *serving* contract (``init_state`` + ``produce_chunk``)
    that every decoder implements.
  * :class:`RLDecoder` — adds the on-policy RL rollout surface (behavior + differentiable
    recompute log-prob). Only decoders that are RL rollout backends implement it.

The concrete decoders, one per paradigm:

  * :class:`FlowDecoder` (``RLDecoder``) — the N-step flow-matching Euler loop (pi0.5,
    GR00T, LingBot-VLA); wraps a :class:`~embodiinfer.policies.base.FlowVLAPolicy`'s
    ``denoise_step``/``flow_schedule`` + the flow-SDE rollout log-prob.
  * :class:`ParallelDecoder` (``RLDecoder``) — a single forward pass with a categorical
    action-token head (OpenVLA-OFT); no time schedule, no noise seed.
  * ``CosmosDiffusionDecoder`` (plain ``ActionDecoder``, in ``policies/cosmos/``) — an
    EDM/rectified-flow diffusion sampler (Cosmos WAM); a planning decoder, *not* an RL
    rollout backend.
  * :class:`AutoregressiveDecoder` (plain ``ActionDecoder``) — eager token-by-token
    generation with policy-owned recurrent memory (ActiveVLN). Recurrent candidate
    branching is a separate capability and is not implied by token-level log-probs.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..models.schedulers.flow import euler_step
from ..types import DecodeTrace

if TYPE_CHECKING:
    from .base import FlowVLAPolicy, MemoryState, PrefixState


@dataclass
class DecodeResult:
    """Structured output of a deterministic decoder invocation."""

    actions: torch.Tensor
    behavior_logprob: torch.Tensor | None = None
    recompute_state: object | None = None
    next_memory: MemoryState | None = None
    traces: list[DecodeTrace] | None = None


class ActionDecoder(abc.ABC):
    """Turns an encoded prefix into an action chunk — the serving contract the engine drives.

    Every decoder implements this pair: stage a decode state, then produce the chunk. The
    on-policy RL rollout surface (behavior/recompute log-prob) is a separate capability
    declared by :class:`RLDecoder`, so a planning-only decoder is not forced to fake it.
    """

    @abc.abstractmethod
    def init_state(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor | None:
        """The decode state the engine stages before generation.

        Flow/diffusion decoders return the initial noise; a single-pass or autoregressive
        decoder returns ``None`` (nothing is seeded before the forward).
        """

    def state_shape(self, batch_size: int) -> tuple[int, ...] | None:
        """Return the static staged-state shape, when the decoder has one."""
        del batch_size
        return None

    def decode(
        self,
        state: torch.Tensor | None,
        prefix: PrefixState,
        num_steps: int,
        bucket: int,
        graphs,
        *,
        generator: torch.Generator | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DecodeResult:
        """Run deterministic generation and return a structured result.

        Existing decoders inherit this behavior-preserving wrapper. Stateful or
        autoregressive decoders override it to attach final memory and traces.
        """
        del generator, cancelled
        return DecodeResult(actions=self.produce_chunk(state, prefix, num_steps, bucket, graphs))

    @abc.abstractmethod
    def produce_chunk(
        self, state: torch.Tensor | None, prefix: PrefixState, num_steps: int, bucket: int, graphs
    ) -> torch.Tensor:
        """Deterministic generation: prefix (+ staged state) -> actions ``[bucket, H, A]``.

        ``graphs`` is the engine's :class:`~embodiinfer.engine.graph.GraphManager` (or ``None``
        when CUDA graphs are disabled); the decoder uses it to replay a captured loop.
        """


class RLDecoder(ActionDecoder):
    """An :class:`ActionDecoder` that also exposes the on-policy RL rollout surface.

    Flow (:class:`FlowDecoder`) and single-pass categorical (:class:`ParallelDecoder`)
    decoders implement this. A planning WAM and a recurrent decoder without candidate
    memory branching remain plain :class:`ActionDecoder` implementations.
    """

    @abc.abstractmethod
    def sample_with_logprob(
        self, prefix: PrefixState, num_steps: int, sigma, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Stochastic rollout for RL.

        Returns ``(actions [B, H, A], behavior_logprob [B], recompute_state)`` where
        ``recompute_state`` is the opaque record :meth:`recompute_logprob` re-scores
        under current parameters. ``prefix`` is already expanded for the group.
        """

    @abc.abstractmethod
    def recompute_logprob(
        self, prefix: PrefixState, recompute_state: object, num_steps: int, sigma
    ) -> torch.Tensor:
        """Differentiable re-score of a sampled rollout -> ``log pi_theta`` ``[B]``."""


class AutoregressiveDecoder(ActionDecoder):
    """Base class for eager token-by-token action generation.

    Concrete policies own token sampling, stop criteria, cache mutation, and trace
    construction by overriding :meth:`decode`. This serving capability does not imply
    branch-aware recurrent rollout; concrete policies may expose direct parity helpers
    without becoming :class:`RLDecoder` instances.
    """

    def init_state(self, batch_size: int, generator: torch.Generator | None = None) -> None:
        del batch_size, generator
        return None

    @abc.abstractmethod
    def decode(
        self,
        state: torch.Tensor | None,
        prefix: PrefixState,
        num_steps: int,
        bucket: int,
        graphs,
        *,
        generator: torch.Generator | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DecodeResult:
        """Generate tokens eagerly and return the final memory plus trace."""

    def produce_chunk(
        self, state: torch.Tensor | None, prefix: PrefixState, num_steps: int, bucket: int, graphs
    ) -> torch.Tensor:
        return self.decode(state, prefix, num_steps, bucket, graphs).actions


class FlowDecoder(RLDecoder):
    """N-step flow-matching Euler loop + flow-SDE rollout log-prob (pi0.5, GR00T).

    Policies own the internal state shape, output conversion, and optional
    likelihood mask. Integration and stochastic trajectory scoring are shared.
    """

    def __init__(self, policy: FlowVLAPolicy):
        self.policy = policy

    def state_shape(self, batch_size: int) -> tuple[int, int, int]:
        """Expose the policy's internal flow shape for static graph buffers."""
        return self.policy.flow_state_shape(batch_size)

    def init_state(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.policy.new_noise(batch_size, generator=generator)

    def produce_chunk(
        self, state: torch.Tensor | None, prefix: PrefixState, num_steps: int, bucket: int, graphs
    ) -> torch.Tensor:
        """Integrate the flow field, then restore the policy's public action representation."""
        return self.policy.finalize_actions(self.integrate(state, prefix, num_steps, bucket, graphs), prefix)

    def integrate(
        self, state: torch.Tensor | None, prefix: PrefixState, num_steps: int, bucket: int, graphs
    ) -> torch.Tensor:
        """Run all flow steps and return model-space actions before output transforms.

        This is the same eager/graph execution used by ``produce_chunk``. Callers
        measuring model execution can time it separately from ``finalize_actions``.
        """
        if state is None:
            raise ValueError("Flow decoding requires an initial state")
        x = state
        policy = self.policy
        device, dtype = x.device, x.dtype
        graph_variant = policy.cuda_graph_variant(prefix)
        graph = None
        if graphs is not None:
            graph = graphs.get(bucket, num_steps, graph_variant, prefix)
            graph.set_prefix(prefix)
        if graphs is not None and graphs.full_loop:
            x = graph.run(x)
        else:
            for t_val, dt in policy.flow_schedule(num_steps):
                t = torch.full((bucket,), t_val, device=device, dtype=dtype)
                velocity = graph.run(x, t) if graph is not None else policy.denoise_step(x, t, prefix)
                x = euler_step(x, velocity, dt)
        return x

    def sample_with_logprob(
        self, prefix: PrefixState, num_steps: int, sigma, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ..engine.rollout.logprob import flow_sample_with_logprob

        x0 = self.policy.new_noise(prefix.batch_size, generator=generator)
        actions, logprob, trajectory = flow_sample_with_logprob(
            self.policy,
            prefix,
            x0,
            num_steps,
            sigma,
            return_trajectory=True,
            generator=generator,
            logprob_mask=self.policy.flow_logprob_mask(x0),
        )
        return self.policy.finalize_actions(actions, prefix), logprob, trajectory

    def recompute_logprob(
        self, prefix: PrefixState, recompute_state: object, num_steps: int, sigma
    ) -> torch.Tensor:
        from ..engine.rollout.logprob import flow_logprob_recompute

        if not isinstance(recompute_state, torch.Tensor) or recompute_state.ndim != 4:
            raise ValueError("Flow recompute state must be a [B, N+1, H, A] tensor trajectory")
        return flow_logprob_recompute(
            self.policy,
            prefix,
            recompute_state,
            num_steps,
            sigma,
            logprob_mask=self.policy.flow_logprob_mask(recompute_state[:, 0]),
        )


class ParallelDecoder(RLDecoder):
    """Single forward pass, discrete action-token head (OpenVLA-OFT)."""

    def __init__(self, policy):
        self.policy = policy

    def init_state(self, batch_size: int, generator: torch.Generator | None = None) -> None:
        del batch_size, generator
        return None

    def produce_chunk(
        self, state: torch.Tensor | None, prefix: PrefixState, num_steps: int, bucket: int, graphs
    ) -> torch.Tensor:
        logits = self.policy.decode_action_logits(prefix, graphs=graphs, bucket=bucket)
        idxs, _ = self.policy.head.sample(logits, do_sample=False)
        return self.policy.head.tokens_to_actions(idxs)

    def sample_with_logprob(
        self, prefix: PrefixState, num_steps: int, sigma, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_steps, sigma
        temperature, top_k = self.policy.sample_temperature, self.policy.sample_top_k
        logits = self.policy.decode_action_logits(prefix)
        idxs, logprob = self.policy.head.sample(
            logits, do_sample=True, temperature=temperature, top_k=top_k, generator=generator
        )
        actions = self.policy.head.tokens_to_actions(idxs)
        return actions, logprob, idxs

    def recompute_logprob(
        self, prefix: PrefixState, recompute_state: object, num_steps: int, sigma
    ) -> torch.Tensor:
        del num_steps, sigma
        temperature, top_k = self.policy.sample_temperature, self.policy.sample_top_k
        logits = self.policy.decode_action_logits(prefix)
        return self.policy.head.recompute_logprob(
            logits, recompute_state, temperature=temperature, top_k=top_k
        )
