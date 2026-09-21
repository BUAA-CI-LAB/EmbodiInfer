"""The VLA policy contract the engine executes against.

Every VLA is factored into two stages the engine schedules:

    1. ``encode_prefix`` — compute-bound, run ONCE per observation. Encodes
       images + language + state into a multimodal prefix and returns the
       cross-attention K/V cache (``PrefixState``). Model-agnostic.
    2. an :class:`~embodiinfer.policies.decoder.ActionDecoder` (``policy.decoder``) that
       turns that prefix into an action chunk. Implementations, one per paradigm: a
       ``FlowDecoder`` (N-step flow-matching denoise loop — pi0.5, GR00T, LingBot-VLA),
       a ``ParallelDecoder`` (single forward pass — OpenVLA-OFT), and a
       ``CosmosDiffusionDecoder`` (EDM/rectified-flow diffusion sampler — Cosmos WAM).

Keeping encode/decode separate is what lets the engine (a) reuse the prefix KV
and (b) CUDA-graph-capture the static-shape decode. Flow models subclass
:class:`FlowVLAPolicy` (which carries ``denoise_step``/``flow_schedule``); the
slim :class:`VLAPolicy` base carries nothing flow-specific. Concrete models
(pi0.5, GR00T, OpenVLA-OFT, Cosmos) implement these via adapters.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from ..models.schedulers.flow import euler_step
from ..types import BatchedObservation, Observation, collate, pad_batch
from .config import VLAPolicyConfig

if TYPE_CHECKING:
    from ..engine.rollout.refit import RefitResult, WeightNameMap
    from .decoder import ActionDecoder


class PolicyBatch(Protocol):
    """A collated, ready-to-run batch. Its concrete layout is policy-specific
    (``BatchedObservation`` for mock, ``Pi05Batch`` for pi0.5); the engine only
    needs the batch size, the request ids, and a device/dtype move."""

    request_ids: list[str]

    @property
    def batch_size(self) -> int: ...

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> PolicyBatch: ...


class MemoryState(Protocol):
    """Opaque policy state committed across environment steps.

    The engine owns the lifecycle but never inspects the concrete layout. A
    recurrent policy must treat a checked-out committed state as immutable and
    return a distinct next state after a successful decode.
    """

    @property
    def seq_len(self) -> int: ...

    def to(self, device: torch.device | str) -> MemoryState: ...

    def expand(self, num_samples: int) -> MemoryState:
        """Return branch-safe candidate state for a shared recurrent prefix."""
        if num_samples != 1:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement branch-safe memory expansion"
            )
        return self

    def compact(self, keep: Sequence[int] | torch.Tensor | None = None) -> MemoryState:
        """Optionally compact recurrent state or select active batch rows."""
        del keep
        return self


class PrefixState(Protocol):
    """The multimodal prefix the engine reuses across denoising steps.

    Computed once by ``encode_prefix`` and consumed unchanged by every
    ``denoise_step``. Concrete layouts differ per model — dense per-layer K/V for
    :class:`MockFlowVLA`, a HuggingFace ``past_key_values`` + pad mask for pi0.5 —
    so the engine depends only on this small interface, never on a layout.
    """

    batch_size: int

    def to(self, device: torch.device | str) -> PrefixState:
        """Move the cached tensors to ``device``."""
        ...

    def expand(self, num_samples: int) -> PrefixState:
        """Broadcast to ``batch_size * num_samples`` candidates (best-of-N)."""
        ...


@dataclass
class DenseKVPrefix:
    """Per-expert-layer cross-attention K/V cache (dense, static shape).

    ``kv`` is a list (one entry per expert layer) of ``(key, value)`` tensors of
    shape ``[B, num_heads, prefix_len, head_dim]``. Computed once and reused
    unchanged across denoising steps — the VLA analogue of a prefill KV cache,
    except it is *static* (the observation does not change within one action
    prediction). Used by ``MockFlowVLA``; pi0.5 has its own ``Pi05Prefix``.
    """

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    batch_size: int

    def to(self, device: torch.device | str) -> DenseKVPrefix:
        return DenseKVPrefix(
            kv=[(k.to(device), v.to(device)) for k, v in self.kv],
            batch_size=self.batch_size,
        )

    def expand(self, num_samples: int) -> DenseKVPrefix:
        if num_samples == 1:
            return self
        # repeat_interleave keeps one observation's candidates contiguous
        # ([obs0_s0, obs0_s1, ..., obs1_s0, ...]).
        kv = [
            (k.repeat_interleave(num_samples, dim=0), v.repeat_interleave(num_samples, dim=0))
            for k, v in self.kv
        ]
        return DenseKVPrefix(kv=kv, batch_size=self.batch_size * num_samples)


class VLAPolicy(abc.ABC, torch.nn.Module):
    """Abstract VLA policy: the slim, model-agnostic contract (``encode_prefix`` + a
    ``decoder``).

    Carries nothing paradigm-specific. Flow models subclass :class:`FlowVLAPolicy` (which
    adds ``denoise_step``/``flow_schedule``); a single-pass policy (OpenVLA-OFT) or a
    diffusion WAM (Cosmos) subclasses this base directly."""

    def __init__(self, config: VLAPolicyConfig):
        super().__init__()
        self._config = config

    @property
    def config(self) -> VLAPolicyConfig:
        return self._config

    @property
    def execution_dtype(self) -> torch.dtype:
        """Dtype of observations and decoder state; mixed policies may override it."""
        return next(self.parameters()).dtype

    @property
    def policy_version(self) -> int:
        """Monotonic version of weights currently visible to inference."""
        from ..engine.rollout.refit import policy_version

        return policy_version(self)

    def refit_state_dict(self, *, keep_vars: bool = False) -> dict[str, torch.Tensor]:
        """Return live policy tensors for a framework-managed zero-copy refit."""
        from ..engine.rollout.refit import refit_state_dict

        return refit_state_dict(self, keep_vars=keep_vars)

    def refit(
        self,
        weights: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
        name_map: WeightNameMap | None = None,
        version: int | None = None,
    ) -> RefitResult:
        """Copy learner weights in place using an optional external name map."""
        from ..engine.rollout.refit import refit_module

        return refit_module(
            self,
            weights,
            strict=strict,
            name_map=name_map,
            version=version,
        )

    def commit_refit(self, *, version: int | None = None) -> int:
        """Publish an externally completed zero-copy refit."""
        from ..engine.rollout.refit import commit_refit

        return commit_refit(self, version=version)

    def on_refit(self, version: int) -> None:
        """Refresh runtime state before ``version`` becomes visible."""
        del version

    @property
    def is_recurrent(self) -> bool:
        """Whether the policy commits opaque state across ``EngineCore.execute`` calls."""
        return False

    @property
    def manages_cuda_graph(self) -> bool:
        """Whether this policy configures a graph-backed native runtime itself."""
        return False

    def configure_runtime(self, *, use_cuda_graph: bool) -> None:
        if use_cuda_graph:
            raise NotImplementedError(f"{type(self).__name__} does not manage CUDA graphs")

    # ---- prefix encode (compute-bound, run once) ----------------------------
    @abc.abstractmethod
    def encode_prefix(self, batch: BatchedObservation, memory: MemoryState | None = None) -> PrefixState:
        """Encode the multimodal prefix and return the cross-attn KV cache."""

    # ---- action decode strategy ---------------------------------------------
    @property
    @abc.abstractmethod
    def decoder(self) -> ActionDecoder:
        """The strategy that turns a prefix into an action chunk.

        Flow policies return a ``FlowDecoder`` (N-step denoise loop); a single-pass
        policy (OpenVLA-OFT) returns a ``ParallelDecoder``. The engine and the RL
        rollout surface depend only on this — never on whether the model denoises."""

    # ---- batch construction (default: the dense BatchedObservation layout) ---
    def collate(self, observations: list[Observation], request_ids: list[str]) -> PolicyBatch:
        """Stack single observations into a batched, ready-to-run ``PolicyBatch``.

        Default builds a ``BatchedObservation``; adapters whose model consumes a
        different layout (e.g. pi0.5's per-camera list) override this.
        """
        return collate(observations, request_ids)

    def pad(self, batch: PolicyBatch, target_batch_size: int) -> PolicyBatch:
        """Pad a batch up to ``target_batch_size`` (for CUDA-graph bucket reuse)."""
        return pad_batch(batch, target_batch_size)

    # ---- CUDA-graph capability (opt-in) -------------------------------------
    @property
    def supports_cuda_graph(self) -> bool:
        """Whether the decode step can be captured as a static-shape CUDA graph.

        A capable policy also implements ``allocate_static_prefix`` and
        ``copy_prefix_into``. Default False: the engine runs the decode eagerly.
        pi0.5 leaves this False (its HF cache is not a static buffer, and at
        4.14B fp32 the loop is compute-bound so graph capture buys little).
        """
        return False

    @property
    def cuda_graph_kind(self) -> str:
        """Which CUDA-graph the engine's ``GraphManager`` captures for this policy.

        One of ``"flow"`` / ``"single_forward"`` / ``"diffusion_step"`` — declared by the
        policy as metadata, so the engine chooses the graph class without sniffing methods
        (cf. vllm-omni's "model declares structure"). Only consulted when
        :attr:`supports_cuda_graph` is True."""
        raise NotImplementedError(
            f"{type(self).__name__}.supports_cuda_graph is True but cuda_graph_kind is not declared"
        )

    def allocate_static_prefix(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> PrefixState:
        """Allocate a zero-filled prefix with static shapes for graph capture."""
        raise NotImplementedError(
            f"{type(self).__name__}.supports_cuda_graph is True but allocate_static_prefix is not implemented"
        )

    def cuda_graph_variant(self, prefix: PrefixState) -> object | None:
        """Return policy metadata that changes the captured operator graph."""
        del prefix
        return None

    def allocate_static_prefix_for_variant(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        variant: object | None,
    ) -> PrefixState:
        """Allocate a graph prefix; policies may specialize it by variant."""
        del variant
        return self.allocate_static_prefix(batch_size, device, dtype)

    def allocate_static_prefix_from_live(
        self,
        prefix: PrefixState,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        variant: object | None,
    ) -> PrefixState:
        """Create a graph prefix from a live prefix; simple policies may reuse allocation."""
        del prefix
        return self.allocate_static_prefix_for_variant(batch_size, device, dtype, variant)

    def copy_prefix_into(self, dst: PrefixState, src: PrefixState) -> None:
        """Copy ``src`` prefix tensors into the pre-allocated static ``dst`` in place."""
        raise NotImplementedError(
            f"{type(self).__name__}.supports_cuda_graph is True but copy_prefix_into is not implemented"
        )


class FlowVLAPolicy(VLAPolicy):
    """A flow-matching VLA: the action chunk is produced by an N-step Euler loop over a
    velocity field cross-attending the cached prefix.

    pi0.5 and GR00T are flow policies. This layer holds everything flow-specific
    (``denoise_step``, ``flow_schedule``, the reference sampler, the noise seed) so the
    slim :class:`VLAPolicy` base — and a non-flow policy like OpenVLA-OFT — carry none of
    it. The generation itself lives in :class:`~embodiinfer.policies.decoder.FlowDecoder`.
    """

    @property
    def cuda_graph_kind(self) -> str:
        return "flow"

    # ---- the flow denoise step (run N times) --------------------------------
    @abc.abstractmethod
    def denoise_step(self, x_t: torch.Tensor, t: torch.Tensor, prefix: PrefixState) -> torch.Tensor:
        """Return the flow-matching velocity field v(x_t, t | prefix).

        Args:
            x_t: [B, horizon, action_dim] current noisy action chunk.
            t:   [B] flow time in [0, 1].
            prefix: cached prefix from ``encode_prefix``.
        Returns:
            [B, horizon, action_dim] velocity.
        """

    # ---- flow-matching integration schedule ---------------------------------
    def flow_schedule(self, num_steps: int) -> list[tuple[float, float]]:
        """Return the list of ``(t_value, dt)`` steps for the Euler integrator.

        Default: ascending t in [0, 1) with dt = +1/N (x0 noise -> x1 action).
        Adapters override this: pi0.5 integrates t from 1 -> 0 with dt = -1/N.
        The engine loop is policy-agnostic and simply follows this schedule.
        """
        dt = 1.0 / num_steps
        return [(i * dt, dt) for i in range(num_steps)]

    def flow_state_shape(self, batch_size: int) -> tuple[int, int, int]:
        """Shape of the integrated state, which may include padded action columns."""
        return batch_size, self.config.action_horizon, self.config.action_dim

    def finalize_actions(self, state: torch.Tensor, prefix: PrefixState) -> torch.Tensor:
        """Convert the final flow state to public actions after integration/graph replay."""
        del prefix
        return state

    def flow_logprob_mask(self, state: torch.Tensor) -> torch.Tensor | None:
        """Select scored state elements with a broadcastable mask; None scores all.

        This affects transition likelihoods only. Sampling and stored trajectories
        retain the full internal state, including any unscored padding columns.
        """
        del state
        return None

    @property
    def decoder(self) -> ActionDecoder:
        d = self.__dict__.get("_decoder")
        if d is None:
            from .decoder import FlowDecoder

            d = FlowDecoder(self)
            self._decoder = d
        return d

    # ---- reference sampler (engine overrides for batching / CUDA graph) ------
    @torch.no_grad()
    def sample_actions(
        self,
        batch: BatchedObservation,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default flow-matching sampler: encode prefix once, integrate N steps.

        This is the *semantic reference*. The engine reproduces the exact same
        computation but with prefix reuse + CUDA-graph replay for the loop.
        """
        cfg = self.config
        num_steps = num_steps or cfg.default_num_steps
        p = next(self.parameters())
        device, dtype = p.device, self.execution_dtype
        B = batch.batch_size
        prefix = self.encode_prefix(batch)
        x = self.new_noise(B, generator=generator) if x0 is None else x0
        for t_val, dt in self.flow_schedule(num_steps):
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.denoise_step(x, t, prefix)
            x = euler_step(x, v, dt)
        return self.finalize_actions(x, prefix)

    def new_noise(
        self, batch_size: int, generator: torch.Generator | None = None, noise_scale: float = 1.0
    ) -> torch.Tensor:
        p = next(self.parameters())
        return noise_scale * torch.randn(
            *self.flow_state_shape(batch_size),
            device=p.device,
            dtype=self.execution_dtype,
            generator=generator,
        )
