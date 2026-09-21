"""Core data types for the embodiinfer engine.

The engine operates on *flow VLA* requests: an observation (images + language +
proprio state) is turned into an action *chunk* by encoding a multimodal prefix
once and then integrating a flow-matching velocity field for a fixed number of
denoising steps. These dataclasses describe that request/response contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def _as_tensor(x: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    if dtype is not None:
        t = t.to(dtype)
    return t


@dataclass
class Observation:
    """A single (unbatched) VLA observation.

    Tensors are stored on CPU; the engine moves them to the device at collate
    time. ``instruction_tokens`` is assumed already tokenized and padded to the
    policy's ``max_lang_len`` so that the multimodal prefix has a static shape
    (a precondition for CUDA-graph capture of the denoising loop).
    """

    images: torch.Tensor  # [num_cam, 3, H, W], float in [0, 1]
    state: torch.Tensor  # [state_dim]
    instruction_tokens: torch.Tensor  # [max_lang_len], long
    instruction: str | None = None
    env_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.images = _as_tensor(self.images, torch.float32)
        self.state = _as_tensor(self.state, torch.float32)
        self.instruction_tokens = _as_tensor(self.instruction_tokens, torch.long)
        if self.images.ndim == 3:  # [3, H, W] -> single camera
            self.images = self.images.unsqueeze(0)


@dataclass
class SampleParams:
    """Per-request sampling controls for the flow-matching sampler."""

    num_steps: int = 10  # solver / denoising steps
    num_samples: int = 1  # best-of-N candidates (RL planning); 1 = greedy
    noise_scale: float = 1.0  # scale of the initial x0 ~ N(0, noise_scale^2 I)
    seed: int | None = None
    temperature: float = 1.0  # scales injected noise for stochastic sampling


@dataclass(frozen=True)
class SessionKey:
    """Stable identity for policy state that persists across environment steps.

    ``env_id`` identifies the environment worker, ``episode_id`` prevents a reused
    worker slot from inheriting an earlier episode's state, and ``rollout_id``
    separates branches sampled from the same episode start.
    """

    env_id: str | int
    episode_id: str | int
    rollout_id: str | int = 0


@dataclass
class Request:
    """An inference request tracked by the scheduler."""

    request_id: str
    observation: Observation
    params: SampleParams = field(default_factory=SampleParams)
    arrival_ns: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _validate_policy_version(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"policy_version must be a non-negative integer; got {value!r}")


def _validate_timing(timing: dict[str, float], *, complete: bool = False) -> None:
    required = {"e2e_ms"}
    if complete:
        required.update({"prefill_ms", "decode_ms"})
    missing = sorted(required.difference(timing))
    if missing:
        raise ValueError(f"timing is missing required fields: {missing}")
    for name, value in timing.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"timing[{name!r}] must be a non-negative number; got {value!r}")


@dataclass
class DecodeTrace:
    """Decoder-native details attached to one returned action chunk."""

    token_ids: torch.Tensor
    token_logprobs: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    text: str | None = None
    parsed_actions: Any = None
    stop_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    policy_version: int = 0
    timing: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_policy_version(self.policy_version)
        if self.timing:
            _validate_timing(self.timing)


@dataclass
class ActionChunk:
    """The engine's response: a chunk of future actions."""

    request_id: str
    actions: torch.Tensor  # [horizon, action_dim]
    logprob: torch.Tensor | None = None  # decoder-native granularity (scalar or token-level)
    value: float | None = None  # best-of-N selected candidate value
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    trace: DecodeTrace | None = None
    policy_version: int = 0
    timing: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_policy_version(self.policy_version)
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative; got {self.latency_ms!r}")
        if self.timing:
            _validate_timing(self.timing)


@dataclass
class TrajectoryRecord:
    """Canonical audit record for one environment action.

    The engine supplies tokens, behavior log-probabilities, parsed actions,
    policy version and timing.  An environment adapter subsequently supplies
    ``executed_action``, ``reward`` and ``done``.  Those adapter-owned fields are
    explicit (rather than hidden in ``meta``) and may be ``None`` only while a
    record is in flight; :meth:`validate_complete` enforces the persisted-record
    contract.
    """

    env_id: str | int
    episode_id: str | int
    step_idx: int
    policy_version: int
    seed: int
    raw_tokens: torch.Tensor
    token_logprobs: torch.Tensor | None
    parsed_action: Any
    executed_action: Any | None
    reward: float | None
    done: bool | None
    timing: dict[str, float]
    recompute_state: Any | None = None

    def __post_init__(self) -> None:
        if self.env_id is None or self.episode_id is None:
            raise ValueError("env_id and episode_id are required")
        if isinstance(self.step_idx, bool) or not isinstance(self.step_idx, int) or self.step_idx < 0:
            raise ValueError(f"step_idx must be a non-negative integer; got {self.step_idx!r}")
        _validate_policy_version(self.policy_version)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer; got {self.seed!r}")

        self.raw_tokens = _as_tensor(self.raw_tokens, torch.long)
        if self.raw_tokens.ndim != 1:
            raise ValueError(f"raw_tokens must be 1-D; got shape {tuple(self.raw_tokens.shape)}")
        if self.token_logprobs is not None:
            self.token_logprobs = _as_tensor(self.token_logprobs, torch.float32)
            if self.token_logprobs.ndim != 1:
                raise ValueError(f"token_logprobs must be 1-D; got shape {tuple(self.token_logprobs.shape)}")
            if self.token_logprobs.numel() != self.raw_tokens.numel():
                raise ValueError("token_logprobs must align one-to-one with raw_tokens")
        elif self.recompute_state is None:
            raise ValueError("token_logprobs or recompute_state is required")
        if self.parsed_action is None:
            raise ValueError("parsed_action is required")
        _validate_timing(self.timing)
        if self.reward is not None and (
            isinstance(self.reward, bool) or not isinstance(self.reward, (int, float))
        ):
            raise ValueError(f"reward must be numeric or None; got {self.reward!r}")
        if self.done is not None and not isinstance(self.done, bool):
            raise ValueError(f"done must be bool or None; got {self.done!r}")

    def validate_complete(self) -> None:
        """Validate fields required before a record is persisted or trained on."""
        missing = [
            name
            for name, value in (
                ("executed_action", self.executed_action),
                ("reward", self.reward),
                ("done", self.done),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"trajectory record is incomplete: missing {missing}")
        _validate_timing(self.timing, complete=True)


@dataclass
class BatchedObservation:
    """A collated batch, ready for a single forward pass on device."""

    images: torch.Tensor  # [B, num_cam, 3, H, W]
    state: torch.Tensor  # [B, state_dim]
    instruction_tokens: torch.Tensor  # [B, max_lang_len]
    request_ids: list[str]
    env_ids: list[int | None]

    @property
    def batch_size(self) -> int:
        return self.images.shape[0]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> BatchedObservation:
        img = self.images.to(device, non_blocking=True)
        state = self.state.to(device, non_blocking=True)
        if dtype is not None:  # float inputs cast to the model dtype; tokens stay long
            img = img.to(dtype)
            state = state.to(dtype)
        return BatchedObservation(
            images=img,
            state=state,
            instruction_tokens=self.instruction_tokens.to(device, non_blocking=True),
            request_ids=self.request_ids,
            env_ids=self.env_ids,
        )


def validate_observation(obs: Observation, config: Any = None) -> None:
    """Check an observation's tensor ranks (and, if ``config`` exposes them, its
    camera/state dimensions), raising :class:`ObservationError` with the expected
    vs received shape rather than letting a wrong shape fail deep in the forward.

    ``config`` is optional and duck-typed: only the ``num_cameras`` / ``state_dim``
    / ``max_lang_len`` fields that exist are checked, so it works for any policy.
    """
    from .exceptions import ObservationError  # local import to keep types.py dependency-free

    if obs.images.ndim != 4:
        raise ObservationError(f"images must be [num_cameras, 3, H, W]; got shape {tuple(obs.images.shape)}")
    if obs.state.ndim != 1:
        raise ObservationError(f"state must be 1-D [state_dim]; got shape {tuple(obs.state.shape)}")
    if obs.instruction_tokens.ndim != 1:
        raise ObservationError(
            f"instruction_tokens must be 1-D [max_lang_len]; got shape {tuple(obs.instruction_tokens.shape)}"
        )
    if config is None:
        return
    checks = [
        ("num_cameras", obs.images.shape[0], "cameras"),
        ("state_dim", obs.state.shape[0], "state dim"),
        ("max_lang_len", obs.instruction_tokens.shape[0], "language tokens"),
    ]
    for field_name, got, label in checks:
        expected = getattr(config, field_name, None)
        if expected is not None and got != expected:
            raise ObservationError(f"expected {expected} {label}; got {got}")


def collate(observations: list[Observation], request_ids: list[str]) -> BatchedObservation:
    """Stack a list of single observations into a batch."""
    return BatchedObservation(
        images=torch.stack([o.images for o in observations], dim=0),
        state=torch.stack([o.state for o in observations], dim=0),
        instruction_tokens=torch.stack([o.instruction_tokens for o in observations], dim=0),
        request_ids=list(request_ids),
        env_ids=[o.env_id for o in observations],
    )


def pad_batch(batch: BatchedObservation, target_batch_size: int) -> BatchedObservation:
    """Pad a batch up to ``target_batch_size`` by repeating the last observation.

    The pad rows are computed but discarded (only the real rows are returned by
    the engine); repeating keeps shapes valid and reuses a captured CUDA graph.
    """
    b = batch.batch_size
    if target_batch_size == b:
        return batch
    pad = target_batch_size - b

    def rep(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0)

    return BatchedObservation(
        images=rep(batch.images),
        state=rep(batch.state),
        instruction_tokens=rep(batch.instruction_tokens),
        request_ids=batch.request_ids + [f"__pad_{i}" for i in range(pad)],
        env_ids=batch.env_ids + [None] * pad,
    )
