"""Real-Time Chunking (RTC) guidance for flow-matching action decoders.

RTC (Black, Galliker & Levine, *Real-Time Execution of Action Chunking Flow
Policies*, arXiv 2506.07339) makes an action-chunking flow policy safe to run
*asynchronously*. Under async inference the chunk computed at time ``t`` is not
applied until ``t + Δ``, so the actions the robot is already committed to
executing during ``Δ`` would be contradicted by a freshly sampled chunk. RTC
resolves this by *inpainting*: the actions of the previous chunk that are
guaranteed to execute are held fixed, and the remainder is generated under a
prefix-weighted guidance term.

Two conditioning modes are provided, both taken from the reference:

``soft``
    Every denoising step computes a correction that pulls the *clean-action
    estimate* toward the frozen prefix, weighted per step, and adds it to the
    velocity. This is the paper's method.
``hard``
    The frozen steps are clamped to the prefix value before and after every
    Euler step, and their flow time is pinned to the action end. This is the
    cheaper fixed-prefix variant (``simulated_delay`` in the reference) and
    needs no autograd.

This module is the numeric core only. It depends on nothing but ``torch`` and a
caller-supplied velocity callable, so it ports verbatim across flow policies
(pi0.5, GR00T, LingBot-VLA) and stays testable on CPU with a synthetic velocity
field.

Conventions
-----------
Integration follows the convention declared by EmbodiInfer's pi0.5 schedule:
``dt = -1/num_steps`` with time running ``1 -> 0`` (noise at ``t = 1``, actions
at ``t = 0``). That matches Physical Intelligence's PyTorch stack. The original
JAX reference runs the opposite direction (``t = 0 -> 1``, ``dt = +1/N``); the
expressions below are the algebraically equivalent translation of it, which is
also what the LeRobot port does.

Fidelity note
-------------
The velocity correction is the **Jacobian-vector product**
``(∂x1_t/∂x_t)ᵀ · ((prev - x1_t) * w)`` through the velocity field, exactly as
in Physical Intelligence's JAX ``realtime_action`` (``jax.vjp`` over the
denoiser). The LeRobot PyTorch port computes ``v_t`` *before* marking ``x_t`` as
requiring grad, which silently collapses that VJP to the identity and degrades
the correction to ``err``. This module follows the original: the velocity is
evaluated inside the grad-enabled region with the state as a differentiable
input, so the true VJP is computed. Tests pin the difference explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch

__all__ = [
    "PrefixAttentionSchedule",
    "RTCGuidance",
    "RTCGuidanceConfig",
    "action_ness",
    "build_rtc_guidance",
    "clamp_prefix",
    "flow_noise_end",
    "guidance_strength",
    "hard_prefix_mask",
    "prefix_weights",
    "rtc_guided_velocity",
]


class PrefixAttentionSchedule(str, Enum):
    """How firmly the committed prefix is held across the action horizon.

    The schedule maps ``(inference_delay, execution_horizon)`` onto a per-step
    weight in ``[0, 1]``. With ``start=2, end=6, total=10`` the ``LINEAR``
    schedule yields ``1 1 4/5 3/5 2/5 1/5 0 0 0 0``: steps before ``start`` are
    fully held, steps from ``start`` to ``end`` ramp down, and steps at or
    beyond ``end`` are free.
    """

    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"


@dataclass(frozen=True, slots=True)
class RTCGuidanceConfig:
    """Static RTC settings shared by every request of a serving session.

    Args:
        execution_horizon: Number of steps from the start of the chunk eligible
            to be held by the prefix. Clamped at build time to the length of the
            supplied leftover, because there is nothing to merge beyond it.
        prefix_attention_schedule: Weight shape across the horizon.
        max_guidance_weight: Upper clamp on the guidance strength. The unclamped
            coefficient is singular at the noise end of the flow, so the clamp is
            load-bearing rather than cosmetic.
        hard_prefix: Use the fixed-prefix clamp instead of soft guidance.
    """

    execution_horizon: int
    prefix_attention_schedule: PrefixAttentionSchedule = PrefixAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    hard_prefix: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.execution_horizon, bool) or not isinstance(self.execution_horizon, int):
            raise TypeError("execution_horizon must be an integer")
        if self.execution_horizon < 0:
            raise ValueError("execution_horizon must be non-negative")
        if not isinstance(self.prefix_attention_schedule, PrefixAttentionSchedule):
            object.__setattr__(
                self,
                "prefix_attention_schedule",
                PrefixAttentionSchedule(self.prefix_attention_schedule),
            )
        weight = float(self.max_guidance_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("max_guidance_weight must be a finite positive number")
        object.__setattr__(self, "max_guidance_weight", weight)
        if not isinstance(self.hard_prefix, bool):
            raise TypeError("hard_prefix must be a boolean")


@dataclass(frozen=True, slots=True)
class RTCGuidance:
    """Everything one RTC-conditioned decode needs, shaped for batching.

    Attributes:
        prev_chunk_left_over: ``[B, H, A]`` unexecuted actions of the previous
            chunk, zero-padded on the right when the leftover is shorter than the
            current horizon. ``None`` means "no previous chunk": conditioning is
            skipped so the first inference of an episode is bit-identical to the
            plain path.
        prefix_weights: ``[H]`` per-step prefix firmness from
            :func:`prefix_weights`.
        inference_delay: Latency in control steps, used by the hard-prefix mask.
        max_guidance_weight: Clamp forwarded from :class:`RTCGuidanceConfig`.
        hard_prefix: Whether the decoder must clamp instead of guide.
    """

    prev_chunk_left_over: torch.Tensor | None
    prefix_weights: torch.Tensor
    inference_delay: int = 0
    max_guidance_weight: float = 10.0
    hard_prefix: bool = False
    row_scale: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        """Whether a previous chunk is available to condition against."""
        return self.prev_chunk_left_over is not None

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> RTCGuidance:
        """Move the conditioning tensors to ``device``."""
        left_over = self.prev_chunk_left_over
        return RTCGuidance(
            prev_chunk_left_over=None if left_over is None else left_over.to(device=device, dtype=dtype),
            prefix_weights=self.prefix_weights.to(device=device),
            inference_delay=self.inference_delay,
            max_guidance_weight=self.max_guidance_weight,
            hard_prefix=self.hard_prefix,
            row_scale=None if self.row_scale is None else self.row_scale.to(device=device),
        )


def prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: PrefixAttentionSchedule = PrefixAttentionSchedule.LINEAR,
) -> torch.Tensor:
    """Return the ``[total]`` prefix-firmness weights for one request.

    ``start`` is the inference delay (pushed down to ``end`` when larger, since
    ``end`` takes precedence), ``end`` is the execution horizon (exclusive), and
    ``total`` is the action horizon ``H``.

    Equivalent to the reference ``get_prefix_weights``; the piecewise
    leading-ones / ramp / trailing-zeros construction is used because it is
    exact in float64/float32 at the endpoints, unlike rebuilding the ramp from
    the closed form.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if start < 0 or end < 0:
        raise ValueError("start and end must be non-negative")
    schedule = PrefixAttentionSchedule(schedule)
    start = min(start, end)

    if schedule is PrefixAttentionSchedule.ZEROS:
        weights = torch.zeros(total)
        weights[:start] = 1.0
    elif schedule is PrefixAttentionSchedule.ONES:
        weights = torch.ones(total)
        weights[end:] = 0.0
    else:
        ramp = _linear_ramp(start, end, total)
        if schedule is PrefixAttentionSchedule.EXP:
            ramp = ramp * torch.expm1(ramp).div(math.e - 1)
        weights = _add_leading_ones(_add_trailing_zeros(ramp, total, end), start, total)

    return weights


def _linear_ramp(start: int, end: int, total: int) -> torch.Tensor:
    """The interior ``linspace(1, 0)`` segment of the linear/exp schedules."""
    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        return torch.tensor([])
    return torch.linspace(1, 0, linspace_steps + 2)[1:-1]


def _add_trailing_zeros(weights: torch.Tensor, total: int, end: int) -> torch.Tensor:
    zeros_len = total - end
    if zeros_len <= 0:
        return weights
    return torch.cat([weights, torch.zeros(zeros_len)])


def _add_leading_ones(weights: torch.Tensor, start: int, total: int) -> torch.Tensor:
    ones_len = min(start, total)
    if ones_len <= 0:
        return weights
    return torch.cat([torch.ones(ones_len), weights])


def build_rtc_guidance(
    prev_chunk_left_over: torch.Tensor | None,
    inference_delay: int,
    config: RTCGuidanceConfig,
    *,
    action_horizon: int | None = None,
    action_dim: int | None = None,
) -> RTCGuidance:
    """Shape a raw leftover prefix into conditioning for one decode.

    Mirrors the reference preprocessing: a leftover shorter than the action
    horizon (or narrower in the action dimension) is right-padded with zeros,
    and the execution horizon is clamped to whatever the leftover actually
    covers. Both rules let a partially consumed or truncated previous chunk
    degrade into weaker conditioning instead of raising on shape.

    Args:
        prev_chunk_left_over: ``[B, T_prev, A_prev]`` or ``[T_prev, A_prev]``
            unexecuted actions, or ``None`` on the first inference of an episode.
        inference_delay: Inference latency in control steps (``Δ``).
        config: Static RTC settings.
        action_horizon: Current chunk length ``H``; defaults to the leftover's.
        action_dim: Current action width ``A``; defaults to the leftover's.
    """
    if inference_delay < 0:
        raise ValueError("inference_delay must be non-negative")

    if prev_chunk_left_over is None:
        horizon = 0 if action_horizon is None else int(action_horizon)
        return RTCGuidance(
            prev_chunk_left_over=None,
            prefix_weights=torch.zeros(max(horizon, 0)),
            inference_delay=inference_delay,
            max_guidance_weight=config.max_guidance_weight,
            hard_prefix=config.hard_prefix,
        )

    left_over = prev_chunk_left_over
    if left_over.ndim == 2:
        left_over = left_over.unsqueeze(0)
    if left_over.ndim != 3:
        raise ValueError(f"prev_chunk_left_over must be 2D or 3D, got shape {tuple(left_over.shape)}")

    batch_size, prefix_len, prefix_dim = left_over.shape
    horizon = int(action_horizon) if action_horizon is not None else prefix_len
    width = int(action_dim) if action_dim is not None else prefix_dim
    if horizon <= 0 or width <= 0:
        raise ValueError("action_horizon and action_dim must be positive")

    execution_horizon = config.execution_horizon
    if execution_horizon > prefix_len:
        execution_horizon = prefix_len

    if prefix_len < horizon or prefix_dim < width:
        padded = torch.zeros(batch_size, horizon, width, device=left_over.device, dtype=left_over.dtype)
        padded[:, :prefix_len, :prefix_dim] = left_over
        left_over = padded

    weights = prefix_weights(inference_delay, execution_horizon, horizon, config.prefix_attention_schedule)
    return RTCGuidance(
        prev_chunk_left_over=left_over,
        prefix_weights=weights,
        inference_delay=inference_delay,
        max_guidance_weight=config.max_guidance_weight,
        hard_prefix=config.hard_prefix,
    )


def flow_noise_end(schedule: list[tuple[float, float]]) -> float:
    """Return the flow time at which the schedule's sample is pure noise.

    EmbodiInfer supports both flow directions: pi0.5 and the openpi-derived stack
    integrate ``t: 1 -> 0`` with ``dt < 0`` (noise at ``t = 1``), while GR00T and
    the mock policy integrate ``t: 0 -> 1`` with ``dt > 0`` (noise at ``t = 0``).
    RTC's arithmetic is written in the reference's "action-ness" coordinate, so
    every caller needs this one bit before it can condition correctly. Deriving it
    from the schedule's sign is the same rule the flow-SDE coefficients use.
    """
    if not schedule:
        raise ValueError("flow schedule must not be empty")
    return 1.0 if schedule[0][1] < 0 else 0.0


def action_ness(time: torch.Tensor | float, noise_at: float) -> torch.Tensor:
    """Convert a flow time to the reference's action-ness coordinate ``s``.

    ``s = 0`` is pure noise, ``s = 1`` is a clean action. The reference (and the
    LeRobot port, which calls it ``tau``) is written in this coordinate, so all
    the guidance constants below are direction-independent once converted.
    """
    time_tensor = torch.as_tensor(time, dtype=torch.float32)
    return (1 - time_tensor) if noise_at == 1.0 else time_tensor


def guidance_strength(action_ness_value: torch.Tensor | float, max_guidance_weight: float) -> torch.Tensor:
    """Closed-form RTC guidance coefficient at action-ness ``s``.

    Port of the reference constants (with ``s`` the reference's own time variable)::

        c        = (1 - s) / s
        inv_r2   = ((1 - s)^2 + s^2) / (1 - s)^2
        strength = min(c * inv_r2, max_guidance_weight)

    evaluated with ``nan_to_num`` so both ends behave: ``s -> 0`` (pure noise)
    saturates at the clamp, while ``s -> 1`` (clean action) produces a ``0 * inf``
    NaN that the reference maps to 0, releasing guidance once the sample no longer
    needs it.
    """
    s = torch.as_tensor(action_ness_value, dtype=torch.float32)
    clamp = torch.as_tensor(float(max_guidance_weight), dtype=torch.float32)
    squared_one_minus_s = (1 - s) ** 2
    inv_r2 = (squared_one_minus_s + s**2) / squared_one_minus_s
    c = torch.nan_to_num((1 - s) / s, posinf=float(max_guidance_weight))
    strength = torch.nan_to_num(c * inv_r2, posinf=float(max_guidance_weight))
    return torch.minimum(strength, clamp)


def rtc_guided_velocity(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    time: torch.Tensor | float,
    guidance: RTCGuidance,
    *,
    noise_at: float = 1.0,
) -> torch.Tensor:
    """One soft-guided denoising step: base velocity plus prefix correction.

    Args:
        denoise_fn: Velocity callable taking ``x_t`` only; the engine binds the
            timestep. It must be differentiable with respect to its input, and
            the correction differentiates *through* it.
        x_t: ``[B, H, A]`` or ``[H, A]`` current noisy chunk.
        time: Flow time in the policy's own direction.
        guidance: Conditioning from :func:`build_rtc_guidance`.
        noise_at: Flow time of pure noise, from :func:`flow_noise_end`. Defaults to
            the pi0.5/openpi convention (``t = 1`` is noise).

    Returns:
        The guided velocity, same shape and dtype as ``x_t``.

    Guidance is applied under :func:`torch.enable_grad` because the engine drives
    decode from inside ``torch.no_grad()``; the returned tensor is detached so the
    surrounding loop stays grad-free. This is why a soft-guided step cannot be
    replayed from a captured CUDA graph of the velocity field — the reference pays
    the same inpainting cost, and it is the overhead VLASH's future-state
    conditioning exists to avoid.
    """
    if not guidance.enabled:
        return denoise_fn(x_t)

    prev = guidance.prev_chunk_left_over
    if prev is None:  # pragma: no cover - guarded by `enabled`
        raise ValueError("RTC guidance requires a previous chunk left-over")

    squeezed = x_t.ndim < 3
    x_in = x_t.unsqueeze(0) if squeezed else x_t
    if prev.ndim < 3:
        prev = prev.unsqueeze(0)
    if prev.shape != x_in.shape:
        raise ValueError(
            f"RTC prefix shape {tuple(prev.shape)} must match the decode state {tuple(x_in.shape)}"
        )

    time_tensor = torch.as_tensor(time, device=x_in.device, dtype=x_in.dtype)
    # The conditioning tensors are built host-side in float32; align them with the
    # decode dtype/device so the elementwise term stays dtype-consistent. For the
    # float32 flow used by the verified reference path this is a no-op. `prefix_weights`
    # is stored as `[H]` and reshaped to `[1, H, 1]` here so it broadcasts across
    # batch and action width exactly as the reference's `.unsqueeze(0).unsqueeze(-1)`.
    prev = prev.to(device=x_in.device, dtype=x_in.dtype)
    weights = guidance.prefix_weights.reshape(1, -1, 1).to(device=x_in.device, dtype=x_in.dtype)

    # Fresh leaf so the caller's own autograd history can never leak into the
    # correction, and so the velocity is built with x differentiated.
    x_grad = x_in.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        base_velocity = denoise_fn(x_grad)
        if base_velocity.shape != x_grad.shape:
            raise ValueError("denoise_fn must preserve the action-chunk shape")
        # Express both flow directions in the reference's action-ness coordinate:
        # s = 0 at noise, s = 1 at a clean action, with v_s the matching velocity.
        s = action_ness(time_tensor, noise_at).to(dtype=x_grad.dtype)
        v_s = -base_velocity if noise_at == 1.0 else base_velocity
        x1_t = x_grad + (1 - s) * v_s
        error = (prev - x1_t) * weights
        if guidance.row_scale is not None:
            error = error * guidance.row_scale.reshape(-1, 1, 1).to(device=x_in.device, dtype=x_in.dtype)
        correction = torch.autograd.grad(x1_t, x_grad, error, retain_graph=False)[0]

    # The coefficient is a 0-dim float32 tensor, so it promotes as a scalar and
    # never widens the result beyond the decode dtype.
    strength = guidance_strength(action_ness(time_tensor, noise_at), guidance.max_guidance_weight)
    result = (base_velocity - strength.to(device=x_in.device) * correction).to(dtype=x_in.dtype)
    result = result.detach()
    return result.squeeze(0) if squeezed else result


def hard_prefix_mask(inference_delay: int, action_horizon: int, device=None) -> torch.Tensor:
    """Boolean ``[H, 1]`` mask selecting the steps pinned to the frozen prefix."""
    if inference_delay < 0:
        raise ValueError("inference_delay must be non-negative")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    index = torch.arange(action_horizon, device=device)
    return (index < min(inference_delay, action_horizon)).unsqueeze(-1)


def clamp_prefix(x_t: torch.Tensor, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Write the frozen prefix rows of ``x_t`` from ``prefix`` wherever ``mask``.

    The prefix is aligned to ``x_t``'s device and dtype first. Conditioning is
    usually assembled host-side from a wire payload (see
    :func:`~embodiinfer.engine.async_inference.serving.batch_rtc_guidance`) while
    ``x_t`` lives wherever the policy runs, so the two disagree by default; a
    CPU/CUDA mismatch here is a deployment bug, and it must not depend on every
    caller remembering to move the prefix.
    """
    if x_t.shape != prefix.shape:
        raise ValueError(f"prefix shape {tuple(prefix.shape)} must match state {tuple(x_t.shape)}")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return torch.where(
        mask.to(device=x_t.device),
        prefix.to(device=x_t.device, dtype=x_t.dtype),
        x_t,
    )
