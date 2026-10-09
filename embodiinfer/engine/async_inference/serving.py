"""Applying async hints at the serving boundary.

The two algorithms attach at different layers, for a reason worth stating.

**VLASH is transport-level.** Rolling the robot state forward is arithmetic on the
request's own ``state`` mapping plus numbers the client already knows. It needs no
model, no prefix, and no engine — only the field names and the pending actions. So
it is applied once in :class:`~embodiinfer.engine.serve.service.PolicyService`,
before any adapter sees the request, and every policy in the catalog inherits it.

**RTC is decode-level.** Prefix guidance conditions the denoising trajectory, so it
belongs to whoever owns the decode loop — an adapter holding an
:class:`~embodiinfer.engine.core.EngineCore`. The helpers here prepare that
conditioning from the wire plan so each adapter does not re-derive it.

Both arrive in the same ``metadata["async"]`` block, are parsed once by
:func:`~embodiinfer.engine.async_inference.contracts.parse_async_plan`, and are
opt-in: a request without the block is byte-for-byte the request it always was.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from .contracts import AsyncInferencePlan, parse_async_plan
from .rtc import RTCGuidance, build_rtc_guidance
from .vlash import apply_vlash_state_plan

__all__ = [
    "apply_request_async_state",
    "batch_rtc_guidance",
    "plan_from_request",
]


def plan_from_request(request) -> AsyncInferencePlan | None:
    """Parse the async block of a raw policy request, or ``None`` if absent."""
    metadata = getattr(request, "metadata", None)
    if metadata is None:
        return None
    return parse_async_plan(metadata)


def apply_request_async_state(request):
    """Return ``request`` with any VLASH state roll-forward already applied.

    Returns the request unchanged when no async block (or no VLASH half) is
    present, so the synchronous path costs one dict lookup. The state mapping is
    the only thing rewritten: identity, ordering, and images are untouched, and
    the transformation is a pure function of the payload, so applying it before
    the idempotency cache is populated is safe and reproducible.
    """
    plan = plan_from_request(request)
    if plan is None or plan.vlash is None:
        return request
    state = request.state
    rolled = apply_vlash_state_plan(state, plan.vlash)
    if rolled == dict(state):
        return request
    return replace(request, state=rolled)


def batch_rtc_guidance(
    plans: Sequence[AsyncInferencePlan | None],
    *,
    action_horizon: int | None = None,
    action_dim: int | None = None,
) -> RTCGuidance | None:
    """Combine the RTC halves of several requests into one batched conditioning.

    Returns ``None`` when nobody asked for RTC, which is the common synchronous
    case and must stay allocation-free. Requests that did ask are stacked in
    order.

    A request that did **not** ask contributes an all-zero leftover and is then
    explicitly excluded through ``row_scale``. The exclusion is necessary rather
    than cosmetic: with the weights shared across the batch, a zero leftover would
    otherwise make the correction pull that row's denoising toward zero actions,
    which is not the same thing as leaving it unconstrained.

    Args:
        plans: One parsed plan (or ``None``) per batch row, in row order.
        action_horizon: Current chunk length ``H``. Defaults to the leftover
            length; supply it when the leftover is shorter than the chunk.
        action_dim: Current action width ``A``. Defaults to the leftover width.

    Raises:
        ValueError: if the RTC requests disagree on the inference delay, on the
            leftover's shape, or on the static config. A batch shares one
            execution clock, so mixed delays describe different timings; merging
            them would silently mis-condition a chunk that the robot then
            executes.
    """
    active = [plan.rtc for plan in plans if plan is not None and plan.rtc is not None and plan.rtc.enabled]
    if not active:
        return None

    reference = active[0]
    for other in active[1:]:
        if other.inference_delay != reference.inference_delay:
            raise ValueError("RTC inference_delay must match across a batch")

    first_grid = reference.prev_chunk_left_over or ()
    horizon = len(first_grid)
    width = len(first_grid[0]) if horizon else 0

    # Shape is checked before the static config so that a genuinely malformed
    # batch reports the mismatch a caller can act on, not a config difference
    # that merely follows from the leftover lengths.
    grids: list[tuple[tuple[float, ...], ...]] = []
    participating: list[bool] = []
    for plan in plans:
        rtc = None if plan is None else plan.rtc
        grid = () if rtc is None or not rtc.enabled else (rtc.prev_chunk_left_over or ())
        if grid and (len(grid) != horizon or len(grid[0]) != width):
            raise ValueError("RTC prev_chunk_left_over shape must match across a batch")
        participating.append(bool(grid))
        grids.append(grid if grid else tuple((0.0,) * width for _ in range(horizon)))

    for other in active[1:]:
        if other.config != reference.config:
            raise ValueError("RTC config must match across a batch")

    stacked = torch.stack([torch.tensor(grid, dtype=torch.float32) for grid in grids])
    guidance = build_rtc_guidance(
        stacked,
        reference.inference_delay,
        reference.config,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    if all(participating):
        return guidance
    scale = torch.tensor([1.0 if flag else 0.0 for flag in participating], dtype=torch.float32)
    return replace(guidance, row_scale=scale.view(-1, 1, 1))
