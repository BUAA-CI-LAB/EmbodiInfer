"""VLASH future-state conditioning for asynchronous VLA inference.

VLASH (Tang et al., *VLASH: Real-Time VLAs via Future-State-Aware Asynchronous
Inference*, arXiv 2512.01031) removes the prediction/execution misalignment that
makes naive asynchronous inference unstable. The robot keeps moving during the
forward pass, so a chunk computed from the observation-time state ``s_t`` is
applied at ``s_{t+Δ}``. VLASH closes that gap by *rolling the robot state
forward* under the actions that are already committed::

    s_{t+Δ} = roll(s_t, a_{t : t+Δ-1})

and conditioning the policy on ``(o_t, s_{t+Δ})`` — the state the robot will
actually be in when the chunk starts executing. The observation is still the
current (stale) one, which is the point: a reaction delay is unavoidable, but the
body state is *known*, so it can be predicted instead of guessed.

Action-space semantics matter and both are supported:

``absolute``
    The action vector is the commanded target, so the state after ``Δ`` steps is
    simply ``pending[Δ-1]``. This is the LeRobot / π0.5 convention and what the
    upstream deployment loop hardcodes (it assigns the last action of the
    executing chunk to ``observation.state``).
``delta``
    The action is an increment, so the state after ``Δ`` steps is
    ``s_t + Σ_{i<Δ} pending[i]``. This is the form the paper writes down.

The ``absolute`` case is a read, not a sum, which is why the reference is a
one-liner: with ``Δ`` equal to the remaining steps of the executing chunk, the
last action *is* the state at handover.

This module is pure math over tensors or plain number sequences, with no model,
engine, or embodiment reference, so it is shared verbatim with the execution
side and pinned by common golden vectors on both sides of the boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

__all__ = [
    "ActionSpaceSemantics",
    "VlashStatePlan",
    "apply_vlash_state_plan",
    "roll_state_forward",
    "roll_state_forward_projected",
]


class ActionSpaceSemantics(str, Enum):
    """How an action vector relates to the robot state it produces."""

    ABSOLUTE = "absolute"
    DELTA = "delta"


def roll_state_forward(
    state: torch.Tensor,
    pending_actions: torch.Tensor,
    delay: int,
    semantics: ActionSpaceSemantics | str = ActionSpaceSemantics.ABSOLUTE,
) -> torch.Tensor:
    """Return the robot state ``Δ`` steps ahead, given committed actions.

    Args:
        state: ``[D]`` or ``[B, D]`` current robot state.
        pending_actions: ``[T, D]`` or ``[B, T, D]`` actions already issued and
            still to be executed, in execution order.
        delay: ``Δ``, the number of pending steps that will elapse before the
            new chunk starts executing. ``0`` returns ``state`` unchanged.
        semantics: ``absolute`` (actions are commanded targets) or ``delta``
            (actions are increments).

    Returns:
        ``[D]`` or ``[B, D]`` rolled-forward state, matching ``state``'s rank.

    Raises:
        ValueError: if ``delay`` is negative, exceeds the number of pending
            actions, or the shapes disagree. ``delay == T`` is allowed and means
            the whole pending buffer is consumed.
    """
    semantics = ActionSpaceSemantics(semantics)
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise TypeError("delay must be an integer")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    squeezed = state.ndim == 1
    state_2d = state.unsqueeze(0) if squeezed else state
    if state_2d.ndim != 2:
        raise ValueError(f"state must be 1D or 2D, got shape {tuple(state.shape)}")

    actions = pending_actions.unsqueeze(0) if pending_actions.ndim == 2 else pending_actions
    if actions.ndim != 3:
        raise ValueError(f"pending_actions must be 2D or 3D, got shape {tuple(pending_actions.shape)}")

    horizon = actions.shape[1]
    if delay > horizon:
        raise ValueError(f"delay {delay} exceeds the {horizon} pending action(s) available")
    if actions.shape[2] != state_2d.shape[1]:
        raise ValueError(
            f"pending action width {actions.shape[2]} does not match state width {state_2d.shape[1]}"
        )
    if actions.shape[0] != state_2d.shape[0]:
        raise ValueError(
            f"pending actions batch {actions.shape[0]} does not match state batch {state_2d.shape[0]}"
        )

    actions = actions.to(device=state_2d.device, dtype=state_2d.dtype)

    if delay == 0:
        rolled = state_2d
    elif semantics is ActionSpaceSemantics.ABSOLUTE:
        rolled = actions[:, delay - 1, :]
    else:
        rolled = state_2d + actions[:, :delay, :].sum(dim=1)

    return rolled.squeeze(0) if squeezed else rolled


def roll_state_forward_projected(
    state: torch.Tensor,
    pending_actions: torch.Tensor,
    delay: int,
    semantics: ActionSpaceSemantics | str,
    action_to_state: tuple[int, ...],
    action_to_state_scale: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Roll a state whose width does not match the action vector.

    VLASH's ``s_{t+Δ} = s_t + Σ a`` assumes one action column per state column,
    which holds when a policy emits absolute joint targets (SO-101) but not in
    general. LIBERO is the counter-example that forced this: its proprioception is
    8-wide (``eef_pos(3) + axis_angle(3) + gripper_qpos(2)``) while its OSC delta
    action is 7-wide, because the two-finger gripper position is not a linear
    function of the single gripper command.

    ``action_to_state[j]`` names the state column action column ``j`` drives, or
    ``-1`` to leave it unmapped — an unmapped action simply does not contribute,
    and an untargeted state column keeps its current value. Mapping is a
    controller fact, so it is supplied rather than guessed.

    ``action_to_state_scale[j]`` is the tracking gain of action column ``j``: how many
    state units one unit of that action actually produces. It is NOT optional in
    practice — measured on LIBERO's ``OSC_POSE`` controller, one unit of position
    action moves the end effector ~0.005 and one unit of rotation ~0.047, so a
    unit-scale roll overstates the position change by ~200x and hands the model a
    wildly wrong execution-time state. The gain is a controller fact and must be
    measured, not assumed.
    """
    semantics = ActionSpaceSemantics(semantics)
    if not action_to_state:
        raise ValueError("action_to_state must not be empty")
    targets = [int(value) for value in action_to_state]
    mapped = [value for value in targets if value >= 0]
    if any(value < -1 for value in targets):
        raise ValueError("action_to_state entries must be >= -1")
    if len(set(mapped)) != len(mapped):
        raise ValueError("action_to_state must not map two action columns onto one state column")

    squeezed = state.ndim == 1
    state_2d = state.unsqueeze(0) if squeezed else state
    actions = pending_actions.unsqueeze(0) if pending_actions.ndim == 2 else pending_actions
    if actions.ndim != 3:
        raise ValueError("pending_actions must be 2D or 3D")
    if len(targets) != actions.shape[2]:
        raise ValueError(
            f"action_to_state has {len(targets)} entries but the action vector is {actions.shape[2]} wide"
        )
    if action_to_state_scale is not None:
        if len(action_to_state_scale) != len(targets):
            raise ValueError(
                f"action_to_state_scale has {len(action_to_state_scale)} entries but the action "
                f"vector is {len(targets)} wide"
            )
        if any(not math.isfinite(float(value)) for value in action_to_state_scale):
            raise ValueError("action_to_state_scale entries must be finite")
    if mapped and max(mapped) >= state_2d.shape[1]:
        raise ValueError(
            f"action_to_state targets column {max(mapped)} but the state has {state_2d.shape[1]} columns"
        )
    if delay < 0 or delay > actions.shape[1]:
        raise ValueError(f"delay {delay} is outside the {actions.shape[1]} pending action(s) supplied")

    actions = actions.to(device=state_2d.device, dtype=state_2d.dtype)
    rolled = state_2d.clone()
    if delay == 0:
        return rolled.squeeze(0) if squeezed else rolled

    scales = list(action_to_state_scale) if action_to_state_scale is not None else [1.0] * len(targets)
    if semantics is ActionSpaceSemantics.DELTA:
        total = actions[:, :delay, :].sum(dim=1)
        for column, target in enumerate(targets):
            if target >= 0:
                rolled[:, target] = rolled[:, target] + scales[column] * total[:, column]
    else:
        final = actions[:, delay - 1, :]
        for column, target in enumerate(targets):
            if target >= 0:
                rolled[:, target] = scales[column] * final[:, column]
    return rolled.squeeze(0) if squeezed else rolled


@dataclass(frozen=True, slots=True)
class VlashStatePlan:
    """A request to re-condition one observation on a rolled-forward state.

    The client supplies *ingredients*, not the answer: the names of the state
    fields that participate, the pending action vectors aligned to those fields,
    the delay, and the action-space semantics. The server performs the roll
    itself, so both sides run the same arithmetic and neither has to trust a
    precomputed override.

    A named field may be a **scalar or a numeric vector**; the vectors are
    concatenated in field order to form the action-vector columns. That mirrors
    how the pi0.5 adapter reads proprioception (one ``observation.state`` entry
    holding the whole joint vector is the normal case, not eight named scalars),
    and it means this contract does not force a client to restructure its state
    mapping to use VLASH.

    Attributes:
        state_fields: ``request.state`` keys to read and overwrite, in action
            column order. Each contributes as many columns as it holds scalars.
        pending_actions: ``[T, D]`` committed actions, in execution order, where
            ``D`` is the total scalar width of the named fields.
        delay: ``Δ`` in control steps.
        semantics: Action-space semantics for the participating fields.
    """

    state_fields: tuple[str, ...]
    pending_actions: tuple[tuple[float, ...], ...]
    delay: int
    semantics: ActionSpaceSemantics = ActionSpaceSemantics.ABSOLUTE
    action_to_state: tuple[int, ...] | None = None
    action_to_state_scale: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.state_fields:
            raise ValueError("state_fields must not be empty")
        if len(set(self.state_fields)) != len(self.state_fields):
            raise ValueError("state_fields must be unique")
        if self.pending_actions:
            widths = {len(row) for row in self.pending_actions}
            if 0 in widths:
                raise ValueError("pending_actions rows must not be empty")
            if len(widths) != 1:
                raise ValueError("pending_actions rows must all have the same width")
            for index, row in enumerate(self.pending_actions):
                if any(not math.isfinite(float(value)) for value in row):
                    raise ValueError(f"pending_actions[{index}] must be finite")
        if self.delay < 0:
            raise ValueError("delay must be non-negative")
        if self.delay > len(self.pending_actions):
            raise ValueError(
                f"delay {self.delay} exceeds the {len(self.pending_actions)} pending action(s) supplied"
            )
        if not isinstance(self.semantics, ActionSpaceSemantics):
            object.__setattr__(self, "semantics", ActionSpaceSemantics(self.semantics))
        if self.action_to_state is not None:
            targets = tuple(int(value) for value in self.action_to_state)
            if not targets:
                raise ValueError("action_to_state must not be empty when provided")
            if len(targets) != self.action_width:
                raise ValueError(
                    f"action_to_state has {len(targets)} entries but pending_actions rows are "
                    f"{self.action_width} wide"
                )
            if any(value < -1 for value in targets):
                raise ValueError("action_to_state entries must be >= -1")
            mapped = [value for value in targets if value >= 0]
            if len(set(mapped)) != len(mapped):
                raise ValueError("action_to_state must not map two action columns onto one state column")
            object.__setattr__(self, "action_to_state", targets)
        if self.action_to_state_scale is not None:
            scales = tuple(float(value) for value in self.action_to_state_scale)
            if len(scales) != self.action_width:
                raise ValueError(
                    f"action_to_state_scale has {len(scales)} entries but pending_actions rows are "
                    f"{self.action_width} wide"
                )
            if any(not math.isfinite(value) for value in scales):
                raise ValueError("action_to_state_scale entries must be finite")
            if self.action_to_state is None:
                raise ValueError("action_to_state_scale requires action_to_state")
            object.__setattr__(self, "action_to_state_scale", scales)

    @property
    def action_width(self) -> int:
        """Number of scalars in each pending action vector (0 when there are none)."""
        return len(self.pending_actions[0]) if self.pending_actions else 0

    @property
    def observation_is_stale(self) -> bool:
        """Whether conditioning actually changed, i.e. whether vision is stale."""
        return self.delay > 0


def _flatten_numeric(value: Any, field: str) -> list[float]:
    """Flatten a scalar or arbitrarily nested numeric sequence to floats."""
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item, field))
        return flattened
    raise ValueError(f"state field {field!r} must hold numbers, got {type(value).__name__}")


def _restore_numeric(original: Any, values: list[float]) -> Any:
    """Write rolled values back in the shape the caller sent them in.

    Scalars stay scalars and single-element sequences stay sequences, so a
    round-trip through VLASH does not silently change a field's type. A nested
    sequence is written back flat, because the roll operates on the concatenated
    vector and any nesting is a presentation detail the adapter flattens anyway.
    """
    if len(values) == 1:
        only = values[0]
        if isinstance(original, bool):
            return bool(only)
        if isinstance(original, int) and float(only).is_integer():
            return int(only)
        return [only] if isinstance(original, (list, tuple)) else only
    return values


def apply_vlash_state_plan(state: Mapping[str, Any], plan: VlashStatePlan) -> dict[str, Any]:
    """Return ``state`` with the plan's fields replaced by their rolled values.

    Only the named fields change; every other key is copied through untouched so a
    policy that also consumes non-proprioceptive state is unaffected. Field widths
    are read from the state itself, so the caller does not have to declare them.

    When the state and the action vector differ in width — LIBERO's 8-wide
    proprioception against its 7-wide OSC delta action — the plan must carry an
    explicit :attr:`VlashStatePlan.action_to_state` projection. Without one, a
    width disagreement is an error rather than a silent reinterpretation of the
    columns, because reinterpreting them would roll the wrong joints.
    """
    missing = [field for field in plan.state_fields if field not in state]
    if missing:
        raise ValueError(f"state is missing VLASH field(s): {', '.join(missing)}")

    current: list[float] = []
    layout: list[tuple[str, int]] = []
    for field in plan.state_fields:
        values = _flatten_numeric(state[field], field)
        if not values:
            raise ValueError(f"state field {field!r} holds no numeric values")
        current.extend(values)
        layout.append((field, len(values)))

    widths = {len(row) for row in plan.pending_actions}
    action_width = widths.pop() if widths else 0
    if plan.action_to_state is not None:
        if max([*plan.action_to_state, -1]) >= len(current):
            raise ValueError(
                f"action_to_state targets a column beyond the {len(current)} state value(s) provided"
            )
    elif len(current) != action_width:
        raise ValueError(
            f"state fields {list(plan.state_fields)} carry {len(current)} value(s) but "
            f"pending_actions rows are {action_width} wide; supply action_to_state to map "
            "action columns onto state columns"
        )

    actions = torch.tensor(plan.pending_actions, dtype=torch.float32)
    flat_state = torch.tensor([current], dtype=torch.float32)
    if plan.action_to_state is not None:
        rolled = roll_state_forward_projected(
            flat_state,
            actions,
            plan.delay,
            plan.semantics,
            plan.action_to_state,
            plan.action_to_state_scale,
        )[0]
    else:
        rolled = roll_state_forward(flat_state, actions, plan.delay, plan.semantics)[0]

    updated = dict(state)
    offset = 0
    for field, width in layout:
        chunk = [float(value) for value in rolled[offset : offset + width]]
        offset += width
        updated[field] = _restore_numeric(state[field], chunk)
    return updated
