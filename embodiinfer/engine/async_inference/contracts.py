"""Transport contracts for asynchronous-inference hints.

EmbodiRun and EmbodiInfer speak a versioned HTTP/WirelessComm API, so the async
algorithms cannot share Python objects across the boundary — they share a
*payload*. This module owns that payload: a single optional ``async`` block
inside the existing step ``metadata``, carrying everything the server needs to
reproduce the execution side's timing decisions.

Keeping it in ``metadata`` rather than widening the step schema is deliberate:
the step schema is frozen and every existing client keeps working unchanged,
because a request without an ``async`` block is exactly the synchronous request
it always was. Both algorithms are opt-in per request.

Schema (``embodiinfer.async.v1``)::

    {
      "schema": "embodiinfer.async.v1",
      "rtc": {
        "prev_chunk_left_over": [[...], ...],
        "inference_delay": 3,
        "execution_horizon": 8,
        "prefix_attention_schedule": "linear",
        "max_guidance_weight": 10.0,
        "hard_prefix": false
      },
      "vlash": {
        "state_fields": ["joint_0", "joint_1"],
        "pending_actions": [[...], ...],
        "delay": 4,
        "action_space": "absolute",
        "action_to_state": [0, 1, 2, 3, 4, 5, -1]
      }
      }
    }
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .rtc import PrefixAttentionSchedule, RTCGuidance, RTCGuidanceConfig, build_rtc_guidance
from .vlash import ActionSpaceSemantics, VlashStatePlan

__all__ = ["ASYNC_SCHEMA", "AsyncInferencePlan", "RTCPlan", "parse_async_plan"]

ASYNC_SCHEMA = "embodiinfer.async.v1"

# Bounds on the inline numeric payload so a malformed or hostile request cannot
# make the server allocate without limit before validation finishes.
MAX_GRID_ROWS = 4096
MAX_GRID_COLUMNS = 4096


@dataclass(frozen=True, slots=True)
class RTCPlan:
    """A parsed, validated RTC request.

    The leftover chunk stays a plain nested tuple so the plan is comparable,
    hashable, and JSON-round-trippable; :meth:`guidance` materializes the tensor
    view the decoder consumes.
    """

    prev_chunk_left_over: tuple[tuple[float, ...], ...] | None
    inference_delay: int
    config: RTCGuidanceConfig

    def __post_init__(self) -> None:
        if self.inference_delay < 0:
            raise ValueError("rtc.inference_delay must be non-negative")
        if self.prev_chunk_left_over is not None:
            width: int | None = None
            for index, row in enumerate(self.prev_chunk_left_over):
                if not row:
                    raise ValueError(f"rtc.prev_chunk_left_over[{index}] must not be empty")
                if any(not math.isfinite(float(value)) for value in row):
                    raise ValueError(f"rtc.prev_chunk_left_over[{index}] must be finite")
                if width is None:
                    width = len(row)
                elif len(row) != width:
                    raise ValueError("rtc.prev_chunk_left_over rows must all have the same width")

    @property
    def enabled(self) -> bool:
        """Whether a previous chunk is present and worth conditioning on."""
        return self.prev_chunk_left_over is not None and bool(self.prev_chunk_left_over)

    def guidance(self, *, action_horizon: int | None = None, action_dim: int | None = None) -> RTCGuidance:
        """Materialize the tensor conditioning for one decode."""
        import torch

        left_over = None
        if self.prev_chunk_left_over is not None:
            left_over = torch.tensor(self.prev_chunk_left_over, dtype=torch.float32)
        if action_horizon is None and left_over is not None:
            action_horizon = int(left_over.shape[0])
        return build_rtc_guidance(
            left_over,
            self.inference_delay,
            self.config,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )


@dataclass(frozen=True, slots=True)
class AsyncInferencePlan:
    """Both async hints for one request; either half may be absent."""

    rtc: RTCPlan | None = None
    vlash: VlashStatePlan | None = None
    schema: str = ASYNC_SCHEMA
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def requested(self) -> bool:
        """Whether the request asked for any async behaviour at all."""
        return self.rtc is not None or self.vlash is not None


def parse_async_plan(metadata: Mapping[str, Any]) -> AsyncInferencePlan | None:
    """Extract the ``async`` block from step metadata.

    Returns ``None`` when the caller supplied no ``async`` block, which keeps the
    synchronous path free of any async branching. A block that is present but
    malformed raises ``ValueError`` so the serving layer reports
    ``invalid_observation`` rather than silently degrading to synchronous
    behaviour — a silent downgrade under an async deployment would be a control
    hazard, not a convenience.
    """
    block = metadata.get("async")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError("metadata.async must be an object")

    schema = block.get("schema", ASYNC_SCHEMA)
    if schema != ASYNC_SCHEMA:
        raise ValueError(f"unsupported async schema {schema!r}, expected {ASYNC_SCHEMA!r}")

    rtc_plan = _parse_rtc(block.get("rtc"))
    vlash_plan = _parse_vlash(block.get("vlash"))

    known = {"schema", "rtc", "vlash"}
    extra = {key: value for key, value in block.items() if key not in known}
    return AsyncInferencePlan(rtc=rtc_plan, vlash=vlash_plan, schema=str(schema), extra=extra)


def _parse_rtc(value: object) -> RTCPlan | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("async.rtc must be an object")

    raw_left_over = value.get("prev_chunk_left_over")
    left_over: tuple[tuple[float, ...], ...] | None
    if raw_left_over is None:
        left_over = None
    else:
        left_over = _numeric_grid(raw_left_over, "async.rtc.prev_chunk_left_over")

    inference_delay = _non_negative_int(value.get("inference_delay", 0), "async.rtc.inference_delay")
    execution_horizon = _non_negative_int(
        value.get("execution_horizon", 0 if left_over is None else len(left_over)),
        "async.rtc.execution_horizon",
    )
    schedule = value.get("prefix_attention_schedule", PrefixAttentionSchedule.LINEAR.value)
    max_weight = _positive_float(value.get("max_guidance_weight", 10.0), "async.rtc.max_guidance_weight")
    hard_prefix = value.get("hard_prefix", False)
    if not isinstance(hard_prefix, bool):
        raise ValueError("async.rtc.hard_prefix must be a boolean")

    config = RTCGuidanceConfig(
        execution_horizon=execution_horizon,
        prefix_attention_schedule=PrefixAttentionSchedule(schedule),
        max_guidance_weight=max_weight,
        hard_prefix=hard_prefix,
    )
    return RTCPlan(prev_chunk_left_over=left_over, inference_delay=inference_delay, config=config)


def _parse_vlash(value: object) -> VlashStatePlan | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("async.vlash must be an object")

    fields = value.get("state_fields")
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence) or not fields:
        raise ValueError("async.vlash.state_fields must be a non-empty list of strings")
    state_fields = tuple(str(item) for item in fields)
    if any(not field for field in state_fields):
        raise ValueError("async.vlash.state_fields entries must not be empty")

    pending = _numeric_grid(value.get("pending_actions"), "async.vlash.pending_actions")
    delay = _non_negative_int(value.get("delay", 0), "async.vlash.delay")
    semantics = value.get("action_space", ActionSpaceSemantics.ABSOLUTE.value)

    # Optional action-column -> state-column projection, needed whenever the two
    # vectors differ in width (LIBERO: 7-wide OSC action, 8-wide proprioception).
    raw_mapping = value.get("action_to_state")
    mapping: tuple[int, ...] | None = None
    if raw_mapping is not None:
        if isinstance(raw_mapping, (str, bytes)) or not isinstance(raw_mapping, Sequence):
            raise ValueError("async.vlash.action_to_state must be a list of integers")
        entries: list[int] = []
        for index, item in enumerate(raw_mapping):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"async.vlash.action_to_state[{index}] must be an integer")
            entries.append(item)
        mapping = tuple(entries)

    raw_scales = value.get("action_to_state_scale")
    scales: tuple[float, ...] | None = None
    if raw_scales is not None:
        if isinstance(raw_scales, (str, bytes)) or not isinstance(raw_scales, Sequence):
            raise ValueError("async.vlash.action_to_state_scale must be a list of numbers")
        collected: list[float] = []
        for index, item in enumerate(raw_scales):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"async.vlash.action_to_state_scale[{index}] must be a number")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"async.vlash.action_to_state_scale[{index}] must be finite")
            collected.append(number)
        scales = tuple(collected)

    return VlashStatePlan(
        state_fields=state_fields,
        pending_actions=pending,
        delay=delay,
        semantics=ActionSpaceSemantics(semantics),
        action_to_state=mapping,
        action_to_state_scale=scales,
    )


def _numeric_grid(value: object, name: str) -> tuple[tuple[float, ...], ...]:
    """Validate a JSON list-of-lists of finite numbers into a nested tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of lists")
    if len(value) > MAX_GRID_ROWS:
        raise ValueError(f"{name} has too many rows (limit {MAX_GRID_ROWS})")
    rows: list[tuple[float, ...]] = []
    for index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"{name}[{index}] must be a list of numbers")
        if len(row) > MAX_GRID_COLUMNS:
            raise ValueError(f"{name}[{index}] has too many columns (limit {MAX_GRID_COLUMNS})")
        numbers: list[float] = []
        for column, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{name}[{index}][{column}] must be a number")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"{name}[{index}][{column}] must be finite")
            numbers.append(number)
        rows.append(tuple(numbers))
    return tuple(rows)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number
