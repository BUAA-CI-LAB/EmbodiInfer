"""Transport contracts for asynchronous-inference hints.

EmbodiRun and EmbodiInfer speak a versioned HTTP/WirelessComm API, so the async
algorithms cannot share Python objects across the boundary — they share a
*payload*. This module owns that payload: a single optional ``async`` block
inside the existing step ``metadata``, carrying everything the server needs to
reproduce the execution side's timing decisions.

Keeping it in ``metadata`` rather than widening the step schema is deliberate:
the step schema is frozen and every existing client keeps working unchanged,
because a request without an ``async`` block is exactly the synchronous request
it always was. RTC is opt-in per request.

Schema (``embodiinfer.async.v1``)::

    {
      "schema": "embodiinfer.async.v1",
      "rtc": {
        "prev_chunk_left_over": [[...], ...],
        "model_space": true,
        "inference_delay": 3,
        "cache_start_index": 10,
        "execution_horizon": 8,
        "prefix_attention_schedule": "linear",
        "max_guidance_weight": 10.0,
        "hard_prefix": false
      }
    }
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .rtc import PrefixAttentionSchedule, RTCGuidance, RTCGuidanceConfig, build_rtc_guidance

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
    view the decoder consumes. Explicit leftovers must declare ``model_space``;
    ordinary execution clients should omit them and send ``inference_delay`` so
    the serving adapter resolves its own model-space session cache. Such clients
    should also send ``cache_start_index``, the exact number of actions already
    executed from that cached chunk. It is intentionally independent of model
    latency (``inference_delay``).
    """

    prev_chunk_left_over: tuple[tuple[float, ...], ...] | None
    inference_delay: int
    config: RTCGuidanceConfig
    model_space: bool = False
    cache_start_index: int | None = None

    def __post_init__(self) -> None:
        if self.inference_delay < 0:
            raise ValueError("rtc.inference_delay must be non-negative")
        if self.cache_start_index is not None and (
            isinstance(self.cache_start_index, bool)
            or not isinstance(self.cache_start_index, int)
            or self.cache_start_index < 0
        ):
            raise ValueError("rtc.cache_start_index must be a non-negative integer or null")
        if not isinstance(self.model_space, bool):
            raise TypeError("rtc.model_space must be a boolean")
        if self.prev_chunk_left_over is not None and not self.model_space:
            raise ValueError(
                "rtc.prev_chunk_left_over is model-space data; set model_space=true "
                "or omit it and send inference_delay for the server session cache"
            )
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
    """The RTC hint for one request, when present."""

    rtc: RTCPlan | None = None
    schema: str = ASYNC_SCHEMA
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def requested(self) -> bool:
        """Whether the request asked for any async behaviour at all."""
        return self.rtc is not None


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

    known = {"schema", "rtc"}
    extra = {key: value for key, value in block.items() if key not in known}
    return AsyncInferencePlan(rtc=rtc_plan, schema=str(schema), extra=extra)


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
    raw_cache_start_index = value.get("cache_start_index")
    cache_start_index = (
        None
        if raw_cache_start_index is None
        else _non_negative_int(raw_cache_start_index, "async.rtc.cache_start_index")
    )
    execution_horizon = _non_negative_int(
        value.get("execution_horizon", 0 if left_over is None else len(left_over)),
        "async.rtc.execution_horizon",
    )
    schedule = value.get("prefix_attention_schedule", PrefixAttentionSchedule.LINEAR.value)
    max_weight = _positive_float(value.get("max_guidance_weight", 10.0), "async.rtc.max_guidance_weight")
    hard_prefix = value.get("hard_prefix", False)
    if not isinstance(hard_prefix, bool):
        raise ValueError("async.rtc.hard_prefix must be a boolean")

    model_space = value.get("model_space", False)
    if not isinstance(model_space, bool):
        raise ValueError("async.rtc.model_space must be a boolean")
    config = RTCGuidanceConfig(
        execution_horizon=execution_horizon,
        prefix_attention_schedule=PrefixAttentionSchedule(schedule),
        max_guidance_weight=max_weight,
        hard_prefix=hard_prefix,
    )
    return RTCPlan(
        prev_chunk_left_over=left_over,
        inference_delay=inference_delay,
        config=config,
        model_space=model_space,
        cache_start_index=cache_start_index,
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
