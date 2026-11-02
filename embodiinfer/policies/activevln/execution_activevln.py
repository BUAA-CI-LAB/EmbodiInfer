"""Opt-in continuation context for observations taken during committed actions.

This is prompt conditioning for the autoregressive policy, not flow RTC guidance.
The deployment runtime owns execution, timing and cancellation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

EXECUTION_PREFIX_SCHEMA = "activevln.execution_prefix.v1"


@dataclass(frozen=True)
class ExecutionPrefix:
    """Remaining R2R cm/degree actions from the immediately preceding model turn."""

    previous_step_id: int
    remaining_actions: tuple[tuple[int, float], ...]

    @classmethod
    def parse(cls, value: object, *, step_id: int | None = None) -> ExecutionPrefix:
        """Reject unsupported fields, invalid magnitudes and out-of-order prefixes."""
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "previous_step_id",
            "remaining_actions",
        }:
            raise ValueError("invalid ActiveVLN execution prefix fields")
        if value["schema"] != EXECUTION_PREFIX_SCHEMA:
            raise ValueError("unsupported ActiveVLN execution prefix schema")
        previous = value["previous_step_id"]
        if type(previous) is not int or previous < 0:
            raise ValueError("previous_step_id must be non-negative")
        if step_id is not None and previous != step_id - 1:
            raise ValueError("execution prefix must reference the previous step")
        rows = value["remaining_actions"]
        if not isinstance(rows, (list, tuple)) or len(rows) > 3:
            raise ValueError("execution prefix must contain at most three actions")
        actions = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError("execution prefix actions must be [id, magnitude]")
            kind, magnitude = row
            if type(kind) is not int or kind not in (1, 2, 3):
                raise ValueError("execution prefix allows only forward/left/right")
            if (
                isinstance(magnitude, bool)
                or not isinstance(magnitude, (int, float))
                or not math.isfinite(magnitude)
                or not 0 < magnitude <= (75 if kind == 1 else 45)
            ):
                raise ValueError("execution prefix magnitude is outside R2R bounds")
            actions.append((kind, float(magnitude)))
        return cls(previous, tuple(actions))

    def to_wire(self) -> dict[str, object]:
        """Return the canonical JSON form used for the server acknowledgement."""
        return {
            "schema": EXECUTION_PREFIX_SCHEMA,
            "previous_step_id": self.previous_step_id,
            "remaining_actions": [[kind, value] for kind, value in self.remaining_actions],
        }

    @property
    def digest(self) -> str:
        """Bind the response to the exact continuation context it consumed."""
        wire = json.dumps(self.to_wire(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(wire.encode()).hexdigest()

    def prompt_context(self) -> str:
        """Describe the immutable tail and request only actions after its completion."""
        names = {1: "move forward", 2: "turn left", 3: "turn right"}
        tail = (
            ", ".join(
                f"{names[kind]} {value:.3f} {'cm' if kind == 1 else 'degrees'}"
                for kind, value in self.remaining_actions
            )
            or "none (the previous actions have finished)"
        )
        return (
            "\nExecution context: this image was captured during the previous action chunk, "
            "not necessarily after it. At image capture, the following remaining actions "
            f"were already committed: {tail}. They will finish before your next actions start. "
            "Plan only the continuation AFTER those actions; do not repeat the committed tail. "
            "A stop response ends navigation after that tail.\n"
        )
