"""Model-neutral, CPU-testable numerical gates and synchronized timing.

Exact storage equality is distinct from numerical equality: signed zero and source
precision are checked before any conversion used to compute error metrics. The
lossless gate is frozen here, never estimated from the observed candidate drift.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    """Predeclared absolute, RMS and relative-L2 budgets, all required to pass."""

    max_abs: float = 0.0
    rms: float = 0.0
    relative_l2: float = 0.0
    exact_bits: bool = True
    relative_floor: float = 1e-12

    def __post_init__(self) -> None:
        values = (self.max_abs, self.rms, self.relative_l2, self.relative_floor)
        if not all(math.isfinite(value) and value >= 0 for value in values) or self.relative_floor == 0:
            raise ValueError("thresholds must be finite/nonnegative and relative_floor positive")


LOSSLESS = Thresholds()


@dataclass(frozen=True)
class Snapshot:
    """An owned numerical array and its exact original contiguous storage bytes."""

    values: np.ndarray
    storage: bytes
    dtype: str

    @classmethod
    def from_numpy(cls, value: np.ndarray) -> Snapshot:
        """Copy an array so future workspace reuse cannot change a reference."""
        array = np.array(value, copy=True, order="C")
        if array.dtype.kind not in "biuf":
            raise TypeError("snapshots require real numeric or boolean arrays")
        array.setflags(write=False)
        return cls(array, array.tobytes(), str(array.dtype))

    def metadata(self) -> dict[str, Any]:
        """Identify the snapshot without embedding large tensor contents in JSON."""
        return dict(
            shape=list(self.values.shape),
            dtype=self.dtype,
            bytes=len(self.storage),
            sha256=hashlib.sha256(self.storage).hexdigest(),
        )


def compare(reference: Snapshot, candidate: Snapshot, thresholds: Thresholds = LOSSLESS) -> dict[str, Any]:
    """Fail closed on empty, nonfinite, shape/dtype, byte or metric differences.

    relative_l2 = RMS(candidate-reference) / max(RMS(reference), relative_floor).
    Metric computation uses FP64 only for analysis; neither model input is cast.
    """
    row: dict[str, Any] = dict(
        reference=reference.metadata(),
        candidate=candidate.metadata(),
        thresholds=asdict(thresholds),
        passed=False,
        exact_bits=False,
        max_abs=None,
        rms=None,
        relative_l2=None,
        reference_rms=None,
        failures=[],
    )
    if reference.values.shape != candidate.values.shape:
        row["failures"].append("shape_mismatch")
        return row
    if not reference.values.size:
        row["failures"].append("empty_tensor")
        return row
    if reference.dtype != candidate.dtype:
        row["failures"].append("dtype_mismatch")
    row["exact_bits"] = reference.dtype == candidate.dtype and reference.storage == candidate.storage
    if not np.isfinite(reference.values).all() or not np.isfinite(candidate.values).all():
        row["failures"].append("nonfinite_tensor")
        return row
    reference64 = reference.values.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = candidate.values.astype(np.float64) - reference64
        max_abs = float(np.max(np.abs(error)))
        # Rescaling avoids overflow from squaring finite large FP64 values.
        rms = _rms(error)
        reference_rms = _rms(reference64)
        relative_l2 = rms / max(reference_rms, thresholds.relative_floor)
    metrics = dict(max_abs=max_abs, rms=rms, relative_l2=relative_l2, reference_rms=reference_rms)
    if not all(math.isfinite(value) for value in metrics.values()):
        row["failures"].append("nonfinite_metric")
        return row
    row.update(metrics)
    for name in ("max_abs", "rms", "relative_l2"):
        if row[name] > getattr(thresholds, name):
            row["failures"].append(name)
    if thresholds.exact_bits and not row["exact_bits"]:
        row["failures"].append("bit_mismatch")
    row["passed"] = not row["failures"]
    return row


def _rms(value: np.ndarray) -> float:
    scale = float(np.max(np.abs(value)))
    if not math.isfinite(scale):
        return scale
    return 0.0 if scale == 0 else float(scale * np.sqrt(np.mean(np.square(value / scale))))


def compare_traces(
    reference: Mapping[str, Snapshot],
    candidate: Mapping[str, Snapshot],
    thresholds: Thresholds = LOSSLESS,
) -> dict[str, Any]:
    """Require complete matching trace keys, including every requested layer/step."""
    missing = sorted(reference.keys() - candidate.keys())
    unexpected = sorted(candidate.keys() - reference.keys())
    rows = {
        key: compare(reference[key], candidate[key], thresholds)
        for key in sorted(reference.keys() & candidate.keys())
    }
    return dict(
        passed=bool(reference)
        and not missing
        and not unexpected
        and all(row["passed"] for row in rows.values()),
        missing=missing,
        unexpected=unexpected,
        quantities=rows,
    )


def gripper_flips(
    reference: Snapshot,
    candidate: Snapshot,
    *,
    index: int,
    threshold: float,
    comparison: str,
) -> dict[str, Any]:
    """Count boolean decision changes for a caller-declared physical action channel.

    ``gt/ge/lt/le`` specify the decision explicitly, including equality behavior.
    This is a threshold-discontinuity check, not a robot task-success metric.
    """
    operators = {"gt": np.greater, "ge": np.greater_equal, "lt": np.less, "le": np.less_equal}
    if comparison not in operators or not math.isfinite(threshold):
        raise ValueError("declare a finite threshold and one of gt/ge/lt/le")
    if reference.values.shape != candidate.values.shape or reference.values.ndim < 1:
        raise ValueError("gripper action shapes must match and contain an action axis")
    if index < 0 or index >= reference.values.shape[-1]:
        raise ValueError("gripper index is outside the physical action dimensions")
    if (
        not reference.values.size
        or not np.isfinite(reference.values).all()
        or not np.isfinite(candidate.values).all()
    ):
        raise ValueError("gripper checks require nonempty finite actions")
    ref, actual = reference.values[..., index], candidate.values[..., index]
    predicate = operators[comparison]
    flipped = predicate(ref, threshold) != predicate(actual, threshold)
    return dict(
        index=index,
        threshold=threshold,
        comparison=comparison,
        semantics=f"decision = physical_action[..., {index}] {comparison} {threshold}",
        decisions=int(flipped.size),
        flips=int(np.count_nonzero(flipped)),
        flip_indices=np.argwhere(flipped).tolist(),
        reference_at_threshold=int(np.count_nonzero(ref == threshold)),
        candidate_at_threshold=int(np.count_nonzero(actual == threshold)),
        minimum_reference_margin=float(np.min(np.abs(ref - threshold))),
        passed=not bool(flipped.any()),
    )


def timing_summary(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize raw synchronized wall times, rejecting invalid or empty samples."""
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("timings must be a nonempty one-dimensional sequence of finite nonnegative samples")
    return dict(
        count=int(values.size),
        mean_ms=float(np.mean(values)),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        min_ms=float(np.min(values)),
        max_ms=float(np.max(values)),
    )


def synchronized_call(
    operation: Callable[[], Any],
    synchronize: Callable[[], Any],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, float]:
    """Return an operation result and synchronized host elapsed milliseconds."""
    synchronize()
    start = clock()
    result = operation()
    synchronize()
    elapsed = (clock() - start) * 1000
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    return result, elapsed


def benchmark_requests(
    request: Callable[[int], Any],
    *,
    cases: int,
    synchronize: Callable[[], Any],
    warmup: int,
    iterations: int,
    before_request: Callable[[int], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Warm every case, then time round-robin whole requests with both boundaries synchronized.

    ``before_request`` may reset a fixed-noise generator outside the measurement.
    There is no instrumentation or device-event allocation inside the boundary.
    """
    if min(cases, warmup, iterations) < 1:
        raise ValueError("cases, per-case warmup and measured iterations must be positive")
    warmup_samples = []
    for case in range(cases):
        synchronize()
        warmup_start = clock()
        for _ in range(warmup):
            if before_request is not None:
                before_request(case)
            request(case)
        synchronize()
        warmup_samples.append(dict(case=case, wall_ms=(clock() - warmup_start) * 1000))
    samples = []
    for iteration in range(iterations):
        case = iteration % cases
        if before_request is not None:
            before_request(case)
        synchronize()
        start = clock()
        request(case)
        synchronize()
        samples.append(dict(iteration=iteration, case=case, wall_ms=(clock() - start) * 1000))
    return dict(
        warmup_per_case=warmup,
        warmup_timing=dict(
            scope="per-case warmup blocks, including generator reset; excluded from steady-state samples",
            raw_samples=warmup_samples,
            total_wall_ms=sum(row["wall_ms"] for row in warmup_samples),
            summary=timing_summary([row["wall_ms"] for row in warmup_samples]),
        ),
        iterations=iterations,
        raw_samples=samples,
        summary=timing_summary([row["wall_ms"] for row in samples]),
        summary_per_case={
            str(case): timing_summary([row["wall_ms"] for row in samples if row["case"] == case])
            for case in range(min(cases, iterations))
        },
    )


def benchmark_paired_requests(
    requests: dict[str, Callable[[int], Any]],
    *,
    cases: int,
    synchronize: Callable[[], None],
    warmup: int,
    iterations: int,
    before_request: Callable[[str, int], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure every mode per round, alternating order to balance phase drift.

    Every call includes synchronization at both boundaries. Generator resets
    occur outside the measured interval. Each mode sees identical case counts,
    with AB then BA ordering for two modes; no outliers are removed.
    """
    if len(requests) < 2 or min(cases, warmup, iterations) < 1:
        raise ValueError("paired timing needs at least two modes and positive counts")
    names = tuple(requests)
    for case in range(cases):
        for name in names:
            for _ in range(warmup):
                if before_request is not None:
                    before_request(name, case)
                requests[name](case)
            synchronize()
    samples = []
    for iteration in range(iterations):
        order = names if iteration % 2 == 0 else names[::-1]
        case = iteration % cases
        for position, name in enumerate(order):
            if before_request is not None:
                before_request(name, case)
            synchronize()
            start = clock()
            requests[name](case)
            synchronize()
            samples.append(
                dict(
                    mode=name,
                    iteration=iteration,
                    case=case,
                    position=position,
                    wall_ms=(clock() - start) * 1000,
                )
            )
    return dict(
        warmup_per_case=warmup,
        iterations_per_mode=iterations,
        order="alternate insertion order and reverse order every round",
        raw_samples=samples,
        modes={
            name: timing_summary([row["wall_ms"] for row in samples if row["mode"] == name]) for name in names
        },
    )
