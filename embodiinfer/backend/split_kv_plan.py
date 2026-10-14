"""Pure launch planning for the existing non-causal split-KV attention kernel.

This module deliberately imports neither Torch nor Triton. Explicit plans are
profiling candidates, not device-specific performance recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_SPLIT_COUNTS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class SplitKVPlan:
    """Static kernel geometry; changing split count changes reduction order."""

    num_splits: int
    block_m: int
    block_n: int
    partition_length: int


def plan_split_kv(
    query_length: int,
    cached_length: int,
    *,
    num_splits: int | None = None,
) -> SplitKVPlan:
    """Plan one current KV token per query, preserving the legacy default.

    ``None`` selects four partitions for at most 64 queries, otherwise one.
    Explicit counts are a bounded power-of-two set supported by the merge
    kernel. They do not select a different dtype, tile size, or attention
    algorithm. Empty partitions are allowed, including when the cache is empty.
    Candidate accuracy and latency must be measured for the actual workload
    before choosing a nondefault count; no hardware heuristic is applied here.
    """
    if isinstance(query_length, bool) or not isinstance(query_length, int) or query_length < 1:
        raise ValueError("query_length must be a positive integer")
    if isinstance(cached_length, bool) or not isinstance(cached_length, int) or cached_length < 0:
        raise ValueError("cached_length must be a nonnegative integer")
    if num_splits is not None and (
        isinstance(num_splits, bool)
        or not isinstance(num_splits, int)
        or num_splits not in SUPPORTED_SPLIT_COUNTS
    ):
        raise ValueError(f"num_splits must be None or one of {SUPPORTED_SPLIT_COUNTS}")
    splits = (4 if query_length <= 64 else 1) if num_splits is None else num_splits
    block_m, block_n = (16 if query_length <= 64 else 32), 64
    length = query_length + cached_length
    partition_length = ((length + splits * block_n - 1) // (splits * block_n)) * block_n
    return SplitKVPlan(splits, block_m, block_n, partition_length)
