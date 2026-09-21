"""Engine-level execution configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Engine-level execution knobs."""

    device: str = "cuda"
    # CUDA "auto" preserves individual parameter/buffer dtypes and uses the
    # policy's execution_dtype for observations and decoder state (proposal 0001).
    # CPU remains FP32; explicit bf16/fp16 uniformly casts the policy.
    dtype: str = "auto"
    max_batch_size: int = 32  # ceiling for a single forward pass
    max_wait_ms: float = 5.0  # scheduler batching window (async engine)
    # optimization toggles (each maps to a claim in docs/en/architecture.md)
    use_cuda_graph: bool = True  # static-shape capture of the denoise loop
    # capture the whole N-step integration as a single graph (one replay per
    # prediction) instead of one graph per step; drives step-to-step host
    # overhead from O(N) to O(1). See docs/proposals/0001-static-loop-capture.md.
    capture_full_loop: bool = False
    reuse_prefix_kv: bool = True  # compute prefix KV once, reuse across steps
    # bucketing: pad batch up to the nearest bucket so graphs are reusable
    batch_buckets: tuple = (1, 2, 4, 8, 16, 32)
    num_steps: int | None = None  # override policy default_num_steps

    def resolve_bucket(self, n: int) -> int:
        """Choose a graph shape within the configured execution ceiling."""
        if not 1 <= n <= self.max_batch_size:
            raise ValueError("batch size must be between 1 and max_batch_size")
        for b in self.batch_buckets:
            if n <= b <= self.max_batch_size:
                return b
        return self.max_batch_size
