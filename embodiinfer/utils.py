"""Small shared utilities."""

from __future__ import annotations

import time

import numpy as np
import torch

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def torch_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def now_ns() -> int:
    return time.perf_counter_ns()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentiles(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}
    a = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
    }


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar time in [0, 1].

    Args:
        t: [B] flow-matching time values.
        dim: embedding dimension (even).
    Returns:
        [B, dim] embedding.
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
