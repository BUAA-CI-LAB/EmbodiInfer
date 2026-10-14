"""Profile explicit split-KV counts without changing the runtime default.

Run from an installed checkout, for example::

    python benchmarks/attention_split.py --atol 0.02 --rtol 0.02 \
        --splits 1 2 4 8 16 --capture --output results/attention-split.json

The example thresholds are illustrative, not an accuracy recommendation. Declare
workload-appropriate tolerances before running. This synthetic operator benchmark
does not establish whole-request action parity or a deployment speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from embodiinfer.backend.split_kv_plan import SUPPORTED_SPLIT_COUNTS, plan_split_kv
from embodiinfer.backend.triton.capability import require_triton
from embodiinfer.layers import get_split_kv_attention_backend


def reference_attention(
    query: torch.Tensor,
    cached_key: torch.Tensor,
    cached_value: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """FP32 eager non-causal reference over exactly the candidate's rounded inputs.

    Materializing K/V and repeating GQA heads is intentional in this untimed
    reference. Padding uses zero/nonzero validity, not arbitrary additive bias.
    Each batch row must retain at least one key.
    """
    repeats = query.shape[1] // key.shape[1]
    full_key = torch.cat((cached_key, key), dim=2).float().repeat_interleave(repeats, dim=1)
    full_value = torch.cat((cached_value, value), dim=2).float().repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query.float(), full_key.transpose(-1, -2)) * query.shape[-1] ** -0.5
    if mask is not None:
        if not (mask == 0).any(dim=-1).all().item():
            raise ValueError("reference requires at least one valid key in each batch row")
        scores = scores.masked_fill(mask != 0, -float("inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), full_value)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict:
    """Gate every element using declared tolerances, rejecting nonfinite outputs."""
    if actual.shape != expected.shape:
        raise ValueError("accuracy comparison requires matching shapes")
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    if not finite:
        return {"passed": False, "finite": False, "max_abs": None, "rmse": None, "bit_exact": False}
    error = actual.float() - expected.float()
    return {
        "passed": bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)),
        "finite": True,
        "max_abs": float(error.abs().max().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "bit_exact": actual.dtype == expected.dtype
        and bool(
            torch.equal(
                actual.reshape(-1).contiguous().view(torch.uint8),
                expected.reshape(-1).contiguous().view(torch.uint8),
            )
        ),
    }


def timing_summary(samples: list[float]) -> dict[str, Any]:
    """Retain every sample and linearly interpolated p50/p95 in milliseconds."""
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("timing samples must be nonempty, finite and nonnegative")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"raw_ms": samples, "p50_ms": percentile(0.5), "p95_ms": percentile(0.95)}


def _measure(fn: Callable[[], Any], stream: torch.cuda.Stream, iterations: int) -> dict:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    host_samples = []
    # Initialize lazy CUDA events outside the measurements.
    for event in (*starts, *ends):
        event.record(stream)
    stream.synchronize()
    for start, end in zip(starts, ends):
        stream.synchronize()
        before = time.perf_counter()
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        host_samples.append((time.perf_counter() - before) * 1000)
    return {
        "cuda_elapsed": timing_summary([start.elapsed_time(end) for start, end in zip(starts, ends)]),
        "synchronized_host": timing_summary(host_samples),
    }


def _warmup(fn: Callable[[], Any], stream: torch.cuda.Stream, iterations: int) -> float:
    stream.synchronize()
    before = time.perf_counter()
    for _ in range(iterations):
        fn()
    stream.synchronize()
    return (time.perf_counter() - before) * 1000


def _provenance(tensor: torch.Tensor) -> dict:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Require numerical thresholds before creating candidates or timing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument(
        "--splits", nargs="+", type=int, choices=SUPPORTED_SPLIT_COUNTS, default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--cached", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, choices=[64, 128, 256], default=256)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask", choices=["none", "padding"], default="padding")
    parser.add_argument("--padding-fraction", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--capture", action="store_true", help="also measure CUDA graph setup and replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        parser.error("atol and rtol must be finite and nonnegative")
    if min(args.batch, args.query_heads, args.kv_heads, args.queries, args.warmup, args.iterations) < 1:
        parser.error("batch, heads, queries, warmup and iterations must be positive")
    if args.cached < 0 or args.device < 0 or args.query_heads % args.kv_heads:
        parser.error("cached/device must be nonnegative and query heads must be a multiple of KV heads")
    if not 0 <= args.padding_fraction <= 1:
        parser.error("padding-fraction must be between zero and one")
    return args


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    """Measure one workload; steady-state timing requires that mode's accuracy gates."""
    if not torch.cuda.is_available():
        raise RuntimeError("attention split benchmark requires CUDA and Triton; no CUDA device is available")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    capability = require_triton(device)
    torch.set_float32_matmul_precision("highest")
    dtype = getattr(torch, args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support BF16")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    def make_tensor(heads: int, length: int) -> torch.Tensor:
        return torch.randn(args.batch, heads, length, args.head_dim, generator=generator).to(device, dtype)

    tensors = (
        make_tensor(args.query_heads, args.queries),
        make_tensor(args.kv_heads, args.cached),
        make_tensor(args.kv_heads, args.cached),
        make_tensor(args.kv_heads, args.queries),
        make_tensor(args.kv_heads, args.queries),
    )
    mask = None
    if args.mask == "padding":
        padding = torch.rand(args.batch, 1, 1, args.cached + args.queries, generator=generator)
        mask = torch.zeros_like(padding).masked_fill_(padding < args.padding_fraction, -float("inf"))
        mask[..., -1] = 0  # Includes fully masked partitions but never a fully masked batch row.
        mask = mask.to(device)
    props = torch.cuda.get_device_properties(device)
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        root / "embodiinfer/backend/split_kv_plan.py",
        root / "embodiinfer/backend/triton/split_kv_attention.py",
        root / "embodiinfer/layers/attention.py",
    ]
    report = {
        "schema_version": 1,
        "scope": "synthetic non-causal split-KV attention; not whole-request performance or action parity",
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": capability.version,
            "gpu": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": props.total_memory,
            "multiprocessors": props.multi_processor_count,
            "nvidia_smi": _command_output(
                "nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader"
            ),
            "revision": _command_output("git", "-C", str(root), "rev-parse", "HEAD"),
            "git_status": _command_output("git", "-C", str(root), "status", "--short"),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
            },
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "inputs": {
            name: _provenance(tensor)
            for name, tensor in zip(("query", "cached_key", "cached_value", "key", "value"), tensors)
        },
        "mask": None if mask is None else _provenance(mask),
        "accuracy_contract": {
            "atol": args.atol,
            "rtol": args.rtol,
            "reference": "FP32 eager QK, softmax, PV using the same rounded FP16/BF16 inputs; TF32 disabled",
            "gates": ["reference", "legacy_default"],
            "graph_gate": "bit-exact equality to eager candidate",
            "reason_for_difference": "KV partitioning changes online softmax and FP32 reduction order",
        },
        "timing_contract": {
            "stream": "one dedicated CUDA stream for every candidate and phase",
            "cuda_elapsed": "start/end CUDA events on that stream, synchronized per sample",
            "synchronized_host": "perf_counter includes event recording, Python dispatch, allocation and end-event synchronization",
            "setup": "first call (possibly JIT/cache load), warmup and graph capture reported separately; excluded from steady state",
            "order": [None, *dict.fromkeys(args.splits)],
            "note": "Sequential synthetic sweep; clocks, thermals and concurrent GPU work are not controlled by this tool",
        },
        "results": [],
    }
    backend = get_split_kv_attention_backend("triton_split_kv")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        stream.synchronize()
        before = time.perf_counter()
        reference = reference_attention(*tensors, mask)
        stream.synchronize()
        report["reference_setup_host_ms"] = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        default = backend.attend_split(*tensors, key_padding_mask=mask)
        stream.synchronize()
        report["default_first_call_host_ms"] = (time.perf_counter() - before) * 1000
        report["default_reference"] = error_metrics(default, reference, atol=args.atol, rtol=args.rtol)
        if not report["default_reference"]["passed"]:
            report["passed"] = False
            return report
        # The correctness gate warms the default specialization. Neither this
        # setup nor first sweep calls establish a cold Triton cache measurement.
        report["timing_contract"]["default_reference_setup"] = (
            "default first call includes any JIT/cache load and precedes sweep; persistent Triton cache is not cleared"
        )
        for num_splits in [None, *dict.fromkeys(args.splits)]:
            row: dict[str, Any] = {
                "requested_splits": num_splits,
                "plan": asdict(plan_split_kv(args.queries, args.cached, num_splits=num_splits)),
            }
            report["results"].append(row)

            def call(num_splits: int | None = num_splits) -> torch.Tensor:
                return backend.attend_split(*tensors, key_padding_mask=mask, num_splits=num_splits)

            row["first_sweep_call_host_ms"] = _warmup(call, stream, 1)
            actual = call()
            row["reference"] = error_metrics(actual, reference, atol=args.atol, rtol=args.rtol)
            row["legacy_default"] = error_metrics(actual, default, atol=args.atol, rtol=args.rtol)
            row["passed"] = row["reference"]["passed"] and row["legacy_default"]["passed"]
            if not row["passed"]:
                continue
            row["eager_warmup_host_ms"] = _warmup(call, stream, args.warmup)
            row["eager"] = _measure(call, stream, args.iterations)
            if args.capture:
                graph = torch.cuda.CUDAGraph()
                stream.synchronize()
                before = time.perf_counter()
                with torch.cuda.graph(graph, stream=stream):
                    captured = call()
                stream.synchronize()
                row["capture_host_ms"] = (time.perf_counter() - before) * 1000
                graph.replay()
                stream.synchronize()
                row["graph_correctness"] = error_metrics(captured, actual, atol=0, rtol=0)
                row["passed"] = row["graph_correctness"]["bit_exact"]
                if row["passed"]:
                    row["graph_warmup_host_ms"] = _warmup(graph.replay, stream, args.warmup)
                    row["graph_replay"] = _measure(graph.replay, stream, args.iterations)
                del graph, captured
    report["passed"] = all(row["passed"] for row in report["results"])
    return report


def main(argv: list[str] | None = None) -> int:
    """Write the complete report and fail the process if an accuracy gate fails."""
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Accuracy gates {'passed' if report['passed'] else 'FAILED'}; report: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
