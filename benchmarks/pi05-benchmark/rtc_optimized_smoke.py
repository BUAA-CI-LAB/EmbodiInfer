"""Fixed-input PI0.5 soft-RTC parity and latency smoke on one CUDA device.

The candidate uses the repository's documented unquantized native recipe.  The
soft RTC VJP deliberately falls back from native denoise graphs/kernels, so the
report records both policy-owned and engine-owned graph counts instead of
assuming that a requested optimization was used.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty list")
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": statistics.fmean(row[key] for row in rows),
            "median": statistics.median(row[key] for row in rows),
            "p95": _percentile([row[key] for row in rows], 0.95),
        }
        for key in ("wall_ms", "prefill_ms", "decode_ms", "e2e_ms")
    }


def _fixed_batch(policy: Any, checkpoint: str, seed: int):
    from embodiinfer.policies.pi05.processor_pi05 import make_processor

    config = policy._lerobot.config
    state_feature = config.input_features["observation.state"]
    state_dim = int(state_feature.shape[0])
    state = torch.linspace(-0.25, 0.25, state_dim, dtype=torch.float32)
    processor = make_processor(policy, checkpoint)
    state = processor.prepare_state(state)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    images: dict[str, torch.Tensor] = {}
    for name, feature in config.image_features.items():
        if "empty_camera" in name:
            continue
        channels, height, width = map(int, feature.shape)
        images[name] = torch.rand((channels, height, width), generator=generator)
    if not images:
        raise RuntimeError("checkpoint has no real camera features")
    batch = processor.prepare(state, images, "pick up the black bowl")
    batch.request_ids = ["rtc-optimized-smoke"]
    return batch


def _run(engine: Any, batch: Any, guidance: Any, seed: int) -> tuple[torch.Tensor, dict[str, float]]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    chunks = engine.execute(
        batch,
        num_steps=10,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        rtc_guidance=guidance,
    )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    if len(chunks) != 1:
        raise RuntimeError("smoke expects batch size one")
    chunk = chunks[0]
    timing = {key: float(chunk.timing[key]) for key in ("prefill_ms", "decode_ms", "e2e_ms")}
    return chunk.actions.clone(), {"wall_ms": wall_ms, **timing}


def _graph_counts(policy: Any, engine: Any) -> dict[str, Any]:
    runtime = policy._runtime.stats() if getattr(policy, "native_inference", False) else None
    manager = getattr(engine, "_graphs", None)
    return {
        "policy_runtime": runtime,
        "engine_manager_created": manager is not None,
        "engine_captured_graphs": len(getattr(manager, "_graphs", {})) if manager is not None else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-abs-gate", type=float, default=0.1)
    parser.add_argument("--rmse-gate", type=float, default=0.02)
    args = parser.parse_args()
    if min(args.warmup, args.iterations) < 1 or args.delay < 0:
        parser.error("warmup/iterations must be positive and delay must be nonnegative")
    if min(args.max_abs_gate, args.rmse_gate) <= 0:
        parser.error("numerical gates must be positive")

    from embodiinfer.engine.async_inference.rtc import (
        PrefixAttentionSchedule,
        RTCGuidanceConfig,
        build_rtc_guidance,
    )
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.config import VLAPolicyConfig
    from embodiinfer.policies.factory import make_policy
    from embodiinfer.policies.pi05 import modeling_pi05
    from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy

    checkpoint = str(args.checkpoint.resolve())
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    baseline = make_policy(
        "pi05",
        checkpoint=checkpoint,
        load_device="cuda",
        attention="sdpa",
        compile_backend="none",
        native_inference=False,
    )
    batch = _fixed_batch(baseline, checkpoint, args.seed)
    baseline_engine = EngineCore(
        baseline,
        EngineConfig(
            device="cuda",
            dtype="auto",
            max_batch_size=1,
            batch_buckets=(1,),
            num_steps=10,
            use_cuda_graph=False,
            capture_full_loop=False,
        ),
    )

    previous, _ = _run(baseline_engine, batch, None, args.seed - 1)
    previous = previous[None, 10:, :]
    guidance_config = RTCGuidanceConfig(
        execution_horizon=10,
        prefix_attention_schedule=PrefixAttentionSchedule.EXP,
        max_guidance_weight=5.0,
        hard_prefix=False,
    )
    guidance = build_rtc_guidance(
        previous,
        args.delay,
        guidance_config,
        action_horizon=baseline.config.action_horizon,
        action_dim=baseline.config.action_dim,
    )

    for _ in range(args.warmup):
        _run(baseline_engine, batch, guidance, args.seed)
    baseline_rows = []
    baseline_action = None
    for _ in range(args.iterations):
        action, timing = _run(baseline_engine, batch, guidance, args.seed)
        baseline_action = action if baseline_action is None else baseline_action
        baseline_rows.append(timing)

    candidate_config = VLAPolicyConfig(
        name="pi0.5",
        action_dim=baseline.config.action_dim,
        action_horizon=baseline.config.action_horizon,
        default_num_steps=baseline.config.default_num_steps,
    )
    candidate = Pi05Policy(
        candidate_config,
        baseline._lerobot,
        attention="sdpa",
        native_inference=True,
        prefix_cuda_graph=True,
        prefix_attention="triton",
        denoise_attention="triton",
        compile_backend="inductor",
    )
    candidate.checkpoint = checkpoint
    candidate_engine = EngineCore(
        candidate,
        EngineConfig(
            device="cuda",
            dtype="auto",
            max_batch_size=1,
            batch_buckets=(1,),
            num_steps=10,
            use_cuda_graph=True,
            capture_full_loop=True,
        ),
    )
    for _ in range(args.warmup):
        _run(candidate_engine, batch, guidance, args.seed)
    candidate_rows = []
    candidate_action = None
    for _ in range(args.iterations):
        action, timing = _run(candidate_engine, batch, guidance, args.seed)
        candidate_action = action if candidate_action is None else candidate_action
        candidate_rows.append(timing)

    if baseline_action is None or candidate_action is None:
        raise AssertionError("measurement produced no actions")
    difference = candidate_action.float() - baseline_action.float()
    max_abs = float(difference.abs().max())
    rmse = float(difference.square().mean().sqrt())
    finite = bool(torch.isfinite(candidate_action).all())
    passed = finite and max_abs <= args.max_abs_gate and rmse <= args.rmse_gate

    report = {
        "schema": "embodiinfer.pi05.rtc_optimized_smoke.v1",
        "checkpoint": checkpoint,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "rtc": {
            "num_steps": 10,
            "cache_start_index": 10,
            "inference_delay": args.delay,
            **asdict(guidance_config),
        },
        "baseline": {
            "policy": {
                "attention": "sdpa",
                "native_inference": False,
                "compile_backend": "none",
            },
            "timing_ms": _summary(baseline_rows),
            "graphs": _graph_counts(baseline, baseline_engine),
        },
        "candidate": {
            "policy": {
                "attention": "sdpa",
                "native_inference": True,
                "prefix_cuda_graph": True,
                "prefix_attention": "triton",
                "denoise_attention": "triton",
                "compile_backend": "inductor",
                "engine_cuda_graph": True,
                "capture_full_loop": True,
            },
            "timing_ms": _summary(candidate_rows),
            "graphs": _graph_counts(candidate, candidate_engine),
            "compiled_cache_counts": {
                "image_encoders": len(modeling_pi05._COMPILED_IMAGE_ENCODERS),
                "prefix_encoders": len(modeling_pi05._COMPILED_PREFIX_ENCODERS),
                "inference_helpers": len(modeling_pi05._COMPILED_INFERENCE_HELPERS),
            },
        },
        "parity": {
            "finite": finite,
            "max_abs": max_abs,
            "rmse": rmse,
            "max_abs_gate": args.max_abs_gate,
            "rmse_gate": args.rmse_gate,
            "passed": passed,
        },
        "soft_rtc_fallbacks": {
            "engine_decode_cuda_graph": True,
            "native_fixed_schedule": True,
            "native_denoise_attention": True,
            "fused_qkv_and_mlp_projections": True,
            "reason": "the exact VJP runs with gradients enabled on the generic eager decoder path",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False, default=str), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
