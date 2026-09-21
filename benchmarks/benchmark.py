"""Inference benchmark: embodiinfer engine vs a HuggingFace-style baseline.

A top-level tool (run ``python benchmarks/benchmark.py --preset small --sweep``), a
sibling of ``examples/`` and ``tests/`` — not part of the importable ``embodiinfer`` library.
It exercises the engine on the synthetic ``mock_flow_vla`` policy to isolate each
engine mechanism; real-model benchmarks live in the box scripts.

Baseline ("hf"): the way you would run a flow VLA with plain HF-style code — one
observation at a time (batch 1), eager, no CUDA graph. Prefix-within-inference
reuse is inherent to the model and present in every config, so what the engine
adds on top is measured cleanly:

    hf            : batch-1, eager                      (baseline)
    embodiinfer-eager    : cross-request batching, eager        (isolates batching)
    embodiinfer-graph    : cross-request batching + CUDA graph  (isolates graph capture)

Reports throughput (obs/s, actions/s) and per-inference latency (p50/p99).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.policies.mock import MockFlowVLA, preset_config
from embodiinfer.types import Observation, collate
from embodiinfer.utils import now_ns, percentiles, resolve_device, sync_if_cuda, torch_dtype


def _make_workload(cfg, n: int) -> list[Observation]:
    rng = np.random.RandomState(0)
    obs = []
    for i in range(n):
        obs.append(
            Observation(
                images=torch.from_numpy(
                    rng.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size).astype(np.float32)
                ),
                state=torch.from_numpy(rng.rand(cfg.state_dim).astype(np.float32)),
                instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
                env_id=i,
            )
        )
    return obs


@torch.no_grad()
def _hf_baseline(policy, obs, num_steps, device, dtype, iters, warmup):
    H, A = policy.config.action_horizon, policy.config.action_dim
    lat = []
    for it in range(iters + warmup):
        t0 = now_ns()
        for o in obs:
            batch = collate([o], ["b"]).to(device)
            batch.images = batch.images.to(dtype)
            batch.state = batch.state.to(dtype)
            x0 = torch.randn(1, H, A, device=device, dtype=dtype)
            policy.sample_actions(batch, num_steps=num_steps, x0=x0)
        sync_if_cuda(device)
        if it >= warmup:
            lat.append((now_ns() - t0) / 1e6)
    return lat


def _engine_run(core, obs, num_steps, iters, warmup):
    batch = collate(obs, [f"e{i}" for i in range(len(obs))])
    lat = []
    for it in range(iters + warmup):
        t0 = now_ns()
        core.execute(batch, num_steps=num_steps)
        sync_if_cuda(core.device)
        if it >= warmup:
            lat.append((now_ns() - t0) / 1e6)
    return lat


def run_benchmark(
    preset="small", num_envs=16, num_steps=10, iters=20, warmup=5, device="cuda", dtype="bfloat16"
):
    dev = resolve_device(device)
    dt = torch_dtype(dtype) if dev.type == "cuda" else torch.float32
    cfg = preset_config(preset)
    H = cfg.action_horizon

    policy = MockFlowVLA(cfg).to(dev).to(dt).eval()
    obs = _make_workload(cfg, num_envs)
    n_params = sum(p.numel() for p in policy.parameters())

    print("\n=== embodiinfer inference benchmark ===")
    print(
        f"preset={preset}  params={n_params / 1e6:.1f}M  num_envs={num_envs}  "
        f"num_steps={num_steps}  horizon={H}  device={dev}  dtype={dtype}\n"
    )

    rows = []

    # baseline
    lat = _hf_baseline(policy, obs, num_steps, dev, dt, iters, warmup)
    rows.append(("hf (bs=1, eager)", lat))

    # embodiinfer eager (batched, no graph)
    core_e = EngineCore(
        policy, EngineConfig(device=device, dtype=dtype, max_batch_size=num_envs, use_cuda_graph=False)
    )
    rows.append(("embodiinfer-eager (batched)", _engine_run(core_e, obs, num_steps, iters, warmup)))

    # embodiinfer graph (batched + CUDA graph)
    if dev.type == "cuda":
        core_g = EngineCore(
            policy, EngineConfig(device=device, dtype=dtype, max_batch_size=num_envs, use_cuda_graph=True)
        )
        rows.append(("embodiinfer-graph (batched+cudagraph)", _engine_run(core_g, obs, num_steps, iters, warmup)))

    base_ms = float(np.mean(rows[0][1]))
    hdr = f"{'config':32s} {'obs/s':>10s} {'actions/s':>12s} {'p50(ms)':>9s} {'p99(ms)':>9s} {'speedup':>8s}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, lat in rows:
        pc = percentiles(lat)
        mean_ms = pc["mean"]
        obs_s = num_envs / (mean_ms / 1e3)
        act_s = obs_s * H
        speedup = base_ms / mean_ms
        print(f"{name:32s} {obs_s:10.1f} {act_s:12.0f} {pc['p50']:9.2f} {pc['p99']:9.2f} {speedup:7.2f}x")
        results[name] = dict(obs_per_s=obs_s, actions_per_s=act_s, **pc, speedup=speedup)
    print()
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--sweep", action="store_true", help="sweep num_envs")
    args = p.parse_args(argv)

    if args.sweep:
        for n in [1, 2, 4, 8, 16, 32, 64]:
            run_benchmark(args.preset, n, args.num_steps, args.iters, args.warmup, args.device, args.dtype)
    else:
        run_benchmark(
            args.preset, args.num_envs, args.num_steps, args.iters, args.warmup, args.device, args.dtype
        )


if __name__ == "__main__":
    main()
