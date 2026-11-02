"""Real pi0.5 inference through the engine (single GPU or data-parallel).

Loads a LeRobot ``pi05`` checkpoint as a first-class embodiinfer policy (embodiinfer owns the
transformer forward; LeRobot only builds and loads the weights) and runs a batch
of observations through :class:`EngineCore`. With ``--gpus > 1`` it holds one
replica per GPU behind a :class:`DataParallelEngine` for lossless throughput
scaling (near-linear across devices).

    python examples/pi05_inference.py --ckpt lerobot/pi05_base --envs 8
    python examples/pi05_inference.py --ckpt lerobot/pi05_base --gpus 4 --envs 32
    python examples/pi05_inference.py --ckpt /models/RLinf-Pi05-LIBERO-SFT \
        --device-type thor --preset nvfp4-fp8 --envs 1
    python examples/pi05_inference.py --ckpt lerobot/pi05_base --envs 1 \
        --cameras 2 --lang-len 160 --dtype bfloat16 --kernel-profile orin_pi05_r2

Requires a CUDA device and the ``pi05`` dependency group
(``uv sync --frozen --no-dev --group pi05``). A tuned kernel profile loads only on
the device and Torch/Triton versions recorded in its ``profile.json``, and each of
its kernels serves only the layouts it was verified on (other calls keep the
default implementation), hence the batch size and shapes above. ``--kernel-profile
none`` runs the same path without tuned kernels.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from embodiinfer import DataParallelEngine, EngineConfig, Observation, make_policy
from embodiinfer.engine import EngineCore


def make_obs(n: int, lang_len: int, cameras: int = 3) -> list[Observation]:
    """A batch of synthetic pi0.5-shaped observations (32-d state)."""
    return [
        Observation(
            images=torch.rand(cameras, 3, 224, 224),
            state=torch.zeros(32),
            instruction_tokens=torch.randint(0, 257152, (lang_len,)),
        )
        for _ in range(n)
    ]


def release_weight_cache(checkpoint: str) -> None:
    """Drop the weight file's page cache, which crowds out CUDA allocations in Orin's unified memory."""
    from transformers.utils import cached_file

    if hasattr(os, "posix_fadvise"):
        with open(cached_file(checkpoint, "model.safetensors"), "rb") as weights:
            os.posix_fadvise(weights.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lerobot/pi05_base", help="LeRobot pi0.5 checkpoint (path or hub id)")
    ap.add_argument("--envs", type=int, default=8, help="number of observations in the batch")
    ap.add_argument("--gpus", type=int, default=1, help="replicas across cuda:0..N-1 (data parallel)")
    ap.add_argument("--lang-len", type=int, default=48)
    ap.add_argument("--cameras", type=int, default=3, help="camera views per synthetic observation")
    ap.add_argument("--dtype", default="auto", help="auto keeps checkpoint dtypes; bfloat16 casts all")
    ap.add_argument("--kernel-profile", help="none, or a tuned kernel profile such as orin_pi05_r2")
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"])
    ap.add_argument("--device-type", choices=["thor", "spark", "orin", "4090"])
    ap.add_argument("--preset", help="strict, bf16 or a prefix-action format pair such as nvfp4-fp8")
    ap.add_argument("--calibration", help="Calibration name or data path")
    ap.add_argument("--horizon", type=int, help="Inference action horizon; optimized profiles require 10")
    ap.add_argument("--steps", type=int, default=10, help="Number of denoising steps")
    ap.add_argument("--optimizations", type=Path, help="Advanced: standalone operator recipe JSON")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("pi0.5 inference needs a CUDA device")
    if torch.cuda.device_count() < args.gpus:
        raise SystemExit(f"requested {args.gpus} GPUs but only {torch.cuda.device_count()} available")
    policy_options = {}
    if args.preset is not None and args.optimizations is not None:
        ap.error("choose --preset or an advanced --optimizations recipe")
    if args.preset is not None and args.device_type is None:
        ap.error("--preset requires --device-type")
    if args.preset is None and (args.device_type is not None or args.calibration is not None):
        ap.error("--device-type and --calibration require --preset")
    if args.preset is not None or args.optimizations is not None:
        from embodiinfer.policies.pi05 import Pi05OptimizationConfig

        if args.attn != "sdpa":
            ap.error("fused inference requires --attn sdpa")
        try:
            options = (
                Pi05OptimizationConfig.from_json(args.optimizations)
                if args.optimizations is not None
                else Pi05OptimizationConfig.from_preset(
                    args.device_type,
                    args.preset,
                    calibration=args.calibration,
                )
            )
        except ValueError as exc:
            ap.error(str(exc))
        if options.fused_mlp and (args.envs != 1 or args.horizon not in (None, 10) or args.steps != 10):
            ap.error("optimized presets require --envs 1, horizon 10 and 10 denoising steps")
        policy_options = dict(optimizations=options)
    policy_options["default_num_steps"] = args.steps
    if args.horizon is not None:
        policy_options["action_horizon"] = args.horizon
    if args.kernel_profile is not None:
        if args.preset is not None or args.optimizations is not None:
            ap.error("choose --kernel-profile or a --preset/--optimizations recipe")
        # Native inference with Triton attention and prefix/denoise graphs, as the profile was measured.
        policy_options.update(
            kernel_profile=args.kernel_profile,
            native_inference=True,
            prefix_cuda_graph=True,
            prefix_attention="triton",
            denoise_attention="triton",
            low_cpu_mem_usage=True,
            load_device="cpu",
        )
    use_graphs = args.preset is not None or args.optimizations is not None or args.kernel_profile is not None

    obs = make_obs(args.envs, args.lang_len, args.cameras)

    def load_policy():
        policy = make_policy("pi05", checkpoint=args.ckpt, attention=args.attn, **policy_options)
        if args.dtype != "auto":
            # Cast before the engine moves the weights: Orin's unified memory cannot hold two FP32 copies.
            policy.to(dtype=getattr(torch, args.dtype))
            release_weight_cache(args.ckpt)
        return policy

    if args.gpus == 1:
        policy = load_policy()
        core = EngineCore(
            policy,
            EngineConfig(device="cuda", max_batch_size=args.envs, capture_full_loop=use_graphs),
        )
        print(f"[pi05] device={core.device} dtype={core.dtype} attn={args.attn}")
        ids = [f"env{i}" for i in range(args.envs)]

        def run():
            return core.execute(policy.collate(obs, ids))
    else:
        cores = []
        for i in range(args.gpus):
            policy = load_policy()
            cores.append(
                EngineCore(
                    policy,
                    EngineConfig(device=f"cuda:{i}", max_batch_size=args.envs, capture_full_loop=use_graphs),
                )
            )
        engine = DataParallelEngine(cores)
        print(f"[pi05] {engine.num_replicas} replicas, {args.envs} obs sharded across them")

        def run():
            return engine.execute(obs)

    run()  # warmup (loads CUDA kernels, captures nothing here)
    t0 = time.perf_counter()
    chunks = run()
    dt = time.perf_counter() - t0

    print(f"[pi05] {len(chunks)} chunks, each {tuple(chunks[0].actions.shape)}")
    print(f"[pi05] {args.envs} obs in {dt * 1e3:.1f} ms -> {args.envs / dt:.1f} obs/s")


if __name__ == "__main__":
    main()
