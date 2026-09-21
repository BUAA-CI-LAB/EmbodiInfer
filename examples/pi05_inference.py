"""Real pi0.5 inference through the engine (single GPU or data-parallel).

Loads a LeRobot ``pi05`` checkpoint as a first-class embodiinfer policy (embodiinfer owns the
transformer forward; LeRobot only builds and loads the weights) and runs a batch
of observations through :class:`EngineCore`. With ``--gpus > 1`` it holds one
replica per GPU behind a :class:`DataParallelEngine` for lossless throughput
scaling (near-linear across devices).

    python examples/pi05_inference.py --ckpt lerobot/pi05_base --envs 8
    python examples/pi05_inference.py --ckpt lerobot/pi05_base --gpus 4 --envs 32

Requires a CUDA device and the ``pi05`` dependency group
(``uv sync --frozen --no-dev --group pi05``).
"""

from __future__ import annotations

import argparse
import time

import torch

from embodiinfer import DataParallelEngine, EngineConfig, Observation, make_policy
from embodiinfer.engine import EngineCore


def make_obs(n: int, lang_len: int) -> list[Observation]:
    """A batch of synthetic pi0.5-shaped observations (3 cameras, 32-d state)."""
    return [
        Observation(
            images=torch.rand(3, 3, 224, 224),
            state=torch.zeros(32),
            instruction_tokens=torch.randint(0, 257152, (lang_len,)),
        )
        for _ in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lerobot/pi05_base", help="LeRobot pi0.5 checkpoint (path or hub id)")
    ap.add_argument("--envs", type=int, default=8, help="number of observations in the batch")
    ap.add_argument("--gpus", type=int, default=1, help="replicas across cuda:0..N-1 (data parallel)")
    ap.add_argument("--lang-len", type=int, default=48)
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("pi0.5 inference needs a CUDA device")
    if torch.cuda.device_count() < args.gpus:
        raise SystemExit(f"requested {args.gpus} GPUs but only {torch.cuda.device_count()} available")

    obs = make_obs(args.envs, args.lang_len)

    if args.gpus == 1:
        policy = make_policy("pi05", checkpoint=args.ckpt, attention=args.attn)
        core = EngineCore(policy, EngineConfig(device="cuda", max_batch_size=args.envs))
        print(f"[pi05] device={core.device} dtype={core.dtype} attn={args.attn}")
        ids = [f"env{i}" for i in range(args.envs)]

        def run():
            return core.execute(policy.collate(obs, ids))
    else:
        cores = []
        for i in range(args.gpus):
            policy = make_policy("pi05", checkpoint=args.ckpt, attention=args.attn)
            cores.append(EngineCore(policy, EngineConfig(device=f"cuda:{i}", max_batch_size=args.envs)))
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
