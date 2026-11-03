"""FlashRT v0.2.0 examples/thor/pi05.py with only model loading and run() swapped to EmbodiInfer.

The inputs, prompt, warmup, seeded action pass, timed loop, action unnormalization and report are
FlashRT's own, so EmbodiInfer and FlashRT share one harness. The seeded action pass feeds FlashRT's
float16 NumPy noise, so the actions are comparable across engines.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROMPT = "pick up the black bowl and place it on the plate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashrt-root", type=Path, default=os.environ.get("FLASHRT_ROOT"))
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="RLinf-Pi05-LIBERO-SFT, LeRobot layout"
    )
    parser.add_argument(
        "--norm-stats", type=Path, help="quantile norm_stats.json used to unnormalize actions"
    )
    parser.add_argument(
        "--preset", default="nvfp4-nvfp4", help="Thor preset, e.g. bf16, fp8-fp8, nvfp4-nvfp4"
    )
    parser.add_argument(
        "--input", type=Path, help="defaults to FlashRT's examples/thor/assets/pi05_libero.npz"
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=os.environ.get(
            "FLASH_RT_PALIGEMMA_TOKENIZER", Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"
        ),
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/flashrt_harness/pi05.npz"))
    parser.add_argument("--reference", type=Path, help="actions of another engine to compare against")
    args = parser.parse_args()
    if args.flashrt_root is None:
        parser.error("set --flashrt-root or FLASHRT_ROOT to a FlashRT v0.2.0 checkout")
    if args.warmup < 0 or args.iters < 1:
        parser.error("warmup must be nonnegative and iters must be positive")
    for name in ("EMBODIINFER_CUTLASS_DIR", "EMBODIINFER_FA4_DIR"):
        if not os.environ.get(name):
            parser.error(f"{name} must be set; without it the Thor kernels fall back to slower paths")

    flashrt = Path(args.flashrt_root).resolve()
    sys.path[:0] = [str(REPO), str(flashrt / "examples" / "thor"), str(flashrt)]
    import sentencepiece
    import torch
    from _validation import metrics, save_report
    from flash_rt.core.utils.actions import unnormalize_actions
    from flash_rt.core.utils.norm_stats import lerobot_candidates, load_norm_stats

    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.config import VLAPolicyConfig
    from embodiinfer.policies.pi05 import Pi05OptimizationConfig
    from embodiinfer.policies.pi05.checkpoints.lerobot import load_lerobot_checkpoint
    from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy

    data = np.load(args.input or flashrt / "examples/thor/assets/pi05_libero.npz", allow_pickle=False)
    observations = [
        dict(image=data[f"img_{i}"], wrist_image=data[f"wrist_{i}"]) for i in range(int(data["n"]))
    ]
    checkpoint = args.checkpoint
    model = load_lerobot_checkpoint(str(checkpoint), load_device="cuda", low_cpu_mem_usage=True)
    policy = Pi05Policy(
        VLAPolicyConfig(name="pi0.5", action_dim=32, action_horizon=10, default_num_steps=10),
        model,
        native_embeddings=True,
        native_inference=True,
        prefix_cuda_graph=True,
        request_cuda_graph=True,
        compile_backend="none",
        prefix_attention="sdpa",
        denoise_attention="sdpa",
        optimizations=Pi05OptimizationConfig.from_preset("thor", args.preset),
    )
    policy.checkpoint = str(checkpoint)
    # Places the policy in CUDA eval mode, which static requests require.
    _engine = EngineCore(
        policy,
        EngineConfig(
            device="cuda",
            dtype="auto",
            max_batch_size=1,
            batch_buckets=(1,),
            use_cuda_graph=True,
            capture_full_loop=True,
        ),
    )
    norm_stats = load_norm_stats(
        [args.norm_stats]
        if args.norm_stats
        else [
            checkpoint / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint / "norm_stats.json",
            *lerobot_candidates(checkpoint),
        ],
        checkpoint_dir=checkpoint,
    )

    # Counterpart of FlashRT's set_prompt + infer: the prompt is encoded once and every call is one
    # graph from pinned uint8 frames to actions.
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    ids = tokenizer.encode(PROMPT, add_bos=True) + tokenizer.encode("\n")
    print(f"prompt tokens: {len(ids)}", flush=True)
    with torch.inference_mode():
        session = policy.static_request(ids, cameras=2)
    frames = session.frames.numpy()

    def run(index, noise=None):
        obs = observations[index]
        np.copyto(frames[0], obs["image"])
        np.copyto(frames[1], obs["wrist_image"])
        raw_actions = session.run(noise=noise)[0].float().numpy()
        return unnormalize_actions(raw_actions, norm_stats)[:, :7]

    for index in range(args.warmup):
        run(index % len(observations))
    actions = []
    for index in range(len(observations)):
        # FlashRT's infer draws its flow noise as np.random.randn(10, 32) in float16.
        np.random.seed(20261007 + index)
        noise = torch.from_numpy(np.random.randn(10, 32).astype(np.float16).astype(np.float32))[None]
        actions.append(np.asarray(run(index, noise)))
    actions = np.stack(actions)
    if actions.shape != (8, 10, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Unexpected actions: {actions.shape}")
    samples = []
    for index in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run(index % len(observations))
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

    report = dict(
        preset=args.preset,
        shape=list(actions.shape),
        warmup=args.warmup,
        iters=args.iters,
        latency_ms=float(np.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
    )
    if args.reference:
        report.update(metrics(actions, np.load(args.reference, allow_pickle=False)["actions"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, actions=actions, samples_ms=samples)
    save_report(args.output.with_suffix(".json"), report)
    print(f"p50={report['latency_ms']:.3f} ms p95={report['p95_ms']:.3f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
