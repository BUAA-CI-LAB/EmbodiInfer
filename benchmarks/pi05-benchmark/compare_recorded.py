"""Compare LeRobot and VVLA on local recorded observations, without a robot service."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.checkpoints.lerobot import load_lerobot_checkpoint
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

MODES = ("lerobot", "vvla_eager", "vvla_eager_graph", "vvla_sdpa", "vvla_sdpa_graph")


def read_observation(path: Path) -> dict[str, Any]:
    """Read sample.json state/task and camera-role-to-image-file mappings."""
    sample = json.loads(path.read_text())
    observation = {
        "observation.state": torch.tensor(sample["state"], dtype=torch.float32),
        "task": sample["task"],
    }
    for role, filename in sample["images"].items():
        with Image.open(path.parent / filename) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        observation[f"observation.images.{role}"] = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
    return observation


def main() -> None:
    """Save per-request timing, complete actions and paired errors after per-mode warmup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", type=Path, nargs="+", required=True, help="Recorded sample.json files")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 0, 2026])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="highest")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this measurement requires CUDA")
    if args.warmup < 1 or args.modes[0] != "lerobot" or len(args.modes) != len(set(args.modes)):
        parser.error("use at least one warmup and unique modes with lerobot first")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision(args.matmul_precision)

    # The checkpoint owns normalization, tokenization and physical action units.
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    reference = load_lerobot_checkpoint(args.checkpoint, load_device="cuda", low_cpu_mem_usage=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    pre = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename="policy_preprocessor.json",
        overrides={
            "tokenizer_processor": {"tokenizer_name": args.tokenizer},
            "device_processor": {"device": "cuda"},
        },
    )
    post = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    config = reference.config
    cfg = VLAPolicyConfig(
        name="pi0.5",
        action_dim=config.max_action_dim,
        action_horizon=config.chunk_size,
        default_num_steps=config.num_inference_steps,
    )
    dim = config.output_features["action"].shape[0]
    observations = [read_observation(path) for path in args.samples]
    expected: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "lerobot": importlib.metadata.version("lerobot"),
        "gpu": torch.cuda.get_device_name(),
        "checkpoint": args.checkpoint,
        "samples": [str(p) for p in args.samples],
        "seeds": args.seeds,
        "warmup": args.warmup,
        "modes": args.modes,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "batch_size": 1,
        "steps": cfg.default_num_steps,
        "horizon": cfg.action_horizon,
        "action_dim": dim,
        "parameter_dtypes": sorted({str(p.dtype) for p in reference.parameters()}),
        "placement": "all parameters on CUDA; no CPU embedding offload",
    }
    (args.out / "conditions.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for mode in args.modes:
        engine = None
        if mode != "lerobot":
            policy = Pi05Policy(
                cfg, reference, attention="eager" if "eager" in mode else "sdpa", native_embeddings=True
            )
            before = {name: p.dtype for name, p in policy.named_parameters()}
            graph = mode.endswith("_graph")
            engine = EngineCore(
                policy,
                EngineConfig(
                    device="cuda",
                    dtype="auto",
                    max_batch_size=1,
                    batch_buckets=(1,),
                    use_cuda_graph=graph,
                    capture_full_loop=graph,
                ),
            )
            assert before == {name: p.dtype for name, p in policy.named_parameters()}
            for seed in args.seeds:
                noise = torch.randn(
                    (1, cfg.action_horizon, cfg.action_dim),
                    device="cuda",
                    dtype=torch.float32,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                )
                assert torch.equal(
                    noise, policy.new_noise(1, torch.Generator(device="cuda").manual_seed(seed))
                )

        def run(
            index: int, seed: int, engine: EngineCore | None = engine, mode: str = mode
        ) -> tuple[np.ndarray, np.ndarray, float, dict]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            batch = pre(observations[index])
            generator = torch.Generator(device="cuda").manual_seed(seed)
            timing = {}
            if engine is None:
                noise = torch.randn(
                    (1, cfg.action_horizon, cfg.action_dim),
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                normalized = reference.predict_action_chunk(batch, noise=noise)
            else:
                native_batch = Pi05Batch.from_lerobot_batch(reference, batch)
                native_batch.request_ids = [str(index)]
                chunk = engine.execute(native_batch, generator=generator)[0]
                normalized = chunk.actions[None, :, :dim].cuda()
                timing = dict(chunk.timing)
            actions = post(normalized.clone()).float().cpu().numpy()
            normalized = normalized.float().cpu().numpy()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if not np.isfinite(actions).all() or not np.isfinite(normalized).all():
                raise RuntimeError(f"nonfinite output in {mode}")
            return normalized, actions, elapsed, timing

        with torch.inference_mode():
            # Warm every shape before timing; prompts may have different lengths.
            for index in range(len(observations)):
                for _ in range(args.warmup):
                    run(index, args.seeds[0])
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for index in range(len(observations)):
                for seed in args.seeds:
                    norm, actions, elapsed, timing = run(index, seed)
                    if mode == "lerobot":
                        expected[index, seed] = (norm.copy(), actions.copy())
                    ref_norm, ref_actions = expected[index, seed]
                    row = {
                        "mode": mode,
                        "sample": index,
                        "seed": seed,
                        "wall_ms": elapsed,
                        "timing": timing,
                        "max_norm_diff": float(np.max(np.abs(norm - ref_norm))),
                        "max_action_diff": float(np.max(np.abs(actions - ref_actions))),
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    }
                    rows.append(row)
                    np.savez(args.out / f"{mode}-{index}-{seed}.npz", normalized=norm, actions=actions)
            with (args.out / "results.jsonl").open("a") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "samples": len(rows),
                        "p50_ms": float(np.percentile([r["wall_ms"] for r in rows], 50)),
                        "p95_ms": float(np.percentile([r["wall_ms"] for r in rows], 95)),
                        "max_action_diff": max(r["max_action_diff"] for r in rows),
                    }
                ),
                flush=True,
            )
        del run, engine
        if mode != "lerobot":
            del policy
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
