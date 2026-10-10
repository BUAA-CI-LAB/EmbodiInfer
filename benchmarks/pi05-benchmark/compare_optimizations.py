"""Measure a frozen pre-PR BF16 checkout and optimized recipes in separate processes.

Run once per checkout and FP32 matmul setting with the same checkpoint, fixture,
warmup and iteration count. The baseline retains its original Inductor, Triton
attention and CUDA Graphs. No benchmark helper imports optimized implementation
code into that process. Results include full actions for separate drift analysis.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch


def digest(path: Path) -> str:
    """Hash an input artifact without depending on its machine-specific path."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_digest(root: Path) -> str:
    """Identify all Python and native sources actually available to a worker."""
    value = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h"):
            value.update(path.relative_to(root).as_posix().encode() + b"\0")
            value.update(path.read_bytes())
    return value.hexdigest()


@torch.inference_mode()
def main() -> None:
    """Time only warmed GPU-input-to-action requests from one isolated checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, help="Optimized combined NVFP4-prefix/FP8-action recipe")
    parser.add_argument("--device", choices=("thor", "spark"), help="Candidate preset device")
    parser.add_argument("--calibration", default="rlinf_libero", help="Calibration name or data path")
    parser.add_argument("--baseline-revision", help="Recorded immutable revision of the baseline export")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.rounds) < 1:
        parser.error("warmup, iterations and rounds must be positive")
    if args.role == "candidate" and (args.recipe is None) == (args.device is None):
        parser.error("candidate requires either --device or a custom --recipe")
    if args.role == "baseline" and (
        args.recipe is not None or args.device is not None or args.baseline_revision is None
    ):
        parser.error("baseline requires --baseline-revision and forbids candidate settings")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    sys.path.insert(0, str(source))

    import embodiinfer
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.config import VLAPolicyConfig
    from embodiinfer.policies.pi05.checkpoints.lerobot import load_lerobot_checkpoint
    from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
    from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

    if Path(embodiinfer.__file__).resolve().parent != source / "embodiinfer":
        raise RuntimeError("Imported package does not belong to the requested isolated checkout")
    if args.role == "baseline" and hasattr(
        sys.modules["embodiinfer.policies.pi05"], "Pi05OptimizationConfig"
    ):
        raise RuntimeError("Baseline checkout contains the new optimization implementation")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(42)
    fixture = torch.load(args.inputs, map_location="cpu", weights_only=False)
    indices = fixture["validation_indices"]
    batches, noises = {}, {}
    for index in indices:
        item = fixture["observations"][index]
        images = [image.cuda() for image in item["images"]]
        batches[index] = Pi05Batch(
            images + [-torch.ones_like(images[0])],
            [torch.ones(1, dtype=torch.bool, device="cuda")] * 2
            + [torch.zeros(1, dtype=torch.bool, device="cuda")],
            item["tokens"].cuda(),
            item["masks"].cuda(),
        )
        noises[index] = torch.from_numpy(
            np.random.default_rng(1000 + index).standard_normal((1, 10, 32), dtype=np.float32)
        ).cuda()
    recipes = {"original_bf16_inductor": None}
    if args.role == "candidate":
        from embodiinfer.policies.pi05 import Pi05OptimizationConfig

        combined = (
            Pi05OptimizationConfig.from_json(args.recipe)
            if args.recipe is not None
            else Pi05OptimizationConfig.from_preset(
                args.device,
                "nvfp4-fp8",
                calibration=args.calibration,
            )
        )
        recipes = {
            "optimized_compat_bf16": Pi05OptimizationConfig(
                hardware=combined.hardware, fused_mlp=True, attention="folded_flash"
            ),
            "optimized_rlinf_bf16": replace(combined, action_layers=(), prefix_layers=()),
            "optimized_rlinf_fp8": replace(combined, prefix_layers=()),
            "optimized_rlinf_prefix_nvfp4": replace(combined, action_layers=()),
            "optimized_rlinf_combined": combined,
        }
    model = load_lerobot_checkpoint(str(args.checkpoint), load_device="cuda", low_cpu_mem_usage=True)
    report = dict(
        role=args.role,
        source_root=str(source),
        source_sha256=source_digest(source / "embodiinfer"),
        benchmark_sha256=digest(Path(__file__)),
        baseline_revision=args.baseline_revision,
        checkpoint_sha256=digest(args.checkpoint / "model.safetensors"),
        inputs_sha256=digest(args.inputs),
        hardware=torch.cuda.get_device_name(),
        versions={
            name: importlib.metadata.version(name) for name in ("torch", "triton", "lerobot", "transformers")
        },
        cuda=torch.version.cuda,
        matmul_precision=args.matmul_precision,
        batch_size=1,
        horizon=10,
        steps=10,
        indices=indices,
        warmup_per_observation=args.warmup,
        iterations_per_round=args.iterations,
        rounds=args.rounds,
        checkpoint_parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
        scope="GPU-ready recorded inputs through prefix and all ten denoise steps; no preprocessing, transfer, postprocessing or cold compilation/capture",
        modes={},
    )
    for name, config in recipes.items():
        baseline = args.role == "baseline"
        print(f"Preparing {name} ({args.matmul_precision})", flush=True)
        kwargs = {} if baseline else {"optimizations": config}
        policy = Pi05Policy(
            VLAPolicyConfig(name="pi0.5", action_dim=32, action_horizon=10, default_num_steps=10),
            model,
            native_embeddings=True,
            native_inference=True,
            prefix_cuda_graph=True,
            compile_backend="inductor" if baseline else "none",
            prefix_attention="triton" if baseline else "sdpa",
            denoise_attention="triton" if baseline else "sdpa",
            **kwargs,
        )
        policy.checkpoint = str(args.checkpoint)
        engine = EngineCore(
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

        def forward(index: int, bound_engine: EngineCore = engine) -> torch.Tensor:
            prefix = bound_engine.policy.encode_prefix(batches[index])
            return bound_engine.policy.decoder.integrate(noises[index], prefix, 10, 1, bound_engine._graphs)

        for index in indices:
            for _ in range(args.warmup):
                forward(index)
        torch.cuda.synchronize()
        outputs, repeat_checks = {}, []
        for index in indices:
            actual = forward(index).clone()
            repeated = forward(index)
            repeat_checks.append(
                dict(
                    index=index,
                    finite=bool(actual.isfinite().all()),
                    repeat_equal=torch.equal(actual, repeated),
                )
            )
            outputs[index] = actual.float().cpu()
        if not all(row["finite"] and row["repeat_equal"] for row in repeat_checks):
            raise RuntimeError("Non-finite or non-deterministic warmed action output")
        torch.save(outputs, args.out / f"{name}-actions.pt")
        before = policy._runtime.stats()
        rows = []
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        for round_index in range(args.rounds):
            for iteration in range(args.iterations):
                index = indices[(iteration + round_index) % len(indices)]
                torch.cuda.synchronize()
                start = time.perf_counter()
                events[0].record()
                prefix = policy.encode_prefix(batches[index])
                events[1].record()
                policy.decoder.integrate(noises[index], prefix, 10, 1, engine._graphs)
                events[2].record()
                events[2].synchronize()
                rows.append(
                    dict(
                        round=round_index,
                        index=index,
                        wall_ms=(time.perf_counter() - start) * 1000,
                        gpu_ms=events[0].elapsed_time(events[2]),
                        prefix_ms=events[0].elapsed_time(events[1]),
                        diffusion_ms=events[1].elapsed_time(events[2]),
                    )
                )
        after = policy._runtime.stats()
        if before != after:
            raise RuntimeError(f"Graph cache changed during timing: {before} -> {after}")
        report["modes"][name] = dict(
            optimization=asdict(config) if config is not None else None,
            compile_backend=policy.compile_backend,
            prefix_attention=policy.prefix_attention,
            denoise_attention=policy.denoise_attention,
            checks=repeat_checks,
            graph_stats_before=before,
            graph_stats_after=after,
            timing=rows,
            summary={
                key: dict(
                    mean=float(np.mean([r[key] for r in rows])),
                    p50=float(np.median([r[key] for r in rows])),
                    p95=float(np.percentile([r[key] for r in rows], 95)),
                )
                for key in ("wall_ms", "gpu_ms", "prefix_ms", "diffusion_ms")
            },
        )
        print(json.dumps(dict(mode=name, summary=report["modes"][name]["summary"])), flush=True)
        (args.out / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
        policy._clear_inference_caches()
        if engine._graphs is not None:
            engine._graphs._graphs.clear()
        del forward, engine, policy
        gc.collect()
        torch.cuda.empty_cache()
    report["imported_package_files"] = {
        name: str(Path(module.__file__).resolve())
        for name, module in tuple(sys.modules.items())
        if name.startswith("embodiinfer") and getattr(module, "__file__", None)
    }
    if any(
        not Path(path).is_relative_to(source / "embodiinfer")
        for path in report["imported_package_files"].values()
    ):
        raise RuntimeError("A package import escaped the isolated source tree")
    if args.role == "baseline" and any(
        name.startswith(("embodiinfer.policies.pi05.inference", "embodiinfer.policies.pi05.optimization"))
        for name in sys.modules
    ):
        raise RuntimeError("Baseline imported new optimization modules")
    (args.out / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
