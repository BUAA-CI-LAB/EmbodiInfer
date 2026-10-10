"""Compare opt-in operators against native Pi05 on identical recorded inputs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.layers import OperatorBackends, paired_gelu_backends, projection_backends
from embodiinfer.layers.attention import attention_backend_capability
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05 import ActionLayerPrecision, Pi05OptimizationConfig
from embodiinfer.policies.pi05.checkpoints.lerobot import load_lerobot_checkpoint
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch


def digest(path: Path) -> str:
    """Hash source, inputs and weights without depending on their machine paths."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_digest() -> str:
    """Identify the actual Python and native operator sources in this checkout."""
    root = Path(__file__).resolve().parents[2] / "embodiinfer"
    value = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h"):
            value.update(path.relative_to(root).as_posix().encode() + b"\0")
            value.update(path.read_bytes())
    return value.hexdigest()


def export_inputs(args: argparse.Namespace) -> None:
    """Export one recorded observation per LIBERO task with the full tokenizer."""
    import benchmark as bench
    from compare import Processor

    processor = Processor(args.checkpoint)
    samples = bench.load_libero(
        dict(root=str(args.dataset), task_limit=10, demos_per_task=1, frames_per_demo=1)
    )
    inputs = []
    for sample in samples:
        prepared = processor.prepare(sample, bench.read_libero(sample))
        inputs.append(
            dict(
                sample_id=sample.sample_id,
                images=[image.to(torch.bfloat16).float() for image in prepared["images"]],
                tokens=prepared["tokens"],
                masks=prepared["masks"],
                state=prepared["state"],
                prompt=prepared["prompt"],
            )
        )
    torch.save(
        dict(
            observations=inputs,
            action_mean=processor.action_stats["action.mean"],
            action_std=processor.action_stats["action.std"] + processor.output_eps,
            vocabulary_size=len(processor.tokenizer),
            calibration_indices=[0, 3, 6, 9],
            validation_indices=[1, 2, 4, 5, 7, 8],
        ),
        args.inputs,
    )
    print(json.dumps(dict(inputs=str(args.inputs), sha256=digest(args.inputs))), flush=True)


def native_batch(item: dict, batch_size: int = 1) -> Pi05Batch:
    """Use two recorded cameras and one explicitly masked empty camera."""
    images = [image.repeat(batch_size, 1, 1, 1).cuda() for image in item["images"]]
    return Pi05Batch(
        images + [-torch.ones_like(images[0])],
        [torch.ones(batch_size, dtype=torch.bool, device="cuda")] * 2
        + [torch.zeros(batch_size, dtype=torch.bool, device="cuda")],
        item["tokens"].repeat(batch_size, 1).cuda(),
        item["masks"].repeat(batch_size, 1).cuda(),
    )


def make_engine(
    reference: torch.nn.Module,
    horizon: int,
    batch_size: int,
    config: Pi05OptimizationConfig | None = None,
    *,
    native: bool = True,
    compiled: bool = False,
) -> EngineCore:
    """Share checkpoint Parameters while giving each candidate independent graphs."""
    policy = Pi05Policy(
        VLAPolicyConfig(name="pi0.5", action_dim=32, action_horizon=horizon, default_num_steps=10),
        reference,
        native_embeddings=True,
        native_inference=native,
        prefix_cuda_graph=native,
        optimizations=config,
        compile_backend="inductor" if compiled else "none",
        prefix_attention="triton" if compiled else "sdpa",
        denoise_attention="triton" if compiled else "sdpa",
    )
    policy.checkpoint = str(reference._validation_checkpoint)
    engine = EngineCore(
        policy,
        EngineConfig(
            device="cuda",
            dtype="auto",
            max_batch_size=batch_size,
            batch_buckets=(batch_size,),
            use_cuda_graph=True,
            capture_full_loop=True,
        ),
    )
    return engine


def run(
    engine: EngineCore, batch: Pi05Batch, noise: torch.Tensor, *, graphs: bool = True
) -> tuple[torch.Tensor, dict[str, float]]:
    """Measure GPU-ready inputs through prefix and all ten denoising steps."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    events[0].record()
    prefix = engine.policy.encode_prefix(batch)
    events[1].record()
    output = engine.policy.decoder.integrate(
        noise, prefix, 10, batch.batch_size, engine._graphs if graphs else None
    )
    events[2].record()
    torch.cuda.synchronize()
    timing = dict(
        wall_ms=(time.perf_counter() - start) * 1000,
        gpu_ms=events[0].elapsed_time(events[2]),
        prefix_ms=events[0].elapsed_time(events[1]),
        diffusion_ms=events[1].elapsed_time(events[2]),
    )
    return output.detach().float().cpu(), timing


def clear(engine: EngineCore) -> None:
    """Release graphs before derived operator storage when switching candidates."""
    engine.policy._clear_inference_caches()
    if engine._graphs is not None:
        engine._graphs._graphs.clear()
    gc.collect()
    torch.cuda.empty_cache()


def calibrate(
    reference: torch.nn.Module,
    batches: list[Pi05Batch],
    horizon: int,
    indices: list[int],
    *,
    prefix: bool = False,
) -> tuple[tuple[float, float], ...]:
    """Collect fixed activation maxima from disjoint BF16 calibration observations."""
    from embodiinfer.backend.torch.activation import TorchPairedGelu
    from embodiinfer.backend.torch.projection import CalibratedProjectionBackend

    towers = reference.model.paligemma_with_expert
    layers = (towers.paligemma.model.language_model if prefix else towers.gemma_expert.model).layers
    maxima = [[0.0, 0.0] for _ in layers]
    gates = {layer.mlp.gate_proj.weight.data_ptr(): i for i, layer in enumerate(layers)}
    downs = {layer.mlp.down_proj.weight.data_ptr(): i for i, layer in enumerate(layers)}

    class RecordPaired(TorchPairedGelu):
        def plan(self, gate_weight, up_weight):
            plan = super().plan(gate_weight, up_weight)
            index = gates.get(gate_weight.data_ptr())

            def execute(inputs):
                if index is not None:
                    maxima[index][0] = max(maxima[index][0], inputs.float().abs().amax().item())
                return plan(inputs)

            return execute

    class RecordProjection(CalibratedProjectionBackend):
        def plan(self, weight, maximum, precision, quantizer, workspace_key):
            plan = super().plan(weight, maximum, precision, quantizer, workspace_key)
            index = downs.get(weight.data_ptr())
            if index is not None:
                original = plan.apply

                def apply(inputs, encoded):
                    maxima[index][1] = max(maxima[index][1], inputs.float().abs().amax().item())
                    return original(inputs, encoded)

                plan.apply = apply
            return plan

    name = "calibration_prefix" if prefix else "calibration_action"
    paired_gelu_backends.register(name, RecordPaired)
    projection_backends.register(name, RecordProjection)
    config = Pi05OptimizationConfig(
        fused_mlp=True,
        operators=OperatorBackends(paired_gelu=name, projection=name),
    )
    engine = make_engine(reference, horizon, 1, config)
    engine.policy.prefix_cuda_graph = False
    for index in indices:
        noise = torch.from_numpy(
            np.random.default_rng(42 + index).standard_normal((1, horizon, 32), dtype=np.float32)
        ).cuda()
        run(engine, batches[index], noise, graphs=False)
    clear(engine)
    del engine
    if any(gate <= 0 or down <= 0 for gate, down in maxima):
        raise RuntimeError(f"Calibration did not observe every {name} projection")
    # A fixed 10% headroom is chosen before inspecting held-out actions.
    return tuple((gate * 1.1, down * 1.1) for gate, down in maxima)


def main() -> None:
    """Export fixtures or report measured latency and frozen action-error gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--source-revision", default=os.environ.get("EMBODIINFER_SOURCE_REVISION"))
    parser.add_argument("--fp32-reference", type=Path, help="Reuse a verified run directory's FP32 oracle")
    parser.add_argument(
        "--precision-map",
        type=Path,
        action="append",
        default=[],
        help="Frozen per-layer formats; recalibrate ranges on calibration observations only",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "native",
            "kv",
            "strict",
            "query_major",
            "folded_flash",
            "paired",
            "paired_flash",
            "fp8",
            "nvfp4",
        ],
    )
    args = parser.parse_args()
    if args.export:
        if args.dataset is None:
            parser.error("--export requires --dataset")
        export_inputs(args)
        return
    if args.out is None or min(args.horizon, args.batch_size, args.warmup, args.iterations) < 1:
        parser.error("positive dimensions/counts and --out are required")
    precision_maps = {}
    for path in args.precision_map:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", path.stem) or path.stem in (
            *args.modes,
            *precision_maps,
        ):
            parser.error("Precision map filenames must be unique simple mode names")
        selection = json.loads(path.read_text())
        for name in ("prefix_layers", "action_layers"):
            for row in selection.get(name, []):
                if set(row) != {"gate_up", "down"}:
                    parser.error("Precision maps select formats only; do not reuse old activation scales")
        precision_maps[path.stem] = selection
    args.modes.extend(precision_maps)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(42)
    fixture = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if fixture["vocabulary_size"] != 257152:
        raise ValueError("Fixtures must use the verified complete PaliGemma vocabulary")
    batches = [native_batch(item, args.batch_size) for item in fixture["observations"]]
    indices = fixture["validation_indices"]
    noises = {
        index: torch.from_numpy(
            np.random.default_rng(1000 + index).standard_normal(
                (args.batch_size, args.horizon, 32), dtype=np.float32
            )
        ).cuda()
        for index in indices
    }
    sha = digest(args.checkpoint / "model.safetensors")
    metadata = dict(
        gpu=str(torch.cuda.get_device_properties(0)),
        versions={
            name: importlib.metadata.version(name) for name in ("torch", "triton", "transformers", "lerobot")
        },
        cuda=torch.version.cuda,
        python=sys.version,
        platform=platform.platform(),
        allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        cpu_threads=torch.get_num_threads(),
        checkpoint_sha256=sha,
        checkpoint_metadata_sha256={
            path.name: digest(path) for path in sorted(args.checkpoint.glob("*.json"))
        },
        inputs_sha256=digest(args.inputs),
        horizon=args.horizon,
        checkpoint_horizon=json.loads((args.checkpoint / "config.json").read_text())["chunk_size"],
        batch_size=args.batch_size,
        steps=10,
        warmup=args.warmup,
        iterations=args.iterations,
        modes=args.modes,
        precision_maps={
            path.stem: dict(sha256=digest(path), selection=precision_maps[path.stem])
            for path in args.precision_map
        },
        seed=42,
        validation_noise="NumPy default_rng(1000 + observation_index), FP32, identical across devices",
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        timing_scope="GPU-ready recorded inputs through prefix and all 10 steps; no preprocessing or action postprocessing",
        calibration_indices=fixture["calibration_indices"],
        validation_indices=indices,
        source_revision=args.source_revision,
        source_sha256=source_digest(),
        forward_contract="EmbodiInfer/LeRobot gelu_pytorch_tanh and original RoPE; numerical validation against this adapter, not task quality or RLinf/JAX alignment",
        benchmark_sha256=digest(Path(__file__)),
    )
    (args.out / "conditions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    reference = load_lerobot_checkpoint(str(args.checkpoint), load_device="cuda", low_cpu_mem_usage=True)
    # LeRobot's optional sample_actions compiler can mutate this global setting
    # during construction; the adapter benchmark must use the recorded value.
    torch.set_float32_matmul_precision(metadata["matmul_precision"])
    reference._validation_checkpoint = args.checkpoint
    original_dtypes = {name: p.dtype for name, p in reference.named_parameters()}
    with torch.inference_mode():
        baseline = make_engine(reference, args.horizon, args.batch_size)
        expected = {index: run(baseline, batches[index], noises[index])[0] for index in indices}
        clear(baseline)
        del baseline
        # Freeze error gates before running any optimized candidate. The FP32
        # reference retains exactly the BF16-rounded checkpoint values and inputs.
        if args.fp32_reference is None:
            reference.float()
            full_precision = make_engine(reference, args.horizon, args.batch_size, native=False)
            fp32 = {
                index: run(full_precision, batches[index], noises[index], graphs=False)[0]
                for index in indices
            }
            clear(full_precision)
            del full_precision
            for name, parameter in reference.named_parameters():
                parameter.data = parameter.data.to(original_dtypes[name])
        else:
            # A CPU-resident oracle avoids a second FP32 model on small devices.
            # Reject mismatched experiments before freezing candidate gates.
            conditions = args.fp32_reference / "conditions.json"
            oracle_metadata = json.loads(conditions.read_text())
            for key in (
                "checkpoint_sha256",
                "inputs_sha256",
                "horizon",
                "batch_size",
                "steps",
                "matmul_precision",
                "validation_indices",
                "validation_noise",
                "forward_contract",
            ):
                if oracle_metadata.get(key) != metadata[key]:
                    raise ValueError(f"External FP32 reference mismatch: {key}")
            oracle = args.fp32_reference / "reference.pt"
            fp32 = torch.load(oracle, map_location="cpu", weights_only=True)["fp32"]
            for index in indices:
                if fp32[index].shape != expected[index].shape or not fp32[index].isfinite().all():
                    raise ValueError(f"Invalid external FP32 actions: observation {index}")
            metadata["fp32_reference_origin"] = dict(
                conditions_sha256=digest(conditions),
                actions_sha256=digest(oracle),
                gpu=oracle_metadata["gpu"],
                versions=oracle_metadata["versions"],
            )
            (args.out / "conditions.json").write_text(json.dumps(metadata, indent=2) + "\n")
        torch.cuda.empty_cache()
        thresholds = {}
        for index in indices:
            error = (expected[index][..., :7] - fp32[index][..., :7]).float()
            thresholds[index] = dict(
                max_abs=max(1e-4, error.abs().max().item() * 2),
                rmse=max(1e-5, error.square().mean().sqrt().item() * 2),
            )
        (args.out / "frozen-tolerances.json").write_text(json.dumps(thresholds, indent=2) + "\n")
        torch.save(dict(baseline=expected, fp32=fp32), args.out / "reference.pt")
        maxima = None
        if (
            (
                any(mode in args.modes for mode in ("fp8", "nvfp4"))
                or any(x.get("action_layers") for x in precision_maps.values())
            )
            and args.batch_size == 1
            and hasattr(torch.nn.functional, "scaled_mm")
        ):
            maxima = calibrate(reference, batches, args.horizon, fixture["calibration_indices"])
            (args.out / "calibration.json").write_text(json.dumps(maxima, indent=2) + "\n")
        prefix_maxima = None
        if any(x.get("prefix_layers") for x in precision_maps.values()):
            prefix_maxima = calibrate(
                reference, batches, args.horizon, fixture["calibration_indices"], prefix=True
            )
            (args.out / "prefix-calibration.json").write_text(json.dumps(prefix_maxima, indent=2) + "\n")
        capability = torch.cuda.get_device_capability()
        hardware = {(11, 0): "thor", (12, 1): "spark"}.get(capability)
        for mode in args.modes:
            config = None
            reason = None
            if mode in precision_maps:
                values = dict(precision_maps[mode])
                for name, ranges in (("action_layers", maxima), ("prefix_layers", prefix_maxima)):
                    if name in values:
                        if ranges is None or len(values[name]) != len(ranges):
                            raise ValueError(
                                f"Precision map {mode} requires one calibrated {name} entry per layer"
                            )
                        values[name] = tuple(
                            ActionLayerPrecision(**row, gate_up_max=gate, down_max=down)
                            for row, (gate, down) in zip(values[name], ranges, strict=True)
                        )
                if "operators" in values:
                    values["operators"] = OperatorBackends(**values["operators"])
                config = Pi05OptimizationConfig(**values, checkpoint_sha256=sha)
                config.to_json(args.out / f"{mode}-recipe.json")
            elif mode == "kv":
                config = Pi05OptimizationConfig(norm_fusion=False)
            elif mode in ("strict", "query_major", "folded_flash"):
                config = Pi05OptimizationConfig(attention=mode if mode != "strict" else "reference")
                if mode != "strict":
                    available, reason = attention_backend_capability(mode)
                    if available:
                        reason = None
            elif mode in ("paired", "paired_flash"):
                if hardware is None or args.horizon != 10 or args.batch_size != 1:
                    reason = "Paired launch profiles cover Thor/Spark B1/horizon10 only"
                else:
                    config = Pi05OptimizationConfig(
                        hardware=hardware,
                        fused_mlp=True,
                        attention="folded_flash" if mode == "paired_flash" else "reference",
                    )
            elif mode in ("fp8", "nvfp4"):
                if capability < ((10, 0) if mode == "nvfp4" else (8, 9)) or maxima is None:
                    reason = "Native low-precision GEMM or calibration unavailable"
                else:
                    config = Pi05OptimizationConfig(
                        action_layers=tuple(
                            ActionLayerPrecision(mode, mode, gate, down) for gate, down in maxima
                        ),
                        checkpoint_sha256=sha,
                    )
                    config.to_json(args.out / f"{mode}-recipe.json")
            elif mode not in ("native", "existing_inductor"):
                raise ValueError(f"Unknown mode {mode}")
            if reason is not None:
                result = dict(mode=mode, skipped=reason)
                (args.out / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result), flush=True)
                continue
            engine = None
            try:
                engine = make_engine(
                    reference, args.horizon, args.batch_size, config, compiled=mode == "existing_inductor"
                )
                for index in indices:
                    for _ in range(args.warmup):
                        run(engine, batches[index], noises[index])
                checks, outputs = [], {}
                for index in indices:
                    actual, _ = run(engine, batches[index], noises[index])
                    repeated, _ = run(engine, batches[index], noises[index])
                    error = actual[..., :7] - expected[index][..., :7]
                    max_abs, rmse = error.abs().max().item(), error.square().mean().sqrt().item()
                    checks.append(
                        dict(
                            index=index,
                            sample_id=fixture["observations"][index]["sample_id"],
                            finite=bool(actual.isfinite().all()),
                            repeat_byteexact=torch.equal(actual, repeated),
                            max_abs=max_abs,
                            rmse=rmse,
                            passed=bool(actual.isfinite().all())
                            and max_abs <= thresholds[index]["max_abs"]
                            and rmse <= thresholds[index]["rmse"],
                        )
                    )
                    outputs[index] = actual
                torch.save(outputs, args.out / f"{mode}-actions.pt")
                graphs_before = engine.policy._runtime.stats()
                torch.cuda.reset_peak_memory_stats()
                rows = [
                    run(engine, batches[indices[i % len(indices)]], noises[indices[i % len(indices)]])[1]
                    for i in range(args.iterations)
                ]
                graphs_after = engine.policy._runtime.stats()
                if graphs_before != graphs_after:
                    raise RuntimeError(
                        f"Graph caches changed during timing: {graphs_before} -> {graphs_after}"
                    )
                result = dict(
                    mode=mode,
                    optimization=asdict(config) if config else None,
                    checks=checks,
                    passed=all(row["passed"] and row["repeat_byteexact"] for row in checks),
                    measured_despite_mismatch=not all(row["passed"] for row in checks),
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    graph_stats_before=graphs_before,
                    graph_stats_after=graphs_after,
                    timing=rows,
                    summary={
                        key: dict(
                            mean=float(np.mean([row[key] for row in rows])),
                            p50=float(np.percentile([row[key] for row in rows], 50)),
                            p95=float(np.percentile([row[key] for row in rows], 95)),
                        )
                        for key in rows[0]
                    },
                )
                (args.out / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps(
                        dict(
                            mode=mode,
                            passed=result["passed"],
                            max_abs=max(row["max_abs"] for row in checks),
                            summary=result["summary"],
                        )
                    ),
                    flush=True,
                )
            except Exception as error:
                result = dict(mode=mode, passed=False, error=str(error), traceback=traceback.format_exc())
                (args.out / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(dict(mode=mode, error=str(error))), flush=True)
            finally:
                if engine is not None:
                    clear(engine)
                del engine


if __name__ == "__main__":
    main()
