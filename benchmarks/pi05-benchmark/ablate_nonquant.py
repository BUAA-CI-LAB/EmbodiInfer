"""Progressive and isolated Pi05 ablations with a frozen byte-equality gate.

This program reuses the graph trace and numerical contracts in validate_nonquant.
Complete-request timings are paired and alternate order. Incremental comparisons
use adjacent stages; cumulative comparisons directly pair each stage with the
current native baseline. They are separate experiments, not multiplied estimates.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from validate_nonquant import (
    LOSSLESS,
    CaptureTrace,
    benchmark_paired_requests,
    clear_engine_graphs,
    compare,
    compare_traces,
    configure_threads,
    digest,
    gripper_flips,
    physical_actions,
    resolve_sampling,
    save_trace,
    source_provenance,
    tensor_snapshot,
)


@dataclass(frozen=True)
class Stage:
    """One explicit, non-quantized operator selection and request capture mode."""

    name: str
    switches: tuple[str, ...] = ()
    request_graph: bool = False
    predecessor: str | None = None
    group: str = "progressive"
    operator_control: bool = False


STAGES = (
    Stage("baseline"),
    Stage("norm", ("norm_fusion",), predecessor="baseline"),
    Stage("norm_kv", ("norm_fusion", "kv_workspace"), predecessor="norm"),
    Stage("norm_kv_prefix", ("norm_fusion", "kv_workspace", "prefix_kv_only"), predecessor="norm_kv"),
    Stage(
        "norm_kv_prefix_context",
        ("norm_fusion", "kv_workspace", "prefix_kv_only", "reuse_action_context"),
        predecessor="norm_kv_prefix",
    ),
    Stage(
        "cumulative_request",
        ("norm_fusion", "kv_workspace", "prefix_kv_only", "reuse_action_context"),
        True,
        "norm_kv_prefix_context",
    ),
    Stage("operator_control", group="isolated", operator_control=True),
    Stage("kv_only", ("kv_workspace",), group="isolated"),
    Stage("prefix_only", ("prefix_kv_only",), group="isolated"),
    Stage("context_only", ("reuse_action_context",), group="isolated"),
    Stage("request_only", request_graph=True, group="isolated"),
    Stage("kv", ("kv_workspace",), predecessor="baseline"),
    Stage("kv_prefix", ("kv_workspace", "prefix_kv_only"), predecessor="kv"),
    Stage(
        "kv_prefix_context",
        ("kv_workspace", "prefix_kv_only", "reuse_action_context"),
        predecessor="kv_prefix",
    ),
    Stage(
        "kv_prefix_context_request",
        ("kv_workspace", "prefix_kv_only", "reuse_action_context"),
        True,
        "kv_prefix_context",
    ),
    Stage("gelu_only", ("prefix_gelu_fusion",), group="isolated"),
    Stage(
        "kv_prefix_context_gelu",
        ("kv_workspace", "prefix_kv_only", "reuse_action_context", "prefix_gelu_fusion"),
        predecessor="kv_prefix_context",
    ),
)


def stage_fields(stage: Stage) -> dict[str, bool] | None:
    """Expand only the tested non-quantized switches; everything else stays default."""
    if not stage.switches and not stage.operator_control:
        return None
    fields = dict(
        norm_fusion=False,
        kv_workspace=False,
        prefix_kv_only=False,
        reuse_action_context=False,
        prefix_gelu_fusion=False,
    )
    fields.update(dict.fromkeys(stage.switches, True))
    return fields


def compare_stage_trace(
    reference: dict, candidate: dict, indices: list[int], threshold: float, comparison: str
) -> dict:
    """Gate all trace quantities and each explicitly selected gripper decision."""
    result = compare_traces(reference, candidate)
    result["grippers"] = [
        gripper_flips(
            reference["actions/physical"],
            candidate["actions/physical"],
            index=index,
            threshold=threshold,
            comparison=comparison,
        )
        for index in indices
    ]
    result["passed"] &= all(row["passed"] for row in result["grippers"])
    return result


@contextmanager
def cuda_profile_range(torch: Any) -> Iterator[None]:
    """Bracket warmed requests for an external CUDA-profiler API capture.

    This emits capture-range triggers only. The caller must inspect the external
    profiler's kernel records before claiming that a CUDA trace was collected.
    """
    torch.cuda.profiler.start()
    try:
        yield None
    finally:
        torch.cuda.profiler.stop()


def parser() -> argparse.ArgumentParser:
    """Declare fixed source inputs, robot restoration and fair timing counts."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--inputs", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--seeds", nargs="+", type=int, default=[1000, 2026])
    result.add_argument("--stages", nargs="+", choices=[stage.name for stage in STAGES])
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--iterations", type=int, default=60)
    result.add_argument("--repeats", type=int, default=2)
    profiling = result.add_mutually_exclusive_group()
    profiling.add_argument(
        "--profile",
        action="store_true",
        help="Export separate warmed complete-request CUDA traces; profiling is excluded from latency samples",
    )
    profiling.add_argument(
        "--profile-cuda-range",
        action="store_true",
        help="Emit separate cudaProfilerStart/Stop windows for Nsight; inspect external kernel records",
    )
    result.add_argument("--physical-action-dim", type=int, required=True)
    result.add_argument("--gripper-indices", nargs="+", type=int, required=True)
    result.add_argument("--gripper-threshold", type=float, required=True)
    result.add_argument("--gripper-comparison", choices=("gt", "ge", "lt", "le"), required=True)
    return result


def main() -> None:
    """Validate every layer/step first, then report paired complete-request samples."""
    args = parser().parse_args()
    if args.out.exists() or min(args.warmup, args.iterations, args.repeats, args.physical_action_dim) < 1:
        raise ValueError("select a new output directory and positive counts")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("fixed noise seeds must be nonempty and unique")
    if not np.isfinite(args.gripper_threshold) or any(
        not 0 <= index < args.physical_action_dim for index in args.gripper_indices
    ):
        raise ValueError("declare valid gripper indices and a finite threshold")
    stages = [
        stage
        for stage in STAGES
        if args.stages is None or stage.name in args.stages or stage.name == "baseline"
    ]
    stage_by_name = {stage.name: stage for stage in stages}

    import torch

    threads = configure_threads(torch, 1, 1)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA and the matching checkpoint environment are required")
    source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source))
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.config import VLAPolicyConfig
    from embodiinfer.policies.pi05 import Pi05OptimizationConfig
    from embodiinfer.policies.pi05.checkpoints.lerobot import load_lerobot_checkpoint
    from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
    from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

    torch.set_float32_matmul_precision("highest")
    fixture = torch.load(args.inputs, map_location="cpu", weights_only=True)
    mean, std = (torch.as_tensor(fixture[key]).float() for key in ("action_mean", "action_std"))
    if (
        mean.shape != std.shape
        or args.physical_action_dim > mean.numel()
        or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all())
    ):
        raise ValueError(
            "physical action statistics must be finite, positive and match the checkpoint dimensions"
        )
    cases = []
    for index in fixture["validation_indices"]:
        item = fixture["observations"][index]
        if "img_masks" not in item:
            images = item["images"] + [-torch.ones_like(item["images"][0])]
            masks = [torch.ones(1, dtype=torch.bool)] * len(item["images"]) + [
                torch.zeros(1, dtype=torch.bool)
            ]
        else:
            images, masks = item["images"], item["img_masks"]
        batch = Pi05Batch(images, masks, item["tokens"], item["masks"], [f"ablation-{index}"])
        for seed in args.seeds:
            cases.append((index, seed, batch))
    if args.iterations < len(cases):
        raise ValueError("each timing repeat must cover every observation/noise case")
    args.out.mkdir(parents=True)
    report: dict[str, Any] = dict(
        status="running",
        passed=False,
        thresholds=asdict(LOSSLESS),
        source=source_provenance(source, args.source_manifest),
        ablation_program_sha256=digest(Path(__file__)),
        fixture_sha256=digest(args.inputs),
        fixture_description=fixture.get("description"),
        restoration=fixture.get("restoration"),
        gripper_semantics=fixture.get("gripper_semantics"),
        gripper_decisions=dict(
            indices=args.gripper_indices, threshold=args.gripper_threshold, comparison=args.gripper_comparison
        ),
        checkpoint_files={
            path.name: digest(path)
            for path in sorted(args.checkpoint.iterdir())
            if path.is_file() and path.suffix in (".json", ".safetensors")
        },
        hardware=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        python=platform.python_version(),
        threads=threads,
        versions={
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "lerobot", "numpy")
        },
        measurement_scope="prepared CPU observations -> H2D -> native prefix -> all denoising steps -> D2H/EngineCore packing -> checkpoint affine action restoration; excludes camera decoding/tokenization, network, loading, cold compilation/capture and warmup",
        stages={
            stage.name: dict(selection=asdict(stage), operator_fields=stage_fields(stage), comparisons=[])
            for stage in stages
        },
        cases=[dict(index=index, seed=seed) for index, seed, _ in cases],
        pairs=[],
        limits="fixed recorded frames, one host/GPU and batch one; no closed-loop task-success claim; all switches are opt-in; gripper classification scope is disclosed independently of continuous robot execution",
    )

    def write() -> None:
        (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write()
    try:
        with torch.inference_mode():
            model = load_lerobot_checkpoint(str(args.checkpoint), load_device="cuda", low_cpu_mem_usage=True)
            sampling = resolve_sampling(model.config, None, None)
            report["sampling"] = sampling
            steps = sampling["resolved"]["steps"]
            config = VLAPolicyConfig(
                name="pi0.5",
                action_dim=sampling["resolved"]["action_dim"],
                action_horizon=sampling["resolved"]["horizon"],
                default_num_steps=steps,
            )
            report["parameter_dtypes"] = dict(Counter(str(value.dtype) for value in model.parameters()))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            def make_engine(stage: Stage) -> Any:
                fields = stage_fields(stage)
                policy = Pi05Policy(
                    config,
                    model,
                    native_embeddings=True,
                    native_inference=True,
                    prefix_cuda_graph=True,
                    request_cuda_graph=stage.request_graph,
                    optimizations=None if fields is None else Pi05OptimizationConfig(**fields),
                    compile_backend="none",
                    prefix_attention="sdpa",
                    denoise_attention="sdpa",
                )
                return EngineCore(
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

            def request(engine: Any, case: int, generator: Any) -> Any:
                chunks = engine.execute(cases[case][2], num_steps=steps, generator=generator)
                return torch.stack([chunk.actions for chunk in chunks])

            def restored(action: Any) -> Any:
                return physical_actions(action, mean, std, args.physical_action_dim)

            references = []
            with torch.cuda.stream(stream):
                for stage in stages:
                    print(f"DIAGNOSE {stage.name}", flush=True)
                    engine = make_engine(stage)
                    generator = torch.Generator(device="cuda")
                    rows = report["stages"][stage.name]["comparisons"]
                    for case in range(len(cases)):
                        clear_engine_graphs(engine)
                        with CaptureTrace(
                            engine.policy, steps, torch.cuda.is_current_stream_capturing
                        ) as recorder:
                            generator.manual_seed(cases[case][1])
                            action = request(engine, case, generator)
                            trace = recorder.snapshot(action)
                            trace["actions/physical"] = tensor_snapshot(restored(action))
                            save_trace(args.out / f"{stage.name}-{case:03d}-trace.npz", trace)
                            generator.manual_seed(cases[case][1])
                            repeat = request(engine, case, generator)
                            repeated_trace = recorder.snapshot(repeat)
                            repeated_trace["actions/physical"] = tensor_snapshot(restored(repeat))
                            rows.append(
                                dict(kind="self_repeat", case=case, **compare_traces(trace, repeated_trace))
                            )
                        if stage.name == "baseline":
                            references.append(trace)
                        else:
                            result = compare_stage_trace(
                                references[case],
                                trace,
                                args.gripper_indices,
                                args.gripper_threshold,
                                args.gripper_comparison,
                            )
                            rows.append(dict(kind="against_baseline", case=case, **result))
                        clear_engine_graphs(engine)
                        generator.manual_seed(cases[case][1])
                        clean = request(engine, case, generator)
                        clean_trace = {
                            "actions/normalized": tensor_snapshot(clean),
                            "actions/physical": tensor_snapshot(restored(clean)),
                        }
                        save_trace(args.out / f"{stage.name}-{case:03d}-clean.npz", clean_trace)
                        for key, value in clean_trace.items():
                            rows.append(
                                dict(
                                    kind="uninstrumented_parity",
                                    case=case,
                                    quantity=key,
                                    **compare(trace[key], value),
                                )
                            )
                    report["stages"][stage.name]["passed"] = all(row["passed"] for row in rows)
                    clear_engine_graphs(engine)
                    del engine, generator
                    gc.collect()
                    torch.cuda.empty_cache()
                    write()
                if args.profile or args.profile_cuda_range:
                    report["profiles"] = {}
                    for stage in stages:
                        if not report["stages"][stage.name]["passed"]:
                            continue
                        engine = make_engine(stage)
                        generator = torch.Generator(device="cuda")
                        for case in range(len(cases)):
                            for _ in range(args.warmup):
                                generator.manual_seed(cases[case][1])
                                restored(request(engine, case, generator))
                        torch.cuda.synchronize()
                        before = engine.policy._runtime.stats()
                        trace_path = args.out / f"{stage.name}-profile.json"
                        print(f"PROFILE enter {stage.name}", flush=True)
                        profile_context = (
                            cuda_profile_range(torch)
                            if args.profile_cuda_range
                            else torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ],
                                record_shapes=True,
                            )
                        )
                        with profile_context as profiler:
                            for case in range(len(cases)):
                                generator.manual_seed(cases[case][1])
                                label = f"complete_request/{stage.name}/{case}"
                                record = (
                                    torch.cuda.nvtx.range(label)
                                    if args.profile_cuda_range
                                    else torch.profiler.record_function(label)
                                )
                                with record:
                                    result = restored(request(engine, case, generator))
                                torch.cuda.synchronize()
                                check = compare(references[case]["actions/physical"], tensor_snapshot(result))
                                if not check["passed"]:
                                    raise RuntimeError("profiled request changed physical actions")
                            print(f"PROFILE collected {stage.name}", flush=True)
                        print(f"PROFILE export {stage.name}", flush=True)
                        events = []
                        if profiler is not None:
                            profiler.export_chrome_trace(str(trace_path))
                            events = json.loads(trace_path.read_text()).get("traceEvents", [])
                        kernels = {}
                        for event in events:
                            if event.get("cat") != "kernel" or "dur" not in event:
                                continue
                            name = event["name"]
                            row = kernels.setdefault(name, dict(calls=0, total_cuda_us=0.0))
                            row["calls"] += 1
                            row["total_cuda_us"] += float(event["dur"])
                        after = engine.policy._runtime.stats()
                        if before != after:
                            raise RuntimeError("profiling constructed new captures after warmup")
                        report["profiles"][stage.name] = dict(
                            scope="warmed complete EngineCore request on the same fixed cases; observer overhead is excluded from all latency samples",
                            backend="cudaProfilerApi" if args.profile_cuda_range else "torch.profiler",
                            cuda_kernels_available=None if args.profile_cuda_range else bool(kernels),
                            kernel_totals=kernels,
                            total_cuda_kernel_us=(
                                None
                                if args.profile_cuda_range
                                else sum(row["total_cuda_us"] for row in kernels.values())
                            ),
                            cache_stats=after,
                            trace_sha256=None if args.profile_cuda_range else digest(trace_path),
                            external_kernel_records_required=args.profile_cuda_range,
                        )
                        clear_engine_graphs(engine)
                        del engine, generator, profiler
                        gc.collect()
                        torch.cuda.empty_cache()
                        write()

                experiments = []
                for stage in stages[1:]:
                    experiments.append(
                        ("cumulative" if stage.group == "progressive" else "isolated", "baseline", stage.name)
                    )
                    if stage.predecessor not in (None, "baseline") and stage.predecessor in stage_by_name:
                        experiments.append(("incremental", stage.predecessor, stage.name))
                for kind, first, second in experiments:
                    if not all(report["stages"][name]["passed"] for name in (first, second)):
                        report["pairs"].append(dict(kind=kind, first=first, second=second, eligible=False))
                        continue
                    print(f"TIMING {kind} {first} vs {second}", flush=True)
                    engines = {name: make_engine(stage_by_name[name]) for name in (first, second)}
                    generators = {name: torch.Generator(device="cuda") for name in engines}

                    def reset(name: str, case: int, bound: Any = generators) -> None:
                        bound[name].manual_seed(cases[case][1])

                    def run(
                        name: str, case: int, bound_engines: Any = engines, bound_generators: Any = generators
                    ) -> Any:
                        return restored(request(bound_engines[name], case, bound_generators[name]))

                    calls = {
                        name: (lambda case, selected=name, call=run: call(selected, case)) for name in engines
                    }
                    trials = []
                    for repetition in range(args.repeats):
                        order = (first, second) if repetition % 2 == 0 else (second, first)
                        trial = benchmark_paired_requests(
                            {name: calls[name] for name in order},
                            cases=len(cases),
                            synchronize=torch.cuda.synchronize,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            before_request=reset,
                        )
                        trial["repetition"] = repetition
                        trials.append(trial)
                    report["pairs"].append(
                        dict(
                            kind=kind,
                            first=first,
                            second=second,
                            eligible=True,
                            trials=trials,
                            cache_stats={
                                name: engine.policy._runtime.stats() for name, engine in engines.items()
                            },
                        )
                    )
                    for engine in engines.values():
                        clear_engine_graphs(engine)
                    del engines, generators, calls, reset, run, engine
                    gc.collect()
                    torch.cuda.empty_cache()
                    write()
            report["passed"] = all(row["passed"] for row in report["stages"].values())
            report["status"] = "passed" if report["passed"] else "failed"
            write()
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        write()
        raise
    print(json.dumps(dict(passed=report["passed"], report=str(args.out / "report.json"))), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
