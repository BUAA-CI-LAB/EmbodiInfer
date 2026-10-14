"""Validate and time Pi05's opt-in whole-request graph without changing precision.

Use a trusted fixture from validate_optimizations.py --export. The fixed
lossless gate covers baseline repeatability, every prefix K/V, every denoising
velocity/state, normalized and affine-restored actions, and gripper decisions.
Instrumented graph captures are diagnostic only. Timing uses fresh, uninstrumented
captures and includes EngineCore.execute, input transfer, packing and restoration.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numerical_validation import (  # noqa: E402
    LOSSLESS,
    Snapshot,
    benchmark_paired_requests,
    benchmark_requests,
    compare,
    compare_traces,
    gripper_flips,
    synchronized_call,
)


def tensor_snapshot(value: Any) -> Snapshot:
    """Own source bytes before converting BF16 to FP32 for NumPy analysis."""
    import torch

    cpu = value.detach().contiguous().cpu()
    storage = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    values = cpu.float().numpy() if cpu.dtype == torch.bfloat16 else cpu.numpy()
    values = np.array(values, copy=True)
    values.setflags(write=False)
    return Snapshot(values, storage, str(cpu.dtype))


def fp32_initial_state(reference: Snapshot, device: Any) -> Any:
    """Promote captured baseline noise values instead of regenerating a dtype-dependent draw."""
    import torch

    values = np.array(reference.values, dtype=np.float32, copy=True, order="C")
    return torch.from_numpy(values).to(device=device)


@contextmanager
def ephemeral_packed_mlp_weights() -> Any:
    """Bound untimed FP32 memory without changing packed GEMM shapes or values.

    The production helper duplicates gate/up Parameters into a reusable packed
    matrix. A full FP32 checkpoint plus all such copies may exceed 24 GB. This
    diagnostic retains the exact helper computation but releases each packed
    copy after its GEMM is enqueued. Torch stream ordering protects its storage.
    This scope is never used for measured requests or CUDA graph capture.
    """
    import torch

    from embodiinfer.policies.pi05 import modeling_pi05

    original = modeling_pi05._fused_linear_pair

    def bounded(inputs: Any, first: Any, second: Any) -> Any:
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ephemeral packed weights are only for untimed eager inference")
        result = original(inputs, first, second)
        first._vvla_fused_pair_weight = None
        return result

    with patch.object(modeling_pi05, "_fused_linear_pair", bounded):
        yield


def digest(path: Path) -> str:
    """Hash an artifact without loading a checkpoint-sized file into RAM."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_provenance(root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """Record source bytes with Git facts, checking an optional host-side manifest.

    Minimal GPU containers may omit Git. Their host can record Git facts before
    launch; every listed runtime/tool byte must still match inside the container.
    A manifest cannot substitute different source bytes; host dirty status is retained.
    """
    paths = sorted(
        path
        for path in (root / "embodiinfer").rglob("*")
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h")
    )
    paths += [Path(__file__).resolve(), root / "benchmarks/numerical_validation.py"]
    files = {str(path.relative_to(root)): digest(path) for path in paths}

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("files") != files:
            raise RuntimeError("host source manifest does not match the measured runtime/tool bytes")
        revision = manifest.get("revision", "")
        if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision):
            raise ValueError("host source manifest requires the full Git commit SHA")
        return dict(
            revision=revision,
            dirty_status=manifest["dirty_status"],
            files=files,
            source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            git_facts="host manifest checked against in-container source bytes",
        )

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return dict(
        revision=git("rev-parse", "HEAD"),
        dirty_status=git("status", "--short"),
        source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        files=files,
    )


class CaptureTrace(AbstractContextManager):
    """Add owned diagnostic clone nodes to actual prefix/loop/request CUDA graphs.

    Hooks never implement or replace inference math. Each post-integration state
    is the next observed denoiser input; the last is the real integrator output.
    Warmup calls are ignored, and CPU snapshots occur only after synchronization.
    """

    def __init__(self, policy: Any, steps: int, capturing: Callable[[], bool]) -> None:
        self.policy = policy
        self.steps = steps
        self.capturing = capturing
        self.buffers: dict[str, Any] = {}
        self.step = 0
        self.patches: list[Any] = []

    def __enter__(self) -> CaptureTrace:
        encode = self.policy._encode_prefix_impl
        denoise = self.policy._denoise_step_impl
        integrate = self.policy._runtime.integrate

        def traced_encode(*args: Any, **kwargs: Any) -> Any:
            prefix = encode(*args, **kwargs)
            if self.capturing():
                self.buffers["prefix/mask"] = prefix.prefix_pad_masks.clone()
                for layer, (key, value) in enumerate(prefix.kv):
                    self.buffers[f"prefix/{layer:03d}/key"] = key.clone()
                    self.buffers[f"prefix/{layer:03d}/value"] = value.clone()
            return prefix

        def traced_denoise(state: Any, time: Any, *args: Any, **kwargs: Any) -> Any:
            active = self.capturing()
            if active:
                if self.step >= self.steps:
                    raise RuntimeError("unexpected extra denoising calls in diagnostic capture")
                self.buffers[f"state/{self.step:03d}"] = state.clone()
                self.buffers[f"time/{self.step:03d}"] = time.clone()
            velocity = denoise(state, time, *args, **kwargs)
            if active:
                self.buffers[f"velocity/{self.step:03d}"] = velocity.clone()
                self.step += 1
            return velocity

        def traced_integrate(*args: Any, **kwargs: Any) -> Any:
            result = integrate(*args, **kwargs)
            if self.capturing():
                self.buffers[f"state/{self.steps:03d}"] = result.clone()
            return result

        self.patches = [
            patch.object(self.policy, "_encode_prefix_impl", traced_encode),
            patch.object(self.policy, "_denoise_step_impl", traced_denoise),
            patch.object(self.policy._runtime, "integrate", traced_integrate),
        ]
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for item in reversed(self.patches):
            item.stop()

    def snapshot(self, actions: Any) -> dict[str, Snapshot]:
        """Require every native layer and step to have actually been captured."""
        layers = len(self.policy._prefix_tower.layers)
        expected = {"prefix/mask", f"state/{self.steps:03d}"}
        expected |= {f"prefix/{i:03d}/{kind}" for i in range(layers) for kind in ("key", "value")}
        expected |= {f"{kind}/{i:03d}" for i in range(self.steps) for kind in ("state", "time", "velocity")}
        if self.step != self.steps or self.buffers.keys() != expected:
            raise RuntimeError("incomplete graph trace; no per-layer/per-step validation can be claimed")
        result = {key: tensor_snapshot(value) for key, value in self.buffers.items()}
        result["actions/normalized"] = tensor_snapshot(actions)
        return result


def physical_actions(actions: Any, mean: Any, std: Any, dimensions: int) -> Any:
    """Apply the fixture's declared affine scale and mean exactly once."""
    return actions[..., :dimensions] * std[:dimensions] + mean[:dimensions]


def save_trace(path: Path, trace: dict[str, Snapshot]) -> None:
    """Store analysis values and raw source storage for independent offline inspection."""
    arrays = {}
    manifest = {}
    for index, (key, snapshot) in enumerate(sorted(trace.items())):
        arrays[f"values_{index}"] = snapshot.values
        arrays[f"bytes_{index}"] = np.frombuffer(snapshot.storage, dtype=np.uint8)
        manifest[key] = dict(index=index, **snapshot.metadata())
    arrays["manifest_json"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(path, **arrays)


def make_parser() -> argparse.ArgumentParser:
    """Define one reproducible protocol; lossless tolerances are not CLI knobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Host Git facts with checked source hashes for containers without Git",
    )
    parser.add_argument("--out", type=Path, required=True, help="New result directory; never overwrite a run")
    parser.add_argument("--indices", type=int, nargs="+", help="Defaults to fixture validation_indices")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
    parser.add_argument(
        "--horizon",
        type=int,
        help="Defaults to checkpoint chunk_size; explicit shorter values are disclosed as a non-native benchmark",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Defaults to checkpoint num_inference_steps; an explicit override applies identically to both paths",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Repeat each fixed observation across this batch"
    )
    parser.add_argument("--threads", type=int, default=1, help="Explicit Torch intra-op CPU threads")
    parser.add_argument("--interop-threads", type=int, default=1, help="Requested Torch inter-op CPU threads")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="highest")
    parser.add_argument("--attention", choices=("sdpa", "triton"), default="sdpa")
    parser.add_argument("--compile-backend", choices=("none", "inductor"), default="none")
    parser.add_argument(
        "--normalized-only",
        action="store_true",
        help="Validate model-space outputs when robot action statistics are unavailable; physical/gripper validation remains unverified",
    )
    parser.add_argument("--physical-action-dim", type=int)
    parser.add_argument("--gripper-index", type=int)
    parser.add_argument("--gripper-threshold", type=float)
    parser.add_argument("--gripper-comparison", choices=("gt", "ge", "lt", "le"))
    parser.add_argument(
        "--fp32-diagnostic",
        action="store_true",
        help="Separate untimed eager FP32 cast diagnostic; never changes the lossless gate",
    )
    return parser


def configure_threads(torch_api: Any, threads: int, interop_threads: int) -> dict[str, Any]:
    """Apply explicit CPU thread limits early; disclose a previously fixed inter-op pool."""
    if min(threads, interop_threads) < 1:
        raise ValueError("thread counts must be positive")
    torch_api.set_num_threads(threads)
    error = None
    try:
        torch_api.set_num_interop_threads(interop_threads)
    except RuntimeError as exc:
        error = str(exc)
    return dict(
        requested_intraop=threads,
        actual_intraop=torch_api.get_num_threads(),
        requested_interop=interop_threads,
        actual_interop=torch_api.get_num_interop_threads(),
        interop_configuration_error=error,
    )


def resolve_sampling(native: Any, horizon: int | None, steps: int | None) -> dict[str, Any]:
    """Inherit native generation dimensions unless the caller explicitly overrides them."""
    source = dict(action_dim="max_action_dim", horizon="chunk_size", steps="num_inference_steps")
    checkpoint = {key: getattr(native, attribute, None) for key, attribute in source.items()}
    if any(type(value) is not int or value < 1 for value in checkpoint.values()):
        raise ValueError(
            "checkpoint must declare positive max_action_dim, chunk_size and num_inference_steps"
        )
    resolved = dict(
        action_dim=checkpoint["action_dim"],
        horizon=checkpoint["horizon"] if horizon is None else horizon,
        steps=checkpoint["steps"] if steps is None else steps,
    )
    if any(type(value) is not int or value < 1 for value in resolved.values()):
        raise ValueError("explicit horizon and steps must be positive integers")
    if resolved["horizon"] > checkpoint["horizon"]:
        raise ValueError("benchmark horizon cannot exceed the checkpoint chunk_size")
    return dict(
        source={key: f"loaded_checkpoint.config.{attribute}" for key, attribute in source.items()},
        checkpoint=checkpoint,
        resolved=resolved,
        explicit_overrides=dict(horizon=horizon, steps=steps),
        matches_checkpoint=resolved == checkpoint,
        scope="both paths use these identical resolved values; explicit non-native settings do not establish deployment parity",
    )


def clear_engine_graphs(engine: Any) -> None:
    """Drop diagnostic or measured graphs before building the next clean capture."""
    engine.policy._clear_inference_caches()
    if engine._graphs is not None:
        engine._graphs._graphs.clear()


def validate_real_lifecycle(engine: Any, cases: list, steps: int) -> dict[str, Any]:
    """Exercise real-weight stream isolation, owned outputs, refit and recovery.

    The one in-memory action-projection perturbation is restored exactly before
    returning. No checkpoint file or timed inference setting is modified.
    """
    import torch

    policy, runtime = engine.policy, engine.policy._runtime
    prepared = [runtime._prepare_batch(case[2].to(engine.device, engine.dtype))[0] for case in cases]
    noises = [
        policy.decoder.init_state(
            case[2].batch_size, torch.Generator(device=engine.device).manual_seed(case[1])
        )
        for case in cases
    ]
    first = runtime.request(prepared[0], noises[0], steps)
    saved = first.clone()
    changed = runtime.request(prepared[-1], noises[-1], steps)
    repeated = runtime.request(prepared[0], noises[0], steps)
    torch.cuda.synchronize()
    rows = [
        dict(kind="real_repeat", **compare(tensor_snapshot(saved), tensor_snapshot(repeated))),
        dict(kind="real_output_ownership", **compare(tensor_snapshot(saved), tensor_snapshot(first))),
    ]
    if first.data_ptr() == repeated.data_ptr() or first.data_ptr() == changed.data_ptr():
        raise RuntimeError("real request outputs alias a later request")
    caller = torch.cuda.current_stream(engine.device)
    streams = [torch.cuda.Stream(device=engine.device) for _ in range(2)]
    stream_results = []
    for stream in streams:
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            stream_results.append(runtime.request(prepared[0], noises[0], steps))
    torch.cuda.synchronize()
    for index, value in enumerate(stream_results):
        rows.append(
            dict(kind=f"real_stream_{index}", **compare(tensor_snapshot(saved), tensor_snapshot(value)))
        )
    weight = policy._m.action_out_proj.weight
    original = weight.detach().clone()
    try:
        weight[0, 0].add_(0.01)
        old = list(runtime.request_graphs.values())
        policy.on_refit(policy.policy_version + 1)
        if runtime.request_graphs or not all(graph.captured._closed for graph in old):
            raise RuntimeError("real refit retained a captured request")
        refit = runtime.request(prepared[0], noises[0], steps)
        policy.request_cuda_graph = False
        prefix = policy.encode_prefix(prepared[0])
        reference = policy.decoder.integrate(noises[0], prefix, steps, prepared[0].batch_size, engine._graphs)
        rows.append(
            dict(
                kind="real_refit_two_stage_parity",
                **compare(tensor_snapshot(reference), tensor_snapshot(refit)),
            )
        )
        if torch.equal(saved, refit):
            raise RuntimeError("refit perturbation did not affect the measured output")
    finally:
        weight.copy_(original)
        policy.request_cuda_graph = True
        policy.on_refit(policy.policy_version + 1)
    restored = runtime.request(prepared[0], noises[0], steps)
    rows.append(dict(kind="real_refit_restore", **compare(tensor_snapshot(saved), tensor_snapshot(restored))))
    runtime.clear()
    original_encode = policy._encode_prefix_impl

    def fail_in_capture(*args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("real-weight injected capture failure")
        return original_encode(*args, **kwargs)

    with patch.object(policy, "_encode_prefix_impl", fail_in_capture):
        try:
            runtime.request(prepared[0], noises[0], steps)
        except RuntimeError as error:
            if "injected capture failure" not in str(error):
                raise
        else:
            raise RuntimeError("injected real-weight capture did not fail")
    if runtime.request_graphs or torch.cuda.current_stream(engine.device).cuda_stream != caller.cuda_stream:
        raise RuntimeError("failed real capture published a graph or changed the caller stream")
    recovered = runtime.request(prepared[0], noises[0], steps)
    rows.append(
        dict(kind="real_capture_recovery", **compare(tensor_snapshot(saved), tensor_snapshot(recovered)))
    )
    return dict(
        passed=all(row["passed"] for row in rows),
        comparisons=rows,
        output_storage_independent=True,
        distinct_streams=2,
        refit_invalidated=True,
        capture_failure_recovered=True,
        weight_restored=True,
    )


def main() -> None:
    """Run fixed-input graph parity before reporting uninstrumented full-request times."""
    parser = make_parser()
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.warmup,
            args.iterations,
            args.threads,
            args.interop_threads,
        )
        < 1
    ):
        parser.error("shape, thread and timing counts must be positive")
    if any(value is not None and value < 1 for value in (args.horizon, args.steps)):
        parser.error("explicit horizon and steps must be positive")
    physical_fields = (
        args.physical_action_dim,
        args.gripper_index,
        args.gripper_threshold,
        args.gripper_comparison,
    )
    if args.normalized_only:
        if any(value is not None for value in physical_fields):
            parser.error("--normalized-only cannot be combined with physical action or gripper settings")
    elif any(value is None for value in physical_fields):
        parser.error(
            "physical validation requires action dimension and explicit gripper index/threshold/comparison"
        )
    elif (
        args.physical_action_dim < 1
        or not 0 <= args.gripper_index < args.physical_action_dim
        or not np.isfinite(args.gripper_threshold)
    ):
        parser.error("declare a positive action dimension, finite gripper threshold and valid index")
    if args.out.exists():
        parser.error("--out must not exist; select a new run directory")

    import torch

    thread_settings = configure_threads(torch, args.threads, args.interop_threads)
    if not torch.cuda.is_available():
        parser.error("this Pi05 adapter requires CUDA and a matching checkpoint environment")
    source_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source_root))

    import embodiinfer
    from embodiinfer.backend.torch.request_graph import cuda_execution_signature
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.config import VLAPolicyConfig
    from embodiinfer.policies.pi05.checkpoints.lerobot import load_lerobot_checkpoint
    from embodiinfer.policies.pi05.inference.runtime import _stream_key, _tensor_key
    from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
    from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

    if Path(embodiinfer.__file__).resolve().parent != source_root / "embodiinfer":
        raise RuntimeError("imported runtime is not from the benchmark checkout")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(42)
    fixture = torch.load(args.inputs, map_location="cpu", weights_only=True)
    indices = args.indices if args.indices is not None else fixture["validation_indices"]
    if (
        not indices
        or any(index < 0 or index >= len(fixture["observations"]) for index in indices)
        or len(set(indices)) != len(indices)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("indices must be valid/nonempty/unique and seeds must be unique")
    mean = std = None
    if not args.normalized_only:
        mean, std = (torch.as_tensor(fixture[key]).float().flatten() for key in ("action_mean", "action_std"))
        if mean.shape != std.shape or args.physical_action_dim > mean.numel():
            parser.error(
                "fixture action mean/std must match and cover the selected physical action dimensions"
            )
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            parser.error("fixture action statistics must be finite with positive effective std")
    cases = []
    for index in indices:
        item = fixture["observations"][index]
        images = [image.repeat(args.batch_size, 1, 1, 1) for image in item["images"]]
        if not images or item["tokens"].shape[0] != 1:
            parser.error("fixture requires camera tensors and one row per observation")
        if "img_masks" in item:
            image_masks = [mask.repeat(args.batch_size) for mask in item["img_masks"]]
            if len(image_masks) != len(images):
                parser.error("explicit camera masks must match the fixture images")
        elif len(images) == 2:
            images += [-torch.ones_like(images[0])]
            image_masks = [torch.ones(args.batch_size, dtype=torch.bool)] * 2 + [
                torch.zeros(args.batch_size, dtype=torch.bool)
            ]
        else:
            parser.error("fixtures with other camera counts must declare img_masks explicitly")
        batch = Pi05Batch(
            images,
            image_masks,
            item["tokens"].repeat(args.batch_size, 1),
            item["masks"].repeat(args.batch_size, 1),
            [f"validation-{index}-{row}" for row in range(args.batch_size)],
        )
        for seed in args.seeds:
            cases.append((index, seed, batch))
    if args.iterations < len(cases):
        parser.error("iterations must cover each fixed observation/noise case at least once")
    device = torch.device("cuda", torch.cuda.current_device())
    validation_stream = torch.cuda.Stream(device=device)
    validation_stream.wait_stream(torch.cuda.current_stream(device))
    args.out.mkdir(parents=True)
    report: dict[str, Any] = dict(
        schema_version=1,
        passed=False,
        status="running",
        lossless_thresholds=asdict(LOSSLESS),
        baseline="same checkout with request_cuda_graph=False; all other switches identical",
        candidate="request_cuda_graph=True",
        measurement_scope="CPU-prepared batch through EngineCore.execute; includes H2D, prefix, every denoise step, D2H and packing, plus affine action restoration when physical validation is enabled; excludes image preprocessing/tokenization, network, checkpoint load and cold capture/compile",
        trace_scope="actual baseline and candidate CUDA graphs with diagnostic-only clone nodes; fresh uninstrumented captures are verified and timed separately",
        invocation=[str(value) for value in sys.argv],
        source=source_provenance(source_root, args.source_manifest),
        checkpoint_files={
            path.name: digest(path)
            for path in sorted(args.checkpoint.iterdir())
            if path.is_file() and path.suffix in (".safetensors", ".json")
        },
        fixture_sha256=digest(args.inputs),
        fixture_description=fixture.get("description", "unspecified fixture provenance"),
        physical_validation=not args.normalized_only,
        versions={
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "lerobot", "transformers")
        },
        python=platform.python_version(),
        hardware=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        cuda=torch.version.cuda,
        threads=thread_settings,
        stream=dict(
            device=str(device),
            cuda_stream=validation_stream.cuda_stream,
            priority=validation_stream.priority,
            scope="one dedicated caller stream for both modes, diagnostic capture requests, warmups, timing and optional FP32; internal capture side streams remain runtime-owned",
        ),
        settings=dict(
            batch_size=args.batch_size,
            horizon=args.horizon,
            steps=args.steps,
            dtype="auto",
            matmul_precision=args.matmul_precision,
            attention=args.attention,
            compile_backend=args.compile_backend,
            prefix_cuda_graph=True,
            capture_full_loop=True,
            quantization=None,
            optimizations=None,
        ),
        action_restoration=dict(
            formula=None if args.normalized_only else "normalized[..., :dimensions] * effective_std + mean",
            dimensions=args.physical_action_dim,
            effective_std_includes_fixture_epsilon=(
                None if args.normalized_only else fixture.get("restoration", {}).get("std_includes_epsilon")
            ),
            fixture_restoration=fixture.get("restoration"),
        ),
        cases=[dict(index=index, seed=seed) for index, seed, _ in cases],
        comparisons=[],
        modes={},
    )

    def write_report() -> None:
        (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write_report()
    try:
        with torch.inference_mode(), torch.cuda.stream(validation_stream):
            model, checkpoint_load_ms = synchronized_call(
                lambda: load_lerobot_checkpoint(
                    str(args.checkpoint), load_device="cuda", low_cpu_mem_usage=True
                ),
                torch.cuda.synchronize,
            )
            report["setup"] = dict(checkpoint_load_wall_ms=checkpoint_load_ms)
            report["checkpoint_parameter_dtypes"] = dict(
                Counter(str(value.dtype) for value in model.parameters())
            )
            sampling = resolve_sampling(model.config, args.horizon, args.steps)
            report["sampling"] = sampling
            args.horizon = sampling["resolved"]["horizon"]
            args.steps = sampling["resolved"]["steps"]
            action_dim = sampling["resolved"]["action_dim"]
            if not args.normalized_only and args.physical_action_dim > action_dim:
                raise ValueError("physical action dimensions exceed checkpoint max_action_dim")
            report["settings"].update(horizon=args.horizon, steps=args.steps, action_dim=action_dim)
            config = VLAPolicyConfig(
                name="pi0.5", action_dim=action_dim, action_horizon=args.horizon, default_num_steps=args.steps
            )

            def engine_for(request_graph: bool) -> Any:
                policy = Pi05Policy(
                    config,
                    model,
                    native_embeddings=True,
                    native_inference=True,
                    prefix_cuda_graph=True,
                    request_cuda_graph=request_graph,
                    compile_backend=args.compile_backend,
                    prefix_attention=args.attention,
                    denoise_attention=args.attention,
                )
                return EngineCore(
                    policy,
                    EngineConfig(
                        device="cuda",
                        dtype="auto",
                        max_batch_size=args.batch_size,
                        batch_buckets=(args.batch_size,),
                        use_cuda_graph=True,
                        capture_full_loop=True,
                    ),
                )

            report["precision_flags"] = dict(
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            )
            baseline_traces: list[dict[str, Snapshot]] = []
            baseline_clean: list[Snapshot] = []
            for mode in ("baseline", "candidate"):
                engine, engine_setup_ms = synchronized_call(
                    lambda selected=mode, factory=engine_for: factory(selected == "candidate"),
                    torch.cuda.synchronize,
                )
                mode_report: dict[str, Any] = dict(
                    engine_construction_wall_ms=engine_setup_ms,
                    caller_cuda_stream=torch.cuda.current_stream(device).cuda_stream,
                    diagnostic_setup=[],
                    clean_setup=[],
                )
                report["modes"][mode] = mode_report
                schedule = engine.policy.flow_schedule(args.steps)
                if mode == "candidate" and schedule != report["schedule"]:
                    raise RuntimeError("candidate changed the integration schedule")
                report["schedule"] = schedule
                generator = torch.Generator(device="cuda")

                def reset(case: int, bound_generator: Any = generator) -> None:
                    bound_generator.manual_seed(cases[case][1])

                def request(case: int, bound_engine: Any = engine, bound_generator: Any = generator) -> Any:
                    chunks = bound_engine.execute(
                        cases[case][2], num_steps=args.steps, generator=bound_generator
                    )
                    return torch.stack([chunk.actions for chunk in chunks])

                def restored_request(case: int, bound_request: Callable[[int], Any] = request) -> Any:
                    action = bound_request(case)
                    return (
                        action
                        if args.normalized_only
                        else physical_actions(action, mean, std, args.physical_action_dim)
                    )

                diagnostic_actions = []
                for case in range(len(cases)):
                    torch.cuda.synchronize()
                    setup_start = time.perf_counter()
                    clear_engine_graphs(engine)
                    with CaptureTrace(
                        engine.policy, args.steps, torch.cuda.is_current_stream_capturing
                    ) as recorder:
                        torch.cuda.synchronize()
                        setup_ms = (time.perf_counter() - setup_start) * 1000
                        reset(case)
                        action, cold_request_ms = synchronized_call(
                            lambda selected=case, bound_request=request: bound_request(selected),
                            torch.cuda.synchronize,
                        )
                        mode_report["diagnostic_setup"].append(
                            dict(
                                case=case,
                                cache_reset_and_hook_setup_wall_ms=setup_ms,
                                cold_request_wall_ms=cold_request_ms,
                                scope="cold instrumented request includes runtime internal warmups, compile/capture, execution and packing; excludes trace serialization; excluded from steady-state samples",
                            )
                        )
                        trace = recorder.snapshot(action)
                        if not args.normalized_only:
                            trace["actions/physical"] = tensor_snapshot(
                                physical_actions(action, mean, std, args.physical_action_dim)
                            )
                        save_trace(args.out / f"{mode}-{case:03d}-trace.npz", trace)
                        if mode == "baseline":
                            baseline_traces.append(trace)
                            reset(case)
                            repeated_action = request(case)
                            torch.cuda.synchronize()
                            repeated = recorder.snapshot(repeated_action)
                            if not args.normalized_only:
                                repeated["actions/physical"] = tensor_snapshot(
                                    physical_actions(repeated_action, mean, std, args.physical_action_dim)
                                )
                            save_trace(args.out / f"baseline-{case:03d}-repeat.npz", repeated)
                            result = compare_traces(trace, repeated)
                            report["comparisons"].append(dict(kind="baseline_repeat", case=case, **result))
                        else:
                            result = compare_traces(baseline_traces[case], trace)
                            if not args.normalized_only:
                                result["gripper"] = gripper_flips(
                                    baseline_traces[case]["actions/physical"],
                                    trace["actions/physical"],
                                    index=args.gripper_index,
                                    threshold=args.gripper_threshold,
                                    comparison=args.gripper_comparison,
                                )
                                result["passed"] &= result["gripper"]["passed"]
                            report["comparisons"].append(dict(kind="candidate_trace", case=case, **result))
                        diagnostic_actions.append(trace["actions/normalized"])
                    del recorder
                    write_report()

                # Remove all diagnostic clone nodes before correctness recheck/timing.
                _, clear_ms = synchronized_call(
                    lambda bound_engine=engine: clear_engine_graphs(bound_engine), torch.cuda.synchronize
                )
                mode_report["diagnostic_cache_teardown_wall_ms"] = clear_ms
                for case in range(len(cases)):
                    cache_before = engine.policy._runtime.stats()
                    reset(case)
                    _, first_request_ms = synchronized_call(
                        lambda selected=case, bound_request=request: bound_request(selected),
                        torch.cuda.synchronize,
                    )
                    cache_after = engine.policy._runtime.stats()
                    torch.cuda.synchronize()
                    warmup_start = time.perf_counter()
                    for _ in range(args.warmup):
                        reset(case)
                        request(case)
                    torch.cuda.synchronize()
                    mode_report["clean_setup"].append(
                        dict(
                            case=case,
                            first_request_wall_ms=first_request_ms,
                            constructed_graphs=cache_before != cache_after,
                            graph_stats_before=cache_before,
                            graph_stats_after=cache_after,
                            warmup_requests=args.warmup,
                            warmup_wall_ms=(time.perf_counter() - warmup_start) * 1000,
                            scope="first clean request includes cache construction only on cache miss; per-case warmup is separate; all excluded from steady-state samples",
                        )
                    )
                    reset(case)
                    actual = tensor_snapshot(request(case))
                    save_trace(args.out / f"{mode}-{case:03d}-clean.npz", {"actions/normalized": actual})
                    report["comparisons"].append(
                        dict(
                            kind=f"{mode}_uninstrumented",
                            case=case,
                            **compare(diagnostic_actions[case], actual),
                        )
                    )
                    if mode == "baseline":
                        baseline_clean.append(actual)
                    else:
                        report["comparisons"].append(
                            dict(
                                kind="candidate_uninstrumented_parity",
                                case=case,
                                **compare(baseline_clean[case], actual),
                            )
                        )
                if (
                    torch.get_num_threads() != thread_settings["actual_intraop"]
                    or torch.get_num_interop_threads() != thread_settings["actual_interop"]
                ):
                    raise RuntimeError("CPU thread settings changed during validation")
                before = engine.policy._runtime.stats()
                timing = benchmark_requests(
                    restored_request,
                    cases=len(cases),
                    synchronize=torch.cuda.synchronize,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    before_request=reset,
                )
                after = engine.policy._runtime.stats()
                mode_report.update(
                    timing=timing,
                    graph_stats_before=before,
                    graph_stats_after=after,
                    graph_cache_stable=before == after,
                )
                if before != after:
                    raise RuntimeError("graph cache changed during the warmed measurement")
                clear_engine_graphs(engine)
                del engine, request, restored_request, reset
                torch.cuda.empty_cache()
                write_report()

            # Fresh captures, paired complete requests, and isolated replay boundaries.
            engines = {name: engine_for(name == "candidate") for name in ("baseline", "candidate")}
            report["real_weight_lifecycle"] = validate_real_lifecycle(engines["candidate"], cases, args.steps)
            report["comparisons"].extend(report["real_weight_lifecycle"]["comparisons"])
            generators = {name: torch.Generator(device=device) for name in engines}

            def paired_reset(name: str, case: int, bound_generators: Any = generators) -> None:
                bound_generators[name].manual_seed(cases[case][1])

            def paired_request(
                name: str, case: int, bound_engines: Any = engines, bound_generators: Any = generators
            ) -> Any:
                chunks = bound_engines[name].execute(
                    cases[case][2], num_steps=args.steps, generator=bound_generators[name]
                )
                action = torch.stack([chunk.actions for chunk in chunks])
                return (
                    action
                    if args.normalized_only
                    else physical_actions(action, mean, std, args.physical_action_dim)
                )

            full_requests = {
                name: (lambda case, selected=name, call=paired_request: call(selected, case))
                for name in engines
            }
            report["paired_complete_request"] = benchmark_paired_requests(
                full_requests,
                cases=len(cases),
                synchronize=torch.cuda.synchronize,
                warmup=args.warmup,
                iterations=args.iterations,
                before_request=paired_reset,
            )
            replay_requests = {}
            for name, engine in engines.items():
                runtime = engine.policy._runtime
                replay_cases = []
                for case, (_, seed, cpu_batch) in enumerate(cases):
                    gpu_batch, all_valid = runtime._prepare_batch(cpu_batch.to(device, engine.dtype))
                    noise = engine.policy.decoder.init_state(
                        args.batch_size, torch.Generator(device=device).manual_seed(seed)
                    )
                    paired_reset(name, case)
                    full_requests[name](case)
                    batch_key = (
                        tuple(_tensor_key(x) for x in gpu_batch.images + gpu_batch.img_masks),
                        _tensor_key(gpu_batch.tokens),
                        _tensor_key(gpu_batch.masks),
                    )
                    stream_modes = (_stream_key(device), cuda_execution_signature())
                    if name == "candidate":
                        key = (
                            *batch_key,
                            _tensor_key(noise),
                            all_valid,
                            args.steps,
                            engine.policy.policy_version,
                            *stream_modes,
                        )
                        graph = runtime.request_graphs[key]
                        replay_cases.append(
                            lambda graph=graph, batch=gpu_batch, state=noise: graph.run(batch, state)
                        )
                    else:
                        prefix_graph = runtime.prefix_graphs[(*batch_key, all_valid, *stream_modes)]
                        loop_graph = runtime.loop_graphs[
                            (
                                _tensor_key(noise),
                                args.steps,
                                _tensor_key(prefix_graph.output.prefix_pad_masks),
                                all_valid,
                                *stream_modes,
                            )
                        ]
                        replay_cases.append(
                            lambda pg=prefix_graph, lg=loop_graph, batch=gpu_batch, state=noise: lg.run(
                                state, pg.run(batch)
                            )
                        )
                replay_requests[name] = lambda case, calls=replay_cases: calls[case]()
                for case in range(len(cases)):
                    actual = tensor_snapshot(replay_requests[name](case))
                    report["comparisons"].append(
                        dict(kind=f"{name}_replay_parity", case=case, **compare(baseline_clean[case], actual))
                    )
            report["paired_graph_replay"] = benchmark_paired_requests(
                replay_requests,
                cases=len(cases),
                synchronize=torch.cuda.synchronize,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            report["paired_graph_replay"]["scope"] = (
                "GPU-resident prepared inputs and fixed noise through static-input copies, prefix/loop replay and owned output clone; baseline includes boundary KV clones/copies; excludes H2D, compaction, D2H, packing and physical restoration"
            )
            report["paired_complete_request"]["scope"] = report["measurement_scope"]
            for engine in engines.values():
                clear_engine_graphs(engine)
            del (
                engines,
                generators,
                full_requests,
                replay_requests,
                replay_cases,
                engine,
                runtime,
                graph,
                prefix_graph,
                loop_graph,
                paired_request,
                paired_reset,
            )
            torch.cuda.empty_cache()

            if args.fp32_diagnostic:
                # A distinct, untimed precision diagnostic. It cannot approve the optimization.
                del model, engine_for
                gc.collect()
                torch.cuda.empty_cache()
                diagnostic_model = load_lerobot_checkpoint(
                    str(args.checkpoint), load_device="cuda", low_cpu_mem_usage=True
                ).float()
                torch.set_float32_matmul_precision("highest")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                diagnostic = Pi05Policy(
                    config,
                    diagnostic_model,
                    native_embeddings=True,
                    native_inference=True,
                    prefix_cuda_graph=False,
                    request_cuda_graph=False,
                    compile_backend="none",
                    prefix_attention="sdpa",
                    denoise_attention="sdpa",
                ).eval()
                diagnostic_rows = []
                with ephemeral_packed_mlp_weights():
                    for case, (_, _seed, batch) in enumerate(cases):
                        noise = fp32_initial_state(baseline_traces[case]["state/000"], device)
                        prefix = diagnostic.encode_prefix(batch.to("cuda", torch.float32))
                        action = (
                            diagnostic.decoder.integrate(noise, prefix, args.steps, args.batch_size, None)
                            .float()
                            .cpu()
                        )
                        diagnostic_rows.append(
                            dict(case=case, **compare(baseline_clean[case], tensor_snapshot(action)))
                        )
                        diagnostic._clear_inference_caches()
                report["fp32_diagnostic"] = dict(
                    gating=False,
                    description="untimed native SDPA FP32 promotion of the same checkpoint and captured initial state; TF32 disabled for matmul/cuDNN; original packed GEMM shapes/computation are retained but their temporary gate/up copies are released after use to bound memory; this diagnostic-only cache policy is not used in timed/captured paths, is not higher-precision source weights, and cannot relax the lossless gate",
                    comparisons=diagnostic_rows,
                )
                diagnostic._clear_inference_caches()

            report["passed"] = all(row["passed"] for row in report["comparisons"])
            report["status"] = "passed" if report["passed"] else "failed"
            report["timings_eligible_for_analysis"] = report["passed"]
            report["measurement_order"] = ["baseline", "candidate"]
            report["limits"] = (
                "single-host fixed-fixture measurements; initial phase timings are sequential, supplementary fresh captures alternate paired order; no closed-loop task-success or fleet-wide claim; normalized-only runs leave physical actions and gripper decisions unverified"
            )
            imported_files = {
                name: str(Path(module.__file__).resolve())
                for name, module in tuple(sys.modules.items())
                if name.startswith("embodiinfer") and getattr(module, "__file__", None)
            }
            if any(
                not Path(path).is_relative_to(source_root / "embodiinfer") for path in imported_files.values()
            ):
                raise RuntimeError("a runtime import escaped the measured source checkout")
            report["imported_package_files"] = imported_files
            write_report()
    except BaseException as exc:
        report["passed"] = False
        report["timings_eligible_for_analysis"] = False
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report()
        raise
    print(json.dumps(dict(passed=report["passed"], report=str(args.out / "report.json"))))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
