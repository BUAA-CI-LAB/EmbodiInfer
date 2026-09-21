"""LingBot-VLA offline benchmark: sampling, timing and inference in one file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

import embodiinfer
from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.policies import make_policy
from embodiinfer.types import Observation

ROOT = Path(embodiinfer.__file__).resolve().parent.parent


def positive(value: Any, name: str) -> int:
    """Reject ambiguous, zero, or negative selection sizes."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def uniform_indices(length: int, count: int) -> tuple[int, ...]:
    """Select distinct integer quantiles, including both endpoints when count > 1."""
    positive(count, "count")
    if count > length:
        raise ValueError(f"requested {count} distinct frames from only {length}")
    if count == 1:
        return (0,)
    return tuple(i * (length - 1) // (count - 1) for i in range(count))


def digest_json(value: Any) -> str:
    """Fingerprint ordered sample identities or a resolved configuration."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class RobotwinSample:
    """One original RoboTwin joint-observation frame with a fixed public instruction."""

    path: Path
    task: str
    demo: int
    frame: int
    instruction: str

    @property
    def sample_id(self) -> str:
        """Identify the same raw frame independently of installation path."""
        return f"robotwin/{self.task}/aloha-agilex_clean_50/episode{self.demo}/{self.frame}"


def load_robotwin(config: dict[str, Any]) -> tuple[RobotwinSample, ...]:
    """Select sorted tasks, numeric first episodes and uniform frames, including endpoints."""
    import h5py

    root = Path(config["root"]).expanduser()
    manifest = json.loads((root / "selection.json").read_text())
    if manifest["repository"] != config["repository"] or manifest["revision"] != config["revision"]:
        raise ValueError("prepared RoboTwin data revision differs from benchmark config")
    tasks = positive(config["task_limit"], "task_limit")
    demos = positive(config["demos_per_task"], "demos_per_task")
    frames = positive(config["frames_per_demo"], "frames_per_demo")
    if len(manifest["tasks"]) < tasks:
        raise ValueError("prepare more tasks before increasing task_limit")
    samples = []
    for task in manifest["tasks"][:tasks]:
        directory = root / task / "aloha-agilex_clean_50"
        # Clean episodes are numbered 0..49. A missing selected episode must fail,
        # never shift the selection to a later available file.
        for demo in range(demos):
            path = directory / "data" / f"episode{demo}.hdf5"
            instructions = json.loads((directory / "instructions" / f"episode{demo}.json").read_text())
            instruction = instructions["seen"][0]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{task}/{demo}: missing public instruction")
            with h5py.File(path) as handle:
                state = handle["joint_action/vector"]
                if len(state.shape) != 2 or state.shape[1] != 14:
                    raise ValueError("LingBot Robotwin checkpoint requires the 14D joint dataset")
                count = len(state)
                for camera in ("head_camera", "left_camera", "right_camera"):
                    if len(handle[f"observation/{camera}/rgb"]) != count:
                        raise ValueError(f"{task}/{demo}: misaligned {camera}")
            samples.extend(
                RobotwinSample(path, task, demo, i, instruction) for i in uniform_indices(count, frames)
            )
    limit = config.get("sample_limit")
    if limit is not None:
        positive(limit, "sample_limit")
        if limit > len(samples):
            raise ValueError("sample_limit exceeds selected RoboTwin frames")
        samples = samples[:limit]
    return tuple(samples)


def read_robotwin(sample: RobotwinSample) -> dict[str, Any]:
    """Decode the three original JPEG observations and read joint state outside the timer."""
    from io import BytesIO

    import h5py

    with h5py.File(sample.path) as handle:
        state = np.asarray(handle["joint_action/vector"][sample.frame], dtype=np.float32)
        images = []
        for camera in ("head_camera", "left_camera", "right_camera"):
            encoded = handle[f"observation/{camera}/rgb"][sample.frame]
            with Image.open(BytesIO(bytes(encoded))) as image:
                images.append(image.convert("RGB"))
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"{sample.sample_id}: invalid joint state")
    return {"state": state, "images": images}


def load_config(default: Path, *, model_choices: tuple[str, ...] = ()) -> tuple[dict[str, Any], Path]:
    """Read the sibling YAML, allowing a separate config for smoke runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument(
        "--validate-data", action="store_true", help="validate selected data without loading weights"
    )
    if model_choices:
        parser.add_argument("--model", choices=model_choices)
    args = parser.parse_args()
    path = args.config.resolve(strict=True)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected an offline benchmark schema_version: 1 mapping")
    for key in ("warmup_calls", "repeats"):
        positive(config[key], key)
    for key in ("output", "output_dir"):
        if key in config:
            output = Path(config[key]).expanduser()
            config[key] = str(output if output.is_absolute() else path.parent / output)
    config["validate_data_only"] = args.validate_data
    if model_choices:
        config["selected_model"] = args.model
    return config, path


def cuda_device(config: dict[str, Any]) -> torch.device:
    """Resolve one CUDA device and establish the recorded numerical settings."""
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("performance measurement requires an available CUDA device")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(device)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.set_float32_matmul_precision("highest")
    return device


def timed_model(
    prefill: Callable[[], Any], decode: Callable[[Any], Any], device: torch.device
) -> tuple[Any, Any, dict[str, float]]:
    """Measure device-ready input through all model generation, before output transforms.

    CUDA events measure prefill and the complete decode loop without a sync between
    stages. The synchronized wall interval also includes host dispatch/sampling.
    Input preprocessing and H2D must finish before entry; output transforms run after
    return. These are elapsed intervals, not a sum of individual kernel durations.
    """
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    events[0].record(torch.cuda.current_stream(device))
    prefix = prefill()
    events[1].record(torch.cuda.current_stream(device))
    result = decode(prefix)
    events[2].record(torch.cuda.current_stream(device))
    events[2].synchronize()
    wall_ms = (time.perf_counter_ns() - start) / 1e6
    return (
        prefix,
        result,
        {
            "prefill_ms": float(events[0].elapsed_time(events[1])),
            "decode_ms": float(events[1].elapsed_time(events[2])),
            "gpu_inference_ms": float(events[0].elapsed_time(events[2])),
            "pure_inference_ms": wall_ms,
        },
    )


def timed_call(callback: Callable[[], dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """Time decoded CPU input through preprocessing, inference, and CPU output."""
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    result = callback()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    output = result.pop("_cpu_output")
    if output.is_floating_point() and not torch.isfinite(output).all():
        raise ValueError("model must return finite output values")
    result["output_sha256"] = hashlib.sha256(output.numpy().tobytes()).hexdigest()
    return {**result, "latency_ms": elapsed_ms}


def output_record(actions: torch.Tensor, token_count: int = 0) -> dict[str, Any]:
    """Materialize the CPU action chunk; numerical checks and hashing happen after timing."""
    actions = actions.detach().float().cpu()
    if actions.ndim != 2 or not actions.numel():
        raise ValueError("model must return a nonempty action chunk")
    return {
        "action_slots": actions.shape[0],
        "action_shape": list(actions.shape),
        "generated_tokens": token_count,
        "_cpu_output": actions,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use total completed calls / total inference time, never mean inverse latency."""
    if not rows:
        raise ValueError("cannot report a benchmark without measured calls")
    latencies = np.asarray([row["latency_ms"] for row in rows], dtype=float)
    if not np.isfinite(latencies).all() or np.any(latencies <= 0):
        raise ValueError("latencies must be positive and finite")
    seconds = float(latencies.sum() / 1000)
    metrics = {
        "calls": len(rows),
        "inference_seconds": seconds,
        "latency_ms": {
            "mean": float(latencies.mean()),
            **{f"p{p}": float(np.percentile(latencies, p)) for p in (50, 95, 99)},
        },
        "calls_per_second": len(rows) / seconds,
        "action_slots_per_second": sum(row["action_slots"] for row in rows) / seconds,
        "generated_tokens_per_second": sum(row["generated_tokens"] for row in rows) / seconds,
    }

    if any("model_timing_ms" in row for row in rows):
        model_metrics = {}
        for name in ("prefill_ms", "decode_ms", "gpu_inference_ms", "pure_inference_ms"):
            values = np.asarray([row["model_timing_ms"][name] for row in rows], dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("model timing must be finite and nonnegative")
            model_metrics[name] = {
                "mean": float(values.mean()),
                **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
            }
        model_seconds = sum(row["model_timing_ms"]["pure_inference_ms"] for row in rows) / 1000
        if model_seconds <= 0:
            raise ValueError("pure model inference time must be positive")
        metrics["model_timing_ms"] = model_metrics
        metrics["model_calls_per_second"] = len(rows) / model_seconds
    return metrics


def provenance(device: torch.device) -> dict[str, Any]:
    """Identify hardware, source contents, interpreter, and installed distributions."""
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "triton",
        "diffusers",
        "OpenDM",
        "liger-kernel",
        "timm",
        "numpy",
        "Pillow",
        "h5py",
        "pyarrow",
        "av",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    digest = hashlib.sha256()
    runtime = Path(embodiinfer.__file__).resolve().parent
    for folder, label in (
        (runtime, "embodiinfer"),
        (Path(__file__).resolve().parent, Path(__file__).resolve().parent.name),
    ):
        for path in sorted(folder.rglob("*.py") if label == "embodiinfer" else folder.glob("*.py")):
            if any(part.startswith(".") for part in path.relative_to(folder).parts):
                continue
            digest.update((label + "/" + str(path.relative_to(folder))).encode())
            digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    commands = {
        "driver_version": ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        "power_mode": ["nvpmodel", "-q"],
    }
    hardware = {}
    for name, command in commands.items():
        if shutil.which(command[0]):
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            hardware[name] = result.stdout.strip() if result.returncode == 0 else None
    check_path = Path(sys.prefix) / "dependency-check.json"
    return {
        "gpu": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "virtual_environment": sys.prefix,
        "packages": versions,
        "source_revision": revision,
        "source_python_sha256": digest.hexdigest(),
        "dependency_check": json.loads(check_path.read_text()) if check_path.is_file() else None,
        **hardware,
    }


def write_report(
    config: dict[str, Any],
    device: torch.device,
    rows: list[dict[str, Any]],
    details: dict[str, Any],
    output: Path,
) -> None:
    """Write one model/dataset report only after all requested measured calls finish."""
    report = {
        "schema": "rlinf_offline_performance_v2",
        "report_created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": provenance(device),
        "batch_size": 1,
        "timing_boundary": "decoded_cpu_rgb_and_state_to_cpu_action_chunk_including_pre_and_postprocessing",
        "excluded": [
            "weights_loading",
            "file_read_and_decode",
            "warmup",
            "simulation",
            "network",
            "report_validation_and_hashing",
        ],
        "model_timing_contract": {
            "prefill_ms": "CUDA elapsed: vision/text/state encoding to reusable prefix",
            "decode_ms": "CUDA elapsed: all generation steps including noise/sampling, before action restoration",
            "gpu_inference_ms": "CUDA elapsed across prefill plus complete decode; no intermediate synchronization",
            "pure_inference_ms": "synchronized wall time across the same model interval, including host dispatch",
            "excluded": ["input_preprocessing", "input_H2D", "output_postprocessing", "output_D2H"],
            "e2e_note": "original CPU observation-to-action scope retained; includes profiling overhead",
        },
        "runtime": "embodiinfer",
        "metrics": summarize(rows),
        "memory": {
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "note": "RSS is process lifetime peak; on Thor CPU/GPU share physical memory, do not add these figures",
        },
        **details,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Atomically publish without replacing an earlier report.
    import os

    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()
    print(json.dumps({"report": str(output), "metrics": report["metrics"]}), flush=True)


RobotwinInference = Callable[[RobotwinSample, dict[str, Any]], dict[str, Any]]


RuntimeStats = Callable[[], dict[str, Any]]


def engine_graph_stats(core: Any) -> dict[str, Any]:
    """Inspect captured engine graphs for benchmark provenance, without changing execution."""
    # EngineCore currently has no public graph diagnostics method. Keep this
    # read-only inspection in benchmark code rather than exposing model internals.
    manager = core._graphs
    entries = [] if manager is None else list(manager._graphs.items())
    return {
        "enabled": manager is not None,
        "capture_count": len(entries),
        "entries": [{"key": repr(key), "kind": type(graph).__name__} for key, graph in entries],
    }


def run(
    default: Path,
    build: Callable[[dict[str, Any], torch.device], tuple[RobotwinInference, dict[str, Any], RuntimeStats]],
) -> None:
    """Run one model on the selected public frames, keeping disk IO outside the timer."""
    config, _ = load_config(default)
    samples = load_robotwin(config["dataset"])
    identities = [sample.sample_id for sample in samples]
    details = {
        "sample_ids": identities,
        "selection_sha256": digest_json(identities),
        "dataset": config["dataset"],
        "selected_samples": len(samples),
    }
    if config["validate_data_only"]:
        for sample in samples:
            read_robotwin(sample)
        print(f"Validated {len(samples)} RoboTwin samples; selection={details['selection_sha256']}")
        return
    if Path(config["output"]).exists():
        raise FileExistsError(f"report already exists: {config['output']}")
    device = cuda_device(config)
    started = time.perf_counter()
    infer, model_details, runtime_stats = build(config, device)
    details.update(model_details)
    details["model_load_seconds"] = time.perf_counter() - started
    rows = []
    with torch.inference_mode():
        started = time.perf_counter()
        warmup_indices = uniform_indices(len(samples), min(config["warmup_calls"], len(samples)))
        for index in range(config["warmup_calls"]):
            sample = samples[warmup_indices[index % len(warmup_indices)]]
            infer(sample, read_robotwin(sample))
        torch.cuda.synchronize(device)
        details["warmup_seconds_including_data_io"] = time.perf_counter() - started
        details["warmup_sample_ids"] = [samples[index].sample_id for index in warmup_indices]
        before = runtime_stats()
        details["runtime_before_measurement"] = before
        if config["cuda_graph"] and not before["graphs"]["capture_count"]:
            raise RuntimeError("CUDA Graph was requested but warmup captured no graph")
        if not before["compiler_stats"].get("unique_graphs", 0):
            raise RuntimeError("Inductor was requested but compiled no graph during warmup")
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(config["seed"])
        for repeat in range(config["repeats"]):
            for index, sample in enumerate(samples):
                raw = read_robotwin(sample)
                row = timed_call(partial(infer, sample, raw), device)
                rows.append({"sample_id": sample.sample_id, "repeat": repeat, **row})
                if index % 25 == 0:
                    print(
                        f"{config['model']}: {index + 1}/{len(samples)}, {row['latency_ms']:.1f} ms",
                        flush=True,
                    )
    after = runtime_stats()
    details["runtime_after_measurement"] = after
    if after["graphs"]["capture_count"] != before["graphs"]["capture_count"]:
        raise RuntimeError("CUDA Graph capture occurred during measurement; extend warmup")
    if after["compiler_stats"].get("unique_graphs", 0) != before["compiler_stats"].get("unique_graphs", 0):
        raise RuntimeError("Compilation occurred during measurement; extend warmup")
    write_report(config, device, rows, details, Path(config["output"]))


def compile_inference(policy: Any, config: dict[str, Any], methods: tuple[str, ...]) -> None:
    """Compile tensor execution while the engine owns CUDA Graph capture."""
    if config["compile_backend"] != "inductor":
        raise ValueError("the optimized profile requires compile_backend: inductor")
    for name in methods:
        setattr(
            policy,
            name,
            torch.compile(getattr(policy, name), backend="inductor", options={"triton.cudagraphs": False}),
        )


def runtime_details(core: Any, policy: Any) -> dict[str, Any]:
    """Record captured graphs, actual attention routing and compiler counters."""
    from torch._dynamo.utils import counters

    return {
        "graphs": engine_graph_stats(core),
        "compile_backend": "inductor",
        "compiler_stats": dict(counters["stats"]),
        "attention": getattr(policy, "attention", None),
    }


def make_core(policy: Any, config: dict[str, Any], device: torch.device) -> EngineCore:
    """Use the same batch-one engine and full-loop capture on both platforms."""
    return EngineCore(
        policy,
        EngineConfig(
            device=str(device),
            dtype=config["dtype"],
            max_batch_size=1,
            batch_buckets=(1,),
            use_cuda_graph=config["cuda_graph"],
            capture_full_loop=config["cuda_graph"],
            reuse_prefix_kv=True,
        ),
    )


class RobotwinProcessor:
    """Apply the published Robotwin joint mapping, bounds_99 statistics and image padding."""

    def __init__(self, norm_stats: Path):
        self.statistics = json.loads(norm_stats.read_text())["norm_stats"]
        for key, width in (("arm.position", 12), ("effector.position", 2)):
            for prefix in ("observation.state", "action"):
                if len(self.statistics[f"{prefix}.{key}"]["q01"]) != width:
                    raise ValueError("expected the official 12D arm and 2D gripper normalization")

    def normalize(self, key: str, values: np.ndarray) -> np.ndarray:
        """Match native bounds_99, including its epsilon and absence of clipping."""
        low = np.asarray(self.statistics[key]["q01"], dtype=np.float32)
        high = np.asarray(self.statistics[key]["q99"], dtype=np.float32)
        return (values - low) / (high - low + 1e-6) * 2 - 1

    def prepare(self, sample: RobotwinSample, raw: dict[str, Any]) -> Observation:
        """Arrange arm12/pad2/gripper2/pad59 and resize/pad RGB to 224 before patchification."""
        import torch.nn.functional as functional

        joint = raw["state"]
        arm = np.concatenate((joint[:6], joint[7:13]))
        gripper = joint[[6, 13]]
        state = torch.zeros(75)
        state[:12] = torch.from_numpy(self.normalize("observation.state.arm.position", arm))
        state[14:16] = torch.from_numpy(self.normalize("observation.state.effector.position", gripper))
        images = []
        for image in raw["images"]:
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
            height, width = pixels.shape[-2:]
            ratio = max(width / 224, height / 224)
            h, w = int(height / ratio), int(width / ratio)
            pixels = functional.interpolate(
                pixels.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            pixels = functional.pad(pixels, (224 - w, 0, 224 - h, 0), value=0)
            images.append(pixels.float() / 255)
        return Observation(
            images=torch.stack(images),
            state=state,
            instruction_tokens=torch.empty(0, dtype=torch.long),
            instruction=sample.instruction,
        )

    def restore(self, actions: torch.Tensor) -> torch.Tensor:
        """Restore physical left/right six-joint plus gripper action columns."""
        actions = actions.detach().float().cpu().numpy()
        groups = []
        for key, indices in (
            ("action.arm.position", slice(0, 12)),
            ("action.effector.position", slice(14, 16)),
        ):
            low = np.asarray(self.statistics[key]["q01"], dtype=np.float32)
            high = np.asarray(self.statistics[key]["q99"], dtype=np.float32)
            groups.append((actions[:, indices] + 1) / 2 * (high - low + 1e-6) + low)
        arm, gripper = groups
        return torch.from_numpy(
            np.concatenate((arm[:, :6], gripper[:, :1], arm[:, 6:], gripper[:, 1:]), axis=-1)
        )


def build(
    config: dict[str, Any], device: torch.device
) -> tuple[RobotwinInference, dict[str, Any], RuntimeStats]:
    """Run the published non-depth Robotwin checkpoint with its native joint workload."""
    processor = RobotwinProcessor(Path(config["norm_stats"]))
    policy = make_policy(
        "lingbot_vla",
        checkpoint=config["checkpoint"],
        backbone_path=config["backbone_path"],
        attention=config["attention"],
    )
    core = make_core(policy, config, device)
    compile_inference(policy, config, ("_vl_forward", "_expert_forward"))

    def infer(sample: RobotwinSample, raw: dict[str, Any]) -> dict[str, Any]:
        batch = policy.collate([processor.prepare(sample, raw)], [sample.sample_id])
        batch = policy.pad(batch, 1).to(device, core.dtype)
        prefix, normalized, timing = timed_model(
            lambda: policy.encode_prefix(batch),
            lambda prefix: policy.decoder.integrate(
                policy.decoder.init_state(1), prefix, config["num_steps"], 1, core._graphs
            ),
            device,
        )
        actions = policy.finalize_actions(normalized, prefix)[0].float().cpu()
        return {**output_record(processor.restore(actions)), "model_timing_ms": timing}

    return (
        infer,
        {
            "policy": "lingbot_vla",
            "checkpoint": config["checkpoint"],
            "checkpoint_repository": config["checkpoint_repository"],
            "checkpoint_revision": config["checkpoint_revision"],
            "output_scope": "physical_joint_action_chunk",
            "action_horizon": 50,
            "output_action_dim": 14,
            "num_steps": config["num_steps"],
            "camera_count": 3,
            "state_layout": "native_left_joint6_gripper_right_joint6_gripper_to_arm12_pad2_gripper2_pad59",
            "image_transform": "bilinear_aspect_resize_black_pad_left_top_224_then_Qwen2.5VL_processor",
            "instruction_selection": "first_seen_instruction_per_episode",
            "normalization_sha256": hashlib.sha256(Path(config["norm_stats"]).read_bytes()).hexdigest(),
        },
        lambda: runtime_details(core, policy),
    )


if __name__ == "__main__":
    run(Path(__file__).with_name("config.yaml"), build)
