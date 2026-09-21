"""pi05 final offline benchmark: sampling, timing and inference in one file."""

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
class LiberoSample:
    """A frame identity in the original LIBERO HDF5 release."""

    path: Path
    demo: str
    frame: int
    instruction: str
    image_convention: str

    @property
    def sample_id(self) -> str:
        """Return a portable identity shared by PI0.5 and Cosmos."""
        return f"libero_10/{self.path.name}/{self.demo}/{self.frame}"


def load_libero(config: dict[str, Any]) -> tuple[LiberoSample, ...]:
    """Select task filenames, numeric demo IDs, then distinct uniform frames."""
    import h5py

    tasks = positive(config["task_limit"], "task_limit")
    demos = positive(config["demos_per_task"], "demos_per_task")
    count = positive(config["frames_per_demo"], "frames_per_demo")
    limit = config.get("sample_limit")
    if limit is not None:
        positive(limit, "sample_limit")
    files = sorted(Path(config["root"]).expanduser().glob("*.hdf5"))
    if len(files) < tasks:
        raise ValueError(f"requested {tasks} task files; only {len(files)} are present")
    samples = []
    for path in files[:tasks]:
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            info = json.loads(data.attrs["problem_info"])
            instruction = info["language_instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path}: missing language instruction")
            convention = data.attrs["macros_image_convention"]
            if isinstance(convention, bytes):
                convention = convention.decode()
            if convention not in ("opengl", "opencv"):
                raise ValueError(f"unsupported stored image convention: {convention}")
            demo_ids = sorted(data.keys(), key=lambda key: int(key.removeprefix("demo_")))
            if len(demo_ids) < demos:
                raise ValueError(f"{path}: fewer than {demos} demos")
            for demo in demo_ids[:demos]:
                obs = data[demo]["obs"]
                length = len(obs["agentview_rgb"])
                for name in ("eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states"):
                    if len(obs[name]) != length:
                        raise ValueError(f"{path}/{demo}: misaligned {name}")
                samples.extend(
                    LiberoSample(path, demo, i, instruction, convention)
                    for i in uniform_indices(length, count)
                )
    if limit is not None and limit > len(samples):
        raise ValueError(f"sample_limit {limit} exceeds selected {len(samples)} samples")
    return tuple(samples if limit is None else samples[:limit])


def read_libero(sample: LiberoSample) -> dict[str, Any]:
    """Read RGB and physical state outside the inference timer, without image transforms."""
    import h5py

    with h5py.File(sample.path, "r") as handle:
        obs = handle[f"data/{sample.demo}/obs"]
        values = {
            key: np.asarray(obs[key][sample.frame])
            for key in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states")
        }
    for key in ("agentview_rgb", "eye_in_hand_rgb"):
        value = values[key]
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError(f"{sample.sample_id}: {key} must be HWC uint8 RGB")
        values[key] = Image.fromarray(value)
    for key, width in (("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)):
        if values[key].shape != (width,) or not np.isfinite(values[key]).all():
            raise ValueError(f"{sample.sample_id}: invalid {key}")
    return values


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
    for name in ("torch", "torchvision", "transformers", "triton", "lerobot", "numpy", "Pillow", "h5py"):
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
    temporary.replace(output)
    print(json.dumps({"report": str(output), "metrics": report["metrics"]}), flush=True)


LiberoInference = Callable[[LiberoSample, dict[str, Any]], dict[str, Any]]


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
    build: Callable[[dict[str, Any], torch.device], tuple[LiberoInference, dict[str, Any], RuntimeStats]],
) -> None:
    """Run one model on the selected public frames, keeping disk IO outside the timer."""
    config, _ = load_config(default)
    samples = load_libero(config["dataset"])
    identities = [sample.sample_id for sample in samples]
    details = {
        "sample_ids": identities,
        "selection_sha256": digest_json(identities),
        "dataset": config["dataset"],
        "selected_samples": len(samples),
    }
    if config["validate_data_only"]:
        for sample in samples:
            read_libero(sample)
        print(f"Validated {len(samples)} LIBERO samples; selection={details['selection_sha256']}")
        return
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
            infer(sample, read_libero(sample))
        torch.cuda.synchronize(device)
        details["warmup_seconds_including_data_io"] = time.perf_counter() - started
        details["warmup_sample_ids"] = [samples[index].sample_id for index in warmup_indices]
        before = runtime_stats()
        details["runtime_before_measurement"] = before
        if config["cuda_graph"] and not before["graphs"]["capture_count"]:
            raise RuntimeError("CUDA Graph was requested but warmup captured no graph")
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(config["seed"])
        for repeat in range(config["repeats"]):
            for index, sample in enumerate(samples):
                raw = read_libero(sample)
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
    write_report(config, device, rows, details, Path(config["output"]))


def validate_tokenizer(checkpoint: str) -> dict[str, Any]:
    """Reject a missing vocabulary before loading weights or measuring inference."""
    from transformers import AutoTokenizer

    preprocessing = json.loads((Path(checkpoint) / "policy_preprocessor.json").read_text())
    names = [
        step["config"]["tokenizer_name"]
        for step in preprocessing["steps"]
        if step.get("registry_name") == "tokenizer_processor"
    ]
    if len(names) != 1:
        raise ValueError("expected one checkpoint tokenizer_processor")
    tokenizer = AutoTokenizer.from_pretrained(names[0], local_files_only=True)
    probes = {
        text: tokenizer.encode(text)
        for text in ("open the drawer", "close the drawer", "pick up the black bowl")
    }
    special_ids = set(tokenizer.all_special_ids)
    if (
        len(tokenizer) < 250_000
        or len({tuple(ids) for ids in probes.values()}) != len(probes)
        or any(not set(ids) - special_ids or tokenizer.unk_token_id in ids for ids in probes.values())
    ):
        raise ValueError(
            "invalid PI0.5 tokenizer vocabulary; Transformers 5 needs a complete "
            "tokenizer.json cache, not only tokenizer.model"
        )
    return {"name": names[0], "vocab_size": len(tokenizer), "probe_token_ids": probes}


def build(
    config: dict[str, Any], device: torch.device
) -> tuple[LiberoInference, dict[str, Any], RuntimeStats]:
    """Use the checkpoint's LeRobot preprocessing and EmbodiInfer execution path."""
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies import make_policy
    from embodiinfer.policies.pi05.processor_pi05 import make_processor

    tokenizer_details = validate_tokenizer(config["checkpoint"])
    policy = make_policy(
        "pi05",
        checkpoint=config["checkpoint"],
        attention="sdpa",
        compile_backend=config["compile_backend"],
        **{
            k: config[k]
            for k in ("native_inference", "prefix_cuda_graph", "denoise_attention", "prefix_attention")
            if k in config
        },
        load_device=str(device),
    )
    processor = make_processor(policy, config["checkpoint"])
    core = EngineCore(
        policy,
        EngineConfig(
            device=str(device),
            dtype=config["dtype"],
            max_batch_size=1,
            batch_buckets=(1,),
            use_cuda_graph=config["cuda_graph"],
            capture_full_loop=config["cuda_graph"],
        ),
    )

    def infer(sample: LiberoSample, raw: dict[str, Any]) -> dict[str, Any]:
        state = processor.prepare_state(
            torch.from_numpy(
                np.concatenate((raw["ee_pos"], raw["ee_ori"], raw["gripper_states"])).astype(np.float32)
            )
        )
        images = {}
        for name, feature in (
            ("agentview_rgb", "observation.images.image"),
            ("eye_in_hand_rgb", "observation.images.image2"),
        ):
            image = raw[name]
            if sample.image_convention == "opengl":
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            images[feature] = torch.from_numpy(processor.resize_image(image, 224, 224))
        batch = processor.prepare(state, images, sample.instruction)
        batch.request_ids = [sample.sample_id]
        batch = policy.pad(batch, 1).to(device, core.dtype)
        prefix, normalized, timing = timed_model(
            lambda: policy.encode_prefix(batch),
            lambda prefix: policy.decoder.integrate(
                policy.decoder.init_state(1), prefix, config["num_steps"], 1, core._graphs
            ),
            device,
        )
        actions = policy.finalize_actions(normalized, prefix)[0].float().cpu()
        return {**output_record(processor.restore_actions(actions, state)), "model_timing_ms": timing}

    def runtime_stats() -> dict[str, Any]:
        from embodiinfer.policies.pi05 import modeling_pi05

        return {
            "graphs": engine_graph_stats(core),
            "compile_backend": policy.compile_backend,
            "native_inference": policy.native_inference,
            "pi05_native": policy._runtime.stats() if policy.native_inference else None,
            "compiled_image_encoders": len(modeling_pi05._COMPILED_IMAGE_ENCODERS),
            "compiled_prefix_encoders": len(modeling_pi05._COMPILED_PREFIX_ENCODERS),
            "compiled_inference_helpers": len(modeling_pi05._COMPILED_INFERENCE_HELPERS),
        }

    return (
        infer,
        {
            "policy": "pi05",
            "output_scope": "physical_action_chunk",
            "checkpoint": config["checkpoint"],
            "tokenizer": tokenizer_details,
            "num_steps": config["num_steps"],
            "action_horizon": policy.config.action_horizon,
            "state_layout": "eef_pos3_axis_angle3_gripper2",
            "image_transform": "stored_opengl_vertical_flip_then_PIL_bilinear_224_then_checkpoint_processor",
        },
        runtime_stats,
    )


if __name__ == "__main__":
    run(Path(__file__).with_name("config.yaml"), build)
