"""StreamVLN quantization benchmark on recorded R2R and RxR trajectories."""

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
from contextlib import nullcontext
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
from embodiinfer.policies import make_policy
from embodiinfer.types import Observation

ROOT = Path(embodiinfer.__file__).resolve().parent.parent


class ModelTimer:
    """Time the existing model path without copying or replacing its computation.

    Begin at device-ready vision input. End before tokenizer text decoding and
    action parsing; token transfers performed by generation remain in the span.
    """

    def __init__(self, policy: Any, device: torch.device):
        self.device = device
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        self.active = False
        self.metrics: dict[str, float] | None = None
        encode_frames = policy.backbone.encode_frames
        decode = policy.decoder.decode
        decode_text = policy.processor.tokenizer.decode

        def vision(*args, **kwargs):
            if not self.active:
                torch.cuda.synchronize(device)
                self.started = time.perf_counter_ns()
                self.events[0].record(torch.cuda.current_stream(device))
                self.active = True
                self.metrics = None
            return encode_frames(*args, **kwargs)

        def generation(*args, **kwargs):
            if not self.active:
                raise RuntimeError("model timing requires the vision encoding entry")
            self.events[1].record(torch.cuda.current_stream(device))
            return decode(*args, **kwargs)

        def text(*args, **kwargs):
            if self.active:
                self.events[2].record(torch.cuda.current_stream(device))
                self.events[2].synchronize()
                self.metrics = {
                    "prefill_ms": float(self.events[0].elapsed_time(self.events[1])),
                    "decode_ms": float(self.events[1].elapsed_time(self.events[2])),
                    "gpu_inference_ms": float(self.events[0].elapsed_time(self.events[2])),
                    "pure_inference_ms": (time.perf_counter_ns() - self.started) / 1e6,
                }
                self.active = False
            return decode_text(*args, **kwargs)

        policy.backbone.encode_frames = vision
        policy.decoder.decode = generation
        policy.processor.tokenizer.decode = text


def quantization_details(policy: Any, device: torch.device) -> dict[str, Any]:
    """Record configured and resolved backends; explicit backends fail instead of falling back."""
    from embodiinfer.models.linear import QuantizedLinear

    groups: dict[str, int] = {}
    scales: dict[str, int] = {}
    for module in policy.modules():
        if isinstance(module, QuantizedLinear):
            probe = torch.empty((1, module.in_features), device=device, dtype=module.compute_dtype)
            key = f"{type(module).__name__}:{module.backend}:{module._resolved_backend(probe)}"
            groups[key] = groups.get(key, 0) + 1
            scale = module.weight_scale
            key = f"{scale.dtype}:{'tensorwise' if scale.ndim == 0 else 'channel_or_block'}"
            scales[key] = scales.get(key, 0) + 1
    return {"linear_groups": groups, "scale_groups": scales, "quantized_linears": sum(groups.values())}


def positive(value: Any, name: str) -> int:
    """Reject ambiguous, zero, or negative selection sizes."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def digest_json(value: Any) -> str:
    """Fingerprint ordered sample identities or a resolved configuration."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class NavigationEpisode:
    """One saved RGB trajectory with the release's first instruction."""

    dataset: str
    episode_id: int
    video: str
    instruction: str
    frames: tuple[Path, ...]
    actions: tuple[int, ...]


def navigation_annotations(path: Path, count: int) -> list[dict[str, Any]]:
    """Choose the first numeric episode IDs before checking local image availability."""
    positive(count, "episode_limit")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("navigation annotations must be a nonempty list")
    records = sorted(records, key=lambda row: (int(row["id"]), row["video"]))
    ids = [int(row["id"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate navigation episode IDs")
    if count > len(records):
        raise ValueError(f"requested {count} episodes, release has {len(records)}")
    return records[:count]


def load_navigation(config: dict[str, Any]) -> tuple[NavigationEpisode, ...]:
    """Require every frame of the selected episodes; never substitute available episodes."""
    root = Path(config["root"]).expanduser().resolve(strict=True)
    rows = navigation_annotations(Path(config["annotations"]), config["episode_limit"])
    episodes = []
    for row in rows:
        video = Path(row["video"])
        if video.is_absolute() or ".." in video.parts:
            raise ValueError(f"unsafe video path: {video}")
        directory = (root / video / "rgb").resolve(strict=True)
        if not directory.is_relative_to(root):
            raise ValueError("video path escapes dataset root")
        frames = tuple(sorted(directory.glob("*.jpg"), key=lambda p: int(p.stem)))
        if not frames or [int(p.stem) for p in frames] != list(range(1, len(frames) + 1)):
            raise ValueError(f"{video}: expected consecutive JPEG frames starting at 001")
        actions = row["actions"]
        if len(actions) != len(frames) or actions[0] != -1:
            raise ValueError(f"{video}: frames/actions must align with initial dummy action -1")
        if any(type(a) is not int or a not in (0, 1, 2, 3) for a in actions[1:]):
            raise ValueError(f"{video}: invalid navigation action")
        instructions = row["instructions"]
        if (
            not isinstance(instructions, list)
            or not instructions
            or not isinstance(instructions[0], str)
            or not instructions[0].strip()
        ):
            raise ValueError(f"{video}: missing first instruction")
        episodes.append(
            NavigationEpisode(
                config["name"],
                int(row["id"]),
                str(video),
                instructions[0],
                frames,
                tuple(actions[1:]) + (0,),
            )
        )
    return tuple(episodes)


def load_config(*, model_choices: tuple[str, ...] = ()) -> tuple[dict[str, Any], Path]:
    """Read one device/precision configuration, optionally selecting one dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("R2R", "RxR"))
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
    if args.dataset:
        config["datasets"] = [spec for spec in config["datasets"] if spec["name"] == args.dataset]
        if not config["datasets"]:
            parser.error(f"{args.dataset} is not present in {path}")
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
    result["actions"] = output.tolist()
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
            ["git", "-C", str(runtime.parent), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pin = runtime.parent / "SOURCE_REVISION"
        revision = pin.read_text().strip() if pin.is_file() else None
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
        "schema": "inference_quantization_performance_v1",
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
        "runtime": "embodiinfer",
        "model_timing_contract": "device-ready vision input through complete generation before text decoding/action parsing; includes generation-internal token transfers and profiling overhead",
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


def read_rgb(path: Path) -> np.ndarray:
    """Decode the recorded image without model-specific transforms."""
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB")).copy()


def select_warmup_episode(
    datasets: list[tuple[dict[str, Any], tuple[NavigationEpisode, ...]]], warmup_calls: int
) -> NavigationEpisode:
    """Use the first selected trajectory long enough for continuous startup capture."""
    for _, episodes in datasets:
        for episode in episodes:
            if len(episode.frames) >= warmup_calls:
                return episode
    raise ValueError(f"no selected trajectory contains {warmup_calls} frames for continuous warmup")


def main() -> None:
    """Replay observations in order, preserving memory only inside one episode."""
    config, path = load_config()
    if config.get("isolate_datasets", False) and len(config["datasets"]) > 1:
        # Match runs measured with a fresh model and dataset-specific warmup.
        # Start children before this process allocates any CUDA resources.
        for spec in config["datasets"]:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(path),
                "--dataset",
                spec["name"],
            ]
            if config["validate_data_only"]:
                command.append("--validate-data")
            subprocess.run(command, check=True)
        return
    datasets = [(spec, load_navigation(spec)) for spec in config["datasets"]]
    step_limit = config["max_steps_per_episode"]
    if step_limit is not None:
        positive(step_limit, "max_steps_per_episode")
    if config["validate_data_only"]:
        for spec, episodes in datasets:
            for episode in episodes:
                for path in episode.frames[:step_limit]:
                    read_rgb(path)
            print(f"Validated {spec['name']}: {len(episodes)} episodes")
        return
    warmup_episode = select_warmup_episode(datasets, config["warmup_calls"])
    device = cuda_device(config)
    started = time.perf_counter()
    policy = make_policy(
        "streamvln",
        checkpoint=config["checkpoint"],
        load_device=config.get("load_device", str(device)),
        quantization=config.get("quantization"),
        dtype=config["dtype"],
        cuda_graph=config["cuda_graph"],
        max_new_tokens=config["max_new_tokens"],
        max_context=config.get("max_context", 32768),
        decode_block_size=config["decode_block_size"],
        cache_history_features=config["history_feature_cache"],
        fast_action_decode=config["fast_action_decode"],
    )
    policy.to(device).eval()
    if not config.get("language_prefill", True):
        policy.backbone.configure_prefill_optimizations(enabled=True, language_prefill=False)
    timer = ModelTimer(policy, device)
    load_seconds = time.perf_counter() - started
    tokenizer_hashes = {
        name: hashlib.sha256((Path(config["checkpoint"]) / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    memory = None

    def infer(rgb: np.ndarray, instruction: str, sample_id: str) -> dict[str, Any]:
        nonlocal memory
        observation = Observation(
            images=torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255),
            state=torch.empty(0),
            instruction_tokens=torch.empty(0, dtype=torch.long),
            instruction=instruction,
        )
        batch = policy.collate([observation], [sample_id])
        prefix = policy.encode_prefix(batch, memory)
        result = policy.decoder.decode(None, prefix, num_steps=1, bucket=1, graphs=None)
        if timer.active or timer.metrics is None:
            raise RuntimeError("complete model timing was not recorded")
        memory = result.next_memory
        trace = result.traces[0]
        output = output_record(result.actions[0], int(trace.token_ids.numel()))
        return {
            **output,
            "cache_length": int(memory.seq_len),
            "stop_reason": trace.stop_reason,
            "model_timing_ms": dict(timer.metrics),
            "invalid_action_text": trace.meta["invalid_action_text"],
            "response_template": trace.meta["response_template"],
        }

    with torch.inference_mode():
        episode = warmup_episode
        if config["cuda_graph"] and config["warmup_calls"] <= policy.processor.window_size:
            raise ValueError("CUDA graph warmup must cover more than one StreamVLN history window")
        started = time.perf_counter()
        context = policy.startup_cuda_graph_capture() if config["cuda_graph"] else nullcontext()
        with context:
            for step, path in enumerate(episode.frames[: config["warmup_calls"]]):
                infer(read_rgb(path), episode.instruction, f"warmup-{step}")
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - started
        memory = None
        policy.clear_preprocessing_caches()
        for spec, episodes in datasets:
            rows = []
            torch.manual_seed(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            policy.reset_cuda_graph_runtime_stats()
            for repeat in range(config["repeats"]):
                for episode in episodes:
                    memory = None
                    for step, path in enumerate(episode.frames[:step_limit]):
                        raw = read_rgb(path)
                        identity = f"{episode.dataset}/{episode.episode_id}/{path.name}"
                        row = timed_call(partial(infer, raw, episode.instruction, identity), device)
                        rows.append(
                            {
                                "sample_id": identity,
                                "episode_id": episode.episode_id,
                                "step": step,
                                "repeat": repeat,
                                **row,
                            }
                        )
                    print(
                        f"{spec['name']} episode {episode.episode_id}: {len(episode.frames[:step_limit])} calls",
                        flush=True,
                    )
            identities = [row["sample_id"] for row in rows if row["repeat"] == 0]
            write_report(
                config,
                device,
                rows,
                {
                    "model": "streamvln",
                    "tokenizer_sha256": tokenizer_hashes,
                    "dataset": spec,
                    "episodes": [ep.episode_id for ep in episodes],
                    "selection_sha256": digest_json(identities),
                    "model_load_seconds": load_seconds,
                    "warmup_seconds_including_data_io": warmup_seconds,
                    "warmup_episode": {
                        "dataset": warmup_episode.dataset,
                        "episode_id": warmup_episode.episode_id,
                        "calls": config["warmup_calls"],
                    },
                    "memory_protocol": "reset_per_episode;recorded_RGB_order;generated_response_history",
                    "output_scope": "navigation_action_chunk",
                    "graphs": policy.cuda_graph_capture_stats(),
                    "quantization": quantization_details(policy, device),
                    "language_prefill_enabled": policy.backbone._language_prefill_optimizations,
                    "cuda_graph_runtime": policy.cuda_graph_runtime_stats(),
                },
                Path(config["output_dir"]) / f"streamvln-{spec['name'].lower()}.json",
            )


if __name__ == "__main__":
    main()
