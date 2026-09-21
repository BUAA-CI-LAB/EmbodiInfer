"""qwenvl final offline benchmark: sampling, timing and inference in one file."""

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
from PIL import Image, ImageOps

import embodiinfer
from embodiinfer.policies import make_policy
from embodiinfer.policies.navida.modeling_navida import NaViDAMemory, _parse_navida_actions
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    QwenVLNMemory as LowLevelMemory,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    _parse_low_level_action,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicMemory as PanoramicMemory,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    parse_panoramic_action,
)
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


def timed_generation(
    generate: Callable[[Callable[[], None]], Any], device: torch.device
) -> tuple[Any, dict[str, float]]:
    """Measure all model generation, marking prefill once without an intermediate sync."""
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    marked = False

    def mark_prefill() -> None:
        nonlocal marked
        if marked:
            raise RuntimeError("prefill timing was marked more than once")
        events[1].record(torch.cuda.current_stream(device))
        marked = True

    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    events[0].record(torch.cuda.current_stream(device))
    result = generate(mark_prefill)
    events[2].record(torch.cuda.current_stream(device))
    events[2].synchronize()
    wall_ms = (time.perf_counter_ns() - start) / 1e6
    if not marked:
        raise RuntimeError("generation did not mark the end of prefill")
    return result, {
        "prefill_ms": float(events[0].elapsed_time(events[1])),
        "decode_ms": float(events[1].elapsed_time(events[2])),
        "gpu_inference_ms": float(events[0].elapsed_time(events[2])),
        "pure_inference_ms": wall_ms,
    }


def timed_model(
    prefill: Callable[[], Any], decode: Callable[[Any], Any], device: torch.device
) -> tuple[Any, Any, dict[str, float]]:
    """Time staged tensor inputs through model generation, excluding CPU transforms."""

    def generate(mark_prefill):
        prefix = prefill()
        mark_prefill()
        return prefix, decode(prefix)

    (prefix, result), timing = timed_generation(generate, device)
    return prefix, result, timing


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


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    instruction: str
    observation: Observation
    history_frames: tuple[torch.Tensor, ...]
    history_responses: tuple[str, ...]
    image_paths: dict[str, Any]

    @property
    def memory_source(self) -> str:
        return "manifest" if self.history_frames or self.history_responses else "empty"


def _memory(kind: str, sample: ManifestSample):
    if kind == "low":
        memory_type = LowLevelMemory
    elif kind == "panoramic":
        memory_type = PanoramicMemory
    else:
        memory_type = NaViDAMemory
    return memory_type(
        frames=sample.history_frames,
        responses=sample.history_responses,
    )


def _signature(
    kind: str,
    result: tuple[str, torch.Tensor, list[float]],
    candidate_count: int,
) -> dict[str, Any]:
    text, token_ids, _ = result
    try:
        if kind == "low":
            actions = _parse_low_level_action(text)
        elif kind == "panoramic":
            actions = parse_panoramic_action(text, candidate_count)
        else:
            actions = _parse_navida_actions(text)
    except Exception as exc:
        return {
            "text": text,
            "token_ids": token_ids.detach().long().cpu().tolist(),
            "actions": None,
            "action_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "text": text,
        "token_ids": token_ids.detach().long().cpu().tolist(),
        "actions": actions.tolist(),
        "action_error": None,
    }


PROFILES = {
    "low": "qwen2.5-vl-3b-r2r-low-level",
    "panoramic": "qwen2.5-vl-3b-r2r-panoramic",
    "navida": "navida",
}


ACTION_TEXT = {0: "Stop", 1: "Move", 2: "Left", 3: "Right"}


def read_images(episode: NavigationEpisode, step: int, history: int, kind: str) -> dict[int, Image.Image]:
    """Decode the required published RGB frames outside the inference timer."""
    indices = set(range(step - history, step + 1))
    if kind == "panoramic":
        indices.update((step - 2, step - 1, step + 1, step + 2))
    result = {}
    for index in indices:
        with Image.open(episode.frames[index]) as source:
            result[index] = source.convert("RGB")
    return result


def prepare_sample(
    episode: NavigationEpisode, step: int, history: int, kind: str, images: dict[int, Image.Image]
) -> ManifestSample:
    """Build the recorded-history input; panoramic derivatives are explicitly shape-only."""

    def tensor(index: int, size: tuple[int, int]) -> torch.Tensor:
        image = ImageOps.fit(images[index], size, method=Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)

    size = (960, 240) if kind == "panoramic" else (320, 240)
    frames = tuple(tensor(i, size) for i in range(step - history, step))
    responses = (
        ()
        if kind == "navida"
        else tuple(ACTION_TEXT[episode.actions[i]] for i in range(step - history, step))
    )
    metadata = {}
    if kind == "panoramic":
        metadata = {
            "candidate_images": [tensor(i, (320, 240)) for i in (step - 2, step - 1, step + 1, step + 2)],
            "candidates": [
                {"relative_angle": angle, "distance": distance}
                for angle, distance in zip((-90.0, -30.0, 30.0, 90.0), (0.5, 0.25, 0.25, 0.5))
            ],
            "distance_traveled": 0.0,
        }
    observation = Observation(
        images=tensor(step, size),
        state=torch.empty(0),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction=episode.instruction,
        metadata=metadata,
    )
    return ManifestSample(
        f"{episode.dataset}/{episode.episode_id}/{episode.frames[step].name}",
        episode.instruction,
        observation,
        frames,
        responses,
        {"source": str(episode.frames[step])},
    )


def main() -> None:
    """Measure each model in a fresh process so model residency and RSS stay interpretable."""
    config, config_path = load_config(Path(__file__).with_name("config.yaml"), model_choices=tuple(PROFILES))
    kind = config["selected_model"]
    if kind is None:
        for name in PROFILES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(config_path),
                "--model",
                name,
            ]
            if config["validate_data_only"]:
                command.append("--validate-data")
            subprocess.run(command, check=True)
        return
    history = positive(config["history_frames"], "history_frames")
    count = positive(config["samples_per_episode"], "samples_per_episode")
    datasets = []
    for spec in config["datasets"]:
        episodes = load_navigation(spec)
        samples = []
        for episode in episodes:
            # Use the same admissible frame indices for all three profiles.
            start = max(history, 2)
            available = len(episode.frames) - 2 - start
            steps = [start + i for i in uniform_indices(available, count)]
            samples.extend((episode, step) for step in steps)
        datasets.append((spec, samples))
    if config["validate_data_only"]:
        for spec, samples in datasets:
            for episode, step in samples:
                prepare_sample(episode, step, history, kind, read_images(episode, step, history, kind))
            print(f"Validated {kind}/{spec['name']}: {len(samples)} samples")
        return
    device = cuda_device(config)
    model = config["models"][kind]
    started = time.perf_counter()
    options = {
        "checkpoint": model["checkpoint"],
        "max_new_tokens": model["max_new_tokens"],
        "compile_backend": model["compile_backend"],
        "load_device": str(device),
    }
    if kind != "navida":
        options["attention_backend"] = model["attention_backend"]
        options["compile_text_buckets"] = tuple(model.get("compile_text_buckets", ()))
        options["history_image_cache"] = model.get("history_image_cache", "rgb_bytes")
    policy = make_policy(PROFILES[kind], **options)
    policy.to(device=device, dtype=getattr(torch, config["dtype"])).eval()
    if policy.runner.configure_cuda_graph(config["cuda_graph"]) != config["cuda_graph"]:
        raise RuntimeError("requested CUDA Graph execution was not enabled")
    load_seconds = time.perf_counter() - started

    def infer(episode: NavigationEpisode, step: int, images: dict[int, Image.Image]) -> dict[str, Any]:
        sample = prepare_sample(episode, step, history, kind, images)
        runner = policy.runner
        encoded = runner._encode_batch([sample.observation], [_memory(kind, sample)])
        if kind == "navida":
            if not runner.cuda_graph_enabled:
                raise ValueError("the timed NaViDA profile requires the verified CUDA Graph generation path")
            (tokens, _), timing = timed_generation(
                lambda mark: runner._navida_graph_generate(encoded, prefill_complete=mark), device
            )
        else:

            def prefill():
                if runner.cuda_graph_enabled:
                    return runner._manual_graph_logits(encoded)
                if runner.torch_compile_enabled:
                    return runner._graph_logits(encoded)
                return runner.model(**encoded, use_cache=False).logits[:, -1]

            _, tokens, timing = timed_model(prefill, lambda logits: logits.argmax(-1, keepdim=True), device)
        text = runner.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        result = (text, tokens[0], [])
        signature = _signature(kind, result, len(sample.observation.metadata.get("candidates", ())))
        token_ids = result[1].detach().cpu()
        return {
            "action_slots": len(signature["actions"]) if signature["actions"] is not None else 0,
            "generated_tokens": int(token_ids.numel()),
            "parse_error": signature["action_error"],
            "model_timing_ms": timing,
            "_cpu_output": token_ids,
        }

    with torch.inference_mode():
        for spec, samples in datasets:
            started = time.perf_counter()
            warmup_calls = max(
                config["warmup_calls"], len(samples) if config.get("warmup_all_samples") else 0
            )
            for index in range(warmup_calls):
                episode, step = samples[index % len(samples)]
                infer(episode, step, read_images(episode, step, history, kind))
                if index % 25 == 0:
                    print(f"{kind}/{spec['name']} warmup: {index + 1}/{warmup_calls}", flush=True)
            torch.cuda.synchronize(device)
            warmup_seconds = time.perf_counter() - started
            graphs_before = policy.runner.manual_graph_stats()
            torch.manual_seed(config["seed"])
            random.seed(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            rows = []
            for repeat in range(config["repeats"]):
                for index, (episode, step) in enumerate(samples):
                    images = read_images(episode, step, history, kind)
                    row = timed_call(partial(infer, episode, step, images), device)
                    rows.append(
                        {
                            "sample_id": f"{episode.dataset}/{episode.episode_id}/{episode.frames[step].name}",
                            "episode_id": episode.episode_id,
                            "step": step,
                            "repeat": repeat,
                            **row,
                        }
                    )
                    if index % 10 == 0:
                        print(
                            f"{kind}/{spec['name']}: {index + 1}/{len(samples)}, {row['latency_ms']:.1f} ms",
                            flush=True,
                        )
            identities = [row["sample_id"] for row in rows if row["repeat"] == 0]
            graphs_after = policy.runner.manual_graph_stats()
            if graphs_after["capture_count"] != graphs_before["capture_count"]:
                raise RuntimeError("CUDA Graph capture occurred during measurement; extend warmup")
            if config["cuda_graph"] and graphs_after["replay_count"] <= graphs_before["replay_count"]:
                raise RuntimeError("CUDA Graph was requested but no measured replay occurred")
            if kind != "navida":
                before_compile = graphs_before["torch_compile"]
                after_compile = graphs_after["torch_compile"]
                for key in ("attempts", "failures", "cache_entries"):
                    if before_compile[key] != after_compile[key]:
                        raise RuntimeError("compilation occurred during measurement; extend warmup")
            write_report(
                config,
                device,
                rows,
                {
                    "model": PROFILES[kind],
                    "checkpoint": model,
                    "dataset": spec,
                    "selection_sha256": digest_json(identities),
                    "model_load_seconds": load_seconds,
                    "warmup_seconds_including_data_io": warmup_seconds,
                    "warmup_calls": warmup_calls,
                    "input_profile": "temporal_RGB_shape_compatible_not_official_panorama"
                    if kind == "panoramic"
                    else "recorded_RGB",
                    "history_protocol": f"independent_snapshot_{history}_recorded_frames;expert_response_history_for_Qwen;empty_responses_for_NaViDA",
                    "panoramic_geometry": "fixed_shape_fixture_angles_and_distances_not_measured"
                    if kind == "panoramic"
                    else None,
                    "graphs_before_measurement": graphs_before,
                    "graphs_after_measurement": graphs_after,
                    "output_scope": "generated_navigation_response_and_action_parse",
                    "parse_error_calls": sum(row["parse_error"] is not None for row in rows),
                },
                Path(config["output_dir"]) / f"{kind}-{spec['name'].lower()}.json",
            )


if __name__ == "__main__":
    main()
