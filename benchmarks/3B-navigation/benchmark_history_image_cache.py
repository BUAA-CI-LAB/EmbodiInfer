#!/usr/bin/env python3
"""Benchmark formal recurrent RGB preprocessing on a real-image manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch

from embodiinfer.models.qwen25_vl.history_image_cache import (
    HistoryImageCache,
    HistoryImageCacheKey,
)
from embodiinfer.policies.qwen_r2r_low.contract import (
    LOW_LEVEL_IMAGE_SIZE,
    LOW_LEVEL_SYSTEM_PROMPT_SHA256,
    R2R_PREPROCESSOR_SHA256 as LOW_PROCESSOR_SHA256,
    QwenR2RLowMemory,
)
from embodiinfer.policies.qwen_r2r_low.runner import QwenR2RLowRunner
from embodiinfer.policies.qwen_r2r_panoramic.contract import (
    PANORAMIC_IMAGE_SIZE,
    PANORAMIC_SYSTEM_PROMPT_SHA256,
    R2R_PREPROCESSOR_SHA256 as PANORAMIC_PROCESSOR_SHA256,
    QwenR2RPanoramicMemory,
)
from embodiinfer.policies.qwen_r2r_panoramic.runner import QwenR2RPanoramicRunner
from embodiinfer.types import Observation


Profile = Literal["low", "panoramic"]
Mode = Literal["none", "rgb_bytes"]


class _IdentityGraphRuntime:
    @staticmethod
    def bucket_encoded(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return encoded


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    total_seconds = sum(values) / 1000
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "steps_per_second": (
            len(values) / total_seconds if total_seconds else 0.0
        ),
    }


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    steps = value.get("steps") if isinstance(value, dict) else value
    if not isinstance(steps, list) or not steps:
        raise ValueError("manifest must contain a non-empty steps list")
    if not all(isinstance(step, dict) for step in steps):
        raise ValueError("every manifest step must be an object")
    return steps


def _resolve(path: str, root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255)


def _load_visuals(
    steps: list[dict[str, Any]],
    profile: Profile,
    root: Path,
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        image_field = "image" if profile == "low" else "panorama"
        if not isinstance(step.get(image_field), str):
            raise ValueError(f"step {index} is missing {image_field}")
        candidates: list[dict[str, Any]] = []
        if profile == "panoramic":
            values = step.get("candidates")
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"panoramic step {index} needs non-empty candidates"
                )
            for candidate_index, value in enumerate(values):
                if not isinstance(value, dict) or not isinstance(
                    value.get("image"), str
                ):
                    raise ValueError(
                        f"step {index} candidate {candidate_index} needs image"
                    )
                angle = float(value.get("relative_angle"))
                distance = float(value.get("distance"))
                if not np.isfinite(angle) or not np.isfinite(distance):
                    raise ValueError("candidate geometry must be finite")
                candidates.append(
                    {
                        "image": _image_tensor(
                            _resolve(value["image"], root)
                        ),
                        "relative_angle": angle,
                        "distance": distance,
                    }
                )
        loaded.append(
            {
                "current": _image_tensor(_resolve(step[image_field], root)),
                "candidates": tuple(candidates),
                "instruction": str(step.get("instruction", "")),
                "response": str(step.get("response", "")),
                "distance_traveled": float(
                    step.get("distance_traveled", 0.0)
                ),
                "move_possible": bool(step.get("move_possible", True)),
            }
        )
    return loaded


def _contract_values(profile: Profile) -> tuple[str, str]:
    if profile == "low":
        return LOW_PROCESSOR_SHA256, LOW_LEVEL_SYSTEM_PROMPT_SHA256
    return PANORAMIC_PROCESSOR_SHA256, PANORAMIC_SYSTEM_PROMPT_SHA256


def _load_verified_processor(
    checkpoint: Path,
    profile: Profile,
):
    from transformers import (
        AutoTokenizer,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
        Qwen2_5_VLProcessor,
    )

    expected_processor_hash, expected_prompt_hash = _contract_values(profile)
    processor_path = checkpoint / "preprocessor_config.upstream.json"
    if not processor_path.is_file():
        processor_path = checkpoint / "preprocessor_config.json"
    processor_bytes = processor_path.read_bytes()
    actual_processor_hash = _sha256(processor_bytes)
    if actual_processor_hash != expected_processor_hash:
        raise ValueError(
            f"unexpected official processor config: {processor_path}"
        )

    prompt_path = checkpoint / "system_prompt.txt"
    prompt_bytes = prompt_path.read_bytes()
    actual_prompt_hash = _sha256(prompt_bytes)
    if actual_prompt_hash != expected_prompt_hash:
        raise ValueError(f"unexpected official system prompt: {prompt_path}")

    processor_config = json.loads(processor_bytes)
    processor_config.pop("image_processor_type", None)
    processor_config.pop("processor_class", None)
    processor_config["size"] = {
        "shortest_edge": int(processor_config["min_pixels"]),
        "longest_edge": int(processor_config["max_pixels"]),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    chat_template_path = checkpoint / "chat_template.json"
    if chat_template_path.is_file():
        chat_template = json.loads(
            chat_template_path.read_text(encoding="utf-8")
        )["chat_template"]
    else:
        chat_template = tokenizer.chat_template
    if not chat_template:
        raise FileNotFoundError(f"chat template is missing from {checkpoint}")
    processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(**processor_config),
        video_processor=Qwen2VLVideoProcessor(**processor_config),
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    processor.tokenizer.padding_side = "left"
    return (
        processor,
        prompt_bytes.decode("utf-8"),
        {
            "processor_path": str(processor_path),
            "processor_sha256": actual_processor_hash,
            "system_prompt_path": str(prompt_path),
            "system_prompt_sha256": actual_prompt_hash,
            "chat_template_source": str(chat_template_path)
            if chat_template_path.is_file()
            else "tokenizer_config",
        },
    )


def _cache_key(profile: Profile) -> HistoryImageCacheKey:
    if profile == "low":
        return HistoryImageCacheKey(
            profile="low_level",
            kind="low",
            size=LOW_LEVEL_IMAGE_SIZE,
            resize=True,
            processor_sha256=LOW_PROCESSOR_SHA256,
        )
    return HistoryImageCacheKey(
        profile="panoramic",
        kind="panorama",
        size=PANORAMIC_IMAGE_SIZE,
        resize=False,
        processor_sha256=PANORAMIC_PROCESSOR_SHA256,
    )


def _make_runner(
    *,
    profile: Profile,
    mode: Mode,
    processor,
    system_prompt: str,
):
    runner_type = (
        QwenR2RLowRunner if profile == "low" else QwenR2RPanoramicRunner
    )
    runner = object.__new__(runner_type)
    runner.profile = "low_level" if profile == "low" else "panoramic"
    runner.processor = processor
    runner.system_prompt = system_prompt
    runner.history_image_cache_mode = mode
    runner.history_image_cache_enabled = mode == "rgb_bytes"
    runner.history_image_cache_key = _cache_key(profile)
    runner.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=0)
    )
    if profile == "low":
        runner.graph_runtime = _IdentityGraphRuntime()
    else:
        runner.bucket_encoded = lambda encoded: encoded
    return runner


def _observation(profile: Profile, step: dict[str, Any]) -> Observation:
    metadata = {
        "distance_traveled": step["distance_traveled"],
        "move_possible": step["move_possible"],
    }
    if profile == "panoramic":
        metadata.update(
            {
                "candidate_images": [
                    candidate["image"] for candidate in step["candidates"]
                ],
                "candidates": [
                    {
                        "relative_angle": candidate["relative_angle"],
                        "distance": candidate["distance"],
                    }
                    for candidate in step["candidates"]
                ],
            }
        )
    return Observation(
        images=step["current"].unsqueeze(0),
        state=torch.empty(0),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction=step["instruction"],
        metadata=metadata,
    )


def _tensor_signature(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": _sha256(value.view(torch.uint8).numpy().tobytes()),
    }


def _encoded_signature(
    encoded: dict[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    return {
        name: _tensor_signature(encoded[name])
        for name in sorted(encoded)
    }


def _run_session(
    *,
    profile: Profile,
    mode: Mode,
    steps: list[dict[str, Any]],
    processor,
    system_prompt: str,
    timed: bool,
) -> dict[str, object]:
    runner = _make_runner(
        profile=profile,
        mode=mode,
        processor=processor,
        system_prompt=system_prompt,
    )
    memory_type = (
        QwenR2RLowMemory
        if profile == "low"
        else QwenR2RPanoramicMemory
    )
    cache = (
        HistoryImageCache.enabled()
        if mode == "rgb_bytes"
        else HistoryImageCache()
    )
    frames: tuple[torch.Tensor, ...] = ()
    responses: tuple[str, ...] = ()
    latencies: list[float] = []
    signatures: list[dict[str, dict[str, object]]] = []
    counters = {
        "current_renders": 0,
        "history_renders": 0,
        "candidate_renders": 0,
        "hits": 0,
        "misses": 0,
        "appends": 0,
        "evictions": 0,
    }
    render_calls = {"low": 0, "panorama": 0, "candidate": 0}
    original_render = runner._render_image

    def counted_render(frame: torch.Tensor, kind: str):
        render_calls[kind] += 1
        return original_render(frame, kind)

    runner._render_image = counted_render

    for step in steps:
        observation = _observation(profile, step)
        memory = memory_type(
            frames=frames,
            responses=responses,
            history_image_cache=cache,
        )
        hits = 0
        misses = 0
        history_renders = 0
        if mode == "rgb_bytes":
            for frame_index in range(len(frames)):
                if cache.get(frame_index, runner.history_image_cache_key) is None:
                    misses += 1
                    history_renders += 1
                else:
                    hits += 1
        else:
            history_renders = len(frames)

        before_calls = dict(render_calls)
        started = time.perf_counter_ns() if timed else 0
        if mode == "rgb_bytes":
            encoded, entries = runner._prepare_batch_with_history_entries(
                [observation], [memory]
            )
            current_entry = entries[0]
            if current_entry is None:
                raise RuntimeError("rgb_bytes preparation omitted current entry")
            before_entries = len(cache.entries)
            cache = runner.commit_history_image_cache(
                memory, current_entry
            )
            counters["appends"] += 1
            counters["evictions"] += max(
                0, before_entries + 1 - len(cache.entries)
            )
        else:
            encoded = runner._prepare_batch([observation], [memory])
            current_entry = None
        if timed:
            latencies.append(
                (time.perf_counter_ns() - started) / 1_000_000
            )

        current_kind = "low" if profile == "low" else "panorama"
        current_kind_calls = (
            render_calls[current_kind] - before_calls[current_kind]
        )
        expected_current_kind_calls = history_renders + 1
        if current_kind_calls != expected_current_kind_calls:
            raise RuntimeError(
                "formal runner render count disagrees with cache accounting"
            )
        candidate_count = (
            len(step["candidates"]) if profile == "panoramic" else 0
        )
        candidate_calls = (
            render_calls["candidate"] - before_calls["candidate"]
        )
        if candidate_calls != candidate_count:
            raise RuntimeError(
                "panoramic candidates must render exactly once per step"
            )

        signatures.append(_encoded_signature(dict(encoded)))
        counters["hits"] += hits
        counters["misses"] += misses
        counters["history_renders"] += history_renders
        counters["current_renders"] += 1
        counters["candidate_renders"] += candidate_count
        frames = (*frames, step["current"].detach().cpu().clone())
        responses = (*responses, step["response"])
        memory_type(
            frames=frames,
            responses=responses,
            history_image_cache=cache,
        )

    counters.update(
        {
            "resident_bytes": cache.bytes_used,
            "resident_entries": len(cache.entries),
            "base_frame_index": cache.base_frame_index,
        }
    )
    result: dict[str, object] = {
        "mode": mode,
        "steps": len(steps),
        "counters": counters,
        "processor_tensors": signatures,
    }
    if timed:
        result["step_latency_ms"] = latencies
        result["summary"] = _summary(latencies)
    return result


def _pooled_mode(sessions: list[dict[str, object]]) -> dict[str, object]:
    latencies = [
        latency
        for session in sessions
        for latency in session["step_latency_ms"]
    ]
    totals: dict[str, int] = {}
    for session in sessions:
        for name, value in session["counters"].items():
            totals[name] = totals.get(name, 0) + int(value)
    return {
        **_summary(latencies),
        "steps": len(latencies),
        "sessions": len(sessions),
        "counters_total": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("low", "panoramic"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=True)
    steps = _load_visuals(
        _load_manifest(manifest), args.profile, manifest.parent
    )
    processor, system_prompt, checkpoint_contract = (
        _load_verified_processor(checkpoint, args.profile)
    )

    warmups = {
        mode: _run_session(
            profile=args.profile,
            mode=mode,
            steps=steps,
            processor=processor,
            system_prompt=system_prompt,
            timed=False,
        )
        for mode in ("none", "rgb_bytes")
    }
    reference = warmups["none"]["processor_tensors"]
    parity = warmups["rgb_bytes"]["processor_tensors"] == reference

    execution_order: list[dict[str, object]] = []
    sessions: list[dict[str, object]] = []
    per_mode: dict[str, list[dict[str, object]]] = {
        "none": [],
        "rgb_bytes": [],
    }
    for iteration in range(args.iterations):
        order = (
            ("none", "rgb_bytes", "rgb_bytes", "none")
            if iteration % 2 == 0
            else ("rgb_bytes", "none", "none", "rgb_bytes")
        )
        for slot, mode in enumerate(order):
            session = _run_session(
                profile=args.profile,
                mode=mode,
                steps=steps,
                processor=processor,
                system_prompt=system_prompt,
                timed=True,
            )
            session["iteration"] = iteration
            session["slot"] = slot
            session["order_index"] = len(sessions)
            execution_order.append(
                {
                    "order_index": len(sessions),
                    "iteration": iteration,
                    "slot": slot,
                    "mode": mode,
                }
            )
            if session["processor_tensors"] != reference:
                parity = False
            sessions.append(session)
            per_mode[mode].append(session)

    result = {
        "status": "pass" if parity else "parity_failure",
        "scope": "formal_runner_real_image_continuous_preprocessing_not_model_e2e",
        "visual_source": "real_images_from_manifest",
        "runner_construction": "object_new_without_model_load",
        "profile": args.profile,
        "checkpoint": str(checkpoint),
        "checkpoint_contract": checkpoint_contract,
        "manifest": str(manifest),
        "steps_per_session": len(steps),
        "abba_iterations": args.iterations,
        "untimed_warmup": {
            mode: {
                "steps": warmups[mode]["steps"],
                "counters": warmups[mode]["counters"],
            }
            for mode in ("none", "rgb_bytes")
        },
        "execution_order": execution_order,
        "sessions": sessions,
        "pooled_step": {
            mode: _pooled_mode(per_mode[mode])
            for mode in ("none", "rgb_bytes")
        },
        "parity": {
            "all_processor_tensor_dtype_shape_hash_exact": parity,
            "reference_processor_tensors": reference,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "timing": {
            "included": [
                "formal prompt and image interleave",
                "history/current/candidate PIL preparation",
                "history cache lookup and RGB reconstruction",
                "official Qwen processor",
                "successful current entry commit",
            ],
            "excluded": [
                "per-mode warmup",
                "checkpoint processor/tokenizer load",
                "manifest parse and image decode",
                "model load and inference",
            ],
            "schedule": "ABBA on even iterations, BAAB on odd iterations",
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if parity else 3


if __name__ == "__main__":
    raise SystemExit(main())
