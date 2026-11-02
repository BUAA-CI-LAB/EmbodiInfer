"""Replay independent ActiveVLN histories in native vLLM 0.30.0 request batches.

Uses the selected B1 optimization profile and the EmbodiInfer batch slot order.
The complete batch interval includes the slowest request, including STOP checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import cuda_device, digest_json, load_navigation, provenance, read_rgb
from benchmark_vllm import save
from benchmark_vllm_modern import (
    ModernVLLMReplay,
    drain_inflight,
    parse_actions_for,
    stopping_prefix,
)
from PIL import Image

from embodiinfer.policies.activevln.prompt_activevln import (
    actions_to_tensor,
    render_turn_text,
)


@dataclass
class ReplaySession:
    """Own one episode's full private input history across observation batches."""

    identity: str
    episode: Any
    step: int = 0
    history: list[int] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)


def summarize_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Weight amortized cost by actual observations, including partially filled tails."""
    if not batches:
        return {}
    count = sum(batch["observations"] for batch in batches)
    e2e = np.asarray([batch["latency_ms"] for batch in batches])
    forward = np.asarray([batch["model_timing_ms"]["pure_inference_ms"] for batch in batches])
    return {
        "batches": len(batches),
        "observations": count,
        "mean_batch_occupancy": count / len(batches),
        "batch_e2e_ms": {
            "mean": float(e2e.mean()),
            **{f"p{p}": float(np.percentile(e2e, p)) for p in (50, 95, 99)},
        },
        "batch_forward_ms": {
            "mean": float(forward.mean()),
            **{f"p{p}": float(np.percentile(forward, p)) for p in (50, 95, 99)},
        },
        "amortized_e2e_ms_per_observation": float(e2e.sum() / count),
        "amortized_forward_ms_per_observation": float(forward.sum() / count),
        "observations_per_second": float(count * 1000 / e2e.sum()),
    }


def collect_outputs(
    engine: Any,
    tokenizer: Any,
    identities: list[str],
    timer: Any,
    action_space: str = "r2r",
) -> tuple[dict[str, Any], int]:
    """Route unordered cumulative outputs and stop each request independently."""
    records = {
        identity: {"token_ids": [], "native_returned_tokens": 0, "stop_reason": None}
        for identity in identities
    }
    if len(records) != len(identities):
        raise ValueError("request identities must be unique")
    finished: set[str] = set()
    steps = 0
    while engine.has_unfinished_requests():
        outputs = engine.step()
        steps += 1
        for output in outputs:
            identity = output.request_id
            if identity not in records or identity in finished:
                raise RuntimeError(f"unexpected native request output: {identity}")
            if len(output.outputs) != 1:
                raise RuntimeError("greedy replay requires exactly one candidate")
            record = records[identity]
            complete = list(output.outputs[0].token_ids)
            previous = record["token_ids"]
            if complete[: len(previous)] != previous:
                raise RuntimeError("native cumulative output rewrote retained tokens")
            if complete:
                timer.first_token()
            tokens, reason = stopping_prefix(tokenizer, complete, len(previous), action_space)
            record.update(token_ids=tokens, native_returned_tokens=len(complete))
            if reason is not None:
                record["stop_reason"] = reason
                finished.add(identity)
                engine.abort_request([identity])
            elif output.finished:
                record["stop_reason"] = output.outputs[0].finish_reason or "max_tokens"
                finished.add(identity)
    if finished != set(identities) or any(not row["token_ids"] for row in records.values()):
        raise RuntimeError("native engine ended before every request produced a complete output")
    return records, steps


class BatchedVLLMReplay(ModernVLLMReplay):
    """Submit independent episode requests together, retaining native paged KV caches."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(
            config,
            device,
            speculation="ngram_gpu",
            speculative_tokens=16,
            compile_vision=True,
            async_scheduling=False,
        )
        original_execute = self.runner.execute_model

        def observe(scheduled: Any, *args: Any, **kwargs: Any) -> Any:
            for request in scheduled.scheduled_new_reqs:
                self.cached_by_request[request.req_id] = request.num_computed_tokens
            self.scheduled_batch_sizes[len(scheduled.num_scheduled_tokens)] += 1
            self.preemptions += len(scheduled.preempted_req_ids or ())
            return original_execute(scheduled, *args, **kwargs)

        self.runner.execute_model = observe
        original_encoder = self.encoder_manager.execute

        def observe_images(mm_kwargs: dict[str, Any]) -> Any:
            self.vision_items += mm_kwargs["image_grid_thw"].numel() // 3
            return original_encoder(mm_kwargs)

        self.encoder_manager.execute = observe_images

    def audit(self, request: Any, image: Image.Image, turn_text: str) -> dict[str, Any]:
        """Verify actual raw patches, normalized BF16 pixels, grid and delta tokens."""
        expected = self.processor(text=[turn_text], images=[image], return_tensors="pt", padding=False)
        raw = self.processor(
            text=[turn_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            do_rescale=False,
            do_normalize=False,
        )
        feature = request.mm_features[-1].data
        if feature is None:
            raise RuntimeError("current image missing from native request")
        actual = feature.get_data()
        with torch.inference_mode():
            normalized = self.audit_normalizer(actual["pixel_values"].to(self.device), torch.bfloat16).cpu()
        result = {
            "tokens_equal": request.prompt_token_ids[-expected.input_ids.shape[1] :]
            == expected.input_ids[0].tolist(),
            "raw_pixels_equal": torch.equal(actual["pixel_values"], raw.pixel_values),
            "normalized_bf16_equal": torch.equal(normalized, expected.pixel_values.to(torch.bfloat16)),
            "grid_equal": torch.equal(actual["image_grid_thw"].reshape(1, 3), expected.image_grid_thw),
            "grid": expected.image_grid_thw.tolist(),
            "pixel_sha256": hashlib.sha256(expected.pixel_values.numpy().tobytes()).hexdigest(),
        }
        if not all(
            result[key] for key in ("tokens_equal", "raw_pixels_equal", "normalized_bf16_equal", "grid_equal")
        ):
            raise ValueError(f"native input audit failed: {result}")
        return result

    def call_batch(
        self, sessions: list[ReplaySession], images: list[Any], *, audit_inputs: bool = False
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Time decoded RGB through the final CPU actions of all submitted requests."""
        if not sessions or len(sessions) != len(images) or len(sessions) > self.config["batch_size"]:
            raise ValueError("invalid observation batch")
        if len({session.identity for session in sessions}) != len(sessions):
            raise ValueError("episode identities must be unique")
        if self.engine.has_unfinished_requests():
            raise RuntimeError("previous observation batch is still active")
        self.timer.reset()
        self.prefix_cached_tokens = self.forward_tokens = self.scheduled_draft_tokens = 0
        self.verified_draft_tokens = self.accepted_draft_tokens = 0
        self.graph_modes = Counter()
        self.cached_by_request: dict[str, int] = {}
        self.scheduled_batch_sizes: Counter[int] = Counter()
        self.vision_items = self.preemptions = 0
        manager = self.encoder_manager
        hits, misses = manager.graph_hits, manager.graph_misses
        torch.cuda.synchronize(self.device)
        start = time.perf_counter_ns()
        identities, requests, turns = [], [], []
        for session, rgb in zip(sessions, images, strict=True):
            turn = render_turn_text(
                self.processor,
                session.episode.instruction,
                initial=not session.images,
                action_space=self.action_space,
            )
            session.history.extend(self.tokenizer.encode(turn, add_special_tokens=False))
            session.images.append(Image.fromarray(rgb))
            session.image_ids.append(f"{session.identity}-frame-{len(session.images)}")
            identity = f"batch-{self.call_index}-request-{len(identities)}"
            (prompt,) = self.engine.renderer.render_cmpl(
                [
                    {
                        "prompt_token_ids": session.history.copy(),
                        "multi_modal_data": {"image": session.images.copy()},
                        "multi_modal_uuids": {"image": session.image_ids.copy()},
                        "mm_processor_kwargs": {"use_fast": False},
                        "cache_salt": session.identity,
                    }
                ]
            )
            self.engine.add_request(identity, prompt, self.sampling)
            identities.append(identity)
            requests.append(self.last_processed)
            turns.append(turn)
        self.call_index += 1
        added = time.perf_counter_ns()
        records, steps = collect_outputs(
            self.engine, self.tokenizer, identities, self.timer, self.action_space
        )
        drain_steps = drain_inflight(self.engine, self.core)
        timing = self.timer.finish()
        rows = []
        for session, identity, request in zip(sessions, identities, requests, strict=True):
            record = records[identity]
            tokens = record["token_ids"]
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            parsed = parse_actions_for(self.action_space, decoded)
            actions, mask = actions_to_tensor(parsed)
            session.history.extend(tokens)
            rows.append(
                {
                    **record,
                    "episode_id": session.episode.episode_id,
                    "step": session.step,
                    "text": decoded,
                    "actions": actions.tolist(),
                    "action_mask": mask.tolist(),
                    "parsed_valid": parsed.valid,
                    "generated_tokens": len(tokens),
                    "prompt_tokens": len(request.prompt_token_ids),
                    "cache_length": len(request.prompt_token_ids) + len(tokens),
                    "history_images": len(session.images),
                    "prefix_cached_tokens": self.cached_by_request.get(request.request_id, 0),
                }
            )
        torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        batch = {
            "observations": len(sessions),
            "latency_ms": elapsed,
            "model_timing_ms": timing,
            "cpu_preparation_and_admission_ms": (added - start) / 1e6,
            "vision_calls": self.timer.vision_calls,
            "vision_items": self.vision_items,
            "vision_graph_hits": manager.graph_hits - hits,
            "vision_graph_misses": manager.graph_misses - misses,
            "graph_modes": dict(self.graph_modes),
            "scheduled_batch_sizes": dict(self.scheduled_batch_sizes),
            "preemptions": self.preemptions,
            "engine_steps": steps + drain_steps,
            "drain_steps": drain_steps,
            "forward_tokens": self.forward_tokens,
            "scheduled_draft_tokens": self.scheduled_draft_tokens,
            "verified_draft_tokens": self.verified_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
        }
        if (
            batch["vision_graph_misses"]
            or not batch["vision_graph_hits"]
            or self.vision_items != len(sessions)
        ):
            raise RuntimeError(f"vision graph failed to cover exactly the current observations: {batch}")
        for row, session, request, turn in zip(rows, sessions, requests, turns, strict=True):
            if audit_inputs and session.step < 2:
                row["input_audit"] = self.audit(request, session.images[-1], turn)
        return rows, batch


def run(config: dict[str, Any], output: Path, limit: int | None = None) -> int:
    """Measure a fresh native engine, preserving partial results and OOM evidence."""
    batch_size = config["batch_size"]
    if (
        type(batch_size) is not int
        or batch_size not in (1, 2, 4, 8)
        or len(config["datasets"]) != 1
        or config["do_sample"]
    ):
        raise ValueError("require one split, batch 1/2/4 and greedy decoding")
    episodes = load_navigation(config["datasets"][0])
    selected = [
        {
            "episode_id": ep.episode_id,
            "frames": len(ep.frames) if limit is None else min(limit, len(ep.frames)),
        }
        for ep in episodes
    ]
    device = cuda_device(config)
    torch.set_num_threads(config["cpu_threads"])
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "activevln.vllm_batch_replay.v1",
        "status": "running",
        "phase": "loading",
        "reference_config": config,
        "batch_size": batch_size,
        "dataset": config["datasets"][0]["name"],
        "selection": selected,
        "selection_sha256": digest_json(selected),
        "expected_observations": sum(ep["frames"] for ep in selected),
        "rows": [],
        "batches": [],
        "environment": {**provenance(device), "vllm": importlib.metadata.version("vllm")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "benchmark_vllm_batch.py",
                "benchmark_vllm_modern.py",
                "benchmark_vllm.py",
                "benchmark.py",
            )
        },
        "timing_boundary": {
            "e2e": "All decoded CPU RGBs through full-history input processing, complete batched generation and final CPU action tensors; excludes disk decoding, startup, audits and report writing.",
            "forward": "Synchronized wall time from first native vision graph after image H2D through all requests' prefill/decode and token-stop checks; includes scheduler/dispatch gaps, before final CPU action parsing.",
            "prefill_decode": "First generated token of the batch divides the CUDA elapsed interval; mixed prefill/decode may overlap across requests, so these are diagnostic intervals, not isolated phase costs.",
        },
        "history": "Full private history and immutable image UUIDs per episode. Episode cache salt disallows cross-episode prefix reuse; completed cache blocks use native LRU eviction. STOP ends generation, but all saved frames are consumed.",
        "slot_refill": "Preserve active slot order; remove episodes after final saved frame and append the next episode in numeric selection order.",
        "accuracy_scope": "Fixed-frame replay; no closed-loop SR/SPL claim.",
    }
    runtime = None
    measured = False
    try:
        save(output, report)
        start = time.monotonic()
        runtime = BatchedVLLMReplay(config, device)
        report["engine_options"] = runtime.engine_options
        report["phase"] = "warmup"
        save(output, report)
        warmup = next(ep for ep in episodes if len(ep.frames) >= config["warmup_calls"])
        sessions = [ReplaySession(f"warmup-{i}", warmup) for i in range(batch_size)]
        for step in range(config["warmup_calls"]):
            rgb = read_rgb(warmup.frames[step])
            runtime.call_batch(sessions, [rgb] * batch_size)
            for session in sessions:
                session.step += 1
        runtime.reset()
        del sessions, session, rgb
        report["startup_seconds"] = time.monotonic() - start
        report["warmup"] = {
            "batch_calls": config["warmup_calls"],
            "observations": batch_size * config["warmup_calls"],
            "episode_id": warmup.episode_id,
        }
        report["startup_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        torch.cuda.reset_peak_memory_stats(device)
        report["phase"] = "measurement"
        measured = True
        save(output, report)
        pending = iter(episodes)
        sessions = []
        while True:
            while len(sessions) < batch_size:
                ep = next(pending, None)
                if ep is None:
                    break
                sessions.append(ReplaySession(f"measure-{ep.episode_id}", ep))
            if not sessions:
                break
            report["inflight_batch"] = [
                {"episode_id": s.episode.episode_id, "step": s.step} for s in sessions
            ]
            images = [read_rgb(s.episode.frames[s.step]) for s in sessions]
            rows, batch = runtime.call_batch(sessions, images, audit_inputs=True)
            index = len(report["batches"])
            report["batches"].append({"batch_index": index, **batch})
            report["rows"].extend({"batch_index": index, **row} for row in rows)
            for session in sessions:
                session.step += 1
            sessions = [
                s
                for s in sessions
                if s.step < (len(s.episode.frames) if limit is None else min(limit, len(s.episode.frames)))
            ]
            if len(report["batches"]) % 25 == 0:
                report["metrics"] = summarize_batches(report["batches"])
                save(output, report)
                print(
                    json.dumps(
                        {
                            "batches": len(report["batches"]),
                            "observations": len(report["rows"]),
                            "metrics": report["metrics"],
                        }
                    ),
                    flush=True,
                )
        if len(report["rows"]) != report["expected_observations"]:
            raise RuntimeError("incomplete replay coverage")
        report.update(
            status="complete" if limit is None and len(episodes) == 48 else "partial_probe", phase="finished"
        )
        report.pop("inflight_batch", None)
        code = 0
    except Exception as exc:
        report.update(
            status="oom" if isinstance(exc, torch.OutOfMemoryError) else "error",
            exception={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        code = 2
    finally:
        report["metrics"] = summarize_batches(report["batches"])
        report["memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "peak_scope": "measurement" if measured else "startup_until_failure",
        }
        if runtime is not None:
            report["optimization_evidence"] = runtime.optimization_evidence()
        save(output, report)
        print(
            json.dumps(
                {key: report[key] for key in ("status", "phase", "metrics", "exception") if key in report}
            ),
            flush=True,
        )
        if runtime is not None:
            runtime.engine.engine_core.shutdown()
    return code


def main() -> None:
    """Extend an unchanged reference batch config with native vLLM execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps-per-episode", type=int)
    args = parser.parse_args()
    if args.max_steps_per_episode is not None and args.max_steps_per_episode < 1:
        parser.error("max-steps-per-episode must be positive")
    raise SystemExit(run(json.loads(args.config.read_text()), args.output, args.max_steps_per_episode))


if __name__ == "__main__":
    main()
