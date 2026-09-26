"""Recorded ActiveVLN tensor-batch throughput and memory, with complete OOM evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from benchmark import (
    NavigationEpisode,
    cuda_device,
    digest_json,
    load_navigation,
    make_observation,
    provenance,
    read_rgb,
    timed_model,
)

from embodiinfer.policies import make_policy
from embodiinfer.policies.activevln.cache_activevln import ActiveVLNMemory


@dataclass
class ReplaySlot:
    """An independently growing episode history, never shared with another slot."""

    episode: NavigationEpisode
    step: int = 0
    memory: ActiveVLNMemory | None = None


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically retain completed calls, including when a later batch fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate batch latency from amortized observation cost and throughput."""
    if not batches:
        return {}
    latencies = np.asarray([row["latency_ms"] for row in batches])
    forward = np.asarray([row["model_timing_ms"]["pure_inference_ms"] for row in batches])
    count = sum(row["observations"] for row in batches)
    return {
        "batches": len(batches),
        "observations": count,
        "mean_batch_occupancy": count / len(batches),
        "batch_e2e_ms": {
            "mean": float(latencies.mean()),
            **{f"p{x}": float(np.percentile(latencies, x)) for x in (50, 95, 99)},
        },
        "batch_forward_ms": {
            "mean": float(forward.mean()),
            **{f"p{x}": float(np.percentile(forward, x)) for x in (50, 95, 99)},
        },
        "batch_prefill_ms": {
            "mean": float(np.mean([row["model_timing_ms"]["prefill_ms"] for row in batches])),
        },
        "batch_decode_ms": {
            "mean": float(np.mean([row["model_timing_ms"]["decode_ms"] for row in batches])),
        },
        "amortized_e2e_ms_per_observation": float(latencies.sum() / count),
        "amortized_forward_ms_per_observation": float(forward.sum() / count),
        "observations_per_second": float(count * 1000 / latencies.sum()),
    }


def run(config: dict[str, Any], output: Path) -> int:
    """Measure one split in one fresh process, retaining failure stage and batch identities."""
    if len(config["datasets"]) != 1:
        raise ValueError("batch memory measurement requires exactly one split per fresh process")
    batch_size = config["batch_size"]
    if type(batch_size) is not int or batch_size not in (1, 2, 4, 8):
        raise ValueError("batch_size must be 1, 2, 4 or 8 (8 is a local lane extension)")
    if config.get("do_sample", False):
        raise ValueError("tensor batching currently requires greedy decoding")
    spec = config["datasets"][0]
    episodes = load_navigation(spec)
    limit = config.get("max_steps_per_episode")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_steps_per_episode must be a positive integer or null")
    selected = [
        {
            "episode_id": ep.episode_id,
            "frames": len(ep.frames) if limit is None else min(limit, len(ep.frames)),
        }
        for ep in episodes
    ]
    expected_frames = sum(ep["frames"] for ep in selected)
    report: dict[str, Any] = {
        "schema": "activevln.tensor_batch_benchmark.v2",
        "status": "running",
        "phase": "loading",
        "config": config,
        "dataset": spec["name"],
        "batch_size": batch_size,
        "selection": selected,
        "selection_sha256": digest_json(selected),
        "expected_observations": expected_frames,
        "rows": [],
        "batches": [],
        "timing_boundary": "Decoded CPU RGB to CPU action chunks for the entire batch; model interval covers packed vision, text prefill and all batched decode steps plus private KV snapshots. Disk decoding, startup/capture, and report hashing are excluded.",
        "accuracy_scope": "Fixed replay only, not navigation SR/SPL. BF16 batch/padding shapes can change outputs. Prior B=1 navigation scores do not certify this batched implementation.",
        "batching_mode": "padded_tensor_batch",
        "slot_refill": "Preserve episode order; remove finished slots, refill from numeric episode order; retain generated history until the final recorded frame, including after predicted STOP.",
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    device = cuda_device(config)
    torch.set_num_threads(config["cpu_threads"])
    report["environment"] = provenance(device)
    runtime = None
    measured = False
    try:
        start = time.perf_counter()
        policy = make_policy(
            "activevln",
            checkpoint=config["checkpoint"],
            revision=config["checkpoint_revision"],
            attention=config["attention"],
            max_new_tokens=config["max_new_tokens"],
            max_context=config["max_context"],
            do_sample=config.get("do_sample", False),
            repetition_penalty=config["repetition_penalty"],
            action_space=config.get("action_space", "r2r"),
        )
        policy.to(device=device, dtype=getattr(torch, config["dtype"])).eval()
        report["model_load_seconds"] = time.perf_counter() - start
        report["phase"] = "workspace"
        save_report(output, report)
        with torch.inference_mode():
            runtime = policy.create_batched_runtime(
                batch_size=batch_size,
                workspace_tokens=config["graph_workspace_tokens"],
                query_bucket_size=config["query_bucket_size"],
                cuda_graph=config["cuda_graph"],
                fused_ops=config["fused_ops"],
                split_attention=config["split_attention"],
                tree_decode=config.get("tree_decode", False),
                tree_fp32_projection=config.get("tree_fp32_projection", False),
                tree_repeat_actions=config.get("tree_repeat_actions", 1),
                kv_pool_tokens=config.get("kv_pool_tokens"),
            )
            report["phase"] = "graph_capture"
            save_report(output, report)
            start = time.perf_counter()
            probes = [make_observation(read_rgb(ep.frames[0]), ep.instruction) for ep in episodes]
            runtime.prewarm(probes, context_buckets=config["prewarm_context_buckets"])
            del probes
            report["graph_capture_seconds"] = time.perf_counter() - start
            report["phase"] = "warmup"
            save_report(output, report)
            warmup_calls = config["warmup_calls"]
            warmup = next(ep for ep in episodes if len(ep.frames) >= warmup_calls)
            memories = [None] * batch_size
            start = time.perf_counter()
            for step in range(warmup_calls):
                observation = make_observation(read_rgb(warmup.frames[step]), warmup.instruction)
                prepared = runtime.prepare([observation] * batch_size, memories)
                generations = runtime.generate(runtime.prefill(prepared))
                memories = [generation.memory for generation in generations]
                del prepared, generations
            del memories, observation
            gc.collect()
            torch.cuda.synchronize(device)
            report["warmup_seconds"] = time.perf_counter() - start
            report["warmup"] = {
                "batch_calls": warmup_calls,
                "observations": warmup_calls * batch_size,
                "episode_id": warmup.episode_id,
            }
            report["startup_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            torch.cuda.reset_peak_memory_stats(device)
            runtime.reset_stats()
            report["phase"] = "measurement"
            measured = True
            pending = iter(episodes)
            slots: list[ReplaySlot] = []
            while True:
                while len(slots) < batch_size:
                    episode = next(pending, None)
                    if episode is None:
                        break
                    slots.append(ReplaySlot(episode))
                if not slots:
                    break
                report["inflight_batch"] = [
                    {
                        "episode_id": slot.episode.episode_id,
                        "step": slot.step,
                        "cache_length": 0 if slot.memory is None else slot.memory.seq_len,
                    }
                    for slot in slots
                ]
                images = [read_rgb(slot.episode.frames[slot.step]) for slot in slots]
                torch.cuda.synchronize(device)
                start_ns = time.perf_counter_ns()
                observations = [
                    make_observation(image, slot.episode.instruction)
                    for image, slot in zip(images, slots, strict=True)
                ]
                prepared = runtime.prepare(observations, [slot.memory for slot in slots])
                prefix, generations, timing = timed_model(
                    lambda prepared=prepared: runtime.prefill(prepared), runtime.generate, device
                )
                results = [policy.decoder.finalize_generation(generation) for generation in generations]
                actions = [result.actions[0].detach().float().cpu() for result in results]
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                batch_index = len(report["batches"])
                report["batches"].append(
                    {
                        "batch_index": batch_index,
                        "observations": len(slots),
                        "latency_ms": elapsed_ms,
                        "model_timing_ms": timing,
                    }
                )
                for slot, result, output_actions in zip(slots, results, actions, strict=True):
                    trace = result.traces[0]
                    if not torch.isfinite(output_actions).all():
                        raise ValueError("nonfinite CPU actions")
                    slot.memory = result.next_memory
                    report["rows"].append(
                        {
                            "batch_index": batch_index,
                            "episode_id": slot.episode.episode_id,
                            "step": slot.step,
                            "token_ids": trace.token_ids.cpu().tolist(),
                            "text": trace.text,
                            "stop_reason": trace.stop_reason,
                            "action_mask": trace.meta["parsed_action_mask"].tolist(),
                            "actions": output_actions.tolist(),
                            "output_sha256": hashlib.sha256(output_actions.numpy().tobytes()).hexdigest(),
                            "parsed_valid": trace.parsed_actions.valid,
                            "cache_length": slot.memory.seq_len,
                        }
                    )
                    slot.step += 1
                slots = [
                    slot
                    for slot in slots
                    if slot.step
                    < (len(slot.episode.frames) if limit is None else min(limit, len(slot.episode.frames)))
                ]
                del prepared, prefix, generations, results, actions, observations, images, result, slot
                if len(report["batches"]) % 25 == 0:
                    report["metrics"] = summarize_batches(report["batches"])
                    save_report(output, report)
                    print(
                        json.dumps(
                            {
                                "batches": len(report["batches"]),
                                "observations": len(report["rows"]),
                                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                            }
                        ),
                        flush=True,
                    )
        assert len(report["rows"]) == expected_frames
        report["status"] = "complete" if limit is None and len(episodes) == 48 else "partial_probe"
        report["phase"] = "finished"
        report.pop("inflight_batch", None)
        return_code = 0
    except Exception as exc:
        report["status"] = "oom" if isinstance(exc, torch.OutOfMemoryError) else "error"
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 2
    finally:
        report["metrics"] = summarize_batches(report["batches"])
        report["measurement_started"] = measured
        report["memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "peak_scope": "measurement" if measured else "startup_until_failure",
        }
        if runtime is not None:
            report["runtime"] = runtime.stats()
        save_report(output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "phase": report["phase"],
                    "metrics": report["metrics"],
                    "memory": report["memory"],
                }
            ),
            flush=True,
        )
    return return_code


def main() -> None:
    """Read an explicit batch configuration and keep its raw report outside Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(yaml.safe_load(args.config.read_text()), args.output))


if __name__ == "__main__":
    main()
