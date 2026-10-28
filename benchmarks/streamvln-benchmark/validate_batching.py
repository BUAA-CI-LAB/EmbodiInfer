"""Paired recorded-episode replay through native StreamVLN tensor batches."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from benchmark import cuda_device, load_navigation, provenance, read_rgb, select_warmup_episode

from embodiinfer.policies import make_policy
from embodiinfer.policies.streamvln.processing import make_streamvln_observation


def main() -> None:
    """Measure complete batches and retain per-episode output/state comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--vision-kernel", choices=["hf", "sdpa", "fused"], default="sdpa")
    parser.add_argument("--tree-budget", type=int, default=31)
    parser.add_argument("--tree-refresh", type=int, default=64)
    parser.add_argument("--no-tree", action="store_true")
    parser.add_argument("--padded-prefill", action="store_true")
    parser.add_argument(
        "--kv-pool-tokens", type=int, help="total scratch budget, capped at each batch's full capacity"
    )
    args = parser.parse_args()
    if 1 not in args.batch_sizes or min(args.batch_sizes) < 1:
        parser.error("positive --batch-sizes must include 1 as the paired reference")
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("highest")
    device = cuda_device(config)
    datasets = [(spec, load_navigation(spec)) for spec in config["datasets"]]
    episodes = [(spec["name"], episode) for spec, rows in datasets for episode in rows]
    warmup = select_warmup_episode(datasets, config["warmup_calls"])
    options = dict(
        fast_action_decode=False,
        gpu_preprocess=True,
        gpu_history_features=True,
        fused_ops=True,
        split_attention=True,
        decode_linear=True,
        tree_decode=not args.no_tree,
        tree_budget=args.tree_budget,
        tree_refresh=args.tree_refresh,
        vision_kernel=args.vision_kernel,
    )
    policy = (
        make_policy(
            "streamvln",
            checkpoint=config["checkpoint"],
            load_device=str(device),
            dtype=config["dtype"],
            max_context=config.get("max_context", 32768),
            max_new_tokens=config["max_new_tokens"],
            cuda_graph=True,
            **options,
        )
        .to(device)
        .eval()
    )
    report = dict(
        config=config,
        options=options,
        environment=provenance(device),
        conditions=dict(
            warmup_calls=config["warmup_calls"],
            matmul_precision="highest",
            note="shared-device exploratory replay; no closed-loop success evaluation",
        ),
        batches={},
    )
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            runtime = policy.create_batched_runtime(
                batch_size=batch_size,
                packed_prefill=not args.padded_prefill,
                kv_pool_tokens=(
                    None
                    if args.kv_pool_tokens is None
                    else min(args.kv_pool_tokens, batch_size * policy.backbone.max_context)
                ),
            )
            memories = [None] * batch_size
            with runtime.startup_capture():
                for path in warmup.frames[: config["warmup_calls"]]:
                    observation = make_streamvln_observation(
                        torch.from_numpy(read_rgb(path)).permute(2, 0, 1), warmup.instruction, uint8=True
                    )
                    results = runtime.infer(
                        [observation] * batch_size, [f"warmup/{i}" for i in range(batch_size)], memories
                    )
                    memories = [r.next_memory for r in results]
            runtime.reset_stats()
            torch.cuda.reset_peak_memory_stats(device)
            rows, timings = [], []
            for offset in range(0, len(episodes), batch_size):
                group = episodes[offset : offset + batch_size]
                memories = [None] * len(group)
                paths = [episode.frames[: config.get("max_steps_per_episode")] for _, episode in group]
                for step in range(max(map(len, paths))):
                    active = [i for i, frames in enumerate(paths) if step < len(frames)]
                    raw = [read_rgb(paths[i][step]) for i in active]
                    identities = [
                        f"{group[i][0]}/{group[i][1].episode_id}/{paths[i][step].name}" for i in active
                    ]
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    observations = [
                        make_streamvln_observation(
                            torch.from_numpy(rgb).permute(2, 0, 1), group[i][1].instruction, uint8=True
                        )
                        for i, rgb in zip(active, raw, strict=True)
                    ]
                    results = runtime.infer(observations, identities, [memories[i] for i in active])
                    outputs = []
                    for i, identity, result in zip(active, identities, results, strict=True):
                        memories[i] = result.next_memory
                        outputs.append(
                            dict(
                                sample_id=identity,
                                tokens=result.traces[0].token_ids.cpu().tolist(),
                                actions=result.actions.cpu().tolist(),
                                cache_length=result.next_memory.seq_len,
                            )
                        )
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter() - started) * 1000
                    timings.append(dict(batch_size=len(active), latency_ms=elapsed))
                    rows.extend(outputs)
            report["batches"][str(batch_size)] = dict(
                rows=rows,
                timings=timings,
                stats=runtime.stats(),
                mean_batch_ms=float(np.mean([t["latency_ms"] for t in timings])),
                amortized_ms_per_frame=sum(t["latency_ms"] for t in timings) / len(rows),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            )
            print(
                json.dumps(
                    {
                        "batch_size": batch_size,
                        **{
                            k: v
                            for k, v in report["batches"][str(batch_size)].items()
                            if k not in ("rows", "timings")
                        },
                    }
                ),
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del runtime, memories, results, outputs
            gc.collect()
            torch.cuda.empty_cache()
    reference = {row["sample_id"]: row for row in report["batches"]["1"]["rows"]}
    for batch in report["batches"].values():
        batch["mismatches_vs_batch1"] = [
            dict(
                sample_id=row["sample_id"],
                fields=[
                    key
                    for key in ("tokens", "actions", "cache_length")
                    if row[key] != reference[row["sample_id"]][key]
                ],
            )
            for row in batch["rows"]
            if row != reference[row["sample_id"]]
        ]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(batch["mismatches_vs_batch1"] for batch in report["batches"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
