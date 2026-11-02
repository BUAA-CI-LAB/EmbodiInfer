"""Paired recorded-input checks for StreamVLN device preprocessing and speculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from benchmark import cuda_device, load_navigation, provenance, read_rgb, select_warmup_episode

from embodiinfer.policies import make_policy
from embodiinfer.policies.streamvln.image_gpu import StreamVLNImageTransform
from embodiinfer.policies.streamvln.processing import make_streamvln_observation


def main() -> None:
    """Replay each profile independently, preserving only its own episode history.

    The reference uses serial tokens and disables the optional response-envelope
    shortcut so cache lengths and token traces can be compared directly. E2E
    includes CPU observation construction through owned CPU output. Per-stage
    timers synchronize the same stages in every profile and add probe overhead.
    This is validation timing, not a replacement for the formal benchmark.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speculative-tokens", type=int, default=4, choices=range(2, 9))
    parser.add_argument("--tree-budget", type=int, default=31)
    parser.add_argument("--tree-refresh", type=int, default=64)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=(
            "reference",
            "reference_repeat",
            "existing_fast",
            "gpu_preprocess",
            "gpu_history",
            "speculative",
            "combined",
            "activevln_serial",
            "activevln_tree",
            "activevln_tree_cublas",
            "activevln_tree_fp32",
            "activevln_tree_text",
            "activevln_tree_sdpa",
            "activevln_tree_sdpa_repeat",
        ),
        help="Select profiles; reference must be included. Defaults to all profiles.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(config["seed"])
    datasets = [(spec, load_navigation(spec)) for spec in config["datasets"]]
    device = cuda_device(config)
    policy = (
        make_policy(
            "streamvln",
            checkpoint=config["checkpoint"],
            load_device=str(device),
            dtype=config["dtype"],
            max_context=config.get("max_context", 32768),
            cuda_graph=config["cuda_graph"],
            max_new_tokens=config["max_new_tokens"],
            decode_block_size=config["decode_block_size"],
            cache_history_features=config["history_feature_cache"],
            fast_action_decode=False,
            tree_refresh=args.tree_refresh,
        )
        .to(device)
        .eval()
    )
    transform = StreamVLNImageTransform(policy.processor.image_processor)
    profiles = {
        "reference": dict(
            gpu_preprocess=False, gpu_history_features=False, speculative_tokens=0, fast_action_decode=False
        ),
        "existing_fast": dict(
            gpu_preprocess=False, gpu_history_features=False, speculative_tokens=0, fast_action_decode=True
        ),
        "gpu_preprocess": dict(
            gpu_preprocess=True, gpu_history_features=False, speculative_tokens=0, fast_action_decode=False
        ),
        "gpu_history": dict(
            gpu_preprocess=False, gpu_history_features=True, speculative_tokens=0, fast_action_decode=False
        ),
        "speculative": dict(
            gpu_preprocess=False,
            gpu_history_features=False,
            speculative_tokens=args.speculative_tokens,
            fast_action_decode=False,
        ),
        "combined": dict(
            gpu_preprocess=True,
            gpu_history_features=True,
            speculative_tokens=args.speculative_tokens,
            fast_action_decode=False,
        ),
    }
    optimized = dict(
        gpu_preprocess=True,
        gpu_history_features=True,
        speculative_tokens=0,
        fast_action_decode=False,
        fused_ops=True,
        split_attention=True,
        attention_kernel="gqa",
        decode_linear=True,
        vision_kernel="fused",
        tree_budget=args.tree_budget,
        tree_refresh=args.tree_refresh,
        uint8_observations=True,
    )
    profiles.update(
        {
            "activevln_serial": dict(optimized, tree_decode=False),
            "activevln_tree": dict(optimized, tree_decode=True),
            "activevln_tree_cublas": dict(optimized, tree_decode=True, decode_linear=False),
            "activevln_tree_fp32": dict(
                optimized, tree_decode=True, decode_linear=False, tree_fp32_projection=True
            ),
            "activevln_tree_text": dict(optimized, tree_decode=True, vision_kernel="hf"),
            "activevln_tree_sdpa": dict(optimized, tree_decode=True, vision_kernel="sdpa"),
        }
    )
    profiles["reference_repeat"] = profiles["reference"].copy()
    profiles["activevln_tree_sdpa_repeat"] = profiles["activevln_tree_sdpa"].copy()
    if args.profiles is not None:
        if "reference" not in args.profiles:
            parser.error("--profiles must include reference")
        profiles = {name: settings for name, settings in profiles.items() if name in args.profiles}

    def configure(name):
        switches = profiles[name]
        policy.processor.clear_runtime_caches()
        policy.processor.gpu_image_transform = transform if switches["gpu_preprocess"] else None
        policy.speculative_tokens = switches["speculative_tokens"]
        policy.fast_action_decode = switches["fast_action_decode"]
        policy.gpu_history_features = switches["gpu_history_features"]
        policy.configure_optimizations(
            **{
                key: switches[key]
                for key in (
                    "tree_decode",
                    "tree_budget",
                    "tree_refresh",
                    "fused_ops",
                    "split_attention",
                    "attention_kernel",
                    "decode_linear",
                    "tree_fp32_projection",
                    "vision_kernel",
                )
                if key in switches
            }
        )

    def infer(rgb, instruction, identity, memory):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        observation = make_streamvln_observation(
            torch.from_numpy(rgb).permute(2, 0, 1),
            instruction,
            uint8=profiles[name].get("uint8_observations", False),
        )
        batch = policy.collate([observation], [identity])
        prepared = policy.prepare_prefix(batch, memory)
        torch.cuda.synchronize(device)
        after_prepare = time.perf_counter()
        prefix = policy.encode_prepared_prefix(prepared)
        torch.cuda.synchronize(device)
        after_prefill = time.perf_counter()
        generation = policy.decoder.generate_tokens(prefix)
        torch.cuda.synchronize(device)
        after_decode = time.perf_counter()
        result = policy.decoder.finalize_generation(prefix, generation)
        trace = result.traces[0]
        row = dict(
            sample_id=identity,
            tokens=trace.token_ids.cpu().tolist(),
            logprobs=trace.token_logprobs.cpu().tolist(),
            actions=result.actions.cpu().tolist(),
            mask=trace.action_mask.cpu().tolist(),
            cache_length=result.next_memory.seq_len,
            effective_cache_length=result.next_memory.seq_len + len(result.next_memory.pending_token_ids),
            stop_reason=trace.stop_reason,
            pending_tokens=list(result.next_memory.pending_token_ids),
        )
        torch.cuda.synchronize(device)
        row["latency_ms"] = (time.perf_counter() - start) * 1000
        row["prepare_ms"] = (after_prepare - start) * 1000
        row["prefill_ms"] = (after_prefill - after_prepare) * 1000
        row["decode_ms"] = (after_decode - after_prefill) * 1000
        return row, result.next_memory

    warmup_calls = config["warmup_calls"]
    if config["cuda_graph"] and warmup_calls <= policy.processor.window_size:
        raise ValueError("graph warmup must cover the history-window transition")
    warmup_episode = select_warmup_episode(datasets, warmup_calls)
    reports = {}
    with torch.inference_mode():
        for name in profiles:
            memory = None
            configure(name)
            context = policy.startup_cuda_graph_capture() if config["cuda_graph"] else nullcontext()
            with context:
                memory = None
                for step, path in enumerate(warmup_episode.frames[:warmup_calls]):
                    _, memory = infer(read_rgb(path), warmup_episode.instruction, f"warmup/{step}", memory)
                print(f"warmup {name}: {warmup_calls}", flush=True)
            torch.manual_seed(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            policy.reset_cuda_graph_runtime_stats()
            rows = []
            for spec, episodes in datasets:
                for episode in episodes:
                    memory = None
                    for path in episode.frames[: config.get("max_steps_per_episode")]:
                        identity = f"{spec['name']}/{episode.episode_id}/{path.name}"
                        row, memory = infer(read_rgb(path), episode.instruction, identity, memory)
                        rows.append(row)
                    print(f"{name} {spec['name']}/{episode.episode_id}: {len(rows)} calls", flush=True)
            reports[name] = dict(
                rows=rows,
                settings=profiles[name],
                mean_ms={
                    key: float(np.mean([row[key] for row in rows]))
                    for key in ("latency_ms", "prepare_ms", "prefill_ms", "decode_ms")
                },
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                graphs=policy.cuda_graph_capture_stats(),
                counters=policy.cuda_graph_runtime_stats(),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".partial.json").write_text(
                json.dumps(dict(config=config, profiles=reports), indent=2) + "\n"
            )
    comparisons = {}
    reference = reports["reference"]["rows"]
    # The existing envelope shortcut defers EOS to the next prefill. Compare
    # the effective logical prefix while retaining raw cache/pending fields.
    keys = ("sample_id", "tokens", "actions", "mask", "effective_cache_length", "stop_reason")
    for name, report in reports.items():
        if name == "reference":
            continue
        mismatches, logprob_max_abs = [], 0.0
        for expected, actual in zip(reference, report["rows"], strict=True):
            changed = [key for key in keys if expected[key] != actual[key]]
            if changed:
                mismatches.append(dict(sample_id=expected["sample_id"], fields=changed))
            if expected["tokens"] == actual["tokens"]:
                logprob_max_abs = max(
                    logprob_max_abs,
                    float(np.max(np.abs(np.asarray(expected["logprobs"]) - np.asarray(actual["logprobs"])))),
                )
        comparisons[name] = dict(
            exact_tokens_actions_state=not mismatches, mismatches=mismatches, logprob_max_abs=logprob_max_abs
        )
    pairwise = {}
    for first, a in reports.items():
        for second, b in reports.items():
            if first >= second:
                continue
            pairwise[f"{first}/{second}"] = dict(
                token_mismatch_frames=sum(
                    x["tokens"] != y["tokens"] for x, y in zip(a["rows"], b["rows"], strict=True)
                ),
                action_mismatch_frames=sum(
                    x["actions"] != y["actions"] for x, y in zip(a["rows"], b["rows"], strict=True)
                ),
                latency_ratio=a["mean_ms"]["latency_ms"] / b["mean_ms"]["latency_ms"],
            )
    output = dict(
        pairwise=pairwise,
        schema="streamvln_optimization_validation_v1",
        config=config,
        environment=provenance(device),
        checkpoint_files_sha256={
            name: hashlib.sha256((Path(config["checkpoint"]) / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            )
            if (Path(config["checkpoint"]) / name).is_file()
        },
        conditions=dict(
            batch_size=1,
            warmup_calls_per_profile=warmup_calls,
            seed=config["seed"],
            matmul_precision="highest",
            note="stage synchronization adds probe overhead; loading, warmup and file IO excluded; device exclusivity not established",
        ),
        profiles=reports,
        comparisons=comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), comparisons=comparisons)), flush=True)
    if any(not comparison["exact_tokens_actions_state"] for comparison in comparisons.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
