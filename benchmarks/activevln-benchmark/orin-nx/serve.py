"""Serve the optimized device profiles using the shared batch HTTP adapter."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from serve_batch import TensorBatchAdapter

from embodiinfer.engine.serve.batching import BatchedServingAdapter
from embodiinfer.engine.serve.http_server import PolicyHttpService, create_http_server
from embodiinfer.policies import make_policy


def main() -> None:
    """Prewarm the measured configuration and serve coalesced session requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--max-wait-ms", type=float, default=1000)
    parser.add_argument("--port", type=int, default=18767)
    parser.add_argument("--streamed-loading", action="store_true")
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "frozen_activevln_benchmark", args.source / "benchmarks/activevln-benchmark/benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    config = json.loads(args.config.read_text())
    device = benchmark.cuda_device(config)
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    from streamed_loading import streamed_loading

    if config.get("quantization") or config.get("vision_quantization"):
        raise ValueError("Calibration requires the unchanged BF16 base configuration")
    with streamed_loading() if args.streamed_loading else nullcontext():
        policy = make_policy(
            "activevln",
            checkpoint=config["checkpoint"],
            revision=config["checkpoint_revision"],
            attention=config["attention"],
            max_new_tokens=config["max_new_tokens"],
            max_context=config["max_context"],
            do_sample=config["do_sample"],
            repetition_penalty=config["repetition_penalty"],
            action_space=config.get("action_space", "r2r"),
        )
        policy.to(device=device, dtype=getattr(torch, config["dtype"])).eval()
    quantization = {"layer_count": 0, "layers": []}
    if args.calibration is not None:
        from calibrated_int8 import convert

        quantization = convert(policy, args.calibration, config)
        gc.collect()
        torch.cuda.empty_cache()
    elif config.get("precision_experiment"):
        raise ValueError("INT8 requires --calibration")
    episodes = [ep for entry in config["datasets"] for ep in benchmark.load_navigation(entry)]

    def make_observation(image: np.ndarray, instruction: str) -> Any:
        observation = benchmark.make_observation(image, instruction)
        if config.get("uint8_observations", False):
            observation.metadata["rgb_uint8"] = image
        return observation

    manifest = Path(config["navigation_manifest"])
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected_digest = config["navigation"]["selections"][config["evaluation_dataset"]]["sha256"]
    if manifest_digest != expected_digest:
        raise ValueError("Navigation manifest changed after configuration was frozen")
    navigation = json.loads(manifest.read_text())["episodes"]
    with torch.inference_mode():
        runtime = policy.create_batched_runtime(
            batch_size=config["batch_size"],
            workspace_tokens=config["graph_workspace_tokens"],
            query_bucket_size=config["query_bucket_size"],
            cuda_graph=config["cuda_graph"],
            fused_ops=config["fused_ops"],
            split_attention=config["split_attention"],
            tree_decode=config.get("tree_decode", False),
            tree_fp32_projection=config.get("tree_fp32_projection", False),
            tree_repeat_actions=config.get("tree_repeat_actions", 1),
            kv_pool_tokens=config.get("kv_pool_tokens"),
            grammar_tree_budget=config.get("grammar_tree_budget", 0),
            grammar_prefix_weight=config.get("grammar_prefix_weight", 1.0),
            grammar_local_weight=config.get("grammar_local_weight", 1.0),
            grammar_refresh=config.get("grammar_refresh", 64),
            sampling_seed=config.get("sampling_seed", 0),
            gpu_preprocess=config.get("gpu_preprocess", False),
            attention_kernel=config.get("attention_kernel", "split"),
            incremental_kv_commit=config.get("incremental_kv_commit", False),
            resident_kv=config.get("resident_kv", False),
            vision_kernel=config.get("vision_kernel", "hf"),
            skip_padded_queries=config.get("skip_padded_queries", False),
            packed_prefill=config.get("packed_prefill", False),
            packed_prefill_bucket=config.get("packed_prefill_bucket", 64),
            decode_linear=config.get("decode_linear", False),
        )
        # Navigation sensors always produce 640x480 RGB. Share one dummy image
        # across instructions, avoiding hundreds of full image copies on NX.
        dummy = make_observation(np.zeros((480, 640, 3), dtype=np.uint8), "shape probe")
        instructions = sorted(
            {row["instruction"]["instruction_text"] for row in navigation}
            | {ep.instruction for ep in episodes}
        )
        probes = [replace(dummy, instruction=text) for text in instructions]
        runtime.prewarm(probes, context_buckets=config["prewarm_context_buckets"])
        del probes, dummy
        warmup = next(ep for ep in episodes if len(ep.frames) >= config["warmup_calls"])
        memories = [None] * config["batch_size"]
        for frame in warmup.frames[: config["warmup_calls"]]:
            observation = make_observation(benchmark.read_rgb(frame), warmup.instruction)
            prepared = runtime.prepare([observation] * config["batch_size"], memories)
            generations = runtime.generate(runtime.prefill(prepared))
            memories = [g.memory for g in generations]
            del prepared, generations
        del memories, observation
        runtime.reset_resident_kv()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        runtime.reset_stats()

    adapter_benchmark = SimpleNamespace(make_observation=make_observation, timed_model=benchmark.timed_model)
    adapter = TensorBatchAdapter(
        policy,
        runtime,
        adapter_benchmark,
        device,
        args.ready_file.with_name("tensor-batches.jsonl"),
    )
    scheduler = BatchedServingAdapter(adapter, max_batch=config["batch_size"], max_wait_ms=args.max_wait_ms)
    service = PolicyHttpService(
        scheduler,
        token=os.environ["ACTIVEVLN_BENCHMARK_TOKEN"],
        maximum_sessions=config["batch_size"],
        maximum_images=1,
        maximum_image_bytes=8 * 1024**2,
        max_body_bytes=10 * 1024**2,
    )
    server = create_http_server(service, host="127.0.0.1", port=args.port)
    environment = benchmark.provenance(device)
    snapshot_path = args.source / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else None
    if snapshot is not None:
        for relative, expected in snapshot["files"].items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source changed: {relative}")
    ready = {
        "pid": os.getpid(),
        "port": server.server_address[1],
        "config": config,
        "source": str(args.source),
        "source_snapshot": snapshot,
        "environment": environment,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        "startup_seconds": time.monotonic() - started,
        "navigation_manifest_sha256": manifest_digest,
        "shape_probe_instructions": len(instructions),
        "warmup": {
            "dataset": warmup.dataset,
            "episode_id": warmup.episode_id,
            "calls": config["warmup_calls"],
        },
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "scheduler": scheduler.capabilities(),
        "runtime": runtime.stats(),
        "quantization": quantization,
        "inference_commit": snapshot["base_commit"] if snapshot else environment["source_revision"],
        "loading_adapter_sha256": hashlib.sha256(
            Path(__file__).with_name("streamed_loading.py").read_bytes()
        ).hexdigest(),
    }
    temporary = args.ready_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(ready, indent=2) + "\n")
    temporary.replace(args.ready_file)
    print(f"Ready: port {server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    finally:
        scheduler.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
