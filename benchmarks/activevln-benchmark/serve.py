"""Serve a frozen replay configuration over the existing versioned policy HTTP API.

This benchmark-only adapter keeps model calls and timing identical to the replay.
Episode execution and navigation metrics stay in the downstream simulator runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from embodiinfer.engine.serve.contracts import ModelAction, ModelResult
from embodiinfer.engine.serve.http_server import PolicyHttpService, create_http_server
from embodiinfer.policies import make_policy
from embodiinfer.types import Observation


class ReplayAdapter:
    """Serialize benchmark inference with private memory committed after success."""

    action_space = "activevln.r2r.discrete.v1"

    def __init__(self, policy: Any, benchmark: Any, device: torch.device) -> None:
        self.policy, self.benchmark, self.device = policy, benchmark, device
        self.memories: dict[str, Any] = {}
        self.lock = threading.RLock()
        self.last_output: torch.Tensor | None = None

    def capabilities(self) -> dict[str, Any]:
        """Expose the same decoded action grammar as the offline benchmark."""
        return {
            "model": "activevln",
            "action_space": self.action_space,
            "recurrent": True,
            "max_batch_size": 1,
        }

    @staticmethod
    def observation(rgb: np.ndarray, instruction: str) -> Observation:
        """Build the unchanged decoded-CPU observation at the timing boundary."""
        return Observation(
            images=torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255),
            state=torch.empty(0),
            instruction_tokens=torch.empty(0, dtype=torch.long),
            instruction=instruction,
        )

    def call(self, rgb: np.ndarray, instruction: str, session: str, identity: str) -> dict[str, Any]:
        """Run the original preprocess/forward/postprocess sequence with scoped memory."""
        batch = self.policy.collate([self.observation(rgb, instruction)], [identity])
        prepared = self.policy.prepare_prefix(batch, self.memories.get(session))
        _, generation, timing = self.benchmark.timed_model(
            lambda: self.policy.encode_prepared_prefix(prepared),
            self.policy.decoder.generate_tokens,
            self.device,
        )
        result = self.policy.decoder.finalize_generation(generation)
        trace = result.traces[0]
        output = self.benchmark.output_record(result.actions[0], int(trace.token_ids.numel()))
        self.last_output = output["_cpu_output"]
        self.memories[session] = result.next_memory
        return {
            **output,
            "cache_length": int(result.next_memory.seq_len),
            "stop_reason": trace.stop_reason,
            "model_timing_ms": timing,
            "_trace": trace,
        }

    def infer(self, request: Any) -> ModelResult:
        """Decode transport bytes before timing; preserve full output diagnostics."""
        if len(request.images) != 1 or request.images[0].name != "observation.images.rgb":
            raise ValueError("expected one observation.images.rgb image")
        with Image.open(io.BytesIO(request.images[0].data)) as source:
            rgb = np.array(source.convert("RGB"))
        with self.lock, torch.inference_mode():
            # Captured graph buffers are tied to the main default CUDA stream.
            torch.cuda.set_device(self.device)
            with torch.cuda.stream(torch.cuda.default_stream(self.device)):
                record = self.benchmark.timed_call(
                    lambda: self.call(rgb, request.instruction, request.session_id, request.request_id),
                    self.device,
                )
                rows = self.last_output.tolist()
                graph_stats = getattr(self.policy, "cuda_graph_stats", lambda: {})()
        return ModelResult(
            action_space=self.action_space,
            actions=(ModelAction("discrete_chunk", {"rows": rows, **record, "graph_stats": graph_stats}),),
            timing={"e2e_ms": record["latency_ms"], **record["model_timing_ms"]},
            policy_revision="frozen-benchmark",
        )

    def reset(self, session_id: str) -> None:
        """Release recurrent state at every HTTP session boundary."""
        with self.lock:
            self.memories.pop(session_id, None)


def main() -> None:
    """Warm up a pinned snapshot, freeze captures, then expose its HTTP endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    benchmark_path = args.source / "benchmarks/activevln-benchmark/benchmark.py"
    spec = importlib.util.spec_from_file_location("frozen_activevln_benchmark", benchmark_path)
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    config = json.loads(args.config.read_text())
    device = benchmark.cuda_device(config)
    torch.set_num_threads(config["cpu_threads"])
    started = time.monotonic()
    policy = make_policy(
        "activevln",
        checkpoint=config["checkpoint"],
        revision=config["checkpoint_revision"],
        attention=config["attention"],
        max_new_tokens=config["max_new_tokens"],
        max_context=config["max_context"],
        do_sample=config["do_sample"],
        repetition_penalty=config["repetition_penalty"],
    )
    policy.to(device=device, dtype=getattr(torch, config["dtype"])).eval()
    adapter = ReplayAdapter(policy, benchmark, device)
    datasets = [(entry, benchmark.load_navigation(entry)) for entry in config["datasets"]]
    warmup = next(
        ep for _, episodes in datasets for ep in episodes if len(ep.frames) >= config["warmup_calls"]
    )
    # Older frozen sources predate this option. Only request it when the
    # matching replay configuration explicitly enables the newer projection.
    projection_options = {"tree_fp32_projection": True} if config.get("tree_fp32_projection", False) else {}
    with torch.inference_mode():
        capture = (
            policy.startup_cuda_graph_capture(
                query_bucket_size=config["query_bucket_size"],
                fused_ops=config["fused_ops"],
                split_attention=config["split_attention"],
                tree_decode=config["tree_decode"],
                tree_repeat_actions=config["tree_repeat_actions"],
                workspace_tokens=config["graph_workspace_tokens"],
                **projection_options,
            )
            if config.get("cuda_graph", False)
            else nullcontext()
        )
        with capture:
            for i, frame in enumerate(warmup.frames[: config["warmup_calls"]]):
                adapter.call(benchmark.read_rgb(frame), warmup.instruction, "warmup", f"warmup-{i}")
            if config.get("prewarm_context_buckets"):
                probes = (
                    adapter.observation(benchmark.read_rgb(frame), ep.instruction)
                    for _, episodes in datasets
                    for ep in episodes
                    for frame in ep.frames[:2]
                )
                policy.prewarm_cuda_graphs(probes, context_buckets=config["prewarm_context_buckets"])
        adapter.reset("warmup")
        torch.cuda.synchronize(device)
    service = PolicyHttpService(
        adapter,
        token=os.environ["ACTIVEVLN_BENCHMARK_TOKEN"],
        maximum_sessions=1,
        maximum_images=1,
        maximum_image_bytes=8 * 1024**2,
        max_body_bytes=10 * 1024**2,
    )
    server = create_http_server(service, host="127.0.0.1", port=0)
    ready = {
        "pid": os.getpid(),
        "port": server.server_address[1],
        "config": config,
        "source": str(args.source),
        "environment": benchmark.provenance(device),
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        "startup_seconds": time.monotonic() - started,
    }
    temporary = args.ready_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(ready, indent=2) + "\n")
    temporary.replace(args.ready_file)
    print(f"Ready: port {server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
