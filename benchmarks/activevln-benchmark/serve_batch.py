"""Expose a frozen tensor-batch runtime through the versioned policy HTTP API.

The adapter may live outside --source so an existing measured snapshot remains
immutable. Its own hash is recorded separately from the runtime provenance.
No simulator or navigation metric code belongs in this process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from embodiinfer.engine.serve.batching import BatchedServingAdapter
from embodiinfer.engine.serve.contracts import ModelAction, ModelResult, RawPolicyRequest
from embodiinfer.engine.serve.http_server import PolicyHttpService, create_http_server
from embodiinfer.policies import make_policy


class TensorBatchAdapter:
    """Commit independent session memories only after every batch result succeeds."""

    action_space = "activevln.r2r.discrete.v1"

    def __init__(
        self, policy: Any, runtime: Any, benchmark: Any, device: torch.device, evidence: Path
    ) -> None:
        self.policy, self.runtime, self.benchmark, self.device = policy, runtime, benchmark, device
        self.evidence = evidence
        self.memories: dict[str, Any] = {}
        self.lock = threading.RLock()
        self.index = 0
        self.failed = False

    def capabilities(self) -> dict[str, Any]:
        """Advertise actual tensor capacity and independent recurrent sessions."""
        return {
            "model": "activevln",
            "action_space": self.action_space,
            "recurrent": True,
            "max_batch_size": self.runtime.batch_size,
        }

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Allow a singleton tail through the same tensor runtime."""
        return self.infer_batch([request])[0]

    def infer_batch(self, requests: Sequence[RawPolicyRequest]) -> list[ModelResult]:
        """Decode PNG outside the timer, then execute one ordered tensor batch.

        The benchmark controller provides stable slot IDs; arrival order must not
        change padding or per-row histories. A failed model call invalidates this
        adapter so partial static workspace writes cannot be reused by a retry.
        """
        if not requests or len(requests) > self.runtime.batch_size:
            raise ValueError("invalid tensor batch occupancy")
        slots = [request.state.get("benchmark_slot") for request in requests]
        if any(type(slot) is not int or not 0 <= slot < self.runtime.batch_size for slot in slots):
            raise ValueError("each request needs a valid benchmark_slot")
        if len(set(slots)) != len(slots) or len({r.session_id for r in requests}) != len(requests):
            raise ValueError("duplicate slot or session in tensor batch")
        order = sorted(range(len(requests)), key=lambda row: slots[row])
        ordered = [requests[row] for row in order]
        images = []
        for request in ordered:
            if len(request.images) != 1 or request.images[0].name != "observation.images.rgb":
                raise ValueError("expected one observation.images.rgb image")
            with Image.open(io.BytesIO(request.images[0].data)) as source:
                images.append(np.array(source.convert("RGB")))
        with self.lock, torch.inference_mode():
            if self.failed:
                raise RuntimeError("a previous model failure invalidated this benchmark process")
            torch.cuda.set_device(self.device)
            with torch.cuda.stream(torch.cuda.default_stream(self.device)):
                evidence = {
                    "index": self.index,
                    "observations": len(ordered),
                    "rows": [
                        {
                            "slot": r.state["benchmark_slot"],
                            "session_id": r.session_id,
                            "step_id": r.step_id,
                            "cache_length": getattr(self.memories.get(r.session_id), "seq_len", 0),
                        }
                        for r in ordered
                    ],
                }
                try:
                    torch.cuda.synchronize(self.device)
                    start = time.perf_counter_ns()
                    observations = [
                        self.benchmark.make_observation(image, request.instruction)
                        for image, request in zip(images, ordered, strict=True)
                    ]
                    prepared = self.runtime.prepare(
                        observations, [self.memories.get(r.session_id) for r in ordered]
                    )
                    _, generations, timing = self.benchmark.timed_model(
                        lambda: self.runtime.prefill(prepared), self.runtime.generate, self.device
                    )
                    decoded = [self.policy.decoder.finalize_generation(g) for g in generations]
                    actions = [result.actions[0].detach().float().cpu() for result in decoded]
                    torch.cuda.synchronize(self.device)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    outputs = []
                    for result, action in zip(decoded, actions, strict=True):
                        if not torch.isfinite(action).all():
                            raise ValueError("nonfinite model action")
                        trace = result.traces[0]
                        values = {
                            "rows": action.tolist(),
                            "token_ids": trace.token_ids.cpu().tolist(),
                            "text": trace.text,
                            "parsed_valid": trace.parsed_actions.valid,
                            "stop_reason": trace.stop_reason,
                            "cache_length": result.next_memory.seq_len,
                            "action_mask": trace.meta["parsed_action_mask"].tolist(),
                            "output_sha256": hashlib.sha256(action.numpy().tobytes()).hexdigest(),
                            "tensor_batch": {
                                "index": self.index,
                                "observations": len(ordered),
                                "capacity": self.runtime.batch_size,
                            },
                        }
                        outputs.append(
                            ModelResult(
                                self.action_space,
                                (ModelAction("discrete_chunk", values),),
                                {"e2e_ms": elapsed_ms, **timing},
                                "frozen-tensor-benchmark",
                            )
                        )
                    evidence.update(
                        status="complete",
                        latency_ms=elapsed_ms,
                        model_timing_ms=timing,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                        runtime={
                            k: v for k, v in self.runtime.stats().items() if not isinstance(v, (list, dict))
                        },
                    )
                    self._record(evidence)
                except Exception as error:
                    self.failed = True
                    evidence.update(
                        status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
                    self._record(evidence)
                    raise
                self.memories.update(
                    {r.session_id: result.next_memory for r, result in zip(ordered, decoded, strict=True)}
                )
                self.index += 1
                restored = [None] * len(requests)
                for row, output in zip(order, outputs, strict=True):
                    restored[row] = output
                return restored

    def _record(self, evidence: dict[str, Any]) -> None:
        with self.evidence.open("a") as stream:
            stream.write(json.dumps(evidence, allow_nan=False) + "\n")

    def reset(self, session_id: str) -> None:
        """Release one session only, after its HTTP step has drained."""
        with self.lock:
            self.memories.pop(session_id, None)


def main() -> None:
    """Prewarm the measured configuration and serve coalesced session requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--max-wait-ms", type=float, default=1000)
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
    episodes = [ep for entry in config["datasets"] for ep in benchmark.load_navigation(entry)]
    with torch.inference_mode():
        runtime = policy.create_batched_runtime(
            batch_size=config["batch_size"],
            workspace_tokens=config["graph_workspace_tokens"],
            query_bucket_size=config["query_bucket_size"],
            cuda_graph=config["cuda_graph"],
            fused_ops=config["fused_ops"],
            split_attention=config["split_attention"],
            tree_decode=config["tree_decode"],
            tree_fp32_projection=config["tree_fp32_projection"],
            tree_repeat_actions=config["tree_repeat_actions"],
            kv_pool_tokens=config["kv_pool_tokens"],
        )
        probes = [
            benchmark.make_observation(benchmark.read_rgb(ep.frames[0]), ep.instruction) for ep in episodes
        ]
        runtime.prewarm(probes, context_buckets=config["prewarm_context_buckets"])
        del probes
        warmup = next(ep for ep in episodes if len(ep.frames) >= config["warmup_calls"])
        memories = [None] * config["batch_size"]
        for frame in warmup.frames[: config["warmup_calls"]]:
            observation = benchmark.make_observation(benchmark.read_rgb(frame), warmup.instruction)
            prepared = runtime.prepare([observation] * config["batch_size"], memories)
            generations = runtime.generate(runtime.prefill(prepared))
            memories = [g.memory for g in generations]
            del prepared, generations
        del memories, observation
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        runtime.reset_stats()
    adapter = TensorBatchAdapter(
        policy, runtime, benchmark, device, args.ready_file.with_name("tensor-batches.jsonl")
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
    server = create_http_server(service, host="127.0.0.1", port=0)
    ready = {
        "pid": os.getpid(),
        "port": server.server_address[1],
        "config": config,
        "source": str(args.source),
        "environment": benchmark.provenance(device),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        "startup_seconds": time.monotonic() - started,
        "scheduler": scheduler.capabilities(),
        "runtime": runtime.stats(),
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
