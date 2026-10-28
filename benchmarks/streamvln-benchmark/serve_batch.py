"""Serve independent StreamVLN episodes in measured, explicit tensor batches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import logging
import os
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import provenance, read_rgb
from PIL import Image

from embodiinfer.engine.serve.batching import BatchedServingAdapter
from embodiinfer.engine.serve.contracts import ModelAction, ModelResult, RawPolicyRequest
from embodiinfer.engine.serve.http_server import PolicyHttpService, create_http_server
from embodiinfer.policies import make_policy
from embodiinfer.policies.streamvln.processing import make_streamvln_observation


def primitive_observation(rgb: np.ndarray, instruction: str, step: int, *, uint8: bool) -> Any:
    """Preserve all ordered primitive frames, with one current frame per turn."""
    frames = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    observation = make_streamvln_observation(
        frames[-1], instruction, uint8=uint8, metadata={"stream_step_id": step}
    )
    observation.images = frames if uint8 else frames.float().div_(255)
    return observation


class TensorBatchAdapter:
    """Commit independent histories only after all outputs in one batch succeed."""

    action_space = "streamvln.action_chunk.v1"

    def __init__(self, policy: Any, runtime: Any, device: torch.device, *, uint8: bool) -> None:
        self.policy, self.runtime, self.device, self.uint8 = policy, runtime, device, uint8
        self.memories: dict[str, Any] = {}
        self.lock = threading.RLock()
        self.index = 0
        self.failed = False

    def capabilities(self) -> dict[str, Any]:
        """Advertise the tensor capacity used by the coalescing scheduler."""
        return dict(
            model="streamvln",
            action_space=self.action_space,
            recurrent=True,
            max_batch_size=self.runtime.batch_size,
        )

    @torch.inference_mode()
    def call_batch(
        self,
        images: Sequence[np.ndarray],
        instructions: Sequence[str],
        sessions: Sequence[str],
        identities: Sequence[str],
        steps: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Measure one tensor call from decoded RGB through owned CPU actions.

        The complete batch latency is repeated in each row, with a shared batch
        index. Consumers must deduplicate that index before computing throughput.
        PNG decoding, scheduler delay and HTTP transport are outside this timer.
        """
        if self.failed:
            raise RuntimeError("a previous model failure invalidated this benchmark process")
        if not 1 <= len(images) <= self.runtime.batch_size or any(
            len(values) != len(images) for values in (instructions, sessions, identities, steps)
        ):
            raise ValueError("batch inputs must align within runtime capacity")
        if len(set(sessions)) != len(sessions) or len(set(identities)) != len(identities):
            raise ValueError("batch sessions and request identities must be unique")
        try:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            observations = [
                primitive_observation(rgb, instruction, step, uint8=self.uint8)
                for rgb, instruction, step in zip(images, instructions, steps, strict=True)
            ]
            results = self.runtime.infer(observations, identities, [self.memories.get(s) for s in sessions])
            if len(results) != len(images):
                raise RuntimeError("tensor runtime returned an incomplete batch")
            actions = [result.actions[0].detach().float().cpu() for result in results]
            masks = [result.traces[0].action_mask.detach().bool().cpu() for result in results]
            torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - started) * 1000
            records = []
            for result, action, mask in zip(results, actions, masks, strict=True):
                if not torch.isfinite(action).all():
                    raise ValueError("nonfinite model action")
                trace = result.traces[0]
                records.append(
                    dict(
                        data=action[mask].tolist(),
                        parsed_valid=not trace.meta["invalid_action_text"],
                        truncated_actions=trace.meta["truncated_actions"],
                        tokens=trace.token_ids.cpu().tolist(),
                        logprobs=trace.token_logprobs.cpu().tolist(),
                        text=trace.text,
                        stop_reason=trace.stop_reason,
                        cache_length=result.next_memory.seq_len,
                        stream_step_id=result.next_memory.step_id - 1,
                        slow_frame_indices=list(result.next_memory.slow_frame_indices),
                        effective_cache_length=result.next_memory.seq_len
                        + len(result.next_memory.pending_token_ids),
                        timing=dict(e2e_ms=elapsed),
                        graph_stats=self.runtime.stats(),
                        cuda_allocated_mib=torch.cuda.memory_allocated(self.device) / 2**20,
                        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(self.device) / 2**20,
                        cuda_reserved_mib=torch.cuda.memory_reserved(self.device) / 2**20,
                        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(self.device) / 2**20,
                        tensor_batch=dict(
                            index=self.index, observations=len(results), capacity=self.runtime.batch_size
                        ),
                    )
                )
        except BaseException:
            self.failed = True
            raise
        self.memories.update(
            {session: result.next_memory for session, result in zip(sessions, results, strict=True)}
        )
        self.index += 1
        return records

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Run a singleton tail through the same tensor runtime."""
        return self.infer_batch([request])[0]

    def infer_batch(self, requests: Sequence[RawPolicyRequest]) -> list[ModelResult]:
        """Sort stable episode slots, execute once, then restore HTTP arrival order."""
        slots = [request.state.get("benchmark_slot") for request in requests]
        if (
            not requests
            or len(requests) > self.runtime.batch_size
            or any(type(slot) is not int or not 0 <= slot < self.runtime.batch_size for slot in slots)
        ):
            raise ValueError("each request requires a valid benchmark_slot within capacity")
        if len(set(slots)) != len(slots) or len({r.session_id for r in requests}) != len(requests):
            raise ValueError("duplicate slot or session in tensor batch")
        order = sorted(range(len(requests)), key=lambda row: slots[row])
        ordered = [requests[row] for row in order]
        images, steps = [], []
        for request in ordered:
            if not 1 <= len(request.images) <= 4:
                raise ValueError("expected one to four consecutive primitive RGB frames")
            step = request.metadata.get("stream_step_id")
            if type(step) is not int or step < 0:
                raise ValueError("stream_step_id must identify the current primitive observation")
            frames = []
            for i, image in enumerate(request.images):
                if image.name != f"observation.images.rgb.{i}":
                    raise ValueError("primitive RGB frames must be ordered and consecutively named")
                with Image.open(io.BytesIO(image.data)) as source:
                    frames.append(np.array(source.convert("RGB")))
            images.append(np.stack(frames))
            steps.append(step)
        with self.lock, torch.inference_mode():
            torch.cuda.set_device(self.device)
            with torch.cuda.stream(torch.cuda.default_stream(self.device)):
                records = self.call_batch(
                    images,
                    [r.instruction for r in ordered],
                    [r.session_id for r in ordered],
                    [r.request_id for r in ordered],
                    steps,
                )
        restored: dict[int, ModelResult] = {}
        for row, record in zip(order, records, strict=True):
            timing = record.pop("timing")
            restored[row] = ModelResult(
                self.action_space, (ModelAction("action_chunk", record),), timing, "frozen-tensor-benchmark"
            )
        return [restored[row] for row in range(len(requests))]

    def reset(self, session_id: str) -> None:
        """Release the session and any scratch-residency reference to its memory."""
        with self.lock:
            memory = self.memories.pop(session_id, None)
            if memory is not None:
                self.runtime.resident = [None if item is memory else item for item in self.runtime.resident]


def main() -> None:
    """Warm and freeze a tensor profile before accepting any measured request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-wait-ms", type=float, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text())
    count = config["batch_size"]
    if type(count) is not int or count not in (1, 2, 4, 8):
        raise ValueError("benchmark batch_size must be one of 1, 2, 4, 8")
    warmup = config["warmup"]
    stride = warmup.get("stride", 4)
    if type(stride) is not int or not 1 <= stride <= 4 or (len(warmup["frames"]) - 1) % stride:
        raise ValueError("warmup requires an initial frame followed by complete primitive strides")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    started = time.monotonic()
    policy = make_policy("streamvln", **config["policy"]).to(device).eval()
    if len(warmup["frames"]) <= policy.processor.window_size:
        raise ValueError("warmup must cover the primitive history-window transition")
    prefill_plan = None
    with torch.inference_mode():
        runtime = policy.create_batched_runtime(batch_size=count, **config.get("batch_runtime", {}))
        adapter = TensorBatchAdapter(policy, runtime, device, uint8=config["uint8_observations"])
        capture = runtime.startup_capture() if policy.cuda_graph else nullcontext()
        with capture:
            start = 0
            for step in range(0, len(warmup["frames"]), stride):
                rgb = np.stack([read_rgb(Path(frame)) for frame in warmup["frames"][start : step + 1]])
                adapter.call_batch(
                    [rgb] * count,
                    [warmup["instruction"]] * count,
                    [f"warmup-{i}" for i in range(count)],
                    [f"warmup-{i}-{step}" for i in range(count)],
                    [step] * count,
                )
                start = step + 1
            if config.get("prefill_warmup_manifest"):
                from prefill_shapes import plan_prefill_shapes

                if policy.fast_action_decode:
                    raise ValueError("prompt shape census requires fast_action_decode=False")
                manifest = Path(config["prefill_warmup_manifest"]).read_bytes()
                if hashlib.sha256(manifest).hexdigest() != config["prefill_warmup_manifest_sha256"]:
                    raise ValueError("prefill warmup manifest changed after configuration was frozen")
                feature = next(
                    value for value in adapter.memories["warmup-0"].frame_features if value is not None
                )
                prefill_plan = plan_prefill_shapes(
                    policy.processor,
                    [
                        episode["instruction"]["instruction_text"]
                        for episode in json.loads(manifest)["episodes"]
                    ],
                    vision_tokens=feature.shape[0],
                    batch_size=count,
                    token_bucket=runtime.packed_prefill_bucket,
                    group_size=policy.backbone.num_heads // policy.backbone.num_kv_heads,
                    workspace_tokens=runtime.capacity,
                    kv_pool_tokens=runtime.pool_tokens,
                    max_new_tokens=policy.max_new_tokens,
                    max_graphs=config.get("prefill_warmup_max_graphs", 2048),
                )
                runtime.prewarm_prefill(prefill_plan["shapes"])
            if config.get("prefill_warmup_shapes"):
                runtime.prewarm_prefill(config["prefill_warmup_shapes"])
        for i in range(count):
            adapter.reset(f"warmup-{i}")
        adapter.index = 0
        runtime.reset_stats()
        policy.clear_preprocessing_caches()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("empty service token")
    scheduler = BatchedServingAdapter(adapter, max_batch=count, max_wait_ms=args.max_wait_ms)
    service = PolicyHttpService(
        scheduler,
        token=token,
        maximum_sessions=count,
        maximum_images=4,
        maximum_image_bytes=8 * 1024**2,
        max_body_bytes=10 * 1024**2,
    )
    server = create_http_server(service, host="127.0.0.1", port=args.port)
    ready = dict(
        pid=os.getpid(),
        port=server.server_address[1],
        config=config,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        environment=dict(
            provenance(device), allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", "default")
        ),
        startup_seconds=time.monotonic() - started,
        runtime=runtime.stats(),
        scheduler=scheduler.capabilities(),
        warmup_model_calls=1 + (len(warmup["frames"]) - 1) // stride,
        prefill_warmup_shapes=config.get("prefill_warmup_shapes", []),
        prefill_warmup_plan=prefill_plan,
        timing_contract="decoded CPU RGB to owned CPU actions; shared batch latency; deduplicate tensor_batch.index",
    )
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
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
