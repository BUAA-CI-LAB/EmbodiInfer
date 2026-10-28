"""Expose a frozen StreamVLN benchmark through the versioned policy HTTP API.

Simulation and navigation metrics belong to the downstream EmbodiRun runner.
The measured interval matches the recorded-input benchmark and excludes PNG
decoding, network transfer, checkpoint loading and startup graph capture.
"""

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
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import provenance, read_rgb, timed_model
from PIL import Image

from embodiinfer.engine.serve.contracts import ModelAction, ModelResult, RawPolicyRequest
from embodiinfer.engine.serve.http_server import PolicyHttpService, create_http_server
from embodiinfer.policies import make_policy
from embodiinfer.policies.streamvln.processing import make_streamvln_observation


class BenchmarkAdapter:
    """Serialize B1 inference and retain only each session's own generated history."""

    action_space = "streamvln.action_chunk.v1"

    def __init__(self, policy: Any, device: torch.device, *, uint8: bool) -> None:
        self.policy, self.device, self.uint8 = policy, device, uint8
        self.memories: dict[str, Any] = {}
        self.lock = threading.RLock()

    def capabilities(self) -> dict[str, Any]:
        """Describe the frozen model's existing action contract."""
        return dict(model="streamvln", action_space=self.action_space, recurrent=True, max_batch_size=1)

    @torch.inference_mode()
    def call(
        self,
        rgb: np.ndarray,
        instruction: str,
        session: str,
        identity: str,
        *,
        stream_step_id: int | None = None,
    ) -> dict[str, Any]:
        """Measure decoded CPU RGB to owned CPU actions, including all model stages."""
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        frames = torch.from_numpy(rgb[None] if rgb.ndim == 3 else rgb).permute(0, 3, 1, 2)
        metadata = {} if stream_step_id is None else {"stream_step_id": stream_step_id}
        observation = make_streamvln_observation(frames[-1], instruction, uint8=self.uint8, metadata=metadata)
        if len(frames) > 1:
            observation.images = frames if self.uint8 else frames.float().div_(255)
        batch = self.policy.collate([observation], [identity])
        prepared = self.policy.prepare_prefix(batch, self.memories.get(session))
        prefix, generation, timing = timed_model(
            lambda: self.policy.encode_prepared_prefix(prepared),
            self.policy.decoder.generate_tokens,
            self.device,
        )
        result = self.policy.decoder.finalize_generation(prefix, generation)
        trace = result.traces[0]
        actions = result.actions[0].detach().float().cpu()
        mask = trace.action_mask.detach().bool().cpu()
        torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - start) * 1000
        record = dict(
            data=actions[mask].tolist(),
            parsed_valid=not trace.meta["invalid_action_text"],
            truncated_actions=trace.meta["truncated_actions"],
            tokens=trace.token_ids.cpu().tolist(),
            logprobs=trace.token_logprobs.cpu().tolist(),
            text=trace.text,
            stop_reason=trace.stop_reason,
            cache_length=result.next_memory.seq_len,
            stream_step_id=result.next_memory.step_id - 1,
            slow_frame_indices=list(result.next_memory.slow_frame_indices),
            effective_cache_length=result.next_memory.seq_len + len(result.next_memory.pending_token_ids),
            timing=dict(e2e_ms=elapsed, **timing),
            graph_stats=self.policy.cuda_graph_runtime_stats(),
            cuda_allocated_mib=torch.cuda.memory_allocated(self.device) / 2**20,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(self.device) / 2**20,
            cuda_reserved_mib=torch.cuda.memory_reserved(self.device) / 2**20,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(self.device) / 2**20,
        )
        self.memories[session] = result.next_memory
        return record

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Decode transport bytes outside timing and use the capture stream."""
        if not 1 <= len(request.images) <= 4:
            raise ValueError("expected one to four consecutive primitive RGB frames")
        frames = []
        for i, image in enumerate(request.images):
            if image.name != f"observation.images.rgb.{i}":
                raise ValueError("primitive RGB frames must be ordered and consecutively named")
            with Image.open(io.BytesIO(image.data)) as source:
                frames.append(np.array(source.convert("RGB")))
        rgb = np.stack(frames)
        step = request.metadata.get("stream_step_id")
        if type(step) is not int or step < 0:
            raise ValueError("stream_step_id must identify the current primitive observation")
        with self.lock, torch.inference_mode():
            torch.cuda.set_device(self.device)
            with torch.cuda.stream(torch.cuda.default_stream(self.device)):
                record = self.call(
                    rgb, request.instruction, request.session_id, request.request_id, stream_step_id=step
                )
        timing = record.pop("timing")
        return ModelResult(
            action_space=self.action_space,
            actions=(ModelAction("action_chunk", record),),
            timing=timing,
            policy_revision="frozen-benchmark",
        )

    def reset(self, session_id: str) -> None:
        """Release recurrent tensors at the session boundary."""
        with self.lock:
            self.memories.pop(session_id, None)


def main() -> None:
    """Load, warm up, freeze captures, and publish provenance before serving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    started = time.monotonic()
    policy = make_policy("streamvln", **config["policy"]).to(device).eval()
    adapter = BenchmarkAdapter(policy, device, uint8=config["uint8_observations"])
    warmup = config["warmup"]
    stride = warmup.get("stride", 1)
    if type(stride) is not int or not 1 <= stride <= 4 or (len(warmup["frames"]) - 1) % stride:
        raise ValueError("warmup must contain an initial frame followed by complete strides of one to four")
    if config["policy"].get("cuda_graph", True) and len(warmup["frames"]) <= policy.processor.window_size:
        raise ValueError("graph warmup must cover the history-window transition")
    capture = policy.startup_cuda_graph_capture() if policy.cuda_graph else nullcontext()
    with torch.inference_mode(), capture:
        start = 0
        for step in range(0, len(warmup["frames"]), stride):
            frames = np.stack([read_rgb(Path(frame)) for frame in warmup["frames"][start : step + 1]])
            adapter.call(frames, warmup["instruction"], "warmup", f"warmup-{step}", stream_step_id=step)
            start = step + 1
    adapter.reset("warmup")
    policy.clear_preprocessing_caches()
    policy.reset_cuda_graph_runtime_stats()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("empty service token")
    service = PolicyHttpService(
        adapter,
        token=token,
        maximum_sessions=1,
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
        graph_capture=policy.cuda_graph_capture_stats(),
        warmup_model_calls=1 + (len(warmup["frames"]) - 1) // stride,
    )
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
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
