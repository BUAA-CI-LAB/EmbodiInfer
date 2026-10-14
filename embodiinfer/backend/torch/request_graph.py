"""Model-independent lifetime for a private, whole-request CUDA graph.

The adapter owns shapes, static input buffers and the forward function. This
module owns capture serialization, stream affinity, output ownership and teardown.
It never changes precision, kernels, sampling, or the forward's operation order.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import torch

# Shared with engine.graph: lazy capture has process-global CUDA runtime state.
CUDA_GRAPH_CAPTURE_LOCK = threading.RLock()


def cuda_execution_signature() -> tuple:
    """Key precision and deterministic modes that CUDA replay cannot re-evaluate.

    Mode changes choose a new capture instead of silently reusing old arithmetic.
    This does not change any global setting. Applications must not mutate process
    global Torch flags concurrently with inference.
    """
    try:
        autocast = torch.is_autocast_enabled("cuda")
        autocast_dtype = torch.get_autocast_dtype("cuda")
    except TypeError:  # PyTorch 2.1/2.2 compatibility.
        autocast = torch.is_autocast_enabled()
        autocast_dtype = torch.get_autocast_gpu_dtype()
    matmul = torch.backends.cuda.matmul
    return (
        autocast,
        autocast_dtype,
        torch.is_autocast_cache_enabled(),
        torch.get_float32_matmul_precision(),
        matmul.allow_tf32,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction,
        getattr(matmul, "allow_fp16_accumulation", False),
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.enabled,
        tuple(
            (name, getattr(torch.backends.cuda, name, lambda: None)())
            for name in (
                "flash_sdp_enabled",
                "mem_efficient_sdp_enabled",
                "math_sdp_enabled",
                "cudnn_sdp_enabled",
                "fp16_bf16_reduction_math_sdp_allowed",
            )
        ),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


class CapturedRequest:
    """Capture one adapter-owned forward and return independent replay outputs.

    Inputs must be allocated before construction and kept alive by the adapter.
    ``update_inputs`` runs on the creating stream, before replay. Callers must not
    mutate weights while a request is in flight; invalidate before refit/migration.
    A separate instance is required for every stream and static input signature.
    ``release`` runs only after device work is complete and graph storage is reset.
    """

    def __init__(
        self,
        forward: Callable[[], torch.Tensor],
        device: torch.device,
        *,
        execution_context: Callable[[], AbstractContextManager] = nullcontext,
        release: Callable[[], None] | None = None,
        warmup: int = 3,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CapturedRequest requires a CUDA device")
        if warmup < 1:
            raise ValueError("CapturedRequest requires at least one warmup")
        self.device = device
        self.stream = torch.cuda.current_stream(device)
        self._signature = cuda_execution_signature()
        if self._signature[0] and self._signature[2]:
            raise ValueError(
                "CUDA request capture requires autocast cache_enabled=False when autocast is enabled"
            )
        self._release = release
        self._lock = threading.RLock()
        self._closed = False
        self.graph = torch.cuda.CUDAGraph()
        try:
            with CUDA_GRAPH_CAPTURE_LOCK, torch.cuda.device(device), execution_context():
                try:
                    capture_stream = torch.cuda.Stream(device=device)
                    capture_stream.wait_stream(self.stream)
                    with torch.cuda.stream(capture_stream):
                        for _ in range(warmup):
                            forward()
                    self.stream.wait_stream(capture_stream)
                    with torch.cuda.graph(
                        self.graph, stream=capture_stream, capture_error_mode="thread_local"
                    ):
                        self.output = forward()
                finally:
                    # capture_end can raise before torch.cuda.graph restores its stream.
                    torch.cuda.set_stream(self.stream)
        except BaseException:
            self.close()
            raise

    def run(self, update_inputs: Callable[[], None]) -> torch.Tensor:
        """Update inputs and replay atomically; the returned tensor never aliases the graph."""
        with self._lock, torch.cuda.device(self.device):
            if self._closed:
                raise RuntimeError("CapturedRequest is closed")
            if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
                raise RuntimeError("CapturedRequest must replay on its owning stream")
            if cuda_execution_signature() != self._signature:
                raise RuntimeError("CapturedRequest precision/determinism mode changed; recapture required")
            update_inputs()
            self.graph.replay()
            return self.output.clone()

    def close(self, *, synchronize: bool = True) -> None:
        """Invalidate the capture and free adapter workspaces after pending work completes.

        ``synchronize=False`` is only for callers that already synchronized the
        entire device. Closing repeatedly is harmless; closed graphs cannot replay.
        """
        with self._lock:
            if self._closed:
                return
            if synchronize:
                torch.cuda.synchronize(self.device)
            self.graph.reset()
            self._closed = True
            release, self._release = self._release, None
            if release is not None:
                release()
