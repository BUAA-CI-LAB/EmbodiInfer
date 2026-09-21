"""Compiled CUDA Graph runners for StreamVLN vision and language prefill."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .backbone import StreamVLNBackbone, StreamVLNCache


def _compile(call: Callable[[torch.Tensor], torch.Tensor]):
    if not hasattr(torch, "compile"):
        return call
    return torch.compile(
        call,
        dynamic=False,
        fullgraph=False,
        mode="max-autotune-no-cudagraphs",
    )


class _StartupCaptureRunner:
    """Allow graph creation only inside an explicit startup-capture phase."""

    def __init__(self) -> None:
        self._capture_active = False
        self._capture_frozen = False
        self.graph_replays = 0
        self.eager_fallbacks = 0

    @property
    def capture_active(self) -> bool:
        return self._capture_active

    @property
    def capture_frozen(self) -> bool:
        return self._capture_frozen

    def begin_startup_capture(self) -> None:
        if self._capture_active:
            raise RuntimeError("StreamVLN startup graph capture is already active")
        if self._capture_frozen:
            raise RuntimeError("StreamVLN startup graph capture is already frozen")
        self._capture_active = True

    def finish_startup_capture(self) -> None:
        if not self._capture_active:
            raise RuntimeError("StreamVLN startup graph capture is not active")
        self._capture_active = False
        self._capture_frozen = True

    def abort_startup_capture(self) -> None:
        self._capture_active = False
        self._capture_frozen = False

    def reset_runtime_stats(self) -> None:
        self.graph_replays = 0
        self.eager_fallbacks = 0


class _CapturedTensorCall:
    def __init__(
        self,
        call: Callable[[torch.Tensor], torch.Tensor],
        example: torch.Tensor,
        pool,
        *,
        compile_call: bool = True,
    ) -> None:
        self.input = torch.empty_like(example)
        self.input.copy_(example)
        self.call = _compile(call) if compile_call else call

        # Compile before entering stream capture. Two side-stream executions
        # populate allocator and library workspaces used by the captured path.
        self.call(self.input)
        torch.cuda.synchronize(example.device)
        current = torch.cuda.current_stream(example.device)
        warmup = torch.cuda.Stream(device=example.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                self.call(self.input)
        current.wait_stream(warmup)
        current.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.output = self.call(self.input)

    def replay(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape != self.input.shape or inputs.dtype != self.input.dtype:
            raise ValueError("CUDA Graph input shape or dtype changed")
        self.input.copy_(inputs)
        self.graph.replay()
        return self.output


class _CapturedPrefillCall:
    def __init__(
        self,
        call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        example: torch.Tensor,
        past_length: int,
        pool,
    ) -> None:
        self.input = torch.empty_like(example)
        self.input.copy_(example)
        self.position = torch.tensor(
            [past_length],
            dtype=torch.long,
            device=example.device,
        )
        self.call = call
        self.call(self.input, self.position)
        torch.cuda.synchronize(example.device)
        current = torch.cuda.current_stream(example.device)
        warmup = torch.cuda.Stream(device=example.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                self.call(self.input, self.position)
        current.wait_stream(warmup)
        current.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.output = self.call(self.input, self.position)

    def replay(self, inputs: torch.Tensor, past_length: int) -> torch.Tensor:
        self.input.copy_(inputs)
        self.position.fill_(past_length)
        self.graph.replay()
        return self.output


class StreamVLNVisionRunner(_StartupCaptureRunner):
    """Compile and capture fixed-shape SigLIP feature extraction."""

    def __init__(self, backbone: StreamVLNBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = None
        self.graphs: dict[tuple[tuple[int, ...], torch.dtype, torch.device], _CapturedTensorCall] = {}

    def run(self, pixel_values: torch.Tensor) -> torch.Tensor:
        key = (tuple(pixel_values.shape), pixel_values.dtype, pixel_values.device)
        graph = self.graphs.get(key)
        if graph is None:
            if not self.capture_active:
                self.eager_fallbacks += 1
                return self.backbone._encode_frames_eager(pixel_values)
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph = _CapturedTensorCall(
                self.backbone._encode_frames_eager,
                pixel_values,
                self.pool,
            )
            self.graphs[key] = graph
        self.graph_replays += 1
        return graph.replay(pixel_values)


class StreamVLNPrefillRunner(_StartupCaptureRunner):
    """Compile and capture fixed-address Qwen2 prefill states."""

    def __init__(self, backbone: StreamVLNBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = None
        self.cache: StreamVLNCache | None = None
        self.graphs: dict[
            tuple[int, int, torch.dtype, torch.device],
            _CapturedPrefillCall,
        ] = {}

    def _ensure_cache(
        self,
        inputs: torch.Tensor,
        cache: StreamVLNCache | None,
    ) -> StreamVLNCache:
        if self.cache is None:
            self.cache = self.backbone._allocate_cache(inputs)
        if cache is not None and (
            cache.key_storage.data_ptr() != self.cache.key_storage.data_ptr()
            or cache.value_storage.data_ptr() != self.cache.value_storage.data_ptr()
        ):
            raise ValueError("StreamVLN prefill graph cache storage changed")
        return self.cache

    def _cache_matches(self, cache: StreamVLNCache | None) -> bool:
        return (
            cache is None
            or self.cache is None
            or (
                cache.key_storage.data_ptr() == self.cache.key_storage.data_ptr()
                and cache.value_storage.data_ptr() == self.cache.value_storage.data_ptr()
            )
        )

    def _run_eager(
        self,
        inputs: torch.Tensor,
        cache: StreamVLNCache | None,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        self.eager_fallbacks += 1
        # Reuse the graph-owned cache for a new window even when its exact
        # prefill shape is uncommon. Later common shapes can then safely replay
        # graphs against the same fixed storage.
        working_cache = self.cache.advance(0) if cache is None and self.cache is not None else cache
        return self.backbone._forward_embeddings_eager(
            inputs,
            working_cache,
            project_logits=False,
            return_hidden_states=True,
        )

    def run(
        self,
        inputs: torch.Tensor,
        cache: StreamVLNCache | None,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        past_length = 0 if cache is None else cache.seq_len
        query_length = int(inputs.shape[1])
        cache_end = past_length + query_length
        if cache_end > self.backbone.max_context:
            raise RuntimeError("StreamVLN prefill graph exceeded KV capacity")
        key_bucket = max(512, 1 << (cache_end - 1).bit_length())
        key = (query_length, key_bucket, inputs.dtype, inputs.device)
        graph = self.graphs.get(key)
        if graph is None and not self.capture_active:
            return self._run_eager(inputs, cache)
        if not self._cache_matches(cache):
            if not self.capture_active:
                return self._run_eager(inputs, cache)
            raise ValueError("StreamVLN prefill graph cache storage changed during startup")
        static_cache = self._ensure_cache(inputs, cache)
        if cache_end > static_cache.capacity:
            raise RuntimeError("StreamVLN prefill graph exceeded KV capacity")
        if graph is None:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()

            def prefill(
                embeddings: torch.Tensor,
                position: torch.Tensor,
            ) -> torch.Tensor:
                return self.backbone.prefill_embeddings_graph(
                    embeddings,
                    static_cache,
                    position,
                    key_bucket,
                )

            graph = _CapturedPrefillCall(
                prefill,
                inputs,
                past_length,
                self.pool,
            )
            self.graphs[key] = graph
        self.graph_replays += 1
        hidden = graph.replay(inputs, past_length)
        return static_cache.advance(cache_end), hidden


__all__ = ["StreamVLNPrefillRunner", "StreamVLNVisionRunner"]
