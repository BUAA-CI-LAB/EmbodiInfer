"""CUDA Graph capture and replay runtime owned by the Low policy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal

import torch

from ...models.qwen25_vl import (
    Qwen25VLCompileRuntime,
    apply_qwen25_vl_text_bucket,
    normalize_qwen25_vl_text_buckets,
    prepare_qwen25_vl_next_token_inputs,
    qwen25_vl_bucket_schema,
    qwen25_vl_model_config_sha256,
    qwen25_vl_text_bucket_key,
)
from .forward import QwenR2RLowNextTokenForward


@dataclass
class CapturedNextTokenGraph:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_logits: torch.Tensor
    shape_key: tuple
    capture_ms: float
    compiled_callable: object | None = None
    replay_count: int = 0


class QwenR2RLowGraphRuntime:
    def __init__(
        self,
        model: torch.nn.Module,
        profile: str,
        *,
        attention_backend: str = "torch_sdpa",
        compile_backend: Literal["none", "inductor"] = "none",
        compile_cache_dir: str | Path | None = None,
        compile_text_buckets: tuple[int, ...] = (),
        pad_token_id: int | None = None,
        processor_contract: dict[str, object] | None = None,
    ):
        self.model = model
        self.profile = profile
        requested_buckets = normalize_qwen25_vl_text_buckets(compile_text_buckets)
        self.compile_text_buckets = requested_buckets
        if self.compile_text_buckets and pad_token_id is None:
            raise ValueError("processor.tokenizer.pad_token_id is required for compile text buckets")
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.use_text_attention_mask = bool(self.compile_text_buckets)
        self._unmasked_forward = QwenR2RLowNextTokenForward(
            model,
            attention_backend=attention_backend,
            use_text_attention_mask=False,
        )
        self._masked_forward = QwenR2RLowNextTokenForward(
            model,
            attention_backend=attention_backend,
            use_text_attention_mask=True,
        )
        self.forward = self._masked_forward if self.use_text_attention_mask else self._unmasked_forward
        self._compile_runtime = Qwen25VLCompileRuntime(
            compile_backend,
            compile_cache_dir=compile_cache_dir,
            cache_identity={
                "profile": profile,
                "attention_backend": attention_backend,
                "dtype": str(next(model.parameters()).dtype),
                "model_config_sha256": qwen25_vl_model_config_sha256(model.config),
                "processor_contract": dict(processor_contract or {}),
                "bucket_schema": qwen25_vl_bucket_schema(profile, self.compile_text_buckets),
                "text_attention_mask_policy": "bucket_or_zero_mask_v1",
                "pad_token_id": self.pad_token_id,
            },
        )
        self._manual_graph_cache: dict[tuple, CapturedNextTokenGraph] = {}
        self._manual_graph_captures = 0
        self._manual_graph_replays = 0
        self._execution_lock = RLock()

    def native_inputs(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        with self._execution_lock:
            return self._native_inputs_unlocked(encoded)

    def _native_inputs_unlocked(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if attention_mask.ndim != 2 or not bool(attention_mask[:, -1].detach().all().item()):
            raise ValueError("native/manual graph forward requires the final token to be unmasked")
        self.use_text_attention_mask = bool(self.compile_text_buckets) or not bool(
            attention_mask.detach().ne(0).all().item()
        )
        self.forward = self._masked_forward if self.use_text_attention_mask else self._unmasked_forward
        prepared = prepare_qwen25_vl_next_token_inputs(
            self.model,
            input_ids=input_ids,
            pixel_values=encoded["pixel_values"],
            attention_mask=attention_mask,
            image_grid_thw=encoded["image_grid_thw"],
            video_grid_thw=encoded.get("video_grid_thw"),
            second_per_grid_ts=encoded.get("second_per_grid_ts"),
        )
        self.forward.set_vision_bounds(prepared.window_bounds, prepared.full_bounds)
        self.forward.select_attention_backend(
            device=prepared.tensors[1].device,
            dtype=prepared.tensors[1].dtype,
        )
        return prepared.tensors

    def bucket_encoded(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bucketed, _ = apply_qwen25_vl_text_bucket(
            encoded,
            pad_token_id=self.pad_token_id,
            buckets=self.compile_text_buckets,
        )
        return bucketed

    def uncaptured_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        with self._execution_lock:
            return self._uncaptured_logits_unlocked(encoded)

    def _uncaptured_logits_unlocked(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.native_inputs(encoded)
        key = self.graph_key(encoded, inputs)
        execute = self._execution_callable(key, inputs)
        return execute(*inputs)

    def graph_key(self, encoded: dict[str, torch.Tensor], inputs: tuple[torch.Tensor, ...]) -> tuple:
        with self._execution_lock:
            return self._graph_key_unlocked(encoded, inputs)

    def _graph_key_unlocked(
        self, encoded: dict[str, torch.Tensor], inputs: tuple[torch.Tensor, ...]
    ) -> tuple:
        tensor_key = tuple(
            (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))
            for tensor in inputs
        )
        grid_key = tuple(encoded["image_grid_thw"].detach().cpu().reshape(-1).tolist())
        layout_key = (self.forward.window_bounds, self.forward.full_bounds)
        backend_key = self.forward.attention_backend_cache_key()
        compile_key = self._compile_runtime.cache_key()
        bucket_key = qwen25_vl_text_bucket_key(encoded, self.compile_text_buckets)
        self._compile_runtime.record_bucket(bucket_key)
        resolved = str(self.forward.attention_backend_stats()["resolved"])
        persistent_execution_key = (
            self.profile,
            backend_key,
            compile_key,
            self.use_text_attention_mask,
            bucket_key,
            tensor_key,
            grid_key,
            layout_key,
        )
        cache_manifest_key = self._compile_runtime.persistent_cache_key(
            persistent_execution_key,
            inputs,
            resolved_attention_backend=resolved,
        )
        return (
            self.profile,
            backend_key,
            compile_key,
            cache_manifest_key,
            self.use_text_attention_mask,
            bucket_key,
            tensor_key,
            grid_key,
            layout_key,
        )

    @property
    def torch_compile_enabled(self) -> bool:
        return self._compile_runtime.enabled

    def _execution_callable(self, key: tuple, inputs: tuple[torch.Tensor, ...]):
        resolved = self.forward.attention_backend_stats()["resolved"]
        return self._compile_runtime.get_callable(
            self.forward,
            key,
            inputs,
            resolved_attention_backend=str(resolved),
            persistent_execution_key=key[:3] + key[4:],
        )

    def captured_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        with self._execution_lock:
            return self._captured_logits_unlocked(encoded)

    def _captured_logits_unlocked(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.native_inputs(encoded)
        if not inputs[0].is_cuda:
            raise RuntimeError("manual CUDA Graph requires CUDA tensors")
        key = self.graph_key(encoded, inputs)
        entry = self._manual_graph_cache.get(key)
        if entry is None:
            if len(self._manual_graph_cache) >= 8:
                raise RuntimeError("manual CUDA Graph shape cache is full (maximum 8 entries)")
            static_inputs = tuple(
                torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device)
                for tensor in inputs
            )
            for target, source in zip(static_inputs, inputs, strict=True):
                target.copy_(source)
            execute = self._execution_callable(key, static_inputs)
            device = inputs[0].device
            current_stream = torch.cuda.current_stream(device)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream), torch.inference_mode():
                for _ in range(3):
                    execute(*static_inputs)
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            started = time.perf_counter_ns()
            with torch.inference_mode(), torch.cuda.graph(graph):
                static_logits = execute(*static_inputs)
            torch.cuda.synchronize(device)
            entry = CapturedNextTokenGraph(
                graph=graph,
                static_inputs=static_inputs,
                static_logits=static_logits,
                shape_key=key,
                capture_ms=(time.perf_counter_ns() - started) / 1000000.0,
                compiled_callable=execute if self.torch_compile_enabled else None,
            )
            self._manual_graph_cache[key] = entry
            self._manual_graph_captures += 1
            entry.graph.replay()
            entry.replay_count += 1
            self._manual_graph_replays += 1
        else:
            for target, source in zip(entry.static_inputs, inputs, strict=True):
                target.copy_(source)
            entry.graph.replay()
            entry.replay_count += 1
            self._manual_graph_replays += 1
        return entry.static_logits.clone()

    def stats(self) -> dict[str, object]:
        with self._execution_lock:
            return self._stats_unlocked()

    def _stats_unlocked(self) -> dict[str, object]:
        entries = [
            {
                "shape_key": list(entry.shape_key),
                "capture_ms": entry.capture_ms,
                "replay_count": entry.replay_count,
            }
            for entry in self._manual_graph_cache.values()
        ]
        return {
            "capture_count": self._manual_graph_captures,
            "replay_count": self._manual_graph_replays,
            "cache_entries": len(entries),
            "entries": entries,
            "attention_backend": self.forward.attention_backend_stats(),
            "torch_compile": self._compile_runtime.stats(),
        }


__all__ = ["CapturedNextTokenGraph", "QwenR2RLowGraphRuntime"]
