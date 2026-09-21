"""Native Qwen2.5-VL-3B R2R panoramic navigation policy."""

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
    qwen25_vl_bucket_schema,
    qwen25_vl_model_config_sha256,
    qwen25_vl_text_bucket_key,
)
from .forward import QwenR2RPanoramicNextTokenForward


@dataclass
class CapturedNextTokenGraph:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_logits: torch.Tensor
    shape_key: tuple
    capture_ms: float
    compiled_callable: object | None = None
    replay_count: int = 0


class QwenR2RPanoramicGraphRuntime:
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
        self.cuda_graph_enabled = False
        self.cuda_graph_requested = False
        requested_buckets = normalize_qwen25_vl_text_buckets(compile_text_buckets)
        self.compile_text_buckets = requested_buckets
        if self.compile_text_buckets and pad_token_id is None:
            raise ValueError("processor.tokenizer.pad_token_id is required for compile text buckets")
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.use_text_attention_mask = bool(self.compile_text_buckets)
        self._unmasked_graph_decoder = QwenR2RPanoramicNextTokenForward(
            model,
            attention_backend=attention_backend,
            use_text_attention_mask=False,
        )
        self._masked_graph_decoder = QwenR2RPanoramicNextTokenForward(
            model,
            attention_backend=attention_backend,
            use_text_attention_mask=True,
        )
        self.graph_decoder = (
            self._masked_graph_decoder if self.use_text_attention_mask else self._unmasked_graph_decoder
        )
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

    def configure_cuda_graph(self, enabled: bool) -> bool:
        with self._execution_lock:
            tensor_parallel = getattr(self, "tensor_parallel", None)
            if enabled and tensor_parallel is not None and tensor_parallel.enabled:
                raise ValueError("Qwen tensor parallelism does not yet support CUDA Graph capture")
            self.cuda_graph_requested = enabled
            device = next(self.model.parameters()).device
            effective = enabled and device.type == "cuda"
            self.cuda_graph_enabled = effective
            return effective

    def _native_inputs(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        with self._execution_lock:
            return self._native_inputs_unlocked(encoded)

    def _native_inputs_unlocked(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        qwen = self.model.model
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if attention_mask.ndim != 2 or not bool(attention_mask[:, -1].detach().all().item()):
            raise ValueError("native/manual graph forward requires the final token to be unmasked")
        self.use_text_attention_mask = bool(self.compile_text_buckets) or not bool(
            attention_mask.detach().ne(0).all().item()
        )
        self.graph_decoder = (
            self._masked_graph_decoder if self.use_text_attention_mask else self._unmasked_graph_decoder
        )
        grid_thw = encoded["image_grid_thw"]
        visual = qwen.visual
        rotary = visual.rot_pos_emb(grid_thw)
        window_index, cu_window = visual.get_window_index(grid_thw)
        window_index = window_index.to(device=input_ids.device)
        reverse_indices = torch.argsort(window_index)
        cu_window_seqlens = torch.unique_consecutive(
            torch.tensor(cu_window, device=input_ids.device, dtype=torch.int32)
        )
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        window_values = cu_window_seqlens.detach().cpu().to(torch.int64).tolist()
        full_values = cu_seqlens.detach().cpu().to(torch.int64).tolist()
        window_bounds = tuple(
            (int(start), int(end))
            for start, end in zip(window_values[:-1], window_values[1:], strict=True)
            if end > start
        )
        full_bounds = tuple(
            (int(start), int(end))
            for start, end in zip(full_values[:-1], full_values[1:], strict=True)
            if end > start
        )
        self.graph_decoder.set_vision_bounds(window_bounds, full_bounds)
        self.graph_decoder.select_attention_backend(
            device=encoded["pixel_values"].device,
            dtype=encoded["pixel_values"].dtype,
        )
        seq_len = encoded["pixel_values"].shape[0]
        unit = visual.spatial_merge_unit
        rotary = rotary.reshape(seq_len // unit, unit, -1)[window_index]
        rotary = rotary.reshape(seq_len, -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        rotary_cos, rotary_sin = rotary.cos(), rotary.sin()
        position_ids, _ = qwen.get_rope_index(
            input_ids,
            encoded["image_grid_thw"],
            encoded.get("video_grid_thw"),
            second_per_grid_ts=encoded.get("second_per_grid_ts"),
            attention_mask=attention_mask,
        )
        return (
            input_ids,
            encoded["pixel_values"],
            attention_mask,
            position_ids,
            window_index,
            reverse_indices,
            cu_window_seqlens,
            cu_seqlens,
            rotary_cos,
            rotary_sin,
        )

    def bucket_encoded(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bucketed, _ = apply_qwen25_vl_text_bucket(
            encoded,
            pad_token_id=self.pad_token_id,
            buckets=self.compile_text_buckets,
        )
        return bucketed

    def _graph_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        with self._execution_lock:
            return self._graph_logits_unlocked(encoded)

    def _graph_logits_unlocked(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self._native_inputs(encoded)
        key = self._manual_graph_key(encoded, inputs)
        execute = self._execution_callable(key, inputs)
        return execute(*inputs)

    def _manual_graph_key(self, encoded: dict[str, torch.Tensor], inputs: tuple[torch.Tensor, ...]) -> tuple:
        with self._execution_lock:
            return self._manual_graph_key_unlocked(encoded, inputs)

    def _manual_graph_key_unlocked(
        self, encoded: dict[str, torch.Tensor], inputs: tuple[torch.Tensor, ...]
    ) -> tuple:
        tensor_key = tuple(
            (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))
            for tensor in inputs
        )
        grid_key = tuple(encoded["image_grid_thw"].detach().cpu().reshape(-1).tolist())
        layout_key = self.graph_decoder.window_bounds, self.graph_decoder.full_bounds
        backend_key = self.graph_decoder.attention_backend_cache_key()
        compile_key = self._compile_runtime.cache_key()
        bucket_key = qwen25_vl_text_bucket_key(encoded, self.compile_text_buckets)
        self._compile_runtime.record_bucket(bucket_key)
        resolved = str(self.graph_decoder.attention_backend_stats()["resolved"])
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
        resolved = self.graph_decoder.attention_backend_stats()["resolved"]
        return self._compile_runtime.get_callable(
            self.graph_decoder,
            key,
            inputs,
            resolved_attention_backend=str(resolved),
            persistent_execution_key=key[:3] + key[4:],
        )

    def _manual_graph_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        with self._execution_lock:
            return self._manual_graph_logits_unlocked(encoded)

    def _manual_graph_logits_unlocked(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self._native_inputs(encoded)
        if not inputs[0].is_cuda:
            raise RuntimeError("manual CUDA Graph requires CUDA tensors")
        key = self._manual_graph_key(encoded, inputs)
        entry = self._manual_graph_cache.get(key)
        if entry is None:
            if len(self._manual_graph_cache) >= 8:
                raise RuntimeError("manual CUDA Graph shape cache is full (maximum 8 entries)")
            static_inputs = tuple(
                torch.empty_strided(
                    tensor.shape,
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
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
                capture_ms=(time.perf_counter_ns() - started) / 1e6,
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

    def manual_graph_stats(self) -> dict[str, object]:
        with self._execution_lock:
            return self._manual_graph_stats_unlocked()

    def _manual_graph_stats_unlocked(self) -> dict[str, object]:
        capture_ms = [entry.capture_ms for entry in self._manual_graph_cache.values()]
        entry_replays = [entry.replay_count for entry in self._manual_graph_cache.values()]
        return {
            "capture_count": self._manual_graph_captures,
            "replay_count": self._manual_graph_replays,
            "cache_entries": len(self._manual_graph_cache),
            "capture_ms": capture_ms,
            "entry_replays": entry_replays,
            "next_token": {
                "capture_count": self._manual_graph_captures,
                "replay_count": self._manual_graph_replays,
                "cache_entries": len(self._manual_graph_cache),
            },
            "attention_backend": self.graph_decoder.attention_backend_stats(),
            "torch_compile": self._compile_runtime.stats(),
        }
