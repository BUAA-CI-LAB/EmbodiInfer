"""Deterministic torch.compile ownership for Qwen2.5-VL next-token forwards."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

import torch

from .compile_cache import Qwen25VLPersistentCompileCache

Qwen25VLCompileBackend = Literal["none", "inductor"]
QWEN25_VL_COMPILE_ABI = "qwen25_vl_next_token_compile_v4"
_SUPPORTED_COMPILE_BACKENDS = ("none", "inductor")
_INDUCTOR_OPTIONS: tuple[tuple[str, bool], ...] = tuple(
    sorted(
        {
            "triton.cudagraphs": False,
            "emulate_precision_casts": True,
        }.items()
    )
)


@dataclass(frozen=True)
class Qwen25VLCompileSelection:
    requested: Qwen25VLCompileBackend
    resolved: Literal["eager", "inductor"]
    fallback_reason: str | None
    compile_abi: str
    backend: str | None
    fullgraph: bool
    dynamic: bool
    inductor_cudagraphs: bool
    emulate_precision_casts: bool
    inductor_options: tuple[tuple[str, bool], ...]
    torch_version: str

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.resolved,
            self.compile_abi,
            self.backend,
            self.fullgraph,
            self.dynamic,
            self.inductor_cudagraphs,
            self.emulate_precision_casts,
            self.inductor_options,
            self.torch_version,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "fallback_reason": self.fallback_reason,
            "compile_abi": self.compile_abi,
            "backend": self.backend,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "inductor_cudagraphs": self.inductor_cudagraphs,
            "emulate_precision_casts": self.emulate_precision_casts,
            "inductor_options": dict(self.inductor_options),
            "torch_version": self.torch_version,
        }


@dataclass
class CompiledQwen25VLForward:
    execution_key: tuple[object, ...]
    compiled_callable: Callable[..., torch.Tensor]
    first_call_wall_ms: float = 0.0
    warmup_calls: int = 0
    calls: int = 0

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.compiled_callable(*inputs)


def normalize_qwen25_vl_compile_backend(value: str) -> Qwen25VLCompileBackend:
    if value not in _SUPPORTED_COMPILE_BACKENDS:
        choices = ", ".join(_SUPPORTED_COMPILE_BACKENDS)
        raise ValueError(f"compile_backend must be one of {choices}; got {value!r}")
    return value  # type: ignore[return-value]


class Qwen25VLCompileRuntime:
    """Own exact-shape compiled callables without eager fallback."""

    def __init__(
        self,
        compile_backend: Qwen25VLCompileBackend = "none",
        *,
        max_entries: int = 8,
        compile_abi: str = QWEN25_VL_COMPILE_ABI,
        target: str | None = None,
        compile_cache_dir: str | Path | None = None,
        cache_identity: Mapping[str, object] | None = None,
    ) -> None:
        requested = normalize_qwen25_vl_compile_backend(compile_backend)
        self._persistent_cache = (
            Qwen25VLPersistentCompileCache(
                compile_cache_dir,
                {
                    "compile_abi": compile_abi,
                    "inductor_options": dict(_INDUCTOR_OPTIONS),
                    "target": target,
                    "runtime": dict(cache_identity or {}),
                },
            )
            if requested == "inductor" and compile_cache_dir is not None
            else None
        )
        if requested == "inductor":
            self._validate_inductor_options(_INDUCTOR_OPTIONS)
        self.selection = Qwen25VLCompileSelection(
            requested=requested,
            resolved="eager" if requested == "none" else "inductor",
            fallback_reason=None,
            compile_abi=compile_abi,
            backend=None if requested == "none" else "inductor",
            fullgraph=requested != "none",
            dynamic=False,
            inductor_cudagraphs=False,
            emulate_precision_casts=True,
            inductor_options=_INDUCTOR_OPTIONS,
            torch_version=str(torch.__version__),
        )
        self.max_entries = max_entries
        self.target = target
        self._entries: dict[tuple[object, ...], CompiledQwen25VLForward] = {}
        self._attempts = 0
        self._failures = 0
        self._discarded_entries = 0
        self._clear_count = 0
        self._retired_direct_callable_calls = 0
        self._bucket_ids: set[tuple[object, ...]] = set()
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.selection.resolved == "inductor"

    def cache_key(self) -> tuple[object, ...]:
        return self.selection.cache_key

    def persistent_cache_key(
        self,
        execution_key: tuple[object, ...],
        example_inputs: tuple[torch.Tensor, ...],
        *,
        resolved_attention_backend: str | None,
    ) -> str | None:
        if not self.enabled or self._persistent_cache is None:
            return None
        return self._persistent_cache.execution_fingerprint(
            execution_key,
            example_inputs,
            resolved_attention_backend=resolved_attention_backend,
        )

    def record_bucket(self, bucket_id: tuple[object, ...]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._bucket_ids.add(bucket_id)

    @staticmethod
    def _validate_inductor_options(
        options: tuple[tuple[str, bool], ...],
    ) -> None:
        try:
            from torch._inductor import list_options

            supported = set(list_options())
        except Exception as exc:
            raise RuntimeError(
                "torch.compile backend 'inductor' was requested, but the "
                "installed PyTorch Inductor option registry is unavailable"
            ) from exc
        missing = sorted(name for name, _ in options if name not in supported)
        if missing:
            raise RuntimeError(
                "torch.compile backend 'inductor' requires unsupported PyTorch "
                f"Inductor option(s): {', '.join(missing)}; "
                f"installed torch={torch.__version__}"
            )

    @property
    def cache_entries(self) -> int:
        with self._lock:
            return len(self._entries)

    def discard(self, execution_key: tuple[object, ...]) -> bool:
        """Discard one quiescent compiled callable and retain lifetime counters."""
        with self._lock:
            entry = self._entries.pop(execution_key, None)
            if entry is None:
                return False
            self._retired_direct_callable_calls += entry.calls
            self._discarded_entries += 1
            return True

    def clear(self) -> int:
        """Release all quiescent compiled callables and retain lifetime counters."""
        with self._lock:
            count = len(self._entries)
            if count == 0:
                return 0
            self._retired_direct_callable_calls += sum(entry.calls for entry in self._entries.values())
            self._discarded_entries += count
            self._clear_count += 1
            self._entries.clear()
            return count

    @staticmethod
    def _synchronize(inputs: tuple[torch.Tensor, ...]) -> None:
        for tensor in inputs:
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
                return

    def get_callable(
        self,
        module: torch.nn.Module,
        execution_key: tuple[object, ...],
        example_inputs: tuple[torch.Tensor, ...],
        *,
        resolved_attention_backend: str | None,
        persistent_execution_key: tuple[object, ...] | None = None,
        before_call: Callable[[], None] | None = None,
    ) -> Callable[..., torch.Tensor]:
        if not self.enabled:
            return module
        cache_execution_key = execution_key if persistent_execution_key is None else persistent_execution_key
        expected_fingerprint = self.persistent_cache_key(
            cache_execution_key,
            example_inputs,
            resolved_attention_backend=resolved_attention_backend,
        )
        if resolved_attention_backend is not None and resolved_attention_backend != "torch_sdpa":
            with self._lock:
                self._failures += 1
            raise RuntimeError(
                "torch.compile supports only resolved torch_sdpa attention; "
                f"got {resolved_attention_backend!r}"
            )
        entry = self._entries.get(execution_key)
        if entry is not None:
            return entry
        with self._lock:
            entry = self._entries.get(execution_key)
            if entry is not None:
                return entry
            if len(self._entries) >= self.max_entries:
                raise RuntimeError(
                    f"Qwen2.5-VL torch.compile cache is full (maximum {self.max_entries} entries)"
                )
        self._attempts += 1
        try:
            lifecycle = (
                self._persistent_cache.compilation_lifecycle(
                    cache_execution_key,
                    example_inputs,
                    resolved_attention_backend=resolved_attention_backend,
                    expected_fingerprint=expected_fingerprint,
                )
                if self._persistent_cache is not None
                else nullcontext(None)
            )
            with lifecycle as persistent_entry:
                compiled = torch.compile(
                    module,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    options=dict(self.selection.inductor_options),
                )
                entry = CompiledQwen25VLForward(execution_key, compiled)
                started = time.perf_counter_ns()
                with torch.inference_mode():
                    if before_call is not None:
                        before_call()
                    entry(*example_inputs)
                    self._synchronize(example_inputs)
                    entry.first_call_wall_ms = (time.perf_counter_ns() - started) / 1e6
                with torch.inference_mode():
                    for _ in range(3):
                        if before_call is not None:
                            before_call()
                        entry(*example_inputs)
                        entry.warmup_calls += 1
                    self._synchronize(example_inputs)
                if self._persistent_cache is not None:
                    assert persistent_entry is not None
                    self._persistent_cache.publish(persistent_entry)
        except Exception:
            self._failures += 1
            raise
        self._entries[execution_key] = entry
        return entry

    def stats(
        self,
        *,
        compile_active: bool | None = None,
        inactive_reason: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            active = self.enabled if compile_active is None else compile_active
            if active:
                reason = None
            elif inactive_reason is not None:
                reason = inactive_reason
            elif not self.enabled:
                reason = "compile_backend_none"
            else:
                reason = "compile_not_active"
            active_direct_calls = sum(entry.calls for entry in self._entries.values())
            direct_calls = self._retired_direct_callable_calls + active_direct_calls
            persistent_cache = (
                self._persistent_cache.stats()
                if self._persistent_cache is not None
                else Qwen25VLPersistentCompileCache.unconfigured_stats()
            )
            result = self.selection.as_dict()
            if self.target is not None:
                result["target"] = self.target
            result.update(
                {
                    "compile_active": active,
                    "inactive_reason": reason,
                    "cache_entries": len(self._entries),
                    "attempts": self._attempts,
                    "failures": self._failures,
                    "direct_callable_calls": direct_calls,
                    "calls": direct_calls,
                    "calls_semantics": ("direct_callable_calls_only_excludes_cuda_graph_replays"),
                    "discarded_entries": self._discarded_entries,
                    "clear_count": self._clear_count,
                    "bucket_ids": [list(bucket_id) for bucket_id in sorted(self._bucket_ids, key=repr)],
                    "persistent_cache": persistent_cache,
                    "entries": [
                        {
                            "execution_key": entry.execution_key,
                            "first_call_wall_ms": entry.first_call_wall_ms,
                            "warmup_calls": entry.warmup_calls,
                            "direct_callable_calls": entry.calls,
                            "calls": entry.calls,
                        }
                        for entry in self._entries.values()
                    ],
                }
            )
            return result


__all__ = [
    "QWEN25_VL_COMPILE_ABI",
    "CompiledQwen25VLForward",
    "Qwen25VLCompileBackend",
    "Qwen25VLCompileRuntime",
    "Qwen25VLCompileSelection",
    "normalize_qwen25_vl_compile_backend",
]
