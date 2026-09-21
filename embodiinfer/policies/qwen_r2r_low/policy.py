"""VVLA decoder, policy adapter, and registered Low builder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ...types import DecodeTrace
from ..base import MemoryState, VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import AutoregressiveDecoder, DecodeResult
from ..factory import register_policy
from .contract import (
    QwenR2RLowBatch,
    QwenR2RLowMemory,
    QwenR2RLowPrefix,
    parse_low_action,
)
from .runner import QwenR2RLowRunner


class QwenR2RLowDecoder(AutoregressiveDecoder):
    def __init__(self, policy: QwenR2RLowPolicy):
        self.policy = policy

    def decode(self, state, prefix, num_steps, bucket, graphs, *, generator=None, cancelled=None):
        del state, num_steps, bucket, graphs
        if cancelled is not None and cancelled():
            from ...exceptions import SessionCancelledError

            raise SessionCancelledError("navigation request cancelled")
        cache_infer = getattr(self.policy.runner, "infer_with_history_entry", None)
        inference = (
            cache_infer(prefix.observation, prefix.memory)
            if callable(cache_infer)
            else self.policy.runner.infer(prefix.observation, prefix.memory)
        )
        if len(inference) == 4:
            text, token_ids, entropies, current_cache_entry = inference
        elif len(inference) == 3:
            text, token_ids, entropies = inference
            current_cache_entry = None
        else:
            raise RuntimeError("qwen_r2r_low runner returned an invalid inference row")
        actions = parse_low_action(text)
        frame = prefix.observation.images[0].detach().cpu().clone()
        frames = (*prefix.memory.frames, frame)
        responses = (*prefix.memory.responses, text)
        commit_cache = getattr(self.policy.runner, "commit_history_image_cache", None)
        history_image_cache = (
            commit_cache(prefix.memory, current_cache_entry)
            if current_cache_entry is not None and callable(commit_cache)
            else prefix.memory.history_image_cache.advance_without_entry(
                frame_index=len(prefix.memory.frames)
            )
        )
        memory = QwenR2RLowMemory(frames, responses, history_image_cache)
        trace = DecodeTrace(
            token_ids=token_ids.detach().long().cpu(),
            text=text,
            parsed_actions=actions.tolist(),
            stop_reason="model",
            meta={
                "profile": self.policy.profile,
                "runtime_mode": getattr(
                    self.policy.runner,
                    "runtime_mode",
                    "manual_cudagraph" if getattr(self.policy, "cuda_graph_enabled", False) else "eager",
                ),
                "compile_active": getattr(self.policy.runner, "compile_active", False),
                "compile_inactive_reason": getattr(self.policy.runner, "compile_inactive_reason", None),
                "cuda_graph_requested": self.policy.cuda_graph_requested,
                "cuda_graph_confirmed": self.policy.cuda_graph_enabled,
                "token_entropies": entropies,
            },
        )
        return DecodeResult(actions=actions.unsqueeze(0), next_memory=memory, traces=[trace])


class QwenR2RLowPolicy(VLAPolicy):
    def __init__(self, name: str, runner: QwenR2RLowRunner, profile: str):
        if profile != "low_level":
            raise ValueError("QwenR2RLowPolicy only accepts profile low_level")
        horizon = 1
        super().__init__(
            VLAPolicyConfig(
                name=name, action_dim=2, action_horizon=horizon, default_num_steps=1, dtype="bfloat16"
            )
        )
        self.runner = runner
        self.model = runner.model
        self.profile = profile
        self.cuda_graph_enabled = False
        self.cuda_graph_requested = False
        self._decoder = QwenR2RLowDecoder(self)

    @property
    def is_recurrent(self):
        return True

    @property
    def manages_cuda_graph(self):
        return True

    def configure_runtime(self, *, use_cuda_graph: bool):
        self.cuda_graph_requested = use_cuda_graph
        self.cuda_graph_enabled = self.runner.configure_cuda_graph(use_cuda_graph)
        self.model = self.runner.model

    @property
    def decoder(self):
        return self._decoder

    def collate(self, observations, request_ids):
        if len(observations) != 1 or len(request_ids) != 1:
            raise ValueError(f"{self.config.name} requires batch size 1")
        return QwenR2RLowBatch(list(observations), list(request_ids))

    def pad(self, batch, target_batch_size):
        if target_batch_size != 1:
            raise ValueError("navigation policies cannot be padded")
        return batch

    def encode_prefix(self, batch, memory: MemoryState | None = None):
        if not batch.observations[0].instruction:
            raise ValueError("navigation instruction is required")
        if memory is None:
            cache_factory = getattr(self.runner, "new_history_image_cache", None)
            cache = cache_factory() if callable(cache_factory) else None
            memory = QwenR2RLowMemory(history_image_cache=cache) if cache is not None else QwenR2RLowMemory()
        return QwenR2RLowPrefix(batch.observations[0], memory)


def create_qwen_r2r_low_policy(
    name,
    checkpoint,
    max_new_tokens,
    overrides,
    *,
    attention_backend="torch_sdpa",
    compile_backend: Literal["none", "inductor"] = "none",
    compile_cache_dir: str | Path | None = None,
    compile_text_buckets: tuple[int, ...] = (),
    history_image_cache: Literal["none", "rgb_bytes"] = "rgb_bytes",
    load_device: str | None = None,
    tensor_parallel_size: int = 1,
    tensor_parallel_group=None,
):
    if overrides:
        raise ValueError(f"unknown {name} overrides: {sorted(overrides)}")
    if checkpoint is None:
        raise ValueError(f"{name} requires a local checkpoint")
    if not Path(checkpoint).exists():
        raise ValueError(f"checkpoint must be local: {checkpoint}")
    if max_new_tokens != 1:
        raise ValueError(f"{name} is an official single-forward, one-token policy")
    return QwenR2RLowPolicy(
        name,
        QwenR2RLowRunner(
            checkpoint,
            "low_level",
            max_new_tokens=max_new_tokens,
            execute_chunks=1,
            attention_backend=attention_backend,
            compile_backend=compile_backend,
            compile_cache_dir=compile_cache_dir,
            compile_text_buckets=compile_text_buckets,
            history_image_cache=history_image_cache,
            load_device=load_device,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_group=tensor_parallel_group,
        ),
        "low_level",
    )


@register_policy("qwen2.5-vl-3b-r2r-low-level")
def build_qwen_r2r_low(
    checkpoint=None,
    *,
    max_new_tokens=1,
    attention_backend="torch_sdpa",
    compile_backend: Literal["none", "inductor"] = "none",
    compile_cache_dir: str | Path | None = None,
    compile_text_buckets: tuple[int, ...] = (),
    history_image_cache: Literal["none", "rgb_bytes"] = "rgb_bytes",
    load_device: str | None = None,
    tensor_parallel_size: int = 1,
    tensor_parallel_group=None,
    **overrides,
):
    return create_qwen_r2r_low_policy(
        "qwen2.5-vl-3b-r2r-low-level",
        checkpoint,
        max_new_tokens,
        overrides,
        attention_backend=attention_backend,
        compile_backend=compile_backend,
        compile_cache_dir=compile_cache_dir,
        compile_text_buckets=compile_text_buckets,
        history_image_cache=history_image_cache,
        load_device=load_device,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_group=tensor_parallel_group,
    )


__all__ = [
    "QwenR2RLowDecoder",
    "QwenR2RLowPolicy",
    "create_qwen_r2r_low_policy",
    "build_qwen_r2r_low",
]
