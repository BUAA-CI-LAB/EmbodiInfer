"""EmbodiInfer policy, decoder, builder, and registry integration for NaViDA."""

from __future__ import annotations

from pathlib import Path

from ...types import DecodeTrace
from ..base import MemoryState, VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import AutoregressiveDecoder, DecodeResult
from ..factory import register_policy
from .contract import (
    NAVIDA_MAX_ATOMIC_ACTIONS,
    NAVIDA_MAX_STORED_FRAMES,
    NaViDABatch,
    NaViDAMemory,
    NaViDAPrefix,
    parse_navida_actions,
)
from .runner import NaViDARunner


class NaViDADecoder(AutoregressiveDecoder):
    def __init__(self, policy: NaViDAPolicy):
        self.policy = policy

    def decode(
        self,
        state,
        prefix,
        num_steps,
        bucket,
        graphs,
        *,
        generator=None,
        cancelled=None,
    ):
        del state, num_steps, bucket, graphs, generator
        if cancelled is not None and cancelled():
            from ...exceptions import SessionCancelledError

            raise SessionCancelledError("navigation request cancelled")
        text, token_ids, entropies = self.policy.runner.infer(prefix.observation, prefix.memory)
        actions = parse_navida_actions(text, self.policy.runner.execute_chunks)
        frame = prefix.observation.images[0].detach().cpu()
        frames = (*prefix.memory.frames, frame)[-NAVIDA_MAX_STORED_FRAMES:]
        memory = NaViDAMemory(frames=frames, responses=())
        trace = DecodeTrace(
            token_ids=token_ids.detach().long().cpu(),
            text=text,
            parsed_actions=actions.tolist(),
            stop_reason="model",
            meta={
                "profile": "navida",
                "runtime_mode": ("manual_cudagraph" if self.policy.cuda_graph_enabled else "eager"),
                "cuda_graph_requested": self.policy.cuda_graph_requested,
                "cuda_graph_confirmed": self.policy.cuda_graph_enabled,
                "token_entropies": entropies,
            },
        )
        return DecodeResult(actions=actions.unsqueeze(0), next_memory=memory, traces=[trace])


class NaViDAPolicy(VLAPolicy):
    def __init__(self, name: str, runner: NaViDARunner):
        super().__init__(
            VLAPolicyConfig(
                name=name,
                action_dim=2,
                action_horizon=NAVIDA_MAX_ATOMIC_ACTIONS,
                default_num_steps=1,
                dtype="bfloat16",
            )
        )
        self.runner = runner
        self.model = runner.model
        self.profile = "navida"
        self.cuda_graph_enabled = False
        self.cuda_graph_requested = False
        self._decoder = NaViDADecoder(self)

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
        return NaViDABatch(list(observations), list(request_ids))

    def pad(self, batch, target_batch_size):
        if target_batch_size != 1:
            raise ValueError("navigation policies cannot be padded")
        return batch

    def encode_prefix(self, batch, memory: MemoryState | None = None):
        if not batch.observations[0].instruction:
            raise ValueError("navigation instruction is required")
        return NaViDAPrefix(batch.observations[0], memory or NaViDAMemory())


def _build(
    name,
    checkpoint,
    max_new_tokens,
    execute_chunks,
    overrides,
    *,
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
    if execute_chunks != 2:
        raise ValueError("NaViDA v2 executes exactly the first two predicted chunks")
    return NaViDAPolicy(
        name,
        NaViDARunner(
            checkpoint,
            max_new_tokens=max_new_tokens,
            execute_chunks=execute_chunks,
            load_device=load_device,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_group=tensor_parallel_group,
        ),
    )


@register_policy("navida")
def build_navida(
    checkpoint=None,
    *,
    max_new_tokens=512,
    execute_chunks=2,
    compile_backend="none",
    load_device: str | None = None,
    tensor_parallel_size: int = 1,
    tensor_parallel_group=None,
    **overrides,
):
    if compile_backend != "none":
        raise ValueError(
            "NaViDA compile_backend must be 'none': stochastic history parity "
            "has not been admitted for torch.compile/Inductor"
        )
    return _build(
        "navida",
        checkpoint,
        max_new_tokens,
        execute_chunks,
        overrides,
        load_device=load_device,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_group=tensor_parallel_group,
    )
