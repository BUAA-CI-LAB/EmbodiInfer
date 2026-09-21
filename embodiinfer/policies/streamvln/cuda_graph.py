"""CUDA Graph runner for a fixed StreamVLN autoregressive token block."""

from __future__ import annotations

import torch

from ...backend.triton import (
    GreedyWorkspace,
    fused_greedy,
    graph_gqa_available,
    mark_penalized,
)
from ...models.streamvln import StreamVLNBackbone, StreamVLNCache


class StreamVLNDecodeGraph:
    """Capture several dependent decode tokens as one graph replay.

    ``block_size`` tokens are appended to the fixed-address KV storage. For an
    intermediate block, the graph also selects the token that seeds the next
    block; a terminal block omits that final LM-head/sampling pass. EOS remains a
    logical length decision made by the decoder after one synchronization per
    block. Physical KV writes after an early EOS are overwritten from the
    committed logical length by the next turn.
    """

    def __init__(
        self,
        backbone: StreamVLNBackbone,
        cache: StreamVLNCache,
        workspace: GreedyWorkspace,
        repetition_penalty: float,
        *,
        block_size: int = 4,
        produce_next: bool = True,
    ) -> None:
        if block_size <= 0:
            raise ValueError("StreamVLN decode graph block_size must be positive")
        if not self.is_available() or not cache.key_storage.is_cuda:
            raise RuntimeError("StreamVLN CUDA Graph requires CUDA and Triton")
        self.backbone = backbone
        self.block_size = int(block_size)
        self.produce_next = bool(produce_next)
        self.key_pointer = cache.key_storage.data_ptr()
        self.value_pointer = cache.value_storage.data_ptr()
        self.workspace_pointer = workspace.penalized.data_ptr()
        self.workspace = workspace
        self.repetition_penalty = repetition_penalty
        device = cache.key_storage.device
        self.input_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.input_logprob = torch.zeros((1, 1), dtype=torch.float32, device=device)
        self.position = torch.tensor([cache.seq_len], dtype=torch.long, device=device)
        self.output_tokens = torch.empty((1, self.block_size), dtype=torch.long, device=device)
        self.output_logprobs = torch.empty(
            (1, self.block_size),
            dtype=torch.float32,
            device=device,
        )
        self.next_token = torch.empty((1, 1), dtype=torch.long, device=device) if self.produce_next else None
        self.next_logprob = (
            torch.empty((1, 1), dtype=torch.float32, device=device) if self.produce_next else None
        )

        saved_penalized = workspace.penalized.clone()
        current = torch.cuda.current_stream(device)
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                self._run(cache)
        current.wait_stream(warmup)
        current.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self._run(cache)
        workspace.penalized.copy_(saved_penalized)

    def _run(self, cache: StreamVLNCache) -> None:
        current_token = self.input_token
        current_logprob = self.input_logprob
        for index in range(self.block_size):
            self.output_tokens[:, index : index + 1].copy_(current_token)
            self.output_logprobs[:, index : index + 1].copy_(current_logprob)
            mark_penalized(self.workspace, current_token)
            position = self.position + index
            hidden = self.backbone.decode_token_graph(cache, current_token, position)
            needs_selection = index + 1 < self.block_size or self.produce_next
            if not needs_selection:
                continue
            logits = self.backbone.project_logits(hidden)
            selected_token, selected_logprob = fused_greedy(
                logits,
                self.workspace,
                self.repetition_penalty,
            )
            if index + 1 < self.block_size:
                current_token = selected_token
                current_logprob = selected_logprob
            else:
                if self.next_token is None or self.next_logprob is None:
                    raise RuntimeError("intermediate StreamVLN graph has no next-token buffers")
                self.next_token.copy_(selected_token)
                self.next_logprob.copy_(selected_logprob)

    @staticmethod
    def is_available() -> bool:
        return graph_gqa_available()

    def matches(self, cache: StreamVLNCache, workspace: GreedyWorkspace) -> bool:
        return (
            cache.key_storage.data_ptr() == self.key_pointer
            and cache.value_storage.data_ptr() == self.value_pointer
            and workspace.penalized.data_ptr() == self.workspace_pointer
        )

    def replay(
        self,
        cache: StreamVLNCache,
        token: torch.Tensor,
        logprob: torch.Tensor,
        workspace: GreedyWorkspace,
    ) -> tuple[
        StreamVLNCache,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.matches(cache, workspace):
            raise ValueError("CUDA Graph cache storage changed")
        if cache.seq_len + self.block_size > cache.capacity:
            raise RuntimeError("StreamVLN cache capacity exceeded")
        self.input_token.copy_(token.reshape(1, 1))
        self.input_logprob.copy_(logprob.reshape(1, 1))
        self.position.fill_(cache.seq_len)
        self.graph.replay()
        return (
            cache.advance(cache.seq_len + self.block_size),
            self.output_tokens.clone(),
            self.output_logprobs.clone(),
            None if self.next_token is None else self.next_token.clone(),
            None if self.next_logprob is None else self.next_logprob.clone(),
        )


__all__ = ["StreamVLNDecodeGraph"]
