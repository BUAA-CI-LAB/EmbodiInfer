"""Self-hosted StreamVLN policy with transactional SlowFast episode state."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ...backend.triton import (
    GreedyWorkspace,
    fused_greedy,
    mark_penalized,
    supports_fused_greedy,
)
from ...exceptions import SessionCancelledError
from ...layers.linear import INT8Config, QuantizationConfig, parse_quantization_config
from ...models.linear import quantize_linear
from ...models.streamvln import StreamVLNBackbone, StreamVLNCache, load_streamvln_checkpoint
from ...types import DecodeTrace, Observation
from ..base import VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import AutoregressiveDecoder, DecodeResult
from ..factory import register_policy
from .cuda_graph import StreamVLNDecodeGraph, StreamVLNVerificationGraph
from .processing import StreamVLNBatch, StreamVLNInputs, StreamVLNProcessor
from .prompt import STOP, VISION_PHRASE, parse_symbolic_actions

if TYPE_CHECKING:
    from ...engine.core import EngineCore
    from ...engine.serve.contracts import ServingAdapter
    from .batching import StreamVLNBatchedRuntime


@dataclass(frozen=True)
class StreamVLNMemory:
    cache: StreamVLNCache | None
    frame_bank: tuple[torch.Tensor, ...]
    step_id: int
    instruction: str
    frame_features: tuple[torch.Tensor | None, ...] = ()
    slow_frame_indices: tuple[int, ...] = ()
    pending_token_ids: tuple[int, ...] = ()

    @property
    def seq_len(self) -> int:
        return 0 if self.cache is None else self.cache.seq_len

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> StreamVLNMemory:
        cache = None if self.cache is None else self.cache.to(device, dtype)
        # Host-offloaded features stay on the host. Opt-in device features move
        # with the session, without borrowing graph-owned output storage.
        features = tuple(
            feature.to(device=device, dtype=dtype or feature.dtype)
            if feature is not None and feature.is_cuda
            else feature
            for feature in self.frame_features
        )
        return replace(self, cache=cache, frame_features=features)


@dataclass(frozen=True)
class StreamVLNPrefix:
    memory: StreamVLNMemory
    next_hidden: torch.Tensor
    response_prefix_ids: torch.Tensor | None = None
    response_prefix_hiddens: torch.Tensor | None = None
    reference_cache_length: int | None = None
    batch_size: int = 1

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> StreamVLNPrefix:
        return StreamVLNPrefix(
            memory=self.memory.to(device, dtype),
            next_hidden=self.next_hidden.to(
                device=device,
                dtype=dtype or self.next_hidden.dtype,
            ),
            response_prefix_ids=(
                None if self.response_prefix_ids is None else self.response_prefix_ids.to(device=device)
            ),
            response_prefix_hiddens=(
                None
                if self.response_prefix_hiddens is None
                else self.response_prefix_hiddens.to(
                    device=device,
                    dtype=dtype or self.response_prefix_hiddens.dtype,
                )
            ),
            reference_cache_length=self.reference_cache_length,
        )

    def expand(self, num_samples: int) -> StreamVLNPrefix:
        if num_samples != 1:
            raise ValueError("StreamVLN recurrent decoding supports one branch per replica")
        return self


@dataclass(frozen=True)
class StreamVLNPreparedPrefix:
    """Preprocessed current input and committed history, ready for model encoding."""

    committed: StreamVLNMemory
    frame_bank: tuple[torch.Tensor, ...]
    inputs: StreamVLNInputs
    base_cache: StreamVLNCache | None
    input_ids: torch.Tensor
    current_pixels: torch.Tensor
    memory_pixels: torch.Tensor | None
    position_shift: int
    use_response_template: bool
    response_prefix_ids: torch.Tensor | None
    intermediate_pixels: tuple[tuple[int, torch.Tensor], ...] = ()
    stream_stride: int = 1


@dataclass(frozen=True)
class StreamVLNGeneration:
    """Completed token generation before text decoding and action parsing."""

    cache: StreamVLNCache
    token_ids: torch.Tensor
    token_logprobs: torch.Tensor
    stop_reason: str
    pending_token_ids: tuple[int, ...]
    used_response_template: bool


_ResponseTemplateResult = tuple[
    StreamVLNCache,
    list[torch.Tensor],
    list[torch.Tensor],
    tuple[int, ...],
]


class StreamVLNDecoder(AutoregressiveDecoder):
    def __init__(self, policy: StreamVLNPolicy) -> None:
        self.policy = policy
        self.decode_block_size = policy.decode_block_size
        self._workspace: GreedyWorkspace | None = None
        self._cuda_graphs: dict[tuple[int, bool], StreamVLNDecodeGraph] = {}
        self._verification_graphs: dict[int, StreamVLNVerificationGraph] = {}
        self._tree_runtime = None
        self._startup_capture_active = False
        self._startup_capture_frozen = False
        self.graph_replays = 0
        self.eager_fallbacks = 0
        self.speculation_stats = dict.fromkeys(
            (
                "speculative_blocks",
                "speculative_proposed_tokens",
                "speculative_accepted_tokens",
                "speculative_fallback_tokens",
                "speculative_prefix_accepts",
                "speculative_graph_replays",
                "speculative_eager_fallbacks",
            ),
            0,
        )

    def begin_startup_graph_capture(self) -> None:
        if self._startup_capture_active:
            raise RuntimeError("StreamVLN startup graph capture is already active")
        if self._startup_capture_frozen:
            raise RuntimeError("StreamVLN startup graph capture is already frozen")
        self._startup_capture_active = True

    def finish_startup_graph_capture(self) -> None:
        if not self._startup_capture_active:
            raise RuntimeError("StreamVLN startup graph capture is not active")
        self._startup_capture_active = False
        self._startup_capture_frozen = True

    def abort_startup_graph_capture(self) -> None:
        self._cuda_graphs.clear()
        self._verification_graphs.clear()
        self._tree_runtime = None
        self._startup_capture_active = False
        self._startup_capture_frozen = False

    def reset_cuda_graph_runtime(self) -> None:
        self._cuda_graphs.clear()
        self._verification_graphs.clear()
        self._tree_runtime = None
        self._startup_capture_active = False
        self._startup_capture_frozen = False

    def reset_runtime_stats(self) -> None:
        self.graph_replays = 0
        self.eager_fallbacks = 0
        for name in self.speculation_stats:
            self.speculation_stats[name] = 0
        if self._tree_runtime is not None:
            for name in self._tree_runtime.counters:
                self._tree_runtime.counters[name] = 0

    def graph_capture_stats(self) -> dict[str, int | bool]:
        return {
            "active": self._startup_capture_active,
            "frozen": self._startup_capture_frozen,
            "decode": len(self._cuda_graphs),
            "verification": len(self._verification_graphs),
            "tree": 0 if self._tree_runtime is None else len(self._tree_runtime.graphs),
        }

    def _get_workspace(self, hidden: torch.Tensor) -> GreedyWorkspace:
        vocab_size = self.policy.backbone.llm.lm_head.weight.shape[0]
        if (
            self._workspace is None
            or self._workspace.vocab_size != vocab_size
            or self._workspace.penalized.device != hidden.device
        ):
            if self._startup_capture_frozen:
                raise RuntimeError("StreamVLN decode workspace changed after startup graph capture")
            self._workspace = GreedyWorkspace.allocate(vocab_size, hidden.device)
            self._cuda_graphs.clear()
            self._verification_graphs.clear()
            self._tree_runtime = None
        self._workspace.reset()
        return self._workspace

    def verify_candidate_tokens(self, cache: StreamVLNCache, tokens: torch.Tensor) -> torch.Tensor:
        """Verify a causal block, replaying graphs captured only during startup."""
        width = tokens.shape[1]
        graph = self._verification_graphs.get(width)
        if graph is None or not graph.matches(cache):
            if self._startup_capture_active:
                graph = StreamVLNVerificationGraph(self.policy.backbone, cache, width)
                self._verification_graphs[width] = graph
            else:
                self.speculation_stats["speculative_eager_fallbacks"] += 1
                position = torch.tensor([cache.seq_len], dtype=torch.long, device=tokens.device)
                return self.policy.backbone.decode_tokens_graph(cache, tokens, position)
        self.speculation_stats["speculative_graph_replays"] += 1
        return graph.replay(cache, tokens)

    def _append_token(
        self,
        cache: StreamVLNCache,
        token: torch.Tensor,
        workspace: GreedyWorkspace,
        generated: list[torch.Tensor],
        *,
        need_next: bool,
    ) -> tuple[StreamVLNCache, torch.Tensor | None, torch.Tensor | None]:
        cache, hidden = self.policy.append_token(cache, token)
        if not need_next:
            return cache, None, None
        next_token, next_logprob = self._select_token(hidden, workspace, generated)
        return cache, next_token, next_logprob

    def _get_cuda_graph(
        self,
        cache: StreamVLNCache,
        workspace: GreedyWorkspace,
        block_size: int,
        produce_next: bool,
    ) -> StreamVLNDecodeGraph | None:
        key = (block_size, produce_next)
        graph = self._cuda_graphs.get(key)
        if graph is None or not graph.matches(cache, workspace):
            if not self._startup_capture_active:
                self.eager_fallbacks += 1
                return None
            graph = StreamVLNDecodeGraph(
                self.policy.backbone,
                cache,
                workspace,
                self.policy.repetition_penalty,
                block_size=block_size,
                produce_next=produce_next,
            )
            self._cuda_graphs[key] = graph
        self.graph_replays += 1
        return graph

    def _decode_remaining_eager(
        self,
        cache: StreamVLNCache,
        token: torch.Tensor,
        logprob: torch.Tensor,
        workspace: GreedyWorkspace,
        generated: list[torch.Tensor],
        logprobs: list[torch.Tensor],
        remaining: int,
        *,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[StreamVLNCache, list[torch.Tensor], list[torch.Tensor], str]:
        stop_reason = "max_tokens"
        for step in range(remaining):
            if cancelled is not None and cancelled():
                raise SessionCancelledError("StreamVLN generation was cancelled")
            current_token = token.clone()
            current_logprob = logprob.clone()
            generated.append(current_token)
            logprobs.append(current_logprob)
            workspace.penalized[current_token.reshape(-1)] = True
            is_eos = int(current_token.item()) in self.policy.eos_token_ids
            need_next = not is_eos and step + 1 < remaining
            cache, token, logprob = self._append_token(
                cache,
                current_token,
                workspace,
                generated,
                need_next=need_next,
            )
            if is_eos:
                stop_reason = "eos"
                break
            if need_next and (token is None or logprob is None):
                raise RuntimeError("StreamVLN eager decode did not produce its continuation token")
        return cache, generated, logprobs, stop_reason

    def _decode_cuda_blocks(
        self,
        cache: StreamVLNCache,
        token: torch.Tensor,
        logprob: torch.Tensor,
        workspace: GreedyWorkspace,
        *,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[StreamVLNCache, list[torch.Tensor], list[torch.Tensor], str]:
        generated: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        # fused_greedy returns workspace-owned static buffers. A first-time graph
        # capture also samples during warmup and would otherwise overwrite the
        # initial token before the real replay consumes it.
        token = token.clone()
        logprob = logprob.clone()
        remaining = self.policy.max_new_tokens
        stop_reason = "max_tokens"
        while remaining:
            if cancelled is not None and cancelled():
                raise SessionCancelledError("StreamVLN generation was cancelled")
            available = cache.capacity - cache.seq_len
            if available <= 0:
                raise RuntimeError("StreamVLN cache capacity exceeded")
            block_size = min(self.decode_block_size, remaining, available)
            produce_next = remaining > block_size
            graph = self._get_cuda_graph(
                cache,
                workspace,
                block_size,
                produce_next,
            )
            if graph is None:
                return self._decode_remaining_eager(
                    cache,
                    token,
                    logprob,
                    workspace,
                    generated,
                    logprobs,
                    remaining,
                    cancelled=cancelled,
                )
            block_start = cache.seq_len
            advanced, block_tokens, block_logprobs, next_token, next_logprob = graph.replay(
                cache,
                token,
                logprob,
                workspace,
            )
            token_values = block_tokens[0].tolist()
            if cancelled is not None and cancelled():
                raise SessionCancelledError("StreamVLN generation was cancelled")
            used = block_size
            for index, value in enumerate(token_values):
                if value in self.policy.eos_token_ids:
                    used = index + 1
                    stop_reason = "eos"
                    break
            generated.append(block_tokens[:, :used])
            logprobs.append(block_logprobs[:, :used])
            if stop_reason == "eos":
                cache = advanced.advance(block_start + used)
                break
            cache = advanced
            remaining -= block_size
            if not remaining:
                break
            if next_token is None or next_logprob is None:
                raise RuntimeError("StreamVLN decode block did not produce its continuation token")
            token, logprob = next_token, next_logprob
        return cache, generated, logprobs, stop_reason

    def _verify_response_prefix(
        self,
        prefix: StreamVLNPrefix,
        workspace: GreedyWorkspace,
        *,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]] | None:
        """Accept the prefilled assistant envelope only after full-vocabulary checks."""
        expected = prefix.response_prefix_ids
        hiddens = prefix.response_prefix_hiddens
        if expected is None or hiddens is None or hiddens.ndim != 3:
            return None
        length = expected.numel()
        if length >= self.policy.max_new_tokens or hiddens.shape[1] != length + 1:
            return None
        expected = expected.reshape(1, length)
        generated, logprobs = [], []
        for index in range(length):
            if cancelled is not None and cancelled():
                raise SessionCancelledError("StreamVLN generation was cancelled")
            token, score = self._select_token(hiddens[:, index], workspace, generated)
            generated.append(token.clone())
            logprobs.append(score.clone())
            mark_penalized(workspace, expected[:, index : index + 1])
        if torch.cat(generated, dim=1).tolist() != expected.tolist():
            return None
        return generated, logprobs

    def _decode_response_template(
        self,
        prefix: StreamVLNPrefix,
        workspace: GreedyWorkspace,
        *,
        cancelled: Callable[[], bool] | None,
    ) -> _ResponseTemplateResult | None:
        """Decode the verified fixed response envelope and four action tokens.

        The response-prefix tokens are still selected from the full vocabulary
        with the normal repetition penalty. The fast path is accepted only
        when every prefix token, every action token, and the terminal EOS match
        the published response grammar; otherwise the caller rolls the logical
        cache length back and executes the reference decoder.
        """

        expected_prefix = prefix.response_prefix_ids
        prefix_hiddens = prefix.response_prefix_hiddens
        cache = prefix.memory.cache
        if expected_prefix is None or prefix_hiddens is None or cache is None:
            return None
        prefix_length = int(expected_prefix.numel())
        horizon = int(self.policy.config.action_horizon)
        if (
            prefix_length + horizon + 1 > self.policy.max_new_tokens
            or not self.policy.action_token_ids
            or prefix_hiddens.ndim != 3
            or prefix_hiddens.shape[1] != prefix_length + 1
        ):
            return None
        if cancelled is not None and cancelled():
            raise SessionCancelledError("StreamVLN generation was cancelled")

        verified = self._verify_response_prefix(prefix, workspace, cancelled=cancelled)
        if verified is None:
            return None
        generated, logprobs = verified

        token, logprob = self._select_token(
            prefix_hiddens[:, -1],
            workspace,
            generated,
        )
        graph = self._get_cuda_graph(
            cache,
            workspace,
            horizon,
            True,
        )
        if graph is None:
            return None
        advanced, action_tokens, action_logprobs, eos_token, eos_logprob = graph.replay(
            cache,
            token.clone(),
            logprob.clone(),
            workspace,
        )
        if eos_token is None or eos_logprob is None:
            raise RuntimeError("StreamVLN response-template graph did not produce EOS")
        if cancelled is not None and cancelled():
            raise SessionCancelledError("StreamVLN generation was cancelled")
        values = torch.cat((action_tokens, eos_token), dim=1)[0].tolist()
        action_values = values[:-1]
        eos_value = values[-1]
        if (
            any(value not in self.policy.action_token_ids for value in action_values)
            or eos_value not in self.policy.eos_token_ids
        ):
            return None

        generated.extend((action_tokens, eos_token))
        logprobs.extend((action_logprobs, eos_logprob))
        return advanced, generated, logprobs, (eos_value,)

    def _select_token(
        self,
        hidden: torch.Tensor,
        workspace: GreedyWorkspace,
        generated: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy.backbone.project_logits(hidden)
        if supports_fused_greedy(logits, workspace):
            return fused_greedy(
                logits,
                workspace,
                self.policy.repetition_penalty,
            )
        scores = self._apply_repetition_penalty(
            logits.float(),
            generated,
            self.policy.repetition_penalty,
        )
        token = scores.argmax(dim=-1, keepdim=True)
        return token, scores.log_softmax(dim=-1).gather(-1, token)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated: list[torch.Tensor],
        penalty: float,
    ) -> torch.Tensor:
        if penalty == 1.0 or not generated:
            return logits
        adjusted = logits.clone()
        token_ids = torch.unique(torch.cat(generated, dim=1))
        selected = adjusted[:, token_ids]
        selected = torch.where(selected < 0, selected * penalty, selected / penalty)
        adjusted[:, token_ids] = selected
        return adjusted

    @torch.inference_mode()
    def decode(
        self,
        state: torch.Tensor | None,
        prefix: StreamVLNPrefix,
        num_steps: int,
        bucket: int,
        graphs,
        *,
        generator: torch.Generator | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DecodeResult:
        del state, num_steps, bucket, graphs, generator
        return self.finalize_generation(prefix, self.generate_tokens(prefix, cancelled=cancelled))

    @torch.inference_mode()
    def generate_tokens(
        self, prefix: StreamVLNPrefix, *, cancelled: Callable[[], bool] | None = None
    ) -> StreamVLNGeneration:
        """Run the complete existing token loop, leaving text/action conversion to the caller."""
        if prefix.memory.cache is None:
            raise RuntimeError("StreamVLN prefix has no fast KV cache")
        if getattr(self.policy, "tree_decode", False):
            from .tree import generate_tree_tokens

            return generate_tree_tokens(self, prefix, cancelled=cancelled)
        cache = prefix.memory.cache
        hidden = prefix.next_hidden
        generated: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        workspace = self._get_workspace(hidden)
        stop_reason = "max_tokens"
        pending_token_ids: tuple[int, ...] = ()
        used_response_template = False
        if getattr(self.policy, "speculative_tokens", 0):
            from .speculation import decode_speculative

            if prefix.reference_cache_length is not None:
                verified = self._verify_response_prefix(prefix, workspace, cancelled=cancelled)
                if verified is None:
                    if prefix.response_prefix_hiddens is None:
                        raise RuntimeError("StreamVLN response-template hidden states are missing")
                    cache = cache.advance(prefix.reference_cache_length)
                    hidden = prefix.response_prefix_hiddens[:, 0]
                    workspace.reset()
                else:
                    generated, logprobs = verified
                    self.speculation_stats["speculative_prefix_accepts"] += 1
            cache, generated, logprobs, stop_reason = decode_speculative(
                self,
                cache,
                hidden,
                workspace,
                cancelled=cancelled,
                initial_tokens=generated,
                initial_logprobs=logprobs,
            )
            return StreamVLNGeneration(
                cache, torch.cat(generated, dim=1), torch.cat(logprobs, dim=1), stop_reason, (), False
            )
        if (
            self.policy.fast_action_decode
            and self.policy.cuda_graph
            and hidden.is_cuda
            and StreamVLNDecodeGraph.is_available()
        ):
            template_result = self._decode_response_template(
                prefix,
                workspace,
                cancelled=cancelled,
            )
            if template_result is not None:
                cache, generated, logprobs, pending_token_ids = template_result
                stop_reason = "eos"
                used_response_template = True

        if not used_response_template:
            if prefix.reference_cache_length is not None:
                cache = cache.advance(prefix.reference_cache_length)
                if prefix.response_prefix_hiddens is None:
                    raise RuntimeError("StreamVLN response-template hidden states are missing")
                hidden = prefix.response_prefix_hiddens[:, 0]
                workspace = self._get_workspace(hidden)
            token, logprob = self._select_token(hidden, workspace, generated)
            if self.policy.cuda_graph and token.is_cuda and StreamVLNDecodeGraph.is_available():
                cache, generated, logprobs, stop_reason = self._decode_cuda_blocks(
                    cache,
                    token,
                    logprob,
                    workspace,
                    cancelled=cancelled,
                )
            else:
                cache, generated, logprobs, stop_reason = self._decode_remaining_eager(
                    cache,
                    token,
                    logprob,
                    workspace,
                    generated,
                    logprobs,
                    self.policy.max_new_tokens,
                    cancelled=cancelled,
                )

        device = prefix.next_hidden.device
        token_ids = (
            torch.cat(generated, dim=1) if generated else torch.empty((1, 0), device=device, dtype=torch.long)
        )
        token_logprobs = (
            torch.cat(logprobs, dim=1)
            if logprobs
            else torch.empty((1, 0), device=device, dtype=torch.float32)
        )
        return StreamVLNGeneration(
            cache, token_ids, token_logprobs, stop_reason, pending_token_ids, used_response_template
        )

    def finalize_generation(self, prefix: StreamVLNPrefix, generation: StreamVLNGeneration) -> DecodeResult:
        """Decode completed tokens and build the same action chunk, trace and next memory."""
        cache, token_ids = generation.cache, generation.token_ids
        token_logprobs, stop_reason = generation.token_logprobs, generation.stop_reason
        pending_token_ids = generation.pending_token_ids
        used_response_template = generation.used_response_template
        device = prefix.next_hidden.device
        text = self.policy.processor.tokenizer.decode(
            token_ids[0].tolist(),
            skip_special_tokens=True,
        ).strip()
        parsed = parse_symbolic_actions(text, horizon=self.policy.config.action_horizon)
        actions = parsed.actions.unsqueeze(0).to(device=device)
        action_mask = parsed.mask.unsqueeze(0).to(device=device)
        if not parsed.invalid and bool(parsed.mask.any()) and int(parsed.actions[parsed.mask][-1, 0]) == STOP:
            stop_reason = "stop"

        next_memory = replace(
            prefix.memory,
            cache=cache,
            pending_token_ids=pending_token_ids,
        )
        trace = DecodeTrace(
            token_ids=token_ids[0],
            token_logprobs=token_logprobs[0],
            action_mask=action_mask[0],
            text=text,
            parsed_actions=actions[0],
            stop_reason=stop_reason,
            meta={
                "invalid_action_text": parsed.invalid,
                "truncated_actions": parsed.truncated,
                "slow_frame_indices": next_memory.slow_frame_indices,
                "stream_step": next_memory.step_id,
                "response_template": used_response_template,
                "runner_profile": "streamvln_habitat_deterministic_phrase",
            },
        )
        return DecodeResult(
            actions=actions,
            behavior_logprob=token_logprobs.sum(dim=1),
            next_memory=next_memory,
            traces=[trace],
        )


class StreamVLNPolicy(VLAPolicy):
    def __init__(
        self,
        backbone: StreamVLNBackbone,
        processor: StreamVLNProcessor,
        *,
        eos_token_ids: tuple[int, ...] = (151645, 151643),
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.05,
        cuda_graph: bool = True,
        decode_block_size: int = 4,
        cache_history_features: bool = True,
        fast_action_decode: bool = True,
        speculative_tokens: int = 0,
        gpu_history_features: bool = False,
        tree_decode: bool = False,
        tree_budget: int = 31,
        tree_refresh: int = 64,
        fused_ops: bool = False,
        split_attention: bool = False,
        attention_kernel: str = "gqa",
        decode_linear: bool = False,
        tree_fp32_projection: bool = False,
        vision_kernel: str = "hf",
        kv_cache_initial_capacity: int = 0,
        quantized_prefill_graphs: bool = False,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if decode_block_size <= 0:
            raise ValueError("decode_block_size must be positive")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if type(tree_refresh) is not int or tree_refresh < 1:
            raise ValueError("tree_refresh must be a positive integer")
        if type(speculative_tokens) is not int or speculative_tokens not in (0, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError("speculative_tokens must be 0 (disabled) or an integer from 2 to 8")
        if gpu_history_features and not cache_history_features:
            raise ValueError("gpu_history_features requires cache_history_features")
        dtype_name = str(next(backbone.parameters()).dtype).removeprefix("torch.")
        if speculative_tokens and (not cuda_graph or dtype_name != "bfloat16"):
            raise ValueError("StreamVLN speculation requires cuda_graph=True and BF16 decode arithmetic")
        super().__init__(
            VLAPolicyConfig(
                name="streamvln",
                action_dim=2,
                action_horizon=4,
                default_num_steps=1,
                dtype=dtype_name,
            )
        )
        self.backbone = backbone
        self.processor = processor
        self.eos_token_ids = frozenset(eos_token_ids)
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.cuda_graph = cuda_graph
        self.decode_block_size = int(decode_block_size)
        self.cache_history_features = cache_history_features
        self.fast_action_decode = bool(fast_action_decode)
        self.speculative_tokens = speculative_tokens
        self.gpu_history_features = gpu_history_features
        self.action_token_ids = processor.action_token_ids
        self.backbone.configure_prefill_optimizations(
            cuda_graph,
            language_prefill=True,
        )
        self._decoder = StreamVLNDecoder(self)
        self.tree_decode = False
        self.tree_budget = tree_budget
        self.tree_refresh = tree_refresh
        if (
            tree_decode
            or fused_ops
            or split_attention
            or decode_linear
            or tree_fp32_projection
            or vision_kernel != "hf"
        ):
            self.configure_optimizations(
                tree_decode=tree_decode,
                tree_budget=tree_budget,
                tree_refresh=tree_refresh,
                fused_ops=fused_ops,
                split_attention=split_attention,
                attention_kernel=attention_kernel,
                decode_linear=decode_linear,
                tree_fp32_projection=tree_fp32_projection,
                vision_kernel=vision_kernel,
            )
        if (
            type(kv_cache_initial_capacity) is not int
            or kv_cache_initial_capacity < 0
            or (kv_cache_initial_capacity and kv_cache_initial_capacity > backbone.max_context)
        ):
            raise ValueError("kv_cache_initial_capacity must be between zero and max_context")
        if kv_cache_initial_capacity and not (
            getattr(backbone, "quantized_layers", ())
            and tree_decode
            and split_attention
            and attention_kernel == "gqa"
        ):
            raise ValueError(
                "growing KV currently requires quantized tree decoding with grouped-query attention"
            )
        backbone.kv_cache_initial_capacity = kv_cache_initial_capacity
        backbone.cache_generation_reserve = max_new_tokens + (1 << tree_budget.bit_length())
        if type(quantized_prefill_graphs) is not bool:
            raise ValueError("quantized_prefill_graphs must be a boolean")
        if quantized_prefill_graphs and not (
            getattr(backbone, "quantized_layers", ())
            and cuda_graph
            and split_attention
            and attention_kernel == "gqa"
        ):
            raise ValueError(
                "quantized prefill graphs require quantized weights and grouped-query CUDA graphs"
            )
        backbone.quantized_prefill_graphs = quantized_prefill_graphs
        if quantized_prefill_graphs:
            backbone.configure_prefill_optimizations(True, language_prefill=True)

    def configure_optimizations(
        self,
        *,
        tree_decode: bool = False,
        tree_budget: int = 31,
        tree_refresh: int = 64,
        fused_ops: bool = False,
        split_attention: bool = False,
        attention_kernel: str = "gqa",
        decode_linear: bool = False,
        tree_fp32_projection: bool = False,
        vision_kernel: str = "hf",
    ) -> None:
        """Select a profile before starting any session and invalidate old graphs.

        Callers must discard recurrent memories before changing this profile.
        Options preserve model parameters, but changed floating-point execution
        requires real-weight validation. No option is enabled implicitly.
        ``tree_refresh`` bounds how many completed responses reuse proposal
        trees before rebuilding them from the accumulated response counts.
        """
        from ...models.streamvln.optimized import StreamVLNTextOptions

        if self._decoder._startup_capture_active:
            raise RuntimeError("cannot reconfigure StreamVLN during graph capture")
        enabled = any((tree_decode, fused_ops, split_attention, decode_linear, tree_fp32_projection))
        if (enabled or vision_kernel != "hf") and (
            not self.cuda_graph or next(self.backbone.parameters()).dtype != torch.bfloat16
        ):
            raise ValueError("ActiveVLN-style StreamVLN optimizations require CUDA graphs and BF16")
        if tree_decode and self.speculative_tokens:
            raise ValueError("tree_decode and strict repeated speculation are alternative decoders")
        if tree_fp32_projection and not tree_decode:
            raise ValueError("tree_fp32_projection requires tree_decode")
        if tree_fp32_projection and getattr(self.backbone, "quantized_layers", ()):
            raise ValueError("tree_fp32_projection requires floating-point weights")
        if type(tree_budget) is not int or not 1 <= tree_budget <= 127:
            raise ValueError("tree_budget must be an integer from 1 to 127")
        if type(tree_refresh) is not int or tree_refresh < 1:
            raise ValueError("tree_refresh must be a positive integer")
        if vision_kernel not in ("hf", "sdpa", "fused"):
            raise ValueError("vision_kernel must be hf, sdpa or fused")
        options = StreamVLNTextOptions(
            enabled, fused_ops, split_attention, attention_kernel, decode_linear, tree_fp32_projection
        )
        if self.backbone.quantized_prefill_graphs and not (split_attention and attention_kernel == "gqa"):
            raise ValueError("quantized prefill graphs require grouped-query attention")
        self.backbone.configure_prefill_optimizations(False)
        self._decoder.reset_cuda_graph_runtime()
        self.processor.clear_runtime_caches()
        self.backbone.text_options = options
        self.backbone.vision_kernel = vision_kernel
        self.tree_decode, self.tree_budget = tree_decode, tree_budget
        self.tree_refresh = tree_refresh
        self.backbone.configure_prefill_optimizations(self.cuda_graph, language_prefill=True)

    @property
    def is_recurrent(self) -> bool:
        return True

    def build_serving_adapter(
        self,
        *,
        core: EngineCore,
        checkpoint: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> ServingAdapter:
        """Build the raw HTTP adapter while keeping StreamVLN details policy-local."""

        from .serving import StreamVLNServingAdapter

        return StreamVLNServingAdapter(
            core=core,
            checkpoint=checkpoint,
            config=config,
        )

    @property
    def manages_cuda_graph(self) -> bool:
        return True

    def configure_runtime(self, *, use_cuda_graph: bool) -> None:
        if not use_cuda_graph and (self.backbone.text_options.enabled or self.backbone.vision_kernel != "hf"):
            raise ValueError("disable the optimized StreamVLN profile before disabling CUDA graphs")
        self.cuda_graph = bool(use_cuda_graph)
        self.backbone.configure_prefill_optimizations(
            self.cuda_graph,
            language_prefill=True,
        )
        if not self.cuda_graph:
            self._decoder.reset_cuda_graph_runtime()

    def create_batched_runtime(
        self,
        *,
        batch_size: int,
        workspace_tokens: int | None = None,
        kv_pool_tokens: int | None = None,
        packed_prefill: bool = True,
        packed_prefill_bucket: int = 1,
        incremental_kv_commit: bool = True,
        shared_graph_buffers: bool = False,
    ) -> StreamVLNBatchedRuntime:
        """Create explicit tensor batching; generic recurrent serving remains B=1.

        Use eval() and inference_mode(). The returned runtime owns graph scratch
        and returns private episode memories; see StreamVLNBatchedRuntime.infer.
        """
        from .batching import StreamVLNBatchedRuntime

        return StreamVLNBatchedRuntime(
            self,
            batch_size=batch_size,
            workspace_tokens=workspace_tokens,
            kv_pool_tokens=kv_pool_tokens,
            packed_prefill=packed_prefill,
            packed_prefill_bucket=packed_prefill_bucket,
            incremental_kv_commit=incremental_kv_commit,
            shared_graph_buffers=shared_graph_buffers,
        )

    @contextmanager
    def startup_cuda_graph_capture(self) -> Iterator[None]:
        """Capture every native graph before serving or benchmark timing begins."""
        if not self.cuda_graph:
            raise RuntimeError("StreamVLN CUDA Graph execution is disabled")
        self.backbone.begin_startup_graph_capture()
        try:
            self._decoder.begin_startup_graph_capture()
        except BaseException:
            self.backbone.abort_startup_graph_capture()
            raise
        try:
            yield
        except BaseException:
            self._decoder.abort_startup_graph_capture()
            self.backbone.abort_startup_graph_capture()
            raise
        else:
            try:
                self._decoder.finish_startup_graph_capture()
                self.backbone.finish_startup_graph_capture()
            except BaseException:
                self._decoder.abort_startup_graph_capture()
                self.backbone.abort_startup_graph_capture()
                raise

    def cuda_graph_capture_stats(self) -> dict[str, int | bool]:
        backbone = self.backbone.graph_capture_stats()
        decoder = self._decoder.graph_capture_stats()
        if backbone["active"] != decoder["active"] or backbone["frozen"] != decoder["frozen"]:
            raise RuntimeError("StreamVLN CUDA Graph lifecycle state diverged")
        return {
            "active": bool(backbone["active"]),
            "frozen": bool(backbone["frozen"]),
            "vision": int(backbone["vision"]),
            "prefill": int(backbone["prefill"]),
            "decode": int(decoder["decode"]),
            "verification": int(decoder["verification"]),
            "tree": int(decoder["tree"]),
        }

    def cuda_graph_runtime_stats(self) -> dict[str, int]:
        return {
            "tree_refresh": self.tree_refresh,
            **self.backbone.graph_runtime_stats(),
            "decode_replays": self._decoder.graph_replays,
            "decode_eager_fallbacks": self._decoder.eager_fallbacks,
            **self._decoder.speculation_stats,
            **({} if self._decoder._tree_runtime is None else self._decoder._tree_runtime.counters),
        }

    def reset_cuda_graph_runtime_stats(self) -> None:
        self.backbone.reset_graph_runtime_stats()
        self._decoder.reset_runtime_stats()

    def clear_preprocessing_caches(self) -> None:
        self.processor.clear_runtime_caches()

    @property
    def decoder(self) -> StreamVLNDecoder:
        return self._decoder

    def collate(
        self,
        observations: list[Observation],
        request_ids: list[str],
    ) -> StreamVLNBatch:
        return self.processor.collate(observations, request_ids)

    def pad(self, batch: StreamVLNBatch, target_batch_size: int) -> StreamVLNBatch:
        if target_batch_size != 1:
            raise ValueError("StreamVLN recurrent batches cannot be padded beyond one request")
        return batch

    @torch.inference_mode()
    def encode_prefix(
        self,
        batch: StreamVLNBatch,
        memory: StreamVLNMemory | None = None,
    ) -> StreamVLNPrefix:
        """Prepare an observation and encode it through the ordinary model path."""
        return self.encode_prepared_prefix(self.prepare_prefix(batch, memory))

    @torch.inference_mode()
    def prepare_prefix(
        self, batch: StreamVLNBatch, memory: StreamVLNMemory | None = None
    ) -> StreamVLNPreparedPrefix:
        """Apply CPU image/prompt transforms and stage current inputs on the device."""
        observation = batch.observations[0]
        instruction = observation.instruction or ""
        if not instruction.strip():
            raise ValueError("StreamVLN requires a non-empty navigation instruction")
        if observation.images.ndim != 4 or observation.images.shape[0] < 1:
            raise ValueError("StreamVLN requires at least one CHW camera image")

        committed = memory or StreamVLNMemory(
            cache=None,
            frame_bank=(),
            frame_features=(),
            step_id=0,
            instruction=instruction,
        )
        if committed.instruction != instruction:
            raise ValueError("instruction changed inside one StreamVLN episode")
        history_length = (
            len(committed.frame_features) if self.cache_history_features else len(committed.frame_bank)
        )
        if committed.step_id != history_length:
            raise RuntimeError("StreamVLN history and step counter diverged")

        device = next(self.backbone.parameters()).device
        vision_dtype = next(self.backbone.vision_tower.parameters()).dtype
        # Closed-loop callers may execute several actions between model requests.
        # They must supply every newly observed primitive frame in order, ending
        # with stream_step_id. Request sequence numbers remain transport-owned.
        stream_step = observation.metadata.get("stream_step_id", committed.step_id)
        if type(stream_step) is not int or stream_step < committed.step_id:
            raise ValueError("stream_step_id must be an integer at or after the committed frame")
        stride = stream_step - committed.step_id + 1
        if "stream_step_id" in observation.metadata and observation.images.shape[0] != stride:
            raise ValueError("stream_step_id requires every new primitive frame, including the current frame")
        window_start = committed.step_id == 0 or (
            stream_step // self.processor.window_size > (committed.step_id - 1) // self.processor.window_size
        )
        intermediate_pixels = []
        if stride > 1:
            frames = tuple(frame.detach().cpu().clone().contiguous() for frame in observation.images[:-1])
            if self.cache_history_features:
                for step, frame in enumerate(frames, start=committed.step_id):
                    if self.processor.retains_history_feature(step):
                        pixels = (
                            self.processor._preprocess(
                                frame,
                                device=device if self.processor.gpu_image_transform is not None else None,
                            )
                            .unsqueeze(0)
                            .to(device=device, dtype=vision_dtype, non_blocking=True)
                        )
                        intermediate_pixels.append((step, pixels))
                committed = replace(
                    committed,
                    step_id=stream_step,
                    frame_features=committed.frame_features + (None,) * len(frames),
                )
            else:
                committed = replace(committed, step_id=stream_step, frame_bank=committed.frame_bank + frames)
        current = observation.images[-1].detach().cpu().clone().contiguous()
        frame_bank = committed.frame_bank + (current,) if not self.cache_history_features else ()
        if self.speculative_tokens and (device.type != "cuda" or not StreamVLNDecodeGraph.is_available()):
            raise ValueError("StreamVLN speculation requires a CUDA policy and Triton")
        inputs = self.processor.process(
            current,
            committed.instruction,
            step_id=committed.step_id,
            frame_bank=frame_bank if frame_bank else None,
            device=device if self.processor.gpu_image_transform is not None else None,
            window_start=window_start,
        )
        base_cache = None if inputs.window_start else committed.cache
        if not inputs.window_start and base_cache is None:
            raise RuntimeError("StreamVLN fast cache is missing inside an active window")
        device = next(self.backbone.parameters()).device
        input_ids = inputs.input_ids.to(device=device, non_blocking=True)
        pending_token_ids = () if inputs.window_start else committed.pending_token_ids
        use_response_template = (
            self.fast_action_decode
            and not self.speculative_tokens
            and not self.tree_decode
            and self.cuda_graph
            and self.cache_history_features
            and device.type == "cuda"
            and StreamVLNDecodeGraph.is_available()
        )
        response_prefix_ids: torch.Tensor | None = None
        if pending_token_ids or use_response_template:
            leading = torch.tensor(
                pending_token_ids,
                dtype=input_ids.dtype,
                device=device,
            ).reshape(1, -1)
            trailing = torch.tensor(
                self.processor.response_prefix_ids if use_response_template else (),
                dtype=input_ids.dtype,
                device=device,
            ).reshape(1, -1)
            input_ids = torch.cat((leading, input_ids, trailing), dim=1)
            if use_response_template:
                response_prefix_ids = trailing
        position_shift = len(pending_token_ids)
        current_pixels = inputs.current_pixel_values.to(
            device=device,
            dtype=vision_dtype,
            non_blocking=True,
        )
        memory_pixels = (
            inputs.memory_pixel_values.to(device=device, dtype=vision_dtype, non_blocking=True)
            if not self.cache_history_features and inputs.memory_pixel_values is not None
            else None
        )
        return StreamVLNPreparedPrefix(
            committed,
            frame_bank,
            inputs,
            base_cache,
            input_ids,
            current_pixels,
            memory_pixels,
            position_shift,
            use_response_template,
            response_prefix_ids,
            tuple(intermediate_pixels),
            stride,
        )

    @torch.inference_mode()
    def encode_prepared_prefix(self, prepared: StreamVLNPreparedPrefix) -> StreamVLNPrefix:
        """Encode staged pixels/tokens and advance model history without CPU preprocessing."""
        committed, frame_bank, inputs = prepared.committed, prepared.frame_bank, prepared.inputs
        base_cache, input_ids = prepared.base_cache, prepared.input_ids
        current_pixels, position_shift = prepared.current_pixels, prepared.position_shift
        use_response_template = prepared.use_response_template
        response_prefix_ids = prepared.response_prefix_ids
        frame_features = committed.frame_features
        response_prefix_hiddens: torch.Tensor | None = None
        reference_cache_length: int | None = None
        if self.cache_history_features:
            if prepared.intermediate_pixels:
                retained_features = list(frame_features)
                for step, pixels in prepared.intermediate_pixels:
                    features = self.backbone.encode_frames(pixels)[0]
                    retained_features[step] = self._retain_history_feature(features)
                frame_features = tuple(retained_features)
            current_features = self.backbone.encode_frames(current_pixels)[0]
            memory_features = self._selected_history_features(
                frame_features,
                inputs.slow_frame_indices,
                current_features,
            )
            embeddings = self.backbone.prepare_multimodal_feature_embeddings(
                input_ids,
                current_features,
                memory_features,
                image_position=inputs.image_position + position_shift,
                memory_position=(
                    None if inputs.memory_position is None else inputs.memory_position + position_shift
                ),
            )
            cache, hidden = self.backbone.forward_embeddings(
                embeddings,
                base_cache,
                project_logits=False,
                return_hidden_states=use_response_template,
            )
            if use_response_template:
                if response_prefix_ids is None:
                    raise RuntimeError("StreamVLN response-prefix token IDs are missing")
                response_length = int(response_prefix_ids.shape[1])
                if hidden.shape[1] <= response_length:
                    raise RuntimeError("StreamVLN response prefix has no reference hidden state")
                response_prefix_hiddens = hidden[:, -(response_length + 1) :]
                next_hidden = response_prefix_hiddens[:, -1]
                reference_cache_length = cache.seq_len - response_length
            else:
                next_hidden = hidden
            retained = (
                self._retain_history_feature(current_features)
                if self.processor.retains_history_feature(committed.step_id)
                else None
            )
            frame_features = frame_features + (retained,)
        else:
            cache, next_hidden = self.backbone.prefill_turn(
                input_ids,
                current_pixels,
                prepared.memory_pixels,
                base_cache,
                project_logits=False,
                image_position=inputs.image_position + position_shift,
                memory_position=(
                    None if inputs.memory_position is None else inputs.memory_position + position_shift
                ),
            )
        slow_indices = inputs.slow_frame_indices or committed.slow_frame_indices
        working = StreamVLNMemory(
            cache=cache,
            frame_bank=frame_bank,
            frame_features=frame_features,
            step_id=committed.step_id + 1,
            instruction=committed.instruction,
            slow_frame_indices=slow_indices,
            pending_token_ids=(),
        )
        return StreamVLNPrefix(
            memory=working,
            next_hidden=next_hidden,
            response_prefix_ids=response_prefix_ids,
            response_prefix_hiddens=response_prefix_hiddens,
            reference_cache_length=reference_cache_length,
        )

    def _retain_history_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.gpu_history_features:
            if not feature.is_cuda:
                raise ValueError("gpu_history_features requires a CUDA policy")
            # Vision graph output is overwritten on replay; history must own it.
            return feature.detach().clone()
        return self._offload_history_feature(feature)

    @staticmethod
    def _offload_history_feature(feature: torch.Tensor) -> torch.Tensor:
        host = torch.empty(
            feature.shape,
            dtype=feature.dtype,
            device="cpu",
            pin_memory=feature.is_cuda,
        )
        host.copy_(feature.detach(), non_blocking=feature.is_cuda)
        return host

    @staticmethod
    def _selected_history_features(
        frame_features: tuple[torch.Tensor | None, ...],
        indices: tuple[int, ...],
        current_features: torch.Tensor,
    ) -> torch.Tensor | None:
        if not indices:
            return None
        selected = []
        for index in indices:
            try:
                feature = frame_features[index]
            except IndexError as exc:
                raise RuntimeError("StreamVLN slow-memory feature index is missing") from exc
            if feature is None:
                raise RuntimeError("StreamVLN discarded a frame selected by slow memory")
            selected.append(feature)
        if current_features.is_cuda:
            memory = torch.empty(
                (len(selected), *current_features.shape),
                dtype=current_features.dtype,
                device=current_features.device,
            )
            for destination, source in zip(memory, selected, strict=True):
                destination.copy_(source, non_blocking=source.is_cuda or source.is_pinned())
            return memory
        return torch.stack(selected).to(dtype=current_features.dtype)

    @torch.inference_mode()
    def append_token(
        self,
        cache: StreamVLNCache,
        token: torch.Tensor,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        return self.backbone.append_token(cache, token, project_logits=False)


def _quantize_backbone(backbone: StreamVLNBackbone, config: QuantizationConfig) -> None:
    """Quantize Qwen2 transformer projections while retaining multimodal leaves."""

    backbone.fuse_projections()
    quantized: list[str] = []
    for index, layer in enumerate(backbone.llm.model.layers):
        candidates = (
            (layer.self_attn, "qkv_proj", f"llm.layers.{index}.self_attn.qkv_proj"),
            (layer.self_attn, "o_proj", f"llm.layers.{index}.self_attn.o_proj"),
            (layer.mlp, "gate_up_proj", f"llm.layers.{index}.mlp.gate_up_proj"),
            (layer.mlp, "down_proj", f"llm.layers.{index}.mlp.down_proj"),
        )
        for owner, attribute, name in candidates:
            if config.is_ignored(name):
                continue
            setattr(owner, attribute, quantize_linear(getattr(owner, attribute), config))
            quantized.append(name)
    backbone.configure_quantized_projections(tuple(quantized))


@register_policy("streamvln")
def build_streamvln(
    checkpoint: str | Path | None = None,
    *,
    dtype: str = "bfloat16",
    load_device: str | torch.device = "cpu",
    max_new_tokens: int = 128,
    max_context: int = 32768,
    repetition_penalty: float = 1.05,
    cuda_graph: bool = True,
    decode_block_size: int = 4,
    cache_history_features: bool = True,
    fast_action_decode: bool = True,
    speculative_tokens: int = 0,
    gpu_preprocess: bool = False,
    gpu_history_features: bool = False,
    tree_decode: bool = False,
    tree_budget: int = 31,
    tree_refresh: int = 64,
    fused_ops: bool = False,
    split_attention: bool = False,
    attention_kernel: str = "gqa",
    decode_linear: bool = False,
    tree_fp32_projection: bool = False,
    vision_kernel: str = "hf",
    kv_cache_initial_capacity: int = 0,
    quantized_prefill_graphs: bool = False,
    vision_phrase: str = VISION_PHRASE,
    quantization: str | Mapping[str, Any] | QuantizationConfig | None = None,
    **overrides: Any,
) -> VLAPolicy:
    """Load the checkpoint on the serving device without changing its dtype."""
    if checkpoint is None:
        raise ValueError("streamvln requires checkpoint=/path/to/streamvln")
    if overrides:
        unknown = ", ".join(sorted(overrides))
        raise TypeError(f"unknown streamvln overrides: {unknown}")
    quantization_config = parse_quantization_config(quantization)
    loading_options = (
        {"int8_config": quantization_config} if isinstance(quantization_config, INT8Config) else {}
    )
    loaded = load_streamvln_checkpoint(
        checkpoint,
        dtype=dtype,
        max_context=max_context,
        load_device=load_device,
        **loading_options,
    )
    if quantization_config is not None and not isinstance(quantization_config, INT8Config):
        _quantize_backbone(loaded.backbone, quantization_config)
    processor = StreamVLNProcessor(
        loaded.tokenizer,
        loaded.image_processor,
        window_size=32,
        num_history=8,
        vision_phrase=vision_phrase,
        gpu_preprocess=gpu_preprocess,
    )
    return StreamVLNPolicy(
        loaded.backbone,
        processor,
        eos_token_ids=loaded.eos_token_ids,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        cuda_graph=cuda_graph,
        decode_block_size=decode_block_size,
        cache_history_features=cache_history_features,
        fast_action_decode=fast_action_decode,
        speculative_tokens=speculative_tokens,
        gpu_history_features=gpu_history_features,
        tree_decode=tree_decode,
        tree_budget=tree_budget,
        tree_refresh=tree_refresh,
        fused_ops=fused_ops,
        split_attention=split_attention,
        attention_kernel=attention_kernel,
        decode_linear=decode_linear,
        tree_fp32_projection=tree_fp32_projection,
        vision_kernel=vision_kernel,
        kv_cache_initial_capacity=kv_cache_initial_capacity,
        quantized_prefill_graphs=quantized_prefill_graphs,
    )


__all__ = ["StreamVLNMemory", "StreamVLNPolicy", "build_streamvln"]
