"""Opt-in, cooperatively scheduled text decoding with existing ActiveVLN weights.

This view provides raw model output, not a claim that a navigation checkpoint
can plan or call tools reliably. The caller owns scheduling and tool execution.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ...exceptions import SessionCancelledError
from .modeling_activevln import ActiveVLNPrefix, HostActiveVLNTurn
from .processor_activevln import ProcessedTurn

if TYPE_CHECKING:
    from .modeling_activevln import ActiveVLNPolicy
    from .text_graph import ActiveVLNTextGraph


@dataclass
class ActiveVLNTextState:
    """Private KV for one text request; never a navigation session's memory."""

    prefix: ActiveVLNPrefix
    owner: object
    epoch: int
    max_new_tokens: int
    deadline: float | None
    input_tokens: int
    tokens: list[int] = field(default_factory=list)
    stop_reason: str | None = None


@dataclass(frozen=True)
class ActiveVLNTextResult:
    """Unrepaired generated text and its exact token sequence."""

    text: str
    token_ids: tuple[int, ...]
    input_tokens: int
    stop_reason: str


class ActiveVLNTextDecoder:
    """Borrow one policy's parameters and yield after each generated token.

    ``prepare`` is one bounded prefill operation. A scheduler can interleave
    ``step`` calls with higher-priority navigation requests. It must serialize
    GPU execution through the policy/runtime's existing ownership mechanism.
    This object loads no weights and implements greedy decoding only.
    """

    def __init__(self, policy: ActiveVLNPolicy) -> None:
        self.policy = policy
        self._owner = object()
        self._epoch = 0
        self._decode_graph: ActiveVLNTextGraph | None = None

    def enable_decode_graphs(self) -> None:
        """Capture private BF16 decode graphs before admitting requests.

        Call under exclusive GPU ownership on the stream used for subsequent
        steps. Attention stays unpadded and uses the eager reference backend.
        Existing text state is invalidated; no additional weights are loaded.
        """
        from .text_graph import ActiveVLNTextGraph

        self.invalidate()
        self._decode_graph = ActiveVLNTextGraph(self.policy)

    def invalidate(self) -> None:
        """Reject outstanding text state after reset, refit or owner shutdown."""
        self._epoch += 1
        self._decode_graph = None

    def _check(self, state: ActiveVLNTextState, cancelled: Callable[[], bool] | None = None) -> None:
        if state.owner is not self._owner or state.epoch != self._epoch:
            raise SessionCancelledError("ActiveVLN text state belongs to an invalidated owner")
        if state.stop_reason == "cancelled" or (cancelled is not None and cancelled()):
            state.stop_reason = "cancelled"
            raise SessionCancelledError("ActiveVLN text request was cancelled")
        if state.deadline is not None and time.monotonic() >= state.deadline:
            state.stop_reason = "cancelled"
            raise SessionCancelledError("ActiveVLN text request deadline expired")

    def prepare(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        images: Sequence[Any] = (),
        max_new_tokens: int = 64,
        max_input_tokens: int = 2048,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ActiveVLNTextState:
        """Render a standalone conversation and allocate independently owned KV.

        Images use the owning Qwen processor's PIL/array input contract. Both
        budgets are checked before GPU prefill; deadlines are absolute monotonic
        times. No navigation history is read, modified or reset.
        """
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be an integer in [1, 512]")
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if not messages or any(
            message.get("role") not in {"system", "user", "assistant", "tool"} for message in messages
        ):
            raise ValueError("a conversation needs messages with supported roles")
        if (cancelled is not None and cancelled()) or (deadline is not None and time.monotonic() >= deadline):
            raise SessionCancelledError("ActiveVLN text request expired before processing")
        processor = self.policy._processor._processor
        text = processor.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        encoded = processor(text=[text], images=list(images) or None, padding=False, return_tensors="pt")
        count = encoded["input_ids"].shape[1]
        if count > max_input_tokens or count + max_new_tokens > self.policy.max_context:
            raise ValueError("text request exceeds its input/context budget")
        vision = self.policy.qwen.config.vision_config
        patch_width = int(vision.in_channels) * int(vision.temporal_patch_size) * int(vision.patch_size) ** 2
        turn = ProcessedTurn(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            pixel_values=encoded.get("pixel_values", torch.empty((0, patch_width))),
            image_grid_thw=encoded.get("image_grid_thw", torch.empty((0, 3), dtype=torch.long)),
            serialized_prompt=text,
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        host = HostActiveVLNTurn(turn, *self.policy._positions(turn), initial=True)
        if (cancelled is not None and cancelled()) or (deadline is not None and time.monotonic() >= deadline):
            raise SessionCancelledError("ActiveVLN text request expired before prefill")
        with torch.inference_mode():
            # Recomputing inverse frequencies on CUDA can differ from the
            # loaded model's rotary buffer and flip a close greedy choice.
            # This private text path bypasses the navigation graph workspace.
            prefix = self.policy._encode_prepared_prefix(
                self.policy.prepare_device(host, None), checkpoint_rotary=True
            )
        return ActiveVLNTextState(prefix, self._owner, self._epoch, max_new_tokens, deadline, count)

    def step(self, state: ActiveVLNTextState, *, cancelled: Callable[[], bool] | None = None) -> bool:
        """Generate at most one token; return whether another decode step is needed."""
        self._check(state, cancelled)
        if state.stop_reason is not None:
            return False
        try:
            with torch.inference_mode():
                token = state.prefix.next_logits.argmax(dim=-1, keepdim=True)
                value = int(token.item())
                if value in self.policy.eos_token_ids:
                    stop_reason = "eos"
                elif len(state.tokens) + 1 >= state.max_new_tokens:
                    stop_reason = "max_tokens"
                else:
                    stop_reason = None
                    if self._decode_graph is None:
                        memory, logits = self.policy._append_token(
                            state.prefix.memory, token, checkpoint_rotary=True
                        )
                    else:
                        memory, logits = self._decode_graph.append_token(state.prefix.memory, token)
        except BaseException:
            # Append may have written part of an in-place KV buffer. This
            # request must not retry from a partially updated prefix.
            state.stop_reason = "cancelled"
            raise
        # Completed requests never consume another token. Skip its unused
        # forward/KV append, exactly as reference autoregressive generation does.
        if stop_reason is None:
            state.prefix = ActiveVLNPrefix(memory, logits)
        state.tokens.append(value)
        state.stop_reason = stop_reason
        # The word "stop" in text is not interpreted as a navigation command.
        return state.stop_reason is None

    def result(self, state: ActiveVLNTextState) -> ActiveVLNTextResult:
        """Return a completed response without interpreting or repairing its text."""
        self._check(state)
        if state.stop_reason is None:
            raise ValueError("text generation is not complete")
        return ActiveVLNTextResult(
            text=self.policy.tokenizer.decode(state.tokens, skip_special_tokens=True),
            token_ids=tuple(state.tokens),
            input_tokens=state.input_tokens,
            stop_reason=state.stop_reason,
        )
