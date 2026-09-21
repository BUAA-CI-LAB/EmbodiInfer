from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from embodiinfer.models.streamvln import StreamVLNCache
from embodiinfer.models.streamvln.backbone import (
    _compiled_prefill_block_tail,
    _linear,
    _PackedLinear,
)
from embodiinfer.policies.streamvln.policy import StreamVLNDecoder, StreamVLNPolicy
from embodiinfer.policies.streamvln.processing import StreamVLNProcessor
from embodiinfer.policies.streamvln.prompt import render_streamvln_prompt
from embodiinfer.types import Observation


class _Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        return {"input_ids": [len(text) % 17 + 1]}


class _ImageProcessor:
    def preprocess(self, frame: torch.Tensor, *, return_tensors: str):
        assert return_tensors == "pt"
        return {"pixel_values": frame.unsqueeze(0)}


class _VisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


def test_packed_projection_storage_preserves_independent_gemms() -> None:
    first = nn.Linear(5, 3, bias=False)
    second = nn.Linear(5, 2, bias=False)
    inputs = torch.randn(2, 4, 5)
    packed = _PackedLinear(first, second)

    actual = _linear(packed, inputs)
    expected = torch.cat((first(inputs), second(inputs)), dim=-1)

    assert torch.equal(actual, expected)
    assert packed.weight.shape == (5, 5)


def test_prefill_tail_keeps_gate_and_up_as_independent_gemms(monkeypatch) -> None:
    shapes: list[tuple[int, ...]] = []
    linear = torch.nn.functional.linear

    def traced_linear(inputs, weight, bias=None):
        shapes.append(tuple(weight.shape))
        return linear(inputs, weight, bias)

    monkeypatch.setattr("embodiinfer.models.streamvln.backbone.F.linear", traced_linear)
    _compiled_prefill_block_tail(
        torch.randn(1, 2, 4),
        torch.randn(1, 2, 4),
        torch.randn(4),
        torch.randn(6, 4),
        torch.randn(4, 3),
        torch.randn(4),
        1e-6,
    )

    assert shapes == [(3, 4), (3, 4), (4, 3)]


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.vision_tower = _VisionTower()
        self.vision_batches: list[int] = []
        self.memory_batches: list[int] = []
        self.prefill_optimizations = False
        self.language_prefill_optimizations = False
        self.startup_capture_active = False
        self.startup_capture_frozen = False

    def configure_prefill_optimizations(
        self,
        enabled: bool,
        *,
        language_prefill: bool = True,
    ) -> None:
        self.prefill_optimizations = bool(enabled)
        self.language_prefill_optimizations = bool(enabled and language_prefill)
        if not enabled:
            self.startup_capture_active = False
            self.startup_capture_frozen = False

    def begin_startup_graph_capture(self) -> None:
        self.startup_capture_active = True

    def finish_startup_graph_capture(self) -> None:
        self.startup_capture_active = False
        self.startup_capture_frozen = True

    def abort_startup_graph_capture(self) -> None:
        self.startup_capture_active = False
        self.startup_capture_frozen = False

    def graph_capture_stats(self) -> dict[str, int | bool]:
        return {
            "active": self.startup_capture_active,
            "frozen": self.startup_capture_frozen,
            "vision": 0,
            "prefill": 0,
        }

    def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        self.vision_batches.append(int(pixels.shape[0]))
        values = pixels.mean(dim=(1, 2, 3))
        return values[:, None, None].expand(-1, 2, 4).clone()

    def prepare_multimodal_feature_embeddings(
        self,
        input_ids: torch.Tensor,
        current_features: torch.Tensor,
        memory_features: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        del input_ids, kwargs
        self.memory_batches.append(0 if memory_features is None else int(memory_features.shape[0]))
        pieces = [current_features]
        if memory_features is not None:
            pieces.insert(0, memory_features.reshape(-1, memory_features.shape[-1]))
        return torch.cat(pieces).unsqueeze(0)

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
        cache: StreamVLNCache | None,
        *,
        project_logits: bool,
        return_hidden_states: bool = False,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        assert not project_logits
        if cache is None:
            storage = torch.empty((1, 1, 1, 128, 1))
            cache = StreamVLNCache(storage, storage.clone(), 0)
        hidden = embeddings if return_hidden_states else embeddings[:, -1]
        return cache.advance(cache.seq_len + embeddings.shape[1]), hidden

    def prefill_turn(
        self,
        input_ids: torch.Tensor,
        current_pixels: torch.Tensor,
        memory_pixels: torch.Tensor | None,
        cache: StreamVLNCache | None,
        **kwargs,
    ) -> tuple[StreamVLNCache, torch.Tensor]:
        del kwargs
        all_pixels = current_pixels
        memory_count = 0
        if memory_pixels is not None:
            memory_count = int(memory_pixels.shape[0])
            all_pixels = torch.cat((memory_pixels, current_pixels))
        features = self.encode_frames(all_pixels)
        memory_features = features[:memory_count] if memory_count else None
        embeddings = self.prepare_multimodal_feature_embeddings(
            input_ids,
            features[-1],
            memory_features,
        )
        return self.forward_embeddings(embeddings, cache, project_logits=False)


def _policy(*, cache_history_features: bool) -> tuple[StreamVLNPolicy, _Backbone]:
    backbone = _Backbone()
    processor = StreamVLNProcessor(
        _Tokenizer(),
        _ImageProcessor(),
        window_size=4,
        num_history=2,
    )
    policy = StreamVLNPolicy(
        backbone,
        processor,
        max_new_tokens=4,
        cuda_graph=False,
        decode_block_size=4,
        cache_history_features=cache_history_features,
        fast_action_decode=False,
    )
    return policy, backbone


def _observation(step: int) -> Observation:
    return Observation(
        images=torch.full((1, 3, 2, 2), step / 10.0),
        state=torch.empty(0),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction="go forward",
    )


def test_slow_memory_prompt_matches_published_streamvln_wording() -> None:
    prompt = render_streamvln_prompt(
        "go forward",
        window_start=True,
        include_memory=True,
    )

    assert "You have visited these areas <memory>." in prompt
    assert "historical observations" not in prompt


def _encode_sequence(policy: StreamVLNPolicy, steps: int):
    memory = None
    hidden = []
    for step in range(steps):
        batch = policy.collate([_observation(step)], [str(step)])
        prefix = policy.encode_prefix(batch, memory)
        memory = prefix.memory
        hidden.append(prefix.next_hidden.clone())
    return memory, hidden


def test_history_feature_cache_reuses_slow_frames_without_changing_hidden() -> None:
    cached, cached_backbone = _policy(cache_history_features=True)
    reference, reference_backbone = _policy(cache_history_features=False)

    cached_memory, cached_hidden = _encode_sequence(cached, 5)
    reference_memory, reference_hidden = _encode_sequence(reference, 5)

    assert cached_backbone.vision_batches == [1, 1, 1, 1, 1]
    assert reference_backbone.vision_batches == [1, 1, 1, 1, 3]
    assert cached_backbone.memory_batches[-1] == 2
    assert [feature is not None for feature in cached_memory.frame_features] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert cached_memory.frame_bank == ()
    assert len(reference_memory.frame_bank) == 5
    for actual, expected in zip(cached_hidden, reference_hidden, strict=True):
        assert torch.equal(actual, expected)


def test_history_feature_cache_builds_uncommitted_state_without_mutating_input() -> None:
    policy, _ = _policy(cache_history_features=True)
    first = policy.encode_prefix(policy.collate([_observation(0)], ["0"]), None)
    committed = first.memory

    second = policy.encode_prefix(
        policy.collate([_observation(1)], ["1"]),
        committed,
    )

    assert committed.step_id == 1
    assert len(committed.frame_features) == 1
    assert second.memory.step_id == 2
    assert len(second.memory.frame_features) == 2


def test_streamvln_runtime_controls_its_native_graph_switch() -> None:
    policy, backbone = _policy(cache_history_features=True)
    assert policy.manages_cuda_graph
    policy.configure_runtime(use_cuda_graph=True)
    assert policy.cuda_graph and backbone.prefill_optimizations
    assert backbone.language_prefill_optimizations
    policy.configure_runtime(use_cuda_graph=False)
    assert not policy.cuda_graph and not backbone.prefill_optimizations


def test_streamvln_cuda_graph_lifecycle_freezes_after_startup() -> None:
    policy, _ = _policy(cache_history_features=True)
    policy.configure_runtime(use_cuda_graph=True)

    with policy.startup_cuda_graph_capture():
        assert policy.cuda_graph_capture_stats()["active"]

    stats = policy.cuda_graph_capture_stats()
    assert not stats["active"]
    assert stats["frozen"]


def test_streamvln_decoder_uses_eager_fallback_without_runtime_capture() -> None:
    policy = SimpleNamespace(decode_block_size=4)
    decoder = StreamVLNDecoder(policy)

    graph = decoder._get_cuda_graph(
        SimpleNamespace(),
        SimpleNamespace(),
        block_size=4,
        produce_next=False,
    )

    assert graph is None
    assert decoder.eager_fallbacks == 1
    assert not decoder._cuda_graphs


@dataclass
class _GraphReplay:
    tokens: torch.Tensor
    logprobs: torch.Tensor

    def replay(self, cache, token, logprob, workspace):
        del token, logprob, workspace
        size = self.tokens.shape[1]
        return cache.advance(cache.seq_len + size), self.tokens, self.logprobs, None, None


class _EchoGraph:
    def replay(self, cache, token, logprob, workspace):
        del workspace
        return cache.advance(cache.seq_len + 1), token.clone(), logprob.clone(), None, None


@dataclass
class _TemplateGraph:
    tokens: torch.Tensor
    eos_token: torch.Tensor

    def replay(self, cache, token, logprob, workspace):
        del token, logprob, workspace
        size = self.tokens.shape[1]
        action_logprobs = torch.full((1, size), -0.2)
        eos_logprob = torch.tensor([[-0.3]])
        return (
            cache.advance(cache.seq_len + size),
            self.tokens,
            action_logprobs,
            self.eos_token,
            eos_logprob,
        )


def test_decode_block_detaches_initial_workspace_token_before_capture(monkeypatch) -> None:
    policy = SimpleNamespace(
        max_new_tokens=1,
        decode_block_size=1,
        eos_token_ids=frozenset(),
    )
    decoder = StreamVLNDecoder(policy)
    workspace_token = torch.tensor([[3]])

    def capture(*args, **kwargs):
        del args, kwargs
        workspace_token.fill_(9)
        return _EchoGraph()

    monkeypatch.setattr(decoder, "_get_cuda_graph", capture)
    storage = torch.empty((1, 1, 1, 16, 1))
    cache = StreamVLNCache(storage, storage.clone(), 5)

    _, tokens, _, reason = decoder._decode_cuda_blocks(
        cache,
        workspace_token,
        torch.tensor([[-0.1]]),
        SimpleNamespace(),
        cancelled=None,
    )

    assert reason == "max_tokens"
    assert torch.cat(tokens, dim=1).tolist() == [[3]]


def test_decode_block_commits_only_through_early_eos(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = SimpleNamespace(
        max_new_tokens=4,
        decode_block_size=4,
        eos_token_ids=frozenset({9}),
    )
    decoder = StreamVLNDecoder(policy)
    graph = _GraphReplay(
        torch.tensor([[3, 4, 9, 7]]),
        torch.tensor([[-0.1, -0.2, -0.3, -0.4]]),
    )
    monkeypatch.setattr(decoder, "_get_cuda_graph", lambda *args, **kwargs: graph)
    storage = torch.empty((1, 1, 1, 16, 1))
    cache = StreamVLNCache(storage, storage.clone(), 5)

    cache, tokens, logprobs, reason = decoder._decode_cuda_blocks(
        cache,
        torch.tensor([[3]]),
        torch.tensor([[-0.1]]),
        SimpleNamespace(),
        cancelled=None,
    )

    assert reason == "eos"
    assert cache.seq_len == 8
    assert torch.cat(tokens, dim=1).tolist() == [[3, 4, 9]]
    assert torch.cat(logprobs, dim=1).tolist()[0] == pytest.approx([-0.1, -0.2, -0.3])


@pytest.mark.parametrize(
    ("action_tokens", "eos_token", "accepted"),
    [
        ([10, 11, 10, 11], 99, True),
        ([10, 12, 10, 11], 99, False),
        ([10, 11, 10, 11], 98, False),
    ],
)
def test_response_template_validates_actions_and_eos(
    monkeypatch,
    action_tokens,
    eos_token,
    accepted,
) -> None:
    policy = SimpleNamespace(
        max_new_tokens=8,
        decode_block_size=4,
        eos_token_ids=frozenset({99}),
        action_token_ids=frozenset({10, 11}),
        config=SimpleNamespace(action_horizon=4),
    )
    decoder = StreamVLNDecoder(policy)
    selected = iter((1, 2, 3, 10))

    def select(*args, **kwargs):
        del args, kwargs
        token = torch.tensor([[next(selected)]])
        return token, torch.tensor([[-0.1]])

    monkeypatch.setattr(decoder, "_select_token", select)

    monkeypatch.setattr(
        decoder,
        "_get_cuda_graph",
        lambda *args, **kwargs: _TemplateGraph(
            torch.tensor([action_tokens]),
            torch.tensor([[eos_token]]),
        ),
    )
    monkeypatch.setattr("embodiinfer.policies.streamvln.policy.mark_penalized", lambda *args: None)
    storage = torch.empty((1, 1, 1, 32, 1))
    cache = StreamVLNCache(storage, storage.clone(), 8)
    prefix = SimpleNamespace(
        memory=SimpleNamespace(cache=cache),
        response_prefix_ids=torch.tensor([[1, 2, 3]]),
        response_prefix_hiddens=torch.zeros((1, 4, 1)),
    )

    result = decoder._decode_response_template(
        prefix,
        SimpleNamespace(),
        cancelled=None,
    )

    if not accepted:
        assert result is None
        return
    assert result is not None
    next_cache, tokens, _, pending = result
    assert next_cache.seq_len == 12
    assert torch.cat(tokens, dim=1).tolist() == [[1, 2, 3, 10, 11, 10, 11, 99]]
    assert pending == (99,)


@pytest.mark.parametrize("cache_history_features", [False, True])
def test_separate_preparation_preserves_prefill_and_recurrent_history(cache_history_features):
    ordinary, _ = _policy(cache_history_features=cache_history_features)
    split, backbone = _policy(cache_history_features=cache_history_features)
    ordinary_memory = split_memory = None
    for step in range(7):
        observation = _observation(step)
        expected = ordinary.encode_prefix(ordinary.collate([observation], [str(step)]), ordinary_memory)
        calls_before = len(backbone.vision_batches)
        prepared = split.prepare_prefix(split.collate([observation], [str(step)]), split_memory)
        assert len(backbone.vision_batches) == calls_before  # CPU preparation never executes vision.
        actual = split.encode_prepared_prefix(prepared)
        torch.testing.assert_close(actual.next_hidden, expected.next_hidden, rtol=0, atol=0)
        assert actual.memory.step_id == expected.memory.step_id
        assert actual.memory.seq_len == expected.memory.seq_len
        assert actual.memory.slow_frame_indices == expected.memory.slow_frame_indices
        ordinary_memory, split_memory = expected.memory, actual.memory
