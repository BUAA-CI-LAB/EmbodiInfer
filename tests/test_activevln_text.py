"""Text requests borrow policy weights but cannot consume navigation state."""

import time
from types import SimpleNamespace

import pytest
import torch

from embodiinfer.exceptions import SessionCancelledError
from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPolicy, ActiveVLNPrefix
from embodiinfer.policies.activevln.processor_activevln import ProcessedTurn
from embodiinfer.policies.activevln.text_activevln import ActiveVLNTextDecoder


class Processor:
    def apply_chat_template(self, messages, **_):
        return str(messages)

    def __call__(self, **_):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


class Policy:
    def __init__(self):
        self._processor = SimpleNamespace(_processor=Processor())
        self.qwen = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(in_channels=3, temporal_patch_size=2, patch_size=14)
            )
        )
        self.max_context = 100
        self.eos_token_ids = (9,)
        self.prefills = self.decode_calls = 0
        self.tokenizer = SimpleNamespace(
            decode=lambda tokens, **_: "stop" if tokens == [4, 9] else str(tokens)
        )
        self.weights = object()

    @staticmethod
    def _positions(turn):
        return torch.arange(turn.input_ids.shape[1])[None, None].expand(3, 1, -1), turn.input_ids.shape[1]

    @staticmethod
    def prepare_device(host, memory):
        assert memory is None
        return host

    def encode_prepared_prefix(self, _):
        self.prefills += 1
        return ActiveVLNPrefix([], torch.nn.functional.one_hot(torch.tensor([4]), 10).float())

    def _encode_prepared_prefix(self, prepared, *, checkpoint_rotary):
        assert checkpoint_rotary is True
        return self.encode_prepared_prefix(prepared)

    def append_token(self, memory, token):
        self.decode_calls += 1
        memory.append(token.item())
        return memory, torch.nn.functional.one_hot(torch.tensor([9]), 10).float()

    def _append_token(self, memory, token, *, checkpoint_rotary):
        assert checkpoint_rotary is True
        return self.append_token(memory, token)


MESSAGES = [{"role": "user", "content": "Describe the observation."}]


def test_text_yields_and_does_not_interpret_stop_as_a_motion_command():
    policy = Policy()
    decoder = ActiveVLNTextDecoder(policy)
    assert decoder.policy is policy
    state = decoder.prepare(MESSAGES)
    assert decoder.step(state) is True
    assert state.tokens == [4] and policy.decode_calls == 1
    with pytest.raises(ValueError, match="not complete"):
        decoder.result(state)
    assert decoder.step(state) is False
    result = decoder.result(state)
    assert result.text == "stop" and result.token_ids == (4, 9) and result.stop_reason == "eos"
    assert decoder.step(state) is False and policy.decode_calls == 1


def test_interleaved_requests_have_separate_kv_and_share_policy():
    policy = Policy()
    decoder = ActiveVLNTextDecoder(policy)
    left, right = decoder.prepare(MESSAGES), decoder.prepare(MESSAGES)
    assert left.prefix.memory is not right.prefix.memory
    decoder.step(left)
    assert right.tokens == [] and right.prefix.memory == []
    decoder.step(right)
    assert left.prefix.memory == right.prefix.memory == [4]
    assert decoder.policy.weights is policy.weights


def test_cancel_and_owner_invalidation_prevent_late_model_execution():
    policy = Policy()
    decoder = ActiveVLNTextDecoder(policy)
    state = decoder.prepare(MESSAGES)
    with pytest.raises(SessionCancelledError):
        decoder.step(state, cancelled=lambda: True)
    with pytest.raises(SessionCancelledError):
        decoder.result(state)
    assert policy.decode_calls == 0
    state = decoder.prepare(MESSAGES)
    decoder.invalidate()
    with pytest.raises(SessionCancelledError, match="invalidated"):
        decoder.step(state)
    assert policy.decode_calls == 0


def test_foreign_decoder_cannot_use_another_requests_memory():
    policy = Policy()
    left, right = ActiveVLNTextDecoder(policy), ActiveVLNTextDecoder(policy)
    with pytest.raises(SessionCancelledError):
        right.step(left.prepare(MESSAGES))
    assert policy.decode_calls == 0


@pytest.mark.parametrize(
    "limits",
    [{"max_new_tokens": 0}, {"max_new_tokens": True}, {"max_input_tokens": 2}, {"max_new_tokens": 100}],
)
def test_invalid_budgets_fail_before_prefill(limits):
    policy = Policy()
    with pytest.raises(ValueError):
        ActiveVLNTextDecoder(policy).prepare(MESSAGES, **limits)
    assert policy.prefills == 0


def test_expired_and_cancelled_requests_do_not_prefill():
    policy = Policy()
    decoder = ActiveVLNTextDecoder(policy)
    for options in ({"deadline": time.monotonic() - 1}, {"cancelled": lambda: True}):
        with pytest.raises(SessionCancelledError):
            decoder.prepare(MESSAGES, **options)
    assert policy.prefills == 0


def test_token_limit_is_not_reported_as_eos():
    decoder = ActiveVLNTextDecoder(Policy())
    state = decoder.prepare(MESSAGES, max_new_tokens=1)
    assert decoder.step(state) is False
    assert decoder.result(state).stop_reason == "max_tokens"
    assert decoder.policy.decode_calls == 0


def test_text_only_embedding_does_not_call_the_vision_encoder():
    embedding = torch.nn.Embedding(10, 3)
    policy = SimpleNamespace(
        _text=SimpleNamespace(embed_tokens=embedding),
        qwen=SimpleNamespace(config=SimpleNamespace(image_token_id=8)),
    )
    turn = ProcessedTurn(
        torch.tensor([[1, 2]]), torch.ones(1, 2), torch.empty(0, 1176), torch.empty(0, 3), "text", "hash"
    )
    assert torch.equal(ActiveVLNPolicy._embed_turn(policy, turn), embedding(turn.input_ids))
    turn.input_ids = torch.tensor([[1, 8]])
    with pytest.raises(ValueError, match="image tokens require"):
        ActiveVLNPolicy._embed_turn(policy, turn)


def test_failed_append_cannot_reuse_a_partially_mutated_kv():
    policy = Policy()
    decoder = ActiveVLNTextDecoder(policy)
    state = decoder.prepare(MESSAGES)

    def failed_append(memory, token):
        memory.append(token.item())
        raise RuntimeError("injected device failure after a KV write")

    policy.append_token = failed_append
    with pytest.raises(RuntimeError, match="injected device failure"):
        decoder.step(state)
    with pytest.raises(SessionCancelledError, match="cancelled"):
        decoder.step(state)
    with pytest.raises(SessionCancelledError, match="cancelled"):
        decoder.result(state)


def test_graph_enable_invalidates_old_state_and_owner_reset_releases_workspace(monkeypatch):
    policy = Policy()
    decoder = ActiveVLNTextDecoder(policy)
    old = decoder.prepare(MESSAGES)
    graph = SimpleNamespace(append_token=policy.append_token)
    monkeypatch.setattr("embodiinfer.policies.activevln.text_graph.ActiveVLNTextGraph", lambda owner: graph)
    decoder.enable_decode_graphs()
    with pytest.raises(SessionCancelledError, match="invalidated"):
        decoder.step(old)
    state = decoder.prepare(MESSAGES)
    assert decoder.step(state) is True
    assert state.tokens == [4] and policy.decode_calls == 1
    decoder.invalidate()
    assert decoder._decode_graph is None
    with pytest.raises(SessionCancelledError, match="invalidated"):
        decoder.step(state)


def test_graph_failure_after_kv_write_cancels_request_without_eager_retry(monkeypatch):
    policy = Policy()
    decoder = ActiveVLNTextDecoder(policy)

    def fail(memory, token):
        memory.append(token.item())
        raise RuntimeError("graph device failure after write")

    monkeypatch.setattr(
        "embodiinfer.policies.activevln.text_graph.ActiveVLNTextGraph",
        lambda owner: SimpleNamespace(append_token=fail),
    )
    decoder.enable_decode_graphs()
    state = decoder.prepare(MESSAGES)
    with pytest.raises(RuntimeError, match="graph device failure"):
        decoder.step(state)
    with pytest.raises(SessionCancelledError, match="cancelled"):
        decoder.step(state)
    assert policy.decode_calls == 0


@pytest.mark.parametrize(
    "config, message",
    [
        ({"text_decode_graphs": True}, "requires shared_text"),
        ({"shared_text": True, "text_decode_graphs": "true"}, "must be a boolean"),
    ],
)
def test_invalid_graph_configuration_fails_before_adapter_allocation(config, message):
    with pytest.raises(ValueError, match=message):
        ActiveVLNPolicy.build_serving_adapter(SimpleNamespace(), core=None, config=config)
