from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from embodiinfer.engine.rollout.generation_backend import GenerationBackend
from embodiinfer.engine.session import SessionStore
from embodiinfer.exceptions import SessionCancelledError, StaleSessionError
from embodiinfer.policies.activevln.cache_activevln import (
    ActiveVLNMemory,
    BranchedActiveVLNMemory,
)
from embodiinfer.policies.activevln.modeling_activevln import (
    ActiveVLNBranchPrefix,
    ActiveVLNPrefix,
    _ActiveVLNDecoder,
)
from embodiinfer.policies.base import MemoryState
from embodiinfer.types import ActionChunk, Observation, SessionKey


def _chunk(length: int = 2):
    values = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
    tokens = torch.arange(length)[None]
    positions = torch.arange(length)[None, None].expand(3, 1, -1)
    return [(values, values + 1)], tokens, torch.ones_like(tokens), positions


@dataclass
class _DefaultMemoryHooks(MemoryState):
    value: int

    @property
    def seq_len(self):
        return self.value

    def to(self, device):
        del device
        return self


def test_memory_state_default_expand_and_compact_hooks():
    memory = _DefaultMemoryHooks(3)
    assert memory.expand(1) is memory
    assert memory.compact() is memory
    with pytest.raises(NotImplementedError, match="branch-safe"):
        memory.expand(2)


def test_session_group_commit_is_atomic_when_one_branch_is_stale():
    store = SessionStore()
    keys = [SessionKey("env", "episode", index) for index in range(2)]
    leases = store.checkout_many(keys)
    store.cancel([keys[1]])
    with pytest.raises(StaleSessionError, match="group commit"):
        store.commit_many(leases, [_FakeMemory(torch.tensor([[1]]))] * 2)
    assert store.committed(keys[0]) is None
    assert store.committed(keys[1]) is None
    leases[0].rollback()


def test_activevln_memory_expand_is_branch_safe_and_compactable():
    kv, tokens, mask, positions = _chunk()
    source = ActiveVLNMemory.from_chunk(kv, tokens, mask, positions, max_length=16)
    expanded = source.expand(3)
    assert isinstance(expanded, BranchedActiveVLNMemory)
    assert expanded.seq_lens == (2, 2, 2)

    kv2, tokens2, mask2, positions2 = _chunk(1)
    positions2 = positions2 + 2
    expanded.branches[0].append_chunk(kv2, tokens2 + 9, mask2, positions2)
    assert source.seq_len == 2
    assert expanded.seq_lens == (3, 2, 2)

    kept = expanded.compact([2, 0])
    assert isinstance(kept, BranchedActiveVLNMemory)
    assert kept.seq_lens == (2, 3)
    assert expanded.compact([1]) is expanded.branches[1]


@dataclass
class _FakeMemory:
    token_ids: torch.Tensor

    @property
    def seq_len(self):
        return self.token_ids.shape[1]

    def fork(self):
        return _FakeMemory(self.token_ids.clone())

    def to(self, device):
        return _FakeMemory(self.token_ids.to(device))


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        ids = [token for token in ids if token != 2]
        if ids == [7]:
            return "stop"
        if ids == [5]:
            return "move forward 25cm"
        return ""


def _logits(token: int):
    logits = torch.full((1, 10), -20.0)
    logits[0, token] = 20.0
    return logits


class _FakeARPolicy:
    repetition_penalty = 1.0
    do_sample = False
    temperature = 1.0
    top_p = 1.0
    eos_token_ids = (2,)
    tokenizer = _Tokenizer()

    def __init__(self, max_new_tokens=3):
        self.max_new_tokens = max_new_tokens

    def append_token(self, memory, token):
        next_memory = _FakeMemory(torch.cat([memory.token_ids, token], dim=1))
        # A forward action is followed by EOS; stop exits before this is read.
        return next_memory, _logits(2)


def _prefix(first_token: int):
    return ActiveVLNPrefix(_FakeMemory(torch.empty(1, 0, dtype=torch.long)), _logits(first_token))


def test_serial_ragged_decoder_alive_mask_covers_stop_and_eos():
    decoder = _ActiveVLNDecoder(_FakeARPolicy())
    prefix = ActiveVLNBranchPrefix((_prefix(7), _prefix(5)))
    result = decoder.decode(None, prefix, 1, 2, None)

    assert result.traces[0].stop_reason == "stop"
    assert result.traces[1].stop_reason == "eos"
    assert result.recompute_state.action_mask.tolist() == [[True, False], [True, True]]
    assert result.behavior_logprob[0, 1].item() == 0.0
    assert isinstance(result.next_memory, BranchedActiveVLNMemory)
    assert result.next_memory.seq_lens == (1, 2)


def test_autoregressive_max_tokens_and_external_cancel():
    decoder = _ActiveVLNDecoder(_FakeARPolicy(max_new_tokens=1))
    result = decoder.decode(None, _prefix(5), 1, 1, None)
    assert result.traces[0].stop_reason == "max_tokens"
    assert result.recompute_state.action_mask.tolist() == [[True]]

    with pytest.raises(SessionCancelledError):
        decoder.decode(None, _prefix(5), 1, 1, None, cancelled=lambda: True)


@dataclass
class _Batch:
    observations: list[Observation]
    request_ids: list[str]

    @property
    def batch_size(self):
        return len(self.observations)

    def to(self, device, dtype=None):
        del device, dtype
        return self


class _SerialPolicy:
    is_recurrent = True

    def collate(self, observations, request_ids):
        return _Batch(observations, request_ids)


class _SerialCore:
    def __init__(self):
        self.calls = []

    def execute(self, batch, num_steps, *, session_ids):
        del num_steps
        self.calls.append((batch.request_ids[0], session_ids[0]))
        return [ActionChunk(batch.request_ids[0], torch.zeros(1, 1))]


def _observation(env_id):
    return Observation(torch.zeros(1, 3, 2, 2), torch.zeros(1), torch.zeros(1), env_id=env_id)


def test_recurrent_multi_env_generation_is_explicit_serial_ragged_fallback():
    backend = GenerationBackend.__new__(GenerationBackend)
    backend.policy = _SerialPolicy()
    backend.core = _SerialCore()
    keys = [SessionKey("a", 1), SessionKey("b", 1)]
    output = backend.generate([_observation("a"), _observation("b")], session_ids=keys)

    assert backend.recurrent_batching_mode == "serial_ragged"
    assert not backend.supports_true_ragged_batching
    assert [chunk.request_id for chunk in output] == ["g0", "g1"]
    assert backend.core.calls == [("g0", keys[0]), ("g1", keys[1])]


class _GroupPolicy(_SerialPolicy):
    def __init__(self):
        self.decoder = _ActiveVLNDecoder(_FakeARPolicy())
        self.encode_calls = 0

    def encode_prefix(self, batch, memory=None):
        del batch
        assert memory is None
        self.encode_calls += 1
        return _prefix(5)


def test_recurrent_group_prefills_once_and_commits_isolated_branches():
    policy = _GroupPolicy()
    backend = GenerationBackend.__new__(GenerationBackend)
    backend.policy = policy
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.pcfg = type("Config", (), {"default_num_steps": 1, "action_horizon": 3, "action_dim": 2})()
    backend.core = type(
        "Core",
        (),
        {
            "_sessions": SessionStore(),
            "config": type("EngineConfig", (), {"num_steps": 1})(),
            "checkout_sessions": lambda self, keys: self._sessions.checkout_many(keys),
            "commit_sessions": lambda self, leases, memories: self._sessions.commit_many(leases, memories),
        },
    )()
    keys = [SessionKey("env", "episode", rollout) for rollout in range(3)]

    samples = backend.sample_group([_observation("env")], 3, session_ids=keys)
    assert policy.encode_calls == 1
    assert samples.actions.shape == (1, 3, 3, 2)
    memories = [backend.core._sessions.committed(key) for key in keys]
    assert [memory.seq_len for memory in memories] == [2, 2, 2]
    assert len({id(memory) for memory in memories}) == 3
