from dataclasses import dataclass

import pytest
import torch

from embodiinfer.engine.async_engine import AsyncEngine
from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.engine.parallel.data_parallel import InProcessReplica
from embodiinfer.engine.rollout.generation_backend import GenerationBackend
from embodiinfer.exceptions import SessionRequiredError, UnsupportedRecurrentModeError
from embodiinfer.policies.base import VLAPolicy
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.decoder import AutoregressiveDecoder, DecodeResult, RLDecoder
from embodiinfer.types import DecodeTrace, Observation, SessionKey


@dataclass
class _Batch:
    tokens: torch.Tensor
    request_ids: list[str]

    @property
    def batch_size(self):
        return self.tokens.shape[0]

    def to(self, device, dtype=None):
        return _Batch(self.tokens.to(device), self.request_ids)


@dataclass
class _Memory:
    tokens: torch.Tensor

    @property
    def seq_len(self):
        return self.tokens.numel()

    def to(self, device):
        return _Memory(self.tokens.to(device))


@dataclass
class _Prefix:
    tokens: torch.Tensor

    @property
    def batch_size(self):
        return self.tokens.shape[0]

    def to(self, device):
        return _Prefix(self.tokens.to(device))

    def expand(self, num_samples):
        return _Prefix(self.tokens.repeat_interleave(num_samples, dim=0))


class _ToyARDecoder(AutoregressiveDecoder):
    def __init__(self):
        self.fail = False
        self.malformed = None

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
        if self.fail:
            raise RuntimeError("decode failed")
        if cancelled is not None and cancelled():
            raise RuntimeError("cancelled")
        response = torch.tensor([[7, 2]], device=prefix.tokens.device)
        next_tokens = torch.cat([prefix.tokens, response], dim=1)
        logprob = torch.tensor([[-0.25, -0.5]], device=prefix.tokens.device)
        trace = DecodeTrace(
            token_ids=response[0],
            token_logprobs=logprob[0],
            action_mask=torch.ones(2, dtype=torch.bool, device=prefix.tokens.device),
            text="move forward 25cm<eos>",
            stop_reason="eos",
        )
        actions = torch.tensor([[[1.0, 25.0]]], device=prefix.tokens.device)
        return DecodeResult(
            actions=actions,
            behavior_logprob=logprob,
            recompute_state=response,
            next_memory=None if self.malformed == "missing_memory" else _Memory(next_tokens[0].clone()),
            traces=[trace],
        )

    def sample_with_logprob(self, prefix, num_steps, sigma, generator=None):
        result = self.decode(None, prefix, num_steps, prefix.batch_size, None, generator=generator)
        return result.actions, result.behavior_logprob, result.recompute_state

    def recompute_logprob(self, prefix, recompute_state, num_steps, sigma):
        del prefix, num_steps, sigma
        return torch.full(recompute_state.shape, -0.25, dtype=torch.float32)


class _ToyRecurrentPolicy(VLAPolicy):
    def __init__(self):
        super().__init__(
            VLAPolicyConfig(
                name="toy_recurrent",
                action_dim=2,
                action_horizon=1,
                default_num_steps=1,
            )
        )
        self._decoder = _ToyARDecoder()

    @property
    def is_recurrent(self):
        return True

    @property
    def decoder(self):
        return self._decoder

    def collate(self, observations, request_ids):
        return _Batch(torch.stack([obs.instruction_tokens for obs in observations]), request_ids)

    def encode_prefix(self, batch, memory=None):
        previous = (
            torch.empty(0, dtype=batch.tokens.dtype, device=batch.tokens.device)
            if memory is None
            else memory.to(batch.tokens.device).tokens
        )
        tokens = torch.cat([previous, batch.tokens[0]])[None]
        return _Prefix(tokens)


def _obs(token):
    return Observation(
        images=torch.zeros(1, 3, 2, 2),
        state=torch.zeros(1),
        instruction_tokens=torch.tensor([token]),
        instruction="test",
    )


def _core():
    return EngineCore(
        _ToyRecurrentPolicy(),
        EngineConfig(device="cpu", max_batch_size=1, use_cuda_graph=False, capture_full_loop=False),
    )


def test_recurrent_two_turns_commit_response_tokens_then_reset():
    core = _core()
    backend = GenerationBackend(core)
    key = SessionKey("env", "episode")

    first = backend.generate([_obs(3)], session_ids=[key])[0]
    assert first.actions.tolist() == [[1.0, 25.0]]
    assert first.trace.token_ids.tolist() == [7, 2]
    assert core._sessions.committed(key).tokens.tolist() == [3, 7, 2]

    backend.generate([_obs(4)], session_ids=[key])
    assert core._sessions.committed(key).tokens.tolist() == [3, 7, 2, 4, 7, 2]

    backend.reset_sessions([key])
    backend.generate([_obs(5)], session_ids=[key])
    assert core._sessions.committed(key).tokens.tolist() == [5, 7, 2]


def test_decode_failure_rolls_back_to_last_committed_memory():
    core = _core()
    backend = GenerationBackend(core)
    key = SessionKey("env", "episode")
    backend.generate([_obs(3)], session_ids=[key])
    committed = core._sessions.committed(key)

    core.policy.decoder.fail = True
    with pytest.raises(RuntimeError, match="decode failed"):
        backend.generate([_obs(4)], session_ids=[key])
    assert core._sessions.committed(key) is committed


def test_malformed_decode_result_rolls_back_to_last_committed_memory():
    core = _core()
    backend = GenerationBackend(core)
    key = SessionKey("env", "episode")
    backend.generate([_obs(3)], session_ids=[key])
    committed = core._sessions.committed(key)

    core.policy.decoder.malformed = "missing_memory"
    with pytest.raises(RuntimeError, match="did not return next_memory"):
        backend.generate([_obs(4)], session_ids=[key])
    assert core._sessions.committed(key) is committed


def test_weight_update_requires_recurrent_sessions_to_be_reset():
    backend = GenerationBackend(_core())
    key = SessionKey("env", "episode")
    backend.generate([_obs(1)], session_ids=[key])

    with pytest.raises(UnsupportedRecurrentModeError, match="reset recurrent sessions"):
        backend.update_weights({})

    backend.reset_sessions([key])
    backend.update_weights({})


def test_recurrent_request_requires_explicit_session():
    backend = GenerationBackend(_core())
    with pytest.raises(SessionRequiredError):
        backend.generate([_obs(1)])


def test_autoregressive_helpers_do_not_claim_generic_rl_capability():
    decoder = _core().policy.decoder
    assert isinstance(decoder, AutoregressiveDecoder)
    assert not isinstance(decoder, RLDecoder)


def test_recurrent_rollout_guards_precede_rl_capability_checks():
    backend = GenerationBackend(_core())

    with pytest.raises(SessionRequiredError, match="one distinct SessionKey"):
        backend.generate_with_logprob([_obs(1)])
    with pytest.raises(SessionRequiredError, match="one distinct SessionKey"):
        backend.sample_group([_obs(1)], group_size=2)
    keys = [SessionKey("env", "episode", branch) for branch in range(2)]
    with pytest.raises(UnsupportedRecurrentModeError, match="branch-safe prefix expansion"):
        backend.sample_group([_obs(1)], group_size=2, session_ids=keys)
    with pytest.raises(UnsupportedRecurrentModeError, match="selected candidate memory"):
        backend.best_of_n([_obs(1)], num_samples=2)


def test_recurrent_runtime_guards_are_explicit():
    policy = _ToyRecurrentPolicy()
    with pytest.raises(UnsupportedRecurrentModeError):
        EngineCore(policy, EngineConfig(device="cpu"))

    core = _core()
    with pytest.raises(UnsupportedRecurrentModeError):
        AsyncEngine(core)
    replica = InProcessReplica(core)
    assert replica.is_recurrent
    with pytest.raises(UnsupportedRecurrentModeError):
        core.execute_pipelined([])
