import pytest
import torch

from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.engine.rollout.generation_backend import GenerationBackend
from embodiinfer.policies.base import DenseKVPrefix, VLAPolicy
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.cosmos.modeling_cosmos import CosmosDiffusionDecoder
from embodiinfer.policies.decoder import (
    ActionDecoder,
    AutoregressiveDecoder,
    FlowDecoder,
    ParallelDecoder,
    RLDecoder,
)
from embodiinfer.types import Observation


class _ServingDecoder(ActionDecoder):
    def init_state(self, batch_size, generator=None):
        del batch_size, generator
        return None

    def produce_chunk(self, state, prefix, num_steps, bucket, graphs):
        del state, prefix, num_steps, graphs
        return torch.arange(bucket, dtype=torch.float32)[:, None, None]


class _ServingPolicy(VLAPolicy):
    def __init__(self):
        super().__init__(
            VLAPolicyConfig(
                name="serving_only",
                action_dim=1,
                action_horizon=1,
                default_num_steps=1,
            )
        )
        self._decoder = _ServingDecoder()

    @property
    def decoder(self):
        return self._decoder

    def encode_prefix(self, batch):
        return batch


class _CategoricalHead:
    def sample(self, logits, **kwargs):
        del kwargs
        batch = logits.shape[0]
        rows = torch.arange(batch, device=logits.device)
        token_ids = rows[:, None].expand(batch, 2)
        # For each group of three candidates, candidate 1 has the largest
        # sequence log-probability: [-4, -1, -2].
        totals = torch.tensor([-4.0, -1.0, -2.0], device=logits.device)[rows % 3]
        return token_ids, totals[:, None].expand(batch, 2) / 2

    def tokens_to_actions(self, token_ids):
        return token_ids[:, :1, None].float()

    def recompute_logprob(self, logits, token_ids, **kwargs):
        del logits, kwargs
        rows = token_ids[:, 0]
        totals = torch.tensor([-4.0, -1.0, -2.0], device=token_ids.device)[rows % 3]
        return totals[:, None].expand_as(token_ids) / 2


class _CategoricalPolicy(VLAPolicy):
    def __init__(self):
        super().__init__(
            VLAPolicyConfig(
                name="categorical",
                action_dim=1,
                action_horizon=1,
                default_num_steps=1,
            )
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.head = _CategoricalHead()
        self.sample_temperature = 1.0
        self.sample_top_k = None
        self._decoder = ParallelDecoder(self)

    @property
    def decoder(self):
        return self._decoder

    def encode_prefix(self, batch, memory=None):
        del memory
        kv = torch.zeros(batch.batch_size, 1, 1, 1, device=self.anchor.device)
        return DenseKVPrefix([(kv, kv)], batch.batch_size)

    def decode_action_logits(self, prefix, graphs=None, bucket=None):
        del graphs, bucket
        return torch.zeros(prefix.batch_size, 2, 8, device=self.anchor.device)


def _observation():
    return Observation(
        images=torch.zeros(1, 3, 2, 2),
        state=torch.zeros(1),
        instruction_tokens=torch.zeros(1, dtype=torch.long),
    )


def test_decoder_capability_tree_matches_runtime_contracts():
    assert issubclass(FlowDecoder, RLDecoder)
    assert issubclass(ParallelDecoder, RLDecoder)
    assert issubclass(AutoregressiveDecoder, ActionDecoder)
    assert not issubclass(AutoregressiveDecoder, RLDecoder)
    assert issubclass(CosmosDiffusionDecoder, ActionDecoder)
    assert not issubclass(CosmosDiffusionDecoder, RLDecoder)


def test_action_decoder_default_decode_preserves_serving_output():
    decoder = _ServingDecoder()
    result = decoder.decode(None, object(), num_steps=1, bucket=2, graphs=None)

    assert result.actions.tolist() == [[[0.0]], [[1.0]]]
    assert result.next_memory is None
    assert result.behavior_logprob is None


def test_generation_backend_rejects_stateless_non_rl_decoder_before_batch_prep():
    core = EngineCore(
        _ServingPolicy(),
        EngineConfig(device="cpu", max_batch_size=1, use_cuda_graph=False),
    )
    backend = GenerationBackend(core)

    with pytest.raises(TypeError, match="is not an RLDecoder"):
        backend.generate_with_logprob([])


def test_backend_preserves_categorical_token_logprobs_and_reduces_best_of_n():
    backend = GenerationBackend(
        EngineCore(
            _CategoricalPolicy(),
            EngineConfig(device="cpu", max_batch_size=8, use_cuda_graph=False),
        )
    )
    observations = [_observation(), _observation()]

    actions, logprob = backend.generate_with_logprob(observations, num_samples=3)

    assert actions.shape == (2, 3, 1, 1)
    assert logprob.shape == (2, 3, 2)
    torch.testing.assert_close(logprob.sum(dim=-1), torch.tensor([[-4.0, -1.0, -2.0]] * 2))

    picked = backend.best_of_n(observations, num_samples=3)

    assert [item.actions.item() for item in picked] == [1.0, 4.0]
    assert [item.value for item in picked] == [-1.0, -1.0]
    assert all(item.logprob.shape == (2,) for item in picked)
