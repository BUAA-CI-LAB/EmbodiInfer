"""Regression coverage for fp32 action state with a bf16 backbone."""

from types import SimpleNamespace

import pytest
import torch

from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.policies.base import FlowVLAPolicy
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy, Pi05Prefix
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch


class MixedFlow(FlowVLAPolicy):
    def __init__(self):
        super().__init__(VLAPolicyConfig(action_dim=2, action_horizon=3, default_num_steps=2))
        self.backbone = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.bfloat16))
        self.action_proj = torch.nn.Linear(2, 2)

    @property
    def execution_dtype(self):
        return self.action_proj.weight.dtype

    def encode_prefix(self, batch):
        return self.backbone.float()

    def denoise_step(self, x_t, t, prefix):
        assert x_t.dtype == t.dtype == self.action_proj.weight.dtype
        return self.action_proj(x_t) + t[:, None, None] + prefix

    def pad(self, batch, target_batch_size):
        assert target_batch_size == batch.batch_size
        return batch


def batch(device="cpu"):
    return Pi05Batch(
        [],
        [],
        torch.ones(1, 2, dtype=torch.long, device=device),
        torch.ones(1, 2, dtype=torch.bool, device=device),
        ["sample"],
    )


def test_flow_noise_and_time_use_action_dtype():
    policy = MixedFlow()
    assert policy.new_noise(1).dtype == torch.float32
    assert policy.decoder.init_state(1).dtype == torch.float32
    direct = policy.sample_actions(batch(), generator=torch.Generator().manual_seed(7))
    state = policy.decoder.init_state(1, torch.Generator().manual_seed(7))
    decoded = policy.decoder.produce_chunk(state, policy.encode_prefix(batch()), 2, 1, None)
    torch.testing.assert_close(decoded, direct, atol=0, rtol=0)


@pytest.mark.gpu
def test_engine_auto_preserves_mixed_weights():
    policy = MixedFlow().cuda()
    core = EngineCore(policy, EngineConfig(device="cuda", dtype="auto", use_cuda_graph=False))
    assert policy.backbone.dtype == torch.bfloat16
    assert policy.action_proj.weight.dtype == core.dtype == torch.float32
    a = core.execute(batch("cuda"), generator=torch.Generator(device="cuda").manual_seed(7))[0].actions
    b = policy.sample_actions(batch("cuda"), generator=torch.Generator(device="cuda").manual_seed(7))[0].cpu()
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_pi05_static_kv_preserves_prefix_dtype():
    policy = Pi05Policy.__new__(Pi05Policy)
    FlowVLAPolicy.__init__(policy, VLAPolicyConfig())
    attn = SimpleNamespace(head_dim=4, k_proj=SimpleNamespace(out_features=4))
    policy._prefix_tower = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    policy._cached_prefix_meta = (9, torch.bool, torch.bfloat16)
    static = policy.allocate_static_prefix(1, torch.device("cpu"), torch.float32)
    assert static.kv[0][0].shape == (1, 1, 9, 4)
    assert static.kv[0][0].dtype == static.kv[0][1].dtype == torch.bfloat16
    assert static.prefix_pad_masks.dtype == torch.bool


def test_pi05_compact_graph_keeps_live_kv_dtype():
    policy = Pi05Policy.__new__(Pi05Policy)
    FlowVLAPolicy.__init__(policy, VLAPolicyConfig())
    policy.native_inference = True
    key = torch.randn(1, 1, 5, 4, dtype=torch.bfloat16)
    live = Pi05Prefix([(key, key.clone())], torch.ones(1, 5, dtype=torch.bool), all_valid=True)
    static = policy.allocate_static_prefix_from_live(live, 1, torch.device("cpu"), torch.float32)
    assert static.kv[0][0].dtype == torch.bfloat16
    assert static.kv[0][0].data_ptr() != key.data_ptr()
    torch.testing.assert_close(static.kv[0][0], key, atol=0, rtol=0)
    with pytest.raises(ValueError, match="batch must match"):
        policy.allocate_static_prefix_from_live(live, 2, torch.device("cpu"), torch.float32)


def test_engine_cpu_retains_fp32_behavior():
    core = EngineCore(MixedFlow(), EngineConfig(device="cpu", dtype="auto", use_cuda_graph=False))
    assert core.dtype == torch.float32
    assert {p.dtype for p in core.policy.parameters()} == {torch.float32}


@pytest.mark.gpu
def test_engine_explicit_dtype_still_casts_uniformly():
    core = EngineCore(MixedFlow(), EngineConfig(device="cuda", dtype="bfloat16", use_cuda_graph=False))
    assert core.dtype == torch.bfloat16
    assert {p.dtype for p in core.policy.parameters()} == {torch.bfloat16}
