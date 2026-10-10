"""Pi05 workspace ownership and native execution guards."""

from types import SimpleNamespace

import pytest
import torch

from embodiinfer.layers import OperatorRequest
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.inference.config import Pi05OptimizationConfig
from embodiinfer.policies.pi05.inference.operators import Pi05OperatorPlans
from embodiinfer.policies.pi05.inference.runtime import _compact_all_valid
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy, Pi05Prefix
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch


def test_openpi_rlinf_numerics_reject_silent_eager_fallback():
    policy = SimpleNamespace(
        _optimization_config=Pi05OptimizationConfig(numerics="openpi_rlinf", activation="gelu_pytorch_exact"),
        _native_enabled=lambda: False,
    )
    with pytest.raises(RuntimeError, match="CUDA native inference"):
        Pi05Policy._get_optimizations(policy)
    policy._optimization_config = Pi05OptimizationConfig()
    assert Pi05Policy._get_optimizations(policy) is None


def test_folded_prefix_gathers_holes_without_reordering_valid_tokens():
    tokens = torch.tensor([[9, 8, 7, 6, 5]])
    batch = Pi05Batch(
        [torch.zeros(1, 3, 2, 2)],
        [torch.tensor([True])],
        tokens,
        torch.tensor([[True, False, True, False, True]]),
        ["a"],
    )
    compacted = _compact_all_valid(batch)
    assert compacted.tokens.tolist() == [[9, 7, 5]]
    assert compacted.masks.all()
    assert compacted.request_ids == ["a"]
    assert torch.equal(batch.tokens, tokens)


def workspace_runtime(monkeypatch):
    import embodiinfer.policies.pi05.inference.operators as optimization

    monkeypatch.setattr(optimization, "_stream_key", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runtime = Pi05OperatorPlans.__new__(Pi05OperatorPlans)
    attn = object()
    runtime.policy = SimpleNamespace(
        config=VLAPolicyConfig(action_horizon=2),
        _expert_tower=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
    )
    runtime.config = Pi05OptimizationConfig(norm_fusion=False)
    runtime.kv_buffers = {}
    runtime.active_kv = None
    runtime.active_sources = {}
    runtime._scope = None
    return runtime, attn


def test_kv_workspace_refreshes_prefix_reuses_storage_and_does_not_mutate_source(monkeypatch):
    runtime, attn = workspace_runtime(monkeypatch)
    prefix = Pi05Prefix(
        [(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))], torch.ones(1, 3, dtype=torch.bool)
    )
    preserved = [t.clone() for t in prefix.kv[0]]
    suffix = (torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
    with runtime.decode_context(prefix):
        first = runtime.combine_kv(attn, *suffix, prefix.kv[0])
        for result, cached, current in zip(first, prefix.kv[0], suffix):
            assert torch.equal(result, torch.cat((cached, current), dim=2))
        pointers = [x.data_ptr() for x in first]
        with pytest.raises(ValueError, match="active decode prefix"):
            runtime.combine_kv(attn, *suffix, tuple(t.clone() for t in prefix.kv[0]))
    for actual, expected in zip(prefix.kv[0], preserved):
        assert torch.equal(actual, expected)
    for value in prefix.kv[0]:
        value.add_(10)
    with runtime.decode_context(prefix):
        second = runtime.combine_kv(attn, *suffix, prefix.kv[0])
        assert [x.data_ptr() for x in second] == pointers
        for result, cached, current in zip(second, prefix.kv[0], suffix):
            assert torch.equal(result, torch.cat((cached, current), dim=2))
    assert runtime.active_kv is None
    assert runtime.active_sources == {}


def test_workspace_failure_unwinds_context(monkeypatch):
    runtime, attn = workspace_runtime(monkeypatch)
    prefix = Pi05Prefix(
        [(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4))], torch.ones(1, 3, dtype=torch.bool)
    )
    with pytest.raises(ValueError, match="warmed workspace layout"), runtime.decode_context(prefix):
        runtime.combine_kv(attn, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), prefix.kv[0])
    assert runtime.active_kv is None


@pytest.mark.gpu
def test_openpi_rlinf_rotary_registry_preserves_reference_bytes():
    from embodiinfer.layers import rotary_backends
    from embodiinfer.policies.pi05.embeddings import openpi_rlinf_rope_tables

    request = OperatorRequest("cuda", torch.bfloat16, cuda_graph=True)
    reference = rotary_backends.get("torch", request)
    migrated = rotary_backends.get("cuda", request)
    inputs = torch.randn(2, 4, 10, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    cosine, sine = openpi_rlinf_rope_tables(torch.arange(10, device="cuda")[None].expand(2, -1), 256)
    expected = reference(inputs, sine, cosine)
    actual = migrated(inputs, sine, cosine)
    assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
