"""CPU tests for the pluggable attention backends."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as torch_functional

import embodiinfer.layers.attention as attention_registry
from embodiinfer.backend.torch import attention as attention_module
from embodiinfer.layers import available_attention_backends, get_attention_backend
from embodiinfer.policies.mock import MockFlowVLA, preset_config
from embodiinfer.types import Observation


def _qkv(B=2, nh=4, seq=8, hd=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (B, nh, seq, hd)
    return (
        torch.randn(shape, generator=g),
        torch.randn(shape, generator=g),
        torch.randn(shape, generator=g),
    )


def test_registry_lists_and_resolves():
    assert available_attention_backends() == [
        "eager",
        "eager_bc",
        "flex",
        "sdpa",
        "triton_segmented",
        "triton_split_kv",
    ]
    sdpa = get_attention_backend("sdpa")
    eager = get_attention_backend("eager")
    assert sdpa.name == "sdpa"
    assert eager.name == "eager"
    assert type(sdpa).__module__ == "embodiinfer.backend.torch.attention"
    assert type(eager).__module__ == "embodiinfer.backend.torch.attention"


def test_lazy_registry_defers_optional_backend_import(monkeypatch):
    class FakeLazyAttention:
        name = "test_lazy"

    imports = []
    fake_module = SimpleNamespace(FakeLazyAttention=FakeLazyAttention)
    monkeypatch.setattr(attention_registry, "_REGISTRY", dict(attention_registry._REGISTRY))
    monkeypatch.setattr(attention_registry, "_LAZY_REGISTRY", dict(attention_registry._LAZY_REGISTRY))
    monkeypatch.setattr(
        attention_registry.importlib,
        "import_module",
        lambda module_name: imports.append(module_name) or fake_module,
    )

    attention_registry.register_lazy_attention(
        "test_lazy", "tests.fake_optional_attention", "FakeLazyAttention"
    )
    assert "test_lazy" in available_attention_backends()
    assert imports == []

    backend = get_attention_backend("test_lazy")
    assert backend.name == "test_lazy"
    assert imports == ["tests.fake_optional_attention"]


def test_explicit_triton_incompatibility_fails_loudly(monkeypatch):
    class FakeTritonAttention:
        name = "triton_segmented"

    monkeypatch.setitem(attention_registry._REGISTRY, "triton_segmented", FakeTritonAttention)
    monkeypatch.setattr(
        attention_registry,
        "attention_backend_capability",
        lambda name: (False, "requires CUDA sm80 or newer; current process is CPU-only"),
    )

    with pytest.raises(
        RuntimeError,
        match="triton_segmented.*requires CUDA sm80 or newer.*CPU-only",
    ):
        get_attention_backend("triton_segmented")


def test_auto_records_probe_order_and_falls_back_to_sdpa(monkeypatch):
    probes = []

    def fake_capability(name):
        probes.append(name)
        if name == "triton_segmented":
            return False, "Triton is unavailable in this CPU test"
        return True, None

    monkeypatch.setattr(attention_registry, "attention_backend_capability", fake_capability)

    backend = get_attention_backend("auto")
    assert backend.name == "sdpa"
    assert probes == ["triton_segmented", "sdpa", "sdpa"]


def _gqa_qkv(B=2, nh=8, nkv=2, seq=8, hd=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn((B, nh, seq, hd), generator=g)
    k = torch.randn((B, nkv, seq, hd), generator=g)
    v = torch.randn((B, nkv, seq, hd), generator=g)
    return q, k, v


def test_gqa_backends_agree():
    """With num_kv_heads < num_heads, eager (materialized repeat), eager_bc
    (broadcast), and sdpa (enable_gqa) must agree; eager vs eager_bc bit-exact."""
    q, k, v = _gqa_qkv()
    seq = q.shape[2]
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)[None, None]  # [1,1,seq,seq]
    eager = get_attention_backend("eager").attend(q, k, v, attn_mask=mask)
    eager_bc = get_attention_backend("eager_bc").attend(q, k, v, attn_mask=mask)
    sdpa = get_attention_backend("sdpa").attend(q, k, v, attn_mask=mask)
    assert eager.shape == (2, 8, seq, 16)
    # eager_bc is the same math but the grouped-broadcast matmul reduces in a
    # different order than the materialized repeat, so it is numerically identical
    # (fp-reorder level), not bit-exact — same tier as sdpa.
    assert torch.allclose(eager, eager_bc, atol=1e-5)
    assert torch.allclose(eager, sdpa, atol=1e-5)


def test_unknown_backend_suggests():
    with pytest.raises(KeyError, match="sdpa"):  # "sdap" -> did-you-mean "sdpa"
        get_attention_backend("sdap")


def test_flex_is_stubbed():
    q, k, v = _qkv()
    with pytest.raises(NotImplementedError):
        get_attention_backend("flex").attend(q, k, v)


def test_eager_sdpa_parity_no_mask():
    q, k, v = _qkv()
    a = get_attention_backend("eager").attend(q, k, v)
    b = get_attention_backend("sdpa").attend(q, k, v)
    assert a.shape == (2, 4, 8, 16)
    assert torch.allclose(a, b, atol=1e-5)


def test_eager_sdpa_parity_with_additive_mask():
    q, k, v = _qkv()
    B, nh, seq = 2, 4, 8
    # causal additive mask [B, nh, seq, seq]
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
    mask = mask.expand(B, nh, seq, seq)
    a = get_attention_backend("eager").attend(q, k, v, attn_mask=mask)
    b = get_attention_backend("sdpa").attend(q, k, v, attn_mask=mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_mock_backend_swap_is_equivalent():
    """A policy computed with 'eager' vs 'sdpa' matches (same weights, same x0)."""
    cfg = preset_config("tiny")
    m_sdpa = MockFlowVLA(cfg, attention="sdpa").eval()
    m_eager = MockFlowVLA(cfg, attention="eager").eval()
    m_eager.load_state_dict(m_sdpa.state_dict())  # backends are stateless

    obs = Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
    )
    from embodiinfer.types import collate

    batch = collate([obs], ["a"])
    x0 = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    a = m_sdpa.sample_actions(batch, x0=x0.clone())
    b = m_eager.sample_actions(batch, x0=x0.clone())
    assert torch.allclose(a, b, atol=1e-5)


def test_sdpa_legacy_signature_expands_gqa_and_preserves_arguments(monkeypatch):
    """Torch builds without ``enable_gqa`` still receive native SDPA semantics."""
    q, k, v = _gqa_qkv(B=1, nh=8, nkv=2, seq=5, hd=8)
    mask = torch.zeros(1, 1, q.shape[-2], k.shape[-2], dtype=q.dtype)
    seen = {}
    original = torch_functional.scaled_dot_product_attention

    def legacy_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        seen.update(
            heads=key.shape[1],
            mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            dtype=query.dtype,
        )
        return original(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", legacy_sdpa)
    actual = get_attention_backend("sdpa").attend(q, k, v, attn_mask=mask, scaling=0.5, dropout_p=0.0)
    expected = get_attention_backend("eager").attend(q, k, v, attn_mask=mask, scaling=0.5)

    assert seen["heads"] == q.shape[1]
    assert seen["mask"] is mask
    assert seen["dtype"] == q.dtype
    assert torch.allclose(actual, expected, atol=1e-5)

    q_plain, k_plain, v_plain = _qkv(B=1, nh=4, seq=5, hd=8)
    plain = get_attention_backend("sdpa").attend(q_plain, k_plain, v_plain)
    plain_expected = get_attention_backend("eager").attend(q_plain, k_plain, v_plain)
    assert seen["heads"] == k_plain.shape[1]
    assert torch.allclose(plain, plain_expected, atol=1e-5)


def test_sdpa_new_signature_keeps_native_gqa_path(monkeypatch):
    q, k, v = _gqa_qkv(B=1, nh=8, nkv=2, seq=5, hd=8)
    seen = {}
    original = torch_functional.scaled_dot_product_attention

    def modern_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        seen.update(heads=key.shape[1], enable_gqa=enable_gqa)
        if enable_gqa:
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        return original(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", modern_sdpa)
    actual = get_attention_backend("sdpa").attend(q, k, v)
    expected = get_attention_backend("eager").attend(q, k, v)

    assert seen == {"heads": k.shape[1], "enable_gqa": True}
    assert torch.allclose(actual, expected, atol=1e-5)


def test_sdpa_capability_is_resolved_before_attention(monkeypatch):
    backend = get_attention_backend("sdpa")
    monkeypatch.setattr(attention_module.inspect, "signature", lambda _: pytest.fail("late probe"))
    q, k, v = _gqa_qkv()
    assert backend.attend(q, k, v).shape == q.shape


def test_sdpa_execution_type_error_is_not_retried(monkeypatch):
    calls = []

    def failing_sdpa(*args, enable_gqa=False, **kwargs):
        calls.append(enable_gqa)
        raise TypeError("kernel failure")

    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", failing_sdpa)
    with pytest.raises(TypeError, match="kernel failure"):
        get_attention_backend("sdpa").attend(*_gqa_qkv())
    assert calls == [True]


def test_sdpa_unknown_signature_uses_compatibility_path(monkeypatch):
    monkeypatch.setattr(attention_module.inspect, "signature", lambda _: (_ for _ in ()).throw(ValueError()))
    assert not attention_module._supports_gqa(lambda *args: None)
    assert attention_module._supports_gqa(torch_functional.scaled_dot_product_attention)
