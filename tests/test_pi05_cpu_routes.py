"""CPU regression tests for PI0.5 eager parity and dynamic-time semantics."""

from types import SimpleNamespace

import torch

import embodiinfer.policies.pi05.modeling_pi05 as modeling_pi05
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy, Pi05Prefix, _adarms_projection, _mlp
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch


class _CountingProjection:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.output.expand(x.shape[:-1] + (self.output.shape[-1],))


class _AttentionBackend:
    def __init__(self):
        self.inputs = None

    def attend(self, q, k, v, **kwargs):
        self.inputs = (q, k, v)
        return v


def _policy_stub(*, attention: str = "eager"):
    policy = SimpleNamespace(
        attention=attention,
        native_embeddings=False,
        compile_backend="none",
        denoise_attention="sdpa",
        _attn=_AttentionBackend(),
    )
    return policy


def test_pi05_eager_attention_keeps_individual_qkv_projections(monkeypatch):
    policy = _policy_stub()
    h = torch.randn(1, 2, 4)
    projections = [_CountingProjection(torch.ones(1, 2, 4) * value) for value in (1.0, 2.0, 3.0)]
    attn = SimpleNamespace(
        head_dim=4,
        q_proj=projections[0],
        k_proj=projections[1],
        v_proj=projections[2],
        o_proj=torch.nn.Identity(),
        scaling=1.0,
    )

    def fail_if_fused(*_args, **_kwargs):
        raise AssertionError("eager PI0.5 must not use fused QKV projection")

    monkeypatch.setattr(modeling_pi05, "_fused_qkv", fail_if_fused)
    cos = torch.ones(1, 2, 4)
    sin = torch.zeros(1, 2, 4)
    Pi05Policy._attn_sublayer(policy, attn, h, cos, sin, None, None, [])

    assert [projection.calls for projection in projections] == [1, 1, 1]
    q, k, v = policy._attn.inputs
    assert torch.equal(q, torch.ones_like(q))
    assert torch.equal(k, torch.ones_like(k) * 2)
    assert torch.equal(v, torch.ones_like(v) * 3)


def test_pi05_tower_routes_eager_mlp_to_individual_projections(monkeypatch):
    captured = []

    def record_mlp(*args, **kwargs):
        captured.append((args, kwargs))
        return args[1]

    monkeypatch.setattr(modeling_pi05, "_mlp", record_mlp)
    policy = _policy_stub()
    policy._tower_forward = Pi05Policy._tower_forward.__get__(policy, type(policy))
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(q_proj=SimpleNamespace(weight=torch.empty(4, 4))),
        input_layernorm=object(),
        post_attention_layernorm=object(),
        mlp=object(),
    )
    tower = SimpleNamespace(
        layers=[layer],
        norm=object(),
        rotary_emb=lambda hidden, positions: (None, None),
    )

    monkeypatch.setattr(modeling_pi05, "_rmsnorm", lambda *args, **kwargs: (args[1], None))
    policy._attn_sublayer = lambda *args, **kwargs: torch.zeros_like(args[1])
    monkeypatch.setattr(
        modeling_pi05,
        "_gated_residual",
        lambda x, y, gate, *args, **kwargs: x,
    )

    hidden = torch.randn(1, 2, 4)
    policy._tower_forward(tower, hidden, torch.zeros(1, 2, dtype=torch.long), None, None)

    assert captured[0][1]["fuse_projections"] is False


def test_pi05_eager_prefix_uses_native_embed_prefix(monkeypatch):
    calls = []
    prefix_embs = torch.randn(1, 3, 4)
    prefix_pad_masks = torch.ones(1, 3, dtype=torch.bool)
    prefix_att_masks = torch.zeros(1, 3, dtype=torch.bool)

    class _Model:
        def embed_prefix(self, images, image_masks, tokens, masks):
            calls.append((images, image_masks, tokens, masks))
            return prefix_embs, prefix_pad_masks, prefix_att_masks

    policy = _policy_stub()
    policy.prefix_attention = "sdpa"
    policy._m = _Model()
    policy._attention_mask_4d = lambda allowed: None
    policy._prefix_tower = object()
    policy._cached_prefix_meta = None
    policy._make_att_2d_masks = lambda pad, att: torch.zeros(1, 3, 3, dtype=torch.bool)
    policy._tower_forward = lambda *args, **kwargs: (
        args[1],
        [(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4))],
    )
    batch = Pi05Batch(
        images=[torch.zeros(1, 3, 2, 2)],
        img_masks=[torch.ones(1, dtype=torch.bool)],
        tokens=torch.ones(1, 3, dtype=torch.long),
        masks=torch.ones(1, 3, dtype=torch.bool),
    )

    def fail_if_batched(*_args, **_kwargs):
        raise AssertionError("eager PI0.5 must use model.embed_prefix")

    monkeypatch.setattr(modeling_pi05, "_embed_prefix_batched", fail_if_batched)
    prefix = Pi05Policy._encode_prefix_impl(policy, batch)

    assert len(calls) == 1
    assert prefix.batch_size == 1
    assert policy._cached_prefix_meta == (3, torch.bool, torch.float32)


def _dynamic_time_policy():
    m = SimpleNamespace(
        action_in_proj=torch.nn.Linear(2, 3, bias=False),
        time_mlp_in=torch.nn.Linear(3, 3),
        time_mlp_out=torch.nn.Linear(3, 3),
        action_out_proj=torch.nn.Linear(3, 2, bias=False),
        config=SimpleNamespace(min_period=1.0, max_period=10.0, chunk_size=1),
    )
    with torch.no_grad():
        m.action_in_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        m.time_mlp_in.weight.fill_(1.0)
        m.time_mlp_in.bias.zero_()
        m.time_mlp_out.weight.fill_(1.0)
        m.time_mlp_out.bias.zero_()
        m.action_out_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    policy = SimpleNamespace(
        _m=m,
        training=False,
        prefix_attention="sdpa",
        native_embeddings=False,
        _sinusoidal=lambda t, dim, **kwargs: t[:, None].expand(-1, dim),
        _suffix_att_masks=lambda length, dtype, device: torch.ones(1, length, dtype=dtype, device=device),
        _make_att_2d_masks=lambda pad, att: torch.zeros(
            pad.shape[0], pad.shape[1], pad.shape[1], dtype=torch.bool, device=pad.device
        ),
        _expert_tower=object(),
    )
    policy._attention_mask_4d = lambda allowed: None
    policy._tower_forward = lambda tower, hidden, positions, mask, adarms_cond, **kwargs: (
        adarms_cond[:, None, :],
        None,
    )
    policy._embed_suffix = Pi05Policy._embed_suffix.__get__(policy, type(policy))
    return policy


def test_pi05_denoise_step_recomputes_time_during_capture(monkeypatch):
    policy = _dynamic_time_policy()
    x_t = torch.zeros(1, 1, 2)
    t = torch.tensor([0.2])
    prefix = Pi05Prefix([], torch.ones(1, 1, dtype=torch.bool), all_valid=True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with torch.no_grad():
        first = Pi05Policy._denoise_step_impl(policy, x_t, t, prefix).clone()
        t.fill_(0.8)
        second = Pi05Policy._denoise_step_impl(policy, x_t, t, prefix).clone()

    assert not torch.equal(first, second)


def test_pi05_adarms_projection_recomputes_same_pointer_during_capture(monkeypatch):
    norm = SimpleNamespace(training=False, dense=torch.nn.Linear(3, 3, bias=False))
    with torch.no_grad():
        norm.dense.weight.copy_(torch.eye(3))
    cond = torch.tensor([[0.1, 0.2, 0.3]])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with torch.no_grad():
        first = _adarms_projection(norm, cond).clone()
        cond.fill_(0.9)
        second = _adarms_projection(norm, cond).clone()

    assert not torch.equal(first, second)


def test_pi05_mlp_reference_branch_is_independent_of_fused_helper(monkeypatch):
    gate = _CountingProjection(torch.full((1, 1, 2), 2.0))
    up = _CountingProjection(torch.full((1, 1, 2), 3.0))
    down = _CountingProjection(torch.ones(1, 1, 2))
    mlp = SimpleNamespace(gate_proj=gate, up_proj=up, down_proj=down)
    monkeypatch.setattr(
        modeling_pi05,
        "_fused_linear_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fused helper used")),
    )

    _mlp(mlp, torch.zeros(1, 1, 2), use_inductor=False, fuse_projections=False)

    assert gate.calls == 1
    assert up.calls == 1
    assert down.calls == 1
