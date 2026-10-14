"""CPU bit-exact parity and policy routing for the linear operator registration."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from embodiinfer.backend.triton.tuned import KernelProfile, TunedKernel
from embodiinfer.engine.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelContext,
)
from embodiinfer.layers import KernelSelection, available_ops, get_op
from embodiinfer.models.linear import QuantizedLinear
from embodiinfer.policies.pi05 import modeling_pi05 as pi05

_DTYPES = (torch.float32, torch.bfloat16)


@pytest.fixture(autouse=True)
def _fixed_cpu_numerics():
    state = torch.get_rng_state()
    precision = torch.get_float32_matmul_precision()
    torch.random.default_generator.manual_seed(20261014)
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_rng_state(state)
        torch.set_float32_matmul_precision(precision)


def _equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def _profile(applies=None):
    assert "linear" in available_ops(), "linear must be registered before profile routing"
    calls = []

    def record(x, weight, bias=None):
        output = F.linear(x, weight, bias)
        args = (x, weight, bias)
        calls.append(
            SimpleNamespace(
                args=args,
                metadata=[None if t is None else (t.shape, t.stride(), t.dtype) for t in args],
                output=output,
            )
        )
        return output

    profile = KernelProfile(
        name="linear_cpu_test",
        root=Path(__file__).parent,
        match={},
        ops={"linear": {}},
        _loaded={"linear": TunedKernel(record, applies)},
    )
    return KernelSelection(profile), calls


def _call_is(call, x, weight, bias):
    for actual, expected in zip(call.args, (x, weight, bias), strict=True):
        assert actual is expected
    assert call.metadata == [None if t is None else (t.shape, t.stride(), t.dtype) for t in (x, weight, bias)]


def _pair(x, first, second, selection):
    assert "kernels" in inspect.signature(pi05._fused_linear_pair).parameters
    return pi05._fused_linear_pair(x, first, second, kernels=selection)


def _dense(in_features, out_features, dtype, bias=True):
    return nn.Linear(in_features, out_features, bias=bias, dtype=dtype).eval()


def _mlp(dtype, bias=True):
    return SimpleNamespace(
        gate_proj=_dense(7, 11, dtype, False),
        up_proj=_dense(7, 11, dtype, False),
        down_proj=_dense(11, 7, dtype, bias),
    )


def _original_mlp(mlp, x, fuse=True):
    if fuse:
        weight = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0).contiguous()
        gate, up = F.linear(x, weight).split((11, 11), dim=-1)
    else:
        gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    return mlp.down_proj(F.gelu(gate, approximate="tanh") * up)


def _positive_control(selection, calls, dtype):
    first, second = _dense(7, 5, dtype, False), _dense(7, 9, dtype, False)
    x = torch.randn(2, 3, 7, dtype=dtype)
    with torch.no_grad():
        result = _pair(x, first, second, selection)
    assert len(calls) == 1
    _call_is(calls[0], x, first._vvla_fused_pair_weight, None)
    for actual, expected in zip(result, calls[0].output.split((5, 9), dim=-1), strict=True):
        _equal(actual, expected)
    calls.clear()


class _Attention:
    def __init__(self):
        self.calls = []

    def attend(self, q, k, v, *, scaling, attn_mask=None):
        self.calls.append((q, k, v, scaling))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scaling)


class _Activation:
    def __init__(self):
        self.calls = []

    def __call__(self, x):
        result = F.gelu(x, approximate="none")
        self.calls.append((x, result))
        return result


def _vision(dtype, selection):
    attention = SimpleNamespace(
        q_proj=_dense(8, 8, dtype),
        k_proj=_dense(8, 8, dtype),
        v_proj=_dense(8, 8, dtype),
        out_proj=_dense(8, 8, dtype),
        num_heads=2,
        head_dim=4,
        scale=0.5,
    )
    layer = SimpleNamespace(
        self_attn=attention,
        layer_norm1=nn.LayerNorm(8, dtype=dtype),
        layer_norm2=nn.LayerNorm(8, dtype=dtype),
        mlp=SimpleNamespace(fc1=_dense(8, 12, dtype), fc2=_dense(12, 8, dtype), activation_fn=_Activation()),
    )
    vision = SimpleNamespace(
        embeddings=SimpleNamespace(
            patch_embedding=nn.Conv2d(3, 8, 2, stride=2, dtype=dtype),
            position_embedding=nn.Embedding(4, 8, dtype=dtype),
            position_ids=torch.arange(4).unsqueeze(0),
        ),
        encoder=SimpleNamespace(layers=[layer]),
        post_layernorm=nn.LayerNorm(8, dtype=dtype),
    )
    pg = SimpleNamespace(
        vision_tower=SimpleNamespace(vision_model=vision),
        multi_modal_projector=SimpleNamespace(linear=_dense(8, 7, dtype)),
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=7)),
    )
    policy = SimpleNamespace(
        _m=SimpleNamespace(paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(model=pg))),
        _kernels=selection,
        _vision_attn=_Attention(),
    )
    slots = [(attention, name) for name in ("q_proj", "k_proj", "v_proj", "out_proj")]
    slots += [(layer.mlp, "fc1"), (layer.mlp, "fc2")]
    return policy, layer, slots


def _original_image(policy, image):
    # Frozen S3–S6 expression sequence, including entry/exit casts and layouts.
    pg = policy._m.paligemma_with_expert.paligemma.model
    vision = pg.vision_tower.vision_model
    emb = vision.embeddings
    h = emb.patch_embedding(image.to(emb.patch_embedding.weight.dtype))
    h = h.flatten(2).transpose(1, 2)
    h = h + emb.position_embedding(emb.position_ids)
    for layer in vision.encoder.layers:
        x = layer.layer_norm1(h)
        a = layer.self_attn
        b, s, _ = x.shape
        q, k, v = [
            proj(x).view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
            for proj in (a.q_proj, a.k_proj, a.v_proj)
        ]
        y = policy._vision_attn.attend(q, k, v, scaling=a.scale)
        h = h + a.out_proj(y.transpose(1, 2).reshape(b, s, -1).contiguous())
        x = layer.layer_norm2(h)
        h = h + layer.mlp.fc2(layer.mlp.activation_fn(layer.mlp.fc1(x)))
    h = vision.post_layernorm(h)
    h = pg.multi_modal_projector.linear(h)
    scale = pg.config.text_config.hidden_size**0.5
    return ((h / scale) * scale).to(image.dtype)


def _track(module):
    module.route_inputs = []
    module.register_forward_hook(lambda owner, args, output: owner.route_inputs.append(args[0]))
    return module


def _check_module_bypass(make, selection, calls, dtype):
    x = torch.randn(2, 3, 7, dtype=dtype)
    mlp = _mlp(dtype)
    mlp.down_proj = make(11, 7, dtype, True)
    expected = _original_mlp(mlp, x, fuse=False)
    mlp.down_proj.route_inputs.clear()
    actual = pi05._mlp(mlp, x, False, fuse_projections=False, kernels=selection)
    _equal(actual, expected)
    assert len(mlp.down_proj.route_inputs) == 1
    assert calls == []

    policy, _, slots = _vision(dtype, selection)
    for owner, name in slots:
        original = getattr(owner, name)
        setattr(owner, name, make(original.in_features, original.out_features, dtype, True))
    image = torch.randn(2, 3, 4, 4, dtype=dtype)
    expected = _original_image(policy, image)
    for owner, name in slots:
        getattr(owner, name).route_inputs.clear()
    _equal(pi05.Pi05Policy._embed_image(policy, image), expected)
    assert all(len(getattr(owner, name).route_inputs) == 1 for owner, name in slots)
    assert calls == []


def test_linear_reference_fp32_bf16_matches_original_expression():
    from embodiinfer.backend.torch import ops

    assert callable(getattr(ops, "linear", None)), "Torch linear reference is missing"
    from embodiinfer.backend.torch.ops import linear

    for dtype in _DTYPES:
        inputs = [torch.randn(*shape, dtype=dtype) for shape in ((7,), (4, 7), (2, 3, 7))]
        inputs.append(torch.randn(2, 7, 3, dtype=dtype).transpose(1, 2))
        assert not inputs[-1].is_contiguous()
        for has_bias in (False, True):
            module = _dense(7, 5, dtype, has_bias)
            for x in inputs:
                args = (x, module.weight, module.bias)
                snapshots = [None if t is None else t.detach().clone() for t in args]
                actual = linear(*args)
                _equal(actual, F.linear(*args))
                _equal(actual, module(x))
                assert actual.dtype == dtype
                for tensor, before in zip(args, snapshots, strict=True):
                    if tensor is not None:
                        _equal(tensor, before)

            x = inputs[2].detach().clone().requires_grad_()
            weight = module.weight.detach().clone().requires_grad_()
            operands = [x, weight]
            if has_bias:
                operands.append(module.bias.detach().clone().requires_grad_())
            reference_operands = [t.detach().clone().requires_grad_() for t in operands]
            actual, expected = linear(*operands), F.linear(*reference_operands)
            upstream = torch.randn_like(actual)
            gradients = torch.autograd.grad(actual, operands, upstream)
            reference_gradients = torch.autograd.grad(expected, reference_operands, upstream)
            for gradient, reference in zip(gradients, reference_gradients, strict=True):
                _equal(gradient, reference)

        for x, weight, bias in (
            (torch.randn(2, 8, dtype=dtype), torch.randn(5, 7, dtype=dtype), None),
            (torch.randn(2, 7, dtype=dtype), torch.randn(5, 7, dtype=dtype), torch.randn(4, dtype=dtype)),
        ):
            with pytest.raises(RuntimeError) as original_error:
                F.linear(x, weight, bias)
            with pytest.raises(RuntimeError) as routed_error:
                linear(x, weight, bias)
            assert str(routed_error.value) == str(original_error.value)


def test_linear_registry_torch_only_and_profile_dispatch():
    assert "linear" in available_ops(), "linear registry entry is missing"
    from embodiinfer.backend.torch.ops import linear

    assert available_ops()["linear"] == ["torch"]
    assert get_op("linear") is linear
    for mode in ("none", "torch"):
        assert KernelSelection(mode).resolve("linear", "torch") is linear
    selection, calls = _profile(lambda x, weight, bias=None: x.shape[-1] == 7)
    op = selection.resolve("linear", "torch")
    assert selection.resolve("linear", "torch") is op
    assert selection.resolve("gated_gelu", "torch") is get_op("gated_gelu")
    for dtype in _DTYPES:
        for has_bias in (False, True):
            module = _dense(7, 5, dtype, has_bias)
            x = torch.randn(2, 3, 7, dtype=dtype)
            _equal(op(x, module.weight, module.bias), module(x))
            _call_is(calls[-1], x, module.weight, module.bias)
            count = len(calls)
            other = _dense(6, 5, dtype, has_bias)
            rejected = torch.randn(2, 3, 6, dtype=dtype)
            _equal(op(rejected, other.weight, other.bias), other(rejected))
            assert len(calls) == count
    assert len(calls) == 4


def test_pi05_linear_pair_profile_preserves_packed_gemm():
    selection, calls = _profile()
    for dtype in _DTYPES:
        first, second = _dense(7, 5, dtype, False), _dense(7, 9, dtype, False)
        x = torch.randn(2, 3, 7, dtype=dtype)
        with torch.no_grad():
            weight = torch.cat((first.weight, second.weight), dim=0).contiguous()
            expected = F.linear(x, weight).split((5, 9), dim=-1)
            actual = _pair(x, first, second, selection)
            assert len(calls) == 1
            packed = first._vvla_fused_pair_weight
            assert packed.shape == (14, 7) and packed.is_contiguous()
            _call_is(calls[0], x, packed, None)
            for result, reference in zip(actual, expected, strict=True):
                _equal(result, reference)
            for mode in ("none", "torch"):
                plain = _pair(x, first, second, KernelSelection(mode))
                for result, reference in zip(plain, expected, strict=True):
                    _equal(result, reference)
            repeated = _pair(x, first, second, selection)
            assert len(calls) == 2
            _call_is(calls[1], x, packed, None)
            for result, reference in zip(repeated, expected, strict=True):
                _equal(result, reference)
            calls.clear()


def test_pi05_mlp_linear_profile_reaches_pair_and_down(monkeypatch):
    selection, calls = _profile()
    other_selection, other_calls = _profile()
    monkeypatch.setattr(pi05, "_rmsnorm", lambda norm, x, *args: (x, None))
    for dtype in _DTYPES:
        for bias in (False, True):
            mlp = _mlp(dtype, bias)
            x = torch.randn(2, 3, 7, dtype=dtype)
            with torch.no_grad():
                expected = _original_mlp(mlp, x)
                _equal(pi05._mlp(mlp, x, False, kernels=selection), expected)
                assert len(calls) == 2
                assert [call.args[1].shape for call in calls] == [(22, 7), (7, 11)]
                _call_is(calls[0], x, mlp.gate_proj._vvla_fused_pair_weight, None)
                gate, up = calls[0].output.split((11, 11), dim=-1)
                _equal(calls[1].args[0], F.gelu(gate, approximate="tanh") * up)
                _call_is(calls[1], calls[1].args[0], mlp.down_proj.weight, mlp.down_proj.bias)
                for mode in ("none", "torch"):
                    _equal(pi05._mlp(mlp, x, False, kernels=KernelSelection(mode)), expected)
                _equal(pi05._mlp(mlp, x, False, kernels=other_selection), expected)
                assert len(calls) == len(other_calls) == 2
                layer = SimpleNamespace(
                    self_attn=SimpleNamespace(q_proj=mlp.gate_proj),
                    input_layernorm=object(),
                    post_attention_layernorm=object(),
                    mlp=mlp,
                )
                tower = SimpleNamespace(layers=[layer], norm=object(), rotary_emb=lambda *args: (None, None))
                policy = SimpleNamespace(
                    native_embeddings=False,
                    attention="sdpa",
                    compile_backend="none",
                    _kernels=selection,
                    _attn_sublayer=lambda attn, hidden, *args: torch.zeros_like(hidden),
                )
                actual, collected = pi05.Pi05Policy._tower_forward(
                    policy, tower, x, torch.zeros(2, 3, dtype=torch.long), None, None
                )
                _equal(actual, (x + torch.zeros_like(x)) + expected)
                assert collected is None
                assert len(calls) == 4 and len(other_calls) == 2
            calls.clear()
            other_calls.clear()


def test_pi05_vision_linear_profile_preserves_original_forward():
    selection, calls = _profile()
    for dtype, image_dtype in (
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ):
        policy, layer, slots = _vision(dtype, selection)
        image = torch.randn(2, 3, 4, 4, dtype=image_dtype)
        with torch.no_grad():
            expected = _original_image(policy, image)
            actual = pi05.Pi05Policy._embed_image(policy, image)
            _equal(actual, expected)
            assert actual.dtype == image_dtype
            assert len(calls) == 6
            assert [call.args[1].shape for call in calls] == [(8, 8)] * 4 + [(12, 8), (8, 12)]
            for call, (owner, name) in zip(calls, slots, strict=True):
                module = getattr(owner, name)
                _call_is(call, call.args[0], module.weight, module.bias)
                assert all(t.dtype == dtype for t in call.args)
                assert call.args[0].shape[:2] == (2, 4)
            assert calls[0].args[0] is calls[1].args[0] is calls[2].args[0]
            assert calls[3].args[0].is_contiguous()
            activation = layer.mlp.activation_fn
            assert len(activation.calls) == 2
            assert activation.calls[1][0] is calls[4].output
            assert calls[5].args[0] is activation.calls[1][1]
            for original, routed in zip(*policy._vision_attn.calls, strict=True):
                if isinstance(original, torch.Tensor):
                    _equal(routed, original)
                else:
                    assert routed == original
            for mode in ("none", "torch"):
                policy._kernels = KernelSelection(mode)
                _equal(pi05.Pi05Policy._embed_image(policy, image), expected)
            assert len(calls) == 6
        calls.clear()


def test_pi05_linear_pair_gradient_and_training_bypass_profile():
    selection, calls = _profile()
    for dtype in _DTYPES:
        _positive_control(selection, calls, dtype)
        for grad_enabled, first_training, second_training in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            first = _track(_dense(7, 5, dtype, False).train(first_training))
            second = _track(_dense(7, 9, dtype, False).train(second_training))
            x = torch.randn(2, 3, 7, dtype=dtype, requires_grad=grad_enabled)
            with torch.set_grad_enabled(grad_enabled):
                expected = first(x), second(x)
                first.route_inputs.clear()
                second.route_inputs.clear()
                actual = _pair(x, first, second, selection)
                assert len(first.route_inputs) == len(second.route_inputs) == 1
                for result, reference in zip(actual, expected, strict=True):
                    _equal(result, reference)
                if grad_enabled:
                    upstream = tuple(torch.randn_like(t) for t in actual)
                    operands = (x, first.weight, second.weight)
                    gradients = torch.autograd.grad(actual, operands, upstream)
                    reference_gradients = torch.autograd.grad(expected, operands, upstream)
                    for gradient, reference in zip(gradients, reference_gradients, strict=True):
                        _equal(gradient, reference)
            assert not hasattr(first, "_vvla_fused_pair_weight")
            assert calls == []


def test_pi05_linear_quantized_branches_bypass_profile(monkeypatch):
    selection, calls = _profile()

    class QuantizedProjection(QuantizedLinear):
        def __init__(self, in_features, out_features, dtype, bias):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.weight = nn.Parameter(torch.ones(out_features, in_features, dtype=torch.int8), False)
            self.compute_dtype = dtype
            self.bias = nn.Parameter(torch.randn(out_features, dtype=dtype)) if bias else None
            self.route_inputs = []
            self.eval()

        @classmethod
        def from_linear(cls, linear, config):
            raise AssertionError("test double is constructed directly")

        @classmethod
        def pack(cls, linears):
            raise AssertionError("test must use the recording packed projection")

        def forward(self, x):
            self.route_inputs.append(x)
            result = x[..., :1].expand(*x.shape[:-1], self.out_features) + 0.25
            return result if self.bias is None else result + self.bias

    for dtype in _DTYPES:
        _positive_control(selection, calls, dtype)
        with torch.no_grad():
            _check_module_bypass(QuantizedProjection, selection, calls, dtype)
            first, second = QuantizedProjection(7, 5, dtype, False), QuantizedProjection(7, 9, dtype, False)
            x = torch.randn(2, 3, 7, dtype=dtype)
            expected = first(x), second(x)
            packing, packed_calls = [], []

            def packed(value, first=first, second=second, packed_calls=packed_calls):
                packed_calls.append(value)
                return torch.cat((first(value), second(value)), dim=-1)

            def pack(projections, owner, name, packing=packing, packed=packed):
                packing.append((projections, owner, name))
                return packed

            monkeypatch.setattr(pi05, "_packed_quantized_projection", pack)
            actual = _pair(x, first, second, selection)
            assert packing == [((first, second), first, "_vvla_fused_pair_quantized")]
            assert len(packed_calls) == 1 and packed_calls[0] is x
            for result, reference in zip(actual, expected, strict=True):
                _equal(result, reference)
            assert calls == []


def test_pi05_linear_subclasses_and_parallel_modules_bypass_profile(monkeypatch):
    selection, calls = _profile()

    class AlteredLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x) + 0.5

    class CallableProjection:
        def __init__(self, in_features, out_features, dtype, bias):
            self.weight = torch.randn(out_features, in_features, dtype=dtype)
            self.bias = torch.randn(out_features, dtype=dtype) if bias else None
            self.training = False
            self.route_inputs = []

        def __call__(self, x):
            self.route_inputs.append(x)
            return F.linear(x, self.weight, self.bias) - 0.5

    reductions = []
    group = object()

    def reduce(output, *, group):
        reductions.append((output.clone(), group))
        output.add_(0.25)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    def column(k, n, dtype, bias):
        dense = _dense(k, n * 2, dtype, bias)
        return _track(ColumnParallelLinear.from_linear(dense, TensorParallelContext(2, 1), name="test"))

    def row(k, n, dtype, bias):
        dense = _dense(k * 2, n, dtype, bias)
        context = TensorParallelContext(2, 1, group)
        return _track(RowParallelLinear.from_linear(dense, context, name="test"))

    for dtype in _DTYPES:
        _positive_control(selection, calls, dtype)
        factories = (
            lambda k, n, dtype, bias: _track(AlteredLinear(k, n, bias=bias, dtype=dtype).eval()),
            column,
            row,
            CallableProjection,
        )
        with torch.no_grad():
            for make in factories:
                _check_module_bypass(make, selection, calls, dtype)
            projection = row(7, 5, dtype, True)
            x = torch.randn(2, 3, 7, dtype=dtype)
            mlp = SimpleNamespace(gate_proj=nn.Identity(), up_proj=nn.Identity(), down_proj=projection)
            activated = F.gelu(x, approximate="tanh") * x
            expected_partial = F.linear(activated, projection.weight, None)
            count = len(reductions)
            actual = pi05._mlp(mlp, x, False, fuse_projections=False, kernels=selection)
            assert len(reductions) == count + 1 and reductions[-1][1] is group
            _equal(reductions[-1][0], expected_partial)
            _equal(actual, (expected_partial + 0.25) + projection.bias)
        with torch.enable_grad(), pytest.raises(RuntimeError, match="inference only"):
            projection.train()
            pi05._mlp(mlp, x, False, fuse_projections=False, kernels=selection)
        assert calls == []
    assert reductions and all(recorded_group is group for _, recorded_group in reductions)


def test_pi05_linear_pair_routes_any_dense_projection_like_base():
    """BASE packs every non-quantized inference pair into one GEMM, whatever its module type."""
    selection, calls = _profile()

    class AlteredLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x) + 0.5

    def column(k, n, dtype):
        dense = _dense(k, n * 2, dtype, False)
        return ColumnParallelLinear.from_linear(dense, TensorParallelContext(2, 1), name="test").eval()

    for dtype in _DTYPES:
        factories = (lambda k, n, dtype: AlteredLinear(k, n, bias=False, dtype=dtype).eval(), column)
        for make in factories:
            for swapped in (False, True):
                first, second = make(7, 5, dtype), _dense(7, 9, dtype, False)
                if swapped:
                    first, second = second, first
                x = torch.randn(2, 3, 7, dtype=dtype)
                weight = torch.cat((first.weight, second.weight), dim=0).contiguous()
                expected = F.linear(x, weight).split((first.out_features, second.out_features), dim=-1)
                with torch.no_grad():
                    actual = _pair(x, first, second, selection)
                assert len(calls) == 1
                _call_is(calls[0], x, first._vvla_fused_pair_weight, None)
                for result, reference in zip(actual, expected, strict=True):
                    _equal(result, reference)
                calls.clear()


def test_pi05_eager_projection_branches_bypass_linear_profile(monkeypatch):
    selection, calls = _profile()

    def unexpected(*args, **kwargs):
        raise AssertionError("eager branch used a fused or native embedding helper")

    monkeypatch.setattr(pi05, "_fused_qkv", unexpected)
    monkeypatch.setattr(pi05, "_embed_prefix_batched", unexpected)
    for dtype in _DTYPES:
        _positive_control(selection, calls, dtype)
        policy = SimpleNamespace(
            attention="eager", compile_backend="none", _kernels=selection, _attn=_Attention()
        )
        attn = SimpleNamespace(
            q_proj=_track(_dense(8, 8, dtype)),
            k_proj=_track(_dense(8, 8, dtype)),
            v_proj=_track(_dense(8, 8, dtype)),
            o_proj=_track(_dense(8, 8, dtype)),
            head_dim=4,
            scaling=0.5,
        )
        h = torch.randn(2, 3, 8, dtype=dtype)
        cos, sin = torch.ones(2, 3, 4, dtype=dtype), torch.zeros(2, 3, 4, dtype=dtype)
        with torch.no_grad():
            q, k, v = [
                proj(h).view(2, 3, 2, 4).transpose(1, 2) for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
            ]
            out = F.scaled_dot_product_attention(q, k, v, scale=0.5)
            expected = attn.o_proj(out.transpose(1, 2).reshape(2, 3, -1))
            for projection in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                projection.route_inputs.clear()
            actual = pi05.Pi05Policy._attn_sublayer(policy, attn, h, cos, sin, None, None, [])
            _equal(actual, expected)
            assert all(
                len(proj.route_inputs) == 1 for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj)
            )
            assert calls == []

            mlp = _mlp(dtype)
            _track(mlp.gate_proj)
            _track(mlp.up_proj)
            x = torch.randn(2, 3, 7, dtype=dtype)
            expected = _original_mlp(mlp, x, fuse=False)
            mlp.gate_proj.route_inputs.clear()
            mlp.up_proj.route_inputs.clear()
            with monkeypatch.context() as patch:
                patch.setattr(pi05, "_fused_linear_pair", unexpected)
                _equal(pi05._mlp(mlp, x, False, fuse_projections=False, kernels=selection), expected)
            assert len(mlp.gate_proj.route_inputs) == len(mlp.up_proj.route_inputs) == 1
            assert len(calls) == 1
            _call_is(calls[0], calls[0].args[0], mlp.down_proj.weight, mlp.down_proj.bias)
            calls.clear()

            prefix_calls = []
            hidden = torch.randn(2, 3, 8, dtype=dtype)
            pads = torch.ones(2, 3, dtype=torch.bool)

            def embed_prefix(*args, prefix_calls=prefix_calls, hidden=hidden, pads=pads):
                prefix_calls.append(args)
                return hidden, pads, torch.zeros_like(pads)

            policy.native_embeddings = False
            policy.prefix_attention = "sdpa"
            policy._m = SimpleNamespace(embed_prefix=embed_prefix)
            policy._embed_prefix = unexpected
            policy._prefix_tower = object()
            policy._make_att_2d_masks = lambda *args: torch.zeros(2, 3, 3, dtype=torch.bool)
            policy._attention_mask_4d = lambda *args: None
            policy._tower_forward = lambda *args, dtype=dtype, **kwargs: (
                args[1],
                [(torch.zeros(2, 1, 3, 8, dtype=dtype), torch.zeros(2, 1, 3, 8, dtype=dtype))],
            )
            batch = SimpleNamespace(
                images=[torch.zeros(2, 3, 4, 4, dtype=dtype)],
                img_masks=[torch.ones(2, dtype=torch.bool)],
                tokens=torch.zeros(2, 3, dtype=torch.long),
                masks=pads,
            )
            prefix = pi05.Pi05Policy._encode_prefix_impl(policy, batch)
            assert prefix.batch_size == 2 and len(prefix_calls) == 1
            assert prefix_calls[0][0] is batch.images
            assert calls == []
