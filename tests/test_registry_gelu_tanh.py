"""CPU bit-exact parity and guarded SigLIP dispatch for tanh GELU."""

import functools
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


@pytest.fixture
def transformers_tanh(monkeypatch):
    # The recorded CPU environment lacks Transformers. Exercise canonical class
    # identity with a scoped double of its gelu_pytorch_tanh implementation.
    module = ModuleType("transformers.activations")

    class PytorchGELUTanh(nn.Module):
        # Mirrors transformers' GELUTanh, including its Python-formula variant.
        def __init__(self, use_gelu_tanh_python=False):
            super().__init__()
            if use_gelu_tanh_python:
                self.act = self._python
            else:
                self.act = functools.partial(F.gelu, approximate="tanh")

        @staticmethod
        def _python(x):
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

        def forward(self, x):
            return self.act(x)

    PytorchGELUTanh.__module__ = module.__name__
    module.PytorchGELUTanh = PytorchGELUTanh
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return PytorchGELUTanh


def _equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(actual, expected)


def _profile(applies=None):
    calls, linears = [], []

    def gelu(x):
        output = F.gelu(x, approximate="tanh")
        calls.append(SimpleNamespace(x=x, metadata=(x.shape, x.stride(), x.dtype), output=output))
        return output

    def linear(x, weight, bias=None):
        output = F.linear(x, weight, bias)
        linears.append(SimpleNamespace(x=x, weight=weight, bias=bias, output=output))
        return output

    profile = KernelProfile(
        name="gelu_tanh_cpu_test",
        root=Path(__file__).parent,
        match={},
        ops={"gelu_tanh": {}, "linear": {}},
        _loaded={"gelu_tanh": TunedKernel(gelu, applies), "linear": TunedKernel(linear, None)},
    )
    return KernelSelection(profile), calls, linears


def _dense(k, n, dtype):
    return nn.Linear(k, n, dtype=dtype).eval()


class _Attention:
    def __init__(self):
        self.calls = []

    def attend(self, q, k, v, *, scaling):
        self.calls.append((q, k, v, scaling))
        return F.scaled_dot_product_attention(q, k, v, scale=scaling)


def _vision(dtype, selection, activation=None):
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
        mlp=SimpleNamespace(
            fc1=_dense(8, 12, dtype),
            fc2=_dense(12, 8, dtype),
            activation_fn=nn.GELU(approximate="tanh") if activation is None else activation,
        ),
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
    policy._embed_image = pi05.Pi05Policy._embed_image.__get__(policy, type(policy))
    return policy, layer


def _original_image(policy, image):
    # Frozen baseline arithmetic, with the existing Torch linear reference
    # expanded to its single F.linear call; activation still calls its module.
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
        qkv = []
        for proj in (a.q_proj, a.k_proj, a.v_proj):
            if type(proj) is nn.Linear:  # noqa: SIM108 — retain the baseline branch structure.
                projected = F.linear(x, proj.weight, proj.bias)
            else:
                projected = proj(x)
            qkv.append(projected.view(b, s, a.num_heads, a.head_dim).transpose(1, 2))
        q, k, v = qkv
        y = policy._vision_attn.attend(q, k, v, scaling=a.scale)
        if type(a.out_proj) is nn.Linear:
            h = h + F.linear(
                y.transpose(1, 2).reshape(b, s, -1).contiguous(), a.out_proj.weight, a.out_proj.bias
            )
        else:
            h = h + a.out_proj(y.transpose(1, 2).reshape(b, s, -1).contiguous())
        x = layer.layer_norm2(h)
        if type(layer.mlp.fc1) is nn.Linear:
            activated = F.linear(x, layer.mlp.fc1.weight, layer.mlp.fc1.bias)
        else:
            activated = layer.mlp.fc1(x)
        activated = layer.mlp.activation_fn(activated)
        if type(layer.mlp.fc2) is nn.Linear:
            h = h + F.linear(activated, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
        else:
            h = h + layer.mlp.fc2(activated)
    h = vision.post_layernorm(h)
    h = pg.multi_modal_projector.linear(h)
    scale = pg.config.text_config.hidden_size**0.5
    return ((h / scale) * scale).to(image.dtype)


def _positive_control(selection, calls, linears, dtype):
    policy, _ = _vision(dtype, selection)
    image = torch.randn(2, 3, 4, 4, dtype=dtype)
    with torch.no_grad():
        _equal(policy._embed_image(image), _original_image(policy, image))
    assert len(calls) == 1, "SigLIP tanh GELU must reach the supplied profile"
    calls.clear()
    linears.clear()


def test_gelu_tanh_reference_fp32_bf16_matches_original_expression():
    from embodiinfer.backend.torch import ops

    assert callable(getattr(ops, "gelu_tanh", None)), "Torch gelu_tanh reference is missing"
    from embodiinfer.backend.torch.ops import gelu_tanh

    for dtype in _DTYPES:
        inputs = [torch.randn(*shape, dtype=dtype) for shape in ((17,), (2, 3, 12))]
        inputs.append(torch.randn(2, 12, 3, dtype=dtype).transpose(1, 2))
        assert not inputs[-1].is_contiguous()
        for x in inputs:
            before = x.clone()
            actual = gelu_tanh(x)
            _equal(actual, F.gelu(x, approximate="tanh"))
            _equal(actual, nn.GELU(approximate="tanh")(x))
            _equal(x, before)

            leaf = x.detach().clone().requires_grad_()
            reference_leaf = x.detach().clone().requires_grad_()
            upstream = torch.randn_like(x)
            actual_grad = torch.autograd.grad(gelu_tanh(leaf), leaf, upstream)[0]
            expected_grad = torch.autograd.grad(
                F.gelu(reference_leaf, approximate="tanh"), reference_leaf, upstream
            )[0]
            _equal(actual_grad, expected_grad)


def test_gelu_tanh_registry_torch_only_and_profile_dispatch():
    assert "gelu_tanh" in available_ops(), "gelu_tanh registry entry is missing"
    from embodiinfer.backend.torch.ops import gelu_tanh

    assert available_ops()["gelu_tanh"] == ["torch"]
    assert get_op("gelu_tanh") is gelu_tanh
    for mode in ("none", "torch"):
        selection = KernelSelection(mode)
        selection.prepare()
        assert selection.resolve("gelu_tanh", "torch") is gelu_tanh
    assert KernelSelection("torch").resolve("gelu_tanh", "triton") is gelu_tanh
    selection, calls, _ = _profile(lambda x: x.shape[-1] == 12)
    op = selection.resolve("gelu_tanh", "torch")
    assert selection.resolve("gelu_tanh", "torch") is op
    assert selection.resolve("gated_gelu") is get_op("gated_gelu")
    for dtype in _DTYPES:
        x = torch.randn(2, 12, 3, dtype=dtype).transpose(1, 2)
        _equal(op(x), F.gelu(x, approximate="tanh"))
        assert calls[-1].x is x
        assert calls[-1].metadata == (x.shape, x.stride(), dtype)
        count = len(calls)
        rejected = torch.randn(2, 3, 7, dtype=dtype)
        _equal(op(rejected), F.gelu(rejected, approximate="tanh"))
        assert len(calls) == count
    assert len(calls) == 2


def test_pi05_gelu_tanh_profile_preserves_original_vision_forward(transformers_tanh):
    for factory in (lambda: nn.GELU(approximate="tanh"), transformers_tanh):
        for dtype, image_dtype in (
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ):
            selection, calls, linears = _profile()
            policy, layer = _vision(dtype, selection, factory())
            image = torch.randn(2, 3, 4, 4, dtype=image_dtype)
            with torch.no_grad():
                expected = _original_image(policy, image)
                actual = policy._embed_image(image)
                _equal(actual, expected)
                assert actual.shape == (2, 4, 7) and actual.dtype == image_dtype
                assert len(calls) == 1, "SigLIP tanh GELU must reach the supplied profile"
                assert len(linears) == 6
                assert calls[0].x is linears[4].output
                assert calls[0].metadata == (torch.Size((2, 4, 12)), (48, 12, 1), dtype)
                assert linears[5].x is calls[0].output
                assert linears[4].weight is layer.mlp.fc1.weight
                assert linears[5].weight is layer.mlp.fc2.weight
                assert linears[4].bias is layer.mlp.fc1.bias
                assert linears[5].bias is layer.mlp.fc2.bias
                assert linears[3].x.is_contiguous()
                for original, routed in zip(*policy._vision_attn.calls, strict=True):
                    if isinstance(original, torch.Tensor):
                        _equal(routed, original)
                    else:
                        assert routed == original
                for mode in ("none", "torch"):
                    policy._kernels = KernelSelection(mode)
                    _equal(policy._embed_image(image), expected)
                assert len(calls) == 1 and len(linears) == 6

            # Preserve differentiability through the complete vision path too.
            policy._kernels = selection
            leaf = image.detach().clone().requires_grad_()
            reference_leaf = image.detach().clone().requires_grad_()
            actual = policy._embed_image(leaf)
            expected = _original_image(policy, reference_leaf)
            _equal(actual, expected)
            upstream = torch.randn_like(actual)
            _equal(
                torch.autograd.grad(actual, leaf, upstream)[0],
                torch.autograd.grad(expected, reference_leaf, upstream)[0],
            )
            assert len(calls) == 2


def test_pi05_gelu_tanh_other_activations_bypass_profile(transformers_tanh):
    class AlteredGELU(nn.GELU):
        def forward(self, x):
            return super().forward(x) + 0.5

    class AlteredTransformersGELU(transformers_tanh):
        def forward(self, x):
            return super().forward(x) - 0.25

    class RecordingCallable:
        def __init__(self):
            self.calls = []

        def __call__(self, x):
            self.calls.append(x)
            return F.gelu(x, approximate="tanh") + 0.125

    for dtype in _DTYPES:
        selection, calls, linears = _profile()
        _positive_control(selection, calls, linears, dtype)
        policy, layer = _vision(dtype, selection)
        image = torch.randn(2, 3, 4, 4, dtype=dtype)
        recorded = RecordingCallable()
        overridden = nn.GELU(approximate="tanh")
        overridden.forward = recorded
        hooked = nn.GELU(approximate="tanh")
        hook_calls = []

        def hook(module, args, output, hook_calls=hook_calls):
            hook_calls.append(args[0])
            return output + 0.25

        hooked.register_forward_hook(hook)
        impostor = type(
            "PytorchGELUTanh",
            (nn.Module,),
            {"__module__": "transformers.activations", "forward": lambda self, x: F.silu(x)},
        )()
        custom = RecordingCallable()
        for activation in (
            nn.GELU(approximate="none"),
            nn.ReLU(),
            custom,
            AlteredGELU(approximate="tanh"),
            AlteredTransformersGELU(),
            transformers_tanh(use_gelu_tanh_python=True),
            overridden,
            hooked,
            impostor,
        ):
            layer.mlp.activation_fn = activation
            with torch.no_grad():
                expected = _original_image(policy, image)
                _equal(policy._embed_image(image), expected)
            assert calls == [], "a different activation must keep its own callable"
        assert len(custom.calls) == len(recorded.calls) == len(hook_calls) == 2


def test_pi05_gelu_tanh_linear_module_fallbacks_are_preserved(monkeypatch):
    class AlteredLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x) + 0.5

    class QuantizedProjection(QuantizedLinear):
        def __init__(self, k, n, dtype):
            super().__init__()
            self.in_features, self.out_features = k, n
            self.weight = nn.Parameter(torch.ones(n, k, dtype=torch.int8), False)
            self.bias = nn.Parameter(torch.randn(n, dtype=dtype))
            self.compute_dtype = dtype

        @classmethod
        def from_linear(cls, linear, config):
            raise AssertionError("test double is constructed directly")

        @classmethod
        def pack(cls, linears):
            raise AssertionError("vision projections must not be packed")

        def forward(self, x):
            return x[..., :1].expand(*x.shape[:-1], self.out_features) + 0.25 + self.bias

    reductions = []
    group = object()

    def reduce(output, *, group):
        reductions.append((output.clone(), group))
        output.add_(0.25)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    def column(k, n, dtype):
        return ColumnParallelLinear.from_linear(
            _dense(k, n * 2, dtype), TensorParallelContext(2, 1), name="gelu_test"
        )

    def row(k, n, dtype):
        return RowParallelLinear.from_linear(
            _dense(k * 2, n, dtype), TensorParallelContext(2, 1, group), name="gelu_test"
        )

    for dtype in _DTYPES:
        selection, calls, linears = _profile()
        _positive_control(selection, calls, linears, dtype)
        for factory in (
            lambda k, n, dtype: AlteredLinear(k, n, dtype=dtype),
            QuantizedProjection,
            column,
            row,
        ):
            policy, layer = _vision(dtype, selection)
            layer.mlp.fc1 = factory(8, 12, dtype).eval()
            layer.mlp.fc2 = factory(12, 8, dtype).eval()
            module_calls = []

            def track(module, args, output, module_calls=module_calls):
                module_calls.append((module, args[0], output))

            layer.mlp.fc1.register_forward_hook(track)
            layer.mlp.fc2.register_forward_hook(track)
            image = torch.randn(2, 3, 4, 4, dtype=dtype)
            with torch.no_grad():
                reductions.clear()
                expected = _original_image(policy, image)
                reference_reductions = list(reductions)
                reductions.clear()
                module_calls.clear()
                actual = policy._embed_image(image)
            _equal(actual, expected)
            assert len(calls) == 1
            assert len(linears) == 4  # Q/K/V and attention output only.
            assert len(module_calls) == 2
            assert module_calls[0][0] is layer.mlp.fc1
            assert module_calls[1][0] is layer.mlp.fc2
            assert calls[0].x is module_calls[0][2]
            assert module_calls[1][1] is calls[0].output
            assert all(
                call.weight is not projection.weight
                for call in linears
                for projection in (layer.mlp.fc1, layer.mlp.fc2)
            )
            assert len(reductions) == len(reference_reductions) == (2 if factory is row else 0)
            for (actual_partial, actual_group), (expected_partial, expected_group) in zip(
                reductions, reference_reductions, strict=True
            ):
                _equal(actual_partial, expected_partial)
                assert actual_group is expected_group is group
            calls.clear()
            linears.clear()


def test_pi05_gelu_tanh_prefix_keeps_policy_local_selection():
    for dtype in _DTYPES:
        selection, calls, _ = _profile()
        other_selection, other_calls, _ = _profile()
        policy, _ = _vision(dtype, selection)
        other = SimpleNamespace(**policy.__dict__)
        other._kernels = other_selection
        other._embed_image = pi05.Pi05Policy._embed_image.__get__(other, type(other))
        policy._prefix_tower = other._prefix_tower = SimpleNamespace(
            embed_tokens=nn.Embedding(9, 7, dtype=dtype)
        )
        batch = SimpleNamespace(
            images=[torch.randn(2, 3, 4, 4, dtype=dtype) for _ in range(2)],
            img_masks=[torch.tensor([True, False]), torch.tensor([False, True])],
            tokens=torch.tensor([[1, 2, 3], [3, 2, 1]]),
            masks=torch.tensor([[True, True, False], [True, True, True]]),
        )
        with torch.no_grad():
            images = [_original_image(policy, image) for image in batch.images]
            lang = policy._prefix_tower.embed_tokens(batch.tokens)
            expected = torch.cat([*images, lang * lang.shape[-1] ** 0.5], dim=1)
            pads = torch.cat([*(mask[:, None].expand(2, 4) for mask in batch.img_masks), batch.masks], dim=1)
            actual = pi05.Pi05Policy._embed_prefix(policy, batch)
            for value, reference in zip(actual, (expected, pads, torch.zeros_like(pads)), strict=True):
                _equal(value, reference)
            assert len(calls) == 2, "each camera must route GELU through its policy's selection"
            assert other_calls == []
            other_actual = pi05.Pi05Policy._embed_prefix(other, batch)
            for value, reference in zip(other_actual, actual, strict=True):
                _equal(value, reference)
            assert len(calls) == len(other_calls) == 2
            assert calls[0].x is not other_calls[0].x


def test_pi05_gelu_tanh_guard_and_fallback_use_one_activation_lookup():
    class ChangingActivationMLP:
        def __init__(self, mlp, activation):
            self.fc1, self.fc2 = mlp.fc1, mlp.fc2
            self.activation = activation
            self.lookups = 0

        @property
        def activation_fn(self):
            self.lookups += 1
            return self.activation if self.lookups == 1 else nn.ReLU()

    for dtype in _DTYPES:
        for approximate in ("tanh", "none"):
            selection, calls, _ = _profile()
            policy, layer = _vision(dtype, selection)
            layer.mlp = ChangingActivationMLP(layer.mlp, nn.GELU(approximate=approximate))
            image = torch.randn(2, 3, 4, 4, dtype=dtype)
            with torch.no_grad():
                expected = _original_image(policy, image)
                assert layer.mlp.lookups == 1
                layer.mlp.lookups = 0
                actual = policy._embed_image(image)
            _equal(actual, expected)
            assert layer.mlp.lookups == 1, "the guard and fallback must use the same activation lookup"
            assert len(calls) == (1 if approximate == "tanh" else 0)


def test_pi05_gelu_tanh_real_transformers_activations():
    """The installed Transformers' SigLIP activation routes; its Python-formula variant does not."""
    activations = pytest.importorskip("transformers.activations")
    native, python = activations.ACT2FN["gelu_pytorch_tanh"], activations.ACT2FN["gelu_python_tanh"]
    # A dense deterministic range: the Python formula rounds differently on part of it.
    x = torch.linspace(-8, 8, 100001)
    assert torch.equal(native(x), F.gelu(x, approximate="tanh"))
    assert not torch.equal(python(x), F.gelu(x, approximate="tanh"))
    assert pi05._is_gelu_tanh(native)
    assert not pi05._is_gelu_tanh(python)
