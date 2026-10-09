"""Migration contracts for instance-local Pi05 fused operators."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.layers import OperatorBackends, OperatorRequest, normalization_backends, paired_gelu_backends
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy, Pi05Prefix, _rmsnorm_math
from embodiinfer.policies.pi05.optimization import Pi05Optimizations
from embodiinfer.policies.pi05.optimization_config import ActionLayerPrecision, Pi05OptimizationConfig
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch
from embodiinfer.policies.pi05.runtime import _compact_all_valid


def test_optimization_recipe_preserves_activation_contract(tmp_path):
    config = Pi05OptimizationConfig(
        hardware="thor",
        fused_mlp=True,
        action_layers=(ActionLayerPrecision(gate_up="fp8", gate_up_max=3.0),),
        checkpoint_sha256="1" * 64,
    )
    path = tmp_path / "recipe.json"
    config.to_json(path)
    assert Pi05OptimizationConfig.from_json(path) == config
    path.write_text('{"hardware": "thor", "action_layers": []}')
    with pytest.raises(ValueError, match="tanh GELU calibration contract"):
        Pi05OptimizationConfig.from_json(path)


@pytest.mark.parametrize(
    "values",
    [
        {"hardware": "4090"},
        {"fused_mlp": True},
        {"attention": "flash"},
        {"activation": "exact"},
        {"norm_fusion": "yes"},
        {"action_layers": (ActionLayerPrecision(down="nvfp4"),)},
        {"checkpoint_sha256": "x" * 64},
        {"checkpoint_sha256": 42},
        {"schema_version": True},
    ],
)
def test_optimization_config_rejects_invalid_contract(values):
    with pytest.raises(ValueError):
        Pi05OptimizationConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"native_inference": False},
        {"native_inference": True, "compile_backend": "inductor"},
        {"native_inference": True, "quantization": "fp8"},
        {"native_inference": True, "tensor_parallel_size": 2},
        {"native_inference": True, "denoise_attention": "triton"},
    ],
)
def test_unsupported_combinations_fail_before_checkpoint_loading(values):
    with pytest.raises(ValueError, match="Fused Pi05|Select migrated attention"):
        Pi05Policy(VLAPolicyConfig(), None, optimizations=Pi05OptimizationConfig(), **values)


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
    import embodiinfer.policies.pi05.optimization as optimization

    monkeypatch.setattr(optimization, "_stream_key", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runtime = Pi05Optimizations.__new__(Pi05Optimizations)
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
@pytest.mark.parametrize("adaptive", [False, True])
def test_strict_norm_residual_matches_torch_rounding(adaptive):
    from embodiinfer.backend.cuda.normalization import NormFusion

    torch.manual_seed(31)
    x, update = (torch.randn(1, 10, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    scale = torch.randn(1024, device="cuda")
    modulation = torch.randn(1, 3072, device="cuda") if adaptive else None
    gate = torch.randn(1, 1, 1024, device="cuda", dtype=torch.bfloat16) if adaptive else None
    plan = NormFusion().plan(x, adaptive=adaptive, mode="pointwise")
    actual, actual_gate = plan.normalize(x, scale=None if adaptive else scale, modulation=modulation)
    expected, expected_gate = _rmsnorm_math(x, scale, 1e-6, modulation)
    assert torch.equal(actual, expected)
    if adaptive:
        assert torch.equal(actual_gate, expected_gate)
    residual, normalized, next_gate = plan.residual_normalize(
        x,
        update,
        gate,
        scale=None if adaptive else scale,
        modulation=modulation,
    )
    expected_residual = x + (update if gate is None else update * gate)
    expected, expected_gate = _rmsnorm_math(expected_residual, scale, 1e-6, modulation)
    assert torch.equal(residual, expected_residual)
    assert torch.equal(normalized, expected)
    if adaptive:
        assert torch.equal(next_gate, expected_gate)


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [8, 16])
def test_fused_gelu_uses_tanh_and_preserves_product_rounding(bits):
    from embodiinfer.backend.cuda.activation import GeluMulFusion
    from embodiinfer.backend.cuda.quantization import CudaQuantizer

    torch.manual_seed(32)
    gate, up = (torch.randn(10, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    backend = CudaQuantizer()
    fusion = GeluMulFusion(backend, approximate="tanh")
    plan = fusion.plan(gate, bits, 0.02 if bits == 8 else None)
    actual = plan.encode(gate, up).output
    hidden = F.gelu(gate, approximate="tanh") * up
    expected = backend.plan(hidden, 8, 0.02).quantize(hidden).output if bits == 8 else hidden
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    # A second invocation must read new inputs, not a pointer-keyed activation cache.
    gate.mul_(0.3)
    actual = plan.encode(gate, up).output
    hidden = F.gelu(gate, approximate="tanh") * up
    expected = backend.plan(hidden, 8, 0.02).quantize(hidden).output if bits == 8 else hidden
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["query_major", "folded_flash"])
def test_registered_attention_matches_mathematical_reference(name):
    from embodiinfer.layers import get_attention_backend

    torch.manual_seed(33)
    q = torch.randn(1, 8, 10, 64, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(1, 1, 91, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    original_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    result = get_attention_backend(name).attend(q, k, v, scaling=0.125)
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=0.125, enable_gqa=True)
    # BF16 probabilities/output and alternative reduction order account for this
    # component-only threshold; it is not an acceptance gate for full actions.
    torch.testing.assert_close(result.float(), expected, rtol=0, atol=0.01)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == original_flag
    if name == "query_major":
        with pytest.raises(ValueError, match="Boolean"):
            get_attention_backend(name).attend(q, k, v, attn_mask=torch.zeros(1, 1, 10, 91, device="cuda"))


def test_tanh_lookup_rejects_exact_arithmetic_launch():
    from embodiinfer.backend.cuda.gelu_lookup import GeluLookup

    lookup = GeluLookup.__new__(GeluLookup)
    lookup.fusion = SimpleNamespace(approximate="tanh")
    with pytest.raises(ValueError, match="exact GELU only"):
        lookup.select("arithmetic")


@pytest.mark.gpu
def test_nvfp4_requires_blackwell():
    from embodiinfer.backend.cuda.quantization import CudaQuantizer

    if torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("negative architecture check runs on pre-Blackwell devices")
    x = torch.zeros(10, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Blackwell"):
        CudaQuantizer().plan(x, 4)


class _TinyNorm(torch.nn.Module):
    def __init__(self, width, adaptive):
        super().__init__()
        self.eps = 1e-6
        self.dense = torch.nn.Linear(width, width * 3) if adaptive else None
        if not adaptive:
            self.weight = torch.nn.Parameter(torch.zeros(width))


class _TinyTower(torch.nn.Module):
    def __init__(self, adaptive):
        super().__init__()
        width, head_dim = 64, 64
        self.embed_tokens = torch.nn.Embedding(32, width, dtype=torch.bfloat16)
        self.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.input_layernorm = _TinyNorm(width, adaptive)
            layer.post_attention_layernorm = _TinyNorm(width, adaptive)
            layer.self_attn = torch.nn.Module()
            layer.self_attn.head_dim = head_dim
            layer.self_attn.scaling = head_dim**-0.5
            for name, output in (
                ("q_proj", 4 * head_dim),
                ("k_proj", head_dim),
                ("v_proj", head_dim),
            ):
                setattr(
                    layer.self_attn, name, torch.nn.Linear(width, output, bias=False, dtype=torch.bfloat16)
                )
            layer.self_attn.o_proj = torch.nn.Linear(4 * head_dim, width, bias=False, dtype=torch.bfloat16)
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = torch.nn.Linear(width, 256, bias=False, dtype=torch.bfloat16)
            layer.mlp.up_proj = torch.nn.Linear(width, 256, bias=False, dtype=torch.bfloat16)
            layer.mlp.down_proj = torch.nn.Linear(256, width, bias=False, dtype=torch.bfloat16)
            self.layers.append(layer)
        self.norm = _TinyNorm(width, adaptive)
        self.rotary_emb = torch.nn.Module()
        self.rotary_emb.register_buffer(
            "inv_freq", 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        )
        self.rotary_emb.attention_scaling = 1.0


def tiny_cuda_policy(config):
    from embodiinfer.layers import get_attention_backend
    from embodiinfer.policies.base import FlowVLAPolicy
    from embodiinfer.policies.pi05.embeddings import make_attention_mask, time_embedding
    from embodiinfer.policies.pi05.runtime import Pi05Runtime

    class TinyPolicy(Pi05Policy):
        def _embed_image(self, image):
            return image.mean((1, 2, 3))[:, None, None].expand(-1, 3, 64).contiguous().bfloat16()

    policy = TinyPolicy.__new__(TinyPolicy)
    FlowVLAPolicy.__init__(policy, VLAPolicyConfig(action_horizon=10, action_dim=8))
    policy._optimization_config = config
    policy._fused_ops = None
    policy._optimization_recipe_stale = False
    policy.native_inference = policy.native_embeddings = True
    policy.prefix_cuda_graph = True
    policy.compile_backend = "none"
    policy.attention = policy.prefix_attention = policy.denoise_attention = "sdpa"
    policy._attn = get_attention_backend("sdpa")
    policy.tensor_parallel = SimpleNamespace(enabled=False)
    policy._native_attention = None
    policy._cached_suffix_att = None
    policy._cached_prefix_meta = None
    policy.checkpoint = None
    policy._make_att_2d_masks, policy._sinusoidal = make_attention_mask, time_embedding
    policy._prefix_tower, policy._expert_tower = _TinyTower(False), _TinyTower(True)
    policy._m = torch.nn.Module()
    policy._m.action_in_proj = torch.nn.Linear(8, 64)
    policy._m.action_out_proj = torch.nn.Linear(64, 8)
    policy._m.time_mlp_in = torch.nn.Linear(64, 64)
    policy._m.time_mlp_out = torch.nn.Linear(64, 64)
    policy._m.config = SimpleNamespace(chunk_size=10, min_period=0.004, max_period=4.0)
    policy._runtime = Pi05Runtime(policy)
    return policy.eval().cuda()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "attention,paired_reference",
    [("reference", False), ("folded_flash", False), ("query_major", False), ("reference", True)],
)
def test_complete_native_graph_replays_changed_observations_and_owns_outputs(attention, paired_reference):
    torch.manual_seed(34)
    policy = tiny_cuda_policy(
        Pi05OptimizationConfig(
            attention=attention,
            fused_mlp=paired_reference,
            operators=OperatorBackends(paired_gelu="torch") if paired_reference else OperatorBackends(),
        )
    )
    parameters = {name: id(value) for name, value in policy.named_parameters()}
    state_keys = tuple(policy.state_dict())
    noise = torch.randn(1, 10, 8, device="cuda")
    previous = saved = None
    with torch.inference_mode():
        for tokens in ([1, 2, 3, 4, 5], [5, 3, 1, 7, 2]):
            batch = Pi05Batch(
                [torch.randn(1, 3, 8, 8, device="cuda")],
                [torch.tensor([True], device="cuda")],
                torch.tensor([tokens], device="cuda"),
                torch.tensor([[True, False, True, False, True]], device="cuda"),
            )
            prefix = policy.encode_prefix(batch)
            expected = policy._runtime.decode(noise, prefix, 10, False)
            actual = policy._runtime.decode(noise, prefix, 10, True)
            torch.cuda.synchronize()
            assert torch.equal(actual, expected)
            if previous is not None:
                assert not torch.equal(previous, actual)
                assert torch.equal(previous, saved)
                assert previous.data_ptr() != actual.data_ptr()
            previous, saved = actual, actual.clone()
        assert {name: id(value) for name, value in policy.named_parameters()} == parameters
        assert tuple(policy.state_dict()) == state_keys
        operators = policy._fused_ops
        assert operators.kv_buffers
        policy.on_refit(1)
        assert policy._fused_ops is None
        assert not operators.kv_buffers
        assert not policy._runtime.loop_graphs
        assert not policy._runtime.prefix_graphs
        assert not policy._runtime.schedules


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive", [False, True])
def test_norm_fp8_encoder_preserves_normalization_rounding(adaptive):
    from embodiinfer.backend.cuda.norm_quant import NormQuantFusion
    from embodiinfer.backend.cuda.normalization import NormFusion
    from embodiinfer.backend.cuda.quantization import CudaQuantizer

    x = torch.randn(1, 10, 1024, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1024, device="cuda")
    modulation = torch.randn(1, 3072, device="cuda") if adaptive else None
    norm = NormFusion()
    quantizer = CudaQuantizer()
    plan = NormQuantFusion(quantizer, norm).plan(x, bits=8, calibrated_scale=0.02, adaptive=adaptive)
    encoded = plan.normalize(x, scale=None if adaptive else scale, modulation=modulation)
    expected, gate = _rmsnorm_math(x, scale, 1e-6, modulation)
    reference = quantizer.plan(expected.contiguous().view(-1, 1024), 8, 0.02).quantize(
        expected.contiguous().view(-1, 1024)
    )
    assert torch.equal(encoded.output.view(torch.uint8), reference.output.view(torch.uint8))
    if adaptive:
        assert torch.equal(encoded.gate, gate)


@pytest.mark.gpu
def test_paired_gemm_uses_device_tanh_lookup():
    from embodiinfer.backend.triton.geglu import lookup_geglu, prefix_geglu
    from embodiinfer.layers.launch import GemmTile

    torch.manual_seed(35)
    x = torch.randn(19, 64, device="cuda", dtype=torch.bfloat16)
    gate, up = (torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2))
    codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
    table = F.gelu(codes.view(torch.bfloat16), approximate="tanh").view(torch.uint16)
    expected = F.gelu(x @ gate, approximate="tanh") * (x @ up)
    actual = lookup_geglu(x, gate, up, table, GemmTile(16, 32, 32, 4))
    # Different GEMM schedules can change rounded gate/up values. Bound this
    # operator's error independently; this does not approve full-model drift.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.02)
    tail = prefix_geglu(x, gate, up, table, GemmTile(32, 32, 32, 4), 16)
    torch.testing.assert_close(tail.float(), expected.float(), rtol=0, atol=0.02)


@pytest.mark.gpu
def test_graph_eviction_releases_only_its_operator_scope():
    from embodiinfer.policies.pi05.runtime import _LoopGraph

    policy = tiny_cuda_policy(Pi05OptimizationConfig())
    noise = torch.randn(1, 10, 8, device="cuda")
    batch = Pi05Batch(
        [torch.randn(1, 3, 8, 8, device="cuda")],
        [torch.tensor([True], device="cuda")],
        torch.tensor([[1, 2]], device="cuda"),
        torch.tensor([[True, True]], device="cuda"),
    )
    with torch.inference_mode():
        prefix = policy.encode_prefix(batch)
        first_output = policy._runtime.decode(noise, prefix, 10, True)
        old = next(iter(policy._runtime.loop_graphs.values()))
        operators = policy._fused_ops
        old_scope = ("graph", id(old.scope))
        assert any(key[-1] == old_scope for key in operators.kv_buffers)
        new = _LoopGraph(policy._runtime, noise, prefix, 10)
        policy._runtime._remember(policy._runtime.loop_graphs, "replacement", new, limit=1)
        assert not any(key[-1] == old_scope for key in operators.kv_buffers)
        assert any(key[-1] == ("graph", id(new.scope)) for key in operators.kv_buffers)
        assert torch.equal(new.run(noise, prefix), first_output)
        policy._clear_inference_caches()


@pytest.mark.gpu
def test_kv_only_route_is_byteexact_to_existing_native_path():
    policy = tiny_cuda_policy(None)
    batch = Pi05Batch(
        [torch.randn(1, 3, 8, 8, device="cuda")],
        [torch.tensor([True], device="cuda")],
        torch.tensor([[1, 2, 3]], device="cuda"),
        torch.tensor([[True, False, True]], device="cuda"),
    )
    noise = torch.randn(1, 10, 8, device="cuda")
    with torch.inference_mode():
        prefix = policy.encode_prefix(batch)
        expected = policy._runtime.decode(noise, prefix, 10, True)
        policy._clear_inference_caches()

        policy._optimization_config = Pi05OptimizationConfig(norm_fusion=False)
        migrated_prefix = policy.encode_prefix(batch)
        for actual_kv, reference_kv in zip(migrated_prefix.kv, prefix.kv):
            for actual, reference in zip(actual_kv, reference_kv):
                assert torch.equal(actual, reference)
        actual = policy._runtime.decode(noise, migrated_prefix, 10, True)
        assert torch.equal(actual, expected)
        policy._clear_inference_caches()


@pytest.mark.gpu
def test_registered_reference_normalizer_can_replace_cuda_without_policy_changes():
    from embodiinfer.backend.torch.normalization import TorchRMSNorm

    class RecordingNorm(TorchRMSNorm):
        calls = 0

        def plan(self, inputs, **options):
            type(self).calls += 1
            return super().plan(inputs, **options)

    normalization_backends.register("test_pi05_reference_norm", RecordingNorm)
    policy = tiny_cuda_policy(Pi05OptimizationConfig())
    batch = Pi05Batch(
        [torch.randn(1, 3, 8, 8, device="cuda")],
        [torch.tensor([True], device="cuda")],
        torch.tensor([[1, 2, 3]], device="cuda"),
        torch.tensor([[True, False, True]], device="cuda"),
    )
    noise = torch.randn(1, 10, 8, device="cuda")
    with torch.inference_mode():
        prefix = policy.encode_prefix(batch)
        expected = policy._runtime.decode(noise, prefix, 10, True)
        policy._clear_inference_caches()
        policy._optimization_config = Pi05OptimizationConfig(
            operators=OperatorBackends(normalization="test_pi05_reference_norm")
        )
        changed_prefix = policy.encode_prefix(batch)
        for actual_kv, expected_kv in zip(changed_prefix.kv, prefix.kv):
            for actual, reference in zip(actual_kv, expected_kv):
                assert torch.equal(actual, reference)
        actual = policy._runtime.decode(noise, changed_prefix, 10, True)
        assert torch.equal(actual, expected)
        assert RecordingNorm.calls > 0
        policy._clear_inference_caches()


@pytest.mark.gpu
def test_paired_backend_registry_packs_original_weight_layout():
    from embodiinfer.layers.launch import GemmTile

    request = OperatorRequest("cuda", torch.bfloat16, shape=(10, 64), cuda_graph=True)
    backend = paired_gelu_backends.get(
        "triton_lookup", request, tile=GemmTile(16, 32, 32, 4), approximate="tanh"
    )
    gate, up = (torch.randn(256, 64, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2))
    x = torch.randn(10, 64, device="cuda", dtype=torch.bfloat16)
    actual = backend.plan(gate, up)(x)
    expected = F.gelu(F.linear(x, gate), approximate="tanh") * F.linear(x, up)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.02)


@pytest.mark.gpu
def test_failed_capture_releases_new_scopes_and_can_retry(monkeypatch):
    policy = tiny_cuda_policy(Pi05OptimizationConfig())
    batch = Pi05Batch(
        [torch.randn(1, 3, 8, 8, device="cuda")],
        [torch.tensor([True], device="cuda")],
        torch.tensor([[1, 2]], device="cuda"),
        torch.tensor([[True, True]], device="cuda"),
    )
    original = policy._encode_prefix_impl
    calls = 0

    def fail_in_capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        output = original(*args, **kwargs)
        if calls == 4:
            raise RuntimeError("injected capture failure")
        return output

    monkeypatch.setattr(policy, "_encode_prefix_impl", fail_in_capture)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="injected capture failure"):
            policy.encode_prefix(batch)
        assert not policy._runtime.prefix_graphs
        assert not policy._fused_ops.norm_plans
        monkeypatch.setattr(policy, "_encode_prefix_impl", original)
        assert policy.encode_prefix(batch).batch_size == 1
        policy._clear_inference_caches()


@pytest.mark.gpu
def test_native_nvfp4_epilogue_matches_separate_quantization():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native NVFP4 positive validation requires Blackwell")
    from embodiinfer.backend.cuda.activation import GeluMulFusion
    from embodiinfer.backend.cuda.quantization import CudaQuantizer

    gate, up = (torch.randn(10, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    quantizer = CudaQuantizer()
    fusion = GeluMulFusion(quantizer, approximate="tanh")
    hidden = F.gelu(gate, approximate="tanh") * up
    expected = quantizer.plan(hidden, 4, 0.002).quantize(hidden)
    actual = fusion.plan(gate, 4, 0.002).encode(gate, up)
    assert torch.equal(actual.output, expected.output)
    assert torch.equal(actual.blocked.view(torch.uint8), expected.blocked.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.parametrize("precision,reference_ops", [("fp8", False), ("fp8", True), ("nvfp4", False)])
def test_mixed_projection_full_loop_graph_and_refit_invalidation(tmp_path, precision, reference_ops):
    import hashlib

    if not hasattr(F, "scaled_mm"):
        pytest.skip("mixed projection integration requires PyTorch's scaled_mm API")
    if precision == "nvfp4" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native NVFP4 GEMM validation requires Blackwell")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"synthetic calibration checkpoint identity")
    config = Pi05OptimizationConfig(
        action_layers=(ActionLayerPrecision(precision, precision, 6.0, 4.0),) * 2,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        operators=(
            OperatorBackends(normalization="torch", quantization="torch", gelu_mul="torch", norm_quant=None)
            if reference_ops
            else OperatorBackends()
        ),
    )
    policy = tiny_cuda_policy(config)
    policy.checkpoint = str(checkpoint)
    parameters = {name: id(value) for name, value in policy.named_parameters()}
    batch = Pi05Batch(
        [torch.randn(1, 3, 8, 8, device="cuda")],
        [torch.tensor([True], device="cuda")],
        torch.tensor([[1, 2, 3]], device="cuda"),
        torch.tensor([[True, True, True]], device="cuda"),
    )
    with torch.inference_mode():
        prefix = policy.encode_prefix(batch)
        for _ in range(2):
            noise = torch.randn(1, 10, 8, device="cuda")
            expected = policy._runtime.decode(noise, prefix, 10, False)
            actual = policy._runtime.decode(noise, prefix, 10, True)
            assert torch.isfinite(actual).all()
            assert torch.equal(actual, expected)
        if reference_ops:
            policy._clear_inference_caches()
            policy._optimization_config = Pi05OptimizationConfig(
                action_layers=config.action_layers, checkpoint_sha256=config.checkpoint_sha256
            )
            cuda_prefix = policy.encode_prefix(batch)
            cuda_output = policy._runtime.decode(noise, cuda_prefix, 10, True)
            assert torch.equal(cuda_output, actual)
        assert {name: id(value) for name, value in policy.named_parameters()} == parameters
        policy.on_refit(1)
        assert policy._fused_ops is None
        with pytest.raises(RuntimeError, match="calibration is stale"):
            policy.encode_prefix(batch)
