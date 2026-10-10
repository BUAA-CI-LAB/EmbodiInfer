"""Pi05 full-loop CUDA Graph, refit and lifetime integration checks."""

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.layers import OperatorBackends, normalization_backends
from embodiinfer.policies.pi05.inference.config import MlpLayerPrecision, Pi05OptimizationConfig
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "attention,paired_reference",
    [("reference", False), ("folded_flash", False), ("query_major", False), ("reference", True)],
)
def test_complete_native_graph_replays_changed_observations_and_owns_outputs(
    make_pi05_policy, attention, paired_reference
):
    if attention != "reference":
        from embodiinfer.layers.attention import attention_backend_capability

        available, reason = attention_backend_capability(attention)
        if not available:
            pytest.skip(reason)
    torch.manual_seed(34)
    policy = make_pi05_policy(
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


def test_graph_eviction_releases_only_its_operator_scope(make_pi05_policy):
    from embodiinfer.policies.pi05.inference.runtime import _LoopGraph

    policy = make_pi05_policy(Pi05OptimizationConfig())
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


def test_kv_only_route_is_byteexact_to_existing_native_path(make_pi05_policy):
    policy = make_pi05_policy(None)
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


def test_registered_reference_normalizer_can_replace_cuda_without_policy_changes(make_pi05_policy):
    from embodiinfer.backend.torch.normalization import TorchRMSNorm

    class RecordingNorm(TorchRMSNorm):
        calls = 0

        def plan(self, inputs, **options):
            type(self).calls += 1
            return super().plan(inputs, **options)

    normalization_backends.register("test_pi05_reference_norm", RecordingNorm)
    policy = make_pi05_policy(Pi05OptimizationConfig())
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


def test_failed_capture_releases_new_scopes_and_can_retry(make_pi05_policy, monkeypatch):
    policy = make_pi05_policy(Pi05OptimizationConfig())
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


def test_rlinf_cache_only_and_shared_context_preserve_graph_actions(make_pi05_policy, monkeypatch):
    from dataclasses import replace

    from embodiinfer.policies.pi05.embeddings import rlinf_time_embedding

    base = Pi05OptimizationConfig(
        numerics="rlinf", activation="gelu_pytorch_exact", operators=OperatorBackends(rotary="cuda")
    )
    policy = make_pi05_policy(base)
    policy._sinusoidal = rlinf_time_embedding
    policy.prefix_cuda_graph = False
    batch = Pi05Batch(
        [torch.randn(1, 3, 8, 8, device="cuda")],
        [torch.tensor([True], device="cuda")],
        torch.tensor([[1, 2, 3]], device="cuda"),
        torch.tensor([[True, False, True]], device="cuda"),
    )
    noise = torch.randn(1, 10, 8, device="cuda")
    with torch.inference_mode():
        reference_prefix = policy.encode_prefix(batch)
        expected = policy._runtime.decode(noise, reference_prefix, 10, False)
        policy._clear_inference_caches()
        policy._optimization_config = replace(base, prefix_kv_only=True, reuse_action_context=True)

        def unused(*_args, **_kwargs):
            raise AssertionError("cache-only prefix executed the final MLP")

        monkeypatch.setattr(policy._prefix_tower.layers[-1].mlp.gate_proj, "forward", unused)
        prefix = policy.encode_prefix(batch)
        for actual_kv, expected_kv in zip(prefix.kv, reference_prefix.kv, strict=True):
            for actual, reference in zip(actual_kv, expected_kv, strict=True):
                assert torch.equal(actual, reference)
        calls = 0
        original = policy._action_context

        def count_context(*args):
            nonlocal calls
            calls += 1
            return original(*args)

        monkeypatch.setattr(policy, "_action_context", count_context)
        assert torch.equal(policy._runtime.decode(noise, prefix, 10, False), expected)
        assert calls == 1
        assert policy._fused_ops.action_context is None
        assert torch.equal(policy._runtime.decode(noise, prefix, 10, True), expected)
        changed = noise.neg()
        assert torch.equal(
            policy._runtime.decode(changed, prefix, 10, True),
            policy._runtime.decode(changed, prefix, 10, False),
        )
        policy._clear_inference_caches()


@pytest.mark.parametrize(
    "tower,precision,reference_ops",
    [
        (tower, precision, reference_ops)
        for tower in ("action", "prefix")
        for precision, reference_ops in (("fp8", False), ("fp8", True), ("nvfp4", False), ("nvfp4", True))
    ]
    + [("both", "nvfp4", False)],
)
def test_mixed_projection_full_loop_graph_and_refit_invalidation(
    make_pi05_policy, tmp_path, precision, reference_ops, tower
):
    import hashlib

    if not hasattr(F, "scaled_mm"):
        pytest.skip("mixed projection integration requires PyTorch's scaled_mm API")
    if precision == "nvfp4" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native NVFP4 GEMM validation requires Blackwell")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"synthetic calibration checkpoint identity")
    formats = {tower: precision} if tower != "both" else {"action": "fp8", "prefix": "nvfp4"}
    config = Pi05OptimizationConfig(
        **{
            f"{scope}_layers": (MlpLayerPrecision(value, value, 6.0, 4.0),) * 2
            for scope, value in formats.items()
        },
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        operators=(
            OperatorBackends(
                normalization="torch",
                quantization="torch" if precision == "fp8" else "cuda",
                gelu_mul="torch",
                norm_quant=None,
            )
            if reference_ops
            else OperatorBackends()
        ),
    )
    policy = make_pi05_policy(config)
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
        for scope, value in formats.items():
            selected = policy._prefix_tower if scope == "prefix" else policy._expert_tower
            assert all(
                policy._fused_ops.mlp_plans[id(layer.mlp)].gate.precision == value
                for layer in selected.layers
            )
        if reference_ops:
            policy._clear_inference_caches()
            policy._optimization_config = Pi05OptimizationConfig(
                action_layers=config.action_layers,
                prefix_layers=config.prefix_layers,
                checkpoint_sha256=config.checkpoint_sha256,
            )
            cuda_prefix = policy.encode_prefix(batch)
            cuda_output = policy._runtime.decode(noise, cuda_prefix, 10, True)
            assert torch.equal(cuda_output, actual)
        assert {name: id(value) for name, value in policy.named_parameters()} == parameters
        policy.on_refit(1)
        assert policy._fused_ops is None
        with pytest.raises(RuntimeError, match="calibration is stale"):
            policy.encode_prefix(batch)
