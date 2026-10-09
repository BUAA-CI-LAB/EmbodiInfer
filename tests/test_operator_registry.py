"""Extension, support checks and reference contracts for reusable operators."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.layers import (
    BackendRegistry,
    OperatorBackends,
    OperatorCapabilities,
    OperatorRequest,
    gelu_mul_backends,
    normalization_backends,
    paired_gelu_backends,
    projection_backends,
    quantization_backends,
)
from embodiinfer.policies.pi05 import Pi05OptimizationConfig


class _Backend:
    capabilities = OperatorCapabilities(
        ("cpu",), (torch.float32,), "test", cuda_graph=False, rank=2, width_multiple=64, bits=(8,)
    )

    def __init__(self, *, value=1):
        self.value = value


def test_lazy_names_do_not_import_implementations():
    registry = BackendRegistry("test")
    registry.register_lazy("optional", "module_that_does_not_exist", "Backend")
    assert registry.available() == ("optional",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("optional", _Backend)
    with pytest.raises(ValueError, match="Unknown test"):
        registry.get("missing", OperatorRequest("cpu", torch.float32))


@pytest.mark.parametrize(
    "operator_request,message",
    [
        (OperatorRequest("cpu", torch.bfloat16), "does not support"),
        (OperatorRequest("cpu", torch.float32, shape=(2, 3, 64)), "rank"),
        (OperatorRequest("cpu", torch.float32, shape=(2, 65)), "divisible"),
        (OperatorRequest("cpu", torch.float32, bits=4), "4-bit"),
        (OperatorRequest("cpu", torch.float32, cuda_graph=True), "CUDA Graph"),
    ],
)
def test_unsupported_requests_fail_before_backend_construction(operator_request, message):
    class MustNotConstruct(_Backend):
        def __init__(self):
            raise AssertionError("constructed an unsupported backend")

    registry = BackendRegistry("test")
    registry.register("explicit", MustNotConstruct)
    with pytest.raises(ValueError, match=message):
        registry.get("explicit", operator_request)


def test_registry_preserves_factory_options_and_requires_capabilities():
    registry = BackendRegistry("test")
    registry.register("plain", _Backend)
    assert registry.get("plain", OperatorRequest("cpu", torch.float32), value=9).value == 9
    registry.register("undeclared", object)
    with pytest.raises(TypeError, match="declare OperatorCapabilities"):
        registry.get("undeclared", OperatorRequest("cpu", torch.float32))


def test_cuda_construction_is_rejected_during_capture(monkeypatch):
    class CudaBackend(_Backend):
        capabilities = OperatorCapabilities(("cuda",), (torch.bfloat16,), "test")

        def __init__(self):
            raise AssertionError("constructed during capture")

    registry = BackendRegistry("test")
    registry.register("cuda", CudaBackend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
        registry.get("cuda", OperatorRequest("cuda:0", torch.bfloat16))


def test_fp4_support_is_checked_before_native_construction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    with pytest.raises(ValueError, match="4-bit encoding requires SM"):
        quantization_backends.get("cuda", OperatorRequest("cuda:0", torch.bfloat16, bits=4))


@pytest.mark.parametrize("adaptive", [False, True])
def test_torch_normalization_contract_and_ephemeral_storage(adaptive):
    backend = normalization_backends.get("torch", OperatorRequest("cpu", torch.bfloat16, shape=(2, 3, 64)))
    x, update = (torch.randn(2, 3, 64).bfloat16() for _ in range(2))
    scale = torch.randn(64)
    modulation = torch.randn(2, 192) if adaptive else None
    gate = torch.randn(2, 1, 64).bfloat16()
    plan = backend.plan(x, adaptive=adaptive)
    with torch.inference_mode():
        residual, actual, actual_gate = plan.residual_normalize(
            x, update, gate, scale=scale, modulation=modulation
        )
        expected_residual = x + update * gate
        assert torch.equal(residual, expected_residual)
        normalized = expected_residual * torch.rsqrt(
            expected_residual.float().square().mean(-1, keepdim=True) + 1e-6
        )
        if adaptive:
            values, shift, expected_gate = modulation[:, None].chunk(3, dim=-1)
            normalized = normalized * (1 + values) + shift
            assert torch.equal(actual_gate, expected_gate.bfloat16())
        else:
            normalized *= 1 + scale
        assert torch.equal(actual, normalized.bfloat16())
        pointer = actual.data_ptr()
        x.mul_(0.2)
        again, _ = plan.normalize(x, scale=scale, modulation=modulation)
        assert again.data_ptr() == pointer


@pytest.mark.parametrize("bits", [8, 16])
def test_reference_gelu_encoding_uses_registered_quantizer(bits):
    request = OperatorRequest("cpu", torch.bfloat16)
    quantizer = quantization_backends.get("torch", request)
    backend = gelu_mul_backends.get("torch", request, backend=quantizer, approximate="tanh")
    gate, up = (torch.randn(7, 64).bfloat16() for _ in range(2))
    scale = 0.01 if bits == 8 else None
    plan = backend.plan(gate, bits, scale)
    with torch.inference_mode():
        result = plan.encode(gate, up)
        hidden = F.gelu(gate, approximate="tanh") * up
        expected = hidden if bits == 16 else (hidden.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        assert torch.equal(result.output.view(torch.uint8), expected.view(torch.uint8))


def test_paired_and_projection_plans_retain_parameters():
    request = OperatorRequest("cpu", torch.bfloat16)
    gate, up = (torch.nn.Parameter(torch.randn(128, 64).bfloat16()) for _ in range(2))
    x = torch.randn(2, 64).bfloat16()
    paired = paired_gelu_backends.get("torch", request, approximate="tanh").plan(gate, up)
    projection = projection_backends.get("torch", request).plan(
        gate, 1.0, "bf16", None, lambda device: ("stream", 0)
    )
    with torch.inference_mode():
        expected = F.gelu(F.linear(x, gate), approximate="tanh") * F.linear(x, up)
        assert torch.equal(paired(x), expected)
        assert torch.equal(projection.apply(x, None), F.linear(x, gate))
        gate.add_(1)
        assert torch.equal(projection.apply(x, None), F.linear(x, gate))


def test_nested_backend_configuration_roundtrips(tmp_path):
    operators = OperatorBackends(
        normalization="torch", quantization="torch", gelu_mul="torch", norm_quant=None, paired_gelu="torch"
    )
    config = Pi05OptimizationConfig(fused_mlp=True, operators=operators)
    path = tmp_path / "operators.json"
    config.to_json(path)
    assert Pi05OptimizationConfig.from_json(path) == config
    with pytest.raises(ValueError, match="nonempty"):
        OperatorBackends(normalization="")
