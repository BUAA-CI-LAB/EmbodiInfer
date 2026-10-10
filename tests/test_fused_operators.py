"""Model-independent CUDA/Triton fusion and rounding contracts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.layers import OperatorRequest, paired_gelu_backends


def _reference_norm(inputs, scale, eps, modulation):
    from embodiinfer.backend.torch.normalization import TorchRMSNorm

    plan = TorchRMSNorm().plan(inputs, adaptive=modulation is not None, eps=eps)
    return plan.normalize(inputs, scale=scale if modulation is None else None, modulation=modulation)


def test_half_width_rotary_rejects_invalid_inputs():
    from embodiinfer.backend.torch.rotary import TorchRotary

    rotation = TorchRotary()
    cosine, sine = torch.ones(1, 2, 1, 2), torch.zeros(1, 2, 1, 2)
    x = torch.ones(1, 2, 3, 4, dtype=torch.bfloat16)
    assert rotation(x, sine, cosine).dtype == torch.bfloat16
    assert cosine.dtype == sine.dtype == torch.float32
    with pytest.raises(ValueError, match="rank-four BF16"):
        rotation(torch.tensor(1), sine, cosine)
    with pytest.raises(ValueError, match="half-width factors"):
        rotation(x, sine.bfloat16(), cosine)


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
    expected, expected_gate = _reference_norm(x, scale, 1e-6, modulation)
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
    expected, expected_gate = _reference_norm(expected_residual, scale, 1e-6, modulation)
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
    from embodiinfer.layers.attention import attention_backend_capability

    available, reason = attention_backend_capability(name)
    if not available:
        pytest.skip(reason)

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
    expected, gate = _reference_norm(x, scale, 1e-6, modulation)
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
def test_thor_lookup_nvfp4_epilogue_preserves_separate_encoder_bytes():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native NVFP4 requires Blackwell")
    from embodiinfer.backend.cuda.gelu_lookup import LookupGeluMul
    from embodiinfer.backend.cuda.quantization import CudaQuantizer

    gate, up = (torch.randn(19, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    quantizer = CudaQuantizer()
    hidden = F.gelu(gate) * up
    expected = quantizer.plan(hidden, 4, 0.002).quantize(hidden)
    actual = LookupGeluMul(quantizer).plan(gate, 4, 0.002).encode(gate, up)
    assert torch.equal(actual.output, expected.output)
    assert torch.equal(actual.blocked.view(torch.uint8), expected.blocked.view(torch.uint8))
