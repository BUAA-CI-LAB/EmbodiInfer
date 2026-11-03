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


def test_operator_norm_keeps_native_adaptive_reference(monkeypatch):
    from embodiinfer.policies.pi05 import modeling_pi05
    from embodiinfer.policies.pi05.inference.operators import Pi05OperatorPlans

    plans = Pi05OperatorPlans.__new__(Pi05OperatorPlans)
    plans.norm = object()
    plans.config = SimpleNamespace(numerics="lerobot", has_quantized_mlp=False)
    calls = []
    monkeypatch.setattr(
        modeling_pi05, "_rmsnorm", lambda *args: calls.append(("norm", args[-1])) or ("h", "g")
    )
    monkeypatch.setattr(
        modeling_pi05, "_gated_residual", lambda *args: calls.append(("residual", args[-1])) or "r"
    )
    assert plans.normalize(None, None, "modulation", native=True) == ("h", "g")
    assert plans.residual_normalize(None, None, None, "gate", "modulation", native=True) == ("r", "h", "g")
    assert calls == [("norm", True), ("residual", True), ("norm", True)]


@pytest.mark.parametrize(
    "numerics,quantized,native",
    [("openpi_rlinf", False, True), ("lerobot", True, True), ("lerobot", False, False)],
)
def test_operator_norm_retains_other_fusion_contracts(monkeypatch, numerics, quantized, native):
    from embodiinfer.policies.pi05.inference.operators import Pi05OperatorPlans

    plans = Pi05OperatorPlans.__new__(Pi05OperatorPlans)
    plans.norm = object()
    plans.config = SimpleNamespace(numerics=numerics, has_quantized_mlp=quantized)
    plan = SimpleNamespace(normalize=lambda *args, **kwargs: ("fused", "gate"))
    monkeypatch.setattr(plans, "_norm_plan", lambda *args: plan)
    assert plans.normalize(None, None, "modulation", native=native) == ("fused", "gate")


@pytest.mark.gpu
def test_operator_norm_matches_native_triton_adaptive_reference():
    from embodiinfer.backend.cuda.normalization import NormFusion
    from embodiinfer.policies.pi05.inference.operators import Pi05OperatorPlans
    from embodiinfer.policies.pi05.modeling_pi05 import _gated_residual, _rmsnorm

    plans = Pi05OperatorPlans.__new__(Pi05OperatorPlans)
    plans.norm = NormFusion()
    plans.config = SimpleNamespace(numerics="lerobot", has_quantized_mlp=False)
    torch.manual_seed(79)
    inputs, update = (torch.randn(1, 50, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    modulation = torch.randn(1, 3072, device="cuda")
    gate = torch.randn(1, 1, 1024, device="cuda", dtype=torch.bfloat16)
    norm = SimpleNamespace(eps=1e-6)
    actual = plans.normalize(norm, inputs, modulation, native=True)
    expected = _rmsnorm(norm, inputs, None, False, modulation, True)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    residual, normalized, next_gate = plans.residual_normalize(
        norm, inputs, update, gate, modulation, native=True
    )
    expected_residual = _gated_residual(inputs, update, gate, False, True)
    expected = _rmsnorm(norm, expected_residual, None, False, modulation, True)
    assert torch.equal(residual, expected_residual)
    assert torch.equal(normalized, expected[0])
    assert torch.equal(next_gate, expected[1])


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


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [8, 16])
def test_packed_geglu_epilogue_matches_separate_inputs(bits):
    from embodiinfer.layers import OperatorRequest, gelu_mul_backends, quantization_backends

    request = OperatorRequest("cuda", torch.bfloat16, cuda_graph=True)
    quantizer = quantization_backends.get(
        "cuda", OperatorRequest("cuda", torch.bfloat16, bits=8, cuda_graph=True)
    )
    fusion = gelu_mul_backends.get("cuda", request, backend=quantizer, approximate="none")
    packed = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    gate, up = packed[:, :256].contiguous(), packed[:, 256:].contiguous()
    scale = 0.05 if bits == 8 else None
    expected = fusion.plan(gate, bits, scale).encode(gate, up).output.clone()
    actual = fusion.plan(gate, bits, scale).encode_packed(packed).output.clone()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive", [False, True])
def test_reduce_norm_and_norm_quant_match_reference_within_bf16_ulp(adaptive):
    from embodiinfer.layers import (
        OperatorRequest,
        norm_quant_backends,
        normalization_backends,
        quantization_backends,
    )

    request = OperatorRequest("cuda", torch.bfloat16, cuda_graph=True)
    q_req = OperatorRequest("cuda", torch.bfloat16, bits=8, cuda_graph=True)
    ref_norm = normalization_backends.get("cuda_strict", request)
    red_norm = normalization_backends.get("cuda_reduce", request)
    quant = quantization_backends.get("cuda", q_req)
    ref_nq = norm_quant_backends.get("cuda_strict", request, quantizer=quant, norm=ref_norm)
    red_nq = (
        norm_quant_backends.get("cuda_reduce", request, request_norm=red_norm)
        if False
        else norm_quant_backends.get("cuda_reduce", request, quantizer=quant, norm=red_norm)
    )

    x = torch.randn(1, 16, 256, device="cuda", dtype=torch.bfloat16)
    upd = torch.randn(1, 16, 256, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 1, 256, device="cuda", dtype=torch.bfloat16) if adaptive else None
    scale = None if adaptive else torch.randn(256, device="cuda", dtype=torch.float32)
    mod = torch.randn(1, 768, device="cuda", dtype=torch.float32) if adaptive else None

    r_res, r_out, _ = ref_norm.plan(x, adaptive=adaptive).residual_normalize(
        x, upd, gate, scale=scale, modulation=mod
    )
    d_res, d_out, _ = red_norm.plan(x, adaptive=adaptive).residual_normalize(
        x, upd, gate, scale=scale, modulation=mod
    )
    assert torch.equal(r_res, d_res)
    assert torch.allclose(r_out.float(), d_out.float(), atol=2e-2, rtol=1e-2)

    rq_res, rq_enc, _ = ref_nq.plan(x, bits=8, calibrated_scale=0.05, adaptive=adaptive).residual_normalize(
        x, upd, gate, scale=scale, modulation=mod
    )
    dq_res, dq_enc, _ = red_nq.plan(x, bits=8, calibrated_scale=0.05, adaptive=adaptive).residual_normalize(
        x, upd, gate, scale=scale, modulation=mod
    )
    assert torch.equal(rq_res, dq_res)
    assert torch.allclose(rq_enc.output.float(), dq_enc.output.float(), atol=8.0, rtol=2e-2)


@pytest.mark.gpu
def test_cutlass_prefix_geglu_rounds_reference_to_nearest():
    from embodiinfer.backend.cuda import cutlass_geglu
    from embodiinfer.backend.torch.nvfp4 import blocked_scales, quantization_thresholds, quantize_weight

    if not cutlass_geglu.available(torch.device("cuda")):
        pytest.skip("CUTLASS GeGLU requires EMBODIINFER_CUTLASS_DIR and Blackwell")
    torch.manual_seed(0)
    rows, depth, features = 300, 512, 256
    thresholds = quantization_thresholds(torch.empty(1, device="cuda")).double()
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64, device="cuda")

    def codes_of(packed):
        return torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)

    def dequantize(packed, scales):
        codes = codes_of(packed)
        values = torch.where(codes >= 8, -table[(codes & 7).long()], table[(codes & 7).long()])
        return (values.reshape(packed.shape[0], -1, 16) * scales.double().unsqueeze(-1)).flatten(1)

    (a, a_scale, a_global), (g, g_scale, g_global), (u, u_scale, u_global) = (
        quantize_weight(torch.randn(n, depth, device="cuda") * s, thresholds.float())
        for n, s in ((rows, 1.0), (features, depth**-0.5), (features, depth**-0.5))
    )
    a_blocked, g_blocked, u_blocked = (
        blocked_scales(s.view(torch.uint8)).view(torch.float8_e4m3fn) for s in (a_scale, g_scale, u_scale)
    )
    x = dequantize(a, a_scale)
    gate = x @ dequantize(g, g_scale).T * float(a_global * g_global)
    up = x @ dequantize(u, u_scale).T * float(a_global * u_global)
    inv = 0.9 * 448 * 6 / float((0.5 * gate * (1 + torch.erf(gate / 2**0.5)) * up).abs().amax())
    target = 0.5 * gate * (1 + torch.erf(gate / 2**0.5)) * up * inv

    out = torch.empty(rows, features // 2, dtype=torch.uint8, device="cuda")
    padded = -(-rows // 128) * 128
    out_blocked = torch.zeros(padded * (features // 16), dtype=torch.float8_e4m3fn, device="cuda")
    scalars = [
        torch.tensor([v], dtype=torch.float32, device="cuda")
        for v in (float(a_global * g_global), float(a_global * u_global), inv)
    ]
    cutlass_geglu.cutlass_geglu_fp4(
        a,
        a_blocked,
        *cutlass_geglu.interleave_gate_up(g, g_blocked, u, u_blocked),
        *scalars,
        out,
        out_blocked,
    )
    torch.cuda.synchronize()

    scales = out_blocked.view(torch.uint8).reshape(padded // 128, features // 64, 32, 4, 4)
    scales = scales.permute(0, 3, 2, 1, 4).reshape(padded, features // 16)[:rows]
    blocks = target.reshape(rows, -1, 16)
    expected_scales = (blocks.abs().amax(-1) / 6).clamp(max=448).float().to(torch.float8_e4m3fn)
    assert (scales == expected_scales.view(torch.uint8)).float().mean() >= 0.999
    normalized = (blocks / scales.view(torch.float8_e4m3fn).double().unsqueeze(-1)).clamp(-6, 6).flatten(1)
    expected = torch.bucketize(normalized.abs(), thresholds).to(torch.uint8) | (
        (normalized < 0).to(torch.uint8) << 3
    )
    assert (codes_of(out) == expected).float().mean() >= 0.995
    error = (dequantize(out, scales.view(torch.float8_e4m3fn)) - target).norm() / target.norm()
    assert error < 0.1


@pytest.mark.gpu
@pytest.mark.parametrize("batch,steps,keys", [(1, 10, 536), (2, 10, 300), (1, 2, 17)])
def test_mqa_decode_attention_matches_reference_and_bhsd_fp4_encoding(batch, steps, keys):
    import ctypes

    from embodiinfer.backend.cuda.mqa_decode_attention import MqaDecodeAttention, supported
    from embodiinfer.backend.cuda.native import build_library

    torch.manual_seed(0)
    heads, width, scale = 8, 256, 0.0625
    q = (torch.randn(batch, steps, heads, width, device="cuda") * 2).bfloat16().transpose(1, 2)
    k = (torch.randn(batch, 1, keys, width, device="cuda") * 2).bfloat16()
    v = torch.randn(batch, 1, keys, width, device="cuda").bfloat16()
    if not supported(q, k, v):
        pytest.skip("split-key decode attention requires Blackwell")
    plan = MqaDecodeAttention(q, k)
    rows, columns = batch * steps, heads * width
    out = torch.empty(rows, columns, device="cuda", dtype=torch.bfloat16)
    plan(q, k, v, scale, output=out)
    folded = q.transpose(1, 2).reshape(batch, steps * heads, width).double()
    expected = (folded @ k[:, 0].double().transpose(1, 2) * scale).softmax(-1) @ v[:, 0].double()
    error = (out.view(batch, steps * heads, width).double() - expected).norm() / expected.norm()
    assert error < 3e-3

    fp4 = torch.empty(rows, columns // 2, device="cuda", dtype=torch.uint8)
    blocked = torch.zeros(-(-rows // 128) * 128 * (columns // 16), device="cuda", dtype=torch.uint8)
    plan(q, k, v, scale, fp4=fp4, blocked=blocked, g_scale=0.01)
    library = build_library("norm_quant", strict=True, specific=True, native_fp4=True).library
    library.cc_bhsd_fp4.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_float]
        + [ctypes.c_int64] * 3
        + [ctypes.c_int] * 4
        + [ctypes.c_void_p]
    )
    library.cc_bhsd_fp4.restype = ctypes.c_int
    bhsd = out.view(batch, steps, heads, width).transpose(1, 2)
    reference = torch.empty_like(fp4)
    reference_blocked = torch.zeros_like(blocked)
    assert (
        library.cc_bhsd_fp4(
            bhsd.data_ptr(),
            reference.data_ptr(),
            reference_blocked.data_ptr(),
            0.01,
            *bhsd.stride()[:3],
            batch,
            heads,
            steps,
            width,
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    torch.cuda.synchronize()
    assert torch.equal(fp4, reference)
    assert torch.equal(blocked, reference_blocked)


@pytest.mark.gpu
def test_action_bridge_matches_separate_projections_and_euler_step():
    from embodiinfer.backend.cuda.action_bridge import ActionBridge

    torch.manual_seed(0)
    in_proj, out_proj = torch.nn.Linear(32, 1024).cuda(), torch.nn.Linear(1024, 32).cuda()
    hidden = torch.randn(1, 10, 1024, device="cuda").bfloat16()
    x = torch.randn(1, 10, 32, device="cuda")
    assert ActionBridge.supports(in_proj, out_proj, x)
    bridge = ActionBridge(in_proj, out_proj)
    x_next, embeds = bridge(hidden, x, -0.1, last=False)
    with torch.no_grad():
        expected = x + out_proj(hidden.float()) * -0.1
        expected_embeds = in_proj(expected).bfloat16()
    torch.testing.assert_close(x_next, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(embeds.float(), expected_embeds.float(), rtol=1e-2, atol=1e-2)
    x_last, none = bridge(hidden, x, -0.1, last=True)
    assert none is None and torch.equal(x_last, x_next)


@pytest.mark.gpu
def test_fa4_prefix_attention_matches_reference_when_configured():
    from embodiinfer.backend.cuda import fa4

    torch.manual_seed(0)
    q = (torch.randn(1, 8, 526, 256, device="cuda") * 2).bfloat16()
    k = (torch.randn(1, 1, 526, 256, device="cuda") * 2).bfloat16()
    v = torch.randn(1, 1, 526, 256, device="cuda").bfloat16()
    if not fa4.supported(q, k, v):
        pytest.skip("FlashAttention-4 is not configured (EMBODIINFER_FA4_DIR)")
    out = fa4.attend(q, k, v, 0.0625)
    expected = (q.double() @ k.double().transpose(-1, -2) * 0.0625).softmax(-1) @ v.double()
    assert out.shape == q.shape
    assert float((out.double() - expected).norm() / expected.norm()) < 3e-3


@pytest.mark.gpu
def test_camera_frame_normalization_rounds_like_host_float32():
    import numpy as np

    from embodiinfer.backend.cuda.camera_frames import normalize_frames

    frames = (
        torch.arange(3 * 9 * 11 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3, 9, 11, 3)
    )
    images = torch.empty(3, 3, 9, 11, device="cuda")
    normalize_frames(frames.cuda(), images)
    host = frames.numpy().astype(np.float32) / np.float32(255) * np.float32(2) - np.float32(1)
    torch.cuda.synchronize()
    assert torch.equal(images.cpu(), torch.from_numpy(host).permute(0, 3, 1, 2))
    with pytest.raises(ValueError, match="uint8"):
        normalize_frames(frames.cuda().float(), images)


@pytest.mark.gpu
@pytest.mark.parametrize("source", [(256, 256), (480, 640), (128, 96), (224, 300)])
def test_frame_resize_matches_pillow_resize_with_pad(source):
    import numpy as np
    from PIL import Image

    from embodiinfer.backend.cuda.camera_frames import FrameResize

    def resize_with_pad(image, size=224):
        original = Image.fromarray(image)
        width, height = original.size
        ratio = max(width / size, height / size)
        resized = original.resize((int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR)
        padded = Image.new(original.mode, (size, size), 0)
        padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
        return np.asarray(padded)

    frames = torch.randint(
        0, 256, (2, *source, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(0)
    )
    resize = FrameResize(source, (224, 224), torch.device("cuda"))
    images = torch.empty(2, 3, 224, 224, device="cuda")
    resize(frames.cuda(), resize.rows_buffer(2, torch.device("cuda")), images)
    resized = np.stack([resize_with_pad(frame) for frame in frames.numpy()])
    host = resized.astype(np.float32) / np.float32(255) * np.float32(2) - np.float32(1)
    torch.cuda.synchronize()
    assert torch.equal(images.cpu(), torch.from_numpy(host).permute(0, 3, 1, 2))
