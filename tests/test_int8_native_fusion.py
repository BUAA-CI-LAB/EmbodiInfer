"""Fused native W8A8 must preserve payload, scales and every output rounding step."""

import pytest
import torch

from embodiinfer.backend.torch.int8 import _quantize_activation_per_row, native_linear, quantize_weight
from embodiinfer.backend.triton.int8_native import linear, quantize_activation, triton_linear
from embodiinfer.layers.linear import parse_int8_config, resolve_linear_backend


def test_native_fused_config_keeps_cpu_reference_routing():
    config = parse_int8_config({"method": "int8", "backend": "native_fused"})
    assert config.backend == "native_fused"
    assert (
        resolve_linear_backend("int8", config.backend, torch.zeros(1, 8), torch.zeros(8, 8), torch.ones(8))
        == "torch"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("execute", [linear, triton_linear])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,depth", [(1, 3584), (8, 512), (16, 768), (17, 3584), (31, 18944), (219, 512)])
def test_native_fusion_exact_payload_scales_output_and_capture(dtype, rows, depth, execute):
    torch.manual_seed(42)
    with torch.inference_mode():
        inputs = torch.randn(rows, depth, device="cuda", dtype=dtype)
        inputs[0].zero_()
        if rows > 1:
            inputs[1] = torch.linspace(-3.5, 3.5, depth, device="cuda").to(dtype)
        before = inputs.clone()
        expected_q, expected_s = _quantize_activation_per_row(inputs)
        actual_q, actual_s = quantize_activation(inputs)
        assert torch.equal(actual_q[:rows], expected_q)
        assert torch.equal(actual_s[:rows], expected_s)
        assert torch.equal(inputs, before)
        if rows <= 16:
            assert actual_q.shape[0] == 17 and (actual_q[rows:] == 0).all()
        weight, scales = quantize_weight(torch.randn(128, depth, device="cuda", dtype=dtype) * 0.01)
        bias = torch.randn(128, device="cuda", dtype=dtype)
        for offset in (None, bias):
            expected = native_linear(inputs, weight, scales, offset)
            actual = execute(inputs, weight, scales, offset)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                execute(inputs, weight, scales, bias)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = execute(inputs, weight, scales, bias)
        inputs.normal_()
        graph.replay()
        torch.testing.assert_close(
            captured, native_linear(inputs, weight, scales, bias), rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.gpu
def test_native_fusion_nonfinite_tiny_and_strided_rows_preserve_conversion():
    with torch.inference_mode():
        inputs = torch.randn(10, 64, device="cuda", dtype=torch.bfloat16)
        inputs[0].fill_(torch.finfo(torch.bfloat16).tiny)
        inputs[1].fill_(-torch.finfo(torch.bfloat16).tiny)
        inputs[2, 0] = float("nan")
        inputs[3, 0] = float("inf")
        inputs[4, 0] = -float("inf")
        inputs = inputs[:, ::2]
        expected_q, expected_s = _quantize_activation_per_row(inputs)
        actual_q, actual_s = quantize_activation(inputs)
        assert torch.equal(actual_q[:10], expected_q)
        torch.testing.assert_close(actual_s[:10], expected_s, rtol=0, atol=0, equal_nan=True)


@pytest.mark.gpu
def test_triton_native_rounding_on_full_streamvln_slow_memory_gate_projection():
    torch.manual_seed(42)
    with torch.inference_mode():
        inputs = torch.randn(1880, 3584, device="cuda", dtype=torch.bfloat16)
        weight, scales = quantize_weight(torch.randn(37888, 3584, device="cuda", dtype=torch.bfloat16) * 0.01)
        expected = native_linear(inputs, weight, scales)
        actual = triton_linear(inputs, weight, scales)
        assert torch.equal(actual, expected)
