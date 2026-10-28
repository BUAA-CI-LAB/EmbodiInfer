"""Preserve dynamic INT8 rounding and input ownership with bounded temporaries."""

import pytest
import torch

from embodiinfer.backend.torch import int8
from embodiinfer.backend.torch.int8 import _quantize_activation_per_row


def _reference(inputs):
    rows = inputs.reshape(-1, inputs.shape[-1])
    maximum = rows.float().abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    return (rows / scale).round().clamp(-127, 127).to(torch.int8), scale


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_int8_activation_exact_payload_scale_and_unchanged_input(dtype, noncontiguous, chunked, monkeypatch):
    if chunked:
        monkeypatch.setattr(int8, "_ACTIVATION_WORKSPACE_BYTES", 512)
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(4, 9, 64, generator=generator).to(dtype)
    inputs[0].zero_()
    inputs[1, 0] = torch.linspace(-127, 127, 64).to(dtype)
    inputs[1, 1] = torch.linspace(-3.5, 3.5, 64).to(dtype)
    inputs[2, 0].fill_(torch.finfo(dtype).tiny)
    inputs[2, 1].fill_(-torch.finfo(dtype).tiny)
    inputs[3, 0, 0] = float("nan")
    inputs[3, 1, 0] = float("inf")
    inputs[3, 2, 0] = -float("inf")
    if noncontiguous:
        inputs = inputs.transpose(0, 1)
    before = inputs.clone()
    expected = _reference(inputs)
    actual = _quantize_activation_per_row(inputs)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(inputs, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_large_prefill_quantization_reduces_peak_memory_without_drift():
    # StreamVLN's down projection at an eight-image slow-memory reset.
    inputs = torch.randn(1880, 18944, device="cuda", dtype=torch.bfloat16)
    before = inputs.clone()
    _quantize_activation_per_row(inputs)
    torch.cuda.synchronize()

    def measure(call):
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = call(inputs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        return result, peak

    expected, reference_peak = measure(_reference)
    actual, optimized_peak = measure(_quantize_activation_per_row)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(inputs, before)
    print(f"activation temporary bytes: reference={reference_peak}, bounded={optimized_peak}")
    assert reference_peak - optimized_peak >= inputs.numel() * 4
