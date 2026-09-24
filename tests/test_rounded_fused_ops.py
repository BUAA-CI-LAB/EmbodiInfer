"""GPU numerical contracts for inference fusion with explicit low-precision rounding."""

import pytest
import torch

from embodiinfer.backend.triton.rounded_ops import rounded_rms_norm, rounded_rope, rounded_swiglu


def test_rounded_operators_reject_cpu_inputs():
    with pytest.raises(ValueError, match="CUDA and Triton"):
        rounded_rms_norm(torch.ones(1, 4), torch.ones(4), 1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 32, 160])
@torch.inference_mode()
def test_rounded_norm_and_swiglu_match_bf16_torch_exactly(rows):
    torch.manual_seed(32)
    x = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    expected_norm = weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(
        x.dtype
    )
    torch.testing.assert_close(rounded_rms_norm(x, weight, 1e-6), expected_norm, atol=0, rtol=0)
    up = torch.randn_like(x)
    torch.testing.assert_close(rounded_swiglu(x, up), torch.nn.functional.silu(x) * up, atol=0, rtol=0)


@pytest.mark.gpu
@torch.inference_mode()
def test_rounded_swiglu_preserves_every_finite_bf16_input_including_signed_zero():
    values = torch.arange(65536, device="cuda", dtype=torch.int32).short().view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    up = torch.full_like(values, 1.125)
    expected = torch.nn.functional.silu(values) * up
    actual = rounded_swiglu(values, up)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1, 160])
@torch.inference_mode()
def test_rounded_rotary_products_match_torch_exactly(length):
    torch.manual_seed(5)
    query = torch.randn(1, length, 16, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    key = torch.randn(1, length, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    angle = torch.randn(length, 128, device="cuda")
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()

    def reference(x):
        rotated = torch.cat((-x[..., 64:], x[..., :64]), dim=-1)
        return x * cos[None, None] + rotated * sin[None, None]

    q_out, k_out = rounded_rope(query, key, cos, sin)
    torch.testing.assert_close(q_out, reference(query), atol=0, rtol=0)
    torch.testing.assert_close(k_out, reference(key), atol=0, rtol=0)
