"""Projection bias must be added before the final BF16 rounding boundary."""

import pytest
import torch

from embodiinfer.backend.torch.linear import linear_fp32_output


def test_fp32_output_linear_rejects_gradient_execution_before_dispatch():
    with pytest.raises(ValueError, match="inference-only"):
        linear_fp32_output(torch.ones(1, 2), torch.ones(1, 2))


@pytest.mark.gpu
@torch.inference_mode()
def test_fp32_output_linear_rounds_after_bias_and_preserves_leading_dimensions():
    inputs = torch.ones(2, 3, 2, dtype=torch.bfloat16, device="cuda")
    weight = torch.tensor([[1, 1 / 256], [-1, -1 / 256]], dtype=torch.bfloat16, device="cuda")
    bias = torch.tensor([1 / 256, -1 / 256], dtype=torch.bfloat16, device="cuda")
    expected = torch.tensor([1 + 1 / 128, -1 - 1 / 128], device="cuda", dtype=torch.bfloat16)
    actual = linear_fp32_output(inputs, weight, bias)
    assert actual.shape == (2, 3, 2)
    torch.testing.assert_close(actual, expected.expand_as(actual), atol=0, rtol=0)
