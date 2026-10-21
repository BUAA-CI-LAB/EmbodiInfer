"""Contracts of the decode-size Triton linears."""

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.backend.triton import decode_linear as module
from embodiinfer.backend.triton.decode_linear import decode_linear, decode_qkv, decode_swiglu, decode_tiles
from embodiinfer.backend.triton.rounded_ops import rounded_swiglu

SHAPES = [(2048, 2048), (2048, 11008), (11008, 2048)]


def _tensor(*shape, generator, scale=1.0):
    return (torch.randn(*shape, generator=generator) * scale).bfloat16().cuda()


def _operands(rows: int, depth: int, width: int, bias: bool, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    inputs = _tensor(rows, depth, generator=generator)
    weight = _tensor(width, depth, generator=generator, scale=0.02)
    offset = _tensor(width, generator=generator, scale=0.1) if bias else None
    return inputs, weight, offset


def _force_triton(monkeypatch):
    monkeypatch.setattr(module, "decode_tiles", lambda *_: (16, 32, 128, 1, 4, 4))


def test_tiles_cover_decode_rows_and_leave_the_rest_to_cublas():
    assert decode_tiles(12, 2048, 2048) is not None
    assert decode_tiles(128, 11008, 2048) is not None
    assert decode_tiles(96, 2048, 11008) is not None
    assert decode_tiles(128, 2048, 11008) is None
    assert decode_tiles(129, 2048, 2048) is None
    assert decode_tiles(12, 256, 2048) is None
    assert decode_tiles(0, 2048, 2048) is None
    assert decode_tiles(12, 2048, 2000) is None


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 4, 12, 36, 72, 96, 128])
@pytest.mark.parametrize(("depth", "width"), SHAPES)
def test_matches_fp32_projection_to_output_rounding(rows, depth, width):
    if decode_tiles(rows, width, depth) is None:
        pytest.skip("cuBLAS shape")
    inputs, weight, bias = _operands(rows, depth, width, bias=depth == width)
    output = decode_linear(inputs[None], weight, bias)
    exact = inputs.double() @ weight.double().T + (0 if bias is None else bias.double())
    assert output.shape == (1, rows, width) and output.dtype == torch.bfloat16
    # One BF16 rounding of the FP32 sum, plus FP32 accumulation slack.
    tolerance = exact.abs() * 2**-8 + 1e-3
    assert ((output[0].double() - exact).abs() <= tolerance).all()
    reference = F.linear(inputs, weight, bias)
    assert (output[0].float() - reference.float()).abs().max() <= reference.float().abs().max() * 2**-7


@pytest.mark.gpu
@pytest.mark.parametrize(("depth", "width"), SHAPES)
def test_rows_depend_on_neither_tiles_nor_row_count(depth, width, monkeypatch):
    inputs, weight, bias = _operands(96, depth, width, bias=True, seed=1)
    reference = decode_linear(inputs, weight, bias)
    assert torch.equal(decode_linear(inputs[:5], weight, bias), reference[:5])
    every_tile = {tiles for table in module._TILES.values() for _, tiles in table}
    for tiles in every_tile:
        monkeypatch.setattr(module, "decode_tiles", lambda *_, tiles=tiles: tiles)
        assert torch.equal(decode_linear(inputs, weight, bias), reference), tiles


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [4, 36, 96, 128, 192])
def test_grouped_qkv_equals_separate_triton_projections(rows, monkeypatch):
    generator = torch.Generator().manual_seed(rows)
    inputs = _tensor(2, rows // 2, 2048, generator=generator)
    weights = [_tensor(width, 2048, generator=generator, scale=0.02) for width in (2048, 256, 256)]
    biases = [_tensor(width, generator=generator, scale=0.1) for width in (2048, 256, 256)]
    grouped = decode_qkv(inputs, weights, biases)
    for output, weight, bias in zip(grouped, weights, biases, strict=True):
        reference = F.linear(inputs, weight, bias).float()
        assert output.shape == reference.shape
        assert (output.float() - reference).abs().max() <= reference.abs().max() * 2**-7
    _force_triton(monkeypatch)
    separate = [decode_linear(inputs, weight, bias) for weight, bias in zip(weights, biases, strict=True)]
    assert all(torch.equal(a, b) for a, b in zip(grouped, separate, strict=True))


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [4, 36, 96, 128, 192])
def test_fused_swiglu_equals_rounded_swiglu_of_triton_projections(rows, monkeypatch):
    generator = torch.Generator().manual_seed(rows)
    inputs = _tensor(rows, 2048, generator=generator)
    gate, up = (_tensor(11008, 2048, generator=generator, scale=0.02) for _ in range(2))
    fused = decode_swiglu(inputs, gate, up)
    reference = rounded_swiglu(F.linear(inputs, gate), F.linear(inputs, up)).float()
    assert (fused.float() - reference).abs().max() <= reference.abs().max() * 2**-6
    _force_triton(monkeypatch)
    assert torch.equal(fused, rounded_swiglu(decode_linear(inputs, gate), decode_linear(inputs, up)))


@pytest.mark.gpu
def test_unsupported_inputs_fall_back_to_cublas():
    inputs, weight, bias = _operands(256, 2048, 2048, bias=True)
    assert torch.equal(decode_linear(inputs, weight, bias), F.linear(inputs, weight, bias))
    narrow = weight[:256].contiguous()
    assert torch.equal(decode_linear(inputs[:8], narrow), F.linear(inputs[:8], narrow))
    transposed = weight.T.contiguous().T
    assert torch.equal(decode_linear(inputs[:8], transposed), F.linear(inputs[:8], transposed))
    single = inputs[:8].float(), weight.float()
    assert torch.equal(decode_linear(*single), F.linear(*single))
    weights, biases = [weight, narrow, narrow], [bias, bias[:256], bias[:256]]
    expected = [F.linear(inputs, w, b) for w, b in zip(weights, biases, strict=True)]
    assert all(map(torch.equal, decode_qkv(inputs, weights, biases), expected))
    wide = torch.cat([weight] * 5)
    expected = rounded_swiglu(F.linear(inputs, wide), F.linear(inputs, wide.flip(0)))
    assert torch.equal(decode_swiglu(inputs, wide, wide.flip(0).contiguous()), expected)
