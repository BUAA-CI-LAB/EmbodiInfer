"""GPU contracts of the vision segment attention and fused QKV RoPE kernels."""

import pytest
import torch

from embodiinfer.backend.triton.vision_attention import segment_attention
from embodiinfer.backend.triton.vision_rope import vision_qkv_rope

# Qwen2.5-VL-3B vision heads; an R2R frame is 352 patches in windows of 64/64/48.
HEADS, WIDTH = 16, 80


def _segments(lengths: list[int], seed: int):
    generator = torch.Generator().manual_seed(seed)
    shape = (sum(lengths), HEADS, WIDTH)
    q, k, v = (torch.randn(shape, generator=generator).bfloat16().cuda() for _ in range(3))
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return q, k, v, offsets


def _exact(q, k, v, offsets) -> torch.Tensor:
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    bounds = offsets.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        qs, ks, vs = (t[start:end].transpose(0, 1).double() for t in (q, k, v))
        weights = torch.softmax(qs @ ks.transpose(1, 2) * WIDTH**-0.5, -1)
        out[start:end] = (weights @ vs).transpose(0, 1)
    return out


def _relative(x: torch.Tensor, exact: torch.Tensor) -> float:
    return float((x.double() - exact).norm() / exact.norm())


def test_vision_kernels_reject_cpu_inputs():
    x = torch.zeros(4, 2, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA and Triton"):
        segment_attention(x, x, x, torch.tensor([0, 4]), 4)
    with pytest.raises(RuntimeError, match="CUDA and Triton"):
        vision_qkv_rope(torch.zeros(4, 96, dtype=torch.bfloat16), torch.zeros(4, 16), torch.zeros(4, 16), 2)


@pytest.mark.gpu
@pytest.mark.parametrize("lengths", [[64, 64, 48] * 8, [352] * 4])
@torch.inference_mode()
def test_segment_attention_error_stays_at_the_bf16_rounding_floor(lengths):
    q, k, v, offsets = _segments(lengths, seed=len(lengths))
    exact = _exact(q, k, v, offsets)
    floor = _relative(exact.bfloat16(), exact)
    faithful = _relative(segment_attention(q, k, v, offsets, max(lengths), block_m=32), exact)
    rounded = _relative(segment_attention(q, k, v, offsets, max(lengths), probability_parts=1), exact)
    assert faithful <= 1.1 * floor
    assert rounded > 1.15 * faithful


@pytest.mark.gpu
@torch.inference_mode()
def test_segment_attention_ignores_neighbouring_segments():
    q, k, v, offsets = _segments([352, 352, 352], seed=1)
    together = segment_attention(q, k, v, offsets, 352)
    middle = slice(352, 704)
    alone = segment_attention(*(t[middle].contiguous() for t in (q, k, v)), offsets[:2], 352)
    assert torch.equal(together[middle], alone)


@pytest.mark.gpu
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_vision_qkv_rope_matches_transformers_bitwise(table_dtype):
    vision = pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    torch.manual_seed(3)
    tokens = 352
    qkv = torch.randn(tokens, 3 * HEADS * WIDTH, device="cuda", dtype=torch.bfloat16)
    angle = torch.randn(tokens, WIDTH // 2, device="cuda").to(table_dtype)
    rotary = torch.cat((angle, angle), dim=-1)
    cos, sin = rotary.cos(), rotary.sin()
    q, k, v = qkv.reshape(tokens, 3, HEADS, WIDTH).permute(1, 0, 2, 3).unbind(0)
    q_expected, k_expected = vision.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q_actual, k_actual, v_actual = vision_qkv_rope(qkv, cos, sin, HEADS)
    assert torch.equal(q_actual, q_expected)
    assert torch.equal(k_actual, k_expected)
    assert torch.equal(v_actual, v)
