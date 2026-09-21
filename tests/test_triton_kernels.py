"""Correctness tests for optional Triton kernels; no performance claims."""

from __future__ import annotations

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.backend.triton import triton_capability
from embodiinfer.backend.triton.packed_rope import (
    rotate_half_rope,
    warmup_rotate_half_rope,
)
from embodiinfer.backend.triton.segmented_attention import (
    segmented_attention,
    warmup_segmented_attention,
)

QWEN_VISION_HEADS = 16
QWEN_VISION_KV_HEADS = 4
QWEN_VISION_HEAD_DIM = 80


@pytest.fixture(scope="module")
def triton_device() -> torch.device:
    capability = triton_capability()
    if not capability.available:
        pytest.skip(f"Triton CUDA unavailable: {capability.reason}")
    return torch.device("cuda", torch.cuda.current_device())


def _offsets(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor((0, *accumulate(lengths)), device=device, dtype=torch.int32)


def _packed_qkv(
    q_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        sum(q_lengths),
        QWEN_VISION_HEADS,
        QWEN_VISION_HEAD_DIM,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        sum(kv_lengths),
        QWEN_VISION_KV_HEADS,
        QWEN_VISION_HEAD_DIM,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        sum(kv_lengths),
        QWEN_VISION_KV_HEADS,
        QWEN_VISION_HEAD_DIM,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return q, k, v


def _segmented_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
) -> torch.Tensor:
    outputs = []
    q_start = 0
    kv_start = 0
    repeats = q.shape[1] // k.shape[1]
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        q_segment = q[q_start : q_start + q_length].transpose(0, 1).unsqueeze(0)
        k_segment = k[kv_start : kv_start + kv_length].transpose(0, 1).unsqueeze(0)
        v_segment = v[kv_start : kv_start + kv_length].transpose(0, 1).unsqueeze(0)
        if repeats != 1:
            k_segment = k_segment.repeat_interleave(repeats, dim=1)
            v_segment = v_segment.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(
            q_segment,
            k_segment,
            v_segment,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
        q_start += q_length
        kv_start += kv_length
    return torch.cat(outputs, dim=0)


def _rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_fp32 = cos.float().unsqueeze(1)
    sin_fp32 = sin.float().unsqueeze(1)
    q_fp32 = q.float()
    k_fp32 = k.float()
    return (
        q_fp32 * cos_fp32 + _rotate_half_reference(q_fp32) * sin_fp32,
        k_fp32 * cos_fp32 + _rotate_half_reference(k_fp32) * sin_fp32,
    )


def _tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.bfloat16:
        return {"rtol": 4e-2, "atol": 5e-2}
    return {"rtol": 2e-2, "atol": 2e-2}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths"),
    [
        ((7, 19, 31), (11, 13, 37)),
        ((396,), (396,)),
        ((1224,), (1224,)),
    ],
    ids=["short-variable-cross-attention", "vision-396", "vision-1224"],
)
@torch.inference_mode()
def test_segmented_attention_matches_per_segment_sdpa(
    triton_device: torch.device,
    dtype: torch.dtype,
    q_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
):
    q, k, v = _packed_qkv(q_lengths, kv_lengths, dtype, triton_device, seed=1234)
    q_offsets = _offsets(q_lengths, triton_device)
    kv_offsets = _offsets(kv_lengths, triton_device)

    actual = segmented_attention(
        q,
        k,
        v,
        q_offsets,
        kv_offsets,
        max_query_length=max(q_lengths),
        max_key_length=max(kv_lengths),
    )
    expected = _segmented_sdpa_reference(q, k, v, q_lengths, kv_lengths)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, **_tolerances(dtype))


@torch.inference_mode()
def test_segmented_attention_rejects_empty_segments(
    triton_device: torch.device,
):
    lengths = (5, 0, 7)
    q, k, v = _packed_qkv(lengths, lengths, torch.float16, triton_device, seed=11)
    offsets = _offsets(lengths, triton_device)

    with pytest.raises(ValueError, match="non-empty increasing segments"):
        segmented_attention(
            q,
            k,
            v,
            offsets,
            max_query_length=7,
            max_key_length=7,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch.inference_mode()
def test_rotate_half_rope_matches_fp32_torch_reference(
    triton_device: torch.device,
    dtype: torch.dtype,
):
    tokens = 396
    generator = torch.Generator(device=triton_device).manual_seed(4321)
    q = torch.randn(
        tokens,
        QWEN_VISION_HEADS,
        QWEN_VISION_HEAD_DIM,
        device=triton_device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        tokens,
        QWEN_VISION_KV_HEADS,
        QWEN_VISION_HEAD_DIM,
        device=triton_device,
        dtype=dtype,
        generator=generator,
    )
    positions = torch.randn(
        tokens,
        QWEN_VISION_HEAD_DIM,
        device=triton_device,
        dtype=torch.float32,
        generator=generator,
    )
    cos = positions.cos().contiguous()
    sin = positions.sin().contiguous()

    actual_q, actual_k = rotate_half_rope(q, k, cos, sin)
    expected_q, expected_k = _rope_reference(q, k, cos, sin)

    torch.testing.assert_close(actual_q.float(), expected_q, **_tolerances(dtype))
    torch.testing.assert_close(actual_k.float(), expected_k, **_tolerances(dtype))


@torch.inference_mode()
def test_warmup_then_cuda_graph_replay_uses_changed_inputs(
    triton_device: torch.device,
):
    if not hasattr(torch.cuda, "CUDAGraph"):
        pytest.skip("PyTorch CUDA Graph API is unavailable")

    dtype = torch.bfloat16
    q_lengths = (11, 23)
    kv_lengths = (17, 29)
    q, k, v = _packed_qkv(q_lengths, kv_lengths, dtype, triton_device, seed=99)
    q_offsets = _offsets(q_lengths, triton_device)
    kv_offsets = _offsets(kv_lengths, triton_device)
    rope_q = q.clone()
    rope_k = k[: sum(q_lengths)].clone()
    positions = torch.randn(
        sum(q_lengths),
        QWEN_VISION_HEAD_DIM,
        device=triton_device,
        dtype=torch.float32,
    )
    cos = positions.cos().contiguous()
    sin = positions.sin().contiguous()

    warmup_segmented_attention(
        q,
        k,
        v,
        q_offsets,
        kv_offsets,
        max_query_length=max(q_lengths),
        max_key_length=max(kv_lengths),
    )
    warmup_rotate_half_rope(rope_q, rope_k, cos, sin)
    torch.cuda.synchronize(triton_device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_attention = segmented_attention(
            q,
            k,
            v,
            q_offsets,
            kv_offsets,
            max_query_length=max(q_lengths),
            max_key_length=max(kv_lengths),
        )
        graph_rope_q, graph_rope_k = rotate_half_rope(rope_q, rope_k, cos, sin)

    graph.replay()
    torch.cuda.synchronize(triton_device)
    first_attention = graph_attention.clone()
    first_rope_q = graph_rope_q.clone()
    first_rope_k = graph_rope_k.clone()

    q.copy_(torch.randn(q.shape, device=triton_device, dtype=dtype))
    k.copy_(torch.randn(k.shape, device=triton_device, dtype=dtype))
    v.copy_(torch.randn(v.shape, device=triton_device, dtype=dtype))
    rope_q.copy_(torch.randn(rope_q.shape, device=triton_device, dtype=dtype))
    rope_k.copy_(torch.randn(rope_k.shape, device=triton_device, dtype=dtype))
    positions.copy_(torch.randn_like(positions))
    cos.copy_(positions.cos())
    sin.copy_(positions.sin())

    graph.replay()
    torch.cuda.synchronize(triton_device)
    expected_attention = _segmented_sdpa_reference(q, k, v, q_lengths, kv_lengths)
    expected_rope_q, expected_rope_k = _rope_reference(rope_q, rope_k, cos, sin)

    assert not torch.equal(first_attention, graph_attention)
    assert not torch.equal(first_rope_q, graph_rope_q)
    assert not torch.equal(first_rope_k, graph_rope_k)
    torch.testing.assert_close(graph_attention, expected_attention, **_tolerances(dtype))
    torch.testing.assert_close(graph_rope_q.float(), expected_rope_q, **_tolerances(dtype))
    torch.testing.assert_close(graph_rope_k.float(), expected_rope_k, **_tolerances(dtype))


def test_segmented_attention_on_cpu_fails_loudly():
    q = torch.randn(4, 2, 8)
    k = torch.randn(4, 1, 8)
    v = torch.randn(4, 1, 8)
    offsets = torch.tensor([0, 4], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="Triton backend is unavailable"):
        segmented_attention(q, k, v, offsets)


def test_rotate_half_rope_on_cpu_fails_loudly():
    q = torch.randn(4, 2, 8)
    k = torch.randn(4, 1, 8)
    cos = torch.randn(4, 8)
    sin = torch.randn(4, 8)

    with pytest.raises(RuntimeError, match="Triton backend is unavailable"):
        rotate_half_rope(q, k, cos, sin)
