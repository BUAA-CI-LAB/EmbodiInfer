"""Partitioned softmax and dynamic causal/tree visibility on CUDA."""

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.backend.triton.split_attention import split_kv_attention


@pytest.mark.gpu
@pytest.mark.parametrize("length,tree", [(1, False), (33, False), (66, True)])
@torch.inference_mode()
def test_split_attention_dynamic_graph_matches_sdpa_math(length, tree):
    torch.manual_seed(42)
    q = torch.randn(1, length, 16, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(1, 2, 4096, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    position = torch.tensor([123], device="cuda")
    ancestors = None
    if tree:
        indices = torch.arange(length, device="cuda")
        ancestors = (indices[:, None] >= indices[None, :]) & (indices[:, None] % 3 == indices[None, :] % 3)

    def callback():
        return split_kv_attention(q, k, v, position, 2048, ancestors=ancestors)

    for _ in range(3):
        callback()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = callback()
    for past in (123, 1891, 0, 257):
        position.fill_(past)
        graph.replay()
        cols = torch.arange(2048, device="cuda")
        rows = torch.arange(length, device="cuda")
        allowed = cols[None] <= past + rows[:, None]
        if tree:
            local = cols - past
            allowed = (cols[None] < past) | (
                ancestors[:, local.clamp(0, length - 1)] & (local[None] >= 0) & (local[None] < length)
            )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(
                q, k[:, :, :2048], v[:, :, :2048], attn_mask=allowed, enable_gqa=True
            )
        # Three BF16 components approximate FP32 probabilities; partitioned
        # reductions differ from SDPA math. One BF16 relative step plus 2e-6
        # near zero is the operator contract, not the model admission criterion.
        torch.testing.assert_close(actual, reference, rtol=1 / 128, atol=2e-6)
        before = actual.clone()
        # Entirely invisible suffix may contain unrelated session data.
        k[:, :, past + length :].mul_(2)
        v[:, :, past + length :].add_(7)
        graph.replay()
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
