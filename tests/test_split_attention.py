"""Partitioned softmax and dynamic causal/tree visibility on CUDA."""

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.backend.triton.split_attention import split_kv_attention, write_batched_kv


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


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [2, 4])
@torch.inference_mode()
def test_batched_split_attention_graph_uses_independent_row_offsets(batch):
    torch.manual_seed(63)
    q = torch.randn(batch, 16, 17, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    offsets = torch.arange(batch, device="cuda") * 113
    for _ in range(3):
        split_kv_attention(q, k, v, offsets, 1024)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = split_kv_attention(q, k, v, offsets, 1024)
    for shift in (0, 219):
        offsets.copy_(torch.arange(batch, device="cuda") * 113 + shift)
        graph.replay()
        allowed = (
            torch.arange(1024, device="cuda")[None, None, :]
            <= offsets[:, None, None] + torch.arange(17, device="cuda")[None, :, None]
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], enable_gqa=True)
        torch.testing.assert_close(actual, reference, rtol=1 / 128, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [2, 4])
@torch.inference_mode()
def test_batched_kv_write_graph_ignores_padding_beyond_capacity(batch):
    torch.manual_seed(71)
    key = torch.randn(batch, 17, 2, 32, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    value = torch.randn_like(key)
    # The caches are views into the same packed layer storage used by the runtime.
    storage = torch.randn(2, batch, 2, 64, 32, device="cuda", dtype=torch.bfloat16)
    offsets = torch.zeros(batch, device="cuda", dtype=torch.long)
    write_batched_kv(key, value, storage[0], storage[1], offsets)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write_batched_kv(key, value, storage[0], storage[1], offsets)
    for shift in (0, 51):
        offsets.copy_(torch.arange(batch, device="cuda") * 3 + shift)
        expected = storage.clone()
        for row, start in enumerate(offsets.tolist()):
            count = min(key.shape[2], storage.shape[-2] - start)
            expected[0, row, :, start : start + count] = key[row, :, :count]
            expected[1, row, :, start : start + count] = value[row, :, :count]
        graph.replay()
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [2, 4])
@torch.inference_mode()
def test_shared_pool_graph_relocates_rows_and_uses_distinct_ancestor_masks(batch):
    torch.manual_seed(73)
    query = 17
    q = torch.randn(batch, 16, query, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512 * batch, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    starts = torch.arange(batch, device="cuda") * 512
    capacities = torch.full((batch,), 512, device="cuda", dtype=torch.long)
    offsets = torch.arange(batch, device="cuda") * 53 + 7
    masks = torch.eye(query, device="cuda", dtype=torch.bool)[None].repeat(batch, 1, 1)

    def callback():
        return split_kv_attention(
            q, k, v, offsets, 512, ancestors=masks, cache_starts=starts, cache_capacities=capacities
        )

    callback()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = callback()
    for reverse in (False, True):
        if reverse:
            starts.copy_(starts.flip(0))
            offsets.add_(97)
        index = torch.arange(query, device="cuda")
        for row in range(batch):
            masks[row] = (index[:, None] >= index[None]) & (
                (index[:, None] % (row + 2)) == (index[None] % (row + 2))
            )
        graph.replay()
        for row, (start, past) in enumerate(zip(starts.tolist(), offsets.tolist(), strict=True)):
            local = torch.arange(512, device="cuda") - past
            allowed = (local[None] < 0) | (
                masks[row, :, local.clamp(0, query - 1)] & ((local >= 0) & (local < query))[None]
            )
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(
                    q[row : row + 1],
                    k[:, :, start : start + 512],
                    v[:, :, start : start + 512],
                    attn_mask=allowed,
                    enable_gqa=True,
                )
            torch.testing.assert_close(actual[row : row + 1], expected, rtol=1 / 128, atol=2e-6)


@pytest.mark.gpu
@torch.inference_mode()
def test_shared_pool_kv_writes_respect_row_capacity_and_inactive_rows():
    key = torch.randn(4, 2, 17, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    storage = torch.randn(2, 1, 2, 256, 32, device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0, 64, 128, 192], device="cuda")
    capacities = torch.full((4,), 64, device="cuda", dtype=torch.long)
    offsets = torch.tensor([60, 55, 31, 0], device="cuda")
    counts = torch.tensor([3, 17, 0, 1], device="cuda")
    expected = storage.clone()
    for row, (start, past, count) in enumerate(
        zip(starts.tolist(), offsets.tolist(), counts.tolist(), strict=True)
    ):
        count = min(count, 64 - past)
        expected[0, 0, :, start + past : start + past + count] = key[row, :, :count]
        expected[1, 0, :, start + past : start + past + count] = value[row, :, :count]
    write_batched_kv(
        key,
        value,
        storage[0],
        storage[1],
        offsets,
        cache_starts=starts,
        cache_capacities=capacities,
        write_lengths=counts,
    )
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
