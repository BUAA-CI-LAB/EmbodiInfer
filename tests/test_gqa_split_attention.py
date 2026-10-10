"""GPU contracts of the grouped-query split-KV attention kernel."""

import itertools

import pytest
import torch

from embodiinfer.backend.triton.gqa_split_attention import (
    PACKED_BLOCK_M,
    gqa_split_kv_attention,
    packed_query_blocks,
)
from embodiinfer.backend.triton.split_attention import split_kv_attention

HEADS, KV_HEADS, WIDTH, CAPACITY = 16, 2, 128, 4096


def _tree(batch: int, nodes: int, generator: torch.Generator) -> torch.Tensor:
    mask = torch.zeros(batch, nodes, nodes, dtype=torch.bool)
    rows = torch.arange(batch)
    for node in range(nodes):
        mask[:, node, node] = True
        if node:
            mask[:, node] |= mask[rows, torch.randint(0, node, (batch,), generator=generator)]
    return mask


def _inputs(batch: int, length: int, key_bucket: int, tree: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, length, HEADS, WIDTH, generator=generator).bfloat16().cuda().transpose(1, 2)
    key, value = (
        torch.randn(1, KV_HEADS, batch * CAPACITY, WIDTH, generator=generator).bfloat16().cuda()
        for _ in range(2)
    )
    position = torch.randint(0, key_bucket - length + 1, (batch,), generator=generator).cuda()
    layout = {
        "ancestors": _tree(batch, length, generator).cuda() if tree else None,
        "cache_starts": (torch.arange(batch) * CAPACITY).cuda(),
        "cache_capacities": torch.full((batch,), CAPACITY).cuda(),
    }
    lengths = torch.randint(1, length + 1, (batch,), generator=generator)
    lengths[0] = length
    return query, key, value, position, layout, lengths.cuda()


def _poison_unwritten_slots(key, value, position, layout, lengths) -> None:
    rows = zip(layout["cache_starts"].tolist(), position.tolist(), lengths.tolist(), strict=True)
    for start, past, count in rows:
        key[:, :, start + past + count : start + CAPACITY] = float("nan")
        value[:, :, start + past + count : start + CAPACITY] = float("nan")


def test_gqa_attention_rejects_cpu_inputs():
    query = torch.zeros(1, HEADS, 1, WIDTH, dtype=torch.bfloat16)
    cache = torch.zeros(1, KV_HEADS, 8, WIDTH, dtype=torch.bfloat16)
    with pytest.raises((RuntimeError, ValueError), match="split KV attention requires"):
        gqa_split_kv_attention(query, cache, cache, torch.zeros(1, dtype=torch.long), 8)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("batch", "length", "key_bucket", "tree"),
    [(4, 300, 2048, False), (4, 33, 4096, True), (8, 1, 1024, False)],
)
@torch.inference_mode()
def test_gqa_attention_matches_per_head_split_kv(batch, length, key_bucket, tree):
    query, key, value, position, layout, _ = _inputs(batch, length, key_bucket, tree, seed=length)
    expected = split_kv_attention(query, key, value, position, key_bucket, **layout)
    actual = gqa_split_kv_attention(query, key, value, position, key_bucket, **layout)
    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("batch", "length", "key_bucket", "tree"),
    [
        (4, 700, 2048, False),
        (8, 400, 1024, False),
        (4, 150, 8192, False),
        (4, 20, 4096, False),
        (4, 33, 8192, True),
        (8, 33, 2048, True),
        (4, 1, 4096, False),
    ],
)
@torch.inference_mode()
def test_query_lengths_skip_padding_without_changing_live_rows(batch, length, key_bucket, tree):
    query, key, value, position, layout, lengths = _inputs(batch, length, key_bucket, tree, seed=length + 1)
    full = gqa_split_kv_attention(query, key, value, position, key_bucket, **layout)
    skipped = gqa_split_kv_attention(query, key, value, position, key_bucket, query_lengths=lengths, **layout)
    for row, count in enumerate(lengths.tolist()):
        assert torch.equal(skipped[row, :, :count], full[row, :, :count])
        assert not skipped[row, :, count:].any()


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("batch", "length", "key_bucket", "tree"), [(8, 400, 1024, False), (4, 33, 4096, True)]
)
@torch.inference_mode()
def test_query_lengths_never_read_slots_past_live_queries(batch, length, key_bucket, tree):
    query, key, value, position, layout, lengths = _inputs(batch, length, key_bucket, tree, seed=length + 3)
    expected = gqa_split_kv_attention(
        query, key, value, position, key_bucket, query_lengths=lengths, **layout
    )
    _poison_unwritten_slots(key, value, position, layout, lengths)
    actual = gqa_split_kv_attention(query, key, value, position, key_bucket, query_lengths=lengths, **layout)
    assert torch.equal(actual, expected)


def test_packed_program_table_lists_each_rows_query_blocks():
    per_block = PACKED_BLOCK_M // 8
    blocks = packed_query_blocks([per_block + 1, 0, per_block], 8, capacity=6)
    assert blocks == [0, 1, 2 * 65536, -1, -1, -1]
    with pytest.raises(ValueError, match="more programs"):
        packed_query_blocks([per_block + 1, 0, per_block], 8, capacity=2)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("batch", "length", "key_bucket"), [(4, 300, 2048), (3, 150, 1024), (5, 70, 512), (2, 373, 4096)]
)
@torch.inference_mode()
def test_packed_queries_match_padded_live_rows_bitwise(batch, length, key_bucket):
    query, key, value, position, layout, lengths = _inputs(batch, length, key_bucket, False, seed=length + 2)
    lengths[-1] = 0
    _poison_unwritten_slots(key, value, position, layout, lengths)
    padded = gqa_split_kv_attention(query, key, value, position, key_bucket, query_lengths=lengths, **layout)
    sizes = lengths.tolist()
    starts = [0, *itertools.accumulate(sizes)][:-1]
    tokens = sum(sizes) + 5
    packed_query = query.new_zeros(1, HEADS, tokens, WIDTH)
    for row, (start, size) in enumerate(zip(starts, sizes, strict=True)):
        packed_query[0, :, start : start + size] = query[row, :, :size]
    capacity = -(-tokens // (PACKED_BLOCK_M // (HEADS // KV_HEADS))) + batch
    packing = (
        torch.tensor(packed_query_blocks(sizes, HEADS // KV_HEADS, capacity)).cuda(),
        torch.tensor(starts).cuda(),
        torch.tensor([length]).cuda(),
    )
    packed = gqa_split_kv_attention(
        packed_query, key, value, position, key_bucket, query_lengths=lengths, packed=packing, **layout
    )
    for row, (start, size) in enumerate(zip(starts, sizes, strict=True)):
        assert torch.equal(packed[0, :, start : start + size], padded[row, :, :size])
