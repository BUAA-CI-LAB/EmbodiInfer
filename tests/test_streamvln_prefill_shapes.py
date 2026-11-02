"""Prompt-only graph census must cover mixed phases and partial batches."""

import itertools
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodiinfer.backend.triton.gqa_split_attention import PACKED_BLOCK_M
from embodiinfer.policies.streamvln.prompt import VISION_PHRASE


def _plan(**overrides):
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks/streamvln-benchmark/prefill_shapes.py")
    )
    processor = SimpleNamespace(
        tokenizer=lambda text, **kwargs: {"input_ids": [1] * len(text.split())},
        vision_phrase=VISION_PHRASE,
        num_history=8,
    )
    options = dict(
        vision_tokens=4,
        batch_size=3,
        token_bucket=32,
        group_size=4,
        workspace_tokens=1024,
        kv_pool_tokens=3072,
        max_new_tokens=16,
    )
    options.update(overrides)
    return module["plan_prefill_shapes"](processor, ["go left", "go through the door then stop"], **options)


def test_prompt_census_covers_every_mixed_phase_and_tail_tensor_shape():
    plan = _plan()
    lengths = set(itertools.chain.from_iterable(plan["phase_lengths"].values()))
    expected = set()
    per_block = PACKED_BLOCK_M // 4
    for count in (1, 2, 3):
        for rows in itertools.product(lengths, repeat=count):
            physical = -(-sum(rows) // 32) * 32
            expected.add((count, physical, -(-physical // per_block) + count - 1))
    assert set(map(tuple, plan["shape_keys"])) == expected
    assert len(plan["shapes"]) == plan["graph_shapes"] == len(expected)
    for key, rows in zip(plan["shape_keys"], plan["shapes"], strict=True):
        assert len(rows) == key[0] and -(-sum(rows) // 32) * 32 == key[1]
        assert set(rows) <= lengths
    assert len(plan["phase_lengths"]["ordinary"]) == 1
    assert min(plan["phase_lengths"]["slow_window"]) > max(plan["phase_lengths"]["initial"])


def test_b1_exact_census_retains_each_distinct_projected_length():
    plan = _plan(batch_size=1, token_bucket=1)
    expected = set(itertools.chain.from_iterable(plan["phase_lengths"].values()))
    assert {row[0] for row in plan["shapes"]} == expected


@pytest.mark.parametrize(
    "options,message",
    [
        ({"kv_pool_tokens": 200}, "shared KV budget"),
        ({"workspace_tokens": 50}, "per-row workspace"),
        ({"max_graphs": 1}, "startup budget"),
        ({"token_bucket": 1}, "multi-row shape census"),
    ],
)
def test_census_rejects_budgets_that_cannot_cover_the_declared_inputs(options, message):
    with pytest.raises(ValueError, match=message):
        _plan(**options)
