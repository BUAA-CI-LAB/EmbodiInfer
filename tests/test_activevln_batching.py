"""Real tensor batching, independent episode history and transactional failures."""

from types import SimpleNamespace

import pytest
import torch

from embodiinfer.exceptions import SessionCancelledError
from embodiinfer.policies.activevln.batching_activevln import ActiveVLNBatchedRuntime
from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPolicy
from embodiinfer.policies.activevln.processor_activevln import ActiveVLNProcessor, ProcessedTurn
from embodiinfer.types import Observation
from test_activevln import _Qwen


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "stop" if ids and ids[0] == 5 else "unknown"


class _Processor:
    tokenizer = _Tokenizer()
    collate = staticmethod(ActiveVLNProcessor.collate)

    def process_turn(self, observation, *, initial):
        del initial
        ids = torch.tensor([[int(x) for x in observation.instruction.split()]])
        return ProcessedTurn(
            ids,
            torch.ones_like(ids),
            torch.empty(0, 1),
            torch.empty(0, 3, dtype=torch.long),
            "",
            str(ids.tolist()),
        )


class _Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, pixels, *, grid_thw):
        del pixels, grid_thw
        self.calls += 1
        return self.weight.new_empty(0, 12)


def _policy():
    torch.manual_seed(19)
    qwen = _Qwen()
    qwen.config.vision_config = SimpleNamespace(spatial_merge_size=2)
    qwen.visual = _Visual()
    return ActiveVLNPolicy(qwen, _Processor(), do_sample=False, max_new_tokens=4, max_context=128).eval()


def _observation(text):
    return Observation(torch.zeros(1, 3, 2, 2), torch.zeros(0), torch.zeros(0, dtype=torch.long), text)


@pytest.mark.parametrize("batch_size", [2, 4])
@torch.inference_mode()
def test_ragged_batch_matches_independent_forward_and_private_caches(batch_size):
    policy = _policy()
    runtime = ActiveVLNBatchedRuntime(policy, batch_size=batch_size, workspace_tokens=64, query_bucket_size=8)
    memories = []
    for row in range(batch_size):
        obs = _observation(" ".join(["3"] * (row + 1)))
        prefix = policy.encode_prefix(policy.collate([obs], [str(row)]))
        memories.append(prefix.memory)
    snapshots = [memory.packed_kv.clone() for memory in memories]
    observations = [
        _observation(" ".join([str(6 + row)] * (batch_size - row + 1))) for row in range(batch_size)
    ]
    prepared = runtime.prepare(observations, memories)
    calls = []
    hook = policy._text.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape)
    )
    before = policy._visual.calls
    prefix = runtime.prefill(prepared)
    hook.remove()
    assert calls == [torch.Size([batch_size, 8, 12])]
    assert policy._visual.calls == before + 1
    generations = runtime.generate(prefix)
    for row, (observation, memory, generation) in enumerate(
        zip(observations, memories, generations, strict=True)
    ):
        reference_prefix = policy.encode_prefix(policy.collate([observation], [str(row)]), memory)
        torch.testing.assert_close(prefix.logits[row], reference_prefix.next_logits[0], rtol=1e-5, atol=1e-6)
        reference = policy.decoder.generate_tokens(reference_prefix)
        assert torch.equal(generation.token_ids, reference.token_ids)
        torch.testing.assert_close(
            generation.memory.packed_kv, reference.memory.packed_kv, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(memory.packed_kv, snapshots[row], rtol=0, atol=0)
        assert generation.memory._packed_kv.data_ptr() != runtime.storage.data_ptr()
    # A different ordering and a newly reset episode cannot inherit another row.
    next_memories = [generations[-1].memory, None]
    next_observations = [_observation("9"), _observation("8 7")]
    before = next_memories[0].packed_kv.clone()
    prefix2 = runtime.prefill(runtime.prepare(next_observations, next_memories))
    for row, (obs, mem) in enumerate(zip(next_observations, next_memories, strict=True)):
        reference = policy.encode_prefix(policy.collate([obs], [str(row)]), mem)
        torch.testing.assert_close(prefix2.logits[row], reference.next_logits[0], rtol=1e-5, atol=1e-6)
    runtime.generate(prefix2)
    torch.testing.assert_close(next_memories[0].packed_kv, before, rtol=0, atol=0)


@torch.inference_mode()
def test_rows_stop_independently_and_done_row_cache_remains_frozen():
    policy = _policy()

    class ScriptedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            sequences = ([5, 6, 7, 8], [1, 2, 9, 10], [1, 1, 2, 2], [1, 1, 1, 1])
            result = hidden.new_full((hidden.shape[0], 32), -10)
            for row, token in enumerate(sequences[min(self.calls, 3)]):
                result[row, token] = 10
            self.calls += 1
            return result

    policy.qwen.lm_head = ScriptedHead()
    runtime = policy.create_batched_runtime(batch_size=4, workspace_tokens=64, query_bucket_size=4)
    prefix = runtime.prefill(runtime.prepare([_observation("3 4")] * 4))
    generations = runtime.generate(prefix)
    assert [g.token_ids.tolist() for g in generations] == [[[5]], [[6, 2]], [[7, 9, 2]], [[8, 10, 2]]]
    assert [g.stop_reason for g in generations] == ["stop", "eos", "eos", "eos"]
    assert [g.memory.seq_len for g in generations] == [3, 4, 5, 5]
    assert [g.memory.next_position for g in generations] == [3, 4, 5, 5]
    assert policy.qwen.lm_head.calls == 4  # one prefill plus three batched forwards
    with pytest.raises(RuntimeError, match="already consumed"):
        runtime.generate(prefix)


@torch.inference_mode()
def test_cancelled_or_stale_batch_never_modifies_committed_memory():
    policy = _policy()
    obs = _observation("3 4")
    memory = policy.encode_prefix(policy.collate([obs], ["a"])).memory
    original = memory.packed_kv.clone()
    runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=64)
    prepared = runtime.prepare([obs, obs], [memory, memory])
    stale = runtime.prefill(prepared)
    current = runtime.prefill(prepared)
    other_runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=64)
    other = other_runtime.prefill(prepared)
    with pytest.raises(RuntimeError, match="stale"):
        other_runtime.generate(stale)
    other_runtime.generate(other)
    with pytest.raises(RuntimeError, match="stale"):
        runtime.generate(stale)
    with pytest.raises(SessionCancelledError):
        runtime.generate(current, cancelled=lambda: True)
    torch.testing.assert_close(memory.packed_kv, original, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="already consumed"):
        runtime.generate(current)
    runtime.generate(runtime.prefill(prepared))
    torch.testing.assert_close(memory.packed_kv, original, rtol=0, atol=0)


@torch.inference_mode()
def test_workspace_growth_preserves_history_and_context_overflow_is_explicit():
    policy = _policy()
    runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=8, query_bucket_size=1)
    obs = _observation(" ".join(["3"] * 12))
    generations = runtime.generate(runtime.prefill(runtime.prepare([obs, obs])))
    assert runtime.capacity_growths == 1
    assert all(g.memory.seq_len >= 13 for g in generations)
    policy.max_context = 13
    before = generations[0].memory.packed_kv.clone()
    with pytest.raises(ValueError, match="context limit"):
        runtime.prepare([obs], [generations[0].memory])
    torch.testing.assert_close(generations[0].memory.packed_kv, before, rtol=0, atol=0)


@torch.inference_mode()
def test_padded_query_and_stopped_row_at_context_boundary_preserve_valid_cache():
    policy = _policy()
    policy.max_context = 8
    old = policy.encode_prefix(policy.collate([_observation("3 3 3 3 3 3")], ["old"])).memory
    reference = policy.encode_prefix(policy.collate([_observation("4 5")], ["old"]), old).memory

    class ScriptedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            result = hidden.new_full((hidden.shape[0], 32), -10)
            result[0, 5] = 10
            result[1, 2 if self.calls >= 2 else 7] = 10
            self.calls += 1
            return result

    policy.qwen.lm_head = ScriptedHead()
    runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=8, query_bucket_size=8)
    # Row 0 has only one real new token, but padding would write 6+8 slots.
    prefix = runtime.prefill(runtime.prepare([_observation("4"), _observation("6 7 8")], [old, None]))
    generations = runtime.generate(prefix)
    assert [g.memory.seq_len for g in generations] == [8, 6]
    assert [g.stop_reason for g in generations] == ["stop", "eos"]
    torch.testing.assert_close(generations[0].memory.packed_kv, reference.packed_kv, rtol=1e-5, atol=1e-6)


def test_batched_runtime_rejects_training_and_sampling():
    policy = _policy()
    with pytest.raises(ValueError, match="inference_mode"):
        policy.create_batched_runtime(batch_size=2, workspace_tokens=64)
    with torch.inference_mode():
        policy.do_sample = True
        with pytest.raises(ValueError, match="greedy"):
            policy.create_batched_runtime(batch_size=2, workspace_tokens=64)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("pooled", [False, True])
@torch.inference_mode()
def test_batched_trees_match_serial_greedy_paths_logprobs_and_kv(batch_size, pooled):
    from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree

    policy = _policy()
    policy.max_new_tokens = 7
    policy.eos_token_ids = (99,)
    policy._processor.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "unknown")
    observations = [_observation(" ".join([str(row + 3)] * (row + 2))) for row in range(batch_size)]
    references = [
        policy.decoder.generate_tokens(policy.encode_prefix(policy.collate([obs], [str(row)])))
        for row, obs in enumerate(observations)
    ]
    paths = [result.token_ids[0, :4].tolist() for result in references]
    # Proposals intentionally end early; uncovered continuations still use the
    # full-vocabulary greedy choice and cannot be replaced by a proposed suffix.
    groups = {}
    for path in paths:
        groups.setdefault(path[0], []).append(path)
    trees = tuple(
        ActionTokenTree.from_paths(str(root), group + [[root, 31, 30]], "cpu")
        for root, group in groups.items()
    )
    runtime = policy.create_batched_runtime(
        batch_size=batch_size,
        workspace_tokens=64,
        kv_pool_tokens=128 * batch_size if pooled else None,
        query_bucket_size=1,
    )
    runtime.action_trees = trees
    calls = []
    hook = policy._text.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda module, args: calls.append(tuple(args[0].shape))
    )
    results = runtime.generate(runtime.prefill(runtime.prepare(observations)))
    hook.remove()
    assert all(shape[0] == batch_size for shape in calls)
    assert any(shape[1] > 1 for shape in calls[1:])
    assert runtime.counters["tree_accepted_tokens"] >= batch_size * 4
    assert runtime.counters["tree_fallback_tokens"] > 0
    for actual, expected in zip(results, references, strict=True):
        assert actual.stop_reason == expected.stop_reason
        torch.testing.assert_close(actual.token_ids, expected.token_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual.token_logprobs, expected.token_logprobs, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.memory.packed_kv, expected.memory.packed_kv, rtol=1e-5, atol=1e-6)
    snapshots = [result.memory.packed_kv.clone() for result in results]
    next_memories = [result.memory for result in results]
    prepared = runtime.prepare(observations, next_memories)
    hits = runtime.counters["memory_resident_hits"]
    prefix = runtime.prefill(prepared)
    assert runtime.counters["memory_resident_hits"] == hits + batch_size
    checks = 0

    def cancel_after_verification():
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(SessionCancelledError):
        runtime.generate(prefix, cancelled=cancel_after_verification)
    for memory, snapshot in zip(next_memories, snapshots, strict=True):
        torch.testing.assert_close(memory.packed_kv, snapshot, rtol=0, atol=0)


@torch.inference_mode()
def test_pooled_rows_relocate_and_reorder_without_changing_committed_histories():
    policy = _policy()
    policy.max_context = 4096
    runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=2048, kv_pool_tokens=4096)
    observations = [_observation("3 4"), _observation("6 7 8")]
    first = runtime.generate(runtime.prefill(runtime.prepare(observations)))
    memories = [first[1].memory, first[0].memory]
    snapshots = [m.packed_kv.clone() for m in memories]
    # Force a real segment-plan change without allocating large CPU attention.
    runtime._row_capacities = [0, 0]
    prepared = runtime.prepare(observations, memories)
    prefix = runtime.prefill(prepared)
    assert runtime.counters["pool_relocations"] == 2
    results = runtime.generate(prefix)
    for row, (obs, memory) in enumerate(zip(observations, memories, strict=True)):
        reference = policy.decoder.generate_tokens(
            policy.encode_prefix(policy.collate([obs], [str(row)]), memory)
        )
        torch.testing.assert_close(results[row].token_ids, reference.token_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            results[row].memory.packed_kv, reference.memory.packed_kv, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(memory.packed_kv, snapshots[row], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [1, 2, 4])
@torch.inference_mode()
def test_batch_graphs_share_shape_buffers_and_replay_mixed_tree_layouts(batch):
    from test_activevln import _Text

    policy = _policy()
    policy.qwen.model = _Text(128)
    policy.qwen.lm_head = torch.nn.Linear(128, 32, bias=False)
    policy.qwen.config.text_config = SimpleNamespace(
        num_key_value_heads=1, rope_scaling={"mrope_section": [8, 12, 12]}, rope_theta=1_000_000.0
    )
    policy.max_context = 1024
    policy.to(device="cuda", dtype=torch.bfloat16)
    runtime = policy.create_batched_runtime(
        batch_size=batch,
        workspace_tokens=1024,
        kv_pool_tokens=1024 * batch,
        cuda_graph=True,
        fused_ops=True,
        split_attention=True,
        query_bucket_size=1,
    )
    for tree in (False, True):
        graphs = runtime.tree_graphs if tree else runtime.text_graphs
        for query in (3, 5):
            for key in (512, 1024):
                graphs[(query, key)] = runtime._capture_batch(query, key, tree=tree)
            assert graphs[(query, 512)].hidden.data_ptr() == graphs[(query, 1024)].hidden.data_ptr()
            assert graphs[(query, 512)].output.data_ptr() == graphs[(query, 1024)].output.data_ptr()
    assert len(runtime._graph_buffers) == 4
    runtime._plan_rows([510] * batch, 5, batch)
    for query, past, tree in ((5, 509, True), (3, 103, False), (5, 61, True), (3, 511, False)):
        hidden = torch.randn(batch, query, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(query, device="cuda")[None, None].expand(3, batch, -1) + past
        offsets = [past - row * 7 for row in range(batch)]
        ancestors = None
        if tree:
            indices = torch.arange(query, device="cuda")
            ancestors = torch.stack(
                [
                    (indices[:, None] >= indices[None])
                    & ((indices[:, None] + row) % 2 == (indices[None] + row) % 2)
                    for row in range(batch)
                ]
            )
        snapshot = runtime.storage.clone()
        actual = runtime._forward(hidden, positions, offsets, ancestors=ancestors)
        runtime.storage.copy_(snapshot)
        runtime.use_cuda_graph = False
        expected = runtime._forward(hidden, positions, offsets, ancestors=ancestors)
        runtime.use_cuda_graph = True
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Re-prewarming must discard episode segment metadata before dummy reads.
    runtime.vision = lambda turn: torch.empty(0, 128, device="cuda")
    runtime.prewarm([_observation("3 4 5")], context_buckets=[512, 1024])
    assert runtime._row_capacities == [0] * batch
    assert runtime._cache_layout[0].tolist() == [0] * batch
    assert runtime._cache_layout[1].tolist() == [runtime.storage.shape[-2]] * batch
    runtime._plan_rows([510] * batch, 5, batch)
    assert runtime._row_capacities == [1024] * batch


@torch.inference_mode()
def test_mixed_tree_rows_preserve_eos_stop_fallback_and_per_row_budget():
    from types import MethodType

    from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree

    policy = _policy()
    scripts = ([3, 4, 2], [5], [9, 10, 2], [6, 7, 8, 9, 10, 2])

    class PositionHead(torch.nn.Module):
        def forward(self, hidden):
            scores = hidden.new_full((*hidden.shape[:-1], 32), -10)
            for row, script in enumerate(scripts):
                index = (hidden[row, ..., 0].long() + 1).clamp(0, len(script) - 1)
                choice = torch.tensor(script)[index]
                scores[row].scatter_(-1, choice[..., None], 10)
            return scores

    policy.qwen.lm_head = PositionHead()
    runtime = policy.create_batched_runtime(batch_size=4, workspace_tokens=64, query_bucket_size=1)
    runtime.action_trees = (
        ActionTokenTree.from_paths("left", [[3, 4, 2], [3, 11, 12]], "cpu"),
        ActionTokenTree.from_paths("right", [[6, 7, 8, 9, 10, 2]], "cpu"),
    )
    shapes = []

    def forward(self, hidden, positions, past, *, ancestors=None, write_lengths=None):
        del ancestors
        shapes.append(tuple(hidden.shape))
        result = torch.zeros_like(hidden)
        result[..., 0] = positions[0] - 2
        for row, count in enumerate(write_lengths):
            if count:
                self._row_kv(row, past[row], past[row] + count).copy_(positions[0, row, :count, None])
        return result

    runtime._forward = MethodType(forward, runtime)
    results = runtime.generate(runtime.prefill(runtime.prepare([_observation("1 1")] * 4)))
    assert [g.token_ids.tolist() for g in results] == [[[3, 4, 2]], [[5]], [[9, 10, 2]], [[6, 7, 8, 9]]]
    assert [g.stop_reason for g in results] == ["eos", "stop", "eos", "max_tokens"]
    assert [s[:2] for s in shapes] == [(4, 2), (4, 6), (4, 1), (4, 1)]
    for result in results:
        expected = torch.arange(result.memory.seq_len)[:, None].expand(result.memory.seq_len, 6)
        torch.testing.assert_close(result.memory.packed_kv[0, 0, 0, 0], expected.float(), rtol=0, atol=0)


@torch.inference_mode()
def test_batched_tree_penalties_match_independent_prefix_and_ancestor_histories():
    from embodiinfer.policies.activevln.speculation_activevln import (
        ActionTokenTree,
        batched_tree_greedy_scores,
        tree_greedy_scores,
    )

    policy = _policy()
    histories = [
        policy.encode_prefix(policy.collate([_observation(text)], [str(row)])).memory
        for row, text in enumerate(("1 3 3", "2 4"))
    ]
    trees = [
        ActionTokenTree.from_paths("a", [[3, 4, 3], [3, 7], [8, 4]], "cpu"),
        ActionTokenTree.from_paths("b", [[4, 8, 4], [4, 3]], "cpu"),
    ]
    query = max(t.token_ids.shape[1] for t in trees)
    logits = torch.randn(2, query, 32, generator=torch.Generator().manual_seed(62))
    paths = torch.zeros(2, query, 3, dtype=torch.long)
    valid = torch.zeros_like(paths, dtype=torch.bool)
    seen = torch.zeros(2, 32, dtype=torch.bool)
    for row, (tree, memory) in enumerate(zip(trees, histories, strict=True)):
        size, depth = tree.path_ids.shape
        paths[row, :size, :depth], valid[row, :size, :depth] = tree.path_ids, tree.path_valid
        seen[row].scatter_(0, memory.token_ids[0], True)
    selected, scores = batched_tree_greedy_scores(logits, seen, paths, valid, torch.tensor([1, 2]), 1.05)
    for row, (tree, memory) in enumerate(zip(trees, histories, strict=True)):
        size = tree.token_ids.shape[1]
        expected_ids, expected_scores = tree_greedy_scores(logits[row, :size], memory, tree, 1.05)
        torch.testing.assert_close(selected[row, :size], expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(scores[row, :size], expected_scores, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("batch,query", [(1, 289), (2, 144), (4, 173)])
@pytest.mark.parametrize("tree", [False, True])
@torch.inference_mode()
def test_single_partition_context_graph_alias_preserves_exact_forward_and_kv(batch, query, tree):
    from test_activevln import _Attention, _Text

    policy = _policy()
    policy.qwen.model = _Text(256)
    for layer in policy._text.layers:
        layer.self_attn = _Attention(256, heads=16, kv_heads=2)
    policy.qwen.lm_head = torch.nn.Linear(256, 32, bias=False)
    policy.qwen.config.text_config = SimpleNamespace(
        num_key_value_heads=2, rope_scaling={"mrope_section": [2, 3, 3]}, rope_theta=1_000_000.0
    )
    policy.max_context = 1024
    policy.to(device="cuda", dtype=torch.bfloat16)
    runtime = policy.create_batched_runtime(
        batch_size=batch,
        workspace_tokens=1024,
        kv_pool_tokens=batch * 1024,
        cuda_graph=True,
        fused_ops=True,
        split_attention=True,
        query_bucket_size=1,
    )
    runtime._capture_contexts(query, [512, 1024], tree=tree)
    graphs = runtime.tree_graphs if tree else runtime.text_graphs
    assert graphs[(query, 512)].graph is graphs[(query, 1024)].graph
    runtime._capture_contexts(1, [512, 1024])
    assert runtime.text_graphs[(1, 512)].graph is not runtime.text_graphs[(1, 1024)].graph
    runtime._plan_rows([600] * batch, query, batch)
    for past in (50, 600):
        runtime.storage.normal_()
        hidden = torch.randn(batch, query, 256, device="cuda", dtype=torch.bfloat16)
        offsets = [past - row * 3 for row in range(batch)]
        positions = torch.arange(query, device="cuda")[None, None].expand(3, batch, -1)
        positions = positions + torch.tensor(offsets, device="cuda")[None, :, None]
        ancestors = None
        if tree:
            indices = torch.arange(query, device="cuda")
            ancestors = torch.stack(
                [
                    (indices[:, None] >= indices[None])
                    & ((indices[:, None] + row) % 2 == (indices[None] + row) % 2)
                    for row in range(batch)
                ]
            )
        original = runtime.storage.clone()
        actual = runtime._forward(hidden, positions, offsets, ancestors=ancestors)
        actual_kv = runtime.storage.clone()
        runtime.storage.copy_(original)
        runtime.use_cuda_graph = False
        expected = runtime._forward(hidden, positions, offsets, ancestors=ancestors)
        runtime.use_cuda_graph = True
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_kv, runtime.storage, rtol=0, atol=0)
