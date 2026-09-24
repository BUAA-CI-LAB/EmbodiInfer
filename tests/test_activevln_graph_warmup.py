"""CPU coverage of startup shape planning and disposable-workspace ownership."""

from threading import RLock
from types import SimpleNamespace

import pytest
import torch

from embodiinfer.policies.activevln.cuda_graph import ActiveVLNGraphRuntime, plan_graph_shapes
from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree, build_action_trees


def test_shape_plan_separates_empty_history_from_recurring_and_decode_queries():
    keys = [512, 1024, 4096, 16384, 20000]
    actual = set(plan_graph_shapes([337, 337], [145, 160], keys, query_bucket_size=32, max_context=20000))
    expected = {(352, 512)} | {(q, key) for q in (1, 160) for key in keys}
    assert actual == expected
    exact = plan_graph_shapes([337], [145], [512, 1024], query_bucket_size=1, max_context=20000)
    assert (337, 512) in exact and (145, 1024) in exact
    assert (337, 1024) not in exact


@pytest.mark.parametrize("buckets", [[True], [511], [513], [32768], [512.0], ["512"]])
def test_shape_plan_rejects_buckets_that_cannot_match_runtime(buckets):
    with pytest.raises(ValueError, match="context buckets"):
        plan_graph_shapes([100], [50], buckets, query_bucket_size=32, max_context=20000)


@pytest.mark.parametrize("query", [0, -1, True, 100.5])
def test_shape_plan_rejects_invalid_input_lengths(query):
    with pytest.raises(ValueError, match="query lengths"):
        plan_graph_shapes([query], [], [512], query_bucket_size=32, max_context=1024)


def _runtime():
    runtime = ActiveVLNGraphRuntime.__new__(ActiveVLNGraphRuntime)
    runtime.policy = SimpleNamespace(
        max_context=1024,
        parameters=lambda: iter([torch.zeros(1)]),
        _text=SimpleNamespace(embed_tokens=torch.nn.Embedding(12, 4)),
        _lm_head=lambda hidden: hidden,
    )
    runtime.device = torch.device("cpu")
    runtime.lock = RLock()
    runtime.query_bucket_size = 32
    runtime.tree_fp32_projection = False
    runtime.capture_enabled = True
    runtime._validate = lambda: None  # Exercise lifecycle, not CUDA capture, on CPU.
    runtime.storage = torch.full((1, 2, 1, 1, 1024, 1), 9.0)
    runtime.action_trees = (ActionTokenTree.from_paths("test", [[1, 2], [1, 3]], "cpu"),)
    runtime.text_graphs, runtime.tree_graphs = {}, {}
    runtime._resident_memory = SimpleNamespace(cache=torch.tensor([17.0]))
    runtime._resident_length = 1
    return runtime


@torch.inference_mode()
def test_prewarm_captures_missing_shapes_and_invalidates_only_disposable_workspace():
    runtime = _runtime()
    committed = runtime._resident_memory
    calls = []

    def capture(hidden, positions, length, query, key, tree=None):
        calls.append((query, key, tree is not None))
        assert length == key - query
        assert positions.shape == (3, 1, query)
        return SimpleNamespace(output=hidden)

    runtime._capture_text = capture
    runtime.text_graphs[(160, 512)] = "existing capture"
    plan = runtime.prewarm_text([337], [145], [512, 1024])
    assert runtime.text_graphs[(160, 512)] == "existing capture"
    assert set(runtime.text_graphs) == set(plan)
    assert set(runtime.tree_graphs) == {("test", 512), ("test", 1024)}
    assert calls.count((160, 512, False)) == 0
    assert len(calls) == len(plan) - 1 + 2
    assert runtime._resident_memory is None and runtime._resident_length == 0
    assert committed.cache.item() == 17
    assert torch.count_nonzero(runtime.storage) == 0


@torch.inference_mode()
def test_prewarm_failure_cannot_leave_a_stale_workspace_binding():
    runtime = _runtime()
    committed = runtime._resident_memory

    def capture(*args, **kwargs):
        raise RuntimeError("injected capture failure")

    runtime._capture_text = capture
    with pytest.raises(RuntimeError, match="injected capture failure"):
        runtime.prewarm_text([337], [145], [512])
    assert runtime._resident_memory is None and runtime._resident_length == 0
    assert committed.cache.item() == 17


def test_prewarm_is_rejected_after_startup_without_touching_workspace():
    runtime = _runtime()
    committed = runtime._resident_memory
    runtime.capture_enabled = False
    with pytest.raises(RuntimeError, match="only allowed during startup"):
        runtime.prewarm_text([337], [145], [512])
    assert runtime._resident_memory is committed
    assert torch.all(runtime.storage == 9)


def test_repeated_phrase_candidates_keep_early_eos_and_both_prompt_boundaries():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return list(map(ord, text))

    tokenizer = Tokenizer()
    eos = 1000
    trees = build_action_trees(tokenizer, [eos], torch.device("cpu"), repeat_actions=3)
    for tree, space in zip(trees, ("", " "), strict=True):
        for phrase in ("move forward 25cm", "turn left 30 degrees", "turn right 45 degrees"):
            for count in (1, 2, 3):
                text = space + ", ".join([phrase] * count)
                for suffix in ([eos], [ord(",")]):
                    parent = -1
                    for token in list(map(ord, text)) + suffix:
                        parent = tree.children[(parent, token)]
                    route = tree.path_ids[parent, tree.path_valid[parent]].tolist()
                    assert route == list(map(ord, text)) + suffix


@pytest.mark.parametrize("count", [0, 4, True, 1.5])
def test_repeated_phrase_candidates_reject_unbounded_or_invalid_work(count):
    with pytest.raises(ValueError, match="integer from 1 to 3"):
        build_action_trees(None, [2], torch.device("cpu"), repeat_actions=count)


def test_root_partition_preserves_all_proposed_paths_and_removes_rejected_roots():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            # Keep a distinct continuation boundary while exposing phrase roots.
            offset = 100 if text.startswith(" ") else 0
            return [ord(char) + offset for char in text.strip()]

    tokenizer = Tokenizer()
    whole = build_action_trees(tokenizer, [1000], torch.device("cpu"), repeat_actions=3)
    parts = build_action_trees(tokenizer, [1000], torch.device("cpu"), repeat_actions=3, partition_roots=True)

    def paths(trees):
        return {
            tuple(tree.path_ids[row, tree.path_valid[row]].tolist())
            for tree in trees
            for row in range(tree.token_ids.shape[1])
        }

    assert paths(parts) == paths(whole)
    assert len({tree.name for tree in parts}) == len(parts)
    assert len(parts) > len(whole)
    assert all(sum(parent == -1 for parent, _ in tree.children) == 1 for tree in parts)
    assert max(tree.token_ids.shape[1] for tree in parts) < max(tree.token_ids.shape[1] for tree in whole)


@pytest.mark.parametrize("query", [1, 8])
@torch.inference_mode()
def test_graph_workspace_overflow_falls_back_without_changing_public_history(query):
    runtime = _runtime()
    runtime.policy.max_context = 2048  # Public memory is larger than graph scratch.
    runtime.reset_stats()
    runtime.capture_enabled = False
    memory = SimpleNamespace(length=1024 if query == 1 else 1000)
    original_length = memory.length
    result = runtime.forward(torch.zeros(1, query, 4), torch.zeros(3, 1, query, dtype=torch.long), memory)
    assert result is None
    assert memory.length == original_length
    assert runtime.policy.max_context == 2048
    assert runtime._resident_memory is None
    assert torch.all(runtime.storage == 9)
    stage = "decode" if query == 1 else "prefill"
    assert runtime.counters[f"{stage}_fallbacks"] == 1


def test_shape_prewarm_uses_physical_workspace_limit_before_touching_storage():
    runtime = _runtime()
    runtime.policy.max_context = 2048
    resident = runtime._resident_memory
    with pytest.raises(ValueError, match="context buckets"):
        runtime.prewarm_text([337], [145], [2048])
    assert runtime._resident_memory is resident
    assert torch.all(runtime.storage == 9)


@pytest.mark.parametrize("capacity", [0, -1, True, 2049])
def test_invalid_workspace_capacity_is_rejected_before_device_allocation(capacity):
    with pytest.raises(ValueError, match="workspace_tokens"):
        ActiveVLNGraphRuntime(SimpleNamespace(max_context=2048), workspace_tokens=capacity)
