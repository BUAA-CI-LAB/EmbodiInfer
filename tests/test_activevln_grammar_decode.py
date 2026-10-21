"""Batched grammar-tree decoding: scripted outputs, linear KV and exact host helpers."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from embodiinfer.policies.activevln import grammar_tree_activevln as grammar
from embodiinfer.policies.activevln.batching_activevln import ActiveVLNBatchedRuntime
from test_activevln_batching import _observation, _policy

_EOS, _UNKNOWN = 2, 26
_PIECES = {
    "turn": 3,
    " turn": 4,
    "move": 5,
    " move": 6,
    "stop": 7,
    " stop": 8,
    " left": 9,
    " right": 10,
    " forward": 11,
    " ": 12,
    ",": 13,
    " degrees": 14,
    "cm": 15,
    **{str(digit): 16 + digit for digit in range(10)},
}


class _GrammarTokenizer:
    """Greedy longest match over the action grammar's pieces."""

    text = {token: piece for piece, token in _PIECES.items()}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids, start = [], 0
        while start < len(text):
            piece = max((p for p in _PIECES if text.startswith(p, start)), key=len, default=None)
            ids.append(_UNKNOWN if piece is None else _PIECES[piece])
            start += 1 if piece is None else len(piece)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join("" if token == _EOS else self.text.get(token, "?") for token in map(int, ids))


def _scripted_runtime(scripts, pooled):
    policy = _policy()
    policy._processor.tokenizer = _GrammarTokenizer()
    policy.max_new_tokens = 16

    class PositionHead(torch.nn.Module):
        def forward(self, hidden):
            scores = hidden.new_full((*hidden.shape[:-1], 32), -10.0)
            for row, script in enumerate(scripts):
                index = (hidden[row, ..., 0].long() + 1).clamp(0, len(script) - 1)
                scores[row].scatter_(-1, torch.tensor(script)[index][..., None], 10.0)
            return scores

    def forward(self, hidden, positions, past, *, ancestors=None, write_lengths=None):
        del ancestors
        result = torch.zeros_like(hidden)
        result[..., 0] = positions[0] - 2
        for row, count in enumerate(write_lengths):
            if count:
                self._row_kv(row, past[row], past[row] + count).copy_(positions[0, row, :count, None])
        return result

    policy.qwen.lm_head = PositionHead()
    runtime = policy.create_batched_runtime(
        batch_size=len(scripts),
        workspace_tokens=64,
        query_bucket_size=1,
        grammar_tree_budget=8,
        kv_pool_tokens=128 * len(scripts) if pooled else None,
    )
    runtime._forward = MethodType(forward, runtime)
    return policy, runtime


@pytest.mark.parametrize("pooled", [False, True])
@torch.inference_mode()
def test_grammar_trees_follow_the_model_and_keep_linear_kv(pooled):
    encode = _GrammarTokenizer().encode
    scripts = (
        encode("move forward 25cm, stop"),
        encode("turn left 15 degrees") + [_EOS],
        [_UNKNOWN, _UNKNOWN, _EOS],
        encode("move forward 100cm, turn right 90 degrees, move forward 75cm"),
    )
    policy, runtime = _scripted_runtime(scripts, pooled)
    moved = []
    keep = runtime._keep_tree_nodes
    runtime._keep_tree_nodes = lambda kept: (moved.extend(kept), keep(kept))
    results = runtime.generate(runtime.prefill(runtime.prepare([_observation("1 1")] * len(scripts))))
    assert [g.token_ids[0].tolist() for g in results] == [*scripts[:3], scripts[3][: policy.max_new_tokens]]
    assert [g.stop_reason for g in results] == ["stop", "eos", "eos", "max_tokens"]
    # Off-prior digits leave gaps in the accepted node order, which must be compacted.
    assert runtime.counters["tree_accepted_tokens"] > 0 and moved
    for generation in results:
        assert generation.token_logprobs.dtype == torch.float32
        assert generation.token_logprobs.shape == generation.token_ids.shape
        kv = generation.memory.packed_kv[0, 0, 0, 0]
        expected = torch.arange(generation.memory.seq_len, dtype=kv.dtype)[:, None].expand_as(kv)
        torch.testing.assert_close(kv, expected, rtol=0, atol=0)


def test_stop_reasons_are_memoized_within_a_bounded_table(monkeypatch):
    calls = []

    def reference(policy, ids):
        calls.append(tuple(ids))
        return "stop" if ids[-1] == 5 else None

    monkeypatch.setattr(grammar, "_stop_reason", reference)
    stops = grammar.StopReasons(object(), capacity=2)
    assert [stops([1, 5]), stops([1, 2]), stops([1, 5])] == ["stop", None, "stop"]
    assert calls == [(1, 5), (1, 2)]
    stops([3])
    assert stops([1, 5]) == "stop"
    assert calls == [(1, 5), (1, 2), (3,), (1, 5)]


def test_history_sets_match_unique_tokens():
    generator = torch.Generator().manual_seed(3)
    histories = [torch.randint(0, 50, (size,), generator=generator) for size in (1, 7, 40)]
    assert grammar._history_sets(histories, 64) == [set(torch.unique(h).tolist()) for h in histories]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_upload_keeps_shapes_dtypes_and_values(device):
    owner = SimpleNamespace(device=torch.device(device))
    ints = np.arange(6, dtype=np.int64).reshape(2, 3)
    broadcast = np.broadcast_to(np.arange(3, dtype=np.int64), (2, 2, 3))
    first, second = ActiveVLNBatchedRuntime._upload(owner, ints, broadcast)
    ints[:] = -1
    (mask,) = ActiveVLNBatchedRuntime._upload(owner, np.eye(3, dtype=bool))
    assert first.device.type == device and first.dtype == torch.int64 and mask.dtype == torch.bool
    assert first.cpu().tolist() == [[0, 1, 2], [3, 4, 5]]
    assert torch.equal(second.cpu(), torch.arange(3).expand(2, 2, 3))
    assert torch.equal(mask.cpu(), torch.eye(3, dtype=torch.bool))


@pytest.mark.parametrize("pooled", [False, True])
@torch.inference_mode()
def test_kept_tree_nodes_move_like_per_row_index_select(pooled):
    runtime = _policy().create_batched_runtime(
        batch_size=2, workspace_tokens=64, kv_pool_tokens=256 if pooled else None
    )
    if pooled:
        runtime._row_starts = [0, 128]
    runtime.storage.copy_(torch.randn(runtime.storage.shape, generator=torch.Generator().manual_seed(5)))
    kept = [(0, 5, [0, 2, 3]), (1, 9, [0, 1, 4, 6])]
    expected = runtime.storage.clone()
    actual, runtime.storage = runtime.storage, expected
    for row, start, nodes in kept:
        index = torch.tensor(nodes)
        moved = runtime._row_kv(row, start, start + max(nodes) + 1).index_select(-2, index)
        runtime._row_kv(row, start, start + len(nodes)).copy_(moved)
    runtime.storage = actual
    runtime._keep_tree_nodes(kept)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_reused_commit_buffer_keeps_its_layer_views():
    runtime = _policy().create_batched_runtime(batch_size=1, workspace_tokens=64)
    first = runtime.generate(runtime.prefill(runtime.prepare([_observation("3 4")])))[0].memory
    second = runtime.generate(runtime.prefill(runtime.prepare([_observation("5")], [first])))[0].memory
    assert second._packed_kv is first._packed_kv
    assert all(new is old for new, old in zip(second.layers, first.layers, strict=True))
    for index, (key, value) in enumerate(second.visible_kv()):
        torch.testing.assert_close(key, second.packed_kv[index, 0], rtol=0, atol=0)
        torch.testing.assert_close(value, second.packed_kv[index, 1], rtol=0, atol=0)
