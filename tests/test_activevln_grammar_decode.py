"""Independent grammar helpers and device-upload layouts."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodiinfer.policies.activevln import grammar_tree_activevln as grammar
from embodiinfer.policies.activevln.batching_activevln import ActiveVLNBatchedRuntime


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


@pytest.mark.parametrize("top_p", [1.0, 0.8])
def test_sampled_tokens_and_logprobs_follow_the_same_distribution(top_p):
    policy = SimpleNamespace(do_sample=True, temperature=0.7, top_p=top_p)
    logits = torch.tensor([[0.4, -0.3, 0.1, 0.8]])
    scaled = logits.float() / policy.temperature
    ordered, indices = scaled.sort(descending=True, dim=-1)
    removed = ordered.softmax(-1).cumsum(-1) > top_p
    removed[:, 1:] = removed[:, :-1].clone()
    removed[:, 0] = False
    expected = scaled.scatter(1, indices, ordered.masked_fill(removed, -torch.inf)).log_softmax(-1)
    generator = torch.Generator().manual_seed(42)
    tokens, scores = grammar.select_tokens(logits.expand(20000, -1), policy, generator)
    torch.testing.assert_close(scores, expected[0, tokens], atol=1e-6, rtol=0)
    observed = torch.bincount(tokens, minlength=4).float() / tokens.numel()
    # 0.015 exceeds four binomial standard deviations at this sample size.
    torch.testing.assert_close(observed, expected.exp()[0], atol=0.015, rtol=0)


@pytest.fixture(scope="module")
def checkpoint_policy():
    checkpoint = os.environ.get("VVLA_ACTIVEVLN_CKPT")
    if not checkpoint or not torch.cuda.is_available():
        pytest.skip("set VVLA_ACTIVEVLN_CKPT and run on CUDA")
    from embodiinfer.policies import make_policy

    return (
        make_policy(
            "activevln",
            checkpoint=checkpoint,
            attention="eager",
            max_context=2048,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )


@pytest.mark.gpu
@pytest.mark.activevln
@pytest.mark.parametrize("budget", [0, 31])
@torch.no_grad()
def test_sampled_grammar_tree_checkpoint_graph_replay(checkpoint_policy, budget):
    from embodiinfer.policies.activevln.batching_activevln import PreparedActiveVLNBatch
    from embodiinfer.types import Observation

    policy = checkpoint_policy
    prepared = tuple(
        policy.prepare_device(
            policy.prepare_host(
                Observation(
                    images=torch.linspace(0, 1, 3 * 56 * 56).reshape(3, 56, 56),
                    state=torch.empty(0),
                    instruction_tokens=torch.empty(0, dtype=torch.long),
                    instruction=instruction,
                ),
                initial=True,
            ),
            None,
        )
        for instruction in ("Move forward to the doorway.", "Turn left and stop by the door.")
    )
    reference = None
    for cuda_graph in (False, True):
        runtime = policy.create_batched_runtime(
            batch_size=2,
            grammar_tree_budget=budget,
            sampling_seed=42,
            workspace_tokens=512,
            query_bucket_size=32,
            cuda_graph=cuda_graph,
        )
        batch = PreparedActiveVLNBatch(prepared)
        rng_state = runtime.sampling_generator.get_state()
        runtime.prewarm_prepared(batch)
        runtime.prewarm_prepared(batch)
        assert torch.equal(rng_state, runtime.sampling_generator.get_state())
        prefix = runtime.prefill(batch)
        if cuda_graph:
            with pytest.raises(RuntimeError, match="pending generation"):
                runtime.prewarm_prepared(batch)
        generations = runtime.generate(prefix)
        if budget:
            assert runtime.counters["tree_accepted_tokens"] > 0
        if reference is None:
            reference = generations
            continue
        assert runtime.counters["prefill_replays"] > 0
        assert runtime.counters["tree_replays" if budget else "decode_replays"] > 0
        for actual, expected in zip(generations, reference, strict=True):
            torch.testing.assert_close(actual.token_ids, expected.token_ids, atol=0, rtol=0)
            torch.testing.assert_close(actual.token_logprobs, expected.token_logprobs, atol=0, rtol=0)
            torch.testing.assert_close(
                actual.memory.position_ids, expected.memory.position_ids, atol=0, rtol=0
            )
