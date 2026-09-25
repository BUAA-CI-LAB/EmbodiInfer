"""Batch replay must isolate histories and measure real rather than nominal occupancy."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def batch_module(activevln_benchmark_modules):
    return activevln_benchmark_modules["benchmark_vllm_batch"]


class FakeEngine:
    def __init__(self, steps):
        self.steps = list(steps)
        self.aborted = []

    def has_unfinished_requests(self):
        return bool(self.steps)

    def step(self):
        return self.steps.pop(0)

    def abort_request(self, identities):
        self.aborted.extend(identities)


def output(identity, tokens, finished=False):
    return SimpleNamespace(
        request_id=identity,
        finished=finished,
        outputs=[SimpleNamespace(token_ids=tokens, finish_reason="length" if finished else None)],
    )


def test_unordered_outputs_stop_independently_and_discard_speculative_suffix(batch_module):
    tokenizer = SimpleNamespace(
        decode=lambda tokens, **kwargs: "stop" if 8 in tokens else "move forward 25cm"
    )
    engine = FakeEngine(
        [
            [output("b", [1]), output("a", [7, 8, 9])],
            [output("b", [1, 2, 151645, 9], True)],
        ]
    )
    result, steps = batch_module.collect_outputs(
        engine, tokenizer, ["a", "b"], SimpleNamespace(first_token=lambda: None)
    )
    assert steps == 2
    assert result["a"]["token_ids"] == [7, 8]
    assert result["a"]["stop_reason"] == "stop"
    assert result["b"]["token_ids"] == [1, 2, 151645]
    assert result["b"]["stop_reason"] == "eos"
    assert engine.aborted == ["a", "b"]


@pytest.mark.parametrize(
    "steps,match",
    [
        ([[output("unknown", [1], True)]], "unexpected"),
        ([[output("a", [1], True)]], "every request"),
        ([[output("a", [1])], [output("a", [2], True)]], "rewrote"),
        ([[output("a", [], True), output("b", [1], True)]], "every request"),
    ],
)
def test_incomplete_or_corrupted_outputs_fail(batch_module, steps, match):
    with pytest.raises(RuntimeError, match=match):
        batch_module.collect_outputs(
            FakeEngine(steps),
            SimpleNamespace(decode=lambda *args, **kwargs: "move forward 25cm"),
            ["a", "b"],
            SimpleNamespace(first_token=lambda: None),
        )


def test_amortization_counts_partial_tail(batch_module):
    rows = [
        {"observations": 4, "latency_ms": 100, "model_timing_ms": {"pure_inference_ms": 80}},
        {"observations": 1, "latency_ms": 60, "model_timing_ms": {"pure_inference_ms": 40}},
    ]
    stats = batch_module.summarize_batches(rows)
    assert stats["mean_batch_occupancy"] == 2.5
    assert stats["batch_e2e_ms"]["mean"] == 80
    assert stats["amortized_e2e_ms_per_observation"] == 32
    assert stats["amortized_forward_ms_per_observation"] == 24
    assert stats["observations_per_second"] == 31.25


def test_episode_histories_are_private(batch_module):
    a, b = [batch_module.ReplaySession(name, None) for name in ("a", "b")]
    a.history.append(1)
    a.images.append("rgb")
    a.image_ids.append("a-frame-1")
    assert not b.history and not b.images and not b.image_ids


def test_refill_keeps_surviving_history_and_episode_cache_salt(batch_module, monkeypatch):
    import numpy as np
    import torch

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    runtime = batch_module.BatchedVLLMReplay.__new__(batch_module.BatchedVLLMReplay)
    runtime.config = {"batch_size": 2}
    runtime.device = torch.device("cpu")
    runtime.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "initial")
    runtime.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [100 if text == "initial" else 101],
        decode=lambda tokens, **kwargs: "stop" if 8 in tokens else "move forward 25cm",
    )
    runtime.sampling = object()
    runtime.call_index = 0
    runtime.core = SimpleNamespace(batch_queue=[])
    runtime.timer = SimpleNamespace(
        reset=lambda: None,
        first_token=lambda: None,
        finish=lambda: {"pure_inference_ms": 0.0},
        vision_calls=1,
    )
    manager = SimpleNamespace(graph_hits=0, graph_misses=0)
    runtime.encoder_manager = manager

    class Engine:
        renderer = SimpleNamespace(render_cmpl=lambda prompts: prompts)

        def __init__(self):
            self.pending = []
            self.prompts = []

        def add_request(self, identity, prompt, params):
            self.pending.append(identity)
            self.prompts.append(prompt)
            runtime.last_processed = SimpleNamespace(
                request_id=identity, prompt_token_ids=prompt["prompt_token_ids"]
            )

        def has_unfinished_requests(self):
            return bool(self.pending)

        def step(self):
            pending, self.pending = self.pending, []
            runtime.vision_items = len(pending)
            manager.graph_hits += 1
            return [output(identity, [7, 8, 9]) for identity in reversed(pending)]

        def abort_request(self, identities):
            pass

    runtime.engine = Engine()
    a, b, c = [
        batch_module.ReplaySession(name, SimpleNamespace(episode_id=name, instruction="walk"))
        for name in ("a", "b", "c")
    ]
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    first, _ = runtime.call_batch([a, b], [rgb, rgb])
    a.step += 1
    second, _ = runtime.call_batch([a, c], [rgb, rgb])
    prompts = runtime.engine.prompts
    assert [p["cache_salt"] for p in prompts] == ["a", "b", "a", "c"]
    assert prompts[2]["prompt_token_ids"] == [100, 7, 8, 101]
    assert prompts[3]["prompt_token_ids"] == [100]
    assert prompts[2]["multi_modal_uuids"]["image"] == ["a-frame-1", "a-frame-2"]
    assert prompts[3]["multi_modal_uuids"]["image"] == ["c-frame-1"]
    assert b.history == [100, 7, 8]
    assert [row["episode_id"] for row in first + second] == ["a", "b", "a", "c"]
    assert all(row["token_ids"] == [7, 8] for row in first + second)
