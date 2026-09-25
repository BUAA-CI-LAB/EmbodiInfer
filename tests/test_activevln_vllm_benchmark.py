"""The external-engine benchmark must retain tokens and original stop semantics."""

import importlib.metadata
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def benchmark(monkeypatch, activevln_benchmark_modules):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    return activevln_benchmark_modules["benchmark_vllm"]


@pytest.mark.parametrize("tokens,reason", [([7, 151645], "eos"), ([8], "stop"), ([7, 7], "max_tokens")])
def test_complete_generation_and_own_history_survive_next_observation(benchmark, tokens, reason):
    runtime = benchmark.NativeVLLMReplay.__new__(benchmark.NativeVLLMReplay)
    runtime.device = torch.device("cpu")
    runtime.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "initial")
    runtime.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [100 if text == "initial" else 101],
        decode=lambda values, **kwargs: "stop" if 8 in values else "move forward 25cm",
    )
    runtime.history, runtime.images, runtime.call_index = [], [], 0
    runtime.sampling = object()
    runtime.timer = SimpleNamespace(
        reset=lambda: None,
        first_token=lambda: None,
        finish=lambda: {"pure_inference_ms": 1.0},
        vision_calls=1,
    )

    class Engine:
        def add_request(self, identity, prompt, sampling):
            self.identity, self.prompt, self.index, self.active = identity, prompt, 0, True
            runtime.last_processed = SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"])

        def has_unfinished_requests(self):
            return self.active

        def step(self):
            self.index += 1
            if self.index == len(tokens) and reason != "stop":
                self.active = False
            return [
                SimpleNamespace(
                    request_id=self.identity, outputs=[SimpleNamespace(token_ids=tokens[: self.index])]
                )
            ]

        def abort_request(self, identities):
            assert identities == [self.identity]
            self.active = False

    runtime.engine = Engine()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    first = runtime.call(rgb, "walk")
    assert first["token_ids"] == tokens and first["stop_reason"] == reason
    assert runtime.history == [100, *tokens]
    second = runtime.call(rgb, "walk")
    assert runtime.engine.prompt["prompt_token_ids"] == [100, *tokens, 101]
    assert second["history_images"] == 2
    assert second["parsed_valid"]
    assert second["cache_length"] == 2 + 2 * len(tokens)


def test_forward_interval_rejects_missing_vision_or_unfinished_generation(benchmark):
    timer = benchmark.ModelInterval.__new__(benchmark.ModelInterval)
    timer.started_ns = None
    timer.prefill_recorded = False
    with pytest.raises(RuntimeError, match="no vision"):
        timer.first_token()
    with pytest.raises(RuntimeError, match="incomplete"):
        timer.finish()


def test_placeholder_rule_deduplication_retains_identity_and_first_match_order(benchmark):
    first, equal_but_distinct, last = [], [], [1]
    original = {"image": [first, equal_but_distinct, first, last, equal_but_distinct], "video": [last, last]}
    unique = benchmark.unique_update_rules(original)
    assert [id(x) for x in unique["image"]] == [id(first), id(equal_but_distinct), id(last)]
    assert [id(x) for x in unique["video"]] == [id(last)]
    assert len(original["image"]) == 5


@pytest.mark.parametrize("count", [0, 1, 3, 16, 128])
def test_native_placeholder_matches_are_unchanged_for_full_histories(benchmark, count):
    pytest.importorskip("vllm")
    if importlib.metadata.version("vllm") != "0.8.5.post1":
        pytest.skip("frontend regression belongs to the separately pinned 0.8.5 engine")
    from vllm.multimodal.processing import PromptReplacement, find_mm_placeholders

    rule = PromptReplacement(
        modality="image", target=[99], replacement=lambda index: [99] * (2 + index % 3)
    ).bind(SimpleNamespace())
    # Ordinary tokens before/between images exercise the repeated failed scans.
    tokens = [1, 2, 3] * 100
    for index in range(count):
        tokens += [99] * (2 + index % 3) + [4, 5, 6] * 100
    repeated = {"image": [rule] * count}
    original = find_mm_placeholders(repeated, tokens, {"image": count})
    deduplicated = find_mm_placeholders(benchmark.unique_update_rules(repeated), tokens, {"image": count})
    assert original == deduplicated
    assert len(original.get("image", [])) == count
