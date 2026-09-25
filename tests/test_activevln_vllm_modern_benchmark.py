"""Modern speculative blocks must preserve the original per-token stop contract."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def modern(activevln_benchmark_modules):
    return activevln_benchmark_modules["benchmark_vllm_modern"]


@pytest.mark.parametrize(
    "tokens,checked,expected,reason",
    [
        ([1, 151645, 2, 3], 0, [1, 151645], "eos"),
        ([1, 151643, 2], 1, [1, 151643], "eos"),
        ([2, 3, 4, 5, 151645], 0, [2, 3, 4], "stop"),
        ([2, 3, 4, 5], 2, [2, 3, 4], "stop"),
        ([2, 3], 0, [2, 3], None),
        ([1, 1, 1], 1, [1, 1, 1], None),
    ],
)
def test_stopping_prefix_preserves_earliest_stop_inside_native_block(
    modern, tokens, checked, expected, reason
):
    def decode(values, **kwargs):
        if values == [2]:
            return "s"
        if values == [2, 3]:
            return "st"
        if values[:3] == [2, 3, 4]:
            return "stop" if len(values) == 3 else "stop move forward 25cm"
        return "move forward 25cm"

    assert modern.stopping_prefix(SimpleNamespace(decode=decode), tokens, checked) == (expected, reason)


def test_forward_interval_requires_actual_encoder_graph_manager(modern):
    with pytest.raises(RuntimeError, match="did not initialize"):
        modern.EncoderInterval(None, None)


def test_speculative_tokens_after_stop_never_enter_next_observation(modern, monkeypatch):
    import numpy as np
    import torch

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    runtime = modern.ModernVLLMReplay.__new__(modern.ModernVLLMReplay)
    runtime.device = torch.device("cpu")
    runtime.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "initial")
    runtime.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [100 if text == "initial" else 101],
        decode=lambda tokens, **kwargs: "stop" if 8 in tokens else "move forward 25cm",
    )
    runtime.history, runtime.images, runtime.image_ids = [], [], []
    runtime.episode_index = runtime.call_index = 0
    runtime.sampling = object()
    runtime.core = SimpleNamespace(batch_queue=[])
    manager = SimpleNamespace(graph_hits=0, graph_misses=0)
    runtime.encoder_manager = manager
    runtime.timer = SimpleNamespace(
        reset=lambda: None,
        first_token=lambda: None,
        finish=lambda: {"pure_inference_ms": 0.0},
        vision_calls=1,
    )

    class Engine:
        renderer = SimpleNamespace(render_cmpl=lambda prompts: prompts)

        def add_request(self, identity, prompt, params):
            self.identity, self.prompt, self.active = identity, prompt, True
            runtime.last_processed = SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"])

        def has_unfinished_requests(self):
            return self.active

        def step(self):
            self.active = False
            manager.graph_hits += 1
            return [SimpleNamespace(request_id=self.identity, outputs=[SimpleNamespace(token_ids=[7, 8, 9])])]

        def abort_request(self, identities):
            assert identities == [self.identity]
            self.active = False

    runtime.engine = Engine()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    first = runtime.call(rgb, "walk")
    assert first["token_ids"] == [7, 8]
    assert first["native_returned_tokens"] == 3
    assert first["stop_reason"] == "stop"
    second = runtime.call(rgb, "walk")
    assert runtime.engine.prompt["prompt_token_ids"] == [100, 7, 8, 101]
    assert second["history_images"] == 2
    assert runtime.image_ids == ["ep-0-frame-1", "ep-0-frame-2"]


def test_async_pending_work_is_drained_before_next_observation(modern):
    core = SimpleNamespace(batch_queue=["older", "newer"])

    def step():
        core.batch_queue.pop(0)
        return []

    assert modern.drain_inflight(SimpleNamespace(step=step), core) == 2
    assert not core.batch_queue
