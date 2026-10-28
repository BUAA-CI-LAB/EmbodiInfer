"""Batch transport must preserve primitive histories and commit atomically."""

import importlib.util
import io
import runpy
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest


@pytest.fixture
def adapter_type(monkeypatch):
    folder = Path(__file__).parents[1] / "benchmarks/streamvln-benchmark"
    spec = importlib.util.spec_from_file_location("benchmark", folder / "benchmark.py")
    benchmark = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    spec.loader.exec_module(benchmark)
    for name in ("synchronize", "set_device"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    for name in ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    return runpy.run_path(str(folder / "serve_batch.py"))["TensorBatchAdapter"]


class Runtime:
    batch_size = 2

    def __init__(self):
        self.received = []
        self.resident = [None, None]
        self.fail = False

    def infer(self, observations, identities, memories):
        self.received.append((observations, identities, memories))
        results = []
        for row, (observation, memory) in enumerate(zip(observations, memories, strict=True)):
            step = observation.metadata["stream_step_id"]
            start = 0 if memory is None else memory.step_id
            assert observation.images.shape[0] == step - start + 1
            trace = SimpleNamespace(
                action_mask=torch.tensor([True]),
                meta=dict(invalid_action_text=False, truncated_actions=False),
                token_ids=torch.tensor([int(observation.images[-1, 0, 0, 0])]),
                token_logprobs=torch.tensor([-0.5]),
                text="↑",
                stop_reason="eos",
            )
            actions = torch.tensor([[[1, float("nan") if self.fail and row == 1 else 0.25]]])
            next_memory = SimpleNamespace(
                step_id=step + 1, seq_len=step + 10, pending_token_ids=(), slow_frame_indices=()
            )
            results.append(SimpleNamespace(actions=actions, traces=[trace], next_memory=next_memory))
        self.resident[: len(results)] = [result.next_memory for result in results]
        return results

    def stats(self):
        return dict(calls=len(self.received))


def request(slot, *, step=0, first=0, name=None):
    images = []
    for frame in range(first, step + 1):
        stream = io.BytesIO()
        Image.fromarray(np.full((4, 4, 3), 20 * (slot + 1) + frame, dtype=np.uint8)).save(
            stream, format="PNG"
        )
        images.append(RawImage(f"observation.images.rgb.{len(images)}", "image/png", stream.getvalue()))
    return RawPolicyRequest(
        session_id=f"session-{slot}",
        request_id=name or f"request-{slot}-{step}",
        step_id=step,
        instruction=f"go {slot}",
        state={"benchmark_slot": slot},
        images=tuple(images),
        metadata={"stream_step_id": step},
    )


def test_batch_preserves_arrival_order_independent_strides_and_session_reset(adapter_type):
    runtime = Runtime()
    adapter = adapter_type(None, runtime, torch.device("cpu"), uint8=True)
    results = adapter.infer_batch([request(1), request(0)])
    assert [r.actions[0].values["tokens"] for r in results] == [[40], [20]]
    assert [o.instruction for o in runtime.received[0][0]] == ["go 0", "go 1"]
    assert all(
        r.actions[0].values["tensor_batch"] == dict(index=0, observations=2, capacity=2) for r in results
    )
    assert results[0].timing == results[1].timing
    results = adapter.infer_batch([request(0, step=4, first=1), request(1, step=2, first=1)])
    assert [r.actions[0].values["stream_step_id"] for r in results] == [4, 2]
    assert [o.images[:, 0, 0, 0].tolist() for o in runtime.received[1][0]] == [[21, 22, 23, 24], [41, 42]]
    remaining = adapter.memories["session-1"]
    adapter.reset("session-0")
    assert list(adapter.memories) == ["session-1"] and adapter.memories["session-1"] is remaining
    assert runtime.resident == [None, remaining]
    tail = adapter.infer(request(1, step=3, first=3))
    assert tail.actions[0].values["tensor_batch"] == dict(index=2, observations=1, capacity=2)


def test_batch_failure_publishes_no_partial_session_state(adapter_type):
    runtime = Runtime()
    adapter = adapter_type(None, runtime, torch.device("cpu"), uint8=True)
    adapter.infer_batch([request(0), request(1)])
    committed = dict(adapter.memories)
    runtime.fail = True
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.infer_batch([request(0, step=2, first=1), request(1, step=4, first=1)])
    assert adapter.memories == committed and adapter.index == 1
    with pytest.raises(RuntimeError, match="invalidated"):
        adapter.infer(request(0, step=1, first=1))
    assert len(runtime.received) == 2


@pytest.mark.parametrize(
    "malformed", ["duplicate_slot", "duplicate_session", "frame_order", "missing_step", "too_many_frames"]
)
def test_malformed_batch_is_rejected_before_any_model_call(adapter_type, malformed):
    runtime = Runtime()
    adapter = adapter_type(None, runtime, torch.device("cpu"), uint8=True)
    first, second = request(0), request(1)
    if malformed == "duplicate_slot":
        second = replace(second, state=first.state)
    elif malformed == "duplicate_session":
        second = replace(second, session_id=first.session_id)
    elif malformed == "frame_order":
        first = replace(first, images=(replace(first.images[0], name="observation.images.rgb.1"),))
    elif malformed == "missing_step":
        first = replace(first, metadata={})
    else:
        first = request(0, step=4)
    with pytest.raises(ValueError):
        adapter.infer_batch([first, second])
    assert not runtime.received and not adapter.memories and not adapter.failed
