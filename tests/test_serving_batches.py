"""Cross-session batching, row isolation, and bounded shutdown on CPU."""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.serve.batching import BatchedServingAdapter
from embodiinfer.engine.serve.contracts import ModelAction, ModelResult, RawPolicyRequest, ServeError
from embodiinfer.engine.serve.service import PolicyService
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch
from embodiinfer.policies.pi05.serving import Pi05ServingAdapter, Pi05ServingConfig
from embodiinfer.types import ActionChunk


def request(index=0):
    return RawPolicyRequest(f"session-{index}", f"request-{index}", 0, "pick", {"state": [index]}, (), {})


class BatchAdapter:
    action_space = "test.actions"

    def __init__(self):
        self.batches = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = None
        self.resets = []

    def capabilities(self):
        return {"action_space": self.action_space}

    def infer(self, item):
        return self.infer_batch([item])[0]

    def infer_batch(self, items):
        self.batches.append(list(items))
        self.started.set()
        assert self.release.wait(5)
        if self.failure:
            raise self.failure
        return [ModelResult(self.action_space, (ModelAction("value", dict(r.state)),), {}) for r in items]

    def reset(self, session_id):
        self.resets.append(session_id)


@pytest.mark.parametrize("size", [1, 2, 3])
def test_window_executes_one_batch_and_routes_each_row(size):
    adapter = BatchAdapter()
    service = BatchedServingAdapter(adapter, max_batch=size, max_wait_ms=1000)
    try:
        with ThreadPoolExecutor(size) as pool:
            results = list(pool.map(service.infer, [request(i) for i in range(size)]))
        assert [r.actions[0].values["state"] for r in results] == [[i] for i in range(size)]
        assert [len(b) for b in adapter.batches] == [size]
    finally:
        service.shutdown()


def test_partial_window_and_failed_model_do_not_stall_next_request():
    adapter = BatchAdapter()
    service = BatchedServingAdapter(adapter, max_batch=3, max_wait_ms=1)
    try:
        adapter.failure = RuntimeError("model failed")
        with pytest.raises(RuntimeError, match="model failed"):
            service.infer(request())
        adapter.failure = None
        assert service.infer(request(1)).actions[0].values == {"state": [1]}
    finally:
        service.shutdown()


def test_bounded_queue_shutdown_and_active_call():
    adapter = BatchAdapter()
    adapter.release.clear()
    service = BatchedServingAdapter(adapter, max_batch=1, max_pending=1)
    with ThreadPoolExecutor(3) as pool:
        active = pool.submit(service.infer, request())
        assert adapter.started.wait(2)
        queued = pool.submit(service.infer, request(1))
        with service._condition:
            assert service._condition.wait_for(lambda: len(service._pending) == 1, timeout=2)
        with pytest.raises(ServeError) as error:
            service.infer(request(2))
        assert error.value.status == 429
        stopping = pool.submit(service.shutdown)
        with pytest.raises(ServeError) as error:
            queued.result(timeout=2)
        assert error.value.status == 503
        adapter.release.set()
        assert active.result(timeout=2).actions[0].values == {"state": [0]}
        stopping.result(timeout=2)
    with pytest.raises(ServeError):
        service.infer(request(3))


def test_session_idempotency_survives_cross_session_batching():
    adapter = BatchAdapter()
    batched = BatchedServingAdapter(adapter, max_batch=3, max_wait_ms=1000)
    service = PolicyService(batched)
    requests = []
    for i in range(3):
        session = service.open_session(
            {
                "schema": "embodiinfer.policy.session.v1",
                "robot_id": str(i),
                "action_space": adapter.action_space,
            }
        )
        requests.append(replace(request(i), session_id=session["session_id"]))
    try:
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(service.step, r) for r in [*requests, requests[0]]]
            results = [f.result(timeout=3) for f in futures]
        assert results[0] == results[-1]
        assert len(adapter.batches) == 1 and len(adapter.batches[0]) == 3
        assert [r["session_id"] for r in results[:3]] == [r.session_id for r in requests]
    finally:
        batched.shutdown()


def test_reset_waits_for_admitted_step():
    adapter = BatchAdapter()
    adapter.release.clear()
    batched = BatchedServingAdapter(adapter, max_batch=1)
    service = PolicyService(batched)
    session = service.open_session(
        {"schema": "embodiinfer.policy.session.v1", "robot_id": "robot", "action_space": adapter.action_space}
    )["session_id"]
    try:
        with ThreadPoolExecutor(2) as pool:
            step = pool.submit(service.step, replace(request(), session_id=session))
            assert adapter.started.wait(2)
            reset = pool.submit(service.reset, session, {"request_id": "reset-0"})
            assert not adapter.resets
            adapter.release.set()
            step.result(timeout=2)
            reset.result(timeout=2)
        assert adapter.resets == [session]
    finally:
        adapter.release.set()
        batched.shutdown()


class Processor:
    def prepare(self, state, images, prompt):
        token = state.long().reshape(1, 1)
        return Pi05Batch(
            [torch.zeros(1, 3, 2, 2)],
            [torch.ones(1, dtype=torch.bool)],
            token,
            torch.ones_like(token, dtype=torch.bool),
        )

    def restore_actions(self, actions, state):
        return actions + state  # relative actions must use the matching row's state


def pi_adapter():
    adapter = object.__new__(Pi05ServingAdapter)
    adapter._lock = threading.Lock()
    adapter._config = Pi05ServingConfig(("state",), ("image",), 2)
    adapter._processor = Processor()
    adapter._state_vector = lambda state: torch.tensor(state["state"], dtype=torch.float32)
    adapter._image_tensor_stack = lambda images: (torch.zeros(1, 3, 2, 2), ("image",))
    batches = []

    def execute(batch):
        batches.append(batch)
        return [
            ActionChunk(request_id=rid, actions=batch.tokens[i].float().repeat(2, 1), latency_ms=1)
            for i, rid in enumerate(batch.request_ids)
        ]

    adapter._core = SimpleNamespace(execute=execute, policy_version=0)
    return adapter, batches


def test_pi05_preparation_and_state_restoration_match_single_rows_exactly():
    adapter, batches = pi_adapter()
    rows = [request(i) for i in (1, 2, 3)]
    singles = [adapter.infer(r) for r in rows]
    results = adapter.infer_batch(rows)
    assert results == singles
    assert batches[-1].batch_size == 3
    assert batches[-1].request_ids == [r.request_id for r in rows]
    assert batches[-1].masks.tolist() == [[True], [True], [True]]
    assert [r.actions[0].values["data"] for r in results] == [[[2.0], [2.0]], [[4.0], [4.0]], [[6.0], [6.0]]]


def test_bad_pi05_row_does_not_poison_valid_peers():
    adapter, batches = pi_adapter()
    prepare = adapter._processor.prepare

    def checked(state, images, prompt):
        if state.item() < 0:
            raise ValueError("invalid state")
        return prepare(state, images, prompt)

    adapter._processor.prepare = checked
    results = adapter.infer_batch([request(1), request(-1), request(3)])
    assert isinstance(results[1], ValueError)
    assert isinstance(results[0], ModelResult) and isinstance(results[2], ModelResult)
    assert batches[-1].batch_size == 2


def test_graph_bucket_respects_non_power_of_two_ceiling():
    config = EngineConfig(max_batch_size=3)
    assert [config.resolve_bucket(n) for n in (1, 2, 3)] == [1, 2, 3]
    with pytest.raises(ValueError):
        config.resolve_bucket(4)


def test_shared_cli_builds_batch_wrapper_only_when_requested(monkeypatch):
    from embodiinfer.engine.serve import factory

    adapter = BatchAdapter()
    monkeypatch.setattr(factory, "make_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(factory, "EngineCore", lambda *args: None)
    monkeypatch.setattr(factory, "_build_adapter", lambda *args: adapter)
    parser = argparse.ArgumentParser()
    factory.add_policy_arguments(parser)
    assert factory.build_serving_adapter(parser.parse_args([])) is adapter
    batched = factory.build_serving_adapter(parser.parse_args(["--max-batch", "3", "--max-wait-ms", "2"]))
    try:
        assert batched.capabilities()["max_batch_size"] == 3
        assert batched.capabilities()["max_wait_ms"] == 2
    finally:
        batched.shutdown()
    for flags in (["--max-batch", "0"], ["--max-wait-ms", "nan"]):
        with pytest.raises(ValueError):
            factory.build_serving_adapter(parser.parse_args(flags))
    unsupported = SimpleNamespace(
        action_space="other", capabilities=lambda: {}, infer=lambda r: None, reset=lambda s: None
    )
    monkeypatch.setattr(factory, "_build_adapter", lambda *args: unsupported)
    with pytest.raises(ValueError, match="does not implement"):
        factory.build_serving_adapter(parser.parse_args(["--max-batch", "3"]))


@pytest.mark.gpu
def test_three_row_graph_and_partial_batches_match_same_shape_eager():
    from embodiinfer import Observation, preset_config
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.factory import make_policy
    from embodiinfer.types import collate

    policy = make_policy("mock_flow_vla", preset="tiny")
    cfg = preset_config("tiny")
    eager = EngineCore(policy, EngineConfig(device="cuda", max_batch_size=3, use_cuda_graph=False))
    graph = EngineCore(policy, EngineConfig(device="cuda", max_batch_size=3, capture_full_loop=True))
    for size in (3, 1, 2, 3):
        observations = [
            Observation(
                torch.zeros(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
                torch.zeros(cfg.state_dim),
                torch.zeros(cfg.max_lang_len, dtype=torch.long),
            )
            for _ in range(size)
        ]
        batch = collate(observations, [str(i) for i in range(size)])
        reference = eager.execute(batch, generator=torch.Generator(device="cuda").manual_seed(42))
        actual = graph.execute(batch, generator=torch.Generator(device="cuda").manual_seed(42))
        for a, b in zip(reference, actual, strict=True):
            torch.testing.assert_close(a.actions, b.actions, rtol=0, atol=0)
