"""Shared-model transport, scheduling, identity and request lifetime contracts."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
import torch
from PIL import Image

from embodiinfer.engine.serve.batching import BatchedServingAdapter
from embodiinfer.engine.serve.contracts import ServeError
from embodiinfer.engine.serve.http_server import PolicyHttpService, create_http_server
from embodiinfer.engine.serve.text import ExecutionGate, parse_text_request
from embodiinfer.policies.activevln.text_activevln import ActiveVLNTextResult
from embodiinfer.policies.activevln.text_serving import ActiveVLNSharedTextAdapter


class Navigation:
    action_space = "test.action.v1"

    def __init__(self):
        self.policy = torch.nn.Linear(1, 1)
        self.policy.policy_version = 0
        self.resets = []
        self.calls = []

    def capabilities(self):
        return {"action_space": self.action_space}

    def infer(self, request):
        self.calls.append(request)
        return request

    def infer_batch(self, requests):
        return [self.infer(request) for request in requests]

    def reset(self, session):
        self.resets.append(session)


class Decoder:
    def __init__(self, policy):
        self.policy = policy
        self.calls = []
        self.steps = 0
        self.invalidated = False

    def prepare(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return object()

    def step(self, state):
        self.steps += 1
        return self.steps < 2

    def result(self, state):
        return ActiveVLNTextResult("raw, not JSON", (4, 9), 7, "eos")

    def invalidate(self):
        self.invalidated = True


def adapter():
    navigation = Navigation()
    result = ActiveVLNSharedTextAdapter(navigation, navigation.policy)
    result.decoder = Decoder(navigation.policy)
    return result


def payload(owner=None, **updates):
    value = {
        "schema": "vvla.text.request.v1",
        "request_id": "request-1",
        "model_instance": owner.capabilities()["shared_text"]["model_instance"] if owner else "instance-1",
        "messages": [{"role": "user", "content": "What is ahead?"}],
        "max_new_tokens": 32,
        "timeout_s": 10,
    }
    return {**value, **updates}


def test_shared_parameters_raw_output_and_independent_navigation_reset():
    owner = adapter()
    assert owner.decoder.policy is owner.adapter.policy
    request = parse_text_request(payload(owner))
    owner.infer("navigation")
    owner.reset("session-1")
    result = owner.infer_text(request)
    assert result["model_instance"] == request.model_instance
    assert result["request_id"] == request.request_id
    assert result["text"] == "raw, not JSON" and result["token_ids"] == [4, 9]
    assert owner.adapter.calls == ["navigation"] and owner.adapter.resets == ["session-1"]
    assert not owner.decoder.invalidated
    with pytest.raises(ValueError, match="same policy"):
        ActiveVLNSharedTextAdapter(owner.adapter, torch.nn.Linear(1, 1))


def test_decode_graphs_reject_unsupported_device_at_startup():
    navigation = Navigation()
    with pytest.raises(ValueError, match="BF16 CUDA"):
        ActiveVLNSharedTextAdapter(navigation, navigation.policy, decode_graphs=True)
    with pytest.raises(ValueError, match="boolean"):
        ActiveVLNSharedTextAdapter(navigation, navigation.policy, decode_graphs="true")


def test_identity_refit_expiry_and_shutdown_reject_before_prefill():
    owner = adapter()
    request = parse_text_request(payload(owner))
    with pytest.raises(ServeError) as caught:
        owner.infer_text(replace(request, model_instance="foreign"))
    assert caught.value.status == 409
    with pytest.raises(ServeError) as caught:
        owner.infer_text(replace(request, deadline=time.monotonic() - 1))
    assert caught.value.status == 408
    owner.policy.policy_version += 1
    assert "shared_text" not in owner.capabilities()
    with pytest.raises(ServeError) as caught:
        owner.infer_text(request)
    assert caught.value.status == 409 and not owner.decoder.calls
    owner.policy.policy_version -= 1
    owner.shutdown()
    with pytest.raises(ServeError) as caught:
        owner.infer_text(request)
    assert caught.value.status == 503 and owner.decoder.invalidated


def test_only_one_live_text_cache_and_slot_is_released_on_error():
    owner = adapter()
    request = parse_text_request(payload(owner))
    entered, release = threading.Event(), threading.Event()

    def fail(messages, **kwargs):
        entered.set()
        assert release.wait(3)
        raise ValueError("bad input")

    owner.decoder.prepare = fail
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.infer_text, request)
        assert entered.wait(3)
        try:
            with pytest.raises(ServeError) as caught:
                owner.infer_text(request)
            assert caught.value.status == 429
        finally:
            release.set()
        with pytest.raises(ServeError, match="bad input"):
            future.result(timeout=3)
    owner.decoder = Decoder(owner.policy)
    assert owner.infer_text(request)["text"] == "raw, not JSON"


def test_navigation_admission_precedes_next_text_token():
    gate = ExecutionGate()
    order = []

    def work(name, priority):
        with gate.enter(priority=priority, deadline=time.monotonic() + 3):
            order.append(name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        with gate.enter():
            navigation = pool.submit(work, "navigation", True)
            with gate._condition:
                assert gate._condition.wait_for(lambda: gate._priority_waiters == 1, timeout=1)
            text = pool.submit(work, "text", False)
        navigation.result(timeout=3)
        text.result(timeout=3)
    assert order == ["navigation", "text"]
    with pytest.raises(ServeError), gate.enter(priority=True, deadline=time.monotonic() - 1):
        pytest.fail("expired request was admitted")
    with gate.enter(deadline=time.monotonic() + 1):
        pass


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "unknown"},
        {"model_instance": ""},
        {"request_id": "../escape"},
        {"timeout_s": float("nan")},
        {"timeout_s": True},
        {"timeout_s": 61},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"messages": []},
        {"messages": [{"role": [], "content": "x"}]},
        {"messages": [{"role": "user", "content": "x" * 65537}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "url": "file:///x"}]}]},
        {"image_base64": "%%%"},
        {"image_base64": "eA=="},
        {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
    ],
)
def test_transport_rejects_malformed_or_unbounded_requests(changes):
    with pytest.raises(ServeError):
        parse_text_request(payload(**changes))


def test_image_reaches_same_decoder_and_oversized_image_is_rejected():
    owner = adapter()
    for size, accepted in [((4, 3), True), ((641, 480), False)]:
        buffer = io.BytesIO()
        Image.new("RGB", size).save(buffer, "PNG")
        request = parse_text_request(
            payload(
                owner,
                messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Look"}]}],
                image_base64=base64.b64encode(buffer.getvalue()).decode(),
            )
        )
        if accepted:
            owner.infer_text(request)
            assert owner.decoder.calls[-1][1]["images"][0].size == size
        else:
            with pytest.raises(ServeError, match="640x480"):
                owner.infer_text(request)


def test_batch_wrapper_preserves_text_capability_and_closes_owner():
    owner = adapter()
    wrapper = BatchedServingAdapter(owner, max_batch=2)
    request = parse_text_request(payload(owner))
    assert wrapper.infer_text(request)["model_instance"] == request.model_instance
    wrapper.shutdown()
    assert owner.decoder.invalidated
    with pytest.raises(ServeError) as caught:
        wrapper.infer_text(request)
    assert caught.value.status == 503
    plain = BatchedServingAdapter(Navigation(), max_batch=2)
    with pytest.raises(ServeError) as caught:
        plain.infer_text(request)
    assert caught.value.status == 404
    plain.shutdown()


def test_http_text_route_uses_existing_auth_and_capability_identity():
    owner = adapter()
    service = PolicyHttpService(owner, token="test-token", max_body_bytes=1024**2)
    server = create_http_server(service, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = build_opener(ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_port}/v1/text"
    try:
        body = json.dumps(payload(owner)).encode()
        with pytest.raises(HTTPError) as caught:
            client.open(Request(url, data=body), timeout=3)
        assert caught.value.code == 401 and not owner.decoder.calls
        request = Request(url, data=body, headers={"Authorization": "Bearer test-token"})
        with client.open(request, timeout=3) as response:
            result = json.load(response)
        assert result["text"] == "raw, not JSON"
        assert result["model_instance"] == owner.capabilities()["shared_text"]["model_instance"]
    finally:
        server.shutdown()
        server.server_close()
        owner.shutdown()
        thread.join(timeout=3)
