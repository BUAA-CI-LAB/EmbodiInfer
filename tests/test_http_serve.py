from __future__ import annotations

import json
import socket
import threading
from http.server import ThreadingHTTPServer

import pytest

from embodiinfer.engine.serve.contracts import ModelAction, ModelResult, ServeError
from embodiinfer.engine.serve.http_server import PolicyHttpHandler, PolicyHttpService


class FakeAdapter:
    action_space = "pi05.action_chunk.v1"

    def __init__(self) -> None:
        self.calls = 0
        self.resets: list[str] = []
        self.failure: Exception | None = None

    def capabilities(self):
        return {"model": "fake", "action_space": self.action_space}

    def infer(self, request):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return ModelResult(
            action_space=self.action_space,
            actions=(ModelAction("action_chunk", {"data": [[float(request.step_id)]]}),),
            timing={"policy_ms": 1.0},
        )

    def reset(self, session_id: str) -> None:
        self.resets.append(session_id)


def _service(adapter=None, **kwargs):
    return PolicyHttpService(
        adapter or FakeAdapter(),
        token=None,
        max_body_bytes=1024 * 1024,
        **kwargs,
    )


def _session(service):
    return service.open_session(
        {
            "schema": "embodiinfer.policy.session.v1",
            "robot_id": "fr3",
            "action_space": "pi05.action_chunk.v1",
        }
    )["session_id"]


def _multipart(session_id: str, request_id: str, *, step_id: int = 0, state_value: float = 0.0, boundary="b"):
    metadata = {
        "schema": "embodiinfer.policy.step.v1",
        "session_id": session_id,
        "request_id": request_id,
        "step_id": step_id,
        "instruction": "move",
        "state": {"joint_positions_rad": [state_value] * 7},
        "images": [{"name": "camera-0.jpg", "mime_type": "image/jpeg"}],
        "metadata": {},
    }
    pieces = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\nContent-Type: application/json\r\n\r\n'.encode()
        + json.dumps(metadata).encode()
        + b"\r\n",
        f'--{boundary}\r\nContent-Disposition: form-data; name="image_0"; filename="camera.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
        + b"jpeg-bytes\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return f"multipart/form-data; boundary={boundary}", b"".join(pieces)


def _step(service, session_id, request_id, **kwargs):
    content_type, body = _multipart(session_id, request_id, **kwargs)
    return service.step(
        session_id=session_id,
        content_type=content_type,
        body=body,
        idempotency_key=request_id,
    )


def test_idempotency_is_semantic_and_rejects_changed_payload() -> None:
    adapter = FakeAdapter()
    service = _service(adapter)
    session_id = _session(service)
    first = _step(service, session_id, "request-0", boundary="first")
    retried = _step(service, session_id, "request-0", boundary="second")
    assert retried == first
    assert adapter.calls == 1

    with pytest.raises(ServeError, match="different content") as caught:
        _step(service, session_id, "request-0", state_value=1.0, boundary="third")
    assert caught.value.status == 409
    assert adapter.calls == 1


def test_reset_retires_step_keys_and_is_itself_idempotent() -> None:
    adapter = FakeAdapter()
    service = _service(adapter)
    session_id = _session(service)
    _step(service, session_id, "request-0")
    body = {"request_id": "reset-0"}
    first = service.reset(session_id=session_id, body=body, idempotency_key="reset-0")
    assert service.reset(session_id=session_id, body=body, idempotency_key="reset-0") == first
    assert adapter.resets == [session_id]

    with pytest.raises(ServeError) as caught:
        _step(service, session_id, "request-0")
    assert caught.value.status == 409
    _step(service, session_id, "request-1")
    assert adapter.calls == 2


def test_invalid_adapter_input_is_a_422_without_committing_step() -> None:
    adapter = FakeAdapter()
    adapter.failure = ValueError("bad state")
    service = _service(adapter)
    session_id = _session(service)
    with pytest.raises(ServeError) as caught:
        _step(service, session_id, "request-0")
    assert caught.value.status == 422
    adapter.failure = None
    result = _step(service, session_id, "request-1")
    assert result["step_id"] == 0


def test_close_waits_for_inflight_step_and_rejects_future_steps() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingAdapter(FakeAdapter):
        def infer(self, request):
            entered.set()
            assert release.wait(timeout=2)
            return super().infer(request)

    adapter = BlockingAdapter()
    service = _service(adapter)
    session_id = _session(service)
    step_thread = threading.Thread(target=_step, args=(service, session_id, "request-0"))
    close_thread = threading.Thread(target=service.close, args=(session_id,))
    step_thread.start()
    assert entered.wait(timeout=2)
    close_thread.start()
    assert close_thread.is_alive()
    release.set()
    step_thread.join(timeout=2)
    close_thread.join(timeout=2)
    assert not step_thread.is_alive()
    assert not close_thread.is_alive()
    assert adapter.resets == [session_id]
    with pytest.raises(ServeError) as caught:
        _step(service, session_id, "request-1", step_id=1)
    assert caught.value.status == 404


def test_session_limit_is_fail_closed() -> None:
    service = _service(maximum_sessions=1)
    _session(service)
    with pytest.raises(ServeError) as caught:
        _session(service)
    assert caught.value.status == 429


def test_failed_close_blocks_work_but_allows_cleanup_retry() -> None:
    class Adapter(FakeAdapter):
        def reset(self, session_id):
            super().reset(session_id)
            if len(self.resets) == 1:
                raise RuntimeError("cleanup failed")

    adapter = Adapter()
    service = _service(adapter, maximum_sessions=1)
    session_id = _session(service)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        service.close(session_id)
    with pytest.raises(ServeError, match="closed"):
        _step(service, session_id, "step-0")
    with pytest.raises(ServeError, match="closed"):
        service.reset(session_id=session_id, body={"request_id": "reset-0"}, idempotency_key="reset-0")
    service.close(session_id)
    assert adapter.resets == [session_id, session_id]
    assert _session(service) != session_id
    with pytest.raises(ServeError, match="unknown"):
        service.close(session_id)


@pytest.mark.parametrize(
    ("path", "headers", "token", "status"),
    [
        ("/v1/sessions", b"Content-Length: 2\r\n", None, 413),
        ("/v1/sessions", b"Content-Length: invalid\r\n", None, 400),
        ("/v1/sessions", b"Transfer-Encoding: chunked\r\n", None, 400),
        ("/v1/sessions", b"", None, 411),
        ("/v1/sessions", b"Content-Length: 2\r\n", "secret", 401),
        ("/unknown", b"Content-Length: 2\r\n", None, 404),
    ],
)
def test_rejected_body_closes_connection(path, headers, token, status) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), PolicyHttpHandler)
    server.service = PolicyHttpService(FakeAdapter(), token=token, max_body_bytes=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(server.server_address, timeout=2) as connection:
            connection.sendall(
                f"POST {path} HTTP/1.1\r\nHost: localhost\r\n".encode()
                + headers
                + b"\r\n{}GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            )
            chunks = []
            while chunk := connection.recv(8192):
                chunks.append(chunk)
        response = b"".join(chunks)
        assert response.startswith(f"HTTP/1.1 {status} ".encode())
        assert b"Connection: close\r\n" in response
        assert response.count(b"HTTP/1.1 ") == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
