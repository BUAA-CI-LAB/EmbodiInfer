"""Exercise the documented client against the real HTTP transport with a fake policy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from embodiinfer.engine.serve.contracts import ModelAction, ModelResult
from embodiinfer.engine.serve.http_server import PolicyHttpHandler, PolicyHttpService


@pytest.mark.parametrize("infer_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_http_client_opens_steps_and_closes_authenticated_session(
    tmp_path: Path, infer_fails: bool, cleanup_fails: bool
) -> None:
    class Adapter:
        action_space = "example.action.v1"

        def __init__(self):
            self.requests = []
            self.resets = []

        def capabilities(self):
            return {"action_space": self.action_space}

        def infer(self, request):
            self.requests.append(request)
            if infer_fails:
                raise ValueError("invalid observation")
            return ModelResult(
                action_space=self.action_space,
                actions=(ModelAction("action_chunk", {"data": [[0.1, 0.2]]}),),
                timing={"policy_ms": 1.0},
            )

        def reset(self, session_id):
            self.resets.append(session_id)
            if cleanup_fails:
                raise RuntimeError("cleanup failed")

    adapter = Adapter()
    server = ThreadingHTTPServer(("127.0.0.1", 0), PolicyHttpHandler)
    server.service = PolicyHttpService(adapter, token="test-token", max_body_bytes=1024 * 1024)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"observation.state": [1.0, 2.0]}))
    image = tmp_path / "front.png"
    image.write_bytes(b"encoded-image-fixture")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "examples/http_client.py"),
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}",
                "--state",
                str(state),
                "--image",
                f"observation.images.front={image}",
                "--instruction",
                "pick up the cube",
                "--token-env",
                "EXAMPLE_TEST_TOKEN",
            ],
            env={**os.environ, "EXAMPLE_TEST_TOKEN": "test-token"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if infer_fails or cleanup_fails:
            assert result.returncode != 0
            expected = "422" if infer_fails else "500"
            assert f"HTTP Error {expected}" in result.stderr.splitlines()[-1]
        else:
            assert result.returncode == 0, result.stderr
        if not infer_fails:
            reply = json.loads(result.stdout)
            assert reply["schema"] == "embodiinfer.policy.step.result.v1"
            assert reply["step_id"] == 0
            assert reply["actions"][0]["values"]["data"] == [[0.1, 0.2]]
        assert len(adapter.requests) == 1
        request = adapter.requests[0]
        assert request.instruction == "pick up the cube"
        assert request.state == {"observation.state": [1.0, 2.0]}
        assert request.images[0].name == "observation.images.front"
        assert request.images[0].data == b"encoded-image-fixture"
        assert request.session_id in adapter.resets
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
