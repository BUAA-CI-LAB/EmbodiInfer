"""Send one observation to the versioned HTTP policy API; see docs/en/serving.md.

Uses only the Python standard library. Images and state must match the server's
adapter and checkpoint. This client retrieves actions; it never executes them.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import urllib.request
import uuid
from pathlib import Path


def main() -> None:
    """Open a session, infer one step, print the response, and close the session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--state", type=Path, required=True, help="JSON object of named state fields")
    parser.add_argument("--image", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--robot-id", default="example-client")
    parser.add_argument("--token-env", help="Environment variable containing the Bearer token")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    headers = {}
    if args.token_env:
        headers["Authorization"] = "Bearer " + os.environ[args.token_env]
    state = json.loads(args.state.read_text())
    if not isinstance(state, dict):
        parser.error("--state must contain a JSON object")
    images = []
    for item in args.image:
        name, separator, filename = item.partition("=")
        if not separator or not name or not filename:
            parser.error("--image must be NAME=PATH")
        mime = mimetypes.guess_type(filename)[0]
        if mime not in {"image/jpeg", "image/png"}:
            parser.error("images must have .jpg, .jpeg, or .png extensions")
        images.append((name, mime, Path(filename).read_bytes()))

    def request(method: str, path: str, body: bytes | None = None, **extra: str) -> dict:
        req = urllib.request.Request(endpoint + path, data=body, method=method, headers=headers | extra)
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            return json.load(response)

    capabilities = request("GET", "/v1/capabilities")
    session = request(
        "POST",
        "/v1/sessions",
        json.dumps(
            {
                "schema": "embodiinfer.policy.session.v1",
                "robot_id": args.robot_id,
                "action_space": capabilities["adapter"]["action_space"],
            }
        ).encode(),
        **{"Content-Type": "application/json"},
    )
    path = "/v1/sessions/" + session["session_id"]
    try:
        request_id = "example-" + uuid.uuid4().hex
        metadata = {
            "schema": "embodiinfer.policy.step.v1",
            "session_id": session["session_id"],
            "request_id": request_id,
            "step_id": 0,
            "instruction": args.instruction,
            "state": state,
            "images": [{"name": name, "mime_type": mime} for name, mime, _ in images],
            "metadata": {},
        }
        boundary = "embodiinfer-" + uuid.uuid4().hex
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n'
            "Content-Type: application/json\r\n\r\n".encode()
            + json.dumps(metadata).encode()
            + b"\r\n"
        ]
        for index, (_, mime, data) in enumerate(images):
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="image_{index}"; '
                f'filename="image_{index}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
                + data
                + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode())
        result = request(
            "POST",
            path + "/steps",
            b"".join(parts),
            **{"Content-Type": "multipart/form-data; boundary=" + boundary, "Idempotency-Key": request_id},
        )
        print(json.dumps(result, indent=2))
    except BaseException:
        try:
            request("DELETE", path)
        except Exception:
            logging.warning("Session cleanup failed for %s", session["session_id"], exc_info=True)
        raise
    else:
        request("DELETE", path)


if __name__ == "__main__":
    main()
