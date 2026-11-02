"""HTTP serving entry point for EmbodiInfer policies.

The protocol is model- and robot-agnostic:

* Clients send raw observations (state + encoded image bytes) as multipart form.
* A model adapter converts raw data into model-native tensors, runs inference,
  and returns raw actions.
* Deploy-layer or downstream code performs the model-to-robot conversion.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import re
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .batching import BatchedServingAdapter
from .contracts import (
    ServeError,
    ServingAdapter,
    parse_json,
    parse_step,
)
from .factory import add_policy_arguments, build_serving_adapter
from .service import PolicyService

_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_LOG = logging.getLogger(__name__)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ServeError(400, "invalid_request", f"{name} is invalid")
    return value


def _json(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


class PolicyHttpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        try:
            if self.path.rstrip("/") == "/healthz":
                self._write_json(
                    200,
                    {
                        "schema": "embodiinfer.health.v1",
                        "status": "ok",
                        "service": "embodiinfer-http",
                    },
                )
                return
            self._require_token_if_needed()
            if self.path.rstrip("/") == "/v1/capabilities":
                self._write_json(
                    200,
                    {
                        "schema": "embodiinfer.policy.capabilities.v1",
                        "server": "embodiinfer-http",
                        "adapter": self.server.service.adapter.capabilities(),
                    },
                )
                return
            raise ServeError(404, "not_found", "endpoint not found")
        except ServeError as error:
            self._write_error(error)
        except Exception:
            self._write_internal_error()

    def do_POST(self) -> None:
        try:
            self._require_token_if_needed()
            parsed = urlsplit(self.path)
            path = parsed.path.rstrip("/")
            if path == "/v1/sessions":
                payload = parse_json(self._read_body(), name="session body")
                session = self.server.service.open_session(payload)
                self._write_json(200, session)
                return
            if path.endswith("/steps"):
                parts = path.split("/")
                if len(parts) != 5 or parts[1] != "v1" or parts[2] != "sessions" or parts[4] != "steps":
                    raise ServeError(404, "not_found", "endpoint not found")
                session_id = _identifier(parts[3], "session_id")
                content_type = self.headers.get("content-type", "")
                idem = self.headers.get("idempotency-key")
                result = self.server.service.step(
                    session_id=session_id,
                    content_type=content_type,
                    body=self._read_body(),
                    idempotency_key=idem,
                )
                self._write_json(200, result)
                return
            if path.endswith("/reset"):
                parts = path.split("/")
                if len(parts) != 5 or parts[1] != "v1" or parts[2] != "sessions" or parts[4] != "reset":
                    raise ServeError(404, "not_found", "endpoint not found")
                session_id = _identifier(parts[3], "session_id")
                payload = parse_json(self._read_body(), name="reset body")
                result = self.server.service.reset(
                    session_id=session_id, body=payload, idempotency_key=self.headers.get("idempotency-key")
                )
                self._write_json(200, result)
                return
            raise ServeError(404, "not_found", "endpoint not found")
        except ServeError as error:
            self._write_error(error)
        except Exception:
            self._write_internal_error()

    def do_DELETE(self) -> None:
        try:
            self._require_token_if_needed()
            parsed = urlsplit(self.path)
            path = parsed.path.rstrip("/")
            parts = path.split("/")
            if len(parts) != 4 or parts[1] != "v1" or parts[2] != "sessions":
                raise ServeError(404, "not_found", "endpoint not found")
            session_id = _identifier(parts[3], "session_id")
            self.server.service.close(session_id)
            self._write_json(200, {"schema": "embodiinfer.policy.session.close.v1", "ok": True})
        except ServeError as error:
            self._write_error(error)
        except Exception:
            self._write_internal_error()

    def _read_body(self) -> bytes:
        if self.headers.get("transfer-encoding") is not None:
            raise ServeError(400, "unsupported_transfer_encoding", "chunked request bodies are unsupported")
        raw_length = self.headers.get("content-length")
        if raw_length is None:
            raise ServeError(411, "length_required", "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ServeError(400, "invalid_content_length", "Content-Length must be an integer") from exc
        if length < 0:
            raise ServeError(400, "invalid_content_length", "Content-Length must be non-negative")
        max_bytes = self.server.service.max_body_bytes
        if length > max_bytes:
            raise ServeError(413, "request_too_large", "request body exceeds server limit")
        return self.rfile.read(length)

    def _write_json(self, code: int, payload: Mapping[str, Any]) -> None:
        raw = _json(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def _write_error(self, error: ServeError) -> None:
        # Rejected requests may still have unread body bytes on the connection.
        self.close_connection = True
        self._write_json(
            error.status, {"schema": "embodiinfer.error.v1", "code": error.code, "message": str(error)}
        )

    def _write_internal_error(self) -> None:
        _LOG.exception("unhandled EmbodiInfer HTTP request failure")
        self._write_error(ServeError(500, "internal_error", "internal server error"))

    def log_message(self, fmt: str, *args: object) -> None:
        # reduce noisy default logging; keep handler-compatible for local tailing
        pass

    def _require_token_if_needed(self) -> None:
        expected = self.server.service.token
        if expected is None:
            return
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        supplied = header[len(prefix) :] if header.startswith(prefix) else ""
        if not hmac.compare_digest(supplied, expected):
            raise ServeError(401, "unauthorized", "missing or invalid Bearer token")


class PolicyHttpService:
    """Adapt HTTP request values to the transport-neutral policy service."""

    def __init__(
        self,
        adapter: ServingAdapter,
        *,
        token: str | None,
        max_body_bytes: int,
        maximum_images: int = 8,
        maximum_image_bytes: int = 16 * 1024 * 1024,
        maximum_sessions: int = 1024,
        idempotency_cache_size: int = 1024,
    ) -> None:
        self.token = token
        self.max_body_bytes = max_body_bytes
        self.maximum_images = maximum_images
        self.maximum_image_bytes = maximum_image_bytes
        self.policy_service = PolicyService(
            adapter,
            maximum_sessions=maximum_sessions,
            idempotency_cache_size=idempotency_cache_size,
        )

    @property
    def adapter(self) -> ServingAdapter:
        return self.policy_service.adapter

    @property
    def action_space(self) -> str:
        return self.policy_service.action_space

    def open_session(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.policy_service.open_session(request)

    def step(
        self,
        *,
        session_id: str,
        content_type: str,
        body: bytes,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        request = parse_step(
            content_type=content_type,
            body=body,
            route_session_id=session_id,
            idempotency_key=idempotency_key,
            maximum_images=self.maximum_images,
            maximum_image_bytes=self.maximum_image_bytes,
        )
        return self.policy_service.step(request)

    def reset(
        self, *, session_id: str, body: Mapping[str, Any], idempotency_key: str | None
    ) -> dict[str, Any]:
        if idempotency_key is None:
            raise ServeError(400, "invalid_request", "Idempotency-Key header is required for reset")
        request_id = _identifier(body.get("request_id"), "request_id")
        if request_id != idempotency_key:
            raise ServeError(400, "idempotency_mismatch", "Idempotency-Key must equal request_id")
        return self.policy_service.reset(session_id, body)

    def close(self, session_id: str) -> None:
        self.policy_service.close(session_id)


def create_http_server(service: PolicyHttpService, *, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), PolicyHttpHandler)
    server.service = service  # dynamic attribute used by handler
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    add_policy_arguments(parser)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=None, help="optional Bearer token")
    parser.add_argument("--max-body-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-image-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=1024)
    parser.add_argument("--idempotency-cache-size", type=int, default=1024)
    args = parser.parse_args(argv)

    try:
        adapter = build_serving_adapter(args)
    except ValueError as error:
        parser.error(str(error))
    service = PolicyHttpService(
        adapter,
        token=args.token,
        max_body_bytes=args.max_body_bytes,
        maximum_images=args.max_images,
        maximum_image_bytes=args.max_image_bytes,
        maximum_sessions=args.max_sessions,
        idempotency_cache_size=args.idempotency_cache_size,
    )
    server = create_http_server(service, host=args.host, port=args.port)
    server.service = service  # dynamic attribute used by the request handler
    print(
        f"[embodiinfer-http] policy={args.policy} device={args.device} "
        f"action_space={service.action_space} listen={args.host}:{args.port}"
    )
    try:
        server.serve_forever()
    finally:
        if isinstance(adapter, BatchedServingAdapter):
            adapter.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
