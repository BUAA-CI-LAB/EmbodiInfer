"""Transport-neutral session and step semantics for policy serving."""

from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .contracts import (
    ModelAction,
    RawPolicyRequest,
    ServeError,
    ServingAdapter,
    payload_fingerprint,
    request_fingerprint,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ServeError(400, "invalid_request", f"{name} is invalid")
    return value


@dataclass
class _CachedResponse:
    fingerprint: str
    response: dict[str, Any]


@dataclass
class _Session:
    session_id: str
    action_space: str
    metadata: dict[str, Any]
    revision: int = 0
    next_step_id: int = 0
    closed: bool = False
    responses: OrderedDict[str, _CachedResponse] = field(default_factory=OrderedDict)
    seen_requests: OrderedDict[str, str] = field(default_factory=OrderedDict)
    reset_requests: OrderedDict[str, _CachedResponse] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class PolicyService:
    """Serve policy operations independently of their network transport."""

    def __init__(
        self,
        adapter: ServingAdapter,
        *,
        maximum_sessions: int = 1024,
        idempotency_cache_size: int = 1024,
    ) -> None:
        if maximum_sessions < 1 or idempotency_cache_size < 1:
            raise ValueError("session and idempotency limits must be positive")
        self.adapter = adapter
        self.maximum_sessions = maximum_sessions
        self.idempotency_cache_size = idempotency_cache_size
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()

    @property
    def action_space(self) -> str:
        return self.adapter.action_space

    def open_session(self, request: Mapping[str, Any]) -> dict[str, Any]:
        # Schema strings are the wire contract with clients: every payload is
        # tagged "embodiinfer.<area>.<name>.v<version>" and is validated
        # verbatim. Bumping a shape requires a new ".vN" suffix, never a silent
        # change to an existing one.
        if request.get("schema") != "embodiinfer.policy.session.v1":
            raise ServeError(400, "unsupported_schema", "unsupported session schema")
        robot_id = _identifier(request.get("robot_id"), "robot_id")
        action_space = _identifier(request.get("action_space"), "action_space")
        if action_space != self.action_space:
            raise ServeError(400, "invalid_action_space", f"server expects {self.action_space!r}")
        metadata = request.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ServeError(400, "invalid_request", "metadata must be an object")
        session_id = f"session-{uuid.uuid4().hex}"
        session = _Session(session_id=session_id, action_space=action_space, metadata=dict(metadata))
        with self._sessions_lock:
            if len(self._sessions) >= self.maximum_sessions:
                raise ServeError(429, "too_many_sessions", "server session limit reached")
            self._sessions[session_id] = session
        return {
            "schema": "embodiinfer.policy.session.v1",
            "session_id": session_id,
            "session_revision": session.revision,
            "robot_id": robot_id,
            "action_space": action_space,
        }

    def step(self, request: RawPolicyRequest) -> dict[str, Any]:
        """Execute one ordered, idempotent policy step."""

        session = self._sessions.get(request.session_id)
        if session is None:
            raise ServeError(404, "session_not_found", "session_id is unknown")
        with session.lock:
            if session.closed:
                raise ServeError(404, "session_not_found", "session_id is closed")
            fingerprint = request_fingerprint(request)
            cached = session.responses.get(request.request_id)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise ServeError(
                        409,
                        "idempotency_conflict",
                        "request_id was reused with different content",
                    )
                session.responses.move_to_end(request.request_id)
                return cached.response
            if request.request_id in session.seen_requests:
                raise ServeError(
                    409,
                    "stale_idempotency_key",
                    "request_id belongs to an earlier session revision",
                )
            if request.step_id < session.next_step_id:
                raise ServeError(409, "step_id_too_old", "step_id already committed")
            if request.step_id != session.next_step_id:
                raise ServeError(409, "out_of_order_step", "step_id must be monotonic")
            start = perf_counter() * 1000.0
            try:
                result = self.adapter.infer(request)
            except ServeError:
                raise
            except (TypeError, ValueError) as error:
                raise ServeError(422, "invalid_observation", str(error)) from error
            response = {
                "schema": "embodiinfer.policy.step.result.v1",
                "request_id": request.request_id,
                "session_id": request.session_id,
                "step_id": request.step_id,
                "session_revision": session.revision + 1,
                "action_space": result.action_space,
                "actions": [self._serialize_action(action) for action in result.actions],
                "timing": {**result.timing, "policy_ms": perf_counter() * 1000.0 - start},
                "policy_revision": result.policy_revision,
            }
            session.revision += 1
            session.next_step_id = request.step_id + 1
            self._remember(session.seen_requests, request.request_id, fingerprint)
            self._remember(
                session.responses,
                request.request_id,
                _CachedResponse(fingerprint, response),
            )
            return response

    def reset(self, session_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise ServeError(404, "session_not_found", "session_id is unknown")
        request_id = _identifier(body.get("request_id"), "request_id")
        with session.lock:
            if session.closed:
                raise ServeError(404, "session_not_found", "session_id is closed")
            fingerprint = payload_fingerprint(body)
            cached = session.reset_requests.get(request_id)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise ServeError(
                        409,
                        "idempotency_conflict",
                        "reset request_id was reused with different content",
                    )
                session.reset_requests.move_to_end(request_id)
                return cached.response
            self.adapter.reset(session_id)
            session.revision += 1
            session.next_step_id = 0
            session.responses.clear()
            response = {
                "schema": "embodiinfer.policy.reset.v1",
                "session_id": session_id,
                "session_revision": session.revision,
            }
            self._remember(
                session.reset_requests,
                request_id,
                _CachedResponse(fingerprint, response),
            )
            return response

    def close(self, session_id: str) -> None:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ServeError(404, "session_not_found", "session_id is unknown")
        with session.lock:
            if self._sessions.get(session_id) is not session:
                raise ServeError(404, "session_not_found", "session_id is closed")
            # Block new work immediately, but allow close to retry failed cleanup.
            session.closed = True
            self.adapter.reset(session_id)
            with self._sessions_lock:
                if self._sessions.get(session_id) is session:
                    self._sessions.pop(session_id)
            session.responses.clear()
            session.seen_requests.clear()
            session.reset_requests.clear()

    def _remember(self, cache: OrderedDict, key: str, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.idempotency_cache_size:
            cache.popitem(last=False)

    @staticmethod
    def _serialize_action(action: ModelAction) -> dict[str, Any]:
        return {"type": action.kind, "values": dict(action.values)}


__all__ = ["PolicyService"]
