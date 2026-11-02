"""Opt-in raw text serving and cooperative execution, independent of model type."""

from __future__ import annotations

import base64
import binascii
import math
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .contracts import ServeError, _identifier


@dataclass(frozen=True)
class TextRequest:
    """A bounded conversation borrowing an attested navigation model instance."""

    request_id: str
    model_instance: str
    messages: list[dict[str, Any]]
    image: bytes | None
    max_new_tokens: int
    deadline: float


@runtime_checkable
class TextServingAdapter(Protocol):
    """Optional text generation; the adapter owns model identity and GPU scheduling."""

    def infer_text(self, request: TextRequest) -> Mapping[str, Any]:
        """Return raw text and usage without parsing or executing tools."""
        ...


def parse_text_request(value: Mapping[str, Any]) -> TextRequest:
    """Validate the JSON text transport before any model work or image decoding."""
    if value.get("schema") != "vvla.text.request.v1":
        raise ServeError(400, "invalid_schema", "expected vvla.text.request.v1")
    request_id = _identifier(value.get("request_id"), "request_id")
    identity = _identifier(value.get("model_instance"), "model_instance")
    timeout = value.get("timeout_s", 60)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ServeError(400, "invalid_timeout", "timeout_s must be in (0, 60]")
    budget = value.get("max_new_tokens", 64)
    if type(budget) is not int or not 1 <= budget <= 512:
        raise ServeError(400, "invalid_budget", "max_new_tokens must be in [1, 512]")
    messages = value.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ServeError(400, "invalid_messages", "expected 1 to 64 messages")
    text_chars = image_count = 0
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ServeError(400, "invalid_messages", "each message needs role and content")
        if not isinstance(message["role"], str) or message["role"] not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ServeError(400, "invalid_messages", "unsupported message role")
        content = message["content"]
        if isinstance(content, str):
            text_chars += len(content)
            continue
        if not isinstance(content, list) or not content:
            raise ServeError(400, "invalid_messages", "content must be text or nonempty blocks")
        for block in content:
            if isinstance(block, dict) and block == {"type": "image"}:
                image_count += 1
            elif (
                isinstance(block, dict)
                and set(block) == {"type", "text"}
                and block["type"] == "text"
                and isinstance(block["text"], str)
            ):
                text_chars += len(block["text"])
            else:
                raise ServeError(400, "invalid_messages", "unsupported content block")
    if text_chars > 65536:
        raise ServeError(413, "text_too_large", "conversation exceeds character budget")
    encoded = value.get("image_base64")
    image = None
    if encoded is not None:
        if not isinstance(encoded, str) or not 0 < len(encoded) <= 8 * 1024**2:
            raise ServeError(413, "image_too_large", "invalid encoded image size")
        try:
            image = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ServeError(400, "invalid_image", "image is not valid base64") from error
    if image_count != int(image is not None):
        raise ServeError(400, "invalid_messages", "image blocks must match the single supplied image")
    return TextRequest(request_id, identity, messages, image, budget, time.monotonic() + timeout)


class ExecutionGate:
    """Serialize shared execution and prefer waiting navigation between text tokens.

    Admission does not preempt an in-flight prefill or token. Owners must bound
    those operations and use one GPU stream across all callers.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._busy = False
        self._priority_waiters = 0

    @contextmanager
    def enter(self, *, priority: bool = False, deadline: float | None = None) -> Iterator[None]:
        """Acquire execution ownership, respecting an absolute monotonic deadline."""
        with self._condition:
            if priority:
                self._priority_waiters += 1
            try:
                while self._busy or (not priority and self._priority_waiters):
                    wait = None if deadline is None else deadline - time.monotonic()
                    if wait is not None and wait <= 0:
                        raise ServeError(408, "text_deadline", "execution admission deadline expired")
                    self._condition.wait(wait)
                if deadline is not None and time.monotonic() >= deadline:
                    raise ServeError(408, "text_deadline", "execution admission deadline expired")
                self._busy = True
            finally:
                if priority:
                    self._priority_waiters -= 1
                    self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()
