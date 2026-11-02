"""Shared-parameter text access for an existing ActiveVLN navigation adapter."""

from __future__ import annotations

import io
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image, UnidentifiedImageError

from ...engine.serve.contracts import (
    BatchServingAdapter,
    ModelResult,
    RawPolicyRequest,
    ServeError,
    ServingAdapter,
)
from ...engine.serve.text import ExecutionGate, TextRequest
from ...exceptions import SessionCancelledError
from .text_activevln import ActiveVLNTextDecoder

if TYPE_CHECKING:
    from .modeling_activevln import ActiveVLNPolicy


class ActiveVLNSharedTextAdapter:
    """Wrap navigation and raw text around exactly one policy object.

    The navigation adapter must expose its owning ``policy``. All model calls
    must pass through this wrapper. Text has private KV and yields after each
    token to waiting navigation. Only one text request is admitted at a time.
    This advertises raw generation, not reliable planning or tool use.
    """

    def __init__(
        self, adapter: ServingAdapter, policy: ActiveVLNPolicy, *, decode_graphs: bool = False
    ) -> None:
        if type(decode_graphs) is not bool:
            raise ValueError("decode_graphs must be a boolean")
        if getattr(adapter, "policy", None) is not policy:
            raise ValueError("text and navigation must own the same policy object")
        self.adapter, self.policy = adapter, policy
        self.action_space = adapter.action_space
        self.decoder = ActiveVLNTextDecoder(policy)
        self._gate = ExecutionGate()
        self._text_slot = threading.BoundedSemaphore(1)
        self._identity = uuid.uuid4().hex
        self._version = policy.policy_version
        self._closed = False
        if decode_graphs:
            with self._execution():
                self.decoder.enable_decode_graphs()

    def capabilities(self) -> dict[str, Any]:
        """Attest shared parameter ownership without a tool-use accuracy claim."""
        value = self.adapter.capabilities()
        if self.policy.policy_version == self._version and not self._closed:
            value = {
                **value,
                "shared_text": {
                    "schema": "vvla.shared_text.v1",
                    "model_instance": self._identity,
                    "shared_parameters": True,
                    "max_input_tokens": 4096,
                    "max_new_tokens": 512,
                    "max_image_pixels": 640 * 480,
                    "max_concurrent_text": 1,
                },
            }
        return value

    @contextmanager
    def _execution(self, *, priority: bool = False, deadline: float | None = None) -> Iterator[None]:
        with self._gate.enter(priority=priority, deadline=deadline):
            if self._closed:
                raise ServeError(503, "server_stopping", "shared model service is stopping")
            device = next(self.policy.parameters()).device
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
                        yield
                else:
                    yield

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Preserve the navigation adapter's transaction and output contract."""
        with self._execution(priority=True):
            return self.adapter.infer(request)

    def infer_batch(self, requests: Sequence[RawPolicyRequest]) -> Sequence[ModelResult | Exception]:
        """Keep native batch execution behind the same GPU ownership gate."""
        if not isinstance(self.adapter, BatchServingAdapter):
            if len(requests) != 1:
                raise ValueError("navigation adapter does not support batching")
            return [self.infer(requests[0])]
        with self._execution(priority=True):
            return self.adapter.infer_batch(requests)

    def reset(self, session_id: str) -> None:
        """Reset only the navigation session; independent text KV is unaffected."""
        with self._execution(priority=True):
            self.adapter.reset(session_id)

    def _check_identity(self, request: TextRequest) -> None:
        if request.model_instance != self._identity or self.policy.policy_version != self._version:
            raise ServeError(
                409, "stale_model", "model identity changed; recreate the serving adapter after refit"
            )

    def infer_text(self, request: TextRequest) -> dict[str, Any]:
        """Generate bounded raw text with separate KV and navigation priority."""
        self._check_identity(request)
        if not self._text_slot.acquire(blocking=False):
            raise ServeError(429, "text_busy", "one text request is already in flight")
        started = time.perf_counter()
        try:
            images = []
            if request.image is not None:
                try:
                    with Image.open(io.BytesIO(request.image)) as frame:
                        if frame.format not in {"PNG", "JPEG"} or frame.width * frame.height > 640 * 480:
                            raise ValueError("text requires JPEG/PNG with at most 640x480 pixels")
                        images = [frame.convert("RGB")]
                except (UnidentifiedImageError, OSError) as error:
                    raise ValueError("invalid text image") from error
            with self._execution(deadline=request.deadline):
                self._check_identity(request)
                state = self.decoder.prepare(
                    request.messages,
                    images=images,
                    max_new_tokens=request.max_new_tokens,
                    max_input_tokens=4096,
                    deadline=request.deadline,
                )
            while True:
                with self._execution(deadline=request.deadline):
                    self._check_identity(request)
                    more = self.decoder.step(state)
                    if not more:
                        result = self.decoder.result(state)
                        break
            return {
                "schema": "vvla.text.response.v1",
                "request_id": request.request_id,
                "model_instance": self._identity,
                "text": result.text,
                "token_ids": list(result.token_ids),
                "input_tokens": result.input_tokens,
                "stop_reason": result.stop_reason,
                "elapsed_s": time.perf_counter() - started,
            }
        except SessionCancelledError as error:
            raise ServeError(408, "text_deadline", str(error)) from error
        except ValueError as error:
            raise ServeError(400, "invalid_text_request", str(error)) from error
        finally:
            self._text_slot.release()

    def shutdown(self) -> None:
        """Reject new calls and invalidate text state after the current GPU step."""
        with self._gate.enter(priority=True):
            self._closed = True
            self.decoder.invalidate()
        close = getattr(self.adapter, "shutdown", None)
        if close is not None:
            close()
