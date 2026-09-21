"""Bounded request coalescing shared by synchronous serving transports."""

from __future__ import annotations

import math
import threading
from collections import deque
from concurrent.futures import Future
from time import monotonic
from typing import Any

from .contracts import BatchServingAdapter, ModelResult, RawPolicyRequest, ServeError


class BatchedServingAdapter:
    """Coalesce independent session calls into one adapter batch.

    PolicyService still owns session ordering and idempotency. One lazy worker
    owns the model call, so static graph buffers are never replayed concurrently.
    """

    def __init__(
        self,
        adapter: BatchServingAdapter,
        *,
        max_batch: int,
        max_wait_ms: float = 5.0,
        max_pending: int = 1024,
    ) -> None:
        if not isinstance(adapter, BatchServingAdapter):
            raise ValueError("adapter does not implement cross-session batch serving")
        if type(max_batch) is not int or max_batch < 1 or type(max_pending) is not int or max_pending < 1:
            raise ValueError("batch and queue limits must be positive integers")
        if isinstance(max_wait_ms, bool) or not math.isfinite(max_wait_ms) or max_wait_ms < 0:
            raise ValueError("max_wait_ms must be finite and nonnegative")
        self.adapter = adapter
        self.action_space = adapter.action_space
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.max_pending = max_pending
        self._condition = threading.Condition()
        self._pending: deque[tuple[RawPolicyRequest, Future[ModelResult]]] = deque()
        self._worker: threading.Thread | None = None
        self._closed = False

    def capabilities(self) -> dict[str, Any]:
        """Expose the configured limits, not a claim about observed batch size."""
        return {
            **self.adapter.capabilities(),
            "execution_mode": "dynamic_batch",
            "max_batch_size": self.max_batch,
            "max_wait_ms": self.max_wait_ms,
            "max_pending_requests": self.max_pending,
        }

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Submit an admitted session step and wait for its own result."""
        future: Future[ModelResult] = Future()
        with self._condition:
            if self._closed:
                raise ServeError(503, "server_stopping", "batch service is stopping")
            if len(self._pending) >= self.max_pending:
                raise ServeError(429, "queue_full", "batch service queue is full")
            self._pending.append((request, future))
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name="policy-batch", daemon=True)
                self._worker.start()
            self._condition.notify_all()
        return future.result()

    def reset(self, session_id: str) -> None:
        """Delegate session reset after PolicyService has drained its step."""
        self.adapter.reset(session_id)

    def shutdown(self) -> None:
        """Reject queued work and wait for the single in-flight model call."""
        with self._condition:
            self._closed = True
            while self._pending:
                _, future = self._pending.popleft()
                future.set_exception(ServeError(503, "server_stopping", "batch service is stopping"))
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if self._closed:
                    return
                deadline = monotonic() + self.max_wait_ms / 1000
                while len(self._pending) < self.max_batch and not self._closed:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._closed:
                    return
                batch = [self._pending.popleft() for _ in range(min(self.max_batch, len(self._pending)))]
            try:
                results = self.adapter.infer_batch([request for request, _ in batch])
                if len(results) != len(batch) or any(
                    not isinstance(r, (ModelResult, Exception)) for r in results
                ):
                    raise RuntimeError("batch adapter returned invalid outcomes")
            except Exception as error:
                results = [error] * len(batch)
            for (_, future), result in zip(batch, results, strict=True):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
