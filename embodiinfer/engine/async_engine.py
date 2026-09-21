"""Asynchronous engine with request-level continuous batching.

The VLA analogue of vLLM's continuous batching is *request-level*, not
token-level: parallel environments emit observations at slightly different
times, and the scheduler coalesces whatever requests are ready within a short
window (``max_wait_ms``) into one padded batch. Because each request's compute
shape is fixed, forming the batch is trivial compared to the ragged, growing
sequences vLLM must manage.

Two scheduling modes share the same request queue and windowing:

  * default (``pipeline=False``) — each window is run through ``core.execute``
    (prefill then denoise, synchronously);
  * pipelined (``pipeline=True``) — a depth-1 software pipeline that overlaps the
    current window's denoise with the *next* window's prefill on two CUDA streams
    (``core._pipeline_step``). This only helps when observations actually arrive
    staggered (denoise of one window running while the next window is still
    filling); with a CUDA graph absent it degrades to the synchronous behaviour.

The pipelined mode is additive — the synchronous path is unchanged — so enabling
it never alters results, only timing.
"""

from __future__ import annotations

import asyncio
import contextlib

from ..exceptions import UnsupportedRecurrentModeError
from ..types import ActionChunk, Observation, Request, SampleParams
from .core import EngineCore

_QueueItem = tuple[Request, "asyncio.Future"]


class AsyncEngine:
    """Request-level continuous batching over one :class:`EngineCore`.

    Parallel environments emit observations at slightly different times, so the scheduler
    coalesces whatever requests are ready inside a short window (``max_wait_ms``) into one
    padded batch. Each request's compute shape is fixed, which makes forming the batch
    trivial next to the ragged, growing sequences a token-level scheduler has to manage.

    With ``pipeline=True`` one window's prefill overlaps the previous window's denoising on
    separate CUDA streams, which pays off only when requests arrive spread out over time.
    Recurrent policies are rejected: this scheduler does not yet preserve session ordering.
    """

    def __init__(self, core: EngineCore, pipeline: bool = False):
        if core.policy.is_recurrent:
            raise UnsupportedRecurrentModeError(
                "AsyncEngine batching/pipeline does not yet preserve recurrent session ordering"
            )
        self.core = core
        self.cfg = core.config
        self.pipeline = pipeline
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None
        self._req_counter = 0

    async def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    async def generate(self, observation: Observation, params: SampleParams | None = None) -> ActionChunk:
        """Submit one observation; returns when its batch has been executed."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._req_counter += 1
        req = Request(
            request_id=f"r{self._req_counter}",
            observation=observation,
            params=params or SampleParams(),
        )
        await self._queue.put((req, fut))
        return await fut

    async def _run_loop(self) -> None:
        if self.pipeline:
            await self._run_loop_pipelined()
        else:
            await self._run_loop_sync()

    # ---- default: one synchronous execute per window ------------------------
    async def _run_loop_sync(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect_window(loop, self._queue.get())
            await self._execute(batch, loop)

    async def _execute(self, batch: list[_QueueItem], loop: asyncio.AbstractEventLoop) -> None:
        reqs = [r for r, _ in batch]
        futs = [f for _, f in batch]
        num_steps = reqs[0].params.num_steps  # batch assumed homogeneous in steps
        collated = self.core.policy.collate([r.observation for r in reqs], [r.request_id for r in reqs])
        try:
            results = await loop.run_in_executor(None, self.core.execute, collated, num_steps)
            self._set(futs, results)
        except Exception as exc:  # propagate to all waiters in the batch
            self._fail(futs, exc)

    # ---- pipelined: overlap window i's denoise with window i+1's prefill -----
    async def _run_loop_pipelined(self) -> None:
        loop = asyncio.get_running_loop()
        max_wait = self.cfg.max_wait_ms / 1000.0
        # One window may be staged (prefilled) but not yet denoised; its futures /
        # step count travel alongside it until the next window overlaps its denoise.
        staged = None
        prev_futs: list[asyncio.Future] | None = None
        prev_steps = 0
        while True:
            # Get the next window's first request. If a batch is already staged,
            # don't block forever — flush it if nothing new arrives within max_wait.
            try:
                first = await (
                    asyncio.wait_for(self._queue.get(), max_wait) if staged is not None else self._queue.get()
                )
            except asyncio.TimeoutError:
                await self._flush(loop, staged, prev_futs, prev_steps)
                staged, prev_futs = None, None
                continue
            window = await self._collect_window(loop, None, first=first)
            reqs = [r for r, _ in window]
            futs = [f for _, f in window]
            num_steps = reqs[0].params.num_steps
            collated = self.core.policy.collate([r.observation for r in reqs], [r.request_id for r in reqs])
            try:
                prev_actions, staged = await loop.run_in_executor(
                    None, self.core._pipeline_step, staged, collated, num_steps, None
                )
            except Exception as exc:
                self._fail(futs, exc)
                if prev_futs is not None:
                    self._fail(prev_futs, exc)
                staged, prev_futs = None, None
                continue
            if prev_actions is not None and prev_futs is not None:
                self._set(prev_futs, prev_actions)
            prev_futs, prev_steps = futs, num_steps

    async def _flush(
        self, loop: asyncio.AbstractEventLoop, staged, futs: list[asyncio.Future] | None, num_steps: int
    ) -> None:
        """Drain a staged (prefilled) batch: denoise it with no concurrent prefill."""
        if staged is None or futs is None:
            return
        try:
            actions, _ = await loop.run_in_executor(
                None, self.core._pipeline_step, staged, None, num_steps, None
            )
            self._set(futs, actions)
        except Exception as exc:
            self._fail(futs, exc)

    # ---- shared windowing + future helpers ----------------------------------
    async def _collect_window(self, loop, get_first, first=None) -> list[_QueueItem]:
        """Coalesce ready requests into one window (up to ``max_batch_size`` or
        ``max_wait_ms``). Pass ``get_first`` (an awaitable) to block for the first
        item, or ``first`` to seed the window with an already-taken item."""
        max_batch = self.cfg.max_batch_size
        max_wait = self.cfg.max_wait_ms / 1000.0
        batch: list[_QueueItem] = [first] if first is not None else [await get_first]
        deadline = loop.time() + max_wait
        while len(batch) < max_batch:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        return batch

    @staticmethod
    def _set(futs: list[asyncio.Future], results: list[ActionChunk]) -> None:
        for fut, res in zip(futs, results):
            if not fut.done():
                fut.set_result(res)

    @staticmethod
    def _fail(futs: list[asyncio.Future], exc: Exception) -> None:
        for fut in futs:
            if not fut.done():
                fut.set_exception(exc)
