"""The synchronous execution core.

``EngineCore.execute`` runs one batch through a VLA the way the engine is designed to:
encode the multimodal prefix once (eager prefill), then hand the reused prefix to the
policy's :class:`~embodiinfer.policies.decoder.ActionDecoder` to produce the action chunk —
an N-step flow denoise loop, a single categorical forward, or a diffusion sampler, all
behind the same contract and optionally CUDA-graph-captured. Batches are padded up to a
bucket size so a single captured graph serves a range of real batch sizes. The core is
model-agnostic: it never branches on the paradigm.

This is the object a rollout/generation backend or a serving frontend calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..exceptions import SessionRequiredError, UnsupportedRecurrentModeError
from ..policies.base import PolicyBatch, VLAPolicy
from ..types import ActionChunk, DecodeTrace, SessionKey
from ..utils import now_ns, resolve_device, sync_if_cuda, torch_dtype
from .config import EngineConfig
from .graph import GraphManager
from .session import SessionLease, SessionStore


@dataclass
class _Staged:
    """A batch after prefill, ready for the denoise stage. Separating prefill and
    denoise lets consecutive batches be pipelined (one's denoise overlaps the
    next's prefill)."""

    prefix: object
    x: torch.Tensor | None
    bucket: int
    real_B: int
    request_ids: list[str]
    num_steps: int


class _StageTimer:
    """Wall-clock e2e timing plus stream-accurate CUDA stage timing."""

    def __init__(self, device: torch.device):
        self.device = device
        self._wall_start_ns = now_ns()
        self._prefill_ns: int | None = None
        self._start_event: torch.cuda.Event | None = None
        self._prefill_event: torch.cuda.Event | None = None
        if device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._prefill_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record(torch.cuda.current_stream(device))

    def mark_prefill_complete(self) -> None:
        if self._prefill_event is not None:
            self._prefill_event.record(torch.cuda.current_stream(self.device))
        else:
            self._prefill_ns = now_ns()

    def finish(self) -> tuple[float, dict[str, float]]:
        if self._start_event is not None and self._prefill_event is not None:
            done_event = torch.cuda.Event(enable_timing=True)
            done_event.record(torch.cuda.current_stream(self.device))
            done_event.synchronize()
            prefill_ms = self._start_event.elapsed_time(self._prefill_event)
            decode_ms = self._prefill_event.elapsed_time(done_event)
            done_ns = now_ns()
        else:
            done_ns = now_ns()
            if self._prefill_ns is None:
                raise RuntimeError("prefill completion was not recorded")
            prefill_ms = (self._prefill_ns - self._wall_start_ns) / 1e6
            decode_ms = (done_ns - self._prefill_ns) / 1e6
        e2e_ms = (done_ns - self._wall_start_ns) / 1e6
        return e2e_ms, {"prefill_ms": prefill_ms, "decode_ms": decode_ms, "e2e_ms": e2e_ms}


class EngineCore:
    """The synchronous execution core: encode the prefix once, then decode.

    A batch is encoded into a multimodal prefix eagerly, and the reused prefix is handed
    to the policy's ``ActionDecoder`` to produce action chunks -- an N-step flow denoise
    loop, a single categorical forward, or a diffusion sampler, all behind the same
    contract and optionally CUDA-graph captured. Batches are padded to a bucket size so
    one captured graph serves a range of real batch sizes. The core never branches on the
    paradigm.

    This is what a rollout backend or a serving frontend calls. Recurrent policies run as
    eager, single-item transactions through the session store; stateless policies can
    also be driven concurrently by :class:`~embodiinfer.engine.async_engine.AsyncEngine`.
    """

    def __init__(self, policy: VLAPolicy, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.device = resolve_device(self.config.device)
        self.dtype = self._resolve_dtype(policy)
        self.policy = policy.to(self.device)
        # Auto preserves each checkpoint tensor's dtype on CUDA. A uniform cast
        # can change mixed-precision model outputs as well as memory usage.
        if self.config.dtype != "auto" or self.device.type != "cuda":
            self.policy = self.policy.to(self.dtype)
        self.policy.eval()
        self.pcfg = policy.config
        self._sessions = SessionStore()
        if self.policy.is_recurrent:
            unsupported = []
            if self.config.max_batch_size != 1:
                unsupported.append("max_batch_size must be 1")
            if self.config.use_cuda_graph and not self.policy.manages_cuda_graph:
                unsupported.append("use_cuda_graph must be False")
            if self.config.capture_full_loop:
                unsupported.append("capture_full_loop must be False")
            if unsupported:
                raise UnsupportedRecurrentModeError(
                    "recurrent policies currently require eager B=1 execution: " + ", ".join(unsupported)
                )
            if self.policy.manages_cuda_graph:
                self.policy.configure_runtime(use_cuda_graph=self.config.use_cuda_graph)
        self._graphs: GraphManager | None = None
        # CUDA graphs need a CUDA device and a policy that opts in (static prefix).
        if self.config.use_cuda_graph and self.device.type == "cuda" and self.policy.supports_cuda_graph:
            self._graphs = GraphManager(
                self.policy, self.device, self.dtype, full_loop=self.config.capture_full_loop
            )
        # lazily-created two streams + isolated MemPool for execute_pipelined()
        self._stream_a: torch.cuda.Stream | None = None
        self._stream_b: torch.cuda.Stream | None = None
        self._prefill_pool = None

    @property
    def policy_version(self) -> int:
        """Read-only version stamped by the successful weight-sync path."""
        return self.policy.policy_version

    def _resolve_dtype(self, policy: VLAPolicy) -> torch.dtype:
        """Use FP32 on CPU, the policy's execution dtype for CUDA auto, or an explicit cast."""
        if self.device.type != "cuda":
            return torch.float32
        if self.config.dtype == "auto":
            return policy.execution_dtype
        return torch_dtype(self.config.dtype)

    def _integrate(self, x: torch.Tensor | None, prefix, num_steps: int, bucket: int) -> torch.Tensor:
        """Produce the action chunk from the staged prefix via the policy's decoder.

        Model-agnostic: for a flow policy the decoder runs the N-step denoise loop
        (with the active CUDA-graph mode); for a single-pass policy it does one
        forward. The engine only supplies the staged state, the graph manager, and
        the bucket — it does not know how the chunk is produced."""
        return self.policy.decoder.produce_chunk(x, prefix, num_steps, bucket, self._graphs)

    def _prefill(
        self, batch: PolicyBatch, num_steps: int | None, generator: torch.Generator | None
    ) -> _Staged:
        """Compute-bound stage: pad, move to device, encode the multimodal prefix
        once, and draw the initial noise. Padding + device/dtype move are the
        policy's / batch's concern; the engine stays agnostic to the batch layout."""
        real_B = batch.batch_size
        num_steps = num_steps or self.config.num_steps or self.pcfg.default_num_steps
        bucket = self.config.resolve_bucket(real_B) if self._graphs is not None else real_B
        batch = self.policy.pad(batch, bucket)
        batch = batch.to(self.device, self.dtype)
        prefix = self.policy.encode_prefix(batch)  # eager prefill, once
        # The decoder stages its own initial state: flow seeds noise, a single-pass
        # policy seeds nothing. The engine stays agnostic to what generation needs.
        x = self.policy.decoder.init_state(bucket, generator)
        return _Staged(prefix, x, bucket, real_B, list(batch.request_ids), num_steps)

    def _pack(
        self,
        x: torch.Tensor,
        st: _Staged,
        latency_ms: float,
        timing: dict[str, float] | None = None,
    ) -> list[ActionChunk]:
        actions = x[: st.real_B].float().cpu()  # whole-batch latency; per-request share reported by caller
        timing = dict(timing or {"e2e_ms": latency_ms})
        return [
            ActionChunk(
                request_id=st.request_ids[i],
                actions=actions[i],
                latency_ms=latency_ms,
                meta={"batch_size": st.real_B, "bucket": st.bucket, "num_steps": st.num_steps},
                policy_version=self.policy_version,
                timing=dict(timing),
            )
            for i in range(st.real_B)
        ]

    @staticmethod
    def _trace_to_cpu(
        trace: DecodeTrace | None,
        *,
        policy_version: int = 0,
        timing: dict[str, float] | None = None,
    ) -> DecodeTrace | None:
        if trace is None:
            return None
        return DecodeTrace(
            token_ids=trace.token_ids.detach().cpu(),
            token_logprobs=(
                trace.token_logprobs.detach().float().cpu() if trace.token_logprobs is not None else None
            ),
            action_mask=trace.action_mask.detach().cpu() if trace.action_mask is not None else None,
            text=trace.text,
            parsed_actions=trace.parsed_actions,
            stop_reason=trace.stop_reason,
            meta=dict(trace.meta),
            policy_version=policy_version,
            timing={**trace.timing, **(timing or {})},
        )

    def _execute_recurrent(
        self,
        batch: PolicyBatch,
        num_steps: int | None,
        generator: torch.Generator | None,
        session_ids: Sequence[SessionKey] | None,
    ) -> list[ActionChunk]:
        if batch.batch_size != 1:
            raise UnsupportedRecurrentModeError("recurrent execution currently requires batch_size == 1")
        if session_ids is None or len(session_ids) != 1:
            raise SessionRequiredError("recurrent execution requires exactly one explicit SessionKey")

        lease = self._sessions.checkout(session_ids[0])
        timer = _StageTimer(self.device)
        try:
            steps = num_steps or self.config.num_steps or self.pcfg.default_num_steps
            moved = batch.to(self.device, self.dtype)
            prefix = self.policy.encode_prefix(moved, lease.memory)
            timer.mark_prefill_complete()
            state = self.policy.decoder.init_state(1, generator)
            result = self.policy.decoder.decode(
                state,
                prefix,
                steps,
                1,
                None,
                generator=generator,
                cancelled=lease.cancelled,
            )
            if result.next_memory is None:
                raise RuntimeError("recurrent decoder did not return next_memory")
            if result.actions.shape[0] != 1:
                raise RuntimeError(
                    f"recurrent decoder must return one action row; got shape {tuple(result.actions.shape)}"
                )
            if result.traces is not None and len(result.traces) != 1:
                raise RuntimeError("recurrent decoder traces must align one-to-one with the batch")

            latency_ms, timing = timer.finish()
            logprob = None
            if result.behavior_logprob is not None:
                logprob = result.behavior_logprob[0].detach().float().cpu()
            chunk = ActionChunk(
                request_id=list(moved.request_ids)[0],
                actions=result.actions[0].detach().float().cpu(),
                logprob=logprob,
                latency_ms=latency_ms,
                meta={"batch_size": 1, "bucket": 1, "num_steps": steps, "session": session_ids[0]},
                trace=self._trace_to_cpu(
                    result.traces[0] if result.traces else None,
                    policy_version=self.policy_version,
                    timing=timing,
                ),
                policy_version=self.policy_version,
                timing=timing,
            )
            lease.commit(result.next_memory)
            return [chunk]
        except BaseException:
            lease.rollback()
            raise

    @torch.no_grad()
    def execute(
        self,
        batch: PolicyBatch,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        *,
        session_ids: Sequence[SessionKey] | None = None,
    ) -> list[ActionChunk]:
        if self.policy.is_recurrent:
            return self._execute_recurrent(batch, num_steps, generator, session_ids)
        timer = _StageTimer(self.device)
        st = self._prefill(batch, num_steps, generator)
        timer.mark_prefill_complete()
        x = self._integrate(st.x, st.prefix, st.num_steps, st.bucket)
        latency_ms, timing = timer.finish()
        return self._pack(x, st, latency_ms, timing)

    def reset_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        self._sessions.reset(list(session_ids))

    def checkout_sessions(self, session_ids: Sequence[SessionKey]) -> list[SessionLease]:
        """Acquire a recurrent rollout group without exposing the store itself."""
        return self._sessions.checkout_many(session_ids)

    def commit_sessions(self, leases: Sequence[SessionLease], memories: Sequence[object]) -> None:
        """Atomically publish every branch in a recurrent rollout group."""
        self._sessions.commit_many(leases, memories)

    def cancel_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        self._sessions.cancel(list(session_ids))

    def has_session_state(self) -> bool:
        return self._sessions.has_committed() or self._sessions.has_inflight()

    def _overlap_ready(self) -> bool:
        """Cross-batch overlap pays off only when denoise is a CUDA graph (so it
        does not hold the CPU submit thread — an eager loop there makes two streams
        contend and lose) and a MemPool can isolate the concurrent prefill's
        allocations from the graph replay (required for bit-exactness)."""
        return (
            self._graphs is not None
            and self.device.type == "cuda"
            and hasattr(torch.cuda, "MemPool")
            and hasattr(torch.cuda, "use_mem_pool")
        )

    def _pipeline_step(
        self,
        prev: _Staged | None,
        next_batch: PolicyBatch | None,
        num_steps: int | None,
        generator: torch.Generator | None,
    ) -> tuple[list[ActionChunk] | None, _Staged | None]:
        """One depth-1 pipeline step: overlap ``prev``'s denoise (stream A) with
        ``next_batch``'s prefill (stream B, allocations isolated in a MemPool so the
        concurrent replay stays bit-exact). Returns ``(prev_actions, next_staged)``,
        either side ``None`` when absent (priming or draining the pipeline). Without
        a CUDA graph / MemPool it runs each stage sequentially — same results, no
        overlap — so callers need no branch.

        This is the reusable step behind both :meth:`execute_pipelined` (a fixed
        list of batches) and the async engine's pipelined loop (batches arriving
        over time).
        """
        if self.policy.is_recurrent:
            raise UnsupportedRecurrentModeError("recurrent policies do not support pipelined execution")
        if not self._overlap_ready():
            prev_actions = None
            if prev is not None:
                t0 = now_ns()
                x = self._integrate(prev.x, prev.prefix, prev.num_steps, prev.bucket)
                sync_if_cuda(self.device)
                prev_actions = self._pack(x, prev, (now_ns() - t0) / 1e6)
            staged = self._prefill(next_batch, num_steps, generator) if next_batch is not None else None
            return prev_actions, staged

        if self._stream_a is None:
            self._stream_a = torch.cuda.Stream()
            self._stream_b = torch.cuda.Stream()
            self._prefill_pool = torch.cuda.MemPool()
        sA, sB = self._stream_a, self._stream_b
        t0 = now_ns()
        x = None
        if prev is not None:
            with torch.cuda.stream(sA):  # denoise prev (graph replay)
                x = self._integrate(prev.x, prev.prefix, prev.num_steps, prev.bucket)
        staged = None
        if next_batch is not None:  # prefill next, isolated so it can't alias the replay
            with torch.cuda.stream(sB), torch.cuda.use_mem_pool(self._prefill_pool):
                staged = self._prefill(next_batch, num_steps, generator)
        if prev is not None:
            sA.synchronize()
        if next_batch is not None:
            sB.synchronize()  # finish before stream A reads it next step
        prev_actions = self._pack(x, prev, (now_ns() - t0) / 1e6) if prev is not None else None
        return prev_actions, staged

    @torch.no_grad()
    def execute_pipelined(
        self,
        batches: list[PolicyBatch],
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> list[list[ActionChunk]]:
        """Execute a sequence of batches, overlapping each batch's denoise with the
        next batch's prefill (see :meth:`_pipeline_step`). Returns one
        ``ActionChunk`` list per input batch, bit-identical to :meth:`execute` on
        each — only timing differs.

        This is the value of staggered arrival (async rollout serving): denoise is
        memory-bound and prefill compute-bound, so they overlap on orthogonal
        resources. For a single synchronous batch there is nothing to overlap
        (splitting one batch is a net loss, denoise being batch-insensitive), so
        this falls back to sequential :meth:`execute` — as it also does without a
        CUDA graph / MemPool. Callers can always use this API.
        """
        if self.policy.is_recurrent:
            raise UnsupportedRecurrentModeError("recurrent policies do not support pipelined execution")
        if len(batches) <= 1 or not self._overlap_ready():
            return [self.execute(b, num_steps, generator) for b in batches]
        results: list[list[ActionChunk]] = []
        prev: _Staged | None = None
        for b in batches:
            prev_actions, prev = self._pipeline_step(prev, b, num_steps, generator)
            if prev_actions is not None:
                results.append(prev_actions)
        prev_actions, _ = self._pipeline_step(prev, None, num_steps, generator)  # drain the last
        if prev_actions is not None:
            results.append(prev_actions)
        return results
