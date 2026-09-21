"""Data-parallel replicas for lossless throughput scaling.

For a compute-bound VLA (e.g. pi0.5 in fp32), a single GPU saturates and batching
plateaus. The lossless way past the plateau is data parallelism: hold a full
policy replica on each GPU, shard the incoming requests across replicas, and run
them concurrently. No numerics change — each request is served by exactly one
replica exactly as a single-GPU engine would serve it.

The module is factored into three replaceable pieces so the policy for *where* a
request runs, *how* replicas run concurrently, and *what* a replica is are all
independent:

    * :class:`Replica`        — one policy copy pinned to one device.
    * :class:`Dispatcher`     — assigns requests to replicas (round-robin / load).
    * :class:`ReplicaExecutor`— runs the per-replica work concurrently.

:class:`DataParallelEngine` wires them together, preserves caller request order
and ids, and turns a replica failure into a typed :class:`ReplicaExecutionError`.

Concurrency backend. The default :class:`ThreadedExecutor` uses one OS thread per
replica. This is not a GIL bottleneck for the reason it would be in pure Python:
a replica spends its time inside CUDA kernel launches and device execution, both
of which release the GIL, so replicas on distinct devices run in true parallel
(measured near-linear across 4x H200). The Python-side pre-processing each thread
does before launching (collate / tokenize) *does* hold the GIL and would
serialize as the replica count grows; a :class:`ProcessExecutor` (stub below)
would remove that at the cost of tensor IPC and per-process weight loading. At the
current few-replica scale the threaded pre-processing is not the bottleneck.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch

from ...exceptions import (
    ReplicaExecutionError,
    SessionError,
    SessionRequiredError,
    UnsupportedRecurrentModeError,
)
from ...types import ActionChunk, Observation, SessionKey
from ..core import EngineCore

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Replica: one policy copy on one device.                                     #
# --------------------------------------------------------------------------- #
class Replica(Protocol):
    """One full policy copy pinned to one device.

    A replica turns a list of observations into a list of action chunks, in the
    same order, tagging each chunk with the caller's request id. It also exposes
    a coarse ``load`` signal (for load-aware dispatch), a ``healthy`` flag (set
    false after a failure so the dispatcher stops routing to it), and an
    idempotent ``shutdown``.
    """

    replica_id: int
    device: str
    is_recurrent: bool

    def execute(
        self,
        observations: list[Observation],
        num_steps: int | None = None,
        request_ids: list[str] | None = None,
        *,
        session_ids: Sequence[SessionKey] | None = None,
    ) -> list[ActionChunk]: ...

    def reset_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        """Reset recurrent state owned by this replica."""
        ...

    def cancel_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        """Cancel in-flight recurrent work owned by this replica."""
        ...

    def load(self) -> float:
        """A coarse busy-ness signal for load-aware dispatch (e.g. in-flight count)."""
        ...

    def healthy(self) -> bool:
        """Whether the replica is up and has not failed or been shut down."""
        ...

    def shutdown(self) -> None:
        """Release the replica's resources. Idempotent."""
        ...


class InProcessReplica:
    """A :class:`Replica` backed by an in-process :class:`EngineCore`.

    Collates the assigned observations with the caller's request ids (no id
    rewriting) and runs them through the core on the core's own device. A failing
    ``execute`` marks the replica unhealthy and re-raises; the owning engine
    catches that to fence the replica off.
    """

    def __init__(self, core: EngineCore, replica_id: int = 0) -> None:
        self.core = core
        self.replica_id = replica_id
        self.device = str(core.device)
        self.is_recurrent = core.policy.is_recurrent
        self._failed = False
        self._shutdown = False
        self._inflight = 0
        self._state_lock = threading.Lock()
        # EngineCore and GraphManager own mutable staging buffers. Serialize the
        # complete collate/execute region, rather than only the CUDA launch, so
        # concurrent callers can never race one replica's graph inputs.
        self._execution_lock = threading.Lock()

    def _device_context(self):
        device = torch.device(self.core.device)
        if device.type == "cuda":
            # CUDA's current device is thread-local. ThreadPoolExecutor workers
            # otherwise inherit an arbitrary/default device on their first use.
            return torch.cuda.device(device)
        return nullcontext()

    def execute(
        self,
        observations: list[Observation],
        num_steps: int | None = None,
        request_ids: list[str] | None = None,
        *,
        session_ids: Sequence[SessionKey] | None = None,
    ) -> list[ActionChunk]:
        if request_ids is None:
            request_ids = [f"r{self.replica_id}:{j}" for j in range(len(observations))]
        elif len(request_ids) != len(observations):
            raise ValueError("request_ids length must match observations length")
        if session_ids is not None and len(session_ids) != len(observations):
            raise ValueError("session_ids length must match observations length")
        if self.is_recurrent and session_ids is None:
            raise SessionRequiredError("recurrent replica execution requires one SessionKey per observation")
        if self.is_recurrent and len(observations) != 1:
            raise ValueError("recurrent replica execution requires exactly one observation")

        with self._state_lock:
            if self._failed or self._shutdown:
                raise ReplicaExecutionError(
                    [],
                    num_failed=len(observations),
                    message=f"replica {self.replica_id} is not healthy",
                )
            self._inflight += len(observations)
        try:
            with self._execution_lock, self._device_context():
                # A preceding queued call may have failed while this call was
                # waiting for the per-replica lock.
                with self._state_lock:
                    if self._failed or self._shutdown:
                        raise ReplicaExecutionError(
                            [],
                            num_failed=len(observations),
                            message=f"replica {self.replica_id} is not healthy",
                        )
                if self.is_recurrent:
                    # Keep the replica contract atomic: one recurrent call is
                    # exactly one B=1 session transaction. DataParallelEngine
                    # sequences multiple assigned turns and records one outcome
                    # per transaction, so a later failure cannot make earlier
                    # committed rows look unexecuted.
                    if session_ids is None:  # guarded before entering the replica lock
                        raise RuntimeError("recurrent session validation was bypassed")
                    batch = self.core.policy.collate(observations, request_ids)
                    return self.core.execute(batch, num_steps, session_ids=session_ids)

                batch = self.core.policy.collate(observations, request_ids)
                return self.core.execute(batch, num_steps, session_ids=session_ids)
        except SessionError:
            # Busy/cancelled/stale leases are request lifecycle failures, not a
            # broken model replica. The episode remains pinned and may continue.
            raise
        except Exception:
            with self._state_lock:
                self._failed = True
            raise
        finally:
            with self._state_lock:
                self._inflight -= len(observations)

    def reset_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        # Wait for an already-routed turn to finish before clearing its state.
        # DataParallelEngine holds the affected sessions' lifecycle gates, so
        # they cannot be routed again before reset and unpin complete.
        with self._execution_lock, self._device_context():
            self.core.reset_sessions(session_ids)

    def cancel_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        # Deliberately do not take _execution_lock: cancellation must invalidate
        # a lease while the recurrent decoder is still running.
        self.core.cancel_sessions(session_ids)

    def load(self) -> float:
        with self._state_lock:
            return float(self._inflight)

    def healthy(self) -> bool:
        with self._state_lock:
            return not self._failed and not self._shutdown

    def shutdown(self) -> None:
        with self._state_lock:
            self._shutdown = True


# --------------------------------------------------------------------------- #
# Dispatcher: which replica serves each request.                              #
# --------------------------------------------------------------------------- #
class Dispatcher(Protocol):
    """Assigns a batch of requests to replicas.

    ``assign`` returns one index per request into the provided ``replicas``
    sequence. The engine passes only *healthy* replicas, so a dispatcher never
    has to check health itself.
    """

    def assign(self, num_requests: int, replicas: Sequence[Replica]) -> list[int]: ...


class RoundRobinDispatcher:
    """Cycle through replicas in order, carrying a cursor across calls.

    A batch of 5 over 3 replicas assigns ``[0, 1, 2, 0, 1]``; the next call
    resumes where this one stopped, so a stream of small batches still spreads
    evenly. Stateful but cheap; guarded by a lock so it is safe to share.
    """

    def __init__(self) -> None:
        self._next = 0
        self._lock = threading.Lock()

    def assign(self, num_requests: int, replicas: Sequence[Replica]) -> list[int]:
        k = len(replicas)
        if k == 0:
            raise ValueError("no replicas to dispatch to")
        with self._lock:
            start = self._next
            self._next = (self._next + num_requests) % k
        return [(start + i) % k for i in range(num_requests)]


class LeastLoadedDispatcher:
    """Greedily place each request on the currently least-loaded replica.

    Seeds from each replica's real ``load()`` and increments a virtual counter as
    it assigns, so one batch spreads across replicas weighted by their standing
    load. Ties break to the lowest index (deterministic).
    """

    def assign(self, num_requests: int, replicas: Sequence[Replica]) -> list[int]:
        k = len(replicas)
        if k == 0:
            raise ValueError("no replicas to dispatch to")
        virtual = [r.load() for r in replicas]
        out: list[int] = []
        for _ in range(num_requests):
            j = min(range(k), key=lambda i: virtual[i])
            out.append(j)
            virtual[j] += 1.0
        return out


# --------------------------------------------------------------------------- #
# ReplicaExecutor: how per-replica work runs concurrently.                    #
# --------------------------------------------------------------------------- #
class ReplicaExecutor(Protocol):
    """Runs a set of independent per-replica tasks concurrently.

    Each task is a zero-arg thunk; ``run`` returns their results in input order.
    """

    def run(self, tasks: Sequence[Callable[[], T]]) -> list[T]: ...

    def shutdown(self) -> None: ...


class ThreadedExecutor:
    """One OS thread per replica via a :class:`ThreadPoolExecutor`.

    Threads are the right backend here because a replica releases the GIL for the
    duration of its CUDA work (kernel launch + device execution), so replicas on
    distinct devices overlap. See the module docstring for the GIL trade-off.
    """

    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._closed = False

    def run(self, tasks: Sequence[Callable[[], T]]) -> list[T]:
        futures = [self._pool.submit(task) for task in tasks]
        return [f.result() for f in futures]

    def shutdown(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=False)


class ProcessExecutor:
    """Multiprocess replica backend — interface only, not implemented.

    Threads (see :class:`ThreadedExecutor`) suffice while the per-replica Python
    pre-processing (collate / tokenize, which holds the GIL) is cheap relative to
    the GPU work (which releases it). Once many replicas serialize on that
    GIL-bound pre-processing, a process-per-replica backend would remove it.

    Implementing it is deferred, not trivial: a process cannot receive the
    in-process thunks :class:`ReplicaExecutor` runs (closures are not picklable),
    so it needs a serializable ``(request, replica_handle)`` task representation,
    a copy of the model weights per process, and tensor IPC to return chunks.
    None of that pays off at the current few-replica scale.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "ProcessExecutor is not implemented; use ThreadedExecutor. "
            "See the class docstring for the GIL trade-off and what a real "
            "implementation would require."
        )


# --------------------------------------------------------------------------- #
# DataParallelEngine: wire replicas + dispatcher + executor together.         #
# --------------------------------------------------------------------------- #
@dataclass
class _Outcome:
    """Result of one replica's slice of a batch (success xor error)."""

    replica: Replica
    indices: list[int]  # original request positions this replica handled
    chunks: list[ActionChunk] | None
    error: BaseException | None


class _SessionGate:
    """Reference-counted lifecycle lock with a stable global acquisition order."""

    def __init__(self, order: int) -> None:
        self.order = order
        self.lock = threading.Lock()
        self.users = 0


class DataParallelEngine:
    """Shard a batch across replicas and gather results in the caller's order.

    ``execute`` dispatches each observation to a replica, runs the replicas
    concurrently, and scatters the action chunks back into input order (chunk
    ``i`` corresponds to observation ``i``), preserving caller request ids.

    A replica that raises is marked unhealthy and fenced off. By default the
    whole ``execute`` then raises :class:`ReplicaExecutionError`; with
    ``retry_on_healthy=True`` its requests are re-dispatched to the remaining
    healthy replicas instead (off by default so failures surface rather than
    hide).
    """

    def __init__(
        self,
        replicas: Sequence[Replica | EngineCore],
        dispatcher: Dispatcher | None = None,
        executor: ReplicaExecutor | None = None,
        retry_on_healthy: bool = False,
    ) -> None:
        if not replicas:
            raise ValueError("need at least one replica")
        # Ergonomics: a bare EngineCore is auto-wrapped as an InProcessReplica.
        self.replicas: list[Replica] = [
            r if not isinstance(r, EngineCore) else InProcessReplica(r, replica_id=i)
            for i, r in enumerate(replicas)
        ]
        replica_ids = [replica.replica_id for replica in self.replicas]
        if any(isinstance(replica_id, bool) or not isinstance(replica_id, int) for replica_id in replica_ids):
            raise ValueError("replica_id must be an integer")
        if len(set(replica_ids)) != len(replica_ids):
            raise ValueError("replica_id values must be unique")

        recurrent_modes = {bool(getattr(replica, "is_recurrent", False)) for replica in self.replicas}
        if len(recurrent_modes) != 1:
            raise ValueError("all replicas must agree on recurrent execution mode")
        self.is_recurrent = recurrent_modes.pop()
        if self.is_recurrent and retry_on_healthy:
            raise UnsupportedRecurrentModeError(
                "recurrent data parallelism cannot retry on another replica because session state is local"
            )

        self.dispatcher = dispatcher or RoundRobinDispatcher()
        self.executor = executor or ThreadedExecutor(max_workers=len(self.replicas))
        self.retry_on_healthy = retry_on_healthy
        self._replica_by_id = {replica.replica_id: replica for replica in self.replicas}
        self._session_affinity: dict[SessionKey, int] = {}
        self._affinity_lock = threading.Lock()
        self._session_gates: dict[SessionKey, _SessionGate] = {}
        self._session_gates_lock = threading.Lock()
        self._next_session_gate_order = 0
        self._closed = False

    @property
    def num_replicas(self) -> int:
        return len(self.replicas)

    @property
    def session_affinity(self) -> dict[SessionKey, int]:
        """Snapshot of episode-to-replica pins, primarily for diagnostics."""
        with self._affinity_lock:
            return dict(self._session_affinity)

    def execute(
        self,
        observations: list[Observation],
        num_steps: int | None = None,
        request_ids: list[str] | None = None,
        *,
        session_ids: Sequence[SessionKey] | None = None,
    ) -> list[ActionChunk]:
        n = len(observations)
        if request_ids is None:
            request_ids = [f"req_{i}" for i in range(n)]
        elif len(request_ids) != n:
            raise ValueError("request_ids length must match observations length")
        if session_ids is None:
            if self.is_recurrent and n:
                raise SessionRequiredError(
                    "recurrent data parallel execution requires one SessionKey per observation"
                )
            normalized_sessions = None
        else:
            if len(session_ids) != n:
                raise ValueError("session_ids length must match observations length")
            normalized_sessions = list(session_ids)
            if len(set(normalized_sessions)) != len(normalized_sessions):
                raise ValueError("session_ids must be unique within one execute call")
        if n == 0:
            return []

        # Keep every involved episode stable from affinity lookup through the
        # final replica result. In particular, reset cannot clear/unpin after
        # routing but before a queued worker actually enters replica.execute.
        if normalized_sessions is not None:
            with self._hold_session_lifecycle(normalized_sessions):
                return self._execute_routed(
                    observations,
                    num_steps,
                    request_ids,
                    normalized_sessions,
                )
        return self._execute_routed(observations, num_steps, request_ids, None)

    def _execute_routed(
        self,
        observations: list[Observation],
        num_steps: int | None,
        request_ids: list[str],
        session_ids: list[SessionKey] | None,
    ) -> list[ActionChunk]:
        n = len(observations)

        out: list[ActionChunk | None] = [None] * n
        pending = list(range(n))
        banned: set[int] = set()  # replicas that errored this call
        errors: list[tuple[int, BaseException]] = []

        while pending:
            outcomes = self._run_indices(
                pending,
                observations,
                request_ids,
                session_ids,
                num_steps,
                banned,
            )
            failed: list[int] = []
            for oc in outcomes:
                if oc.error is None:
                    assert oc.chunks is not None
                    for j, idx in enumerate(oc.indices):
                        out[idx] = oc.chunks[j]
                else:
                    banned.add(oc.replica.replica_id)
                    errors.append((oc.replica.replica_id, oc.error))
                    failed.extend(oc.indices)
            if not failed:
                break
            if self.is_recurrent:
                completed = [request_ids[index] for index, chunk in enumerate(out) if chunk is not None]
                message = None
                if completed:
                    message = (
                        f"{len(failed)} recurrent request(s) failed; successful request(s) "
                        f"{completed!r} already committed and must not be replayed"
                    )
                raise ReplicaExecutionError(
                    errors,
                    num_failed=len(failed),
                    message=message,
                ) from errors[-1][1]
            if not self.retry_on_healthy or not self._has_capacity(banned):
                raise ReplicaExecutionError(errors, num_failed=len(failed)) from errors[-1][1]
            pending = failed

        if any(chunk is None for chunk in out):
            raise RuntimeError("data-parallel gather did not produce one result per request")
        return [chunk for chunk in out if chunk is not None]

    def _has_capacity(self, banned: set[int]) -> bool:
        return any(r.healthy() and r.replica_id not in banned for r in self.replicas)

    def _run_indices(
        self,
        indices: list[int],
        observations: list[Observation],
        request_ids: list[str],
        session_ids: list[SessionKey] | None,
        num_steps: int | None,
        banned: set[int],
    ) -> list[_Outcome]:
        assignment = self._assign_replicas(indices, session_ids, banned)

        groups: dict[int, tuple[Replica, list[int]]] = {}
        for replica, index in zip(assignment, indices, strict=True):
            if replica.replica_id not in groups:
                groups[replica.replica_id] = (replica, [])
            groups[replica.replica_id][1].append(index)

        if self.is_recurrent:
            # A recurrent row commits replica-local state. Keep rows for one
            # replica ordered, but retain an independent outcome for every row
            # so a later failure is never reported as undoing an earlier commit.
            tasks = [
                self._make_recurrent_task(
                    replica,
                    idxs,
                    observations,
                    request_ids,
                    session_ids,
                    num_steps,
                )
                for replica, idxs in groups.values()
            ]
            return [outcome for outcomes in self.executor.run(tasks) for outcome in outcomes]

        tasks = [
            self._make_task(
                replica,
                idxs,
                observations,
                request_ids,
                session_ids,
                num_steps,
            )
            for replica, idxs in groups.values()
        ]
        return self.executor.run(tasks)

    def _assign_replicas(
        self,
        indices: list[int],
        session_ids: list[SessionKey] | None,
        banned: set[int],
    ) -> list[Replica]:
        """Resolve existing episode pins and atomically create new ones."""
        with self._affinity_lock:
            available = [
                replica for replica in self.replicas if replica.healthy() and replica.replica_id not in banned
            ]
            if not available:
                raise ReplicaExecutionError(
                    [], num_failed=len(indices), message="no healthy replicas available"
                )

            available_ids = {replica.replica_id for replica in available}
            assigned: list[Replica | None] = [None] * len(indices)
            new_units: list[int] = []

            for local_index, request_index in enumerate(indices):
                session_id = None if session_ids is None else session_ids[request_index]
                pinned_id = None if session_id is None else self._session_affinity.get(session_id)
                if pinned_id is not None and pinned_id not in available_ids:
                    if self.is_recurrent or not self.retry_on_healthy:
                        error = RuntimeError(
                            f"session {session_id!r} is pinned to unavailable replica {pinned_id}"
                        )
                        raise ReplicaExecutionError([(pinned_id, error)], num_failed=1) from error
                    # Stateless requests have no replica-local memory to lose;
                    # an explicitly enabled retry may safely establish a new pin.
                    self._session_affinity.pop(session_id, None)
                    pinned_id = None

                if pinned_id is not None:
                    assigned[local_index] = self._replica_by_id[pinned_id]
                else:
                    new_units.append(local_index)

            routed = self.dispatcher.assign(len(new_units), available) if new_units else []
            if len(routed) != len(new_units):
                raise ValueError("dispatcher returned the wrong number of assignments")
            for local_index, replica_index in zip(new_units, routed, strict=True):
                if (
                    isinstance(replica_index, bool)
                    or not isinstance(replica_index, int)
                    or replica_index < 0
                    or replica_index >= len(available)
                ):
                    raise ValueError(
                        f"dispatcher returned invalid replica index {replica_index!r}; "
                        f"expected an integer in [0, {len(available)})"
                    )
                replica = available[replica_index]
                assigned[local_index] = replica
                if session_ids is not None:
                    request_index = indices[local_index]
                    self._session_affinity[session_ids[request_index]] = replica.replica_id

            if any(replica is None for replica in assigned):
                raise RuntimeError("dispatcher left a request without a replica")
            return [replica for replica in assigned if replica is not None]

    @staticmethod
    def _make_recurrent_task(
        replica: Replica,
        idxs: list[int],
        observations: list[Observation],
        request_ids: list[str],
        session_ids: list[SessionKey] | None,
        num_steps: int | None,
    ) -> Callable[[], list[_Outcome]]:
        def task() -> list[_Outcome]:
            return [
                DataParallelEngine._make_task(
                    replica,
                    [index],
                    observations,
                    request_ids,
                    session_ids,
                    num_steps,
                )()
                for index in idxs
            ]

        return task

    @staticmethod
    def _make_task(
        replica: Replica,
        idxs: list[int],
        observations: list[Observation],
        request_ids: list[str],
        session_ids: list[SessionKey] | None,
        num_steps: int | None,
    ) -> Callable[[], _Outcome]:
        def task() -> _Outcome:
            obs = [observations[i] for i in idxs]
            ids = [request_ids[i] for i in idxs]
            sessions = None if session_ids is None else [session_ids[i] for i in idxs]
            try:
                chunks = replica.execute(
                    obs,
                    num_steps=num_steps,
                    request_ids=ids,
                    session_ids=sessions,
                )
                if len(chunks) != len(idxs):
                    raise ValueError(
                        f"replica {replica.replica_id} returned {len(chunks)} chunks for {len(idxs)} requests"
                    )
                returned_ids = [chunk.request_id for chunk in chunks]
                if returned_ids != ids:
                    raise ValueError(
                        f"replica {replica.replica_id} returned request ids {returned_ids!r}; "
                        f"expected {ids!r}"
                    )
                return _Outcome(replica=replica, indices=idxs, chunks=chunks, error=None)
            except Exception as exc:  # captured so the executor gathers partial results
                return _Outcome(replica=replica, indices=idxs, chunks=None, error=exc)

        return task

    @staticmethod
    def _unique_sessions(session_ids: Sequence[SessionKey]) -> list[SessionKey]:
        return list(dict.fromkeys(session_ids))

    @contextmanager
    def _hold_session_lifecycle(self, session_ids: Sequence[SessionKey]):
        """Serialize episode turns and reset without leaking one lock per episode."""
        keys = self._unique_sessions(session_ids)
        entries: list[tuple[SessionKey, _SessionGate]] = []
        with self._session_gates_lock:
            for key in keys:
                gate = self._session_gates.get(key)
                if gate is None:
                    gate = _SessionGate(self._next_session_gate_order)
                    self._next_session_gate_order += 1
                    self._session_gates[key] = gate
                gate.users += 1
                entries.append((key, gate))

        # Calls may contain overlapping session sets in different input orders.
        # The creation ordinal is unique and gives every caller one lock order.
        ordered = sorted(entries, key=lambda item: item[1].order)
        acquired: list[_SessionGate] = []
        try:
            for _, gate in ordered:
                gate.lock.acquire()
                acquired.append(gate)
            yield
        finally:
            for gate in reversed(acquired):
                gate.lock.release()
            with self._session_gates_lock:
                for key, gate in entries:
                    gate.users -= 1
                    if gate.users == 0 and self._session_gates.get(key) is gate:
                        del self._session_gates[key]

    def reset_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        """Reset state on its owning replica, then release the episode pin.

        Per-session lifecycle gates prevent the affected episodes from being
        routed between owner lookup, reset, and unpin. Replica work deliberately
        runs outside the global affinity lock so one busy device cannot stop
        unrelated episodes from being routed to other devices.
        """
        unique_sessions = self._unique_sessions(session_ids)
        with self._hold_session_lifecycle(unique_sessions):
            with self._affinity_lock:
                groups: dict[int, list[SessionKey]] = {}
                for session_id in unique_sessions:
                    replica_id = self._session_affinity.get(session_id)
                    if replica_id is not None:
                        groups.setdefault(replica_id, []).append(session_id)

            for replica_id, keys in groups.items():
                self._replica_by_id[replica_id].reset_sessions(keys)
                with self._affinity_lock:
                    for key in keys:
                        if self._session_affinity.get(key) == replica_id:
                            del self._session_affinity[key]

    def cancel_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        """Route cancellation to each owner without releasing its episode pin."""
        with self._affinity_lock:
            groups: dict[int, list[SessionKey]] = {}
            for session_id in self._unique_sessions(session_ids):
                replica_id = self._session_affinity.get(session_id)
                if replica_id is not None:
                    groups.setdefault(replica_id, []).append(session_id)
            for replica_id, keys in groups.items():
                self._replica_by_id[replica_id].cancel_sessions(keys)

    def shutdown(self) -> None:
        """Shut down the executor and every replica. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown()
        for r in self.replicas:
            r.shutdown()

    def __enter__(self) -> DataParallelEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
