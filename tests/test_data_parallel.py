"""CPU unit tests for the data-parallel engine (fake replicas, no CUDA)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
import torch

from embodiinfer.engine.parallel.data_parallel import (
    DataParallelEngine,
    InProcessReplica,
    LeastLoadedDispatcher,
    ProcessExecutor,
    RoundRobinDispatcher,
)
from embodiinfer.exceptions import (
    ReplicaExecutionError,
    SessionBusyError,
    SessionRequiredError,
    UnsupportedRecurrentModeError,
)
from embodiinfer.types import ActionChunk, SessionKey


@dataclass
class _Obs:
    """Minimal stand-in for Observation carrying only an identifying tag."""

    tag: int


class FakeReplica:
    """A :class:`Replica` that echoes each observation's tag into its chunk.

    The chunk's ``meta['tag']`` is the input observation's tag and its
    ``request_id`` is the caller's id, so a test can prove order + id are
    preserved through sharding. Optionally fails, or reports a fixed ``load``.
    """

    def __init__(
        self,
        replica_id: int,
        load: float = 0.0,
        fail: bool = False,
        *,
        recurrent: bool = False,
        result_mode: str = "valid",
    ):
        self.replica_id = replica_id
        self.device = "cpu"
        self.is_recurrent = recurrent
        self._load = load
        self._fail = fail
        self._result_mode = result_mode
        self._failed = False
        self.shutdown_calls = 0
        self.seen: list[int] = []  # tags this replica was asked to serve
        self.seen_sessions: list[SessionKey] = []
        self.reset_calls: list[list[SessionKey]] = []
        self.cancel_calls: list[list[SessionKey]] = []

    def execute(self, observations, num_steps=None, request_ids=None, *, session_ids=None):
        if self._fail:
            self._failed = True
            raise RuntimeError(f"replica {self.replica_id} boom")
        self.seen.extend(o.tag for o in observations)
        if session_ids is not None:
            self.seen_sessions.extend(session_ids)
        chunks = [
            ActionChunk(request_id=request_ids[j], actions=torch.tensor([float(o.tag)]), meta={"tag": o.tag})
            for j, o in enumerate(observations)
        ]
        if self._result_mode == "short":
            return chunks[:-1]
        if self._result_mode == "wrong_id":
            chunks[0].request_id = "wrong"
        return chunks

    def reset_sessions(self, session_ids):
        self.reset_calls.append(list(session_ids))

    def cancel_sessions(self, session_ids):
        self.cancel_calls.append(list(session_ids))

    def load(self) -> float:
        return self._load

    def healthy(self) -> bool:
        return not self._failed

    def shutdown(self) -> None:
        self.shutdown_calls += 1


# --- dispatchers ----------------------------------------------------------- #
def test_round_robin_assignment_and_cursor():
    d = RoundRobinDispatcher()
    replicas = [FakeReplica(i) for i in range(3)]
    assert d.assign(5, replicas) == [0, 1, 2, 0, 1]
    # cursor carries over: next call resumes at 2
    assert d.assign(2, replicas) == [2, 0]


def test_least_loaded_prefers_idle_replica():
    d = LeastLoadedDispatcher()
    replicas = [FakeReplica(0, load=5.0), FakeReplica(1, load=0.0), FakeReplica(2, load=3.0)]
    # both requests go to the idle replica 1 (0 -> 1 still < replica 2's 3)
    assert d.assign(2, replicas) == [1, 1]


def test_least_loaded_spreads_when_balanced():
    d = LeastLoadedDispatcher()
    replicas = [FakeReplica(i, load=0.0) for i in range(3)]
    assert d.assign(4, replicas) == [0, 1, 2, 0]


# --- DataParallelEngine: order + ids --------------------------------------- #
def test_execute_preserves_order_and_request_ids():
    replicas = [FakeReplica(i) for i in range(3)]
    dp = DataParallelEngine(replicas)
    obs = [_Obs(tag=i) for i in range(7)]
    chunks = dp.execute(obs)

    assert len(chunks) == 7
    # chunk i corresponds to observation i, regardless of which replica served it
    for i, c in enumerate(chunks):
        assert c.meta["tag"] == i
        assert c.request_id == f"req_{i}"
    # every request was served exactly once, spread across replicas
    assert sorted(t for r in replicas for t in r.seen) == list(range(7))
    assert all(r.seen for r in replicas)


def test_execute_forwards_caller_request_ids():
    dp = DataParallelEngine([FakeReplica(i) for i in range(2)])
    ids = ["alpha", "beta", "gamma"]
    chunks = dp.execute([_Obs(tag=i) for i in range(3)], request_ids=ids)
    assert [c.request_id for c in chunks] == ids


def test_empty_batch_returns_empty():
    dp = DataParallelEngine([FakeReplica(0)])
    assert dp.execute([]) == []


def test_request_ids_length_mismatch_raises():
    dp = DataParallelEngine([FakeReplica(0)])
    with pytest.raises(ValueError):
        dp.execute([_Obs(0), _Obs(1)], request_ids=["only-one"])


def test_session_ids_length_mismatch_raises():
    dp = DataParallelEngine([FakeReplica(0)])
    with pytest.raises(ValueError, match="session_ids length"):
        dp.execute(
            [_Obs(0), _Obs(1)],
            session_ids=[SessionKey("env", "episode")],
        )


@pytest.mark.parametrize("assignment", [[0], [0, 2], [0, -1], [0, True]])
def test_dispatcher_output_is_validated(assignment):
    class BadDispatcher:
        def assign(self, num_requests, replicas):
            del num_requests, replicas
            return assignment

    dp = DataParallelEngine([FakeReplica(0), FakeReplica(1)], dispatcher=BadDispatcher())
    with pytest.raises(ValueError, match="dispatcher returned"):
        dp.execute([_Obs(0), _Obs(1)])


def test_replica_ids_must_be_unique():
    with pytest.raises(ValueError, match="replica_id values must be unique"):
        DataParallelEngine([FakeReplica(3), FakeReplica(3)])


@pytest.mark.parametrize("result_mode", ["short", "wrong_id"])
def test_replica_result_count_and_ids_are_validated(result_mode):
    dp = DataParallelEngine([FakeReplica(0, result_mode=result_mode)])
    with pytest.raises(ReplicaExecutionError) as exc_info:
        dp.execute([_Obs(0), _Obs(1)])
    assert isinstance(exc_info.value.__cause__, ValueError)


# --- episode affinity ------------------------------------------------------ #
def test_session_is_pinned_to_one_replica_across_calls():
    replicas = [FakeReplica(0), FakeReplica(1)]
    dp = DataParallelEngine(replicas)
    first = SessionKey("env-a", "episode")
    second = SessionKey("env-b", "episode")

    dp.execute([_Obs(0), _Obs(1)], session_ids=[first, second])
    dp.execute([_Obs(2), _Obs(3)], session_ids=[second, first])

    assert dp.session_affinity == {first: 0, second: 1}
    assert replicas[0].seen_sessions == [first, first]
    assert replicas[1].seen_sessions == [second, second]


def test_duplicate_session_turns_are_rejected():
    dp = DataParallelEngine([FakeReplica(0), FakeReplica(1)])
    session = SessionKey("env", "episode")

    with pytest.raises(ValueError, match="session_ids must be unique"):
        dp.execute([_Obs(0), _Obs(1)], session_ids=[session, session])


def test_concurrent_first_turns_create_one_session_pin():
    replicas = [FakeReplica(0), FakeReplica(1)]
    dp = DataParallelEngine(replicas)
    session = SessionKey("env", "episode")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(dp.execute, [_Obs(i)], session_ids=[session]) for i in range(32)]
        for future in futures:
            future.result()

    owners = [replica.replica_id for replica in replicas if replica.seen]
    assert owners == [dp.session_affinity[session]]


def test_same_session_execute_calls_are_serialized_end_to_end():
    class BlockingReplica(FakeReplica):
        def __init__(self, replica_id):
            super().__init__(replica_id)
            self.first_entered = threading.Event()
            self.second_entered = threading.Event()
            self.release_first = threading.Event()
            self.calls = 0
            self.calls_lock = threading.Lock()

        def execute(self, observations, num_steps=None, request_ids=None, *, session_ids=None):
            with self.calls_lock:
                self.calls += 1
                call = self.calls
            if call == 1:
                self.first_entered.set()
                assert self.release_first.wait(timeout=2)
            else:
                self.second_entered.set()
            return super().execute(
                observations,
                num_steps=num_steps,
                request_ids=request_ids,
                session_ids=session_ids,
            )

    owner = BlockingReplica(0)
    dp = DataParallelEngine([owner, FakeReplica(1)])
    session = SessionKey("env", "episode")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(dp.execute, [_Obs(1)], session_ids=[session])
        assert owner.first_entered.wait(timeout=2)

        second_started = threading.Event()

        def run_second():
            second_started.set()
            return dp.execute([_Obs(2)], session_ids=[session])

        second = pool.submit(run_second)
        assert second_started.wait(timeout=2)
        assert not owner.second_entered.wait(timeout=0.05)
        owner.release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert owner.seen == [1, 2]


def test_reset_cannot_unpin_after_routing_before_worker_execution():
    class ObservableResetReplica(FakeReplica):
        def __init__(self, replica_id):
            super().__init__(replica_id)
            self.reset_entered = threading.Event()

        def reset_sessions(self, session_ids):
            self.reset_entered.set()
            super().reset_sessions(session_ids)

    class PausedExecutor:
        def __init__(self):
            self.routed = threading.Event()
            self.release = threading.Event()

        def run(self, tasks):
            self.routed.set()
            assert self.release.wait(timeout=2)
            return [task() for task in tasks]

        def shutdown(self):
            pass

    replica = ObservableResetReplica(0)
    executor = PausedExecutor()
    dp = DataParallelEngine([replica], executor=executor)
    session = SessionKey("env", "episode")

    with ThreadPoolExecutor(max_workers=2) as pool:
        execute = pool.submit(dp.execute, [_Obs(1)], session_ids=[session])
        assert executor.routed.wait(timeout=2)

        reset_started = threading.Event()

        def run_reset():
            reset_started.set()
            dp.reset_sessions([session])

        reset = pool.submit(run_reset)
        assert reset_started.wait(timeout=2)
        assert not replica.reset_entered.wait(timeout=0.05)
        assert session in dp.session_affinity

        executor.release.set()
        execute.result(timeout=2)
        reset.result(timeout=2)

    assert replica.seen == [1]
    assert replica.reset_calls == [[session]]
    assert session not in dp.session_affinity


def test_blocked_reset_does_not_stop_unrelated_replica_routing():
    class BlockingResetReplica(FakeReplica):
        def __init__(self, replica_id):
            super().__init__(replica_id)
            self.reset_entered = threading.Event()
            self.release_reset = threading.Event()

        def reset_sessions(self, session_ids):
            self.reset_entered.set()
            assert self.release_reset.wait(timeout=2)
            super().reset_sessions(session_ids)

    class ObservableReplica(FakeReplica):
        def __init__(self, replica_id):
            super().__init__(replica_id)
            self.execute_entered = threading.Event()

        def execute(self, observations, num_steps=None, request_ids=None, *, session_ids=None):
            self.execute_entered.set()
            return super().execute(
                observations,
                num_steps=num_steps,
                request_ids=request_ids,
                session_ids=session_ids,
            )

    resetting = BlockingResetReplica(0)
    unrelated = ObservableReplica(1)
    dp = DataParallelEngine([resetting, unrelated])
    resetting_session = SessionKey("resetting", "episode")
    unrelated_session = SessionKey("unrelated", "episode")
    dp.execute([_Obs(0)], session_ids=[resetting_session])

    with ThreadPoolExecutor(max_workers=2) as pool:
        reset = pool.submit(dp.reset_sessions, [resetting_session])
        assert resetting.reset_entered.wait(timeout=2)
        execute = pool.submit(dp.execute, [_Obs(1)], session_ids=[unrelated_session])
        routed_while_reset_blocked = unrelated.execute_entered.wait(timeout=0.2)
        if routed_while_reset_blocked:
            execute.result(timeout=2)

        resetting.release_reset.set()
        reset.result(timeout=2)
        execute.result(timeout=2)

    assert routed_while_reset_blocked
    assert dp.session_affinity == {unrelated_session: 1}


# --- error handling -------------------------------------------------------- #
def test_replica_failure_raises_and_marks_unhealthy():
    good = FakeReplica(0)
    bad = FakeReplica(1, fail=True)
    # round-robin so the bad replica is guaranteed a request
    dp = DataParallelEngine([good, bad])
    with pytest.raises(ReplicaExecutionError) as ei:
        dp.execute([_Obs(tag=i) for i in range(4)])
    assert ei.value.num_failed >= 1
    assert 1 in {rid for rid, _ in ei.value.failures}
    assert not bad.healthy()
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_retry_on_healthy_recovers():
    good = FakeReplica(0)
    bad = FakeReplica(1, fail=True)
    dp = DataParallelEngine([good, bad], retry_on_healthy=True)
    chunks = dp.execute([_Obs(tag=i) for i in range(4)])
    assert len(chunks) == 4
    for i, c in enumerate(chunks):
        assert c.meta["tag"] == i
    # all four ended up served by the healthy replica after the retry
    assert sorted(good.seen) == [0, 1, 2, 3]
    assert not bad.healthy()


def test_retry_raises_when_no_healthy_left():
    dp = DataParallelEngine([FakeReplica(0, fail=True)], retry_on_healthy=True)
    with pytest.raises(ReplicaExecutionError):
        dp.execute([_Obs(tag=0)])


# --- lifecycle ------------------------------------------------------------- #
def test_shutdown_is_idempotent_and_reaches_replicas():
    replicas = [FakeReplica(i) for i in range(2)]
    dp = DataParallelEngine(replicas)
    dp.shutdown()
    dp.shutdown()  # idempotent
    assert all(r.shutdown_calls == 1 for r in replicas)


def test_context_manager_shuts_down():
    replicas = [FakeReplica(i) for i in range(2)]
    with DataParallelEngine(replicas) as dp:
        dp.execute([_Obs(tag=0)])
    assert all(r.shutdown_calls == 1 for r in replicas)


def test_process_executor_is_a_documented_stub():
    with pytest.raises(NotImplementedError):
        ProcessExecutor()


# --- recurrent episode parallelism ---------------------------------------- #
@dataclass
class _ToyBatch:
    observations: list[_Obs]
    request_ids: list[str]

    @property
    def batch_size(self):
        return len(self.observations)


class _ToyRecurrentPolicy:
    is_recurrent = True

    def collate(self, observations, request_ids):
        return _ToyBatch(list(observations), list(request_ids))


class _ToyRecurrentCore:
    """CPU core double with replica-local episode memory and a B=1 contract."""

    def __init__(self, *, fail_tags=(), block_first: bool = False):
        self.device = torch.device("cpu")
        self.policy = _ToyRecurrentPolicy()
        self.memory: dict[SessionKey, list[int]] = {}
        self.fail_tags = set(fail_tags)
        self.batch_sizes: list[int] = []
        self.reset_calls: list[list[SessionKey]] = []
        self.cancel_calls: list[list[SessionKey]] = []
        self.block_first = block_first
        self.first_entered = threading.Event()
        self.another_entered = threading.Event()
        self.release_first = threading.Event()
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._active_lock = threading.Lock()

    def execute(self, batch, num_steps=None, *, session_ids=None):
        del num_steps
        assert batch.batch_size == 1
        assert session_ids is not None and len(session_ids) == 1
        with self._active_lock:
            self.calls += 1
            call = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if call > 1:
                self.another_entered.set()
        try:
            if self.block_first and call == 1:
                self.first_entered.set()
                assert self.release_first.wait(timeout=2)
            tag = batch.observations[0].tag
            if tag in self.fail_tags:
                raise SessionBusyError(f"injected failure for tag {tag}")
            session_id = session_ids[0]
            history = self.memory.setdefault(session_id, [])
            history.append(tag)
            self.batch_sizes.append(batch.batch_size)
            return [
                ActionChunk(
                    request_id=batch.request_ids[0],
                    actions=torch.tensor([[float(sum(history))]]),
                    meta={"history": list(history)},
                )
            ]
        finally:
            with self._active_lock:
                self.active -= 1

    def reset_sessions(self, session_ids):
        self.reset_calls.append(list(session_ids))
        for session_id in session_ids:
            self.memory.pop(session_id, None)

    def cancel_sessions(self, session_ids):
        self.cancel_calls.append(list(session_ids))


def _toy_recurrent_dp():
    cores = [_ToyRecurrentCore(), _ToyRecurrentCore()]
    replicas = [InProcessReplica(core, replica_id=index) for index, core in enumerate(cores)]
    return DataParallelEngine(replicas), cores


def test_recurrent_episode_parallel_is_b1_per_replica_and_preserves_memory():
    dp, cores = _toy_recurrent_dp()
    left = SessionKey("left", "episode")
    right = SessionKey("right", "episode")

    first = dp.execute(
        [_Obs(1), _Obs(10)],
        request_ids=["l0", "r0"],
        session_ids=[left, right],
    )
    second = dp.execute(
        [_Obs(20), _Obs(2)],
        request_ids=["r1", "l1"],
        session_ids=[right, left],
    )
    third = dp.execute(
        [_Obs(3), _Obs(30)],
        request_ids=["l2", "r2"],
        session_ids=[left, right],
    )

    assert [chunk.meta["history"] for chunk in first] == [
        [1],
        [10],
    ]
    assert [chunk.meta["history"] for chunk in second] == [
        [10, 20],
        [1, 2],
    ]
    assert [chunk.meta["history"] for chunk in third] == [
        [1, 2, 3],
        [10, 20, 30],
    ]
    assert all(batch_size == 1 for core in cores for batch_size in core.batch_sizes)
    assert dp.session_affinity[left] != dp.session_affinity[right]


def test_recurrent_reset_clears_owner_state_and_releases_affinity():
    dp, cores = _toy_recurrent_dp()
    session = SessionKey("env", "episode")
    dp.execute([_Obs(1)], session_ids=[session])
    owner = dp.session_affinity[session]

    dp.cancel_sessions([session])
    assert session in dp.session_affinity
    assert cores[owner].cancel_calls == [[session]]

    dp.reset_sessions([session])
    assert session not in dp.session_affinity
    assert session not in cores[owner].memory
    assert cores[owner].reset_calls == [[session]]


def test_recurrent_requires_sessions_and_rejects_retry_migration():
    replica = InProcessReplica(_ToyRecurrentCore(), replica_id=0)
    dp = DataParallelEngine([replica])
    with pytest.raises(SessionRequiredError):
        dp.execute([_Obs(1)])
    with pytest.raises(UnsupportedRecurrentModeError, match="cannot retry"):
        DataParallelEngine([replica], retry_on_healthy=True)


def test_recurrent_failure_is_reported_per_row_after_other_rows_commit():
    cores = [_ToyRecurrentCore(), _ToyRecurrentCore(fail_tags={10})]
    replicas = [InProcessReplica(core, replica_id=index) for index, core in enumerate(cores)]
    dp = DataParallelEngine(replicas)
    sessions = [SessionKey(index, "episode") for index in range(4)]

    with pytest.raises(ReplicaExecutionError, match="already committed") as exc_info:
        dp.execute(
            [_Obs(1), _Obs(10), _Obs(2), _Obs(20)],
            request_ids=["ok-0", "failed", "ok-2", "ok-3"],
            session_ids=sessions,
        )

    assert exc_info.value.num_failed == 1
    assert "['ok-0', 'ok-2', 'ok-3']" in str(exc_info.value)
    assert cores[0].memory == {sessions[0]: [1], sessions[2]: [2]}
    assert cores[1].memory == {sessions[3]: [20]}
    assert replicas[1].healthy()


def test_in_process_replica_serializes_complete_core_execution():
    core = _ToyRecurrentCore(block_first=True)
    replica = InProcessReplica(core, replica_id=0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(
            replica.execute,
            [_Obs(0)],
            request_ids=["r0"],
            session_ids=[SessionKey(0, "episode")],
        )
        assert core.first_entered.wait(timeout=2)
        futures = [
            pool.submit(
                replica.execute,
                [_Obs(i)],
                request_ids=[f"r{i}"],
                session_ids=[SessionKey(i, "episode")],
            )
            for i in range(1, 8)
        ]
        assert not core.another_entered.wait(timeout=0.05)
        core.release_first.set()
        first.result(timeout=2)
        for future in futures:
            future.result(timeout=2)

    assert core.max_active == 1


# --- InProcessReplica wrapping a real EngineCore --------------------------- #
def test_engine_cores_are_auto_wrapped():
    from embodiinfer import EmbodiInfer, EngineConfig, Observation, preset_config

    cfg = preset_config("tiny")

    def obs(i):
        return Observation(
            images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
            state=torch.rand(cfg.state_dim),
            instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
            env_id=i,
        )

    cores = [
        EmbodiInfer(
            "mock_flow_vla",
            preset="tiny",
            engine_config=EngineConfig(device="cpu", use_cuda_graph=False),
        ).core
        for _ in range(2)
    ]
    dp = DataParallelEngine(cores)
    assert all(isinstance(r, InProcessReplica) for r in dp.replicas)
    chunks = dp.execute([obs(i) for i in range(5)])
    assert len(chunks) == 5
    for i, c in enumerate(chunks):
        assert c.request_id == f"req_{i}"
        assert c.actions.shape == (cfg.action_horizon, cfg.action_dim)


@pytest.mark.gpu
def test_two_gpu_episode_affinity_with_per_replica_cuda_graphs():
    """Thread workers must enter each core's CUDA device before lazy capture."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    from embodiinfer import EmbodiInfer, EngineConfig, Observation, preset_config

    cfg = preset_config("tiny")

    def obs(seed):
        generator = torch.Generator().manual_seed(seed)
        return Observation(
            images=torch.rand(
                cfg.num_cameras,
                3,
                cfg.image_size,
                cfg.image_size,
                generator=generator,
            ),
            state=torch.rand(cfg.state_dim, generator=generator),
            instruction_tokens=torch.randint(
                0,
                cfg.vocab_size,
                (cfg.max_lang_len,),
                generator=generator,
            ),
            env_id=seed,
        )

    cores = [
        EmbodiInfer(
            "mock_flow_vla",
            preset="tiny",
            engine_config=EngineConfig(
                device=f"cuda:{device}",
                max_batch_size=2,
                use_cuda_graph=True,
                capture_full_loop=False,
            ),
        ).core
        for device in range(2)
    ]
    sessions = [SessionKey(f"env-{index}", "episode") for index in range(4)]
    with DataParallelEngine(cores) as engine:
        first = engine.execute(
            [obs(index) for index in range(4)],
            request_ids=[f"first-{index}" for index in range(4)],
            session_ids=sessions,
        )
        owners = engine.session_affinity
        second = engine.execute(
            [obs(index + 10) for index in reversed(range(4))],
            request_ids=[f"second-{index}" for index in reversed(range(4))],
            session_ids=list(reversed(sessions)),
        )

    assert [chunk.request_id for chunk in first] == [f"first-{index}" for index in range(4)]
    assert [chunk.request_id for chunk in second] == [f"second-{index}" for index in reversed(range(4))]
    assert owners == {session: index % 2 for index, session in enumerate(sessions)}
    assert all((2, None) in core._graphs._graphs for core in cores)
