from dataclasses import dataclass

import pytest

from embodiinfer.engine.session import SessionStore
from embodiinfer.exceptions import SessionBusyError, StaleSessionError
from embodiinfer.types import SessionKey


@dataclass
class _Memory:
    value: int

    @property
    def seq_len(self) -> int:
        return self.value

    def to(self, device):
        return self


def _key(episode=1, rollout=0):
    return SessionKey(env_id="env-0", episode_id=episode, rollout_id=rollout)


def test_session_commit_and_next_checkout():
    store = SessionStore()
    lease = store.checkout(_key())
    assert lease.memory is None
    lease.commit(_Memory(3))

    next_lease = store.checkout(_key())
    assert next_lease.memory == _Memory(3)
    next_lease.commit(_Memory(7))
    assert store.committed(_key()) == _Memory(7)


def test_session_rollback_preserves_committed_memory():
    store = SessionStore()
    first = store.checkout(_key())
    first.commit(_Memory(5))

    failed = store.checkout(_key())
    assert failed.memory == _Memory(5)
    failed.rollback()
    assert store.committed(_key()) == _Memory(5)


def test_reset_and_cancel_are_idempotent():
    store = SessionStore()
    lease = store.checkout(_key())
    lease.commit(_Memory(4))

    store.reset([_key()])
    store.reset([_key()])
    assert store.committed(_key()) is None

    replacement = store.checkout(_key())
    replacement.commit(_Memory(6))
    store.cancel([_key()])
    store.cancel([_key()])
    assert store.committed(_key()) == _Memory(6)


def test_cancelled_turn_preserves_previous_commit_and_rejects_late_write():
    store = SessionStore()
    first = store.checkout(_key())
    first.commit(_Memory(4))
    cancelled = store.checkout(_key())
    store.cancel([_key()])

    assert store.committed(_key()) == _Memory(4)
    with pytest.raises(StaleSessionError):
        cancelled.commit(_Memory(5))


def test_stale_lease_cannot_revive_reset_session():
    store = SessionStore()
    stale = store.checkout(_key())
    store.reset([_key()])

    current = store.checkout(_key())
    current.commit(_Memory(9))

    assert stale.cancelled()
    with pytest.raises(StaleSessionError):
        stale.commit(_Memory(2))
    assert store.committed(_key()) == _Memory(9)


def test_same_session_rejects_concurrent_checkout():
    store = SessionStore()
    lease = store.checkout(_key())
    with pytest.raises(SessionBusyError):
        store.checkout(_key())
    lease.rollback()


def test_different_episode_and_rollout_are_independent():
    store = SessionStore()
    a = store.checkout(_key(episode=1, rollout=0))
    b = store.checkout(_key(episode=1, rollout=1))
    c = store.checkout(_key(episode=2, rollout=0))
    a.commit(_Memory(1))
    b.commit(_Memory(2))
    c.commit(_Memory(3))

    assert store.committed(_key(episode=1, rollout=0)) == _Memory(1)
    assert store.committed(_key(episode=1, rollout=1)) == _Memory(2)
    assert store.committed(_key(episode=2, rollout=0)) == _Memory(3)
