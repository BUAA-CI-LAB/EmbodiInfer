"""Transactional storage for policy state that persists across environment steps."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

from ..exceptions import SessionBusyError, StaleSessionError
from ..policies.base import MemoryState
from ..types import SessionKey


@dataclass
class _SessionSlot:
    memory: MemoryState | None = None
    epoch: int = 0
    active_lease_id: int | None = None


class SessionLease:
    """Exclusive checkout of one session's currently committed memory."""

    def __init__(
        self,
        store: SessionStore,
        key: SessionKey,
        epoch: int,
        lease_id: int,
        memory: MemoryState | None,
    ) -> None:
        self._store = store
        self.key = key
        self.epoch = epoch
        self.lease_id = lease_id
        self.memory = memory
        self._closed = False

    def cancelled(self) -> bool:
        return self._store._lease_cancelled(self)

    def commit(self, memory: MemoryState) -> None:
        if self._closed:
            raise StaleSessionError(f"session lease is already closed: {self.key!r}")
        self._store._commit(self, memory)
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            return
        self._store._rollback(self)
        self._closed = True


class SessionStore:
    """Own committed recurrent state and guard updates with per-session leases.

    Reset/cancel keep an epoch tombstone. Reset clears the committed episode;
    cancel invalidates only the in-flight turn and preserves the last commit.
    A late generation therefore cannot overwrite newer state.
    """

    def __init__(self) -> None:
        self._slots: dict[SessionKey, _SessionSlot] = {}
        self._next_lease_id = 1
        self._lock = threading.Lock()

    def checkout(self, key: SessionKey) -> SessionLease:
        with self._lock:
            slot = self._slots.setdefault(key, _SessionSlot())
            if slot.active_lease_id is not None:
                raise SessionBusyError(f"session already has an in-flight request: {key!r}")
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            slot.active_lease_id = lease_id
            return SessionLease(self, key, slot.epoch, lease_id, slot.memory)

    def checkout_many(self, keys: Sequence[SessionKey]) -> list[SessionLease]:
        """Checkout a group, releasing earlier leases if any later key is busy."""
        leases = []
        try:
            for key in keys:
                leases.append(self.checkout(key))
        except BaseException:
            for lease in leases:
                lease.rollback()
            raise
        return leases

    def commit_many(self, leases: Sequence[SessionLease], memories: Sequence[MemoryState]) -> None:
        """Atomically commit a rollout group after validating every lease."""
        if len(leases) != len(memories):
            raise ValueError("session leases and memories must have identical lengths")
        if not leases:
            raise ValueError("cannot commit an empty session group")
        with self._lock:
            for lease in leases:
                slot = self._slots.get(lease.key)
                if (
                    lease._closed
                    or slot is None
                    or slot.epoch != lease.epoch
                    or slot.active_lease_id != lease.lease_id
                ):
                    raise StaleSessionError(
                        f"session lease was invalidated before group commit: {lease.key!r}"
                    )
            for lease, memory in zip(leases, memories, strict=True):
                slot = self._slots[lease.key]
                slot.memory = memory
                slot.active_lease_id = None
                lease._closed = True

    def reset(self, keys: list[SessionKey] | tuple[SessionKey, ...]) -> None:
        self._invalidate(keys, clear_memory=True)

    def cancel(self, keys: list[SessionKey] | tuple[SessionKey, ...]) -> None:
        self._invalidate(keys, clear_memory=False)

    def reset_all(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                slot.memory = None
                slot.epoch += 1
                slot.active_lease_id = None

    def has_committed(self) -> bool:
        with self._lock:
            return any(slot.memory is not None for slot in self._slots.values())

    def has_inflight(self) -> bool:
        with self._lock:
            return any(slot.active_lease_id is not None for slot in self._slots.values())

    def committed(self, key: SessionKey) -> MemoryState | None:
        """Return the committed object for diagnostics/tests; callers must not mutate it."""
        with self._lock:
            slot = self._slots.get(key)
            return None if slot is None else slot.memory

    def _invalidate(
        self,
        keys: list[SessionKey] | tuple[SessionKey, ...],
        *,
        clear_memory: bool,
    ) -> None:
        with self._lock:
            for key in keys:
                slot = self._slots.setdefault(key, _SessionSlot())
                if clear_memory:
                    slot.memory = None
                slot.epoch += 1
                slot.active_lease_id = None

    def _lease_cancelled(self, lease: SessionLease) -> bool:
        with self._lock:
            slot = self._slots.get(lease.key)
            return (
                lease._closed
                or slot is None
                or slot.epoch != lease.epoch
                or slot.active_lease_id != lease.lease_id
            )

    def _commit(self, lease: SessionLease, memory: MemoryState) -> None:
        with self._lock:
            slot = self._slots.get(lease.key)
            if slot is None or slot.epoch != lease.epoch or slot.active_lease_id != lease.lease_id:
                raise StaleSessionError(f"session lease was invalidated before commit: {lease.key!r}")
            slot.memory = memory
            slot.active_lease_id = None

    def _rollback(self, lease: SessionLease) -> None:
        with self._lock:
            slot = self._slots.get(lease.key)
            if slot is not None and slot.epoch == lease.epoch and slot.active_lease_id == lease.lease_id:
                slot.active_lease_id = None
