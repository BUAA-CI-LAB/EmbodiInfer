"""Exception hierarchy for vvla.

Every error raised by the library derives from :class:`EmbodiInferError`, so callers can
catch the whole family with a single ``except EmbodiInferError``. Concrete subclasses add
structured context (e.g. which replica failed) rather than a bare string message.
"""

from __future__ import annotations


class EmbodiInferError(Exception):
    """Base class for all vvla-raised errors."""


class PolicyNotFoundError(EmbodiInferError):
    """A policy name was requested that no adapter has registered.

    Raised by :func:`~embodiinfer.policies.factory.make_policy`; the message lists the
    registered names and a close-match suggestion when one exists.
    """


class ObservationError(EmbodiInferError):
    """An observation handed to the engine has the wrong shape.

    Raised at the high-level entry (:meth:`~embodiinfer.engine.serve.api.EmbodiInfer.act`) with the
    expected and received shapes, rather than surfacing as a cryptic error deep
    inside the model forward.
    """


class SessionError(EmbodiInferError):
    """Base class for persistent policy-session lifecycle errors."""


class SessionRequiredError(SessionError):
    """A recurrent policy request omitted its explicit session identity."""


class SessionBusyError(SessionError):
    """A second request tried to use a session that already has an active lease."""


class StaleSessionError(SessionError):
    """A reset, cancel, or newer lease invalidated a late session commit."""


class SessionCancelledError(SessionError):
    """A recurrent generation was cancelled before it could commit."""


class UnsupportedRecurrentModeError(EmbodiInferError):
    """The first recurrent implementation was asked to use an unsupported runtime mode."""


class ReplicaExecutionError(EmbodiInferError):
    """A data-parallel replica failed to execute one or more requests.

    Raised by :class:`~embodiinfer.engine.parallel.data_parallel.DataParallelEngine` when a
    replica's ``execute`` raises and the failure is not recovered (either
    ``retry_on_healthy`` is off, or no healthy replica remains to retry on). The
    original exception is chained as ``__cause__``.

    Attributes:
        failures: ``(replica_id, exception)`` pairs, one per failed replica.
        num_failed: number of individual requests left unserved.
    """

    def __init__(
        self,
        failures: list[tuple[int, BaseException]],
        num_failed: int,
        message: str | None = None,
    ) -> None:
        self.failures = failures
        self.num_failed = num_failed
        if message is None:
            replica_ids = sorted({rid for rid, _ in failures})
            last = repr(failures[-1][1]) if failures else "unknown error"
            message = f"{num_failed} request(s) failed on replica(s) {replica_ids}: {last}"
        super().__init__(message)
