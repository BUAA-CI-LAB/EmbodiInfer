"""Compatibility fixes for the pinned Humanize2 runtime used by developer tooling."""

from __future__ import annotations

from functools import wraps
from typing import Any


def configure_humanize() -> None:
    """Install the process-local Codex event fix before opening tooling sessions.

    Humanize2 at the repository's pinned revision remembers every ``error`` until
    ``turn/completed``. Codex 0.162.1 can emit ``idle`` first; the decoder then
    replays a successful turn because of an earlier reconnect notice. Only
    explicit nonterminal notifications are filtered. Terminal errors, cancellation,
    budgets and permissions retain their original handling. Installed packages
    and the user's CLI configuration are never modified.
    """
    from hmz.coganchor.agents.codex import _AppServer

    original = _AppServer._route
    if getattr(original, "_embodiinfer_retry_notice_fix", False):
        return

    @wraps(original)
    def route(self: _AppServer, message: dict[str, Any]) -> None:
        if (
            "id" not in message
            and message.get("method") == "error"
            and (message.get("params") or {}).get("willRetry") is True
        ):
            return
        original(self, message)

    route._embodiinfer_retry_notice_fix = True
    _AppServer._route = route
