"""Replay Codex events through the actual pinned Humanize2 turn decoder."""

from __future__ import annotations

import itertools
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("hmz", reason="requires the optional Humanize2 tooling environment")
from hmz.coganchor.agents.codex import Failed, _AppServer  # noqa: E402
from scripts.kernel_tuning.humanize_compat import configure_humanize  # noqa: E402


def decode(monkeypatch, messages):
    """Exercise real routing, mailbox consumption and completion without a CLI or account."""
    monkeypatch.setattr(_AppServer, "_route", _AppServer._route)
    configure_humanize()
    server = object.__new__(_AppServer)
    server._pending = itertools.count(1)
    server._routing = threading.Lock()
    server._gone = False
    server._waiting = {}
    server._listening = {}
    server._turning = {}
    server._counted = {}
    server._argv = ["codex", "app-server"]

    def write(_request):
        for method, params in messages:
            server._route({"method": method, "params": {"threadId": "thread", "turnId": "turn", **params}})

    server._write = write
    running = SimpleNamespace(turn=None, took=None, spends=None)
    return list(server.turn({"threadId": "thread"}, running))


RECONNECT = ("error", {"willRetry": True, "error": {"message": "Reconnecting... 1/5"}})
ANSWER = ("item/completed", {"item": {"type": "agentMessage", "text": "READY"}})
IDLE = ("thread/status/changed", {"status": {"type": "idle"}})


@pytest.mark.parametrize("with_completion", [False, True])
def test_recovered_turn_does_not_replay_after_idle(monkeypatch, with_completion):
    # Codex 0.162.1 was observed to send idle before turn/completed. The decoder
    # exits on idle, so a late completion cannot clear an earlier retry notice.
    messages = [RECONNECT, ANSWER, IDLE]
    if with_completion:
        messages.append(("turn/completed", {"turn": {"status": "completed"}}))
    events = decode(monkeypatch, messages)
    assert events[-1].kind == "result"
    assert events[-1].text == "READY"


@pytest.mark.parametrize("retry_flag", [False, None, "true"])
def test_terminal_or_unclassified_error_still_fails(monkeypatch, retry_flag):
    error = {"error": {"message": "terminal failure"}}
    if retry_flag is not None:
        error["willRetry"] = retry_flag
    with pytest.raises(Failed, match="terminal failure"):
        decode(monkeypatch, [RECONNECT, ("error", error), IDLE])


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_reconnect_does_not_hide_failed_or_cancelled_completion(monkeypatch, status):
    with pytest.raises(Failed):
        decode(monkeypatch, [RECONNECT, ("turn/completed", {"turn": {"status": status}})])


def test_configuration_is_idempotent(monkeypatch):
    monkeypatch.setattr(_AppServer, "_route", _AppServer._route)
    configure_humanize()
    installed = _AppServer._route
    configure_humanize()
    assert _AppServer._route is installed
