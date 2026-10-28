"""Replay Codex events through the actual pinned Humanize2 turn decoder."""

from __future__ import annotations

import itertools
import queue
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("hmz", reason="requires the optional Humanize2 tooling environment")
from hmz.coganchor.agents.codex import Failed, _AppServer  # noqa: E402


def decode(messages):
    """Exercise real routing, mailbox consumption and completion without a CLI or account."""
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
    # Keep the event replay bounded when no completion was sent. A blocked read
    # must fail this test rather than hanging the entire suite.
    server._take = lambda mailbox: mailbox.get_nowait()
    running = SimpleNamespace(turn=None, took=None, spends=None)
    return list(server.turn({"threadId": "thread"}, running))


RECONNECT = ("error", {"willRetry": True, "error": {"message": "Reconnecting... 1/5"}})
ANSWER = ("item/completed", {"item": {"type": "agentMessage", "text": "READY"}})
IDLE = ("thread/status/changed", {"status": {"type": "idle"}})


@pytest.mark.parametrize("idle_first", [False, True])
def test_recovered_turn_waits_for_successful_completion(idle_first):
    messages = [RECONNECT, ANSWER]
    if idle_first:
        messages.append(IDLE)
    messages.append(("turn/completed", {"turn": {"status": "completed"}}))
    events = decode(messages)
    assert events[-1].kind == "result"
    assert events[-1].text == "READY"


@pytest.mark.parametrize("retry_flag", [False, None, "true"])
def test_terminal_or_unclassified_error_still_fails(retry_flag):
    error = {"error": {"message": "terminal failure"}}
    if retry_flag is not None:
        error["willRetry"] = retry_flag
    with pytest.raises(Failed, match="terminal failure"):
        decode(
            [
                RECONNECT,
                ("error", error),
                IDLE,
                ("turn/completed", {"turn": {"status": "failed", "error": error["error"]}}),
            ]
        )


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_reconnect_does_not_hide_failed_or_cancelled_completion(status):
    with pytest.raises(Failed):
        decode([RECONNECT, IDLE, ("turn/completed", {"turn": {"status": status}})])


def test_idle_without_completion_keeps_waiting():
    with pytest.raises(queue.Empty):
        decode([RECONNECT, ANSWER, IDLE])
