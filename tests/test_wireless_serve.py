from __future__ import annotations

import asyncio
import socket
import unittest

import pytest

from embodiinfer.engine.serve.contracts import ModelAction, ModelResult
from embodiinfer.engine.serve.service import PolicyService
from embodiinfer.engine.serve.wireless_server import (
    REQUEST_TAG,
    RESPONSE_TAG,
    RPC_SCHEMA,
    WirelessPolicyServer,
)

wireless_comm = pytest.importorskip("wireless_comm")
ConnectionClosedError = pytest.importorskip("wireless_comm.errors").ConnectionClosedError


class FakeAdapter:
    action_space = "pi05.action_chunk.v1"

    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self):
        return {"model": "fake", "action_space": self.action_space}

    def infer(self, request):
        self.calls += 1
        return ModelResult(
            action_space=self.action_space,
            actions=(ModelAction("action_chunk", {"data": [[float(request.step_id)]]}),),
            timing={},
        )

    def reset(self, session_id):
        pass


class FakeMultiPeerComm:
    """Expose one idle peer before one active peer to exercise fair admission."""

    def __init__(self) -> None:
        self.idle = wireless_comm.Peer("idle-robot", "127.0.0.1", 10001)
        self.active = wireless_comm.Peer("active-robot", "127.0.0.1", 10002)
        self.sent = asyncio.Event()
        self.sent_peer = None
        self._delivered = False

    def peers(self):
        return (self.idle, self.active)

    async def recv(self, peer, options):
        if peer == self.active and not self._delivered:
            self._delivered = True
            return {}, {
                "schema": RPC_SCHEMA,
                "kind": "request",
                "rpc_id": "health-1",
                "method": "health",
            }
        await asyncio.Future()

    async def send(self, payload, peer, *, piggypayload, options):
        self.sent_peer = peer
        self.sent.set()


class FakeReconnectingComm:
    """One peer that closes its connection, then comes back and sends a request."""

    def __init__(self, *, disconnects: int = 1) -> None:
        self.peer = wireless_comm.Peer("robot-1", "127.0.0.1", 10003)
        self.served = asyncio.Event()
        self.receives = 0
        self.disconnects_raised = 0
        self._disconnects = disconnects

    def peers(self):
        return (self.peer,)

    async def recv(self, peer, options):
        self.receives += 1
        if self.receives <= self._disconnects:
            self.disconnects_raised += 1
            raise ConnectionClosedError("peer closed its connection")
        if self.receives == self._disconnects + 1:
            return {}, {
                "schema": RPC_SCHEMA,
                "kind": "request",
                "rpc_id": "health-after-reconnect",
                "method": "health",
            }
        await asyncio.Future()

    async def send(self, payload, peer, *, piggypayload, options):
        self.served.set()


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def _rpc(comm, peer, method, payload, *, rpc_id, token="secret"):
    metadata = {
        "schema": RPC_SCHEMA,
        "kind": "request",
        "rpc_id": rpc_id,
        "method": method,
    }
    if token is not None:
        metadata["token"] = token
    await comm.send(
        payload,
        peer,
        piggypayload=metadata,
        options=wireless_comm.CommOptions(tag=REQUEST_TAG),
    )
    result, response = await comm.recv(
        peer,
        wireless_comm.CommOptions(tag=RESPONSE_TAG, timeout=2),
    )
    assert response["rpc_id"] == rpc_id
    return result, response


class WirelessPolicyServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        client_peer = wireless_comm.Peer("robot-1", "127.0.0.1", _unused_port())
        self.server_peer = wireless_comm.Peer("inference-1", "127.0.0.1", _unused_port())
        self.client_comm = await wireless_comm.Comm.create(
            local=client_peer,
            peers=[self.server_peer],
        )
        self.server_comm = await wireless_comm.Comm.create(
            local=self.server_peer,
            peers=[client_peer],
        )
        self.adapter = FakeAdapter()
        server = WirelessPolicyServer(
            self.server_comm,
            PolicyService(self.adapter),
            token="secret",
        )
        self.server_task = asyncio.create_task(server.serve_forever())

    async def asyncTearDown(self) -> None:
        self.server_task.cancel()
        await asyncio.gather(self.server_task, return_exceptions=True)
        await self.client_comm.close()
        await self.server_comm.close()

    async def test_wireless_ingress_shares_policy_session_semantics(self) -> None:
        opened, response = await _rpc(
            self.client_comm,
            self.server_peer,
            "open_session",
            {
                "schema": "vvla.policy.session.v1",
                "robot_id": "fr3",
                "action_space": "pi05.action_chunk.v1",
            },
            rpc_id="open-1",
        )
        self.assertEqual(response["status"], 200)
        step = {
            "schema": "vvla.policy.step.v1",
            "session_id": opened["session_id"],
            "request_id": "request-1",
            "step_id": 0,
            "instruction": "move",
            "state": {"joints": [0.0]},
            "metadata": {},
            "images": [
                {
                    "name": "wrist",
                    "mime_type": "image/jpeg",
                    "data": b"jpeg-bytes",
                }
            ],
        }
        first, response = await _rpc(
            self.client_comm,
            self.server_peer,
            "step",
            step,
            rpc_id="step-1",
        )
        retried, _ = await _rpc(
            self.client_comm,
            self.server_peer,
            "step",
            step,
            rpc_id="step-2",
        )

        self.assertEqual(response["status"], 200)
        self.assertEqual(retried, first)
        self.assertEqual(self.adapter.calls, 1)

    async def test_wireless_ingress_returns_structured_remote_errors(self) -> None:
        payload, response = await _rpc(
            self.client_comm,
            self.server_peer,
            "capabilities",
            {},
            rpc_id="unauthorized-1",
            token=None,
        )

        self.assertEqual(response["status"], 401)
        self.assertEqual(response["code"], "unauthorized")
        self.assertEqual(payload["schema"], "vvla.error.v1")

    async def test_real_peer_can_reconnect_without_restarting_server(self) -> None:
        await _rpc(self.client_comm, self.server_peer, "health", {}, rpc_id="before-close")
        client_peer = self.server_comm.peers()[0]
        await self.client_comm.close()
        self.client_comm = await wireless_comm.Comm.create(
            local=client_peer,
            peers=[self.server_peer],
        )
        result, response = await _rpc(
            self.client_comm, self.server_peer, "health", {}, rpc_id="after-reconnect"
        )
        self.assertEqual(response["status"], 200)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(self.server_task.done())

    async def test_idle_peer_does_not_reserve_in_flight_capacity(self) -> None:
        comm = FakeMultiPeerComm()
        server = WirelessPolicyServer(
            comm,
            PolicyService(FakeAdapter()),
            maximum_in_flight=1,
        )
        task = asyncio.create_task(server.serve_forever())
        try:
            await asyncio.wait_for(comm.sent.wait(), timeout=1)
            self.assertEqual(comm.sent_peer, comm.active)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class WirelessPolicyServerDisconnectTests(unittest.IsolatedAsyncioTestCase):
    """Disconnect handling, exercised with a fake transport instead of real peers."""

    async def test_peer_disconnect_does_not_stop_the_server(self) -> None:
        """A robot closing its connection must not take the server down."""

        comm = FakeReconnectingComm(disconnects=1)
        server = WirelessPolicyServer(comm, PolicyService(FakeAdapter()))
        task = asyncio.create_task(server.serve_forever())
        try:
            await asyncio.wait_for(comm.served.wait(), timeout=5)
            self.assertFalse(
                task.done(),
                "serve_forever returned after a peer disconnected",
            )
            self.assertEqual(comm.disconnects_raised, 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
