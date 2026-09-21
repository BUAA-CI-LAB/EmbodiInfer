"""WirelessComm ingress for the versioned EmbodiInfer policy service."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .batching import BatchedServingAdapter
from .contracts import ServeError, parse_structured_step
from .factory import add_policy_arguments, build_serving_adapter
from .service import PolicyService

if TYPE_CHECKING:
    from wireless_comm import Comm, Peer

RPC_SCHEMA = "embodiinfer.policy.rpc.v1"
REQUEST_TAG = 0x56564C41
RESPONSE_TAG = 0x56564C42

# A peer closing its connection ends that session, not the server. Wait briefly
# before receiving again so a peer that is gone for good cannot spin this loop.
PEER_RECONNECT_DELAY_S = 0.5

_LOG = logging.getLogger(__name__)


class WirelessPolicyServer:
    """Dispatch WirelessComm RPC requests to a shared :class:`PolicyService`."""

    def __init__(
        self,
        comm: Comm,
        service: PolicyService,
        *,
        token: str | None = None,
        maximum_images: int = 8,
        maximum_image_bytes: int = 16 * 1024 * 1024,
        maximum_in_flight: int = 64,
    ) -> None:
        if maximum_images < 1 or maximum_image_bytes < 1 or maximum_in_flight < 1:
            raise ValueError("wireless serving limits must be positive")
        self.comm = comm
        self.service = service
        self.token = token
        self.maximum_images = maximum_images
        self.maximum_image_bytes = maximum_image_bytes
        self._capacity = asyncio.Semaphore(maximum_in_flight)
        self._requests: set[asyncio.Task[None]] = set()

    async def serve_forever(self) -> None:
        """Receive requests from every configured peer until cancelled."""

        loops = [
            asyncio.create_task(
                self._serve_peer(peer),
                name=f"embodiinfer-wireless-rx-{peer.node_id}",
            )
            for peer in self.comm.peers()
        ]
        if not loops:
            raise ValueError("wireless policy server requires at least one peer")
        try:
            await asyncio.gather(*loops)
        finally:
            for loop in loops:
                loop.cancel()
            await asyncio.gather(*loops, return_exceptions=True)
            if self._requests:
                await asyncio.gather(*self._requests, return_exceptions=True)

    async def _serve_peer(self, peer: Peer) -> None:
        from wireless_comm import CommOptions
        from wireless_comm.errors import ConnectionClosedError

        while True:
            try:
                payload, metadata = await self.comm.recv(
                    peer,
                    CommOptions(tag=REQUEST_TAG),
                )
            except ConnectionClosedError:
                # The robot side closes its connection when a run ends. That is a
                # normal end of session, so keep waiting for it to come back
                # instead of letting the error escape through ``serve_forever``,
                # which would stop the receiver loops of every other peer too.
                _LOG.info("peer %s disconnected; waiting for reconnect", peer.node_id)
                await asyncio.sleep(PEER_RECONNECT_DELAY_S)
                continue
            await self._capacity.acquire()
            task = asyncio.create_task(
                self._handle(peer, payload, metadata),
                name=f"embodiinfer-wireless-request-{peer.node_id}",
            )
            self._requests.add(task)
            task.add_done_callback(self._request_done)

    async def _handle(
        self,
        peer: Peer,
        payload: object,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        try:
            try:
                rpc_id, method = self._parse_request_metadata(metadata)
            except ServeError as error:
                _LOG.warning(
                    "rejected malformed wireless request from %s: %s",
                    peer.node_id,
                    error,
                )
                return

            try:
                result = await asyncio.to_thread(
                    self._dispatch,
                    method,
                    payload,
                    metadata,
                )
                response_metadata = {
                    "schema": RPC_SCHEMA,
                    "kind": "response",
                    "rpc_id": rpc_id,
                    "status": 200,
                }
                response_payload: object = result
            except ServeError as error:
                response_metadata = {
                    "schema": RPC_SCHEMA,
                    "kind": "response",
                    "rpc_id": rpc_id,
                    "status": error.status,
                    "code": error.code,
                }
                response_payload = {
                    "schema": "embodiinfer.error.v1",
                    "code": error.code,
                    "message": error.message,
                }
            except Exception:
                _LOG.exception("unhandled EmbodiInfer wireless request failure")
                response_metadata = {
                    "schema": RPC_SCHEMA,
                    "kind": "response",
                    "rpc_id": rpc_id,
                    "status": 500,
                    "code": "internal_error",
                }
                response_payload = {
                    "schema": "embodiinfer.error.v1",
                    "code": "internal_error",
                    "message": "internal server error",
                }

            from wireless_comm import CommOptions

            await self.comm.send(
                response_payload,
                peer,
                piggypayload=response_metadata,
                options=CommOptions(tag=RESPONSE_TAG),
            )
        finally:
            self._capacity.release()

    def _request_done(self, task: asyncio.Task[None]) -> None:
        self._requests.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOG.warning("wireless response delivery failed: %s", error)

    def _dispatch(
        self,
        method: str,
        payload: object,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if method != "health":
            self._require_token(metadata.get("token"))
        if method == "health":
            return {
                "schema": "embodiinfer.health.v1",
                "status": "ok",
                "service": "embodiinfer-wireless",
            }
        if method == "capabilities":
            return {
                "schema": "embodiinfer.policy.capabilities.v1",
                "server": "embodiinfer-wireless",
                "adapter": self.service.adapter.capabilities(),
            }
        if not isinstance(payload, Mapping):
            raise ServeError(400, "invalid_request", "request payload must be an object")
        if method == "open_session":
            return self.service.open_session(payload)
        if method == "step":
            request = parse_structured_step(
                payload,
                maximum_images=self.maximum_images,
                maximum_image_bytes=self.maximum_image_bytes,
            )
            return self.service.step(request)
        if method == "reset":
            session_id = payload.get("session_id")
            body = payload.get("body")
            if not isinstance(session_id, str) or not isinstance(body, Mapping):
                raise ServeError(400, "invalid_request", "reset payload is invalid")
            return self.service.reset(session_id, body)
        if method == "close":
            session_id = payload.get("session_id")
            if not isinstance(session_id, str):
                raise ServeError(400, "invalid_request", "close payload is invalid")
            self.service.close(session_id)
            return {"schema": "embodiinfer.policy.session.close.v1", "ok": True}
        raise ServeError(404, "method_not_found", f"unknown policy method {method!r}")

    @staticmethod
    def _parse_request_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> tuple[str, str]:
        if not isinstance(metadata, Mapping):
            raise ServeError(400, "invalid_rpc", "RPC metadata is required")
        if metadata.get("schema") != RPC_SCHEMA or metadata.get("kind") != "request":
            raise ServeError(400, "invalid_rpc", "unsupported RPC envelope")
        rpc_id = metadata.get("rpc_id")
        method = metadata.get("method")
        if not isinstance(rpc_id, str) or not rpc_id:
            raise ServeError(400, "invalid_rpc", "rpc_id is required")
        if not isinstance(method, str) or not method:
            raise ServeError(400, "invalid_rpc", "method is required")
        return rpc_id, method

    def _require_token(self, supplied: object) -> None:
        if self.token is None:
            return
        value = supplied if isinstance(supplied, str) else ""
        if not hmac.compare_digest(value, self.token):
            raise ServeError(401, "unauthorized", "missing or invalid token")


async def _run(args: argparse.Namespace, service: PolicyService) -> None:
    from wireless_comm import Comm, load_runtime_config

    config = load_runtime_config(args.comm_config)
    comm = await Comm.create(
        local=config.local,
        peers=config.peers,
        bind_host=config.bind_host,
        config=config.comm,
    )
    server = WirelessPolicyServer(
        comm,
        service,
        token=args.token,
        maximum_images=args.max_images,
        maximum_image_bytes=args.max_image_bytes,
        maximum_in_flight=args.max_in_flight,
    )
    print(
        f"[embodiinfer-wireless] policy={args.policy} device={args.device} "
        f"action_space={service.action_space} node={config.local.node_id} "
        f"listen={config.bind_host}:{config.local.port}"
    )
    try:
        await server.serve_forever()
    finally:
        await comm.close()


def main(argv: list[str] | None = None) -> None:
    """Run a EmbodiInfer policy server on a configured WirelessComm node."""

    parser = argparse.ArgumentParser()
    add_policy_arguments(parser)
    parser.add_argument("--comm-config", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-image-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=1024)
    parser.add_argument("--idempotency-cache-size", type=int, default=1024)
    parser.add_argument("--max-in-flight", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        adapter = build_serving_adapter(args)
        service = PolicyService(
            adapter,
            maximum_sessions=args.max_sessions,
            idempotency_cache_size=args.idempotency_cache_size,
        )
        try:
            asyncio.run(_run(args, service))
        finally:
            if isinstance(adapter, BatchedServingAdapter):
                adapter.shutdown()
    except ValueError as error:
        parser.error(str(error))


__all__ = [
    "REQUEST_TAG",
    "RESPONSE_TAG",
    "RPC_SCHEMA",
    "WirelessPolicyServer",
]


if __name__ == "__main__":
    main()
