"""Minimal websocket policy server (optional; needs `pip install ".[serve]"`).

Protocol (msgpack): the client sends {"images", "state", "instruction_tokens",
"num_steps"} and receives {"actions": [H, A]}. Deliberately close to openpi's
websocket client shape so an existing openpi robot client can be pointed at a
vvla server with a thin shim. The async engine batches concurrent connections.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from ...policies.factory import make_policy
from ...types import Observation, SampleParams
from ..async_engine import AsyncEngine
from ..config import EngineConfig
from ..core import EngineCore


async def _serve(engine: AsyncEngine, host: str, port: int):
    import msgpack
    import websockets

    await engine.start()

    async def handler(ws):
        async for raw in ws:
            msg = msgpack.unpackb(raw, raw=False)
            obs = Observation(
                images=np.asarray(msg["images"], dtype=np.float32),
                state=np.asarray(msg["state"], dtype=np.float32),
                instruction_tokens=np.asarray(msg["instruction_tokens"], dtype=np.int64),
            )
            params = SampleParams(num_steps=int(msg.get("num_steps", 10)))
            chunk = await engine.generate(obs, params)
            reply = {"actions": chunk.actions.numpy().tolist(), "latency_ms": chunk.latency_ms}
            await ws.send(msgpack.packb(reply, use_bin_type=True))

    async with websockets.serve(handler, host, port, max_size=None):
        print(f"[vvla] serving on ws://{host}:{port}")
        await asyncio.Future()  # run forever


def main(argv: list | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="mock_flow_vla")
    p.add_argument("--preset", default="small")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    policy = make_policy(args.policy, preset=args.preset)
    core = EngineCore(policy, EngineConfig(device=args.device, max_batch_size=args.max_batch))
    engine = AsyncEngine(core)
    asyncio.run(_serve(engine, args.host, args.port))


if __name__ == "__main__":
    main()
