"""Minimal client for the EmbodiInfer websocket policy server.

Start a server in one terminal (mock policy, no checkpoint needed):

    python -m embodiinfer.engine.serve.server --policy mock_flow_vla --preset small --device cpu --port 8000

then query it from another:

    python examples/serve_client.py --port 8000

The wire protocol (msgpack) is deliberately close to openpi's websocket client,
so an existing openpi robot client can point at an EmbodiInfer server with a thin shim.
Requires the ``serve`` extra (``uv sync --frozen --extra serve``).
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np


async def query(host: str, port: int, num_cameras: int, image_size: int, state_dim: int, lang_len: int):
    import msgpack
    import websockets

    request = {
        "images": np.random.rand(num_cameras, 3, image_size, image_size).astype("float32").tolist(),
        "state": np.random.rand(state_dim).astype("float32").tolist(),
        "instruction_tokens": np.random.randint(0, 32000, size=(lang_len,)).astype("int64").tolist(),
        "num_steps": 10,
    }
    async with websockets.connect(f"ws://{host}:{port}", max_size=None) as ws:
        await ws.send(msgpack.packb(request, use_bin_type=True))
        reply = msgpack.unpackb(await ws.recv(), raw=False)

    actions = np.asarray(reply["actions"])
    print(f"[client] action chunk shape={actions.shape}  server latency={reply['latency_ms']:.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cameras", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--state-dim", type=int, default=14)
    ap.add_argument("--lang-len", type=int, default=48)
    args = ap.parse_args()
    asyncio.run(query(args.host, args.port, args.cameras, args.image_size, args.state_dim, args.lang_len))


if __name__ == "__main__":
    main()
