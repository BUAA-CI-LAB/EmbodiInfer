#!/usr/bin/env python3
"""Serve PI0.5 to OpenPI's LIBERO client over OpenPI's websocket policy protocol.

OpenPI's examples/libero/main.py runs unchanged against this server: it sends observation/image
and observation/wrist_image (224x224 uint8, rotated and resize_with_pad'ed), observation/state
and prompt, and executes the first replan_steps actions of each returned chunk.

--backend ei serves EmbodiInfer static requests of an OpenPI PyTorch checkpoint. Each
--serve NAME:STEPS:RHO:PORT setting gets its own port; settings share one process, its weights and
its low-precision projections, and each request captures its own schedule. --backend flashrt
serves FlashRT's predict() on --port. Requests run one at a time on one event loop, with fresh
sampling noise per call from a seeded generator, as OpenPI's server samples. Replies carry
server_timing.infer_ms, and each setting's per-call latency is written to RESULTS/NAME/latency.json.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


class LatencyLog:
    def __init__(self, path: Path, meta: dict):
        self.path, self.meta, self.values = path, meta, []

    def add(self, milliseconds: float) -> None:
        self.values.append(round(milliseconds, 3))
        if len(self.values) % 200 == 0:
            self.flush()

    def flush(self) -> None:
        values = np.asarray(self.values or [np.nan])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            **self.meta,
            "calls": len(self.values),
            "policy_ms_p50": float(np.median(values)),
            "policy_ms_p95": float(np.percentile(values, 95)),
            "policy_ms": self.values,
        }))


class EmbodiInferSetting:
    """One (steps, rho) setting; it binds a new static request whenever the prompt changes."""

    def __init__(self, policy, processor, steps: int, rho: float, seed: int, log: LatencyLog):
        import torch

        self.policy, self.processor, self.steps, self.rho, self.log = policy, processor, steps, rho, log
        self.generator = torch.Generator().manual_seed(seed)
        self.prompt, self.request = None, None

    def infer(self, obs: dict) -> dict:
        from common import Request

        prompt = str(obs["prompt"])
        if prompt != self.prompt:
            if self.request is not None:
                self.request.close()
            self.request = Request(self.policy, self.processor, prompt, self.steps, self.rho)
            self.prompt = prompt
        start = time.perf_counter()
        self.request.frames[0] = obs["observation/image"]
        self.request.frames[1] = obs["observation/wrist_image"]
        raw = self.request.run(self.request.noise(self.generator))
        actions = self.request.actions(raw, obs["observation/state"])
        self.log.add((time.perf_counter() - start) * 1000)
        return {"actions": actions}


class FlashRtPolicy:
    def __init__(self, checkpoint: str, seed: int, log: LatencyLog):
        import flash_rt

        np.random.seed(seed)
        self.model = flash_rt.load_model(checkpoint, framework="torch", num_views=2, config="pi05", hardware="rtx_sm89")
        self.log = log

    def infer(self, obs: dict) -> dict:
        start = time.perf_counter()
        actions = self.model.predict(
            images=[obs["observation/image"], obs["observation/wrist_image"]], prompt=str(obs["prompt"])
        )
        actions = np.asarray(actions, dtype=np.float32)[:, :7]
        self.log.add((time.perf_counter() - start) * 1000)
        return {"actions": actions}


async def serve(policy, port: int) -> None:
    import websockets
    from openpi_client import msgpack_numpy
    from websockets.asyncio.server import serve as websocket_serve

    async def handler(websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
            except websockets.ConnectionClosed:
                return
            start = time.monotonic()
            try:
                action = policy.infer(obs)
            except Exception:
                # OpenPI's client raises the server traceback it receives as text.
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise
            action["server_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
            await websocket.send(packer.pack(action))

    async with websocket_serve(handler, "0.0.0.0", port, compression=None, max_size=None) as server:
        print(f"serving on 0.0.0.0:{port}", flush=True)
        await server.serve_forever()


def parse_setting(text: str) -> tuple[str, int, float, int]:
    name, steps, rho, port = text.split(":")
    return name, int(steps), float(rho), int(port)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("ei", "flashrt"), default="ei")
    parser.add_argument("--checkpoint", required=True, help="OpenPI PyTorch checkpoint directory")
    parser.add_argument("--results", type=Path, default=Path("runs/libero"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--openpi-config", default=None, help="OpenPI recipe JSON (default: pi05_libero)")
    parser.add_argument("--tokenizer", help="PaliGemma tokenizer.model, if not inside the checkpoint")
    parser.add_argument("--device", default="4090", help="EmbodiInfer preset device")
    parser.add_argument("--preset", default="fp8-fp8")
    parser.add_argument("--calibration", help="activation ranges for this checkpoint (calibrate.py)")
    parser.add_argument("--serve", nargs="+", default=["ei:10:1:8000"], help="EI settings NAME:STEPS:RHO:PORT")
    parser.add_argument("--port", type=int, default=8000, help="FlashRT port")
    args = parser.parse_args()

    logs, served = [], []
    if args.backend == "flashrt":
        log = LatencyLog(args.results / "flashrt" / "latency.json", {"backend": "flashrt"})
        logs.append(log)
        served.append((FlashRtPolicy(args.checkpoint, args.seed, log), args.port))
    else:
        from common import RECIPE, build_policy, load_model, load_recipe

        settings = [parse_setting(text) for text in args.serve]
        model, processor = load_model(args.checkpoint, load_recipe(args.openpi_config or RECIPE, args.tokenizer))
        policy = build_policy(
            model, processor, args.checkpoint,
            device=args.device, preset=args.preset, calibration=args.calibration, steps=settings[0][1],
        )
        for name, steps, rho, port in settings:
            meta = {"backend": "ei", "preset": args.preset, "steps": steps, "rho": rho}
            log = LatencyLog(args.results / name / "latency.json", meta)
            logs.append(log)
            served.append((EmbodiInferSetting(policy, processor, steps, rho, args.seed, log), port))

    def stop(*_):
        # A SystemExit raised inside the event loop does not stop it; flush and leave directly.
        for log in logs:
            log.flush()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)

    async def run_all():
        await asyncio.gather(*(serve(policy, port) for policy, port in served))

    try:
        asyncio.run(run_all())
    finally:
        for log in logs:
            log.flush()


if __name__ == "__main__":
    main()
