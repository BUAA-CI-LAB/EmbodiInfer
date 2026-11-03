#!/usr/bin/env python3
"""Per-call latency of EmbodiInfer static requests for several step counts and schedules.

Every setting binds the same prompt and cycles through recorded LIBERO observations
(observations.py). graph_ms covers session.run alone: frame upload, prefix, the denoise loop and
the action download. call_ms is what serve.py measures per request: it adds the frame copy, the
noise and unnormalization. Run it on an idle GPU.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RECIPE, Request, build_policy, load_model, load_recipe  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--openpi-config", default=str(RECIPE))
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="4090")
    parser.add_argument("--preset", default="fp8-fp8")
    parser.add_argument("--calibration")
    parser.add_argument("--settings", nargs="+", default=["10:1", "4:1", "3:1", "3:2", "2:1", "2:3"],
                        help="STEPS:RHO")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    settings = [(int(steps), float(rho)) for steps, rho in (text.split(":") for text in args.settings)]
    model, processor = load_model(args.checkpoint, load_recipe(args.openpi_config, args.tokenizer))
    policy = build_policy(model, processor, args.checkpoint, device=args.device, preset=args.preset,
                          calibration=args.calibration, steps=settings[0][0])
    data = np.load(args.observations)
    images, wrists, states = data["images"], data["wrists"], data["states"]
    prompt = str(data["prompts"][0])
    generator = torch.Generator().manual_seed(0)
    results = {}
    for steps, rho in settings:
        request = Request(policy, processor, prompt, steps, rho)
        graph_ms, call_ms = [], []
        for index in range(args.warmup + args.iterations):
            sample = index % len(images)
            start = time.perf_counter()
            request.frames[0] = images[sample]
            request.frames[1] = wrists[sample]
            noise = request.noise(generator)
            launched = time.perf_counter()
            raw = request.run(noise)
            finished = time.perf_counter()
            request.actions(raw, states[sample])
            end = time.perf_counter()
            if index >= args.warmup:
                graph_ms.append((finished - launched) * 1000)
                call_ms.append((end - start) * 1000)
        request.close()
        name = f"s{steps}_rho{rho:g}"
        results[name] = {
            "steps": steps,
            "rho": rho,
            "prompt_tokens": request.tokens,
            "graph_ms_p50": round(float(np.median(graph_ms)), 3),
            "graph_ms_p95": round(float(np.percentile(graph_ms, 95)), 3),
            "call_ms_p50": round(float(np.median(call_ms)), 3),
            "call_ms_p95": round(float(np.percentile(call_ms, 95)), 3),
            "call_hz_p50": round(1000.0 / float(np.median(call_ms)), 1),
        }
        print(name, json.dumps(results[name]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
