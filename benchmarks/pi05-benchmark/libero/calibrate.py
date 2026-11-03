#!/usr/bin/env python3
"""FP8 MLP activation ranges for an OpenPI PI0.5 checkpoint, from recorded LIBERO observations.

The BF16 model runs every observation (prefix and a full 10-step denoise loop) and records each
layer's gate/up and down input maxima with validate_optimizations.calibrate, which adds 10%
headroom. The result is a schema-2 calibration for one preset device, bound to the checkpoint's
SHA256. Down projections whose input range exceeds --protect-down-above stay BF16: one per-tensor
FP8 scale cannot serve outliers of that size and the bulk of the activations at once.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import RECIPE, load_model, load_recipe  # noqa: E402
from validate_optimizations import calibrate  # noqa: E402

from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--openpi-config", default=str(RECIPE))
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="4090")
    parser.add_argument("--protect-down-above", type=float, default=4000.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, processor = load_model(str(args.checkpoint), load_recipe(args.openpi_config, args.tokenizer))
    model._validation_checkpoint = args.checkpoint
    cameras = processor.config.image_keys
    data = np.load(args.observations)
    batches = []
    for image, wrist, state, prompt in zip(data["images"], data["wrists"], data["states"], data["prompts"]):
        frames = {key: torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                  for key, frame in zip(cameras, (image, wrist))}
        batch = processor.prepare(torch.from_numpy(state), frames, str(prompt))
        # Static requests run the exact prompt length, so leave out the padding positions.
        length = int(batch.masks.sum())
        batch = Pi05Batch([camera.to(torch.bfloat16).float() for camera in batch.images], batch.img_masks,
                          batch.tokens[:, :length], batch.masks[:, :length])
        batches.append(batch.to("cuda"))
    indices = list(range(len(batches)))
    horizon = processor.config.action_horizon
    with torch.inference_mode():
        towers = {"prefix": calibrate(model, batches, horizon, indices, prefix=True),
                  "action": calibrate(model, batches, horizon, indices, prefix=False)}

    def rows(maxima):
        return [{"gate_up": "fp8", "down": "bf16" if down > args.protect_down_above else "fp8",
                 "gate_up_max": round(float(gate), 4), "down_max": round(float(down), 4)}
                for gate, down in maxima]

    calibration = {
        "schema_version": 2,
        "numerics": "openpi_rlinf",
        "activation": "gelu_pytorch_exact",
        "devices": {args.device: {tower: {"fp8": rows(maxima)} for tower, maxima in towers.items()}},
        "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
        "source": {"observations": len(batches), "headroom": 1.1, "protect_down_above": args.protect_down_above},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calibration, indent=2) + "\n")
    for tower, maxima in towers.items():
        for index, (gate, down) in enumerate(maxima):
            print(f"{tower} {index:2d} gate_up_max={gate:9.3f} down_max={down:9.3f}", flush=True)


if __name__ == "__main__":
    main()
