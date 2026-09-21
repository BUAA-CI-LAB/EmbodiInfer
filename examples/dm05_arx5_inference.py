"""Offline ARX5 smoke inference with Dexmal/DM05-Table30v2-ARX5.

This script only computes and prints an action chunk.  It does not connect to
or command a robot.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from embodiinfer import Observation
from embodiinfer.engine import EngineConfig, EngineCore
from embodiinfer.policies import make_policy
from embodiinfer.policies.dm05.processor_dm05 import pack_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--instruction", default="Press the button.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Repeat three times in ARX5 order: cam_global, cam_side, cam_arm. Uses black images if omitted.",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image and len(args.image) != 3:
        raise ValueError("ARX5 needs exactly three --image values")
    images = (
        [Image.open(path).convert("RGB") for path in args.image]
        if args.image
        else [Image.fromarray(np.zeros((728, 728, 3), dtype=np.uint8)) for _ in range(3)]
    )
    policy = make_policy(
        "dm05",
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        robot_type="ARX5",
        output_action_dim=7,
        action_mode="relative",
        is_history=False,
    )
    core = EngineCore(
        policy,
        EngineConfig(device="cuda", dtype="auto", use_cuda_graph=False),
    )
    pixels, image_sizes = pack_images(images)
    observation = Observation(
        images=pixels,
        state=np.zeros(7, dtype=np.float32),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction=args.instruction,
        metadata={
            "image_sizes": image_sizes,
            "robot_type": "ARX5",
            "state_desc": ["eef"] * 6 + ["gripper"],
        },
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    batch = policy.collate([observation], ["arx5-smoke"])
    chunks = core.execute(batch, num_steps=args.num_steps, generator=generator)
    action = chunks[0].actions
    print(f"shape={tuple(action.shape)} finite={bool(action.isfinite().all())}")
    print(action)


if __name__ == "__main__":
    main()
