#!/usr/bin/env python3
"""Record settled first observations of LIBERO tasks, preprocessed like examples/libero/main.py.

Runs in OpenPI's LIBERO client environment (libero and openpi-client). Each task and initial
state contributes one observation after the 10 settle steps: the base and wrist frames rotated
and resize_with_pad'ed to 224x224, the 8-value state and the prompt. calibrate.py and latency.py
read the resulting .npz.
"""

import argparse
import math
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--inits", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    images, wrists, states, prompts, tags = [], [], [], [], []
    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(args.seed)
            for init in args.inits:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[init])
                for _ in range(10):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                for frames, key in ((images, "agentview_image"), (wrists, "robot0_eye_in_hand_image")):
                    frame = np.ascontiguousarray(obs[key][::-1, ::-1])
                    frames.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, 224, 224)))
                states.append(np.concatenate((
                    obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"],
                )).astype(np.float32))
                prompts.append(task.language)
                tags.append(f"{suite_name}/{task_id}/{init}")
            env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=np.stack(images), wrists=np.stack(wrists), states=np.stack(states),
                        prompts=np.array(prompts), tags=np.array(tags))
    print(f"wrote {args.out}: {len(images)} observations", flush=True)


if __name__ == "__main__":
    main()
