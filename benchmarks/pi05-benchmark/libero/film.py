#!/usr/bin/env python3
"""Film LIBERO episodes under OpenPI's evaluation protocol with the policy rate on screen.

Runs in OpenPI's LIBERO client environment against any server of OpenPI's websocket protocol
(serve.py, or OpenPI's own scripts/serve_policy.py). The episode follows examples/libero/main.py:
256x256 renders, 10 settle steps, frames rotated and resize_with_pad'ed to 224, and the first
replan_steps actions of every chunk. After success it continues for --extra-steps, so the film
also shows the placed object settling.

The film plays in real time: each control step lasts 1/control_hz, and a step that requested a
new chunk also lasts the server's inference time (server_timing.infer_ms). One untimed request
on the settled first observation precedes each episode, so per-prompt setup is not filmed.
Each episode's JSON reports grasp_attempts, the open-to-closed switches of the executed gripper
command (LIBERO: -1 open, +1 close) before success, to tell single grasps from re-grasps.
"""

import argparse
import collections
import json
import math
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy

DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line]


def grasp_attempts(gripper, until):
    closed = [value > 0 for value in gripper[:until]]
    return sum(1 for before, after in zip([False] + closed[:-1], closed) if after and not before)


def write_film(path, frames, step_latency_ms, policy_ms, control_ms, label, prompt, success_step, fps=50):
    import imageio.v2 as imageio

    starts = np.concatenate(([0.0], np.cumsum(np.asarray(step_latency_ms) + control_ms))) / 1000.0
    total = starts[-1]
    complete = starts[success_step] if success_step is not None else None
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    prompt_lines = wrap(prompt, 52)[:2]
    for tick in range(int(total * fps) + fps):
        now = tick / fps
        step = min(int(np.searchsorted(starts, now, side="right")) - 1, len(frames) - 1)
        canvas = np.full((704, 512, 3), 16, np.uint8)
        cv2.putText(canvas, label, (12, 28), FONT, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        canvas[40:552] = cv2.resize(frames[step], (512, 512), interpolation=cv2.INTER_NEAREST)
        seen = [value for value, start in zip(step_latency_ms, starts) if value > 0 and start <= now] or policy_ms[:1]
        median = float(np.median(seen))
        cv2.putText(canvas, f"{1000.0 / median:.1f} Hz", (12, 610), FONT, 1.3, (80, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"{median:.1f} ms median policy", (262, 586), FONT, 0.5, (220, 220, 220), 1,
                    cv2.LINE_AA)
        executed = min(step + 1, len(step_latency_ms))
        cv2.putText(canvas, f"{len(seen)} inferences / {executed} control steps", (262, 608), FONT, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
        if complete is not None and now >= complete:
            status, color = f"task complete at {complete:.1f} s", (120, 230, 140)
        elif step >= len(step_latency_ms):
            status, color = f"stopped at {total:.1f} s", (90, 120, 240)
        else:
            status, color = f"t = {now:.1f} s", (220, 220, 220)
        cv2.putText(canvas, status, (262, 630), FONT, 0.5, color, 1, cv2.LINE_AA)
        for index, line in enumerate(prompt_lines):
            cv2.putText(canvas, line, (12, 664 + 22 * index), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        writer.append_data(canvas)
    writer.close()


def run_episode(env, suite, task, task_id, init, client, args):
    def request(obs):
        render = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        return render, {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(render, 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
            "observation/state": np.concatenate((
                obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"],
            )),
            "prompt": task.language,
        }

    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[init])
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    client.infer(request(obs)[1])
    plan = collections.deque()
    frames, step_latency_ms, policy_ms, gripper = [], [], [], []
    success_step = None
    while len(step_latency_ms) < MAX_STEPS[args.suite]:
        render, element = request(obs)
        latency = 0.0
        if not plan:
            result = client.infer(element)
            plan.extend(result["actions"][: args.replan_steps])
            latency = float(result["server_timing"]["infer_ms"])
            policy_ms.append(latency)
        frames.append(render)
        step_latency_ms.append(latency)
        action = plan.popleft()
        gripper.append(float(action[6]))
        obs, _, done, _ = env.step(action.tolist())
        if done and success_step is None:
            success_step = len(step_latency_ms)
        if success_step is not None and len(step_latency_ms) >= success_step + args.extra_steps:
            break
    frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

    control_ms = 1000.0 / args.control_hz
    report = {
        "suite": args.suite,
        "task": task_id,
        "init": init,
        "prompt": task.language,
        "success": success_step is not None,
        "success_step": success_step,
        "grasp_attempts": grasp_attempts(gripper, success_step or len(gripper)),
        "released_after_success": bool(success_step is not None and any(v < 0 for v in gripper[success_step:])),
        "control_steps": len(step_latency_ms),
        "inferences": len(policy_ms),
        "policy_ms_p50": round(float(np.median(policy_ms)), 3),
        "task_complete_s": (
            round(float(sum(step_latency_ms[:success_step]) + control_ms * success_step) / 1000.0, 3)
            if success_step is not None else None
        ),
        "policy_ms": [round(value, 3) for value in policy_ms],
        "gripper": [round(value, 3) for value in gripper],
    }
    stem = args.out_dir / f"task{task_id}_init{init}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1))
    write_film(str(stem.with_suffix(".mp4")), frames, step_latency_ms, policy_ms, control_ms, args.label,
               task.language, success_step)
    print(json.dumps({key: value for key, value in report.items() if key not in ("policy_ms", "gripper")}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", type=int, nargs="+", default=[0])
    parser.add_argument("--inits", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--extra-steps", type=int, default=40)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--label", required=True, help="title line of the film")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        for init in args.inits:
            run_episode(env, suite, task, task_id, init, client, args)
        env.close()


if __name__ == "__main__":
    main()
