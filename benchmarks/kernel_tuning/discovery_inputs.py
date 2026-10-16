"""Export benchmark data as end-to-end inputs for model operator discovery.

The output is read by ``scripts.kernel_tuning.discovery.drive``. Data selection
reuses each benchmark's own loader, so discovery sees the same images, states,
instructions and episode order as the measurement. Run in the model's runtime::

    python -m benchmarks.kernel_tuning.discovery_inputs pi05 --config benchmarks/pi05-benchmark/config.yaml \\
        --output pi05-inputs.pt --serving-config pi05-serving.json
    python -m benchmarks.kernel_tuning.discovery_inputs qwen --kind low --config benchmarks/qwenvl-benchmark/config.yaml \\
        --output qwen-low-inputs.pt
    python -m benchmarks.kernel_tuning.discovery_inputs streamvln --config benchmarks/streamvln-benchmark/config.yaml \\
        --output streamvln-inputs.pt

pi05 and StreamVLN produce serving requests (their deployment entry); Qwen
navigation profiles produce engine observations. Each episode keeps its frame
order so recurrent memory grows as it does in deployment.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from scripts.kernel_tuning.discovery.drive import save_inputs

from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest

_ROOT = Path(__file__).resolve().parents[2]
_PI05_IMAGES = (
    ("agentview_rgb", "observation.images.image"),
    ("eye_in_hand_rgb", "observation.images.image2"),
)
_PI05_STATE = ("ee_pos", "ee_ori", "gripper_states")


def _benchmark(directory: str) -> ModuleType:
    """Import a benchmark script whose directory name is not a Python package."""
    path = _ROOT / "benchmarks" / directory / "benchmark.py"
    spec = importlib.util.spec_from_file_location(f"_discovery_{directory.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the script executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected a benchmark schema_version: 1 mapping")
    return config


def _png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def pi05(args: argparse.Namespace) -> list[list[RawPolicyRequest]]:
    """LIBERO frames as pi05 serving requests, plus the matching adapter config."""
    from PIL import Image

    module = _benchmark("pi05-benchmark")
    samples = module.load_libero(_config(args.config)["dataset"])[: args.samples]
    requests = []
    for index, sample in enumerate(samples):
        raw = module.read_libero(sample)
        images = []
        for source, field in _PI05_IMAGES:
            image = raw[source]
            if sample.image_convention == "opengl":
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            images.append(RawImage(field, "image/png", _png(image)))
        requests.append(
            RawPolicyRequest(
                session_id=f"libero-{index}",
                request_id=f"libero-{index}",
                step_id=0,
                instruction=sample.instruction,
                state={name: raw[name].tolist() for name in _PI05_STATE},
                images=tuple(images),
                metadata={},
            )
        )
    serving = {
        "state_fields": list(_PI05_STATE),
        "image_fields": [field for _, field in _PI05_IMAGES],
        "policy_kwargs": json.loads(args.policy_kwargs),
    }
    args.serving_config.write_text(json.dumps(serving, indent=2) + "\n", encoding="utf-8")
    # pi05 is stateless: one request per session, all in a single episode.
    return [requests]


def qwen(args: argparse.Namespace) -> list[list[Any]]:
    """Consecutive navigation frames as Qwen engine observations, one episode each."""
    module = _benchmark("qwenvl-benchmark")
    spec = {**_config(args.config)["datasets"][0], "episode_limit": args.episodes}
    episodes = []
    for episode in module.load_navigation(spec):
        # Panoramic inputs read two neighbouring frames on either side of a step.
        steps = range(2, min(len(episode.frames) - 2, 2 + args.steps))
        episodes.append(
            [
                module.prepare_sample(
                    episode, step, 0, args.kind, module.read_images(episode, step, 0, args.kind)
                ).observation
                for step in steps
            ]
        )
    return episodes


def streamvln(args: argparse.Namespace) -> list[list[RawPolicyRequest]]:
    """Consecutive navigation frames as StreamVLN serving requests, one session per episode."""
    from embodiinfer.policies.streamvln.serving import STREAMVLN_IMAGE_FIELD

    module = _benchmark("streamvln-benchmark")
    spec = {**_config(args.config)["datasets"][0], "episode_limit": args.episodes}
    episodes = []
    for episode in module.load_navigation(spec):
        session = f"{episode.dataset}-{episode.episode_id}"
        episodes.append(
            [
                RawPolicyRequest(
                    session_id=session,
                    request_id=f"{session}-{step}",
                    step_id=step,
                    instruction=episode.instruction,
                    state={},
                    images=(RawImage(STREAMVLN_IMAGE_FIELD, "image/jpeg", path.read_bytes()),),
                    metadata={},
                )
                for step, path in enumerate(episode.frames[: args.steps])
            ]
        )
    return episodes


def main(argv: list[str] | None = None) -> None:
    """Write one model's discovery inputs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    models = parser.add_subparsers(dest="model", required=True)
    for name in ("pi05", "qwen", "streamvln"):
        model = models.add_parser(name)
        model.add_argument("--config", type=Path, required=True, help="The model benchmark's YAML config")
        model.add_argument("--output", type=Path, required=True)
        if name == "pi05":
            model.add_argument("--samples", type=int, default=4)
            model.add_argument("--serving-config", type=Path, required=True, help="Adapter config to write")
            model.add_argument(
                "--policy-kwargs", default="{}", help="JSON builder options for the adapter config"
            )
        else:
            model.add_argument("--episodes", type=int, default=2)
            model.add_argument("--steps", type=int, default=8, help="Frames per episode")
        if name == "qwen":
            model.add_argument("--kind", choices=("low", "panoramic"), required=True)
    args = parser.parse_args(argv)
    episodes = {"pi05": pi05, "qwen": qwen, "streamvln": streamvln}[args.model](args)
    save_inputs(args.output, "observations" if args.model == "qwen" else "requests", episodes)
    print(json.dumps({"inputs": str(args.output), "episodes": [len(episode) for episode in episodes]}))


if __name__ == "__main__":
    main()
