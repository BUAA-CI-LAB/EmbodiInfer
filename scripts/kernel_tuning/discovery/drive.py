"""Run recorded inputs through one model's production inference path.

This is the preparation driver for model operator discovery::

    python -m scripts.kernel_tuning capture --model POLICY --output CAPTURE -- \\
        -m scripts.kernel_tuning.discovery.drive --policy POLICY --inputs INPUTS.pt [serving options]

The policy and engine are constructed with the same arguments as the serving
transports. ``requests`` inputs go through the policy's serving adapter, exactly
as an HTTP request would; ``observations`` inputs go through ``policy.collate``
and ``EngineCore.execute``. Recurrent policies keep one session per episode, so
memory grows the way it does in deployment. The capture records whatever this
inference executes; the driver itself chooses no operators or code paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.engine.serve.batching import BatchedServingAdapter
from embodiinfer.engine.serve.contracts import RawPolicyRequest
from embodiinfer.engine.serve.factory import (
    _parse_adapter_config,
    add_policy_arguments,
    build_serving_adapter,
)
from embodiinfer.policies.factory import make_policy
from embodiinfer.types import Observation, SessionKey

_KINDS = {"requests": RawPolicyRequest, "observations": Observation}


def save_inputs(path: Path, kind: str, episodes: Sequence[Sequence[Any]]) -> None:
    """Write episodes of serving requests or engine observations for :func:`main`."""
    if kind not in _KINDS:
        raise ValueError(f"input kind must be one of {sorted(_KINDS)}")
    torch.save({"kind": kind, "episodes": [list(episode) for episode in episodes]}, path)


def load_inputs(path: Path) -> tuple[str, list[list[Any]]]:
    """Read and validate inputs written by :func:`save_inputs`."""
    value = torch.load(path, map_location="cpu", weights_only=False)
    kind = value.get("kind") if isinstance(value, dict) else None
    episodes = value.get("episodes") if kind in _KINDS else None
    if (
        not isinstance(episodes, list)
        or not episodes
        or not all(isinstance(episode, list) and episode for episode in episodes)
        or not all(isinstance(item, _KINDS[kind]) for episode in episodes for item in episode)
    ):
        raise ValueError("inputs must contain non-empty episodes of one kind: requests or observations")
    return kind, episodes


def serve(args: argparse.Namespace, episodes: list[list[RawPolicyRequest]]) -> None:
    """Answer every request through the serving adapter, then end its sessions."""
    adapter = build_serving_adapter(args)
    try:
        for episode in episodes:
            for request in episode:
                adapter.infer(request)
            for session_id in dict.fromkeys(request.session_id for request in episode):
                adapter.reset(session_id)
    finally:
        if isinstance(adapter, BatchedServingAdapter):
            adapter.shutdown()


def execute(args: argparse.Namespace, episodes: list[list[Observation]]) -> None:
    """Execute every observation through the engine, one session per recurrent episode."""
    config = _parse_adapter_config(args.adapter_config)
    policy_kwargs = dict(config.get("policy_kwargs", {}))
    if args.checkpoint is not None:
        policy_kwargs["checkpoint"] = args.checkpoint
    policy = make_policy(args.policy, **policy_kwargs)
    core = EngineCore(
        policy,
        EngineConfig(
            device=args.device,
            dtype=args.dtype,
            max_batch_size=args.max_batch,
            max_wait_ms=args.max_wait_ms,
            num_steps=args.num_steps,
            use_cuda_graph=not args.no_cuda_graph,
            capture_full_loop=args.capture_full_loop,
        ),
    )
    for index, episode in enumerate(episodes):
        key = SessionKey(env_id="discovery", episode_id=index) if policy.is_recurrent else None
        for step, observation in enumerate(episode):
            batch = policy.collate([observation], [f"{index}-{step}"])
            core.execute(batch, session_ids=None if key is None else [key])
        if key is not None:
            core.reset_sessions([key])


def main(argv: list[str] | None = None) -> None:
    """Parse serving options and drive all recorded inputs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_policy_arguments(parser)
    parser.add_argument("--inputs", type=Path, required=True, help="Episodes written by save_inputs")
    args = parser.parse_args(argv)
    kind, episodes = load_inputs(args.inputs)
    (serve if kind == "requests" else execute)(args, episodes)


if __name__ == "__main__":
    main()
