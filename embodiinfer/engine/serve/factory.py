"""Shared command-line construction for policy serving transports."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...policies.base import VLAPolicy
from ...policies.factory import make_policy
from ..config import EngineConfig
from ..core import EngineCore
from .batching import BatchedServingAdapter
from .contracts import ServingAdapter


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model and engine arguments shared by all serving transports."""

    parser.add_argument("--policy", default="pi05")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path or hub id (required for pi05)",
    )
    parser.add_argument(
        "--adapter-config",
        default=None,
        help="JSON file containing adapter config",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--max-wait-ms", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--capture-full-loop", action="store_true")


def build_serving_adapter(args: argparse.Namespace) -> ServingAdapter:
    """Build one model-specific serving adapter from parsed CLI arguments."""

    if type(args.max_batch) is not int or args.max_batch < 1:
        raise ValueError("--max-batch must be a positive integer")
    max_wait_ms = getattr(args, "max_wait_ms", 5.0)
    if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
        raise ValueError("--max-wait-ms must be finite and nonnegative")
    adapter_config = _parse_adapter_config(args.adapter_config)
    policy_kwargs = dict(adapter_config.pop("policy_kwargs", {}))
    if args.checkpoint is not None:
        policy_kwargs["checkpoint"] = args.checkpoint
    policy_kwargs.setdefault("load_device", args.device)
    policy = make_policy(args.policy, **policy_kwargs)
    core = EngineCore(
        policy,
        EngineConfig(
            device=args.device,
            dtype=args.dtype,
            max_batch_size=args.max_batch,
            max_wait_ms=max_wait_ms,
            num_steps=args.num_steps,
            use_cuda_graph=not args.no_cuda_graph,
            capture_full_loop=args.capture_full_loop,
        ),
    )
    adapter = _build_adapter(
        policy,
        core,
        policy_kwargs.get("checkpoint"),
        adapter_config,
    )
    if args.max_batch > 1:
        return BatchedServingAdapter(adapter, max_batch=args.max_batch, max_wait_ms=max_wait_ms)
    return adapter


def _build_adapter(
    policy: VLAPolicy,
    core: EngineCore,
    checkpoint: str | None,
    adapter_config: Mapping[str, Any] | None = None,
) -> ServingAdapter:
    factory = getattr(policy, "build_serving_adapter", None)
    if factory is None:
        raise RuntimeError(f"{policy.__class__.__name__} does not expose build_serving_adapter")
    if not callable(factory):
        raise RuntimeError(f"build_serving_adapter is not callable on {policy.__class__.__name__}")
    return factory(core=core, checkpoint=checkpoint, config=adapter_config)


def _parse_adapter_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("adapter config must be a JSON object")
    return dict(value)


__all__ = ["add_policy_arguments", "build_serving_adapter"]
