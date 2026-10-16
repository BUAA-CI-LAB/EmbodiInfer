"""Model-level numerical gate for an installed kernel profile.

``python -m scripts.kernel_tuning verify --profile NAME --output REPORT -- DRIVER_ARGS``
runs in the model runtime. ``DRIVER_ARGS`` are the serving arguments and recorded
inputs of :mod:`scripts.kernel_tuning.discovery.drive`. The same inputs run three
times, each with a fixed per-input seed:

* reference: FP32 execution with every registry operator on its Torch reference;
* baseline: the requested dtype with ``kernel_profile: torch``;
* candidate: the requested dtype with ``kernel_profile: NAME``.

Continuous outputs are judged over all inputs together, per output name: the
candidate's mean max-abs and mean RMSE error against the reference may exceed the
baseline's by at most 25%, and its worst max-abs error may be at most twice the
baseline's worst (floors 1e-4 and 1e-5). A single input's baseline error is not a
threshold: Torch itself varies between runs by more than that on inputs where it
happens to land close to the reference. Discrete outputs (integers, strings,
booleans) must equal the baseline exactly. See docs/proposals/0009, Section 6.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .contracts import ContractError

MAX_ABS_FLOOR = 1e-4
RMSE_FLOOR = 1e-5
# Mean errors over all inputs may grow by this factor; the worst input by WORST_FACTOR.
MEAN_FACTOR = 1.25
WORST_FACTOR = 2.0
# A baseline this far from the FP32 reference means the two runs did not see the
# same inputs or noise; the reference cannot calibrate a threshold then.
INCOMPARABLE_FRACTION = 0.5


def flatten_outputs(value: Any, prefix: str = "") -> dict[str, tuple[str, Any]]:
    """Map output leaves to ``("continuous", floats)`` or ``("discrete", values)``."""
    import torch

    result: dict[str, tuple[str, Any]] = {}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            result[prefix] = ("continuous", tensor.double().flatten().tolist())
        else:
            result[prefix] = ("discrete", tensor.flatten().tolist())
    elif hasattr(value, "dtype") and hasattr(value, "tolist"):  # NumPy arrays and scalars
        values = value.tolist()
        values = values if isinstance(values, list) else [values]
        flat = _flat(values)
        kind = "continuous" if value.dtype.kind in "fc" else "discrete"
        result[prefix] = (kind, [float(x) for x in flat] if kind == "continuous" else flat)
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            if name in ("timing", "policy_revision"):
                continue  # Wall-clock timing and version strings are not model outputs.
            result.update(flatten_outputs(getattr(value, name), f"{prefix}.{name}"))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            result.update(flatten_outputs(value[key], f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(item, float) for item in value):
            result[prefix] = ("continuous", [float(item) for item in value])
        elif value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            if any(isinstance(item, float) for item in value):
                result[prefix] = ("continuous", [float(item) for item in value])
            else:
                result[prefix] = ("discrete", list(value))
        else:
            for index, item in enumerate(value):
                result.update(flatten_outputs(item, f"{prefix}[{index}]"))
    elif isinstance(value, float):
        result[prefix] = ("continuous", [value])
    else:
        result[prefix] = ("discrete", [value])
    return result


def _flat(values: list[Any]) -> list[Any]:
    return [x for item in values for x in (_flat(item) if isinstance(item, list) else [item])]


def _error(actual: list[float], expected: list[float]) -> tuple[float, float]:
    if len(actual) != len(expected):
        raise ContractError("Output sizes differ between runs")
    if not actual:
        return 0.0, 0.0
    if not all(math.isfinite(x) for x in actual):
        return math.inf, math.inf
    differences = [abs(a - b) for a, b in zip(actual, expected)]
    return max(differences), math.sqrt(sum(d * d for d in differences) / len(differences))


def gate(
    reference: list[dict[str, tuple[str, Any]]],
    baseline: list[dict[str, tuple[str, Any]]],
    candidate: list[dict[str, tuple[str, Any]]],
) -> dict[str, Any]:
    """Compare candidate and baseline errors against the reference over all inputs."""
    if not (len(reference) == len(baseline) == len(candidate)) or not reference:
        raise ContractError("All runs must produce one output per input")
    rows, errors, passed = [], {}, True
    for index, (ref, base, cand) in enumerate(zip(reference, baseline, candidate)):
        if set(ref) != set(base) or set(base) != set(cand):
            raise ContractError(f"input {index}: output structure differs between runs")
        for name in sorted(ref):
            kind = base[name][0]
            if kind == "discrete":
                same = cand[name] == base[name]
                passed &= same
                rows.append({"input": index, "output": name, "kind": kind, "passed": same})
                continue
            if ref[name][0] != "continuous" or cand[name][0] != "continuous":
                raise ContractError(f"input {index} {name}: output kind differs between runs")
            scale = max((abs(x) for x in ref[name][1]), default=0.0)
            base_abs, base_rmse = _error(base[name][1], ref[name][1])
            if base_abs > INCOMPARABLE_FRACTION * max(scale, 1e-6):
                raise ContractError(
                    f"input {index} {name}: baseline differs from the FP32 reference by {base_abs:.3g} "
                    f"(scale {scale:.3g}); the runs did not share inputs/noise, so no threshold is valid"
                )
            cand_abs, cand_rmse = _error(cand[name][1], ref[name][1])
            errors.setdefault(name, []).append((base_abs, base_rmse, cand_abs, cand_rmse))
            rows.append(
                {
                    "input": index,
                    "output": name,
                    "kind": kind,
                    "candidate_max_abs": cand_abs,
                    "candidate_rmse": cand_rmse,
                    "baseline_max_abs": base_abs,
                    "baseline_rmse": base_rmse,
                }
            )
    outputs = {}
    for name, values in sorted(errors.items()):
        base_abs, base_rmse, cand_abs, cand_rmse = (list(column) for column in zip(*values))
        checks = {
            "mean_max_abs": (_mean(cand_abs), max(MAX_ABS_FLOOR, MEAN_FACTOR * _mean(base_abs))),
            "mean_rmse": (_mean(cand_rmse), max(RMSE_FLOOR, MEAN_FACTOR * _mean(base_rmse))),
            "worst_max_abs": (max(cand_abs), max(MAX_ABS_FLOOR, WORST_FACTOR * max(base_abs))),
        }
        ok = all(value <= limit for value, limit in checks.values())
        passed &= ok
        outputs[name] = {
            "passed": ok,
            "ratio": max(value / limit for value, limit in checks.values()),
            **{key: {"candidate": value, "limit": limit} for key, (value, limit) in checks.items()},
            "baseline": {
                "mean_max_abs": _mean(base_abs),
                "mean_rmse": _mean(base_rmse),
                "worst_max_abs": max(base_abs),
            },
        }
    return {
        "passed": bool(passed),
        "inputs": len(reference),
        "outputs": outputs,
        "failed": [name for name, item in outputs.items() if not item["passed"]]
        + [
            f"{row['output']}[{row['input']}]"
            for row in rows
            if row["kind"] == "discrete" and not row["passed"]
        ],
        "worst_ratio": max((item["ratio"] for item in outputs.values()), default=None),
        "rows": rows,
    }


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _run(args: argparse.Namespace, episodes: list[list[Any]], kind: str, profile: str, dtype: str, seed: int):
    """Outputs of every input under one kernel profile and dtype."""
    import torch

    from .discovery import drive

    config = drive._parse_adapter_config(args.adapter_config)
    config.setdefault("policy_kwargs", {})["kernel_profile"] = profile
    outputs = []
    with tempfile.TemporaryDirectory(prefix="kernel-verify-") as scratch:
        path = Path(scratch) / "adapter.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        local = argparse.Namespace(**{**vars(args), "adapter_config": str(path), "dtype": dtype})
        index = 0
        if kind == "requests":
            adapter = drive.build_serving_adapter(local)
            try:
                for episode in episodes:
                    for request in episode:
                        torch.manual_seed(seed + index)
                        outputs.append(flatten_outputs(adapter.infer(request)))
                        index += 1
                    for session in dict.fromkeys(request.session_id for request in episode):
                        adapter.reset(session)
            finally:
                if isinstance(adapter, drive.BatchedServingAdapter):
                    adapter.shutdown()
            del adapter
        else:
            from embodiinfer.engine.config import EngineConfig
            from embodiinfer.engine.core import EngineCore
            from embodiinfer.policies.factory import make_policy
            from embodiinfer.types import SessionKey

            kwargs = dict(config["policy_kwargs"])
            if local.checkpoint is not None:
                kwargs["checkpoint"] = local.checkpoint
            policy = make_policy(local.policy, **kwargs)
            core = EngineCore(
                policy,
                EngineConfig(
                    device=local.device,
                    dtype=dtype,
                    max_batch_size=local.max_batch,
                    num_steps=local.num_steps,
                    use_cuda_graph=not local.no_cuda_graph,
                    capture_full_loop=local.capture_full_loop,
                ),
            )
            for number, episode in enumerate(episodes):
                key = SessionKey(env_id="verify", episode_id=number) if policy.is_recurrent else None
                for step, observation in enumerate(episode):
                    torch.manual_seed(seed + index)
                    batch = policy.collate([observation], [f"{number}-{step}"])
                    outputs.append(
                        flatten_outputs(core.execute(batch, session_ids=None if key is None else [key]))
                    )
                    index += 1
                if key is not None:
                    core.reset_sessions([key])
            del core, policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def run(profile: str, output: Path, driver_args: list[str], *, samples: int | None = None, seed: int = 1000):
    """Run reference, baseline and candidate on the driver inputs and write the gate report."""
    from embodiinfer.engine.serve.factory import add_policy_arguments

    from .discovery.drive import load_inputs

    if output.exists():
        raise ContractError(f"{output} already exists")
    parser = argparse.ArgumentParser(prog="verify driver arguments")
    add_policy_arguments(parser)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args(driver_args)
    kind, episodes = load_inputs(args.inputs)
    if samples is not None:
        remaining, kept = samples, []
        for episode in episodes:
            if remaining <= 0:
                break
            kept.append(episode[:remaining])
            remaining -= len(kept[-1])
        episodes = kept
    dtype = args.dtype if args.dtype != "auto" else "bfloat16"
    runs = {
        "reference": _run(args, episodes, kind, "torch", "float32", seed),
        "baseline": _run(args, episodes, kind, "torch", dtype, seed),
        "candidate": _run(args, episodes, kind, profile, dtype, seed),
    }
    report = gate(runs["reference"], runs["baseline"], runs["candidate"])
    report.update(
        {
            "profile": profile,
            "dtype": dtype,
            "seed": seed,
            "driver_args": driver_args,
            "floors": {"max_abs": MAX_ABS_FLOOR, "rmse": RMSE_FLOOR},
            "factors": {"mean": MEAN_FACTOR, "worst": WORST_FACTOR},
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
