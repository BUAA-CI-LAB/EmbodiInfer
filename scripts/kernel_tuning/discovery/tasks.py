"""Static task construction from verified model captures, with no Torch imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import atomic_json
from ..contracts import ContractError, TaskPackage, read_json
from ..generate import _dump_yaml, tuning_settings, vendor_sources
from ..operators import Operator, Workload
from .contracts import REPLAY_SCHEMA, ModelCapture
from .numerics import SEEDS, execution_precision, policy, task_precision, validate_contract

# FlashInfer's dtypes plus the explicit local replay-schema float64 extension.
_DTYPES = {
    "float64",
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1",
    "int64",
    "int32",
    "int16",
    "int8",
    "bool",
}


def _expression(recipe: dict[str, Any]) -> str:
    kind, value = next(iter(recipe.items()))
    if kind == "tensor":
        return f"x{value}"
    if kind == "constant":
        return repr(value)
    if kind == "torch_value":
        if value.startswith("torch.") and value[6:].isidentifier():
            return value
        return f"torch.device({value!r})"
    if kind in ("tuple", "list"):
        entries = ", ".join(_expression(item) for item in value)
        return f"({entries},)" if kind == "tuple" and value else "()" if kind == "tuple" else f"[{entries}]"
    if kind == "dict":
        return "{" + ", ".join(f"{key!r}: {_expression(item)}" for key, item in value.items()) + "}"
    raise ContractError(f"Unrecognized argument recipe {kind}")


def _return_expressions(
    recipe: dict[str, Any], prefix: str = "result", found: dict[int, str] | None = None
) -> list[str]:
    found = {} if found is None else found
    if "tensor" in recipe:
        found.setdefault(recipe["tensor"], prefix)
    for key in ("tuple", "list"):
        for index, item in enumerate(recipe.get(key, [])):
            _return_expressions(item, f"{prefix}[{index}]", found)
    for key, item in recipe.get("dict", {}).items():
        _return_expressions(item, f"{prefix}[{key!r}]", found)
    return [found[index] for index in sorted(found)]


def _sources(op: dict[str, Any]) -> tuple[str, dict[str, str]]:
    identity = op["identity"]
    files = {}
    if identity["kind"] != "registry":
        raise ContractError("Only registry tasks are supported; recapture with capture-ops")
    module, entry = identity["module"], identity["entry"]
    if module != "embodiinfer.backend.torch.ops" or not entry.isidentifier():
        raise ContractError("Registry tasks take their baseline from embodiinfer.backend.torch.ops")
    for name, key in ((module, "source_digests"), (identity["start_module"], "start_digests")):
        vendored = vendor_sources(name)
        if {path: source_digest for path, (_, source_digest) in vendored.items()} != identity[key]:
            raise ContractError(f"Registry operator source changed since capture: {name}")
        files.update({path: text for path, (text, _) in vendored.items()})
    imports, callable_name = f"import torch\nfrom .vendor.{module} import {entry}\n", entry
    args, kwargs = identity["arguments"]["tuple"]
    names = [f"x{i}" for i in range(len(identity["input_dtypes"]))]
    returns = _return_expressions(identity["returns"]) + [f"x{i}" for i in identity["mutates"]]
    layouts = _production_layouts(op)
    if layouts:
        imports += _RELAYOUT.format(layouts=repr(layouts))
    code = imports + f"\ndef run({', '.join(names)}):\n"
    code += f"    result = {callable_name}(*{_expression(args)}, **{_expression(kwargs)})\n"
    if layouts:
        code += f"    outputs = _relayout(({', '.join(names)},), ({', '.join(returns)},))\n"
        returns = [f"outputs[{i}]" for i in range(len(returns))]
    code += "    return " + (returns[0] if len(returns) == 1 else "(" + ", ".join(returns) + ")") + "\n"
    return code, files


# Registry captures take output layouts from the production call. Where the Torch
# reference lays a result out differently, the baseline copies it into that layout:
# values are unchanged, and the reference meets the same contract as candidates.
_RELAYOUT = """
# Production output layout per recorded input layout: (shape, stride) per output, or
# None where the Torch reference already matches.
_LAYOUTS = {layouts}


def _relayout(inputs, outputs):
    key = tuple((tuple(x.shape), tuple(x.stride()), x.storage_offset()) for x in inputs)
    targets = _LAYOUTS.get(key)
    if targets is None:
        return outputs
    result = []
    for value, target in zip(outputs, targets):
        if target is not None and tuple(value.stride()) != target[1]:
            copy = torch.empty_strided(target[0], target[1], dtype=value.dtype, device=value.device)
            copy.copy_(value)
            value = copy
        result.append(value)
    return tuple(result)
"""


def _production_layouts(op: dict[str, Any]) -> dict[tuple[Any, ...], tuple[Any, ...]]:
    """Recorded input layout -> production output layouts, for cases whose reference differs."""
    layouts = {}
    for case in op["workloads"]:
        reference = case.get("reference_outputs")
        if reference is None:
            continue
        if len(reference) != len(case["outputs"]):
            raise ContractError(f"Reference/production output counts differ in {op['name']}")
        key = tuple((tuple(s["shape"]), tuple(s["stride"]), s["offset"]) for s in case["inputs"])
        layouts[key] = tuple(
            None if ref == spec else (tuple(spec["shape"]), tuple(spec["stride"]))
            for spec, ref in zip(case["outputs"], reference)
        )
    return layouts


def _plan(
    capture: ModelCapture, op: dict[str, Any], overrides: dict[str, Any], hardware_notes: Path | None
) -> dict[str, str | bytes]:
    name = f"registry_{op['identity']['op']}_{op['id']}"
    unsupported = {
        spec["dtype"] for case in op["workloads"] for spec in case["inputs"] + case["outputs"]
    } - _DTYPES
    if unsupported:
        raise ContractError(f"Replay definition needs a custom dtype adapter for {sorted(unsupported)}")
    code, sources = _sources(op)
    # The reference can import snapshotted helper sources through this task's
    # benchmark adapter; its standalone mathematical body remains frozen.
    identity = op["identity"]
    axes, inputs, outputs = {}, {}, {}
    for kind, specs in (("inputs", inputs), ("outputs", outputs)):
        for index, tensor in enumerate(op["workloads"][0][kind]):
            shape = [f"{kind[0]}{index}d{d}" for d in range(len(tensor["shape"]))]
            for axis in shape:
                axes[axis] = {"type": "var"}
            specs[f"{'x' if kind == 'inputs' else 'y'}{index}"] = {"shape": shape, "dtype": tensor["dtype"]}
    definition = {
        "name": name,
        "op_type": "custom",
        "description": f"Captured single operator {op['name']}",
        "tags": [f"model:{capture.manifest['target']['model']}"],
        "axes": axes,
        "inputs": inputs,
        "outputs": outputs,
        "constraints": [],
        "reference": code,
    }
    workloads = []
    cases = {}
    for case in op["workloads"]:
        values = {}
        for kind in ("inputs", "outputs"):
            if len(case[kind]) != len(definition[kind]):
                raise ContractError(f"Output/input count changes in {op['name']}")
            for index, tensor in enumerate(case[kind]):
                spec = list(definition[kind].values())[index]
                if len(spec["shape"]) != len(tensor["shape"]) or spec["dtype"] != tensor["dtype"]:
                    raise ContractError(f"Rank/dtype changes require distinct task signatures: {op['name']}")
                values.update(zip(spec["shape"], tensor["shape"]))
        workloads.append(
            {
                "definition": name,
                "workload": {
                    "uuid": case["id"],
                    "axes": values,
                    "inputs": {key: {"type": "random"} for key in inputs},
                },
                "solution": None,
                "evaluation": None,
            }
        )
        cases[case["id"]] = case
        for tensor in case["inputs"]:
            if "fixture" in tensor:
                sources[tensor["fixture"]] = (capture.root / tensor["fixture"]).read_bytes()
    numerical = None
    precision = {"mode": "bit_exact", **execution_precision(capture.manifest["precision"])}
    if policy(identity) is not None:
        path = capture.root / "calibration.json"
        numerical = read_json(path).get(op["id"]) if path.is_file() else None
        if numerical is None:
            raise ContractError(
                "Missing numerical calibration; run calibrate --captured CAPTURE --output NEW_CAPTURE in the model environment"
            )
        validate_contract(
            numerical, identity, cases, capture.manifest["precision"], capture.manifest["flags"]
        )
        precision = task_precision(numerical, capture.manifest["precision"])
        sources["numerics.json"] = json.dumps(numerical, indent=2) + "\n"
    descriptor = Operator(
        name=name,
        summary=op["name"],
        models=(capture.manifest["target"]["model"],),
        op_type="custom",
        source="",
        entry="run",
        call="",
        axes={},
        inputs={},
        outputs={},
        reference=code,
        workloads=tuple(Workload(case["id"], {}, weight=case["count"]) for case in op["workloads"]),
        notes="Registry operator; the Torch reference is the baseline.",
        precision=precision,
        # Registry operators run inside the model's CUDA graphs in production.
        timing="cuda_graph",
    )
    settings = tuning_settings(descriptor, overrides)
    if settings["precision"] != precision:
        raise ContractError("Model task precision cannot override its frozen numerical contract")
    if numerical is not None and settings["seeds"] != list(SEEDS):
        raise ContractError("Calibrated validation seeds cannot be overridden")
    if 0 not in settings["seeds"]:
        raise ContractError("Model replay must include seed 0 for the unchanged captured fixtures")
    device = next(
        (
            spec["device"]
            for spec in op["workloads"][0]["inputs"] + op["workloads"][0]["outputs"]
            if spec["device"].startswith(capture.manifest["device_type"])
        ),
        None,
    )
    if "device" in overrides and overrides["device"] != device:
        raise ContractError("Capture and evaluator devices differ; prepare on the target device")
    settings["device"] = device
    replay = {
        "schema_version": 1,
        "definition_schema": REPLAY_SCHEMA,
        "operator": op["name"],
        "kind": identity["kind"],
        "mutates": identity["mutates"],
        "cases": cases,
        "torch_version": capture.manifest["torch_version"],
        "flags": capture.manifest["flags"],
        "capture_identity": capture.identity,
        "target": capture.manifest["target"],
    }
    if numerical is not None:
        replay["numerical_identity"] = identity
    sources.update(
        {
            "definition.json": json.dumps(definition, indent=2) + "\n",
            "baseline.py": code,
            "workloads.jsonl": "".join(json.dumps(work) + "\n" for work in workloads),
            "tuning.yaml": _dump_yaml(settings),
            "replay.json": json.dumps(replay, indent=2) + "\n",
            "benchmark.py": "from benchmarks.kernel_tuning.replay_adapter import ReplayAdapter\n\ndef create_adapter(task):\n    return ReplayAdapter(task)\n",
        }
    )
    start = identity["start_module"].replace(".", "/")
    sources["README.md"] = (
        f"# {identity['op']}\n\nTarget: `{capture.manifest['target']['model']}`.\n"
        f"Capture: `{capture.identity}`.\n\n"
        "Write a faster implementation of this operator-registry entry. The baseline to "
        "beat is the Torch reference in baseline.py "
        f"(`vendor/embodiinfer/backend/torch/ops.py::{identity['entry']}`); the repository's "
        f"current implementation, `vendor/{start}.py::{identity['start_entry']}`, is a valid "
        "starting point. Keep the call signature, return outputs with exactly the recorded "
        "shapes, strides and dtypes in replay.json, use return-value style "
        "(destination_passing_style=false), and do not change dtypes, masks or numerical "
        "settings. Every recorded workload is mandatory and weighted by its call count in "
        "the model. The installed kernel is used only for these recorded argument layouts, "
        "so you may specialize on them.\n"
    )
    if _production_layouts(op):
        sources["README.md"] += (
            "\nOutput layouts are those of the implementation the model ran. Where the Torch "
            "reference lays a result out differently (`reference_outputs` in replay.json), "
            "baseline.py copies it into the production layout; the copy is part of the "
            "baseline's time and leaves values unchanged.\n"
        )
    sources["generated.json"] = json.dumps(
        {
            "generator": "scripts.kernel_tuning.regions",
            "capture_identity": capture.identity,
            "operator_id": op["id"],
            "registry_op": identity["op"],
        }
    )
    if hardware_notes is not None:
        sources["HARDWARE.md"] = hardware_notes.read_text(encoding="utf-8")
    if numerical is not None:
        sources["README.md"] += (
            "\nRead numerics.json: all floating elements are compared to the trusted FP64 "
            "reference using frozen per-workload/profile/output bounds. tuning.yaml contains "
            "only their reporting maxima. Recorded inputs, fresh random inputs, zeros and "
            "cancellation inputs must all pass. Reassociation is permitted within these "
            "bounds; changing dtype, masks, accumulation settings or the contract is not.\n"
        )
    if any(spec["dtype"] == "float64" for kind in (inputs, outputs) for spec in kind.values()):
        sources["README.md"] += (
            "\nThis task uses the embodiinfer-replay-v1 float64 definition extension and its "
            "benchmark.py adapter. Keep all float64 inputs, intermediates and outputs at their "
            "declared precision. The upstream FlashInfer 0.1.2 Definition parser alone cannot "
            "load this extension.\n"
        )
    return sources


def render_model(
    capture: ModelCapture,
    destination: Path,
    *,
    overrides: dict[str, Any] | None = None,
    hardware_notes: Path | None = None,
) -> list[TaskPackage]:
    """Build all eligible tasks or reject the entire model before writing any task."""
    capture.require_ready()
    # Recheck snapshots immediately before construction; do not trust a previously loaded object.
    checked = ModelCapture.load(capture.root, capture.manifest["target"]["model"])
    if checked.identity != capture.identity:
        raise ContractError("Capture changed before task generation")
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ContractError("Model tasks require a new or empty destination")
    plans = {}
    failures = []
    for op in capture.operators:
        if op["status"] != "ready":
            continue
        try:
            plans[op["id"]] = _plan(capture, op, overrides or {}, hardware_notes)
        except (ContractError, KeyError, ValueError) as exc:
            failures.append(f"{op['name']}: {exc}")
    if failures:
        raise ContractError("Task coverage gaps; no model batch may start:\n" + "\n".join(failures))
    tasks = []
    for op_id, files in plans.items():
        root = destination / op_id
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8", newline="\n")
        tasks.append(TaskPackage.load(root))
    atomic_json(
        destination / "model.json",
        {
            "capture_identity": capture.identity,
            "target": capture.manifest["target"],
            "tasks": {task.root.name: task.identity for task in tasks},
        },
    )
    return tasks
