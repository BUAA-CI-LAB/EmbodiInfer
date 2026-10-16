"""Portable, immutable model captures, independent of Torch and coding agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import ContractError, digest, read_json, relative_path, tree_hashes

SCHEMA = "embodiinfer-model-capture-v1"
DISPOSITIONS = {"ready", "excluded_quantized", "metadata_only", "host_only", "blocked"}
RNG_OVERLOADS = {"aten.randn.default", "aten.randn.generator"}
REPLAY_SCHEMA = "embodiinfer-replay-v1"


def validate_rng(contract: dict[str, Any] | None, case: dict[str, Any], files: dict[str, str]) -> None:
    """Bind both RNG-state transitions to immutable files for the declared devices."""
    if contract is None:
        if "rng" in case:
            raise ContractError("Undeclared RNG state in workload")
        return
    if not isinstance(contract, dict):
        raise ContractError("Invalid default-generator RNG contract")
    devices = contract.get("devices")
    if (
        set(contract) != {"kind", "devices"}
        or contract["kind"] != "torch_default"
        or not isinstance(devices, list)
        or not 1 <= len(devices) <= 2
        or devices[0] != "cpu"
        or (len(devices) == 2 and not re.fullmatch(r"cuda:\d+", str(devices[1])))
    ):
        raise ContractError("Invalid default-generator RNG contract")
    states = case.get("rng", {})
    if not isinstance(states, dict) or set(states) != {"before", "after"}:
        raise ContractError("RNG replay requires before and after states")
    for phase in states.values():
        if not isinstance(phase, dict) or set(phase) != set(devices):
            raise ContractError("RNG state devices differ from the contract")
        if any(relative_path(path) not in files for path in phase.values()):
            raise ContractError("RNG fixture missing from the capture/task")


def validate_replay(
    replay: dict[str, Any], definition: dict[str, Any], traces: list[dict[str, Any]], hashes: dict[str, str]
) -> None:
    """Validate captured tensor layouts and bind every replay case to its task axes."""
    if (
        replay.get("schema_version") != 1
        or not replay.get("capture_identity")
        or not replay.get("target", {}).get("model")
    ):
        raise ContractError("Invalid model replay identity")
    schema = replay.get("definition_schema")
    if schema not in (None, REPLAY_SCHEMA) or (
        any(
            spec["dtype"] == "float64" for kind in ("inputs", "outputs") for spec in definition[kind].values()
        )
        and schema != REPLAY_SCHEMA
    ):
        raise ContractError("float64 requires the explicit replay definition schema")
    if replay.get("rng") is not None and replay.get("operator") not in RNG_OVERLOADS:
        raise ContractError("Unsupported RNG replay operator")
    if set(replay.get("cases", {})) != {trace["workload"]["uuid"] for trace in traces}:
        raise ContractError("Replay cases must cover every workload exactly")
    if not isinstance(replay.get("mutates"), list) or any(
        type(index) is not int or not 0 <= index < len(definition["inputs"]) for index in replay["mutates"]
    ):
        raise ContractError("Invalid declared input mutations")
    undefined = replay.get("undefined_outputs", [])
    if (
        not isinstance(undefined, list)
        or (undefined and not str(replay.get("operator")).startswith("aten._scaled_dot_product_"))
        or len(set(undefined)) != len(undefined)
        or len(undefined) >= len(definition["outputs"])
        or any(type(index) is not int or not 0 <= index < len(definition["outputs"]) for index in undefined)
    ):
        raise ContractError("Only unused attention dropout state may be excluded from output checks")
    for trace in traces:
        work = trace["workload"]
        case = replay["cases"][work["uuid"]]
        validate_rng(replay.get("rng"), case, hashes)
        for kind in ("inputs", "outputs"):
            specs = case[kind]
            if len(specs) != len(definition[kind]):
                raise ContractError("Replay tensor count differs from the definition")
            for spec, tensor in zip(specs, definition[kind].values()):
                shape = [
                    work["axes"].get(axis, definition["axes"][axis].get("value")) for axis in tensor["shape"]
                ]
                if spec["shape"] != shape or spec["dtype"] != tensor["dtype"]:
                    raise ContractError("Replay tensor shape/dtype differs from workload axes")
                if len(spec["stride"]) != len(shape) or any(
                    type(n) is not int or n < 0 for n in [spec["offset"], *spec["stride"]]
                ):
                    raise ContractError("Invalid replay tensor layout")
                if not isinstance(spec.get("device"), str):
                    raise ContractError("Replay tensor must specify a device")
                if kind == "inputs":
                    if type(spec.get("storage")) is not int or spec["storage"] < 0:
                        raise ContractError("Invalid replay storage group")
                    if "fixture" in spec and relative_path(spec["fixture"]) not in hashes:
                        raise ContractError("Replay fixture missing from the task")
        for key, limit in (("output_aliases", len(case["inputs"])), ("output_groups", len(case["outputs"]))):
            if len(case[key]) != len(case["outputs"]) or any(
                not isinstance(group, list) or any(type(i) is not int or not 0 <= i < limit for i in group)
                for group in case[key]
            ):
                raise ContractError("Invalid output alias contract")


@dataclass(frozen=True)
class ModelCapture:
    """A completed preparation run bound to one actual policy and all observed cases."""

    root: Path
    manifest: dict[str, Any]
    operators: tuple[dict[str, Any], ...]

    @property
    def identity(self) -> str:
        """Content identity used by generated tasks and resumable model batches."""
        return digest(self.manifest)

    @classmethod
    def load(cls, root: Path, model: str | None = None, *, require_ready: bool = True) -> ModelCapture:
        """Verify capture contents and optionally enforce complete device-task coverage."""
        root = root.resolve()
        if not (root / "manifest.json").is_file():
            raise ContractError("Model tuning requires a model capture directory; recapture legacy JSONL")
        manifest = read_json(root / "manifest.json")
        if manifest.get("schema") != SCHEMA:
            raise ContractError("Unsupported model capture schema")
        target = manifest.get("target", {})
        if not target.get("model") or not target.get("type"):
            raise ContractError("Capture has no verified target model")
        if model is not None and target["model"] != model:
            raise ContractError(f"Capture model {target['model']!r} differs from requested model {model!r}")
        hashes = tree_hashes(root)
        hashes.pop("manifest.json", None)
        if hashes != manifest.get("files"):
            raise ContractError("Model capture content changed")
        operators = read_json(root / "operators.json")
        if not isinstance(operators, list) or not operators:
            raise ContractError("Capture contains no operator observations")
        identities = set()
        for op in operators:
            if op.get("status") not in DISPOSITIONS or not op.get("id") or op["id"] in identities:
                raise ContractError("Malformed or duplicate captured operator")
            if op["id"] != digest(op["identity"])[:20] or op["name"] != op["identity"]["name"]:
                raise ContractError("Captured operator identity differs from its content")
            identities.add(op["id"])
            cases = op.get("workloads", [])
            if not cases or len({case["id"] for case in cases}) != len(cases):
                raise ContractError(f"Missing or duplicate workloads: {op['id']}")
            if any(type(case.get("count")) is not int or case["count"] < 1 for case in cases):
                raise ContractError("Captured workload counts must be positive")
            for case in cases:
                if case["id"] != digest({k: v for k, v in case.items() if k not in {"id", "count"}})[:20]:
                    raise ContractError("Captured workload identity differs from its content")
                for tensor in case.get("inputs", []):
                    if "fixture" in tensor and relative_path(tensor["fixture"]) not in hashes:
                        raise ContractError("Captured fixture must be included in the capture directory")
                if op["status"] == "ready":
                    rng = op["identity"].get("rng")
                    if rng is not None and op["name"] not in RNG_OVERLOADS:
                        raise ContractError("Unsupported RNG capture operator")
                    validate_rng(rng, case, hashes)
        capture = cls(root, manifest, tuple(operators))
        if require_ready:
            capture.require_ready()
        return capture

    def require_ready(self) -> None:
        """Reject incomplete runs and any non-quantized operator preparation gaps."""
        gaps = list(self.manifest.get("errors", []))
        gaps += [
            f"{op['name']}: {op.get('reason', 'unsupported')}"
            for op in self.operators
            if op["status"] == "blocked"
        ]
        if not self.manifest.get("complete"):
            gaps.insert(0, "preparation did not complete")
        if gaps:
            raise ContractError("Model capture is incomplete; no tuning may start:\n" + "\n".join(gaps))
        if not any(op["status"] == "ready" for op in self.operators):
            raise ContractError("Model capture has no tunable device computation")
