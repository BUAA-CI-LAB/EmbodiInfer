"""Frozen task packages, portable solutions, and promotion from measured evidence.

This module intentionally imports neither Humanize2 nor GPU libraries. Official
FlashInfer schemas are validated again by the default evaluator before building.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

HMZ_REVISION = "b70d4427f11308442aef20f9b2233dfbbc34a39e"
FLASHINFER_VERSION = "0.1.2"
ROLES = ("baseline", "incumbent", "candidate")


class ContractError(ValueError):
    """A task, solution, or measurement does not satisfy the frozen contract."""


def read_json(path: Path) -> Any:
    """Read JSON while rejecting non-finite numbers and duplicate keys."""
    return parse_json(path.read_text(encoding="utf-8"))


def parse_json(text: str) -> Any:
    """Parse strict JSON, also used for individual official Trace JSONL records."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ContractError(f"Non-finite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def digest(value: Any) -> str:
    """Hash JSON data independently of whitespace and dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def relative_path(value: str) -> str:
    """Validate a portable relative file path before joining it to any workspace."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ContractError(f"Expected a portable relative path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ContractError(f"Unsafe relative path: {value!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{index}" for prefix in ("COM", "LPT") for index in "123456789¹²³")
    if any(part.split(".")[0].upper() in reserved or part.endswith((".", " ")) for part in path.parts):
        raise ContractError(f"Non-portable path: {value!r}")
    return value


def tree_hashes(root: Path) -> dict[str, str]:
    """Hash every task/artifact file, refusing symlinks and generated Python caches."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"Symlinks are not portable task inputs: {path}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_file():
            name = relative_path(path.relative_to(root).as_posix())
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _positive(value: Any, name: str, *, integer: bool = False, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ContractError(f"{name} must be {'an integer' if integer else 'a number'}")
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ContractError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class Precision:
    """Numerical conditions fixed before the first agent turn."""

    mode: str = "bit_exact"
    atol: float = 0.0
    rtol: float = 0.0
    reason: str = ""
    matmul_precision: str = "highest"
    allow_tf32: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("bit_exact", "tolerance"):
            raise ContractError("precision.mode must be bit_exact or tolerance")
        for key in ("atol", "rtol"):
            _positive(getattr(self, key), f"precision.{key}", zero=True)
        if self.mode == "bit_exact" and (self.atol or self.rtol):
            raise ContractError("bit_exact cannot declare nonzero tolerances")
        if self.mode == "tolerance" and (not self.reason.strip() or not (self.atol or self.rtol)):
            raise ContractError("tolerance requires a reason and a nonzero atol or rtol")
        if self.matmul_precision not in ("highest", "high", "medium") or type(self.allow_tf32) is not bool:
            raise ContractError("Invalid matmul precision or allow_tf32")
        if self.allow_tf32 != (self.matmul_precision != "highest"):
            raise ContractError("Use highest with allow_tf32: false, or high/medium with allow_tf32: true")


@dataclass(frozen=True)
class Timing:
    """Timing protocol; each paired round must independently pass promotion."""

    mode: str = "eager"
    warmup: int = 10
    iterations: int = 100
    trials: int = 3
    paired_rounds: int = 2

    def __post_init__(self) -> None:
        if self.mode not in ("eager", "cuda_graph"):
            raise ContractError("timing.mode must be eager or cuda_graph")
        for key in ("warmup", "iterations", "trials", "paired_rounds"):
            _positive(getattr(self, key), f"timing.{key}", integer=True)
        if self.paired_rounds < 2:
            raise ContractError("Promotion requires at least two paired measurement rounds")


@dataclass(frozen=True)
class Search:
    """Whole-run and per-operation limits; failures consume attempts and patience."""

    max_candidates: int = 20
    max_seconds: float = 7200
    patience: int = 5
    agent_timeout_seconds: float = 600
    evaluation_timeout_seconds: float = 300

    def __post_init__(self) -> None:
        for key in ("max_candidates", "patience"):
            _positive(getattr(self, key), f"search.{key}", integer=True)
        for key in ("max_seconds", "agent_timeout_seconds", "evaluation_timeout_seconds"):
            _positive(getattr(self, key), f"search.{key}")


@dataclass(frozen=True)
class Acceptance:
    """Latency reduction and workload regression limits, expressed as fractions."""

    min_improvement: float = 0.03
    max_regression: float = 0.05

    def __post_init__(self) -> None:
        _positive(self.min_improvement, "acceptance.min_improvement")
        _positive(self.max_regression, "acceptance.max_regression", zero=True)
        if self.min_improvement >= 1 or self.max_regression >= 1:
            raise ContractError("Acceptance fractions must be less than 1")


T = TypeVar("T")


def _settings(cls: type[T], data: Any) -> T:
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
        raise ContractError(f"Unknown or malformed {cls.__name__} settings: {data!r}")
    try:
        return cls(**data)
    except (TypeError, AttributeError) as exc:
        raise ContractError(f"Invalid {cls.__name__}: {exc}") from exc


@dataclass(frozen=True)
class Settings:
    """Project settings beside, rather than inside, the official task schemas."""

    language: str
    schema_version: int = 1
    agent: str = ""
    evaluator_python: str = "python"
    device: str = "cuda:0"
    target_hardware: tuple[str, ...] = ("cuda",)
    seeds: tuple[int, ...] = (0, 1, 2)
    weights: dict[str, float] = field(default_factory=dict)
    precision: Precision = field(default_factory=Precision)
    timing: Timing = field(default_factory=Timing)
    search: Search = field(default_factory=Search)
    acceptance: Acceptance = field(default_factory=Acceptance)
    profile: bool = False

    @classmethod
    def load(cls, path: Path) -> Settings:
        """Parse YAML with unknown settings rejected instead of silently ignored."""
        import yaml

        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader: Any, node: Any) -> dict[str, Any]:
            pairs = loader.construct_pairs(node, deep=True)
            if len({key for key, _ in pairs}) != len(pairs):
                raise ContractError("Duplicate YAML setting")
            return dict(pairs)

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        try:
            data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise ContractError(f"Invalid tuning.yaml: {exc}") from exc
        if not isinstance(data, dict):
            raise ContractError("tuning.yaml must contain a mapping")
        for key, kind in (
            ("precision", Precision),
            ("timing", Timing),
            ("search", Search),
            ("acceptance", Acceptance),
        ):
            data[key] = _settings(kind, data.get(key, {}))
        for key in ("seeds", "target_hardware"):
            if key in data:
                if not isinstance(data[key], list):
                    raise ContractError(f"{key} must be a list")
                data[key] = tuple(data[key])
        return _settings(cls, data)

    def __post_init__(self) -> None:
        if self.language not in ("triton", "cuda") or self.schema_version != 1:
            raise ContractError("Use schema_version: 1 and language: triton or cuda")
        for key in ("evaluator_python", "device"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ContractError(f"{key} must be a nonempty string")
        if not isinstance(self.agent, str) or type(self.profile) is not bool:
            raise ContractError("agent must be a string; profile must be a boolean")
        if not self.target_hardware or any(not isinstance(x, str) or not x for x in self.target_hardware):
            raise ContractError("target_hardware must list at least one hardware label")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ContractError("seeds must be nonempty and unique")
        for seed in self.seeds:
            _positive(seed, "seed", integer=True, zero=True)
            if seed >= 2**32:
                raise ContractError("Seeds must fit uint32")
        if not isinstance(self.weights, dict):
            raise ContractError("weights must map workload UUIDs to positive weights")
        for value in self.weights.values():
            _positive(value, "workload weight")


@dataclass(frozen=True)
class TaskPackage:
    """A self-contained task snapshot with a content identity and declared workloads."""

    root: Path
    definition: dict[str, Any]
    workloads: tuple[dict[str, Any], ...]
    settings: Settings
    hashes: dict[str, str]

    @property
    def identity(self) -> str:
        """Identity of the complete task, including data files and custom evaluator."""
        return digest(self.hashes)

    @property
    def workload_ids(self) -> tuple[str, ...]:
        """Stable measurement keys in the task's declared order."""
        return tuple(t["workload"]["uuid"] for t in self.workloads)

    @classmethod
    def load(cls, root: Path) -> TaskPackage:
        """Validate the task structurally without importing or executing its code."""
        try:
            return cls._load(root)
        except (KeyError, TypeError, AttributeError, SyntaxError) as exc:
            raise ContractError(f"Malformed task package: {exc}") from exc

    @classmethod
    def _load(cls, root: Path) -> TaskPackage:
        root = root.resolve(strict=True)
        hashes = tree_hashes(root)
        required = {"README.md", "definition.json", "workloads.jsonl", "baseline.py", "tuning.yaml"}
        if missing := required - hashes.keys():
            raise ContractError(f"Missing task files: {sorted(missing)}")
        definition = read_json(root / "definition.json")
        for key in ("name", "op_type", "reference"):
            if not isinstance(definition.get(key), str) or not definition[key]:
                raise ContractError(f"Definition requires {key}")
        axes = definition.get("axes")
        if not isinstance(axes, dict):
            raise ContractError("Definition axes must be a mapping")
        for axis in axes.values():
            if axis.get("type") not in ("var", "const"):
                raise ContractError("Axis must be var or const")
            if axis["type"] == "const":
                _positive(axis.get("value"), "constant axis", integer=True, zero=True)
        for kind in ("inputs", "outputs"):
            if not isinstance(definition.get(kind), dict) or not definition[kind]:
                raise ContractError(f"Definition requires {kind}")
            for spec in definition[kind].values():
                shape = spec.get("shape")
                if (
                    shape is not None and (not isinstance(shape, list) or set(shape) - axes.keys())
                ) or not spec.get("dtype"):
                    raise ContractError(f"Invalid {kind} tensor spec")
        for source in (definition["reference"], (root / "baseline.py").read_text(encoding="utf-8")):
            functions = [
                n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run"
            ]
            if len(functions) != 1:
                raise ContractError("Reference and baseline must define run")
            args = functions[0].args
            if [a.arg for a in args.posonlyargs + args.args] != list(definition["inputs"]):
                raise ContractError("run argument order must match definition.inputs")
        traces = []
        for line in (root / "workloads.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trace = parse_json(line)
            work = trace.get("workload", {})
            if (
                trace.get("definition") != definition["name"]
                or not isinstance(work.get("uuid"), str)
                or not work["uuid"]
            ):
                raise ContractError("Workload must identify this definition and have a UUID")
            if trace.get("solution") is not None or trace.get("evaluation") is not None:
                raise ContractError("Input workloads must be unevaluated traces")
            variables = {name for name, axis in axes.items() if axis["type"] == "var"}
            if not isinstance(work.get("axes"), dict) or set(work["axes"]) != variables:
                raise ContractError("Workload axes must bind exactly the variable axes")
            for value in work["axes"].values():
                _positive(value, "workload axis", integer=True, zero=True)
            if not isinstance(work.get("inputs"), dict) or set(work["inputs"]) - definition["inputs"].keys():
                raise ContractError("Invalid workload inputs")
            for spec in work["inputs"].values():
                if spec.get("type") not in ("random", "scalar", "safetensors"):
                    raise ContractError("Unsupported workload input source")
                if spec["type"] == "scalar":
                    value = spec.get("value")
                    if not isinstance(value, (bool, int, float)) or not math.isfinite(value):
                        raise ContractError("Scalar input must contain a finite numeric value")
                if spec["type"] == "safetensors" and relative_path(spec["path"]) not in hashes:
                    raise ContractError("Safetensors data must be included inside the task package")
            traces.append(trace)
        settings = Settings.load(root / "tuning.yaml")
        ids = [t["workload"]["uuid"] for t in traces]
        if not ids or len(ids) != len(set(ids)):
            raise ContractError("Workloads must be nonempty and have distinct UUIDs")
        if settings.weights and set(settings.weights) != set(ids):
            raise ContractError("Explicit weights must cover exactly every workload UUID")
        return cls(root, definition, tuple(traces), settings, hashes)

    def verify(self) -> None:
        """Refuse a changed frozen task before trusting another measurement."""
        if tree_hashes(self.root) != self.hashes:
            raise ContractError("Frozen task changed during tuning")

    def baseline(self) -> dict[str, Any]:
        """Represent baseline.py and its helper sources as an official Solution."""
        return {
            "name": f"{self.definition['name']}_baseline",
            "definition": self.definition["name"],
            "author": "task-owner",
            "spec": {
                "language": "python",
                "target_hardware": list(self.settings.target_hardware),
                "entry_point": "baseline.py::run",
                "destination_passing_style": False,
            },
            "sources": [
                {"path": path, "content": (self.root / path).read_text(encoding="utf-8")}
                for path in self.hashes
                if Path(path).suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp")
                and path != "benchmark.py"
            ],
        }


def validate_solution(solution: Any, task: TaskPackage) -> dict[str, Any]:
    """Validate candidate identity and source paths before any filesystem write/build."""
    if not isinstance(solution, dict) or set(solution) - {
        "name",
        "definition",
        "author",
        "description",
        "spec",
        "sources",
    }:
        raise ContractError("Expected an official FlashInfer Solution object")
    for key in ("name", "author"):
        if not isinstance(solution.get(key), str) or not solution[key]:
            raise ContractError(f"Solution requires {key}")
    if solution.get("definition") != task.definition["name"]:
        raise ContractError("Candidate solves a different definition")
    spec = solution.get("spec", {})
    if spec.get("language") != task.settings.language:
        raise ContractError("Candidate language differs from the frozen task")
    if spec.get("target_hardware") != list(task.settings.target_hardware):
        raise ContractError("Candidate target_hardware differs from the frozen task")
    if type(spec.get("destination_passing_style")) is not bool:
        raise ContractError("Declare destination_passing_style explicitly")
    if spec.get("binding") not in (None, "torch", "tvm-ffi"):
        raise ContractError("CUDA binding must be torch or tvm-ffi")
    entry = spec.get("entry_point", "").split("::")
    if len(entry) != 2 or not entry[1].isidentifier():
        raise ContractError("entry_point must be file::symbol")
    relative_path(entry[0])
    sources = solution.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ContractError("Solution requires sources")
    paths = []
    for source in sources:
        path = relative_path(source["path"])
        if not isinstance(source.get("content"), str) or not source["content"]:
            raise ContractError("Source contents must be nonempty UTF-8 text")
        paths.append(path)
    if len({p.casefold() for p in paths}) != len(paths) or entry[0] not in paths:
        raise ContractError("Source paths must be distinct and include the entry point")
    for path in paths:
        if any(parent.as_posix() in paths for parent in PurePosixPath(path).parents if str(parent) != "."):
            raise ContractError("Source file/directory collision")
    return solution


def validate_measurement(task: TaskPackage, report: dict[str, Any]) -> None:
    """Require complete correctness evidence and finite timings for every role/case."""
    try:
        _validate_measurement(task, report)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ContractError(f"Malformed measurement report: {exc}") from exc


def _validate_measurement(task: TaskPackage, report: dict[str, Any]) -> None:
    cfg = task.settings
    expected = {(work, role) for work in task.workload_ids for role in ROLES}
    seen = set()
    for check in report.get("checks", []):
        key = (check.get("workload"), check.get("role"))
        if key not in expected or key in seen or check.get("passed") is not True:
            raise ContractError(f"Failed, unexpected, or duplicate correctness check: {key}")
        seen.add(key)
        if check.get("seeds") != list(cfg.seeds) or check.get("precision") != asdict(cfg.precision):
            raise ContractError("Correctness evidence used a different precision/seed contract")
        if cfg.timing.mode == "cuda_graph" and check.get("graph_replay") is not True:
            raise ContractError("Missing changed-input graph replay check")
        for field_name in ("max_abs_error", "max_rel_error"):
            _positive(check.get(field_name), field_name, zero=True)
            if cfg.precision.mode == "bit_exact" and check[field_name] != 0:
                raise ContractError("bit_exact report contains nonzero error")
    if seen != expected:
        raise ContractError("Incomplete correctness evidence")
    rounds = report.get("rounds", [])
    if len(rounds) != cfg.timing.paired_rounds:
        raise ContractError("Missing paired remeasurement rounds")
    for measurements in rounds:
        if set(measurements) != set(task.workload_ids):
            raise ContractError("Missing or unexpected timed workloads")
        for row in measurements.values():
            if set(row) != set(ROLES):
                raise ContractError("Missing baseline/incumbent/candidate measurements")
            for samples in row.values():
                if not isinstance(samples, list) or len(samples) != cfg.timing.trials:
                    raise ContractError("Missing timing trials")
                for value in samples:
                    _positive(value, "latency_ms")


def promotion(task: TaskPackage, report: dict[str, Any]) -> dict[str, Any]:
    """Decide solely from fixed thresholds, requiring every paired round to pass."""
    validate_measurement(task, report)
    cfg = task.settings
    weights = cfg.weights or dict.fromkeys(task.workload_ids, 1.0)
    decisions = []
    reasons = []
    for index, measurements in enumerate(report["rounds"]):
        means = {
            work: {role: math.fsum(values) / len(values) for role, values in row.items()}
            for work, row in measurements.items()
        }
        aggregate = {
            role: math.fsum(weights[w] * means[w][role] for w in means) / math.fsum(weights.values())
            for role in ROLES
        }
        improvements = {role: 1 - aggregate["candidate"] / aggregate[role] for role in ROLES[:2]}
        decisions.append({"mean_latency_ms": aggregate, "improvement": improvements, "workloads": means})
        for role in ROLES[:2]:
            if improvements[role] + 1e-12 < cfg.acceptance.min_improvement:
                reasons.append(f"round {index + 1}: insufficient improvement over {role}")
            for work in means:
                if means[work]["candidate"] > means[work][role] * (1 + cfg.acceptance.max_regression + 1e-12):
                    reasons.append(f"round {index + 1}: {work} regresses against {role}")
    return {"promoted": not reasons, "reasons": reasons, "rounds": decisions}
