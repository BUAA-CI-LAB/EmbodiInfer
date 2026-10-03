"""Render catalog operators into self-contained KDA task packages.

The baseline is EmbodiInfer's production kernel, copied byte-for-byte from the
repository together with the repository modules it imports. Generation is
static: it reads source files and writes the task; it never imports Torch,
Triton, or the copied kernels.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import struct
import subprocess
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import REPOSITORY
from .contracts import ContractError, Settings, TaskPackage, read_json
from .operators import Operator

VENDOR = "vendor"
MARKER = "generated.json"
PACKAGE = "embodiinfer"
SAFETENSORS_DTYPES = {"int32": ("I32", "<i"), "int64": ("I64", "<q")}


def _text_digest(path: Path) -> str:
    """Digest source text with normalized newlines, so Windows checkouts agree with Linux."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def _module_file(module: str) -> Path:
    """Resolve a dotted repository module to its file, refusing package initializers."""
    base = REPOSITORY.joinpath(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        raise ContractError(
            f"Cannot vendor package {module}: import the defining submodule instead of the package"
        )
    raise ContractError(f"Repository module {module} does not exist")


def _imports(path: Path) -> list[tuple[ast.ImportFrom | ast.Import, list[str]]]:
    """List each repository import in ``path`` with the dotted modules it loads."""
    relative = path.relative_to(REPOSITORY).with_suffix("")
    package = list(relative.parts[:-1])
    found: list[tuple[ast.ImportFrom | ast.Import, list[str]]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == PACKAGE for alias in node.names):
                raise ContractError(f"{path}: use 'from ... import' for repository modules")
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if node.level > len(package):
                raise ContractError(f"{path}: relative import escapes the repository package")
            anchor = package[: len(package) - node.level + 1]
        elif node.module and node.module.split(".")[0] == PACKAGE:
            anchor = []
        else:
            continue
        target = [*anchor, *(node.module or "").split(".")] if node.module else anchor
        target = [part for part in target if part]
        if node.module:
            found.append((node, [".".join(target)]))
        else:  # ``from . import module`` loads sibling modules.
            found.append((node, [".".join([*target, alias.name]) for alias in node.names]))
    return found


def _relative_import(source: str, target: str) -> str:
    """Spell ``target`` relative to the package containing module ``source``."""
    origin = source.split(".")[:-1]
    goal = target.split(".")
    common = 0
    while common < min(len(origin), len(goal)) and origin[common] == goal[common]:
        common += 1
    return "." * (len(origin) - common + 1) + ".".join(goal[common:])


def vendor_sources(entry: str) -> dict[str, tuple[str, str]]:
    """Copy a module and its repository imports; return task paths to (text, source digest).

    Relative imports keep working because the package layout is preserved.
    Absolute ``embodiinfer.`` imports are rewritten to relative form, the only
    edit made to a copied file. Package ``__init__`` modules are never copied, so
    importing one kernel does not import the whole backend.
    """
    pending = [entry]
    result: dict[str, tuple[str, str]] = {}
    while pending:
        module = pending.pop()
        name = f"{VENDOR}/{module.replace('.', '/')}.py"
        if name in result:
            continue
        path = _module_file(module)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        for node, targets in _imports(path):
            for target in targets:
                _module_file(target)
                pending.append(target)
            if not node.level:
                line = node.lineno - 1
                spelled = _relative_import(module, node.module)
                lines[line], count = re.subn(
                    rf"\bfrom\s+{re.escape(node.module)}\s+import\b",
                    f"from {spelled} import",
                    lines[line],
                    count=1,
                )
                if not count:
                    raise ContractError(f"{path}:{node.lineno}: cannot rewrite multi-line import")
        result[name] = ("".join(lines), _text_digest(path))
    return dict(sorted(result.items()))


def safetensors_bytes(tensors: dict[str, tuple[str, tuple[int, ...]]]) -> bytes:
    """Encode one-dimensional integer tensors in the safetensors format without Torch."""
    header: dict[str, Any] = {}
    payload = b""
    for key, (dtype, values) in sorted(tensors.items()):
        code, fmt = SAFETENSORS_DTYPES[dtype]
        data = struct.pack(f"<{len(values)}{fmt[1]}", *values)
        header[key] = {
            "dtype": code,
            "shape": [len(values)],
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def _yaml_value(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``section.key=value`` tuning.yaml overrides; values use YAML syntax."""
    overrides: dict[str, Any] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ContractError(f"Override must be key=value: {item!r}")
        overrides[key.strip()] = _yaml_value(value)
    return overrides


def tuning_settings(operator: Operator, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default contract for a catalog operator, with validated dotted overrides."""
    settings: dict[str, Any] = {
        "schema_version": 1,
        "language": "triton",
        "agent": "",
        "evaluator_python": "python",
        "device": "cuda:0",
        "target_hardware": ["cuda"],
        "seeds": [0, 1, 2],
        "precision": {"matmul_precision": "highest", "allow_tf32": False, **operator.precision},
        "timing": {"mode": operator.timing, "warmup": 10, "iterations": 100, "trials": 3, "paired_rounds": 2},
        "acceptance": {"min_improvement": 0.03, "max_regression": 0.05},
        "search": {
            "max_candidates": 20,
            "max_seconds": 7200,
            "patience": 5,
            "agent_timeout_seconds": 600,
            "evaluation_timeout_seconds": 600,
        },
        "profile": False,
    }
    weights = {work.uuid: work.weight for work in operator.workloads}
    if len(set(weights.values())) > 1:
        settings["weights"] = weights
    for key, value in (overrides or {}).items():
        *parents, leaf = key.split(".")
        node = settings
        for parent in parents:
            if not isinstance(node.get(parent), dict):
                raise ContractError(f"Unknown tuning.yaml section in override: {key}")
            node = node[parent]
        node[leaf] = value
    Settings.load_mapping(settings)
    return settings


def _definition(operator: Operator) -> dict[str, Any]:
    axes = {
        name: {"type": "var"} if value is None else {"type": "const", "value": value}
        for name, value in operator.axes.items()
    }

    def tensors(specs: dict[str, Any]) -> dict[str, Any]:
        return {
            name: {"shape": None if spec.shape is None else list(spec.shape), "dtype": spec.dtype}
            for name, spec in specs.items()
        }

    return {
        "name": f"embodiinfer_{operator.name}",
        "op_type": operator.op_type,
        "description": operator.summary,
        "tags": [f"model:{model}" for model in operator.models],
        "axes": axes,
        "inputs": tensors(operator.inputs),
        "outputs": tensors(operator.outputs),
        "constraints": list(operator.constraints),
        "reference": operator.reference,
    }


def _workload_inputs(operator: Operator, work: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for name, spec in operator.inputs.items():
        if name in work.fixed:
            inputs[name] = {
                "type": "safetensors",
                "path": f"data/{work.uuid}.safetensors",
                "tensor_key": name,
            }
        elif name in work.scalars:
            inputs[name] = {"type": "scalar", "value": work.scalars[name]}
        elif spec.shape is None:
            raise ContractError(f"{operator.name}/{work.uuid}: scalar input {name} needs a value")
        else:
            inputs[name] = {"type": "random"}
    return inputs


def _baseline(operator: Operator) -> str:
    module = operator.source.removesuffix(".py").replace("/", ".")
    arguments = ", ".join(operator.inputs)
    return f'''"""Generated baseline: EmbodiInfer's production kernel, called as in the model.

The kernel source under {VENDOR}/ is copied from the repository; see README.md.
"""

from .{VENDOR}.{module} import {operator.entry}


def run({arguments}):
    return {operator.call}
'''


def _readme(operator: Operator, vendored: dict[str, tuple[str, str]], revision: str, hardware: bool) -> str:
    def shape(spec: Any) -> str:
        return "scalar" if spec.shape is None else "[" + ", ".join(spec.shape) + "]"

    tensors = "\n".join(
        f"| {kind} | `{name}` | {shape(spec)} | {spec.dtype} |"
        for kind, specs in (("input", operator.inputs), ("output", operator.outputs))
        for name, spec in specs.items()
    )
    cases = "\n".join(
        f"| `{work.uuid}` | "
        + ", ".join(f"{k}={v}" for k, v in work.axes.items())
        + "".join(f", {k}={v}" for k, v in work.scalars.items())
        + f" | {work.weight:g} | {work.note} |"
        for work in operator.workloads
    )
    sources = "\n".join(
        f"- `{path}` (source sha256 `{digest[:16]}`)" for path, (_, digest) in vendored.items()
    )
    if operator.precision["mode"] == "bit_exact":
        precision = "Outputs must be **bit-identical** to the reference on every seed."
    else:
        precision = (
            f"Outputs must satisfy `|actual - reference| <= {operator.precision['atol']} + "
            f"{operator.precision['rtol']} * |reference|` elementwise. Reason: {operator.precision['reason']}"
        )
    graph_note = (
        "\nTiming captures each call in a CUDA Graph, so a candidate must not synchronize with the host."
        if operator.timing == "cuda_graph"
        else ""
    )
    hardware_note = (
        "\nTarget-hardware notes supplied for this machine are in `HARDWARE.md`.\n" if hardware else ""
    )
    return f"""# {operator.name}

> Generated by `python -m scripts.kernel_tuning generate` from repository revision
> `{revision}`. Do not edit by hand; change `scripts/kernel_tuning/operators.py`
> or the kernel source, then regenerate.

{operator.summary} Used by: {", ".join(operator.models)}.

| role | name | shape | dtype |
|---|---|---|---|
{tensors}

Constraints: {", ".join(f"`{c}`" for c in operator.constraints) or "none"}.

{operator.notes}

## Baseline

`baseline.py` calls the production kernel `{operator.entry}` from `{operator.source}`
exactly as the inference engine does. It is the latency to beat; the
mathematical reference in `definition.json` defines correctness. The production
kernel and its repository imports are copied verbatim under `{VENDOR}/`:

{sources}

Start from that source. Its input checks show which layouts the engine passes;
a candidate only has to handle the declared definition and workloads.{graph_note}

## Precision

{precision}

## Workloads

Shapes are representative of the listed models, not traced traffic.

| uuid | axes and scalars | weight | where |
|---|---|---|---|
{cases}
{hardware_note}"""


def render(
    operator: Operator,
    destination: Path,
    *,
    overrides: dict[str, Any] | None = None,
    hardware_notes: Path | None = None,
    force: bool = False,
) -> TaskPackage:
    """Write one task package, replacing only a directory this generator created."""
    if destination.exists():
        if not (destination / MARKER).is_file() and any(destination.iterdir()):
            raise ContractError(f"{destination} exists and was not generated; choose another output")
        if not force:
            raise ContractError(f"{destination} exists; pass --force to regenerate it")
        shutil.rmtree(destination)
    module = operator.source.removesuffix(".py").replace("/", ".")
    vendored = vendor_sources(module)
    revision = _git_revision()
    settings = tuning_settings(operator, overrides)
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        files: dict[str, str | bytes] = {
            "definition.json": json.dumps(_definition(operator), indent=2) + "\n",
            "workloads.jsonl": "".join(
                json.dumps(
                    {
                        "definition": f"embodiinfer_{operator.name}",
                        "workload": {
                            "uuid": work.uuid,
                            "axes": work.axes,
                            "inputs": _workload_inputs(operator, work),
                        },
                        "solution": None,
                        "evaluation": None,
                    }
                )
                + "\n"
                for work in operator.workloads
            ),
            "baseline.py": _baseline(operator),
            "tuning.yaml": _dump_yaml(settings),
            "README.md": _readme(operator, vendored, revision, hardware_notes is not None),
        }
        files.update({path: text for path, (text, _) in vendored.items()})
        for work in operator.workloads:
            if work.fixed:
                files[f"data/{work.uuid}.safetensors"] = safetensors_bytes(
                    {name: (operator.inputs[name].dtype, values) for name, values in work.fixed.items()}
                )
        if hardware_notes is not None:
            files["HARDWARE.md"] = hardware_notes.read_text(encoding="utf-8")
        files[MARKER] = (
            json.dumps(
                {
                    "generator": "scripts.kernel_tuning.generate",
                    "operator": operator.name,
                    "repository_revision": revision,
                    "sources": {path: digest for path, (_, digest) in vendored.items()},
                    "catalog_entry": _catalog_entry(operator),
                },
                indent=2,
            )
            + "\n"
        )
        for name, content in files.items():
            path = staging / PurePosixPath(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8", newline="\n")
        TaskPackage.load(staging)
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return TaskPackage.load(destination)


def _catalog_entry(operator: Operator) -> dict[str, Any]:
    entry = asdict(operator)
    entry["workloads"] = [asdict(work) for work in operator.workloads]
    return json.loads(json.dumps(entry))


def _dump_yaml(settings: dict[str, Any]) -> str:
    import yaml

    return (
        "# Generated; see README.md. Machine-specific values may be overridden at generation.\n"
        + yaml.safe_dump(settings, sort_keys=False)
    )


def stale_sources(task: Path) -> list[str]:
    """Copied sources whose repository file has changed since generation."""
    marker = read_json(task / MARKER)
    changed = []
    for path, digest in marker["sources"].items():
        source = REPOSITORY / PurePosixPath(path).relative_to(VENDOR)
        if not source.is_file() or _text_digest(source) != digest:
            changed.append(path)
    return changed
