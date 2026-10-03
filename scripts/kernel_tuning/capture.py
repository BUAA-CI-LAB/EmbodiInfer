"""Record the shapes of production kernel calls while a model actually runs.

``python -m scripts.kernel_tuning capture --output calls.jsonl -- SCRIPT [ARGS]``
executes an unmodified inference or benchmark script in this interpreter with
each catalog operator's production function wrapped. Every call is converted to
the operator's task axes, scalars, and fixed integer inputs, so captured traffic
can replace the catalog's estimated workloads (``generate --captured``).

Run it in the model's own runtime, with CUDA Graphs and compilation disabled:
inside a graph only the capture pass reaches Python, so counts would be lost.
Shapes depend on the configuration and inputs, never on the weight values.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import json
import re
import runpy
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import ContractError
from .operators import Operator, Workload, catalog

Bound = dict[str, Any]


class Skip(Exception):
    """A call outside the task's semantics; it is counted but yields no workload."""


def _bf16(*tensors: Any) -> None:
    for tensor in tensors:
        if str(tensor.dtype) != "torch.bfloat16":
            raise Skip(f"dtype {tensor.dtype}")


def _ada_rms_norm(a: Bound) -> dict[str, Any]:
    x, modulation = a["x"], a["modulation"]
    _bf16(x, modulation)
    batch, tokens, hidden = x.shape
    return {
        "axes": {"B": batch, "T": tokens, "H": hidden, "H3": modulation.shape[1]},
        "scalars": {"eps": a["eps"]},
    }


def _gated_residual(a: Bound) -> dict[str, Any]:
    _bf16(a["residual"], a["update"], a["gate"])
    batch, tokens, hidden = a["residual"].shape
    return {"axes": {"B": batch, "T": tokens, "H": hidden, "G": a["gate"].shape[1]}}


def _gated_gelu(a: Bound) -> dict[str, Any]:
    _bf16(a["gate"], a["up"])
    batch, tokens, width = a["gate"].shape
    return {"axes": {"B": batch, "T": tokens, "I": width}}


def _rotate_qk(a: Bound) -> dict[str, Any]:
    _bf16(a["query"], a["key"], a["cos"], a["sin"])
    batch, heads, size, width = a["query"].shape
    return {"axes": {"B": batch, "HQ": heads, "HK": a["key"].shape[1], "S": size, "D": width}}


def _split_kv_attention(a: Bound) -> dict[str, Any]:
    # A padding mask keeps the task's shapes and work; it only changes which keys count.
    _bf16(a["query"], a["prefix_key"])
    batch, heads, size, width = a["query"].shape
    axes = {"B": batch, "HQ": heads, "HK": a["prefix_key"].shape[1], "P": a["prefix_key"].shape[2]}
    return {"axes": {**axes, "S": size, "D": width}, "scalars": {"scaling": float(a["scaling"])}}


def _rotate_half_rope(a: Bound) -> dict[str, Any]:
    q, k, cos = a["q"], a["k"], a["cos"]
    _bf16(q, k)
    if str(cos.dtype) != "torch.float32":
        raise Skip(f"cos dtype {cos.dtype}")
    if cos.shape[0] != q.shape[0]:
        raise Skip("broadcast cos/sin rows")
    tokens, heads, width = q.shape
    return {"axes": {"T": tokens, "HQ": heads, "HK": k.shape[1], "D": width}}


def _segmented_attention(a: Bound) -> dict[str, Any]:
    q, k, offsets = a["q"], a["k"], a["q_segment_offsets"]
    _bf16(q, k, a["v"])
    if a["kv_segment_offsets"] is not None and a["kv_segment_offsets"] is not offsets:
        raise Skip("separate key/value segments")
    if k.shape[0] != q.shape[0]:
        raise Skip("cross-length attention")
    if a["scaling"] is not None and abs(a["scaling"] - q.shape[-1] ** -0.5) > 1e-12:
        raise Skip("non-default scaling")
    if str(offsets.dtype) != "torch.int32":
        raise Skip(f"offset dtype {offsets.dtype}")
    if _capturing(q):
        raise Skip("offsets unreadable during CUDA Graph capture")
    bounds = [int(x) for x in offsets.tolist()]
    longest = max(end - start for start, end in zip(bounds, bounds[1:]))
    bound = a["max_query_length"] or longest
    if (a["max_key_length"] or longest) != bound:
        raise Skip("different query and key bounds")
    tokens, heads, width = q.shape
    return {
        "axes": {"T": tokens, "HQ": heads, "HK": k.shape[1], "D": width, "N": len(bounds)},
        "scalars": {"max_length": int(bound)},
        "fixed": {"segment_offsets": bounds},
    }


def _rms_norm(a: Bound) -> dict[str, Any]:
    _bf16(a["inputs"])
    width = a["inputs"].shape[-1]
    return {"axes": {"M": a["inputs"].numel() // width, "H": width}, "scalars": {"eps": float(a["epsilon"])}}


def _add_rms_norm(a: Bound) -> dict[str, Any]:
    _bf16(a["residual"], a["update"])
    width = a["residual"].shape[-1]
    return {
        "axes": {"M": a["residual"].numel() // width, "H": width},
        "scalars": {"eps": float(a["epsilon"])},
    }


def _swiglu(a: Bound) -> dict[str, Any]:
    _bf16(a["packed"])
    width = a["packed"].shape[-1]
    return {"axes": {"M": a["packed"].numel() // width, "I": width // 2}}


#: Production arguments -> task axes/scalars/fixed inputs, keyed by catalog name.
MAPPINGS: dict[str, Callable[[Bound], dict[str, Any]]] = {
    "ada_rms_norm": _ada_rms_norm,
    "gated_residual": _gated_residual,
    "gated_gelu": _gated_gelu,
    "rotate_qk": _rotate_qk,
    "split_kv_attention": _split_kv_attention,
    "rotate_half_rope": _rotate_half_rope,
    "segmented_attention": _segmented_attention,
    "rms_norm": _rms_norm,
    "add_rms_norm": _add_rms_norm,
    "swiglu": _swiglu,
}


def _capturing(tensor: Any) -> bool:
    if not tensor.is_cuda:
        return False
    import torch

    return torch.cuda.is_current_stream_capturing()


class Recorder:
    """Convert production calls to task cases; counts identical cases instead of storing calls."""

    def __init__(self) -> None:
        self.cases: Counter[str] = Counter()
        self.skipped: Counter[tuple[str, str]] = Counter()

    def record(self, name: str, bound: Bound) -> None:
        """Count one call. A mapping failure must never break the model being measured."""
        try:
            case = MAPPINGS[name](bound)
        except Skip as exc:
            self.skipped[name, str(exc)] += 1
            return
        except Exception as exc:  # noqa: BLE001 -- recording is best effort by design
            self.skipped[name, f"unmapped call: {type(exc).__name__}: {exc}"] += 1
            return
        self.cases[json.dumps({"operator": name, **case}, sort_keys=True)] += 1

    def rows(self) -> list[dict[str, Any]]:
        """One row per distinct case with its call count, plus counted skips."""
        rows = [{**json.loads(key), "count": count} for key, count in self.cases.most_common()]
        rows += [
            {"operator": name, "skipped": reason, "count": count}
            for (name, reason), count in self.skipped.most_common()
        ]
        return rows


def wrap(name: str, function: Callable[..., Any], recorder: Recorder) -> Callable[..., Any]:
    """Return a drop-in replacement that records each call before running the original."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def recorded(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        recorder.record(name, dict(bound.arguments))
        return function(*args, **kwargs)

    recorded.__wrapped_kernel__ = function  # type: ignore[attr-defined]
    return recorded


def install(operators: list[Operator], recorder: Recorder) -> list[tuple[Any, str, Any]]:
    """Wrap each production function wherever an imported module already holds it.

    Lazy ``from ... import`` inside model code reads the defining module at call
    time; modules importing the backend package later receive the wrapper too.
    Returns what was replaced so the caller can restore it.
    """
    replaced = []
    for operator in operators:
        module = importlib.import_module(operator.source.removesuffix(".py").replace("/", "."))
        original = getattr(module, operator.entry)
        wrapper = wrap(operator.name, original, recorder)
        for holder in list(sys.modules.values()):
            for attribute, value in list(getattr(holder, "__dict__", {}).items()):
                if value is original:
                    replaced.append((holder, attribute, original))
                    setattr(holder, attribute, wrapper)
    return replaced


def run(command: list[str], output: Path, operators: list[Operator]) -> list[dict[str, Any]]:
    """Run ``SCRIPT [ARGS]`` or ``-m MODULE [ARGS]`` as ``__main__`` and write the captured cases."""
    if not command:
        raise ContractError("Give the script to run after --, e.g. -- benchmark.py --config config.yaml")
    recorder = Recorder()
    replaced = install(operators, recorder)
    argv = sys.argv
    try:
        if command[0] == "-m":
            sys.argv = command[1:]
            runpy.run_module(command[1], run_name="__main__", alter_sys=True)
        else:
            script = Path(command[0]).resolve()
            sys.argv = [str(script), *command[1:]]
            sys.path.insert(0, str(script.parent))
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = argv
        for holder, attribute, original in replaced:
            setattr(holder, attribute, original)
        rows = recorder.rows()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def load(path: Path) -> list[dict[str, Any]]:
    """Read a capture file written by :func:`run`."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ContractError(f"{path} contains no captured calls")
    return rows


def apply(operator: Operator, rows: list[dict[str, Any]], *, top: int = 8) -> tuple[Operator, str]:
    """Replace an operator's estimated workloads with its most frequent captured cases.

    Weights are call counts, so the promotion criterion follows real traffic.
    Returns the operator unchanged when the capture saw no usable call of it.
    """
    cases = [row for row in rows if row["operator"] == operator.name and "axes" in row]
    skipped = sum(row["count"] for row in rows if row["operator"] == operator.name and "skipped" in row)
    if not cases:
        return operator, f"{operator.name}: no captured calls ({skipped} skipped); keeping estimated shapes"
    cases.sort(key=lambda row: -row["count"])
    kept, total = cases[:top], sum(row["count"] for row in cases)
    variable = [name for name, value in operator.axes.items() if value is None]
    workloads = []
    for index, row in enumerate(kept):
        if set(row["axes"]) != set(variable):
            raise ContractError(
                f"Capture for {operator.name} binds {sorted(row['axes'])}, expected {variable}"
            )
        label = "-".join(f"{name}{row['axes'][name]}" for name in variable)
        workloads.append(
            Workload(
                re.sub(r"[^A-Za-z0-9-]", "", f"captured{index + 1}-{label}")[:64],
                row["axes"],
                scalars=row.get("scalars", {}),
                fixed={name: tuple(values) for name, values in row.get("fixed", {}).items()},
                weight=float(row["count"]),
                note=f"{row['count']} of {total} captured calls",
            )
        )
    share = sum(row["count"] for row in kept) / total
    summary = f"{operator.name}: {len(kept)} captured shapes covering {share:.0%} of {total} calls"
    if skipped:
        summary += f"; {skipped} calls outside the task skipped"
    return dataclasses.replace(operator, workloads=tuple(workloads)), summary


def apply_file(operators: list[Operator], path: Path | None) -> tuple[list[Operator], list[str]]:
    """Apply a capture file to every selected operator; ``None`` keeps the catalog shapes."""
    if path is None:
        return operators, []
    rows = load(path)
    if unknown := {row["operator"] for row in rows} - catalog().keys():
        raise ContractError(f"Capture names operators outside the catalog: {sorted(unknown)}")
    applied = [apply(operator, rows) for operator in operators]
    return [operator for operator, _ in applied], [summary for _, summary in applied]


__all__ = ["MAPPINGS", "Recorder", "apply", "apply_file", "install", "load", "run", "wrap"]
