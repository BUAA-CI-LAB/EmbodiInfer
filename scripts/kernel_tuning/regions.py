"""Capture operator-registry calls of an end-to-end run as a tunable model capture.

``python -m scripts.kernel_tuning capture-ops --model MODEL --output DIR -- DRIVER``
runs an unmodified inference driver in the model runtime. Calls that the policy
resolves through :mod:`embodiinfer.layers.ops` during the measured steps (after
``--warmup``) are recorded with their real inputs. The result is a model capture
in the discovery format, so ``calibrate``, ``generate`` and ``tune-all`` apply
unchanged. Each operator's identity has ``kind: registry``: its baseline and
its FP64 oracle are the Torch reference in ``embodiinfer.backend.torch.ops``,
and the implementation the model ran is shipped as the agent's starting point.

Record in eager mode (no CUDA graphs, no ``torch.compile``): graph replays and
compiled regions do not call the registry from Python.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import runpy
import sys
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .contracts import ContractError, digest, tree_hashes
from .hotspots import _ENTRIES

REFERENCE_MODULE = "embodiinfer.backend.torch.ops"


def _portable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(v) for v in value]
    return str(value)


class OpCapture:
    """Record registry-operator calls of the measured inference steps."""

    def __init__(
        self,
        model: str,
        output: Path,
        *,
        ops: set[str] | None = None,
        warmup: int = 2,
        steps: int = 8,
        fixture_bytes: int = 1 << 30,
    ) -> None:
        if warmup < 0 or steps <= 0:
            raise ContractError("capture-ops needs warmup >= 0 and steps > 0")
        self.model, self.root, self.ops = model, output.resolve(), ops
        self.warmup, self.steps, self.fixture_bytes = warmup, steps, fixture_bytes
        if self.root.exists():
            raise ContractError(f"{self.root} already exists; captures are immutable")
        self.root.mkdir(parents=True)
        self.started = 0
        self.lock = threading.Lock()
        self.local = threading.local()
        self.operators: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.target: dict[str, Any] | None = None
        self.used_bytes = 0
        self._digests: dict[str, dict[str, str]] = {}
        self._fixtures: dict[str, list[dict[str, Any]]] = {}
        self._restore: list[Callable[[], None]] = []
        import torch

        self.precision = {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        self.flags = {
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "fp16_reduced_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }

    @property
    def measuring(self) -> bool:
        """Whether the current thread is inside a measured inference step."""
        return getattr(self.local, "measured", False)

    def _replace(self, owner: Any, name: str, value: Any) -> None:
        had = name in vars(owner)
        original = vars(owner).get(name)
        setattr(owner, name, value)
        self._restore.append(lambda: setattr(owner, name, original) if had else delattr(owner, name))

    def _replace_everywhere(self, original: Any, value: Any) -> None:
        for module in list(sys.modules.values()):
            if module is None:
                continue
            for name, item in list(vars(module).items()):
                if item is original:
                    self._replace(module, name, value)

    def _step(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if getattr(self.local, "depth", 0):
                return function(*args, **kwargs)
            with self.lock:
                index = self.started
                self.started += 1
            self.local.depth = 1
            self.local.measured = self.warmup <= index < self.warmup + self.steps
            try:
                return function(*args, **kwargs)
            finally:
                self.local.depth = 0
                self.local.measured = False

        return wrapper

    def install(self) -> None:
        """Patch policy construction, inference entries and registry resolution."""
        from embodiinfer.layers.ops import KernelSelection, get_op
        from embodiinfer.policies import factory

        for module_name, owner, methods in _ENTRIES:
            holder = getattr(importlib.import_module(module_name), owner)
            for method in methods:
                if hasattr(holder, method):
                    self._replace(holder, method, self._step(getattr(holder, method)))

        original_factory = factory.make_policy

        @functools.wraps(original_factory)
        def make_policy(name: str, **kwargs: Any) -> Any:
            if name != self.model:
                raise ContractError(f"The driver constructed {name!r}, expected {self.model!r}")
            policy = original_factory(name, **kwargs)
            self.target = {
                "model": name,
                "type": f"{type(policy).__module__}.{type(policy).__qualname__}",
                "config": _portable(kwargs),
            }
            return policy

        self._replace_everywhere(original_factory, make_policy)
        original_resolve = KernelSelection.resolve
        capture = self

        @functools.wraps(original_resolve)
        def resolve(selection: Any, op: str, prefer: str = "torch", *args: Any, **kwargs: Any) -> Any:
            function = original_resolve(selection, op, prefer, *args, **kwargs)
            if capture.ops is not None and op not in capture.ops:
                return function
            implementation = get_op(op, "torch" if selection.mode == "torch" else prefer)
            return capture._recording(op, implementation, function)

        self._replace(KernelSelection, "resolve", resolve)

    def _recording(self, op: str, implementation: Any, function: Any) -> Any:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import torch

            if (
                not self.measuring
                or torch.compiler.is_compiling()
                or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
            ):
                return function(*args, **kwargs)
            return self.record(op, implementation, function, args, kwargs)

        return wrapper

    def _source_digests(self, module: str) -> dict[str, str]:
        if module not in self._digests:
            from .generate import vendor_sources

            self._digests[module] = {path: value for path, (_, value) in vendor_sources(module).items()}
        return self._digests[module]

    def record(self, op: str, implementation: Any, function: Any, args: tuple, kwargs: dict) -> Any:
        """Snapshot one call, derive its output contract from the Torch reference, then run it."""
        from embodiinfer.layers.ops import get_op

        from .discovery.tensors import (
            clone_inputs,
            describe_inputs,
            flatten,
            restore,
            storage_key,
            tensor_outputs,
            tensor_spec,
        )

        try:
            tensors: list[Any] = []
            recipe = flatten((args, kwargs), tensors)
            specs = describe_inputs(tensors)
            reference = get_op(op, "torch")
            local = clone_inputs(tensors)
            ref_args, ref_kwargs = restore(recipe, local)
            result = reference(*ref_args, **ref_kwargs)
            outputs = tensor_outputs(result)
            if not outputs:
                raise ContractError("registry operators must return tensors")
            identity = {
                "kind": "registry",
                "name": f"registry.{op}",
                "op": op,
                "module": REFERENCE_MODULE,
                "entry": op,
                "source_digests": self._source_digests(REFERENCE_MODULE),
                "start_module": implementation.__module__,
                "start_entry": implementation.__name__,
                "start_digests": self._source_digests(implementation.__module__),
                "arguments": recipe,
                "input_dtypes": [spec["dtype"] for spec in specs],
                "input_ranks": [len(spec["shape"]) for spec in specs],
                "mutates": [],
                "returns": flatten(result, []),
            }
            # The first call of each structural case supplies its real inputs;
            # later calls with the same layout only add to the case's count.
            structure = digest([identity, specs])
            if structure not in self._fixtures:
                self._fixtures[structure] = self._snapshot(tensors, specs)
            case = {
                "inputs": self._fixtures[structure],
                "outputs": [tensor_spec(tensor) for tensor in outputs],
                "output_aliases": [
                    [i for i, tensor in enumerate(local) if storage_key(tensor) == storage_key(output)]
                    for output in outputs
                ],
                "output_groups": [
                    [i for i, other in enumerate(outputs) if storage_key(output) == storage_key(other)]
                    for output in outputs
                ],
            }
            identity["output_dtypes"] = [spec["dtype"] for spec in case["outputs"]]
            identity["output_ranks"] = [len(spec["shape"]) for spec in case["outputs"]]
            self._add(identity, case)
        except (ContractError, RuntimeError, TypeError, ValueError) as exc:
            message = f"registry.{op}: {exc}"
            if message not in self.errors:
                self.errors.append(message)
        return function(*args, **kwargs)

    def _snapshot(self, tensors: list[Any], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import torch

        from .discovery.tensors import extent

        specs = [dict(spec) for spec in specs]
        for group in {spec["storage"] for spec in specs}:
            indices = [i for i, spec in enumerate(specs) if spec["storage"] == group]
            tensor = tensors[indices[0]]
            size = max(extent(specs[i]) for i in indices)
            data = tensor.detach().as_strided((size,), (1,), 0).contiguous().view(torch.uint8).cpu()
            data = data.numpy().tobytes()
            path = f"fixtures/{hashlib.sha256(data).hexdigest()}.bin"
            file = self.root / path
            if not file.exists():
                if self.used_bytes + len(data) > self.fixture_bytes:
                    raise ContractError("input fixtures exceed --fixture-bytes; raise the budget")
                file.parent.mkdir(exist_ok=True)
                file.write_bytes(data)
                self.used_bytes += len(data)
            for i in indices:
                specs[i]["fixture"] = path
        return specs

    def _add(self, identity: dict[str, Any], case: dict[str, Any]) -> None:
        op_id = digest(identity)[:20]
        op = self.operators.setdefault(
            op_id,
            {"id": op_id, "name": identity["name"], "identity": identity, "status": "ready", "workloads": []},
        )
        case_id = digest(case)[:20]
        existing = next((item for item in op["workloads"] if item["id"] == case_id), None)
        if existing is None:
            existing = {**case, "id": case_id, "count": 0}
            op["workloads"].append(existing)
        existing["count"] += 1

    def close(self, *, complete: bool) -> None:
        """Restore every patch and publish the hashed capture."""
        import torch

        from .discovery.contracts import SCHEMA
        from .generate import _git_revision

        while self._restore:
            self._restore.pop()()
        if self.target is None:
            self.errors.append("No policy was built through make_policy")
        if self.started < self.warmup + self.steps:
            self.errors.append(
                f"Only {self.started} inference steps ran; capture needs warmup {self.warmup} + {self.steps}"
            )
        recorded = {op["identity"]["op"] for op in self.operators.values()}
        for op in sorted((self.ops or set()) - recorded):
            self.errors.append(
                f"registry.{op} was not called during the measured steps; run the driver without "
                "CUDA graphs or torch.compile so registry calls happen in Python"
            )
        if not self.operators and not self.errors:
            self.errors.append("No registry operator calls were recorded")
        operators = list(self.operators.values())
        atomic_json(self.root / "operators.json", operators)
        atomic_json(
            self.root / "coverage.json",
            {"operators": dict(Counter(op["status"] for op in operators)), "errors": self.errors},
        )
        config = (self.target or {}).get("config", {})
        atomic_json(
            self.root / "manifest.json",
            {
                "schema": SCHEMA,
                "capture": "registry",
                "target": self.target or {},
                "complete": complete,
                "errors": self.errors,
                "window": {"warmup": self.warmup, "steps": self.steps},
                "repository_revision": _git_revision(),
                "torch_version": str(torch.__version__),
                "precision": self.precision,
                "flags": self.flags,
                "device_type": "cuda",
                "compile_backend": config.get("compile_backend", "none"),
                "fixture_budget_bytes": self.fixture_bytes,
                "fixture_bytes": self.used_bytes,
                "files": tree_hashes(self.root),
            },
        )


def run(
    command: list[str],
    output: Path,
    model: str,
    *,
    ops: set[str] | None = None,
    warmup: int = 2,
    steps: int = 8,
    fixture_bytes: int = 1 << 30,
) -> dict[str, Any]:
    """Run an inference driver and write a registry-operator model capture."""
    from .discovery.contracts import ModelCapture

    if not command:
        raise ContractError("Supply the inference driver after --")
    capture = OpCapture(model, output, ops=ops, warmup=warmup, steps=steps, fixture_bytes=fixture_bytes)
    argv, path = sys.argv, sys.path[:]
    complete = False
    try:
        capture.install()
        if command[0] == "-m":
            sys.argv = command[1:]
            runpy.run_module(command[1], run_name="__main__", alter_sys=True)
        else:
            script = Path(command[0]).resolve()
            sys.argv = [str(script), *command[1:]]
            sys.path.insert(0, str(script.parent))
            runpy.run_path(str(script), run_name="__main__")
        complete = True
    except SystemExit as exc:
        complete = exc.code in (None, 0)
        if not complete:
            raise
    finally:
        sys.argv, sys.path = argv, path
        capture.close(complete=complete)
    result = ModelCapture.load(output, model)
    return {
        "model": model,
        "identity": result.identity,
        "operators": {
            op["identity"]["op"]: sum(case["count"] for case in op["workloads"]) for op in result.operators
        },
    }
