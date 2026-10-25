"""Observe one model's existing execution, without replacing its numerical path."""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import runpy
import sys
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from ..artifacts import REPOSITORY, atomic_json
from ..contracts import ContractError, digest, tree_hashes
from ..generate import _git_revision, vendor_sources
from .contracts import SCHEMA
from .numerics import policy
from .rng import contract_for, requires_rng_adapter, undefined_outputs
from .rng import snapshot as rng_snapshot
from .tensors import describe_inputs, extent, flatten, storage_key, tensor_outputs, tensor_spec

_VIEWS = {
    "alias",
    "detach",
    "view",
    "_unsafe_view",
    "transpose",
    "permute",
    "as_strided",
    "expand",
    "squeeze",
    "unsqueeze",
    "slice",
    "select",
    "split",
    "split_with_sizes",
    "unbind",
    "t",
    "as_strided_",
    "transpose_",
    "squeeze_",
    "unsqueeze_",
    "detach_",
    "resize_",
    "resize_as_",
    "set_",
}
_ALLOCATIONS = {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided"}
_HOST = {"item", "_local_scalar_dense", "is_nonzero"}
# Weight loading, casting and device placement are not inference arithmetic.
_COPIES = {"_to_copy", "copy_", "clone", "contiguous", "lift_fresh", "_copy_from", "_copy_from_and_resize"}
# Production inference enters through the model-agnostic engine contract. Every
# operator running below one of these calls, on any thread, belongs to the model.
_ENTRIES = (
    ("embodiinfer.engine.core", "EngineCore", ("__init__", "execute", "execute_pipelined", "_pipeline_step")),
    (
        "embodiinfer.engine.rollout.generation_backend",
        "GenerationBackend",
        ("generate", "generate_with_logprob", "sample_group", "best_of_n"),
    ),
)
_QUANTIZED_ATEN = {
    "_scaled_mm",
    "_weight_int8pack_mm",
    "_weight_int4pack_mm",
    "_convert_weight_to_int4pack",
    "quantize_per_tensor",
    "quantize_per_channel",
    "dequantize",
}
# These contracts describe existing public kernels, not an operator discovery whitelist.
_PURE_KERNELS = {
    "rms_norm",
    "add_rms_norm",
    "ada_rms_norm",
    "gated_residual",
    "swiglu",
    "gated_gelu",
    "rotate_qk",
    "rotate_half_rope",
    "segmented_attention",
    "split_kv_attention",
    "flash_prefill_attention",
    "flash_prefill_attention_graph",
    "gqa_decode_graph",
}
_KERNEL_WRITES = {
    "fused_rope_cache": ("key_cache", "value_cache"),
    "fused_rope_cache_graph": ("key_cache", "value_cache"),
    "fused_prefill_rope_cache_graph": ("key_cache", "value_cache"),
    "fused_greedy": ("workspace",),
    "fused_lm_head_greedy": ("workspace",),
    "mark_penalized": ("workspace",),
}


def _portable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _portable(getattr(value, field.name)) for field in fields(value)}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


class _Observer(TorchDispatchMode):
    def __init__(self, session: CaptureSession):
        super().__init__()
        self.session = session

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        session = self.session
        kwargs = kwargs or {}
        if session.suppressed:
            return func(*args, **kwargs)
        if not session.depth:
            session.outside(func, args, kwargs)
            return func(*args, **kwargs)
        name = str(func)
        short = name.split(".")[1] if name.startswith("aten.") else name
        reason = None
        with session.silence():
            rng = contract_for(name, kwargs)
        if not name.startswith("aten."):
            reason = "External dispatcher operator needs a reference adapter"
        if rng is None and requires_rng_adapter(func, args, kwargs):
            reason = "Random/nondeterministic operator needs an RNG replay adapter"
        writes = []
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is not None and argument.alias_info.is_write:
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                writes.append(value)
        quantized = name.split(".")[0] in {"quantized", "quantized_decomposed"} or short in _QUANTIZED_ATEN
        origin = {"kind": "aten", "name": name, "schema": str(func._schema)}
        if undefined := undefined_outputs(func, args, kwargs):
            origin["undefined_outputs"] = undefined
        return session.call(
            origin,
            func,
            args,
            kwargs,
            writes=writes,
            reason=reason,
            rng=rng if rng is not None and rng["devices"][-1].split(":")[0] == session.device_type else None,
            classification="excluded_quantized"
            if quantized
            else "metadata_only"
            if short in _VIEWS | _ALLOCATIONS
            else "host_only"
            if short in _HOST
            else None,
        )


class CaptureSession:
    """Collect one bound model's calls; failures remain explicit coverage gaps.

    Use ``observe`` around existing preparation execution for custom drivers. The
    CLI additionally binds a target constructed by the public policy factory.
    ``device_type='cpu'`` is useful for deterministic observer unit tests only.
    """

    def __init__(
        self, model: str, output: Path, *, fixture_bytes: int = 64 * 1024 * 1024, device_type: str = "cuda"
    ) -> None:
        if not model or fixture_bytes < 0:
            raise ContractError("Select one model and a nonnegative fixture byte budget")
        self.model, self.root = model, output.resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ContractError("Capture destination must be empty; use a new directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.fixture_bytes, self.used_bytes, self.device_type = fixture_bytes, 0, device_type
        self.target: dict[str, Any] = {}
        self.instance: Any = None
        # Dispatch modes are thread-local, so inference scope is tracked per thread.
        self._local = threading.local()
        self.entries = 0
        self.bypassed: Counter = Counter()
        self._weights: tuple[int, frozenset[int]] = (0, frozenset())
        self.operators: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.replacements: list[tuple[Any, str, Any, bool]] = []
        self._fixtures: dict[str, list[dict[str, Any]]] = {}
        self.graphs: dict[int, Counter] = {}
        self.graph_stack: list[Counter] = []
        self._numerics_bound = False
        self._set_numerics()

    def _set_numerics(self) -> None:
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

    def _check_numerics(self) -> None:
        previous = (self.precision, self.flags)
        self._set_numerics()
        if self._numerics_bound and previous != (self.precision, self.flags):
            self.errors.append("Numerical settings changed during model preparation")
        self._numerics_bound = True

    @property
    def depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @depth.setter
    def depth(self, value: int) -> None:
        self._local.depth = value

    @property
    def suppressed(self) -> int:
        return getattr(self._local, "suppressed", 0)

    @suppressed.setter
    def suppressed(self, value: int) -> None:
        self._local.suppressed = value

    @contextmanager
    def inference(self) -> Iterator[None]:
        """Observe every operator below one inference call on the current thread."""
        observer = None
        if not getattr(self._local, "observing", False):
            observer = _Observer(self)
            observer.__enter__()
            self._local.observing = True
        if not self.depth:
            self.entries += 1
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if observer is not None:
                self._local.observing = False
                observer.__exit__(None, None, None)

    @contextmanager
    def watch(self) -> Iterator[None]:
        """Watch the driver thread outside inference to reject bypassed model arithmetic."""
        self._local.observing = True
        try:
            with _Observer(self):
                yield
        finally:
            self._local.observing = False

    def _entry(self, function: Any) -> Any:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.inference():
                return function(*args, **kwargs)

        return wrapper

    def _weight_storages(self) -> frozenset[int]:
        tensors = [*self.instance.parameters(), *self.instance.buffers()]
        if not tensors:
            return frozenset()
        probe = tensors[0].untyped_storage().data_ptr()
        # Moving or casting the model replaces storages; refresh only then.
        if probe != self._weights[0]:
            self._weights = (probe, frozenset(t.untyped_storage().data_ptr() for t in tensors) - {0})
        return self._weights[1]

    def outside(self, func: Any, args: tuple, kwargs: dict) -> None:
        """Record model-weight arithmetic that bypassed every inference entry."""
        if not isinstance(self.instance, torch.nn.Module):
            return
        name = str(func)
        short = name.split(".")[1] if name.startswith("aten.") else name
        if short in _VIEWS | _ALLOCATIONS | _HOST | _COPIES:
            return
        with self.silence():
            weights = self._weight_storages()
            if any(
                isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() in weights
                for t in tree_leaves((args, kwargs))
            ):
                self.bypassed[name] += 1

    @contextmanager
    def silence(self) -> Iterator[None]:
        """Prevent instrumentation and atomic/quantized internals from becoming tasks."""
        self.suppressed += 1
        try:
            yield
        finally:
            self.suppressed -= 1

    def bind(self, instance: Any, config: dict[str, Any]) -> None:
        """Bind to an actual object rather than trusting an unverified model label."""
        if self.instance is not None and self.instance is not instance:
            raise ContractError("Prepare exactly one target policy instance per capture")
        modules = instance.modules() if isinstance(instance, torch.nn.Module) else (instance,)
        if config.get("compile_backend", "none") != "none" or any(
            hasattr(module, "_orig_mod") for module in modules
        ):
            raise ContractError("Model discovery requires explicit uncompiled preparation")
        self.instance = instance
        self.target = {
            "model": self.model,
            "type": f"{type(instance).__module__}.{type(instance).__qualname__}",
            "config": _portable(config),
        }

    @contextmanager
    def observe(self, instance: Any, **config: Any) -> Iterator[None]:
        """Observe the existing preparation call without running it a second time."""
        self.bind(instance, config)
        with self.inference():
            yield

    def _snapshot(
        self, key: str, tensors: list[torch.Tensor], specs: list[dict[str, Any]], *, real_inputs: bool = False
    ) -> list[dict[str, Any]]:
        if key in self._fixtures:
            return self._fixtures[key]
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            if any(not tensor.is_floating_point() for tensor in tensors):
                raise ContractError("Valid non-floating fixtures must be observed during eager warmup")
            return specs
        for group in {spec["storage"] for spec in specs}:
            indices = [i for i, spec in enumerate(specs) if spec["storage"] == group]
            tensor = tensors[indices[0]]
            size = max(extent(specs[i]) for i in indices)
            byte_count = size * tensor.element_size()
            required = real_inputs or not tensor.is_floating_point()
            if required and byte_count > self.fixture_bytes:
                raise ContractError(
                    "Required input fixture exceeds the capture byte budget; increase --fixture-bytes"
                )
            # Calibrated arithmetic needs real weights too. Other floating tasks
            # retain small fixtures when space permits; indices are never guessed.
            save = required or byte_count <= 1024 * 1024
            if save and not required and self.used_bytes + byte_count > self.fixture_bytes:
                save = False
            if save:
                raw = (
                    tensor.detach()
                    .as_strided((size,), (1,), 0)
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                path = f"fixtures/{hashlib.sha256(raw).hexdigest()}.bin"
                file = self.root / path
                if not file.exists():
                    if self.used_bytes + byte_count > self.fixture_bytes:
                        raise ContractError(
                            "Required real input fixture exceeds the capture byte budget; increase --fixture-bytes"
                        )
                    file.parent.mkdir(exist_ok=True)
                    file.write_bytes(raw)
                    self.used_bytes += len(raw)
                for i in indices:
                    specs[i]["fixture"] = path
        self._fixtures[key] = specs
        return specs

    def _rng_snapshot(self, devices: list[str]) -> dict[str, str]:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise ContractError("RNG fixtures require eager preparation outside CUDA Graph capture")
        result = {}
        for device, state in rng_snapshot(devices).items():
            raw = bytes(state.tolist())
            path = f"fixtures/{hashlib.sha256(raw).hexdigest()}.bin"
            file = self.root / path
            if not file.exists():
                if self.used_bytes + len(raw) > self.fixture_bytes:
                    raise ContractError("Required RNG fixture exceeds the capture byte budget")
                file.parent.mkdir(exist_ok=True)
                file.write_bytes(raw)
                self.used_bytes += len(raw)
            result[device] = path
        return result

    def _add(self, identity: dict[str, Any], case: dict[str, Any], status: str, reason: str | None) -> None:
        op_id = digest(identity)[:20]
        op = self.operators.setdefault(
            op_id,
            {
                "id": op_id,
                "name": identity["name"],
                "identity": identity,
                "status": status,
                "reason": reason,
                "workloads": [],
            },
        )
        if status == "blocked":
            op.update(status=status, reason=reason)
        case_id = digest({k: v for k, v in case.items() if k != "count"})[:20]
        existing = next((w for w in op["workloads"] if w["id"] == case_id), None)
        if existing is None:
            existing = {**case, "id": case_id, "count": 0}
            op["workloads"].append(existing)
        if self.graph_stack:
            self.graph_stack[-1][op_id, case_id] += 1
            # Retain capture-only cases too; replay traffic is counted separately.
        existing["count"] += 1

    def call(
        self,
        origin: dict[str, Any],
        function: Any,
        args: tuple,
        kwargs: dict,
        *,
        writes: list[Any] | None = None,
        reason: str | None = None,
        rng: dict[str, Any] | None = None,
        classification: str | None = None,
    ) -> Any:
        """Record before/after metadata while invoking the original callable exactly once."""
        with self.silence():
            self._check_numerics()
            tensors: list[torch.Tensor] = []
            identity = dict(origin)
            case: dict[str, Any] = {"inputs": [], "outputs": []}
            status = classification or "ready"
            explicit = reason
            try:
                recipe = flatten((args, kwargs), tensors)
                if any(t.is_quantized for t in tensors):
                    classification = status = "excluded_quantized"
                specs = describe_inputs(tensors)
                identity.update(
                    arguments=recipe,
                    input_dtypes=[s["dtype"] for s in specs],
                    input_ranks=[len(s["shape"]) for s in specs],
                )
                case["inputs"] = specs
                if not any(t.device.type == self.device_type for t in tensors):
                    status = classification or "host_only"
                mutated: list[torch.Tensor] = []
                for value in writes or []:
                    flatten(value, mutated)
                identity["mutates"] = [
                    i for i, tensor in enumerate(tensors) if any(tensor is other for other in mutated)
                ]
                if reason:
                    raise ContractError(reason)
                if status == "ready":
                    case["inputs"] = self._snapshot(
                        digest([identity, specs]), tensors, specs, real_inputs=policy(identity) is not None
                    )
                if rng is not None:
                    identity["rng"] = rng
                    case["rng"] = {"before": self._rng_snapshot(rng["devices"])}
            except (ContractError, RuntimeError, TypeError, ValueError) as exc:
                status, reason = (
                    ("excluded_quantized" if classification == "excluded_quantized" else "blocked"),
                    str(exc),
                )
                if explicit is None and classification in {"metadata_only", "host_only"}:
                    status = classification  # Not a task, so no replay recipe is needed.
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                self.errors.append(f"{origin['name']} failed during preparation: {type(exc).__name__}: {exc}")
                raise
            if (
                status == "blocked"
                and explicit is None
                and rng is None
                and not any(
                    isinstance(t, torch.Tensor) and t.device.type == self.device_type
                    for t in tree_leaves((args, kwargs, result))
                )
            ):
                # Host preprocessing with an unrepresentable argument is not a device task.
                status = "host_only"
            try:
                if rng is not None and status != "blocked":
                    case["rng"]["after"] = self._rng_snapshot(rng["devices"])
                if (
                    status == "host_only"
                    and classification is None
                    and any(t.device.type == self.device_type for t in tensor_outputs(result))
                ):
                    status = "ready"
                    # Host inputs were not snapshotted before the call; capture them now.
                    if identity["mutates"]:
                        raise ContractError("Host-to-device calls mutating inputs need a dedicated adapter")
                    case["inputs"] = self._snapshot(digest([identity, specs]), tensors, specs)
                if status not in {"metadata_only", "host_only", "excluded_quantized", "blocked"}:
                    outputs = tensor_outputs(result)
                    identity["returns"] = flatten(result, [])
                    # Mutated tensors are part of the task's observable outputs too.
                    outputs += [tensors[i] for i in identity["mutates"]]
                    if not outputs:
                        raise ContractError("Calls without tensor outputs require a dedicated task adapter")
                    case["outputs"] = [tensor_spec(t) for t in outputs]
                    identity["output_dtypes"] = [s["dtype"] for s in case["outputs"]]
                    identity["output_ranks"] = [len(s["shape"]) for s in case["outputs"]]
                    case["output_aliases"] = [
                        [i for i, tensor in enumerate(tensors) if storage_key(tensor) == storage_key(output)]
                        for output in outputs
                    ]
                    case["output_groups"] = [
                        [i for i, other in enumerate(outputs) if storage_key(output) == storage_key(other)]
                        for output in outputs
                    ]
                else:
                    case["outputs"] = [tensor_spec(t) for t in tensor_outputs(result)]
            except (ContractError, RuntimeError, TypeError, ValueError) as exc:
                if rng is not None or status not in {"metadata_only", "host_only", "excluded_quantized"}:
                    status, reason = "blocked", str(exc)
            self._add(identity, case, status, reason)
            return result

    def _replace(self, holder: Any, name: str, replacement: Any) -> None:
        self.replacements.append((holder, name, getattr(holder, name), name in vars(holder)))
        setattr(holder, name, replacement)

    def _replace_function(self, original: Any, wrapper: Any) -> None:
        for module in list(sys.modules.values()):
            if module is None:
                continue
            for name, value in list(vars(module).items()):
                if value is original:
                    self._replace(module, name, wrapper)

    def _atomic(self, function: Any, *, quantized: bool = False) -> Any:
        source_digests: dict[str, str] | None = None

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal source_digests
            if not self.depth or self.suppressed:
                return function(*args, **kwargs)
            origin = {
                "kind": "backend",
                "name": f"{function.__module__}.{function.__name__}",
                "module": function.__module__,
                "entry": function.__name__,
            }
            if quantized:
                # A module instance is provenance, not a replayable tensor input.
                call_args = args[1:] if args and isinstance(args[0], torch.nn.Module) else args
                return self.call(
                    origin,
                    lambda *a, **kw: function(*args, **kwargs),
                    call_args,
                    kwargs,
                    classification="excluded_quantized",
                )
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            source = Path(inspect.getfile(function))
            origin["source_digest"] = hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()
            names = _KERNEL_WRITES.get(function.__name__, ())
            reason = (
                None
                if function.__name__ in _PURE_KERNELS or names
                else "Backend entry requires an explicit mutation/reference contract"
            )
            try:
                if source_digests is None:
                    source_digests = {
                        path: source_digest
                        for path, (_, source_digest) in vendor_sources(function.__module__).items()
                    }
                origin["source_digests"] = source_digests
            except ContractError as exc:
                reason = str(exc)
            return self.call(
                origin, function, args, kwargs, writes=[bound.arguments[n] for n in names], reason=reason
            )

        return wrapper

    def install(self) -> None:
        """Install process-local quantization, backend, factory, and graph observation hooks."""
        from embodiinfer.models import linear
        from embodiinfer.policies import factory

        if self.model not in factory.available_policies():
            raise ContractError(f"Unknown policy {self.model!r}; choose from {factory.available_policies()}")
        self._replace(
            linear.QuantizedLinear, "forward", self._atomic(linear.QuantizedLinear.forward, quantized=True)
        )
        for name in ("quantize_linear", "pack_quantized_linears"):
            original = getattr(linear, name)
            self._replace_function(original, self._atomic(original, quantized=True))
        for method in ("fp8", "int8", "nvfp4"):
            module = importlib.import_module(f"embodiinfer.backend.torch.{method}")
            for name, function in list(vars(module).items()):
                if (
                    inspect.isfunction(function)
                    and function.__module__ == module.__name__
                    and not name.startswith("_")
                ):
                    self._replace_function(function, self._atomic(function, quantized=True))
        for path in sorted((REPOSITORY / "embodiinfer/backend/triton").glob("*.py")):
            if path.stem in {"__init__", "capability"}:
                continue
            module = importlib.import_module(f"embodiinfer.backend.triton.{path.stem}")
            for name, function in list(vars(module).items()):
                if (
                    inspect.isfunction(function)
                    and function.__module__ == module.__name__
                    and not name.startswith(("_", "supports_", "warmup_"))
                    and not name.endswith(("_available", "_capability"))
                ):
                    self._replace_function(function, self._atomic(function, quantized=path.stem == "fp8"))
        original_factory = factory.make_policy

        @functools.wraps(original_factory)
        def construct(name: str, **kwargs: Any) -> Any:
            if name != self.model:
                raise ContractError(f"Preparation constructed {name!r}, expected {self.model!r}")
            if kwargs.get("compile_backend", "none") != "none":
                raise ContractError("Disable torch.compile explicitly before model discovery")
            policy = original_factory(name, **kwargs)
            self.bind(policy, kwargs)
            # Request preprocessing and serving adapters belong to end-to-end inference.
            if callable(getattr(policy, "collate", None)):
                self._replace(policy, "collate", self._entry(policy.collate))
            build = getattr(policy, "build_serving_adapter", None)
            if callable(build):

                @functools.wraps(build)
                def serving(*args: Any, **kw: Any) -> Any:
                    with self.inference():
                        adapter = build(*args, **kw)
                    for method in ("infer", "infer_batch"):
                        if callable(getattr(adapter, method, None)):
                            self._replace(adapter, method, self._entry(getattr(adapter, method)))
                    return adapter

                self._replace(policy, "build_serving_adapter", serving)
            return policy

        self._replace_function(original_factory, construct)
        for module, owner, methods in _ENTRIES:
            holder = getattr(importlib.import_module(module), owner)
            for method in methods:
                self._replace(holder, method, self._entry(getattr(holder, method)))

        def compile_guard(*args: Any, **kwargs: Any) -> Any:
            raise ContractError("torch.compile is unsupported during model preparation; disable compilation")

        self._replace_function(torch.compile, compile_guard)
        try:
            from triton.runtime import JITFunction
        except ImportError:
            pass
        else:
            original_run = JITFunction.run

            def triton_run(kernel: Any, *args: Any, **kwargs: Any) -> Any:
                if not self.depth or self.suppressed:
                    return original_run(kernel, *args, **kwargs)
                origin = {"kind": "unregistered_triton", "name": f"{kernel.__module__}.{kernel.__name__}"}
                return self.call(
                    origin,
                    lambda *a, **kw: original_run(kernel, *a, **kw),
                    args,
                    kwargs,
                    reason="Raw Triton entry needs a public callable and replay contract",
                )

            self._replace(JITFunction, "run", triton_run)
        graph = torch.cuda.CUDAGraph
        begin, end, replay = graph.capture_begin, graph.capture_end, graph.replay

        def capture_begin(instance: Any, *args: Any, **kwargs: Any) -> Any:
            result = begin(instance, *args, **kwargs)
            self.graph_stack.append(Counter())
            return result

        def capture_end(instance: Any, *args: Any, **kwargs: Any) -> Any:
            result = end(instance, *args, **kwargs)
            self.graphs[id(instance)] = self.graph_stack.pop()
            return result

        def graph_replay(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if self.depth and not self.suppressed and id(instance) not in self.graphs:
                self.errors.append("Model replayed a CUDA graph not observed during this preparation")
            result = replay(instance, *args, **kwargs)
            for (op_id, case_id), count in self.graphs.get(id(instance), {}).items():
                case = next(w for w in self.operators[op_id]["workloads"] if w["id"] == case_id)
                case["count"] += count
            return result

        for name, fn in (
            ("capture_begin", capture_begin),
            ("capture_end", capture_end),
            ("replay", graph_replay),
        ):
            self._replace(graph, name, fn)

    def close(self, *, complete: bool) -> None:
        """Restore all hooks and publish a hashed success or failure capture."""
        for holder, name, original, owned in reversed(self.replacements):
            if owned:
                setattr(holder, name, original)
            else:
                delattr(holder, name)
        self.replacements.clear()
        if not self.target:
            self.errors.append("No target policy was bound through make_policy or observe")
        elif not self.entries:
            self.errors.append(
                "No end-to-end inference ran; drive the model through EngineCore, "
                "GenerationBackend or its serving adapter"
            )
        for name, count in sorted(self.bypassed.items()):
            self.errors.append(
                f"{name} read model weights outside an inference entry ({count} calls); "
                "drive the model through EngineCore, GenerationBackend or its serving adapter"
            )
        operators = list(self.operators.values())
        atomic_json(self.root / "operators.json", operators)
        coverage = dict(Counter(op["status"] for op in operators))
        atomic_json(self.root / "coverage.json", {"operators": coverage, "errors": self.errors})
        atomic_json(
            self.root / "manifest.json",
            {
                "schema": SCHEMA,
                "target": self.target,
                "complete": complete,
                "errors": self.errors,
                "repository_revision": _git_revision(),
                "torch_version": str(torch.__version__),
                "precision": self.precision,
                "flags": self.flags,
                "device_type": self.device_type,
                "compile_backend": "none",
                "fixture_budget_bytes": self.fixture_bytes,
                "fixture_bytes": self.used_bytes,
                "files": tree_hashes(self.root),
            },
        )


def run(
    command: list[str], output: Path, model: str, *, fixture_bytes: int = 64 * 1024 * 1024
) -> dict[str, Any]:
    """Execute an existing model preparation script with model-scoped observation."""
    if not command:
        raise ContractError("Supply the preparation script after --")
    session = CaptureSession(model, output, fixture_bytes=fixture_bytes)
    argv, path = sys.argv, sys.path[:]
    complete = False
    try:
        session.install()
        with session.watch():
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
        session.close(complete=complete)
    from .contracts import ModelCapture

    capture = ModelCapture.load(output, model)
    return {"model": model, "identity": capture.identity, "operators": len(capture.operators)}
