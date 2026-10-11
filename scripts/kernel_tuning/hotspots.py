"""Profile end-to-end inference and rank where GPU time goes.

``python -m scripts.kernel_tuning profile --output DIR -- DRIVER`` runs an
unmodified inference driver (for example ``-m scripts.kernel_tuning.discovery.drive``)
in the model runtime. Each top-level call of a model-agnostic inference entry
(``EngineCore.execute``, ``GenerationBackend`` rollouts, serving ``infer``) is one
step; the first ``warmup`` steps are skipped and the next ``steps`` are profiled.

Every GPU kernel is attributed through its launch (CUDA correlation id) to the
profiling ranges enclosing that launch:

* ``stage:<method>``: policy and decoder protocol methods (``encode_prefix``,
  ``denoise_step``, ``integrate``, ...),
* ``op:<name>``: operator-registry calls (:mod:`embodiinfer.layers.ops`),
* ``attention:<class>``: attention-backend calls,
* ``module:<path>``: ``nn.Module`` calls, with layer indices folded to ``*``.

A kernel's region is its stage ranges plus the innermost operator, attention or
module range enclosing its launch.

Kernels replayed from a CUDA graph keep the ranges active at replay time, so
they are attributed to their stage only; profile a run without CUDA graphs or
compilation for operator-level detail. Ranges are skipped inside
``torch.compile`` tracing, so compiled regions also report at stage level.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import re
import runpy
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError

RANGE = "kt/"
STEP = f"{RANGE}step:"
_ENTRIES = (
    ("embodiinfer.engine.core", "EngineCore", ("execute", "execute_pipelined")),
    (
        "embodiinfer.engine.rollout.generation_backend",
        "GenerationBackend",
        ("generate", "generate_with_logprob", "sample_group", "best_of_n"),
    ),
)
_POLICY_STAGES = ("collate", "encode_prefix", "denoise_step", "finalize_actions")
_DECODER_STAGES = ("decode", "produce_chunk", "integrate", "sample_with_logprob")
_LAYER_INDEX = re.compile(r"(?<=\.)\d+(?=\.|$)")
_KERNEL_ACTIVITIES = {"kernel", "gpu_memset", "gpu_memcpy"}
_LAUNCH_ACTIVITIES = {"cuda_runtime", "cuda_driver"}


@dataclass(frozen=True)
class Range:
    """A profiling range on one host thread, in nanoseconds."""

    name: str
    thread: int
    start: int
    end: int


@dataclass(frozen=True)
class Launch:
    """A host-side launch (kernel, memset/memcpy or graph) and its correlation id."""

    name: str
    thread: int
    start: int
    correlation: int


@dataclass(frozen=True)
class DeviceKernel:
    """A device activity with the correlation id of its launch."""

    name: str
    start: int
    duration_ns: int
    correlation: int


@dataclass(frozen=True)
class Kernel:
    """One profiled kernel attributed to a step and its enclosing ranges."""

    step: int | None
    path: tuple[str, ...]
    name: str
    duration_us: float
    graph: bool


def fold_layers(name: str) -> str:
    """Fold numeric path components so repeated layers aggregate (``layers.3`` -> ``layers.*``)."""
    return _LAYER_INDEX.sub("*", name)


def _enclosing(ranges: Iterable[Range], launches: Iterable[Launch]) -> dict[int, tuple[Range, ...]]:
    """Map each launch's correlation id to its enclosing ranges, outermost first.

    Ranges on one thread nest (``record_function`` scopes), so one sweep per
    thread with a stack of open ranges finds them in O(n log n).
    """
    range_lists: dict[int, list[Range]] = defaultdict(list)
    launch_lists: dict[int, list[Launch]] = defaultdict(list)
    for item in ranges:
        range_lists[item.thread].append(item)
    for launch in launches:
        launch_lists[launch.thread].append(launch)
    result: dict[int, tuple[Range, ...]] = {}
    for thread, thread_launches in launch_lists.items():
        thread_ranges = sorted(range_lists.get(thread, []), key=lambda item: (item.start, -item.end))
        stack: list[Range] = []
        index = 0
        for launch in sorted(thread_launches, key=lambda item: item.start):
            while index < len(thread_ranges) and thread_ranges[index].start <= launch.start:
                while stack and stack[-1].end < thread_ranges[index].start:
                    stack.pop()
                stack.append(thread_ranges[index])
                index += 1
            while stack and stack[-1].end < launch.start:
                stack.pop()
            result[launch.correlation] = tuple(stack)
    return result


def attribute(
    ranges: Iterable[Range], launches: Iterable[Launch], kernels: Iterable[DeviceKernel]
) -> list[Kernel]:
    """Attribute each device kernel to the ranges enclosing its host launch."""
    kernels = list(kernels)
    wanted = {kernel.correlation for kernel in kernels}
    launch_by_id = {launch.correlation: launch for launch in launches if launch.correlation in wanted}
    enclosing_by_id = _enclosing(ranges, launch_by_id.values())
    result = []
    for kernel in kernels:
        launch = launch_by_id.get(kernel.correlation)
        if launch is None:
            result.append(Kernel(None, ("<unlinked>",), kernel.name, kernel.duration_ns / 1000, False))
            continue
        step = None
        stages: list[str] = []
        innermost = None
        for item in enclosing_by_id[kernel.correlation]:
            name = item.name[len(RANGE) :]
            if item.name.startswith(STEP):
                step = int(name[len(STEP) - len(RANGE) :])
            elif name.startswith("stage:"):
                if not stages or stages[-1] != name:
                    stages.append(name)
            else:
                innermost = fold_layers(name)
        path = (*stages, innermost) if innermost else tuple(stages)
        result.append(Kernel(step, path, kernel.name, kernel.duration_ns / 1000, "Graph" in launch.name))
    return result


def summarize(kernels: Iterable[Kernel], steps: int, *, small_us: float = 5.0) -> dict[str, Any]:
    """Aggregate attributed kernels per step into regions ranked by GPU time."""
    measured = [kernel for kernel in kernels if kernel.step is not None]
    if steps <= 0 or not measured:
        raise ContractError("No kernels were profiled inside the measured steps")
    total = sum(kernel.duration_us for kernel in measured)
    regions: dict[tuple[str, ...], dict[str, Any]] = {}
    for kernel in measured:
        region = regions.setdefault(
            kernel.path,
            {"us": 0.0, "kernels": 0, "small": 0, "graph": 0, "names": defaultdict(lambda: [0, 0.0])},
        )
        region["us"] += kernel.duration_us
        region["kernels"] += 1
        region["small"] += kernel.duration_us < small_us
        region["graph"] += kernel.graph
        entry = region["names"][kernel.name]
        entry[0] += 1
        entry[1] += kernel.duration_us
    ranked = []
    for path, region in sorted(regions.items(), key=lambda item: -item[1]["us"]):
        names = sorted(region["names"].items(), key=lambda item: -item[1][1])
        ranked.append(
            {
                "region": "/".join(path) or "<outside ranges>",
                "us_per_step": region["us"] / steps,
                "share": region["us"] / total,
                "kernels_per_step": region["kernels"] / steps,
                "mean_kernel_us": region["us"] / region["kernels"],
                "small_kernel_fraction": region["small"] / region["kernels"],
                "graph_replayed_fraction": region["graph"] / region["kernels"],
                "top_kernels": [
                    {"name": name, "calls_per_step": calls / steps, "us_per_step": us / steps}
                    for name, (calls, us) in names[:5]
                ],
            }
        )
    return {
        "steps": steps,
        "kernel_us_per_step": total / steps,
        "kernels_per_step": len(measured) / steps,
        "small_kernel_us": small_us,
        "regions": ranked,
    }


def render(summary: dict[str, Any], *, limit: int = 30) -> str:
    """Markdown table of the highest-cost regions."""
    lines = [
        f"# Hotspots ({summary['steps']} steps)",
        "",
        f"GPU kernel time per step: {summary['kernel_us_per_step'] / 1000:.2f} ms in "
        f"{summary['kernels_per_step']:.0f} kernels. Small kernels are under {summary['small_kernel_us']:g} us.",
        "",
        "| Region | ms/step | Share | Kernels/step | Mean us | Small | Graph | Top kernel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for region in summary["regions"][:limit]:
        top = region["top_kernels"][0]["name"] if region["top_kernels"] else ""
        lines.append(
            f"| `{region['region']}` | {region['us_per_step'] / 1000:.3f} | {region['share']:.1%} "
            f"| {region['kernels_per_step']:.1f} | {region['mean_kernel_us']:.1f} "
            f"| {region['small_kernel_fraction']:.0%} | {region['graph_replayed_fraction']:.0%} | `{top[:60]}` |"
        )
    return "\n".join(lines) + "\n"


def _compiling() -> bool:
    import torch

    return torch.compiler.is_compiling()


def _ranged(function: Callable[..., Any], name: str) -> Callable[..., Any]:
    from torch.profiler import record_function

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _compiling():
            return function(*args, **kwargs)
        with record_function(RANGE + name):
            return function(*args, **kwargs)

    return wrapper


class ProfileSession:
    """Count inference steps, profile a window of them and label their GPU work."""

    def __init__(self, warmup: int, steps: int, *, modules: bool = True) -> None:
        if warmup < 0 or steps <= 0:
            raise ContractError("profile needs warmup >= 0 and steps > 0")
        self.warmup, self.steps, self.modules = warmup, steps, modules
        self.started = 0
        self.depth = threading.local()
        self.lock = threading.Lock()
        self.profiler: Any = None
        self.trace: Any = None
        self._restore: list[Callable[[], None]] = []
        self._module_names: dict[int, str] = {}
        self._module_ranges = threading.local()

    def _replace(self, owner: Any, name: str, value: Any) -> None:
        """Set ``owner.name`` (class, module or instance) and remember how to undo it."""
        had = name in vars(owner)
        original = vars(owner).get(name)
        setattr(owner, name, value)

        def restore() -> None:
            if had:
                setattr(owner, name, original)
            else:
                delattr(owner, name)

        self._restore.append(restore)

    def _step(self, function: Callable[..., Any]) -> Callable[..., Any]:
        session = self

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            depth = getattr(session.depth, "value", 0)
            if depth:
                return function(*args, **kwargs)
            with session.lock:
                index = session.started
                session.started += 1
            if index == session.warmup:
                session._start()
            measured = session.warmup <= index < session.warmup + session.steps
            session.depth.value = 1
            try:
                if not measured:
                    return function(*args, **kwargs)
                from torch.profiler import record_function

                with record_function(f"{STEP}{index - session.warmup}"):
                    return function(*args, **kwargs)
            finally:
                session.depth.value = 0
                if index == session.warmup + session.steps - 1:
                    session._stop()

        return wrapper

    def _start(self) -> None:
        import torch
        from torch.profiler import ProfilerActivity, profile

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self.profiler = profile(activities=activities)
        self.profiler.start()

    def _stop(self) -> None:
        import torch

        if self.profiler is None or self.trace is not None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.profiler.stop()
        self.trace = self.profiler

    def _bind_policy(self, policy: Any) -> None:
        import torch

        if isinstance(policy, torch.nn.Module):
            self._module_names.update({id(module): name for name, module in policy.named_modules() if name})
        for name in _POLICY_STAGES:
            method = getattr(policy, name, None)
            if callable(method) and not isinstance(method, torch.nn.Module):
                self._replace(policy, name, _ranged(method, f"stage:{name}"))
        decoder = getattr(policy, "decoder", None)
        for name in _DECODER_STAGES:
            method = getattr(decoder, name, None)
            if callable(method):
                self._replace(decoder, name, _ranged(method, f"stage:{name}"))

    def install(self) -> None:
        """Patch inference entries, protocol stages, operator and attention resolution."""
        for module_name, owner, methods in _ENTRIES:
            holder = getattr(importlib.import_module(module_name), owner)
            for method in methods:
                if hasattr(holder, method):
                    self._replace(holder, method, self._step(getattr(holder, method)))
        core = importlib.import_module("embodiinfer.engine.core").EngineCore
        original_init = core.__init__
        session = self

        @functools.wraps(original_init)
        def init(engine: Any, policy: Any, *args: Any, **kwargs: Any) -> None:
            session._bind_policy(policy)
            original_init(engine, policy, *args, **kwargs)

        self._replace(core, "__init__", init)

        from embodiinfer.layers.ops import KernelSelection

        original_resolve = KernelSelection.resolve

        @functools.wraps(original_resolve)
        def resolve(selection: Any, op: str, *args: Any, **kwargs: Any) -> Any:
            return _ranged(original_resolve(selection, op, *args, **kwargs), f"op:{op}")

        self._replace(KernelSelection, "resolve", resolve)

        attention = importlib.import_module("embodiinfer.layers.attention")
        original_backend = attention._resolve_attention_backend

        def resolve_backend(name: str) -> Any:
            cls = original_backend(name)
            for method in ("attend", "attend_split"):
                function = cls.__dict__.get(method)
                if function is not None and not getattr(function, "_kt_ranged", False):
                    ranged = _ranged(function, f"attention:{cls.__name__}")
                    ranged._kt_ranged = True  # type: ignore[attr-defined]
                    self._replace(cls, method, ranged)
            return cls

        self._replace(attention, "_resolve_attention_backend", resolve_backend)
        if self.modules:
            self._install_module_hooks()

    def _install_module_hooks(self) -> None:
        import torch
        from torch.nn.modules import module as module_hooks

        def before(module: Any, _inputs: Any) -> None:
            if _compiling() or self.profiler is None or self.trace is not None:
                return
            name = self._module_names.get(id(module))
            if name is None:
                return
            handle = torch.ops.profiler._record_function_enter_new(f"{RANGE}module:{name}", None)
            stack = getattr(self._module_ranges, "stack", None)
            if stack is None:
                stack = self._module_ranges.stack = []
            stack.append((id(module), handle))

        def after(module: Any, _inputs: Any, _output: Any) -> None:
            if _compiling():
                return
            stack = getattr(self._module_ranges, "stack", None)
            if stack and stack[-1][0] == id(module):
                torch.ops.profiler._record_function_exit._RecordFunction(stack.pop()[1])

        pre = module_hooks.register_module_forward_pre_hook(before)
        post = module_hooks.register_module_forward_hook(after)
        self._restore.append(pre.remove)
        self._restore.append(post.remove)

    def close(self) -> None:
        """Undo every patch, stopping the profiler if the driver ended early."""
        self._stop()
        while self._restore:
            self._restore.pop()()

    def kernels(self) -> list[Kernel]:
        """Attributed kernels of the profiled window."""
        if self.trace is None or self.started < self.warmup + self.steps:
            raise ContractError(
                f"Only {self.started} inference steps ran; profiling needs warmup {self.warmup} + {self.steps}"
            )
        ranges, launches, devices = [], [], []
        for event in self.trace.profiler.kineto_results.events():
            activity = event.activity_type()
            name = event.name()
            if activity == "user_annotation" and name.startswith(RANGE):
                start = event.start_ns()
                ranges.append(Range(name, event.start_thread_id(), start, start + event.duration_ns()))
            elif activity in _LAUNCH_ACTIVITIES:
                launches.append(
                    Launch(name, event.start_thread_id(), event.start_ns(), event.correlation_id())
                )
            elif activity in _KERNEL_ACTIVITIES:
                devices.append(
                    DeviceKernel(name, event.start_ns(), event.duration_ns(), event.correlation_id())
                )
        return attribute(ranges, launches, devices)


def _environment() -> dict[str, Any]:
    import torch

    environment: dict[str, Any] = {"torch": torch.__version__, "python": sys.version.split()[0]}
    try:
        import triton

        environment["triton"] = triton.__version__
    except ImportError:
        environment["triton"] = None
    if torch.cuda.is_available():
        environment["device_name"] = torch.cuda.get_device_name()
        environment["capability"] = list(torch.cuda.get_device_capability())
    return environment


@contextlib.contextmanager
def _driver_argv(command: list[str]) -> Iterator[None]:
    argv, path = sys.argv, sys.path[:]
    try:
        yield
    finally:
        sys.argv, sys.path = argv, path


def run(
    command: list[str],
    output: Path,
    *,
    warmup: int = 10,
    steps: int = 20,
    modules: bool = True,
    trace: bool = False,
) -> dict[str, Any]:
    """Run an inference driver, profile ``steps`` steps after ``warmup`` and write the report."""
    if not command:
        raise ContractError("Supply the inference driver after --")
    if output.exists():
        raise ContractError(f"{output} already exists; choose a new profile directory")
    session = ProfileSession(warmup, steps, modules=modules)
    session.install()
    try:
        with _driver_argv(command):
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
        session.close()
    summary = summarize(session.kernels(), steps)
    summary.update(
        {"command": command, "warmup": warmup, "environment": _environment(), "module_ranges": modules}
    )
    output.mkdir(parents=True)
    (output / "hotspots.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "hotspots.md").write_text(render(summary), encoding="utf-8")
    if trace:
        session.trace.export_chrome_trace(str(output / "trace.json"))
    return summary
