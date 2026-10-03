"""FlashInfer 0.1.2 building/input generation with strict KDA measurement gates.

The upstream default evaluator permits numerical tolerances and only eager
timing. We reuse its builders, generators, and eager timer and implement the
additional frozen precision, paired timing, and graph contracts here.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernel_tuning.contracts import (  # noqa: E402
    FLASHINFER_VERSION,
    ROLES,
    ContractError,
    Precision,
    TaskPackage,
)


def compare_outputs(actual: list[Any], expected: list[Any], precision: Precision) -> tuple[float, float]:
    """Compare all output tensor bytes or a predeclared elementwise tolerance.

    Byte comparison preserves signed zero and NaN payload distinctions. Shape,
    device and dtype are checked first, so conversion cannot conceal an error.
    """
    import torch

    if len(actual) != len(expected):
        raise ContractError("Output count differs from reference")
    max_abs, max_rel = 0.0, 0.0
    for value, reference in zip(actual, expected):
        if (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise ContractError("Output shape, dtype, or device differs from reference")
        lhs = value.detach().contiguous().reshape(-1)
        rhs = reference.detach().contiguous().reshape(-1)
        if precision.mode == "bit_exact":
            if not torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8)):
                raise ContractError("Output bytes differ from the bit-exact reference")
            continue
        if not torch.is_floating_point(lhs):
            if not torch.equal(lhs, rhs):
                raise ContractError("Non-floating outputs must match exactly")
            continue
        if not torch.isfinite(lhs).all().item() or not torch.isfinite(rhs).all().item():
            raise ContractError("Tolerance comparison requires finite reference and candidate outputs")
        lhs, rhs = lhs.to(torch.float64), rhs.to(torch.float64)
        delta = (lhs - rhs).abs()
        if not torch.all(delta <= precision.atol + precision.rtol * rhs.abs()).item():
            raise ContractError("Output exceeds the fixed atol + rtol * abs(reference) threshold")
        if delta.numel():
            max_abs = max(max_abs, delta.max().item())
            max_rel = max(
                max_rel, (delta / rhs.abs().clamp_min(torch.finfo(torch.float64).tiny)).max().item()
            )
    return max_abs, max_rel


class FlashInferAdapter:
    """Measure pure tensor operators on one selected NVIDIA device.

    Tasks requiring aliasing, mutable input state, non-contiguous fixture creation,
    statistical correctness, or other accelerators provide benchmark.py instead.
    """

    def __init__(self, task: TaskPackage) -> None:
        import torch
        from flashinfer_bench.data import Definition, Trace

        if importlib.metadata.version("flashinfer-bench") != FLASHINFER_VERSION:
            raise ContractError(f"This adapter requires flashinfer-bench=={FLASHINFER_VERSION}")
        if not torch.cuda.is_available() or torch.version.hip or not task.settings.device.startswith("cuda"):
            raise ContractError("Default FlashInfer adapter requires a prepared NVIDIA CUDA environment")
        self.task = task
        self.cfg = task.settings
        if self.cfg.language == "triton":
            import triton  # noqa: F401 -- verify the selected builder before opening an agent
        elif not shutil.which("nvcc"):
            raise ContractError("CUDA C++ tasks require nvcc on PATH before tuning")
        self.device = str(torch.device(self.cfg.device))
        torch.cuda.set_device(self.device)
        self.device = f"cuda:{torch.cuda.current_device()}"
        self.definition = Definition.model_validate(task.definition)
        self.traces = [Trace.model_validate(trace) for trace in task.workloads]
        for trace in self.traces:
            axes = {k: v.value for k, v in self.definition.axes.items() if v.type == "const"}
            axes.update(trace.workload.axes)
            for constraint in self.definition.constraints:
                if not eval(constraint, {"__builtins__": {}, "min": min, "max": max}, axes):
                    raise ContractError(f"Workload {trace.workload.uuid} violates {constraint}")
        torch.set_float32_matmul_precision(self.cfg.precision.matmul_precision)
        torch.backends.cudnn.allow_tf32 = self.cfg.precision.allow_tf32
        self._precision_flags = self._flags()
        self._current_stream = torch.cuda.current_stream(self.device).cuda_stream
        self._nvcc = self._command(["nvcc", "--version"])
        self._driver = self._command(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"]
        )
        self._flush: Any = None

    @staticmethod
    def _command(argv: list[str]) -> str:
        if not shutil.which(argv[0]):
            return "unavailable"
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
            return result.stdout.strip() if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable"

    @staticmethod
    def _flags() -> dict[str, Any]:
        import torch

        return {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "current_device": torch.cuda.current_device(),
            "default_dtype": str(torch.get_default_dtype()),
            "default_device": str(torch.get_default_device()),
            "autocast_enabled": torch.is_autocast_enabled("cuda"),
            "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "fp16_reduced_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }

    def _check_flags(self) -> None:
        import torch

        if self._flags() != self._precision_flags:
            raise ContractError("Solution changed the frozen PyTorch precision settings")
        if torch.cuda.current_stream(self.device).cuda_stream != self._current_stream:
            raise ContractError("Solution changed the current CUDA stream")

    def environment(self) -> dict[str, Any]:
        """Fingerprint the interpreter, packages, GPU, compiler, and precision flags."""
        import torch

        self._check_flags()
        properties = torch.cuda.get_device_properties(self.device)
        versions = {}
        for name in (
            "torch",
            "triton",
            "flashinfer-bench",
            "flashinfer-python",
            "apache-tvm-ffi",
            "numpy",
            "safetensors",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "absent"
        return {
            "identity": {
                "hardware": properties.name,
                "device": self.device,
                "capability": list(torch.cuda.get_device_capability(self.device)),
                "total_memory": properties.total_memory,
                "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
                "hostname": platform.node(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "interpreter": sys.executable,
                "packages": versions,
                "cuda": torch.version.cuda,
                "driver": self._driver,
                "compiler": self._nvcc,
                "precision_flags": self._precision_flags,
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "CUDA_LAUNCH_BLOCKING",
                        "NVIDIA_TF32_OVERRIDE",
                        "CUBLAS_WORKSPACE_CONFIG",
                        "TORCH_CUDA_ARCH_LIST",
                    )
                },
            },
            "conditions": {
                "timing": asdict(self.cfg.timing),
                "precision": asdict(self.cfg.precision),
                "seeds": list(self.cfg.seeds),
                "cache_policy": (
                    "cold L2 per sample; input copy/clone excluded; graph samples are device "
                    "time without host launch latency"
                ),
            },
        }

    @staticmethod
    def _seed(seed: int) -> None:
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _clone(inputs: list[Any]) -> list[Any]:
        import torch

        return [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]

    def _inputs(self, workload: Any, seed: int) -> list[Any]:
        from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

        self._seed(seed)
        safe = (
            load_safetensors(self.definition, workload, self.task.root)
            if any(x.type == "safetensors" for x in workload.inputs.values())
            else {}
        )
        return gen_inputs(self.definition, workload, device=self.device, safe_tensors=safe)

    def _outputs(self, result: Any) -> list[Any]:
        import torch

        names = list(self.definition.outputs)
        if isinstance(result, dict):
            if set(result) != set(names):
                raise ContractError("Output dictionary keys differ from definition.outputs")
            result = [result[name] for name in names]
        values = list(result) if isinstance(result, (tuple, list)) else [result]
        if len(values) != len(names):
            raise ContractError("Output count differs from definition.outputs")
        out = []
        for name, value, dtype in zip(names, values, self.definition.torch_output_dtypes):
            if not isinstance(value, torch.Tensor):
                if self.definition.outputs[name].shape is not None:
                    raise ContractError("Tensor output was replaced by a Python scalar")
                value = torch.tensor(value, dtype=dtype, device=self.device)
            out.append(value)
        return out

    def _call(self, runnable: Any, inputs: list[Any]) -> tuple[list[Any], list[Any]]:
        import torch
        from flashinfer_bench.bench.evaluators.utils import allocate_outputs

        args = list(inputs)
        if runnable.metadata.destination_passing_style:
            outputs = allocate_outputs(self.definition, inputs, self.device)
            # A deterministic poison catches most omitted writes, including zero outputs.
            for output in outputs:
                output.view(torch.uint8).fill_(0xA5)
            args += outputs
            runnable(*args)
        else:
            outputs = self._outputs(runnable(*args))
        shapes = self.definition.get_output_shapes(self.definition.get_axes_values_from_inputs(inputs))
        for value, shape, dtype in zip(outputs, shapes, self.definition.torch_output_dtypes):
            if (
                tuple(value.shape) != tuple(shape or ())
                or value.dtype != dtype
                or str(value.device) != self.device
            ):
                raise ContractError("Output violates the definition's shape, dtype, or device")
        return args, outputs

    def _verify_call(
        self, runnable: Any, reference: Any, inputs: list[Any], seed: int
    ) -> tuple[float, float]:
        import torch

        self._seed(seed)
        _, expected = self._call(reference, self._clone(inputs))
        expected = [v.clone() for v in expected]
        self._seed(seed)
        local = self._clone(inputs)
        _, outputs = self._call(runnable, local)
        torch.cuda.synchronize(self.device)
        self._check_flags()
        # Pure operators may alias an input but must not mutate it.
        compare_outputs(
            [x for x in local if isinstance(x, torch.Tensor)],
            [x for x in inputs if isinstance(x, torch.Tensor)],
            Precision(),
        )
        return compare_outputs(outputs, expected, self.cfg.precision)

    def _capture(self, runnable: Any, inputs: list[Any]) -> tuple[Any, list[Any], list[Any]]:
        import torch
        from flashinfer_bench.bench.evaluators.utils import allocate_outputs

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        args = list(inputs)
        is_dps = runnable.metadata.destination_passing_style
        outputs = allocate_outputs(self.definition, inputs, self.device) if is_dps else []
        if is_dps:
            args += outputs
        with torch.cuda.stream(stream):
            for _ in range(self.cfg.timing.warmup):
                runnable(*args)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = runnable(*args)
        if not is_dps:
            outputs = self._outputs(result)
        torch.cuda.synchronize(self.device)
        return graph, args, outputs

    def _verify_graph(self, runnable: Any, reference: Any, workload: Any, seed: int) -> tuple[float, float]:
        import torch

        original = self._inputs(workload, seed)
        static = self._clone(original)
        graph, _, outputs = self._capture(runnable, static)
        error = (0.0, 0.0)
        # A -> B -> A catches cached results and stale graph buffers. Task-specific
        # valid perturbations of fixed fixtures belong to the custom benchmark.
        changed = self._inputs(workload, (seed + 1) % 2**32)
        if not any(
            isinstance(value, torch.Tensor) and not torch.equal(value, original[index])
            for index, value in enumerate(changed)
        ):
            raise ContractError(
                "Graph validation needs changed valid tensor inputs; supply benchmark.py for fixed fixtures"
            )
        for values in (original, changed, original):
            for dst, src in zip(static, values):
                if isinstance(dst, torch.Tensor):
                    dst.copy_(src)
                elif dst != src:
                    raise ContractError(
                        "Graph scalar arguments must stay fixed; use benchmark.py for custom semantics"
                    )
            self._seed(seed)
            _, expected = self._call(reference, self._clone(values))
            expected = [v.clone() for v in expected]
            self._seed(seed)
            graph.replay()
            torch.cuda.synchronize(self.device)
            self._check_flags()
            compare_outputs(
                [x for x in static if isinstance(x, torch.Tensor)],
                [x for x in values if isinstance(x, torch.Tensor)],
                Precision(),
            )
            current = compare_outputs(outputs, expected, self.cfg.precision)
            error = tuple(max(a, b) for a, b in zip(error, current))
        return error

    def _time(self, runnable: Any, inputs: list[Any]) -> float:
        from flashinfer_bench.bench.timing import time_runnable

        args, _ = self._call(runnable, self._clone(inputs))
        if self.cfg.timing.mode == "eager":
            result = time_runnable(
                runnable, args, self.cfg.timing.warmup, self.cfg.timing.iterations, self.device
            )
        else:
            graph, args, _ = self._capture(runnable, self._clone(inputs))
            result = self._time_graph(graph, args, inputs)
        self._check_flags()
        return float(result)

    def _time_graph(self, graph: Any, args: list[Any], inputs: list[Any]) -> float:
        """Mean device time of graph replays, each from a cold L2 and freshly reset inputs.

        FlashInfer's do_bench synchronizes before every timed call. The idle GPU then
        records the start event before the host has launched the replay, so each
        sample also contains host launch latency and its scheduler jitter: several
        microseconds, comparable to the kernels themselves. Here the L2 flush and
        the input reset are queued ahead of the start event instead. The flush keeps
        the device busy while the replay is enqueued, so the events bracket only the
        replay's execution on the device.
        """
        import torch

        if self._flush is None:
            # The same 256 MiB buffer FlashInfer zeroes to evict inputs from L2.
            self._flush = torch.empty(64 * 1024 * 1024, dtype=torch.int, device=self.device)
        tensors = [(dst, src) for dst, src in zip(args, inputs) if isinstance(dst, torch.Tensor)]

        def prepare() -> None:
            self._flush.zero_()
            for dst, src in tensors:
                dst.copy_(src)

        for _ in range(self.cfg.timing.warmup):
            prepare()
            graph.replay()
        count = self.cfg.timing.iterations
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        torch.cuda.synchronize(self.device)
        for start, end in zip(starts, ends):
            prepare()
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize(self.device)
        return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / count

    def evaluate(self, solutions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Verify all seeds before timing and alternate role order in paired rounds."""
        import torch
        from flashinfer_bench.compile import BuilderRegistry
        from flashinfer_bench.data import Solution

        if set(solutions) != set(ROLES):
            raise ContractError("Expected baseline, incumbent, and candidate solutions")
        registry = BuilderRegistry.get_instance()
        reference = registry.build_reference(self.definition)
        runnables = {
            role: registry.build(self.definition, Solution.model_validate(solution))
            for role, solution in solutions.items()
        }
        self._check_flags()
        checks = []
        with torch.no_grad():
            for trace in self.traces:
                for role in ROLES:
                    max_abs, max_rel = 0.0, 0.0
                    for seed in self.cfg.seeds:
                        inputs = self._inputs(trace.workload, seed)
                        error = self._verify_call(runnables[role], reference, inputs, seed)
                        max_abs, max_rel = max(max_abs, error[0]), max(max_rel, error[1])
                        if self.cfg.timing.mode == "cuda_graph":
                            error = self._verify_graph(runnables[role], reference, trace.workload, seed)
                            max_abs, max_rel = max(max_abs, error[0]), max(max_rel, error[1])
                    checks.append(
                        {
                            "workload": trace.workload.uuid,
                            "role": role,
                            "passed": True,
                            "seeds": list(self.cfg.seeds),
                            "precision": asdict(self.cfg.precision),
                            "max_abs_error": max_abs,
                            "max_rel_error": max_rel,
                            "graph_replay": self.cfg.timing.mode == "cuda_graph",
                        }
                    )
            rounds = []
            for round_index in range(self.cfg.timing.paired_rounds):
                measurements = {}
                for trace in self.traces:
                    row = {role: [] for role in ROLES}
                    for trial in range(self.cfg.timing.trials):
                        seed = self.cfg.seeds[trial % len(self.cfg.seeds)]
                        inputs = self._inputs(trace.workload, seed)
                        order = ROLES if (round_index + trial) % 2 == 0 else tuple(reversed(ROLES))
                        for role in order:
                            self._seed(seed)
                            row[role].append(self._time(runnables[role], inputs))
                    measurements[trace.workload.uuid] = row
                rounds.append(measurements)
        return {"checks": checks, "rounds": rounds}

    def profile(self, solutions: dict[str, dict[str, Any]], output: Path) -> dict[str, Any]:
        """Capture optional NCU evidence in a separate worker, never promotion timing."""
        from scripts.kernel_tuning.artifacts import atomic_json

        if not shutil.which("ncu"):
            raise ContractError("profile: true requires ncu on PATH")
        candidate = output / "candidate.json"
        atomic_json(candidate, solutions["candidate"])
        command = [
            "ncu",
            "--target-processes",
            "all",
            "--set",
            "basic",
            "--force-overwrite",
            "--export",
            str(output / "ncu"),
            sys.executable,
            str(Path(__file__).resolve()),
            "--profile-once",
            str(self.task.root),
            str(candidate),
        ]
        with (output / "ncu.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=self.cfg.search.evaluation_timeout_seconds,
            )
        return {
            "command": command,
            "report": str(output / "ncu.ncu-rep"),
            "workload": self.traces[0].workload.uuid,
        }


if __name__ == "__main__":
    # NCU child: one representative workload, excluded from optimization metrics.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.data import Solution
    from scripts.kernel_tuning.contracts import read_json

    if len(sys.argv) != 4 or sys.argv[1] != "--profile-once":
        raise SystemExit("Internal NCU worker; use tuning.yaml profile: true")
    adapter = FlashInferAdapter(TaskPackage.load(Path(sys.argv[2])))
    runnable = BuilderRegistry.get_instance().build(
        adapter.definition, Solution.model_validate(read_json(Path(sys.argv[3])))
    )
    adapter._call(runnable, adapter._inputs(adapter.traces[0].workload, adapter.cfg.seeds[0]))
    import torch

    torch.cuda.synchronize(adapter.device)
