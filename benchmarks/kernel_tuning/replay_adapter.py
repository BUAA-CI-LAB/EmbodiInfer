"""Evaluate discovered calls with captured layouts, aliases, and mutable state."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from scripts.kernel_tuning.contracts import ContractError, Precision, TaskPackage, read_json

from .flashinfer_adapter import FlashInferAdapter, compare_outputs


class ReplayAdapter(FlashInferAdapter):
    """Use the frozen production call as oracle, resetting storage before each sample.

    The normal adapter remains suitable for pure, contiguous tensor tasks. This
    adapter handles input views and declared mutations, and rejects changes to
    the output layout/alias contract as well as incorrect values.
    """

    def __init__(self, task: TaskPackage) -> None:
        import torch

        self.replay = read_json(task.root / "replay.json")
        if self.replay["torch_version"] != str(torch.__version__):
            raise ContractError("Capture and evaluator Torch versions differ; recapture on the target")
        super().__init__(task)
        flags = self.replay["flags"]
        torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags["fp16_reduced_reduction"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags["bf16_reduced_reduction"]
        torch.use_deterministic_algorithms(flags["deterministic_algorithms"])
        self._precision_flags = self._flags()
        self.case: dict[str, Any] = {}
        self._rng: Any = None
        if self.cfg.profile:
            raise ContractError("Discovered tasks need a replay-aware NCU adapter before profile=true")
        if self.replay.get("rng") and self.cfg.timing.mode != "eager":
            raise ContractError("RNG replay currently requires eager timing")

    def _load_definition(self, definition: dict[str, Any]) -> Any:
        from .replay_schema import ReplayDefinition

        return ReplayDefinition.model_validate(definition)

    def evaluate(self, solutions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate stateful RNG calls without leaking their state into the caller."""
        from scripts.kernel_tuning.discovery.rng import preserve

        rng = self.replay.get("rng")
        with preserve(rng["devices"]) if rng else nullcontext():
            return super().evaluate(solutions)

    def _build_reference(self, registry: Any) -> Any:
        from flashinfer_bench.data import Solution

        # Build a package, not a standalone source string: backend calls may use
        # snapshotted relative imports. Both reference and baseline are immutable.
        return registry.build(self.definition, Solution.model_validate(self.task.baseline()))

    def _seed(self, seed: int) -> None:
        # RngReplay resets only the declared generators. The ordinary adapter's
        # global seed would also mutate unrelated CUDA devices.
        if not self.replay.get("rng"):
            super()._seed(seed)

    def _inputs(self, workload: Any, seed: int) -> list[Any]:
        from scripts.kernel_tuning.discovery.tensors import materialize

        self._seed(seed)
        self.case = self.replay["cases"][workload.uuid]
        from scripts.kernel_tuning.discovery.rng import RngReplay

        self._rng = (
            RngReplay(self.replay["rng"], self.case, self.task.root, seed) if self.replay.get("rng") else None
        )
        return materialize(self.case["inputs"], self.task.root, seed)

    @staticmethod
    def _clone(inputs: list[Any]) -> list[Any]:
        from scripts.kernel_tuning.discovery.tensors import clone_inputs

        return clone_inputs(inputs)

    @staticmethod
    def _storage_views(inputs: list[Any]) -> list[Any]:
        from scripts.kernel_tuning.discovery.tensors import describe_inputs, extent

        specs = describe_inputs(inputs)
        buffers = {}
        for tensor, spec in zip(inputs, specs):
            group = spec["storage"]
            if group not in buffers:
                size = max(extent(item) for item in specs if item["storage"] == group)
                buffers[group] = tensor.as_strided((size,), (1,), 0)
        return list(buffers.values())

    @classmethod
    def _reset(cls, destination: list[Any], source: list[Any]) -> None:
        for dst, src in zip(cls._storage_views(destination), cls._storage_views(source)):
            dst.copy_(src)

    def _check_outputs(self, inputs: list[Any], outputs: list[Any]) -> None:
        from scripts.kernel_tuning.discovery.tensors import storage_key, tensor_spec

        if len(outputs) != len(self.case["outputs"]):
            raise ContractError("Output count differs from the captured call")
        for tensor, spec, aliases in zip(outputs, self.case["outputs"], self.case["output_aliases"]):
            if tensor_spec(tensor) != spec:
                raise ContractError("Output layout, dtype, or device differs from the captured call")
            actual = [i for i, value in enumerate(inputs) if storage_key(value) == storage_key(tensor)]
            if actual != aliases:
                raise ContractError("Output aliases differ from the captured call")
        groups = [
            [j for j, other in enumerate(outputs) if storage_key(value) == storage_key(other)]
            for value in outputs
        ]
        if groups != self.case["output_groups"]:
            raise ContractError("Output-to-output aliases differ from the captured call")

    def _call(self, runnable: Any, inputs: list[Any]) -> tuple[list[Any], list[Any]]:
        if runnable.metadata.destination_passing_style:
            raise ContractError("Discovered tasks require return-value style")
        outputs = self._outputs(runnable(*inputs))
        self._check_outputs(inputs, outputs)
        return list(inputs), outputs

    def _verify_call(
        self, runnable: Any, reference: Any, inputs: list[Any], seed: int
    ) -> tuple[float, float]:
        from scripts.kernel_tuning.discovery.rng import preserve

        rng = getattr(self, "_rng", None)
        with preserve(rng.devices) if rng else nullcontext():
            return self._verify_transition(runnable, reference, inputs, seed)

    def _verify_transition(
        self, runnable: Any, reference: Any, inputs: list[Any], seed: int
    ) -> tuple[float, float]:
        import torch
        from scripts.kernel_tuning.discovery.rng import snapshot

        rng = getattr(self, "_rng", None)
        expected_inputs = self._clone(inputs)
        self._seed(seed)
        if rng:
            rng.reset()
        _, expected = self._call(reference, expected_inputs)
        if rng:
            expected_rng = snapshot(rng.devices)
            rng.check_reference(expected_rng)
        expected = [value.clone() for value in expected]
        self._seed(seed)
        local = self._clone(inputs)
        if rng:
            rng.reset()
        _, outputs = self._call(runnable, local)
        if rng:
            rng.check(snapshot(rng.devices), expected_rng)
        torch.cuda.synchronize(self.device)
        self._check_flags()
        # Check every byte of backing storage, including aliases and untouched
        # regions. Only declared mutable storage may use the task's tolerance.
        error = self._check_state(local, expected_inputs, inputs)
        current = compare_outputs(outputs, expected, self.cfg.precision)
        return tuple(max(a, b) for a, b in zip(error, current))

    def _check_state(
        self, actual: list[Any], expected: list[Any], original: list[Any]
    ) -> tuple[float, float]:
        from scripts.kernel_tuning.discovery.tensors import describe_inputs

        groups = describe_inputs(original)
        if describe_inputs(actual) != groups or describe_inputs(expected) != groups:
            raise ContractError("Input layout or aliasing changed during the operator call")
        mutable = {groups[i]["storage"] for i in self.replay["mutates"]}
        error = (0.0, 0.0)
        for index, (lhs, rhs, old) in enumerate(
            zip(self._storage_views(actual), self._storage_views(expected), self._storage_views(original))
        ):
            if index in mutable:
                current = compare_outputs([lhs], [rhs], self.cfg.precision)
                error = tuple(max(a, b) for a, b in zip(error, current))
            else:
                compare_outputs([rhs, lhs], [old, old], Precision())
        return error

    def _capture(self, runnable: Any, inputs: list[Any]) -> tuple[Any, list[Any], list[Any]]:
        import torch

        saved = self._clone(inputs)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(self.cfg.timing.warmup):
                self._reset(inputs, saved)
                self._call(runnable, inputs)
            self._reset(inputs, saved)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _, outputs = self._call(runnable, inputs)
        torch.cuda.synchronize(self.device)
        return graph, inputs, outputs

    def _verify_graph(self, runnable: Any, reference: Any, workload: Any, seed: int) -> tuple[float, float]:
        import torch

        original = self._inputs(workload, seed)
        changed = self._inputs(workload, (seed + 1) % 2**32)
        if not any(not torch.equal(a, b) for a, b in zip(original, changed)):
            raise ContractError(
                "Graph validation requires changed valid inputs; supply a domain-specific adapter"
            )
        static = self._clone(original)
        graph, _, outputs = self._capture(runnable, static)
        error = (0.0, 0.0)
        for values in (original, changed, original):
            expected_inputs = self._clone(values)
            self._seed(seed)
            _, expected = self._call(reference, expected_inputs)
            expected = [value.clone() for value in expected]
            self._reset(static, values)
            self._seed(seed)
            graph.replay()
            torch.cuda.synchronize(self.device)
            self._check_flags()
            self._check_outputs(static, outputs)
            state_error = self._check_state(static, expected_inputs, values)
            current = compare_outputs(outputs, expected, self.cfg.precision)
            error = tuple(max(a, b, c) for a, b, c in zip(error, current, state_error))
        return error

    def _time(self, runnable: Any, inputs: list[Any]) -> float:
        from scripts.kernel_tuning.discovery.rng import preserve

        rng = getattr(self, "_rng", None)
        with preserve(rng.devices) if rng else nullcontext():
            return self._time_samples(runnable, inputs)

    def _time_samples(self, runnable: Any, inputs: list[Any]) -> float:
        import torch

        local = self._clone(inputs)
        if self.cfg.timing.mode == "cuda_graph":
            graph, _, _ = self._capture(runnable, local)
            execute = graph.replay
        else:

            def execute() -> Any:
                return runnable(*local)

        if self._flush is None:
            self._flush = torch.empty(64 * 1024 * 1024, dtype=torch.int, device=self.device)
        pairs = list(zip(self._storage_views(local), self._storage_views(inputs)))

        def prepare() -> None:
            for dst, src in pairs:
                dst.copy_(src)
            if self._rng:
                self._rng.reset()
            # Reset copies touch L2 too; evict them before the timed call.
            self._flush.zero_()

        for _ in range(self.cfg.timing.warmup):
            prepare()
            execute()
        count = self.cfg.timing.iterations
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        torch.cuda.synchronize(self.device)
        for start, end in zip(starts, ends):
            prepare()
            start.record()
            execute()
            end.record()
        torch.cuda.synchronize(self.device)
        self._check_flags()
        return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / count
