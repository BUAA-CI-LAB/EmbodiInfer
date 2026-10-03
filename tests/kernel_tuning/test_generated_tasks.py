"""Generated task packages: static structure on CPU, baseline contracts on a GPU host."""

from __future__ import annotations

import importlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.kernel_tuning.contracts import ContractError, TaskPackage, read_json
from scripts.kernel_tuning.generate import (
    MARKER,
    parse_overrides,
    render,
    safetensors_bytes,
    stale_sources,
    tuning_settings,
    vendor_sources,
)
from scripts.kernel_tuning.operators import OPERATORS, catalog, select


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, TaskPackage]:
    """Every catalog operator, rendered once with default settings."""
    root = tmp_path_factory.mktemp("generated")
    return {operator.name: render(operator, root / operator.name) for operator in OPERATORS}


def test_every_catalog_operator_renders_a_valid_task(generated: dict[str, TaskPackage]) -> None:
    for name, task in generated.items():
        operator = catalog()[name]
        assert task.workload_ids == tuple(work.uuid for work in operator.workloads)
        assert task.settings.precision.mode == operator.precision["mode"]
        baseline = task.baseline()
        paths = {source["path"] for source in baseline["sources"]}
        assert "baseline.py" in paths
        assert f"vendor/{operator.source}" in paths
        assert not stale_sources(task.root)


def test_vendored_kernels_are_verbatim_and_import_closed(generated: dict[str, TaskPackage]) -> None:
    from scripts.kernel_tuning.artifacts import REPOSITORY

    for task in generated.values():
        for path in read_json(task.root / MARKER)["sources"]:
            original = REPOSITORY / Path(path).relative_to("vendor")
            assert (task.root / path).read_bytes() == original.read_bytes().replace(b"\r\n", b"\n")
            for line in (task.root / path).read_text(encoding="utf-8").splitlines():
                assert not line.lstrip().startswith(("from embodiinfer", "import embodiinfer"))
    # packed_rope imports .capability; the closure must carry it.
    assert "vendor/embodiinfer/backend/triton/capability.py" in generated["rotate_half_rope"].hashes


def test_absolute_repository_imports_are_rewritten(monkeypatch, tmp_path: Path) -> None:
    from scripts.kernel_tuning import generate

    package = tmp_path / "embodiinfer/backend/triton"
    package.mkdir(parents=True)
    (package / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "kernel.py").write_text(
        "from embodiinfer.backend.triton.helper import VALUE\nfrom . import helper\n", encoding="utf-8"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "bad.py").write_text("from embodiinfer.backend.triton import helper\n", encoding="utf-8")
    monkeypatch.setattr(generate, "REPOSITORY", tmp_path)
    vendored = vendor_sources("embodiinfer.backend.triton.kernel")
    assert vendored["vendor/embodiinfer/backend/triton/kernel.py"][0].startswith("from .helper import VALUE")
    assert "vendor/embodiinfer/backend/triton/helper.py" in vendored
    with pytest.raises(ContractError, match="package"):
        vendor_sources("embodiinfer.backend.triton.bad")


def test_fixed_integer_inputs_round_trip_through_safetensors() -> None:
    encoded = safetensors_bytes({"offsets": ("int32", (0, 64, 128))})
    (size,) = struct.unpack("<Q", encoded[:8])
    assert size % 8 == 0
    header = json.loads(encoded[8 : 8 + size])
    assert header["offsets"] == {"dtype": "I32", "shape": [3], "data_offsets": [0, 12]}
    assert struct.unpack("<3i", encoded[8 + size :]) == (0, 64, 128)


def test_overrides_are_validated_and_regeneration_is_explicit(tmp_path: Path) -> None:
    operator = catalog()["gated_residual"]
    overrides = parse_overrides(["evaluator_python=/opt/thor/bin/python", "search.max_candidates=4"])
    task = render(operator, tmp_path / "task", overrides=overrides)
    assert task.settings.evaluator_python == "/opt/thor/bin/python"
    assert task.settings.search.max_candidates == 4
    with pytest.raises(ContractError, match="--force"):
        render(operator, tmp_path / "task")
    assert render(operator, tmp_path / "task", overrides=overrides, force=True).identity == task.identity
    with pytest.raises(ContractError):
        tuning_settings(operator, parse_overrides(["search.unknown=1"]))
    with pytest.raises(ContractError, match="Unknown tuning.yaml section"):
        tuning_settings(operator, parse_overrides(["nope.key=1"]))
    (tmp_path / "manual").mkdir()
    (tmp_path / "manual/README.md").write_text("hand-written", encoding="utf-8")
    with pytest.raises(ContractError, match="not generated"):
        render(operator, tmp_path / "manual", force=True)


def test_operator_selection_by_name_and_model() -> None:
    assert {op.name for op in select(models=["pi05"])} == {
        "ada_rms_norm",
        "gated_residual",
        "gated_gelu",
        "rotate_qk",
        "split_kv_attention",
    }
    assert [op.name for op in select(["swiglu"])] == ["swiglu"]
    with pytest.raises(ValueError, match="Unknown operators"):
        select(["nope"])
    with pytest.raises(ValueError, match="Unknown models"):
        select(models=["nope"])


# ---- Target-host checks: the production kernel must satisfy its own task contract ----


def _load_baseline(task: TaskPackage, root: Path) -> Any:
    """Import the baseline the way FlashInfer's Python builder does: as a namespace package."""
    package = f"kernel_task_{task.identity[:12]}"
    for source in task.baseline()["sources"]:
        path = root / package / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source["content"], encoding="utf-8")
    sys.path.insert(0, str(root))
    return importlib.import_module(f"{package}.baseline").run


def _inputs(task: TaskPackage, workload: dict[str, Any], seed: int, device: str) -> list[Any]:
    """Mirror FlashInfer's generator: torch.randn floats, scalars, and safetensors data."""
    import torch

    torch.manual_seed(seed)
    definition = task.definition
    axes = {k: v["value"] for k, v in definition["axes"].items() if v["type"] == "const"}
    axes.update(workload["axes"])
    values = []
    for name, spec in definition["inputs"].items():
        source = workload["inputs"][name]
        dtype = getattr(torch, spec["dtype"])
        if source["type"] == "scalar":
            values.append(source["value"])
        elif source["type"] == "safetensors":
            data = (task.root / source["path"]).read_bytes()
            (size,) = struct.unpack("<Q", data[:8])
            entry = json.loads(data[8 : 8 + size])[source["tensor_key"]]
            start, end = entry["data_offsets"]
            payload = torch.frombuffer(bytearray(data[8 + size + start : 8 + size + end]), dtype=dtype)
            values.append(payload.reshape(entry["shape"]).to(device))
        else:
            values.append(torch.randn([axes[a] for a in spec["shape"]], dtype=dtype, device=device))
    return values


def _gpu_ready() -> str | None:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return "requires Torch and Triton"
    return None if torch.cuda.is_available() else "requires CUDA"


@pytest.mark.gpu
@pytest.mark.parametrize("name", [op.name for op in OPERATORS])
def test_production_baseline_meets_generated_contract(name: str, tmp_path: Path) -> None:
    """Calibrates each contract: a production kernel that fails its own task is a catalog bug."""
    if reason := _gpu_ready():
        pytest.skip(reason)
    import torch
    from scripts.kernel_tuning.artifacts import REPOSITORY

    spec = importlib.util.spec_from_file_location(
        "generated_task_adapter", REPOSITORY / "benchmarks/kernel_tuning/flashinfer_adapter.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    compare = adapter.compare_outputs

    task = render(catalog()[name], tmp_path / "task")
    baseline = _load_baseline(task, tmp_path / "build")
    reference_scope: dict[str, Any] = {}
    exec(compile(task.definition["reference"], "reference", "exec"), reference_scope)
    reference = reference_scope["run"]

    def outputs(value: Any) -> list[Any]:
        return list(value) if isinstance(value, (tuple, list)) else [value]

    device = "cuda"
    precision = task.settings.precision
    for trace in task.workloads:
        work = trace["workload"]
        exact, margin = True, 0.0
        for seed in task.settings.seeds:
            inputs = _inputs(task, work, seed, device)
            expected = [
                v.clone()
                for v in outputs(reference(*[x.clone() if hasattr(x, "clone") else x for x in inputs]))
            ]
            actual = outputs(baseline(*inputs))
            torch.cuda.synchronize()
            compare(actual, expected, precision)
            for value, target in zip(actual, expected):
                exact = exact and torch.equal(value.view(torch.uint8), target.view(torch.uint8))
                if precision.mode == "tolerance" and value.is_floating_point():
                    bound = precision.atol + precision.rtol * target.double().abs()
                    margin = max(margin, ((value.double() - target.double()).abs() / bound).max().item())
        # Report how much of the bound the production kernel uses, to calibrate the catalog.
        print(f"{name}/{work['uuid']}: bit_exact={exact} bound_used={margin:.2f}")
        # The evaluator replays a captured graph with changed inputs; the baseline must allow it.
        static = _inputs(task, work, task.settings.seeds[0], device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            baseline(*static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = outputs(baseline(*static))
        changed = _inputs(task, work, task.settings.seeds[0] + 1, device)
        for dst, src in zip(static, changed):
            if isinstance(dst, torch.Tensor):
                dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        compare(captured, outputs(reference(*changed)), precision)
