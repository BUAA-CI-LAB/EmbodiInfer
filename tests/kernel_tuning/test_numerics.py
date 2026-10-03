"""Byte-level correctness tests use CPU Torch; no kernel tuning is executed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.kernel_tuning.artifacts import REPOSITORY  # noqa: E402
from scripts.kernel_tuning.contracts import ContractError, Precision  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "kernel_test_adapter", REPOSITORY / "benchmarks/kernel_tuning/flashinfer_adapter.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
compare = adapter.compare_outputs


def test_bit_exact_distinguishes_signed_zero_and_nan_payloads() -> None:
    positive = torch.tensor([0.0])
    negative = torch.tensor([-0.0])
    assert torch.equal(positive, negative)
    with pytest.raises(ContractError, match="bytes"):
        compare([positive], [negative], Precision())
    first_nan = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    second_nan = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
    assert compare([first_nan], [first_nan.clone()], Precision()) == (0, 0)
    with pytest.raises(ContractError, match="bytes"):
        compare([first_nan], [second_nan], Precision())


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([torch.ones(2)], [torch.ones(3)]),
        ([torch.ones(2, dtype=torch.float16)], [torch.ones(2)]),
        ([torch.ones(2)], [torch.ones(2), torch.ones(2)]),
    ],
)
def test_shape_dtype_and_output_count_are_strict(actual, expected) -> None:
    with pytest.raises(ContractError):
        compare(actual, expected, Precision())


def test_tolerance_uses_fixed_combined_atol_rtol_threshold() -> None:
    precision = Precision(mode="tolerance", atol=0.1, rtol=0.1, reason="test contract")
    assert compare([torch.tensor([1.19])], [torch.tensor([1.0])], precision)[0] < 0.2
    with pytest.raises(ContractError, match="threshold"):
        compare([torch.tensor([1.21])], [torch.tensor([1.0])], precision)
    with pytest.raises(ContractError, match="finite"):
        compare([torch.tensor([float("nan")])], [torch.tensor([0.0])], precision)
    with pytest.raises(ContractError, match="exactly"):
        compare([torch.tensor([1], dtype=torch.int32)], [torch.tensor([2], dtype=torch.int32)], precision)


def test_bf16_intermediate_rounding_is_observable() -> None:
    generator = torch.Generator().manual_seed(0)
    residual, update, gate = [torch.randn(4096, generator=generator).to(torch.bfloat16) for _ in range(3)]
    reference = residual + (update * gate).to(torch.bfloat16)
    contracted = (residual.float() + update.float() * gate.float()).to(torch.bfloat16)
    assert not torch.equal(reference, contracted)
    with pytest.raises(ContractError, match="bytes"):
        compare([contracted], [reference], Precision())


@pytest.mark.gpu
@pytest.mark.parametrize("language", ["triton", "cuda"])
def test_default_adapter_builds_multi_file_solution(tmp_path: Path, language: str) -> None:
    """Optional target-host smoke test for both builders, with tiny declared cases."""
    import shutil

    import yaml
    from scripts.kernel_tuning.artifacts import atomic_json
    from scripts.kernel_tuning.contracts import TaskPackage, validate_measurement

    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    pytest.importorskip("flashinfer_bench")
    if language == "triton":
        pytest.importorskip("triton")
    elif shutil.which("nvcc") is None:
        pytest.skip("requires the CUDA toolkit")
    task = tmp_path / "task"
    shutil.copytree(REPOSITORY / "benchmarks/kernel_tuning/tasks/gated_residual", task)
    definition = {
        "name": "vector_add_test",
        "op_type": "elementwise",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}, "y": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x, y):\n    return x + y\n",
    }
    atomic_json(task / "definition.json", definition)
    (task / "baseline.py").write_text(definition["reference"])
    (task / "workloads.jsonl").write_text(
        json.dumps(
            {
                "definition": "vector_add_test",
                "workload": {"uuid": "small-tail", "axes": {"N": 257}, "inputs": {}},
                "solution": None,
                "evaluation": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = yaml.safe_load((task / "tuning.yaml").read_text())
    config["language"] = language
    config["timing"].update(mode="cuda_graph", warmup=2, iterations=2, trials=1)
    (task / "tuning.yaml").write_text(yaml.safe_dump(config))
    if language == "triton":
        sources = [
            {"path": "helper.py", "content": "BLOCK = 256\n"},
            {
                "path": "kernel.py",
                "content": """
import torch
import triton
import triton.language as tl
from .helper import BLOCK

@triton.jit
def add(X, Y, Z, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Z + i, tl.load(X + i, i < N, 0) + tl.load(Y + i, i < N, 0), i < N)

def run(x, y):
    out = torch.empty_like(x)
    add[(triton.cdiv(x.numel(), BLOCK),)](x, y, out, x.numel(), BLOCK)
    return out
""",
            },
        ]
        entry = "kernel.py::run"
    else:
        sources = [
            {"path": "helper.cuh", "content": "constexpr int BLOCK = 256;\n"},
            {
                "path": "kernel.cu",
                "content": """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "helper.cuh"
__global__ void add(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}
torch::Tensor run(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    add<<<(x.numel()+BLOCK-1)/BLOCK, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), x.numel());
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
""",
            },
        ]
        entry = "kernel.cu::run"
    package = TaskPackage.load(task)
    solution = {
        "name": "add_test",
        "definition": definition["name"],
        "author": "test",
        "spec": {
            "language": language,
            "target_hardware": ["cuda"],
            "entry_point": entry,
            "destination_passing_style": False,
            **({"binding": "torch"} if language == "cuda" else {}),
        },
        "sources": sources,
    }
    measured = adapter.FlashInferAdapter(package).evaluate(
        {"baseline": package.baseline(), "incumbent": package.baseline(), "candidate": solution}
    )
    validate_measurement(package, measured)
