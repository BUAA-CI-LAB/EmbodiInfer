"""CPU measurement-contract tests and optional CUDA parity/capture checks."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from embodiinfer.backend.split_kv_plan import SUPPORTED_SPLIT_COUNTS
from embodiinfer.backend.triton.capability import triton_capability
from embodiinfer.layers import get_split_kv_attention_backend

_PATH = Path(__file__).resolve().parents[1] / "benchmarks/attention_split.py"
_SPEC = importlib.util.spec_from_file_location("attention_split_benchmark", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_timing_summary_preserves_raw_samples_and_interpolates() -> None:
    samples = [3.0, 1.0, 2.0, 4.0]
    summary = benchmark.timing_summary(samples)
    assert summary["raw_ms"] == samples
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] == pytest.approx(3.85)


@pytest.mark.parametrize("samples", [[], [-1.0], [math.nan], [math.inf]])
def test_timing_summary_rejects_invalid_samples(samples: list[float]) -> None:
    with pytest.raises(ValueError, match="timing"):
        benchmark.timing_summary(samples)


@pytest.mark.parametrize("flags", [[], ["--atol", "0.1"], ["--rtol", "0.1"]])
def test_tolerances_are_required_before_measurement(flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args([*flags, "--output", "unused.json"])


@pytest.mark.parametrize(
    "flags",
    [
        ["--atol", "nan"],
        ["--rtol", "inf"],
        ["--atol", "-1"],
        ["--splits", "3"],
        ["--batch", "0"],
        ["--queries", "0"],
        ["--cached", "-1"],
        ["--kv-heads", "3"],
        ["--warmup", "0"],
        ["--iterations", "0"],
        ["--padding-fraction", "nan"],
        ["--device", "-1"],
    ],
)
def test_invalid_benchmark_contract_is_rejected(flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--atol", "0", "--rtol", "0", "--output", "unused.json", *flags])


def test_error_gate_is_elementwise_and_nonfinite_safe() -> None:
    reference = torch.tensor([0.0, 1.0])
    assert benchmark.error_metrics(reference.clone(), reference, atol=0, rtol=0)["bit_exact"]
    assert not benchmark.error_metrics(reference + 1e-4, reference, atol=0, rtol=0)["passed"]
    assert benchmark.error_metrics(reference + 1e-4, reference, atol=1e-3, rtol=0)["passed"]
    assert not benchmark.error_metrics(reference.half(), reference, atol=0, rtol=0)["bit_exact"]
    assert not benchmark.error_metrics(torch.tensor([-0.0, 1]), reference, atol=0, rtol=0)["bit_exact"]
    strided = torch.arange(6, dtype=torch.float32)[::2]
    assert benchmark.error_metrics(strided, strided.clone(), atol=0, rtol=0)["bit_exact"]
    for value in (float("nan"), float("inf")):
        metrics = benchmark.error_metrics(torch.tensor([value, 1]), reference, atol=1, rtol=1)
        assert metrics == {
            "passed": False,
            "finite": False,
            "max_abs": None,
            "rmse": None,
            "bit_exact": False,
        }
    with pytest.raises(ValueError, match="matching shapes"):
        benchmark.error_metrics(reference[:1], reference, atol=0, rtol=0)


def test_reference_uses_padding_validity_and_gqa() -> None:
    # Equal zero logits over two valid keys => their mean, repeated over heads.
    q = torch.zeros(1, 4, 2, 64, dtype=torch.bfloat16)
    cached_k = torch.zeros(1, 1, 1, 64, dtype=q.dtype)
    k = torch.zeros(1, 1, 2, 64, dtype=q.dtype)
    cached_v = torch.full_like(cached_k, 3)
    v = torch.stack((torch.full_like(k[:, :, 0], 100), torch.full_like(k[:, :, 0], 5)), dim=2)
    mask = torch.tensor([[[[0.0, 2.0, 0.0]]]])
    actual = benchmark.reference_attention(q, cached_k, cached_v, k, v, mask)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, torch.full(q.shape, 4.0))
    with pytest.raises(ValueError, match="at least one valid key"):
        benchmark.reference_attention(q, cached_k, cached_v, k, v, torch.ones_like(mask))


def test_tensor_provenance_records_shape_stride_dtype_and_content() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    first = benchmark._provenance(tensor)
    assert first["shape"] == [4, 3]
    assert first["strides"] == [1, 4]
    assert first["dtype"] == "torch.float32"
    assert first["device"] == "cpu"
    assert len(first["sha256"]) == 64
    assert first["sha256"] == benchmark._provenance(tensor.contiguous())["sha256"]
    assert first["sha256"] != benchmark._provenance(tensor + 1)["sha256"]


def test_run_rejects_cpu_before_setup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("attempted CUDA setup"))
    args = benchmark.parse_args(["--atol", "0", "--rtol", "0", "--output", str(tmp_path / "unused.json")])
    with pytest.raises(RuntimeError, match="requires CUDA and Triton"):
        benchmark.run(args)


@pytest.mark.parametrize("passed", [True, False])
def test_main_writes_gate_result_and_returns_failure_status(
    monkeypatch, tmp_path: Path, passed: bool
) -> None:
    output = tmp_path / "nested" / "report.json"
    expected = {"passed": passed, "results": []}
    monkeypatch.setattr(benchmark, "run", lambda _: expected)
    status = benchmark.main(["--atol", "0", "--rtol", "0", "--output", str(output)])
    assert status == (0 if passed else 1)
    assert json.loads(output.read_text()) == expected


@pytest.fixture
def cuda_device() -> torch.device:
    capability = triton_capability()
    if not capability.available:
        pytest.skip(capability.reason)
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [None, *SUPPORTED_SPLIT_COUNTS])
@pytest.mark.parametrize(
    ("batch", "heads", "kv_heads", "queries", "cached", "width", "masked"),
    [
        (1, 4, 1, 7, 0, 64, False),
        (2, 4, 2, 50, 257, 128, True),
        (1, 2, 2, 65, 67, 256, True),
    ],
)
@torch.inference_mode()
def test_split_candidates_match_reference(
    cuda_device: torch.device,
    dtype: torch.dtype,
    splits: int | None,
    batch: int,
    heads: int,
    kv_heads: int,
    queries: int,
    cached: int,
    width: int,
    masked: bool,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support BF16")
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        generator = torch.Generator(device="cpu").manual_seed(123)

        def tensor(h: int, length: int) -> torch.Tensor:
            # Strided sequence axis exercises the kernel's explicit strides.
            return torch.randn(batch, h, length * 2, width, generator=generator).to(cuda_device, dtype)[
                :, :, ::2
            ]

        tensors = (
            tensor(heads, queries),
            tensor(kv_heads, cached),
            tensor(kv_heads, cached),
            tensor(kv_heads, queries),
            tensor(kv_heads, queries),
        )
        mask = None
        if masked:
            mask = torch.zeros(batch, 1, 1, cached + queries, device=cuda_device)
            mask[..., :cached] = -float("inf")  # Entirely padded and empty partitions must contribute zero.
        backend = get_split_kv_attention_backend("triton_split_kv")
        actual = backend.attend_split(*tensors, key_padding_mask=mask, num_splits=splits)
        reference = benchmark.reference_attention(*tensors, mask)
        # Fixed a priori for the distinct FP16/BF16 dot and softmax rounding.
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(actual.float(), reference, atol=tolerance, rtol=tolerance)
        default = backend.attend_split(*tensors, key_padding_mask=mask)
        torch.testing.assert_close(actual, default, atol=tolerance, rtol=tolerance)
        if splits is None or splits == (4 if queries <= 64 else 1):
            assert torch.equal(actual, default)
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.mark.gpu
def test_microbenchmark_capture_report(cuda_device: torch.device, tmp_path: Path) -> None:
    args = benchmark.parse_args(
        [
            "--atol",
            "0.002",
            "--rtol",
            "0.002",
            "--dtype",
            "float16",
            "--queries",
            "7",
            "--cached",
            "17",
            "--head-dim",
            "64",
            "--splits",
            "1",
            "4",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--capture",
            "--device",
            str(cuda_device.index),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    report = benchmark.run(args)
    assert report["passed"]
    assert [row["requested_splits"] for row in report["results"]] == [None, 1, 4]
    for row in report["results"]:
        assert row["capture_host_ms"] >= 0
        assert row["graph_correctness"]["bit_exact"]
        for mode in ("eager", "graph_replay"):
            assert len(row[mode]["cuda_elapsed"]["raw_ms"]) == 2
            assert len(row[mode]["synchronized_host"]["raw_ms"]) == 2
