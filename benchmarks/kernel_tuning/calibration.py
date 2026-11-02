"""Candidate-free calibration and replay of frozen operator numerical contracts."""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from scripts.kernel_tuning.artifacts import atomic_json
from scripts.kernel_tuning.contracts import ContractError, digest, tree_hashes
from scripts.kernel_tuning.discovery.contracts import ModelCapture
from scripts.kernel_tuning.discovery.numerics import (
    PROFILES,
    SEEDS,
    VERSION,
    execution_precision,
    implementation_digest,
    policy,
    profile_key,
    validate_contract,
)
from scripts.kernel_tuning.discovery.tensors import (
    clone_inputs,
    describe_inputs,
    extent,
    materialize,
    restore,
    tensor_outputs,
)


def environment(device: str) -> dict[str, Any]:
    """Identity of the actual arithmetic environment, checked again at evaluation."""
    cuda = torch.device(device).type == "cuda"
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device,
        "hardware": torch.cuda.get_device_name(device) if cuda else "cpu",
        "capability": list(torch.cuda.get_device_capability(device)) if cuda else [],
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "fp16_reduced_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
        "default_dtype": str(torch.get_default_dtype()),
        "autocast_enabled": torch.is_autocast_enabled("cuda" if cuda else "cpu"),
        "environment": {
            key: os.environ.get(key) for key in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")
        },
    }


@contextmanager
def captured_settings(precision: dict[str, Any], flags: dict[str, Any]) -> Iterator[None]:
    """Use captured arithmetic switches and restore every changed process-global flag."""
    old = environment("cpu")
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_float32_matmul_precision(precision["matmul_precision"])
        torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags["fp16_reduced_reduction"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags["bf16_reduced_reduction"]
        torch.use_deterministic_algorithms(flags["deterministic_algorithms"])
        yield
    finally:
        torch.set_float32_matmul_precision(old["matmul_precision"])
        torch.backends.cudnn.allow_tf32 = old["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old["fp16_reduced_reduction"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old["bf16_reduced_reduction"]
        torch.use_deterministic_algorithms(old["deterministic_algorithms"], warn_only=warn)


def production(identity: dict[str, Any], inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Execute only the captured production implementation, never a candidate."""
    args, kwargs = restore(identity["arguments"], inputs)
    if identity["kind"] != "registry":
        raise ContractError("Only registry operators can be calibrated; recapture with capture-ops")
    fn = getattr(importlib.import_module(identity["module"]), identity["entry"])
    return tensor_outputs(fn(*args, **kwargs))


def _arguments(identity: dict[str, Any], inputs: list[torch.Tensor]) -> dict[str, Any]:
    import inspect

    args, kwargs = restore(identity["arguments"], inputs)
    function = getattr(importlib.import_module(identity["module"]), identity["entry"])
    return {**dict(zip(inspect.signature(function).parameters, args)), **kwargs}


def validation_inputs(
    identity: dict[str, Any],
    case: dict[str, Any],
    root: Path,
    profile: str,
    seed: int,
) -> list[torch.Tensor]:
    """Generate a fixed validation input while retaining masks, indices, layouts and aliases."""
    inputs = materialize(case["inputs"], root, 0)
    if profile == "recorded":
        return inputs
    if (profile, seed) not in PROFILES:
        raise ContractError("Unknown numerical validation profile")
    arguments = _arguments(identity, inputs)
    protected = [
        value for name, value in arguments.items() if "mask" in name and isinstance(value, torch.Tensor)
    ]
    protected_groups = {
        spec["storage"]
        for value, spec in zip(inputs, case["inputs"])
        if any(value is item for item in protected)
    }
    specs = describe_inputs(inputs)
    seen = set()
    for tensor, spec in zip(inputs, specs):
        group = spec["storage"]
        if group in seen or group in protected_groups or not tensor.is_floating_point():
            continue
        seen.add(group)
        size = max(extent(item) for item in specs if item["storage"] == group)
        storage = tensor.as_strided((size,), (1,), 0)
        if profile == "random":
            generator = torch.Generator(device=tensor.device).manual_seed(seed * 1009 + group)
            storage.copy_(torch.randn(size, dtype=tensor.dtype, device=tensor.device, generator=generator))
        elif profile == "zeros":
            storage.zero_()
        else:
            storage.copy_((torch.arange(size, device=tensor.device) % 2) * 2 - 1)
    return inputs


def high_precision(identity: dict[str, Any], inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Evaluate the registry's Torch reference with floating inputs promoted to FP64."""
    if policy(identity) != "registry":
        raise ContractError("No trusted high-precision adapter for this operator")
    promoted = [x.to(torch.float64) if x.is_floating_point() else x for x in inputs]
    return production(identity, promoted)


def derive_bound(baseline: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Derive a per-output all-element bound using only baseline/reference error."""
    if (
        baseline.shape != reference.shape
        or not baseline.is_floating_point()
        or reference.dtype != torch.float64
    ):
        raise ContractError("Calibration requires matching floating outputs and an FP64 reference")
    if not torch.isfinite(baseline).all() or not torch.isfinite(reference).all():
        raise ContractError("Nonfinite baseline/reference cannot calibrate a tolerance")
    info = torch.finfo(baseline.dtype)
    error = (baseline.double() - reference).abs()
    rms = reference.square().mean().sqrt().item() if reference.numel() else 0.0
    residual = (error - info.eps * reference.abs()).clamp_min(0).max().item() if error.numel() else 0.0
    return {
        "dtype": str(baseline.dtype).removeprefix("torch."),
        "reference_dtype": "float64",
        "atol": max(info.eps * rms, 2 * residual, info.tiny),
        "rtol": info.eps,
        "baseline_max_abs": error.max().item() if error.numel() else 0.0,
        "residual_max": residual,
        "rms": rms,
        "epsilon": info.eps,
        "tiny": info.tiny,
    }


def compare_calibrated(
    actual: list[torch.Tensor], reference: list[torch.Tensor], bounds: list[dict[str, Any]]
) -> tuple[float, float]:
    """Enforce frozen per-output thresholds on every finite element, with no matched-ratio escape."""
    if len(actual) != len(reference) or len(actual) != len(bounds):
        raise ContractError("Calibrated output count changed")
    maximum, relative = 0.0, 0.0
    for value, ref, bound in zip(actual, reference, bounds):
        if (
            str(value.dtype).removeprefix("torch.") != bound["dtype"]
            or value.shape != ref.shape
            or value.device != ref.device
        ):
            raise ContractError("Calibrated output shape/dtype/device changed")
        if not torch.isfinite(value).all() or not torch.isfinite(ref).all():
            raise ContractError("Calibrated comparison requires finite outputs")
        error = (value.double() - ref).abs()
        if not torch.all(error <= bound["atol"] + bound["rtol"] * ref.abs()):
            raise ContractError("Output exceeds frozen calibrated tolerance")
        if error.numel():
            maximum = max(maximum, error.max().item())
            # Use the absolute floor as the denominator floor to keep near-zero diagnostics finite.
            relative = max(relative, (error / ref.abs().clamp_min(bound["atol"])).max().item())
    return maximum, relative


def calibrate(capture: ModelCapture, destination: Path) -> ModelCapture:
    """Create a new immutable calibrated capture before any candidate source exists."""
    from scripts.kernel_tuning.discovery.tasks import _sources

    checked = ModelCapture.load(capture.root)
    if checked.identity != capture.identity:
        raise ContractError("Capture changed before calibration")
    if capture.manifest["torch_version"] != str(torch.__version__):
        raise ContractError("Calibration requires the capture's Torch version")
    destination = destination.resolve()
    if destination.exists():
        raise ContractError("Calibration requires a new destination; existing captures/runs are immutable")
    if destination.is_relative_to(capture.root):
        raise ContractError("Calibration destination must be outside its source capture")
    results = {}
    precision, flags = capture.manifest["precision"], capture.manifest["flags"]
    with captured_settings(precision, flags), torch.no_grad():
        for op in capture.operators:
            identity = op["identity"]
            reference = policy(identity) if op["status"] == "ready" else None
            if reference is None:
                continue
            _sources(op)  # Verify backend source/dependency digests before executing them.
            cases = {case["id"]: case for case in op["workloads"]}
            device = op["workloads"][0]["outputs"][0]["device"]
            contract = {
                "version": VERSION,
                "implementation_digest": implementation_digest(),
                "reference": reference,
                "operator_identity": digest(identity),
                "precision": execution_precision(precision),
                "flags": flags,
                "profiles": [list(pair) for pair in PROFILES],
                "seeds": list(SEEDS),
                "environment": environment(device),
                "workloads": {k: digest(v) for k, v in cases.items()},
                "cases": {},
            }
            for key, case in cases.items():
                if any("fixture" not in spec for spec in case["inputs"]):
                    raise ContractError(
                        f"{op['name']}/{key}: real inputs missing; recapture with a larger --fixture-bytes"
                    )
                samples = {}
                for profile, seed in PROFILES:
                    inputs = validation_inputs(identity, case, capture.root, profile, seed)
                    baseline = production(identity, clone_inputs(inputs))
                    ref = high_precision(identity, inputs)
                    if len(baseline) != len(ref) or len(ref) != len(case["outputs"]):
                        raise ContractError("High-precision reference output count differs from capture")
                    bounds = [derive_bound(b, r) for b, r in zip(baseline, ref)]
                    compare_calibrated(baseline, ref, bounds)
                    samples[profile_key(profile, seed)] = bounds
                contract["cases"][key] = samples
                if environment(device) != contract["environment"]:
                    raise ContractError("Numerical environment changed during calibration")
            validate_contract(contract, identity, cases, precision, flags)
            results[op["id"]] = contract
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calibration-", dir=destination.parent) as scratch:
        root = Path(scratch) / "capture"
        shutil.copytree(capture.root, root)
        # Refuse a source changed while calibration was running or while copying.
        if ModelCapture.load(root).identity != capture.identity:
            raise ContractError("Capture changed during calibration")
        atomic_json(root / "calibration.json", results)
        manifest = dict(capture.manifest)
        files = tree_hashes(root)
        files.pop("manifest.json")
        manifest.update(files=files, numerical_policy=VERSION, calibration_source=capture.identity)
        atomic_json(root / "manifest.json", manifest)
        ModelCapture.load(root)
        root.rename(destination)
    return ModelCapture.load(destination)
