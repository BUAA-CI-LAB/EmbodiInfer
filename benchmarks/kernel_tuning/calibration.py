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
    if identity["kind"] == "aten":
        _, name, overload = identity["name"].split(".")
        fn = getattr(getattr(torch.ops.aten, name), overload)
    else:
        fn = getattr(importlib.import_module(identity["module"]), identity["entry"])
    return tensor_outputs(fn(*args, **kwargs))


def _arguments(identity: dict[str, Any], inputs: list[torch.Tensor]) -> dict[str, Any]:
    args, kwargs = restore(identity["arguments"], inputs)
    if identity["kind"] == "aten":
        _, name, overload = identity["name"].split(".")
        schema = getattr(getattr(torch.ops.aten, name), overload)._schema
        return {
            arg.name: args[i] if i < len(args) else kwargs.get(arg.name, arg.default_value)
            for i, arg in enumerate(schema.arguments)
        }
    names = {
        "split_kv_attention": (
            "query",
            "prefix_key",
            "prefix_value",
            "suffix_key",
            "suffix_value",
            "mask",
            "scaling",
        ),
        "segmented_attention": ("q", "k", "v", "q_segment_offsets", "kv_segment_offsets"),
    }[policy(identity)]
    return {**dict(zip(names, args)), **kwargs}


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


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: Any = None,
    scale: float | None = None,
    causal: bool = False,
    gqa: bool = False,
) -> torch.Tensor:
    if gqa and q.shape[-3] != k.shape[-3]:
        k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
        v = v.repeat_interleave(q.shape[-3] // v.shape[-3], dim=-3)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    rows = []
    # Bound oracle scratch memory without changing each row's reduction semantics.
    for start in range(0, q.shape[-2], 128):
        stop = min(start + 128, q.shape[-2])
        scores = (q[..., start:stop, :] @ k.transpose(-1, -2)) * scale
        if causal:
            visible = (
                torch.arange(k.shape[-2], device=q.device)
                <= torch.arange(start, stop, device=q.device)[:, None]
            )
            scores = scores.masked_fill(~visible, float("-inf"))
        if mask is not None:
            local = mask[..., start:stop, :] if mask.ndim >= 2 and mask.shape[-2] != 1 else mask
            scores = (
                scores.masked_fill(~local, float("-inf")) if local.dtype == torch.bool else scores + local
            )
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(torch.isneginf(scores).all(dim=-1, keepdim=True), 0.0, probs)
        rows.append(probs @ v)
    return torch.cat(rows, dim=-2) if rows else q.new_empty((*q.shape[:-1], v.shape[-1]))


def high_precision(identity: dict[str, Any], inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Evaluate the declared mathematical operation in FP64 with unchanged attention semantics."""
    kind = policy(identity)
    if kind is None:
        raise ContractError("No trusted high-precision adapter for this operator")
    promoted = [x.to(torch.float64) if x.is_floating_point() else x for x in inputs]
    args = _arguments(identity, promoted)
    if kind == "aten":
        _, name, overload = identity["name"].split(".")
        if name == "scaled_dot_product_attention":
            if args["dropout_p"] != 0:
                raise ContractError("Attention calibration requires dropout_p=0")
            return [
                _attention(
                    args["query"],
                    args["key"],
                    args["value"],
                    mask=args["attn_mask"],
                    scale=args["scale"],
                    causal=args["is_causal"],
                    gqa=args.get("enable_gqa", False),
                )
            ]
        if "dtype" in args:
            args["dtype"] = torch.float64
        if "half_to_float" in args:
            args["half_to_float"] = False
        if name == "rms_norm" and args.get("eps") is None:
            # The default epsilon belongs to the captured input dtype, not the oracle dtype.
            args["eps"] = torch.finfo(inputs[0].dtype).eps
        fn = getattr(getattr(torch.ops.aten, name), overload)
        return tensor_outputs(fn(**args))
    if kind == "split_kv_attention":
        mask = args.get("mask")
        return [
            _attention(
                args["query"],
                torch.cat((args["prefix_key"], args["suffix_key"]), dim=2),
                torch.cat((args["prefix_value"], args["suffix_value"]), dim=2),
                mask=mask == 0 if mask is not None else None,
                scale=args["scaling"],
                gqa=True,
            )
        ]
    q, k, v = args["q"], args["k"], args["v"]
    qb = args["q_segment_offsets"].tolist()
    kb = args.get("kv_segment_offsets")
    kb = qb if kb is None else kb.tolist()
    output = torch.empty_like(q)
    for qs, qe, ks, ke in zip(qb[:-1], qb[1:], kb[:-1], kb[1:]):
        output[qs:qe] = _attention(
            q[qs:qe].transpose(0, 1),
            k[ks:ke].transpose(0, 1),
            v[ks:ke].transpose(0, 1),
            scale=args.get("scaling"),
            gqa=True,
        ).transpose(0, 1)
    return [output]


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
