"""Static numerical policies and validation of frozen, candidate-free calibration."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from ..contracts import ContractError, digest

VERSION = "baseline-fp64-v1"
SEEDS = (0, 1, 2)
PROFILES = (("recorded", 0), ("random", 1), ("random", 2), ("zeros", 0), ("cancellation", 0))
# Explicit semantic adapters, not substring matching on arbitrary operator names.
_ATEN = {
    "mm",
    "bmm",
    "matmul",
    "mv",
    "dot",
    "addmm",
    "baddbmm",
    "linear",
    "sum",
    "mean",
    "_softmax",
    "softmax",
    "_log_softmax",
    "log_softmax",
    "native_layer_norm",
    "layer_norm",
    "rms_norm",
    "scaled_dot_product_attention",
}
_BACKENDS = {
    "embodiinfer.backend.triton.split_kv_attention.split_kv_attention": "split_kv_attention",
    "embodiinfer.backend.triton.segmented_attention.segmented_attention": "segmented_attention",
}


def implementation_digest() -> str:
    """Bind calibration to the trusted policy/oracle source, including across machines."""
    root = Path(__file__).resolve().parents[3]
    paths = ("scripts/kernel_tuning/discovery/numerics.py", "benchmarks/kernel_tuning/calibration.py")
    return digest(
        {
            name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in paths
        }
    )


def policy(identity: dict[str, Any]) -> str | None:
    """Return a trusted reference adapter, or retain exact validation for other semantics."""
    if identity.get("mutates") or identity.get("rng"):
        return None
    dtypes = identity.get("input_dtypes", []) + identity.get("output_dtypes", [])
    if not any(dtype in {"float16", "bfloat16", "float32"} for dtype in dtypes):
        return None
    if any(dtype.startswith(("float8", "float4")) or dtype == "float64" for dtype in dtypes):
        return None
    if any(dtype not in {"float16", "bfloat16", "float32"} for dtype in identity.get("output_dtypes", [])):
        return None
    name = identity["name"]
    if identity["kind"] == "aten" and name.split(".")[1] in _ATEN:
        return "aten"
    return _BACKENDS.get(name)


def profile_key(profile: str, seed: int) -> str:
    """Stable identity of one real, random, or boundary validation input."""
    return f"{profile}:{seed}"


def execution_precision(captured: dict[str, Any]) -> dict[str, Any]:
    """Extract execution switches without inheriting legacy capture-wide bit exactness."""
    return {key: captured[key] for key in ("matmul_precision", "allow_tf32")}


def task_precision(contract: dict[str, Any], captured: dict[str, Any]) -> dict[str, Any]:
    """Summarize bounds for reporting; evaluation uses each profile/output's own bound."""
    bounds = [bound for case in contract["cases"].values() for sample in case.values() for bound in sample]
    return {
        **execution_precision(captured),
        "mode": "tolerance",
        "atol": max(bound["atol"] for bound in bounds),
        "rtol": max(bound["rtol"] for bound in bounds),
        "reason": "Frozen per-workload/output FP64 baseline calibration in numerics.json; "
        "different accumulation orders are allowed only within those measured bounds.",
    }


def validate_contract(
    contract: dict[str, Any],
    identity: dict[str, Any],
    cases: dict[str, Any],
    precision: dict[str, Any],
    flags: dict[str, Any],
) -> None:
    """Reject incomplete, misbound, or nonfinite calibration before task generation/evaluation."""
    expected = {
        "version": VERSION,
        "implementation_digest": implementation_digest(),
        "reference": policy(identity),
        "operator_identity": digest(identity),
        "precision": execution_precision(precision),
        "flags": flags,
        "profiles": [list(pair) for pair in PROFILES],
        "seeds": list(SEEDS),
    }
    if not expected["reference"] or any(contract.get(k) != v for k, v in expected.items()):
        raise ContractError("Numerical calibration policy/operator/execution settings changed; recalibrate")
    if not contract.get("environment") or set(contract.get("cases", {})) != set(cases):
        raise ContractError("Numerical calibration requires the target environment and every workload")
    for key, case in cases.items():
        if contract.get("workloads", {}).get(key) != digest(case):
            raise ContractError("Numerical calibration workload/fixture identity changed")
        if any("fixture" not in spec for spec in case["inputs"]):
            raise ContractError(
                "Calibration requires complete real input fixtures; recapture with a larger --fixture-bytes"
            )
        samples = contract["cases"][key]
        if set(samples) != {profile_key(*pair) for pair in PROFILES}:
            raise ContractError("Calibration must cover recorded, random and boundary inputs")
        for bounds in samples.values():
            if len(bounds) != len(case["outputs"]):
                raise ContractError("Calibration must cover every output")
            for bound, spec in zip(bounds, case["outputs"]):
                if bound.get("dtype") != spec["dtype"] or bound.get("reference_dtype") != "float64":
                    raise ContractError("Calibration output/reference dtype changed")
                for field in ("atol", "rtol", "baseline_max_abs", "residual_max", "rms", "epsilon", "tiny"):
                    value = bound.get(field)
                    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                        raise ContractError(f"Invalid calibration statistic: {field}")
                atol = max(bound["epsilon"] * bound["rms"], 2 * bound["residual_max"], bound["tiny"])
                if bound["atol"] != atol or bound["rtol"] != bound["epsilon"]:
                    raise ContractError("Calibration threshold does not follow the frozen policy")
