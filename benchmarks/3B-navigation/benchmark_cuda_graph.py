"""Real-image next-token CUDA Graph kernel benchmark for Low and Panoramic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Callable

import torch

from dataset import PROFILES, ManifestSample, load_manifest, profile_kind
from embodiinfer import make_policy
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    QwenVLNMemory as LowLevelMemory,
    _parse_low_level_action,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicMemory as PanoramicMemory,
    parse_panoramic_action,
)


MAX_GRAPH_SHAPES = 8
RTOL = 0.02
ATOL = 0.05
TRITON_HYBRID_HF_RTOL = 0.06
TRITON_HYBRID_HF_ATOL = 0.30
TRITON_HYBRID_TOP50_REQUIRED = 48


def _physical_gpu(device: torch.device) -> str:
    logical_index = device.index or 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(logical_index)
    values = [value.strip() for value in visible.split(",")]
    if logical_index >= len(values):
        raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    return values[logical_index]


def _source_revision() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _memory(kind: str, sample: ManifestSample):
    memory_type = LowLevelMemory if kind == "low" else PanoramicMemory
    return memory_type(
        frames=sample.history_frames,
        responses=sample.history_responses,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _encoded_description(encoded: dict[str, Any]) -> dict[str, Any]:
    description: dict[str, Any] = {}
    for key in sorted(encoded):
        value = encoded[key]
        if torch.is_tensor(value):
            description[key] = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
            }
        else:
            description[key] = {"type": type(value).__name__}
    return description


def _cohorts(
    descriptions: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for sample_id, description in descriptions:
        key = json.dumps(description, sort_keys=True, separators=(",", ":"))
        group = groups.setdefault(
            key,
            {"encoded_tensors": description, "sample_ids": []},
        )
        group["sample_ids"].append(sample_id)
    if len(groups) > MAX_GRAPH_SHAPES:
        counts = [len(group["sample_ids"]) for group in groups.values()]
        raise RuntimeError(
            f"manifest produces {len(groups)} encoded shape cohorts {counts}; "
            f"the manual graph cache admits at most {MAX_GRAPH_SHAPES}. "
            "Split the manifest into shape-homogeneous runs."
        )
    return [
        {"cohort": index, **group}
        for index, group in enumerate(groups.values())
    ]


def _timed_cuda(
    function: Callable[[], torch.Tensor], device: torch.device
) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


def _timing_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    mean_ms = statistics.fmean(values)
    return {
        "latencies_ms": values,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)],
        "kernel_samples_per_second": 1000.0 / mean_ms,
    }


def _comparison(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    top50_required: int | None,
) -> dict[str, Any]:
    left_top1 = left.argmax(-1).detach().cpu().tolist()
    right_top1 = right.argmax(-1).detach().cpu().tolist()
    left_top50 = set(
        left.topk(50, dim=-1).indices.detach().cpu().reshape(-1).tolist()
    )
    right_top50 = set(
        right.topk(50, dim=-1).indices.detach().cpu().reshape(-1).tolist()
    )
    top50_overlap = len(left_top50 & right_top50)
    top1_match = left_top1 == right_top1
    logits_close = bool(torch.allclose(left, right, rtol=rtol, atol=atol))
    top50_pass = top50_required is None or top50_overlap >= top50_required
    return {
        "top1_match": top1_match,
        "logits_close": logits_close,
        "max_abs": float((left - right).abs().max().item()),
        "rtol": rtol,
        "atol": atol,
        "top50_overlap": top50_overlap,
        "top50_required": top50_required,
        "passes_gate": top1_match and logits_close and top50_pass,
    }


def _signature(
    runner,
    logits: torch.Tensor,
    kind: str,
    candidate_count: int,
) -> dict[str, Any]:
    token_ids = logits.argmax(-1, keepdim=True)
    text = runner.processor.batch_decode(
        token_ids, skip_special_tokens=True
    )[0].strip()
    try:
        actions = (
            _parse_low_level_action(text)
            if kind == "low"
            else parse_panoramic_action(text, candidate_count)
        )
        action_values = actions.tolist()
        action_error = None
    except Exception as exc:
        action_values = None
        action_error = f"{type(exc).__name__}: {exc}"
    return {
        "token_ids": token_ids.detach().long().cpu().tolist(),
        "text": text,
        "actions": action_values,
        "action_error": action_error,
    }


def _graph_counters(stats: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(stats.get("capture_count", 0)),
        int(stats.get("replay_count", 0)),
        int(stats.get("cache_entries", 0)),
    )


def _attention_backend_record(
    runner, requested: str, profile: str
) -> dict[str, Any]:
    backend = runner.manual_graph_stats().get("attention_backend")
    if backend is None:
        return {
            "applicable": False,
            "requested": requested,
            "resolved": None,
            "fallback": False,
            "fallback_reason": None,
            "kernel_abi": None,
            "config": {},
            "layers_backend": None,
            "triton_version": None,
            "profile": profile,
        }
    record = dict(backend)
    record["applicable"] = True
    record["fallback"] = record.get("fallback_reason") is not None
    return record


def _torch_compile_record(runner, requested: str) -> dict[str, Any]:
    record = dict(runner.manual_graph_stats().get("torch_compile", {}))
    entries = list(record.get("entries", []))
    record.setdefault("requested", requested)
    record["effective"] = record.get("resolved", "eager")
    record.setdefault("target", "qwen25_vl_next_token_forward")
    record["abi"] = record.get("compile_abi")
    record["first_call_wall_ms"] = [
        float(entry.get("first_call_wall_ms", 0.0)) for entry in entries
    ]
    return record


def _compile_counters(runner, requested: str) -> dict[str, Any]:
    record = _torch_compile_record(runner, requested)
    raw_persistent = record.get("persistent_cache", {})
    persistent = dict(raw_persistent) if isinstance(raw_persistent, dict) else {}
    persistent_state = {
        "configured": bool(persistent.get("configured", False)),
        "schema": persistent.get("schema"),
        "root": persistent.get("root"),
        "launcher_contract_asserted": bool(
            persistent.get("launcher_contract_asserted", False)
        ),
        "libdevice": persistent.get("libdevice"),
        "entries": int(persistent.get("entries", 0)),
        "fingerprint": persistent.get("fingerprint"),
        "manifest_key": persistent.get("manifest_key"),
        "artifact_loaded": bool(persistent.get("artifact_loaded", False)),
        "artifact_published": bool(persistent.get("artifact_published", False)),
        "artifact_publish_skipped": bool(
            persistent.get("artifact_publish_skipped", False)
        ),
        "artifact_sha256": persistent.get("artifact_sha256"),
        "artifact_bytes": int(persistent.get("artifact_bytes", 0)),
        "load_cache_info": persistent.get("load_cache_info"),
        "save_cache_info": persistent.get("save_cache_info"),
        "cache_info_artifacts": persistent.get("cache_info_artifacts"),
        "cache_counters_before": persistent.get("cache_counters_before"),
        "cache_counters_after": persistent.get("cache_counters_after"),
        "cache_counters_delta": persistent.get("cache_counters_delta"),
        "persistent_hit_admission": persistent.get("persistent_hit_admission"),
        "cold_compile_reason": persistent.get("cold_compile_reason"),
        "quarantine_reason": persistent.get("quarantine_reason"),
        "corruption_policy": persistent.get("corruption_policy"),
        "entry_stats": persistent.get("entry_stats", {}),
    }
    if persistent_state["configured"]:
        if not persistent_state["launcher_contract_asserted"]:
            raise RuntimeError("compile cache bootstrap assertion was not recorded")
        if persistent_state["schema"] != "qwen25_vl_inductor_execution_cache_v3":
            raise RuntimeError("compile cache did not use execution-cache schema v3")
        if persistent_state["entries"] < 1:
            raise RuntimeError("compile cache has no execution-scoped entry")
        if (
            not persistent_state["fingerprint"]
            or persistent_state["manifest_key"]
            != persistent_state["fingerprint"]
        ):
            raise RuntimeError("compile cache fingerprint/manifest key is invalid")
        if persistent_state["quarantine_reason"] is not None:
            raise RuntimeError("compile cache quarantined an artifact")
        if (
            not persistent_state["artifact_sha256"]
            or persistent_state["artifact_bytes"] <= 0
        ):
            raise RuntimeError("compile cache has no validated content-addressed blob")
        if not (
            persistent_state["artifact_loaded"]
            or persistent_state["artifact_published"]
        ):
            raise RuntimeError("compile cache neither loaded nor published an artifact")
        if (
            persistent_state["artifact_publish_skipped"]
            and not persistent_state["artifact_loaded"]
        ):
            raise RuntimeError("only a loaded cache hit may skip artifact publish")
        admission = persistent_state["persistent_hit_admission"]
        if persistent_state["artifact_loaded"] and (
            not isinstance(admission, dict) or not admission.get("admitted")
        ):
            raise RuntimeError("loaded compile cache did not satisfy FX/AOT admission")
        entries = persistent_state["entry_stats"]
        if not isinstance(entries, dict) or not entries:
            raise RuntimeError("compile cache has no per-execution stats")
        for fingerprint, entry in entries.items():
            if not isinstance(entry, dict) or entry.get("manifest_key") != fingerprint:
                raise RuntimeError("compile cache execution entry is malformed")
            if entry.get("quarantine_reason") is not None:
                raise RuntimeError("compile cache execution entry was quarantined")
            if entry.get("artifact_loaded"):
                entry_admission = entry.get("persistent_hit_admission", {})
                if (
                    not isinstance(entry_admission, dict)
                    or not entry_admission.get("admitted")
                ):
                    raise RuntimeError(
                        "loaded execution entry did not satisfy FX/AOT admission"
                    )
            elif not entry.get("artifact_published"):
                raise RuntimeError("cold execution entry did not publish an artifact")
    return {
        key: int(record.get(key, 0))
        for key in ("cache_entries", "attempts", "failures")
    } | {
        "bucket_ids": record.get("bucket_ids", []),
        "persistent_cache": persistent_state,
    }


def _native_uncaptured_logits(
    runner, encoded: dict[str, torch.Tensor], kind: str
) -> torch.Tensor:
    if kind == "low":
        inputs = runner.graph_runtime.native_inputs(encoded)
        return runner.graph_runtime.forward(*inputs)
    inputs = runner._native_inputs(encoded)
    return runner.graph_decoder(*inputs)


def _parse_compile_text_buckets(value: str) -> tuple[int, ...]:
    try:
        buckets = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "compile text buckets must be comma-separated positive integers"
        ) from exc
    if not buckets or any(value <= 0 for value in buckets):
        raise argparse.ArgumentTypeError(
            "compile text buckets must be comma-separated positive integers"
        )
    if buckets != tuple(sorted(set(buckets))):
        raise argparse.ArgumentTypeError(
            "compile text buckets must be strictly increasing and unique"
        )
    return buckets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument(
        "--attention-backend",
        choices=("torch_sdpa", "triton", "auto"),
        default="torch_sdpa",
    )
    parser.add_argument(
        "--compile-backend",
        choices=("none", "inductor"),
        default="none",
        help="Compile the self-authored next-token target; JIT is outside timing.",
    )
    parser.add_argument("--compile-cache-dir", type=Path)
    parser.add_argument(
        "--compile-text-buckets",
        type=_parse_compile_text_buckets,
        default=(),
        metavar="S1,S2,...",
    )
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-physical-gpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser


# Keep timing-stability snapshots identical across the kernel and full-policy CLI.
def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.profile == "navida" and args.compile_backend != "none":
        parser.error(
            "NaViDA does not admit --compile-backend=inductor: real 3B "
            "history stochastic paired-seed token parity diverges; use "
            "--compile-backend=none"
        )
    if args.compile_cache_dir is not None and args.compile_backend != "inductor":
        parser.error("--compile-cache-dir requires --compile-backend=inductor")
    if args.compile_text_buckets and args.profile == "navida":
        parser.error("--compile-text-buckets is only supported for Low/Panoramic")
    if (
        args.compile_backend != "none"
        and args.profile != "navida"
        and args.attention_backend != "torch_sdpa"
    ):
        parser.error(
            "--compile-backend=inductor for Low/Panoramic requires "
            "--attention-backend=torch_sdpa; compile never falls back"
        )
    if args.profile == "navida" and args.attention_backend != "torch_sdpa":
        parser.error(
            "NaViDA does not support --attention-backend; use torch_sdpa or "
            "select a Low/Panoramic profile"
        )
    if args.profile == "navida":
        parser.error(
            "NaViDA manual CUDA Graph covers complete StaticCache autoregressive "
            "generation, not this single next-token kernel; use benchmark.py"
        )
    if args.warmup <= 0:
        parser.error("--warmup must be positive")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    try:
        manifest = load_manifest(
            args.manifest,
            args.data_root,
            args.profile,
            args.limit,
            args.provenance,
            require_formal=args.provenance is not None,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("the manual CUDA Graph benchmark requires CUDA")
    actual_gpu = _physical_gpu(device)
    if (
        args.expected_physical_gpu is not None
        and args.expected_physical_gpu != actual_gpu
    ):
        parser.error(
            f"expected physical GPU {args.expected_physical_gpu}, got {actual_gpu}"
        )

    torch.cuda.set_device(device)
    _seed_everything(args.seed)
    policy = make_policy(
        args.profile,
        checkpoint=args.checkpoint,
        attention_backend=args.attention_backend,
        compile_backend=args.compile_backend,
        compile_cache_dir=args.compile_cache_dir,
        compile_text_buckets=args.compile_text_buckets,
    ).eval()
    policy.to(device=device, dtype=torch.bfloat16)
    runner = policy.runner
    kind = profile_kind(args.profile)

    encoded_samples: list[tuple[ManifestSample, dict[str, Any]]] = []
    descriptions: list[tuple[str, dict[str, Any]]] = []
    with torch.inference_mode():
        for sample in manifest.samples:
            encoded = runner._encode_batch(
                [sample.observation], [_memory(kind, sample)]
            )
            description = _encoded_description(encoded)
            descriptions.append((sample.sample_id, description))
            encoded_samples.append((sample, encoded))
    cohorts = _cohorts(descriptions)

    parity_records: list[dict[str, Any]] = []
    capture_warmups: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample, encoded in encoded_samples:
            hf_logits = runner.model(
                **encoded, use_cache=False
            ).logits[:, -1].float()
            native_logits = _native_uncaptured_logits(
                runner, encoded, kind
            ).float()
            uncaptured_logits = runner._graph_logits(encoded).float()
            counters_before = _graph_counters(runner.manual_graph_stats())
            captured_logits = runner._manual_graph_logits(encoded).float()
            counters_after = _graph_counters(runner.manual_graph_stats())
            resolved_backend = runner.manual_graph_stats()["attention_backend"][
                "resolved"
            ]
            triton_kernel = resolved_backend == "triton_hybrid"
            hf_rtol = TRITON_HYBRID_HF_RTOL if triton_kernel else RTOL
            hf_atol = TRITON_HYBRID_HF_ATOL if triton_kernel else ATOL
            hf_top50_required = (
                TRITON_HYBRID_TOP50_REQUIRED if triton_kernel else None
            )
            inductor_compiled = args.compile_backend == "inductor"
            compiled_rtol = 0.06 if inductor_compiled else hf_rtol
            compiled_atol = 0.30 if inductor_compiled else hf_atol
            compiled_top50_required = (
                45 if inductor_compiled else hf_top50_required
            )
            comparisons = {
                "hf_vs_native": _comparison(
                    hf_logits,
                    native_logits,
                    rtol=hf_rtol,
                    atol=hf_atol,
                    top50_required=hf_top50_required,
                ),
                "hf_vs_uncaptured": _comparison(
                    hf_logits,
                    uncaptured_logits,
                    rtol=compiled_rtol,
                    atol=compiled_atol,
                    top50_required=compiled_top50_required,
                ),
                "hf_vs_captured": _comparison(
                    hf_logits,
                    captured_logits,
                    rtol=compiled_rtol,
                    atol=compiled_atol,
                    top50_required=compiled_top50_required,
                ),
                "uncaptured_vs_captured": _comparison(
                    uncaptured_logits,
                    captured_logits,
                    rtol=RTOL,
                    atol=ATOL,
                    top50_required=None,
                ),
                "native_vs_uncaptured": _comparison(
                    native_logits,
                    uncaptured_logits,
                    rtol=compiled_rtol,
                    atol=compiled_atol,
                    top50_required=compiled_top50_required,
                ),
                "native_vs_captured": _comparison(
                    native_logits,
                    captured_logits,
                    rtol=compiled_rtol,
                    atol=compiled_atol,
                    top50_required=compiled_top50_required,
                ),
            }
            candidate_count = len(
                sample.observation.metadata.get("candidates", ())
            )
            signatures = {
                "hf_top_level_eager": _signature(
                    runner, hf_logits, kind, candidate_count
                ),
                "self_authored_native": _signature(
                    runner, native_logits, kind, candidate_count
                ),
                "self_authored_uncaptured": _signature(
                    runner, uncaptured_logits, kind, candidate_count
                ),
                "captured_replay": _signature(
                    runner, captured_logits, kind, candidate_count
                ),
            }
            signature_values = list(signatures.values())
            signatures_exact = all(
                signature == signature_values[0]
                for signature in signature_values[1:]
            )
            actions_valid = all(
                signature["action_error"] is None
                for signature in signature_values
            )
            parity_records.append(
                {
                    "id": sample.sample_id,
                    "instruction": sample.instruction,
                    "image_paths": sample.image_paths,
                    "memory_source": sample.memory_source,
                    "resolved_attention_backend": resolved_backend,
                    "top1": {
                        "hf_top_level_eager": hf_logits.argmax(-1)
                        .detach()
                        .cpu()
                        .tolist(),
                        "self_authored_native": native_logits.argmax(-1)
                        .detach()
                        .cpu()
                        .tolist(),
                        "self_authored_uncaptured": uncaptured_logits.argmax(-1)
                        .detach()
                        .cpu()
                        .tolist(),
                        "captured_replay": captured_logits.argmax(-1)
                        .detach()
                        .cpu()
                        .tolist(),
                    },
                    "comparisons": comparisons,
                    "signatures": signatures,
                    "token_text_action_exact": signatures_exact,
                    "actions_valid": actions_valid,
                }
            )
            capture_warmups.append(
                {
                    "id": sample.sample_id,
                    "capture_delta": counters_after[0] - counters_before[0],
                    "replay_delta": counters_after[1] - counters_before[1],
                    "entry_delta": counters_after[2] - counters_before[2],
                }
            )

    mode_functions = {
        "hf_top_level_eager": lambda encoded: runner.model(
            **encoded, use_cache=False
        ).logits[:, -1],
        **(
            {
                "self_authored_native": lambda encoded: _native_uncaptured_logits(
                    runner, encoded, kind
                )
            }
            if args.compile_backend != "none"
            else {}
        ),
        (
            "torch_compiled_uncaptured"
            if args.compile_backend != "none"
            else "self_authored_uncaptured"
        ): lambda encoded: runner._graph_logits(encoded),
        "captured_replay": lambda encoded: runner._manual_graph_logits(encoded),
    }
    aggregate: dict[str, list[float]] = {name: [] for name in mode_functions}
    per_sample_timings: list[dict[str, Any]] = []
    timed_capture_count = 0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for sample, encoded in encoded_samples:
            sample_timings: dict[str, Any] = {"id": sample.sample_id}
            for name, function in mode_functions.items():
                for _ in range(args.warmup):
                    function(encoded)
                torch.cuda.synchronize(device)
                before = _graph_counters(runner.manual_graph_stats())
                compile_before = _compile_counters(
                    runner, args.compile_backend
                )
                values = [
                    _timed_cuda(lambda f=function, e=encoded: f(e), device)[0]
                    for _ in range(args.iters)
                ]
                after = _graph_counters(runner.manual_graph_stats())
                compile_after = _compile_counters(
                    runner, args.compile_backend
                )
                if compile_before != compile_after:
                    raise RuntimeError(
                        "torch.compile cache changed inside the timed window: "
                        f"before={compile_before}, after={compile_after}"
                    )
                if name == "captured_replay":
                    timed_capture_count += after[0] - before[0]
                aggregate[name].extend(values)
                sample_timings[name] = {
                    **_timing_summary(values),
                    "compile_before_timed": compile_before,
                    "compile_after_timed": compile_after,
                }
            per_sample_timings.append(sample_timings)

    all_parity = all(
        all(
            comparison["passes_gate"]
            for comparison in record["comparisons"].values()
        )
        and record["token_text_action_exact"]
        and record["actions_valid"]
        for record in parity_records
    )
    formal_admitted = (
        manifest.provenance["status"] == "verified"
        and manifest.provenance["formal"]
    )
    status = (
        ("pass" if formal_admitted else "unverified")
        if all_parity and timed_capture_count == 0
        else "parity_failure"
    )
    measurements = {
        name: _timing_summary(values) for name, values in aggregate.items()
    }
    output = {
        "status": status,
        "scope": (
            "B1 fixed-shape pre-encoded multimodal prefill-to-next-token kernel; "
            "compares Transformers top-level eager, self-authored uncaptured "
            "forward, and captured replay; excludes processor, H2D, generation, "
            "parser, EngineCore, memory commit, and simulator"
        ),
        "visual_source": (
            "provenance_consistent_r2r_aligned_rgb"
            if formal_admitted
            else "unverified_manifest_images"
        ),
        "synthetic": manifest.provenance["synthetic_pixels"],
        "profile": args.profile,
        "checkpoint": args.checkpoint,
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "data_root": str(manifest.data_root),
        "provenance": manifest.provenance,
        "sample_count": len(manifest.samples),
        "sample_ids": [sample.sample_id for sample in manifest.samples],
        "samples": [
            {
                "id": sample.sample_id,
                "instruction": sample.instruction,
                "image_paths": sample.image_paths,
                "memory_source": sample.memory_source,
            }
            for sample in manifest.samples
        ],
        "shape_cohorts": cohorts,
        "input_shapes": {
            sample.sample_id: _encoded_description(encoded)
            for sample, encoded in encoded_samples
        },
        "batch_size": 1,
        "seed": args.seed,
        "warmup_per_sample_per_path": args.warmup,
        "iterations_per_sample_per_path": args.iters,
        "tolerances": {
            "torch_hf_vs_raw_native": {
                "rtol": RTOL,
                "atol": ATOL,
                "top50_required": None,
            },
            "bfloat16_inductor_hf_or_raw_vs_compiled": {
                "rtol": 0.06,
                "atol": 0.30,
                "top50_required": 45,
                "semantic_gate": "top1/token/text/action exact",
            },
            "triton_hybrid_hf_vs_native_or_captured": {
                "rtol": TRITON_HYBRID_HF_RTOL,
                "atol": TRITON_HYBRID_HF_ATOL,
                "top50_required": TRITON_HYBRID_TOP50_REQUIRED,
            },
            "native_vs_captured": {
                "rtol": RTOL,
                "atol": ATOL,
                "top50_required": None,
            },
            "compiled_vs_compiled_captured": {
                "rtol": RTOL,
                "atol": ATOL,
                "top50_required": None,
            },
        },
        "parity": {
            "all_top1_and_logits_pass": all_parity,
            "captures_inside_timed_window": timed_capture_count,
            "samples": parity_records,
        },
        "capture_warmups": capture_warmups,
        "measurements": measurements,
        "per_sample_measurements": per_sample_timings,
        "attention_backend": _attention_backend_record(
            runner, args.attention_backend, args.profile
        ),
        "torch_compile": _torch_compile_record(runner, args.compile_backend),
        "compile_shape_cache": {
            "requested_text_buckets": list(args.compile_text_buckets),
            "compile_cache_dir": (
                str(args.compile_cache_dir.resolve())
                if args.compile_cache_dir is not None
                else None
            ),
            "observed_bucket_ids": _torch_compile_record(
                runner, args.compile_backend
            ).get("bucket_ids", []),
            "persistent_cache": _torch_compile_record(
                runner, args.compile_backend
            ).get("persistent_cache", {}),
        },
        "runtime_mode": {
            "uncaptured": (
                "compiled" if args.compile_backend != "none" else "eager"
            ),
            "captured": (
                "compiled_manual_cudagraph"
                if args.compile_backend != "none"
                else "manual_cudagraph"
            ),
        },
        "manual_graph": runner.manual_graph_stats(),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "timing_contract": {
            "clock": "CUDA events around one pre-encoded kernel invocation",
            "included": ["GPU kernel execution"],
            "excluded": [
                "manifest read and SHA256",
                "image disk I/O and PIL decode",
                "processor and tokenization",
                "host-to-device transfer",
                "model load and device transfer",
                "warmup and graph capture",
                "attention backend resolution and Triton JIT compilation",
                "torch.compile/Inductor first call and compile warmups",
                "persistent Mega-Cache artifact load/save",
                "autoregressive generation",
                "text decode and action parser",
                "JSON serialization",
            ],
        },
        "logical_device": str(device),
        "physical_gpu": actual_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "logical_device": str(device),
            "physical_gpu": actual_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_visible_device_count": torch.cuda.device_count(),
        },
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "commit": _source_revision(),
    }
    serialized = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if status == "pass" else (4 if status == "unverified" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
