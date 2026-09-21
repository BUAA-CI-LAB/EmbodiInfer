"""Real-image, correctness-gated full-policy benchmark for 3B navigation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Any

import torch

from dataset import PROFILES, LoadedManifest, ManifestSample, load_manifest, profile_kind
from embodiinfer import make_policy
from embodiinfer.policies.navida.modeling_navida import NaViDAMemory, _parse_navida_actions
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    QwenVLNMemory as LowLevelMemory,
    _parse_low_level_action,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicMemory as PanoramicMemory,
    parse_panoramic_action,
)


MAX_GRAPH_SHAPES = 8


def _physical_gpu(device: torch.device) -> str:
    logical_index = device.index or 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(logical_index)
    identifiers = [item.strip() for item in visible.split(",")]
    if logical_index >= len(identifiers):
        raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    return identifiers[logical_index]


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
    if kind == "low":
        memory_type = LowLevelMemory
    elif kind == "panoramic":
        memory_type = PanoramicMemory
    else:
        memory_type = NaViDAMemory
    return memory_type(
        frames=sample.history_frames,
        responses=sample.history_responses,
    )


def _signature(
    kind: str,
    result: tuple[str, torch.Tensor, list[float]],
    candidate_count: int,
) -> dict[str, Any]:
    text, token_ids, _ = result
    try:
        if kind == "low":
            actions = _parse_low_level_action(text)
        elif kind == "panoramic":
            actions = parse_panoramic_action(text, candidate_count)
        else:
            actions = _parse_navida_actions(text)
    except Exception as exc:
        return {
            "text": text,
            "token_ids": token_ids.detach().long().cpu().tolist(),
            "actions": None,
            "action_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "text": text,
        "token_ids": token_ids.detach().long().cpu().tolist(),
        "actions": actions.tolist(),
        "action_error": None,
    }


def _derived_seed(seed: int, sample_id: str, iteration: int) -> int:
    payload = f"{seed}\0{sample_id}\0{iteration}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_once(
    runner,
    sample: ManifestSample,
    memory,
    kind: str,
    device: torch.device,
) -> tuple[float, dict[str, Any], int]:
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    result = runner.infer_batch([sample.observation], [memory])[0]
    torch.cuda.synchronize(device)
    signature = _signature(
        kind,
        result,
        len(sample.observation.metadata.get("candidates", ())),
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return elapsed_ms, signature, int(result[1].numel())


def _paired_signature_exact(
    reference: dict[str, Any] | None,
    requested_uncaptured: dict[str, Any],
    requested_captured: dict[str, Any],
) -> bool:
    values = [requested_uncaptured, requested_captured]
    if reference is not None:
        values.insert(0, reference)
    return all(value == values[0] for value in values[1:])


def _summary(
    latencies_ms: list[float], generated_tokens: list[int]
) -> dict[str, Any]:
    ordered = sorted(latencies_ms)
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    elapsed_ms = sum(latencies_ms)
    mean_ms = statistics.fmean(latencies_ms)
    samples_per_second = len(latencies_ms) * 1000.0 / elapsed_ms
    return {
        "latencies_ms": latencies_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p95_ms": ordered[p95_index],
        "model_samples_per_second": samples_per_second,
        "samples_per_second": samples_per_second,
        "generated_tokens_per_sample_mean": statistics.fmean(generated_tokens),
        "generated_tokens_per_second": sum(generated_tokens) * 1000.0 / elapsed_ms,
        "total_generated_tokens": sum(generated_tokens),
    }


def _graph_counters(stats: dict[str, Any]) -> tuple[int, int, int]:
    if "navida_decode" in stats:
        stats = stats["navida_decode"]
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
    record.setdefault(
        "target",
        "navida_static_cache_decode"
        if runner.__class__.__name__.startswith("NaViDA")
        else "qwen25_vl_next_token_forward",
    )
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


def _runtime_modes(profile: str, compile_backend: str) -> dict[str, str]:
    if compile_backend == "none":
        return {"uncaptured": "eager", "captured": "manual_cudagraph"}
    return {
        "uncaptured": "compiled",
        "captured": "compiled_manual_cudagraph",
    }


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


def _shape_cohorts(
    runner,
    manifest: LoadedManifest,
    kind: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for sample in manifest.samples:
            encoded = runner._encode_batch(
                [sample.observation], [_memory(kind, sample)]
            )
            description = _encoded_description(encoded)
            key = json.dumps(description, sort_keys=True, separators=(",", ":"))
            group = groups.setdefault(
                key,
                {"encoded_tensors": description, "sample_ids": []},
            )
            group["sample_ids"].append(sample.sample_id)
            del encoded
    torch.cuda.synchronize(device)
    if kind != "navida" and len(groups) > MAX_GRAPH_SHAPES:
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
        help="Compile the self-authored next-token/decode target; JIT is outside timing.",
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
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
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
    if args.warmup <= 0:
        parser.error("--warmup must be positive so graph capture is outside timing")
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
        parser.error("the real-image 3B benchmark requires an available CUDA device")
    actual_physical_gpu = _physical_gpu(device)
    if (
        args.expected_physical_gpu is not None
        and args.expected_physical_gpu != actual_physical_gpu
    ):
        parser.error(
            f"expected physical GPU {args.expected_physical_gpu}, "
            f"got {actual_physical_gpu}"
        )

    torch.cuda.set_device(device)
    kind = profile_kind(args.profile)
    independent_reference_required = (
        args.profile != "navida"
        and args.compile_backend == "inductor"
    )
    reference_signatures: dict[str, dict[str, Any]] = {}
    reference_gate: dict[str, Any] = {
        "required": independent_reference_required,
        "attention_backend": "torch_sdpa" if independent_reference_required else None,
        "compile_backend": "none" if independent_reference_required else None,
        "timed": False,
        "sample_count": 0,
        "wall_seconds": None,
    }
    if independent_reference_required:
        reference_started = time.perf_counter()
        reference_policy = make_policy(
            args.profile,
            checkpoint=args.checkpoint,
            attention_backend="torch_sdpa",
            compile_backend="none",
        ).eval()
        reference_policy.to(device=device, dtype=torch.bfloat16)
        reference_runner = reference_policy.runner
        reference_runner.configure_cuda_graph(False)
        with torch.inference_mode():
            for sample in manifest.samples:
                paired_seed = _derived_seed(args.seed, sample.sample_id, 0)
                _seed_everything(paired_seed)
                _, signature, _ = _run_once(
                    reference_runner,
                    sample,
                    _memory(kind, sample),
                    kind,
                    device,
                )
                reference_signatures[sample.sample_id] = signature
        torch.cuda.synchronize(device)
        reference_gate["sample_count"] = len(reference_signatures)
        reference_gate["wall_seconds"] = time.perf_counter() - reference_started
        del reference_runner
        del reference_policy
        gc.collect()
        torch.cuda.empty_cache()

    policy_kwargs: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "compile_backend": args.compile_backend,
    }
    if args.profile != "navida":
        policy_kwargs["attention_backend"] = args.attention_backend
        policy_kwargs["compile_cache_dir"] = args.compile_cache_dir
        policy_kwargs["compile_text_buckets"] = args.compile_text_buckets
    policy = make_policy(args.profile, **policy_kwargs).eval()
    policy.to(device=device, dtype=torch.bfloat16)
    runner = policy.runner
    stochastic = args.profile == "navida"
    stability_required = not stochastic
    cohorts = _shape_cohorts(runner, manifest, kind, device)
    torch.cuda.empty_cache()

    if not runner.configure_cuda_graph(True):
        raise RuntimeError("manual CUDA Graph could not be enabled for capture planning")
    capture_preparation: list[dict[str, Any]] = []
    for sample in manifest.samples:
        counters_before = _graph_counters(runner.manual_graph_stats())
        capture_seed = _derived_seed(args.seed, sample.sample_id, -2)
        _seed_everything(capture_seed)
        _run_once(runner, sample, _memory(kind, sample), kind, device)
        counters_after = _graph_counters(runner.manual_graph_stats())
        capture_preparation.append(
            {
                "id": sample.sample_id,
                "seed": capture_seed,
                "capture_delta": counters_after[0] - counters_before[0],
                "replay_delta": counters_after[1] - counters_before[1],
                "entry_delta": counters_after[2] - counters_before[2],
            }
        )

    paired_seeded_records: list[dict[str, Any]] = []
    for sample in manifest.samples:
        paired_seed = _derived_seed(args.seed, sample.sample_id, 0)
        memory = _memory(kind, sample)
        runner.configure_cuda_graph(False)
        _seed_everything(paired_seed)
        _, eager_signature, _ = _run_once(runner, sample, memory, kind, device)
        runner.configure_cuda_graph(True)
        counters_before = _graph_counters(runner.manual_graph_stats())
        _seed_everything(paired_seed)
        _, graph_signature, _ = _run_once(runner, sample, memory, kind, device)
        counters_after = _graph_counters(runner.manual_graph_stats())
        paired_seeded_records.append(
            {
                "id": sample.sample_id,
                "seed": paired_seed,
                "reference_torch_sdpa_compile_none": reference_signatures.get(
                    sample.sample_id
                ),
                "eager": eager_signature,
                "manual_cudagraph": graph_signature,
                "token_text_action_exact": _paired_signature_exact(
                    reference_signatures.get(sample.sample_id),
                    eager_signature,
                    graph_signature,
                ),
                "capture_delta": counters_after[0] - counters_before[0],
            }
        )
    runner.configure_cuda_graph(False)

    mode_records: dict[str, list[dict[str, Any]]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    runtime_modes = _runtime_modes(args.profile, args.compile_backend)
    for mode in (runtime_modes["uncaptured"], runtime_modes["captured"]):
        requested_graph = mode == runtime_modes["captured"]
        effective_graph = runner.configure_cuda_graph(requested_graph)
        if requested_graph != effective_graph:
            raise RuntimeError(f"requested {mode}, but the runner selected eager")
        _seed_everything(args.seed)

        latencies: list[float] = []
        token_counts: list[int] = []
        records: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats(device)
        for sample in manifest.samples:
            memory = _memory(kind, sample)
            counters_before = _graph_counters(runner.manual_graph_stats())
            warmup_latencies: list[float] = []
            for _ in range(args.warmup):
                warmup_ms, _, _ = _run_once(
                    runner, sample, memory, kind, device
                )
                warmup_latencies.append(warmup_ms)
            counters_after_warmup = _graph_counters(runner.manual_graph_stats())
            compile_before_timed = _compile_counters(
                runner, args.compile_backend
            )

            signatures: list[dict[str, Any]] = []
            sample_latencies: list[float] = []
            sample_tokens: list[int] = []
            for _ in range(args.iters):
                elapsed_ms, signature, generated = _run_once(
                    runner, sample, memory, kind, device
                )
                sample_latencies.append(elapsed_ms)
                sample_tokens.append(generated)
                signatures.append(signature)
            counters_after_timed = _graph_counters(runner.manual_graph_stats())
            compile_after_timed = _compile_counters(
                runner, args.compile_backend
            )
            if compile_before_timed != compile_after_timed:
                raise RuntimeError(
                    "torch.compile cache changed inside the timed window: "
                    f"before={compile_before_timed}, after={compile_after_timed}"
                )
            latencies.extend(sample_latencies)
            token_counts.extend(sample_tokens)
            records.append(
                {
                    "id": sample.sample_id,
                    "instruction": sample.instruction,
                    "image_paths": sample.image_paths,
                    "memory_source": sample.memory_source,
                    "warmup_latencies_ms": warmup_latencies,
                    "capture_warmup_delta": (
                        counters_after_warmup[0] - counters_before[0]
                    ),
                    "replay_warmup_delta": (
                        counters_after_warmup[1] - counters_before[1]
                    ),
                    "entry_warmup_delta": (
                        counters_after_warmup[2] - counters_before[2]
                    ),
                    "capture_timed_delta": (
                        counters_after_timed[0] - counters_after_warmup[0]
                    ),
                    "compile_before_timed": compile_before_timed,
                    "compile_after_timed": compile_after_timed,
                    "runtime_mode": mode,
                    "latencies_ms": sample_latencies,
                    "generated_tokens": sample_tokens,
                    "signatures": signatures,
                }
            )
        measurements[mode] = {
            **_summary(latencies, token_counts),
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        }
        mode_records[mode] = records

    eager_by_id = {
        record["id"]: record
        for record in mode_records[runtime_modes["uncaptured"]]
    }
    graph_by_id = {
        record["id"]: record
        for record in mode_records[runtime_modes["captured"]]
    }
    mismatch_ids = [
        record["id"]
        for record in paired_seeded_records
        if not record["token_text_action_exact"]
    ]
    invalid_action_ids: list[str] = []
    unstable_ids: list[str] = []
    timed_capture_count = 0
    for sample in manifest.samples:
        eager = eager_by_id[sample.sample_id]
        graphed = graph_by_id[sample.sample_id]
        all_signatures = eager["signatures"] + graphed["signatures"]
        if any(signature["action_error"] is not None for signature in all_signatures):
            invalid_action_ids.append(sample.sample_id)
        if any(
            any(value != signatures[0] for value in signatures[1:])
            for signatures in (eager["signatures"], graphed["signatures"])
        ):
            unstable_ids.append(sample.sample_id)
        timed_capture_count += int(graphed["capture_timed_delta"])

    for record in paired_seeded_records:
        validation_signatures = [
            record["eager"],
            record["manual_cudagraph"],
        ]
        if record["reference_torch_sdpa_compile_none"] is not None:
            validation_signatures.insert(
                0, record["reference_torch_sdpa_compile_none"]
            )
        if any(
            signature["action_error"] is not None
            for signature in validation_signatures
        ) and record["id"] not in invalid_action_ids:
            invalid_action_ids.append(record["id"])

    paired_capture_count = sum(
        int(record["capture_delta"]) for record in paired_seeded_records
    )
    paired_seeded_parity = not mismatch_ids and paired_capture_count == 0
    stability_gate_passed = not stability_required or not unstable_ids
    parity = (
        paired_seeded_parity
        and not invalid_action_ids
        and stability_gate_passed
        and timed_capture_count == 0
    )
    formal_admitted = (
        manifest.provenance["status"] == "verified"
        and manifest.provenance["formal"]
    )
    status = (
        ("pass" if formal_admitted else "unverified")
        if parity
        else "parity_failure"
    )
    sample_trace = [
        {
            "id": sample.sample_id,
            "instruction": sample.instruction,
            "image_paths": sample.image_paths,
            "memory_source": sample.memory_source,
        }
        for sample in manifest.samples
    ]
    output = {
        "status": status,
        "scope": (
            "B1 runner.infer_batch full policy including prompt construction, "
            "processor, H2D, prefill, autoregressive decode, text decode, and "
            "action parser; excludes EngineCore, recurrent commit, and simulator"
        ),
        "visual_source": (
            "provenance_consistent_r2r_aligned_rgb"
            if formal_admitted
            else "unverified_manifest_images"
        ),
        "synthetic": manifest.provenance["synthetic_pixels"],
        "profile": args.profile,
        "stochastic": stochastic,
        "stability_required": stability_required,
        "checkpoint": args.checkpoint,
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "data_root": str(manifest.data_root),
        "provenance": manifest.provenance,
        "sample_count": len(manifest.samples),
        "sample_ids": [sample.sample_id for sample in manifest.samples],
        "samples": sample_trace,
        "shape_cohorts": cohorts,
        "batch_size": 1,
        "seed": args.seed,
        "timing_seed_per_mode": args.seed,
        "warmup_per_sample_per_mode": args.warmup,
        "iterations_per_sample_per_mode": args.iters,
        "timing_contract": {
            "clock": (
                "time.perf_counter_ns with CUDA synchronization around each "
                "runner.infer_batch call"
            ),
            "included": [
                "prompt construction",
                "processor",
                "host-to-device transfer",
                "model prefill and decode",
                "token/text decode",
                "action parser",
            ],
            "excluded": [
                "manifest read and SHA256",
                "image disk I/O and PIL decode",
                "model load and device transfer",
                "encoded shape cohort planning",
                "warmup and CUDA Graph capture",
                "attention backend resolution and Triton JIT compilation",
                "torch.compile/Inductor first call and compile warmups",
                "persistent Mega-Cache artifact load/save",
                "paired seeded eager/manual-graph validation",
                "independent torch_sdpa/compile-none reference runner lifecycle",
                "RNG seed reset",
                "JSON serialization",
            ],
            "capture_excluded_from_timed_samples": timed_capture_count == 0,
        },
        "measurements": measurements,
        "parity": {
            "paired_seeded_parity": paired_seeded_parity,
            "independent_reference_gate": reference_gate,
            "paired_seeded_seed_base": args.seed,
            "paired_seeded_samples": paired_seeded_records,
            "paired_validation_captures": paired_capture_count,
            "all_token_text_action_signatures_match": not mismatch_ids,
            "all_actions_valid": not invalid_action_ids,
            "outputs_stable_within_mode": not unstable_ids,
            "stability_required": stability_required,
            "stability_gate_passed": stability_gate_passed,
            "mismatch_sample_ids": mismatch_ids,
            "invalid_action_sample_ids": invalid_action_ids,
            "unstable_sample_ids": unstable_ids,
            "captures_inside_timed_window": timed_capture_count,
        },
        "capture_preparation": capture_preparation,
        "outputs": mode_records,
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
        "runtime_mode": runtime_modes,
        "manual_graph": runner.manual_graph_stats(),
        "logical_device": str(device),
        "physical_gpu": actual_physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "logical_device": str(device),
            "physical_gpu": actual_physical_gpu,
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
