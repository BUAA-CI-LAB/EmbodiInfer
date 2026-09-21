#!/usr/bin/env python3
"""Pure-stdlib producer/consumer launcher for the Qwen2.5-VL Mega-Cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterator


BOOTSTRAP_ENV = "VVLA_COMPILE_CACHE_BOOTSTRAP_ASSERTION"
BOOTSTRAP_TOKEN = "qwen25_vl_compile_cache_preimport_v1"
LOW_PROFILE = "qwen2.5-vl-3b-r2r-low-level"
PANORAMIC_PROFILE = "qwen2.5-vl-3b-r2r-panoramic"
PROFILES = (LOW_PROFILE, PANORAMIC_PROFILE)
MEGA_CACHE_DIRS = ("vvla-execution-entries", "vvla-content-blobs")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _secure_directory(path: Path, *, empty: bool) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"cache path must be a real directory: {path}")
        if empty and any(path.iterdir()):
            raise RuntimeError(f"producer/fresh cache directory must be empty: {path}")
    else:
        path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)


def _copy_mega_cache(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for name in MEGA_CACHE_DIRS:
        source_dir = source / name
        if not source_dir.is_dir() or source_dir.is_symlink():
            raise RuntimeError(f"producer did not create trusted {name}: {source_dir}")
        destination_dir = destination / name
        shutil.copytree(source_dir, destination_dir)
        destination_dir.chmod(0o700)
        for path in destination_dir.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"Mega-Cache copy contains a symlink: {path}")
            path.chmod(0o700 if path.is_dir() else 0o600)
            if path.is_file():
                copied.append(str(path.relative_to(destination)))
    if not copied:
        raise RuntimeError("producer emitted no per-execution manifests/content blobs")
    return sorted(copied)


def _walk_mappings(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _compile_stats(result: dict[str, object]) -> dict[str, object]:
    for mapping in _walk_mappings(result):
        candidate = mapping.get("torch_compile")
        if isinstance(candidate, dict) and isinstance(
            candidate.get("persistent_cache"), dict
        ):
            return candidate
    for mapping in _walk_mappings(result):
        if "compile_abi" in mapping and isinstance(
            mapping.get("persistent_cache"), dict
        ):
            return mapping
    raise RuntimeError("benchmark JSON does not contain torch_compile stats")


def _child_environment(cache_root: Path, libdevice: Path, gpu: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment[BOOTSTRAP_ENV] = BOOTSTRAP_TOKEN
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root)
    environment["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    environment["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    environment["TRITON_LIBDEVICE_PATH"] = str(libdevice)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    repo = str(Path(__file__).resolve().parents[1])
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        repo if not inherited_pythonpath else repo + os.pathsep + inherited_pythonpath
    )
    environment.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    return environment


def _phase_summary(
    name: str,
    result: dict[str, object],
    *,
    outer_wall_ms: float,
    cache_root: Path,
) -> dict[str, object]:
    compile_stats = _compile_stats(result)
    persistent = compile_stats["persistent_cache"]
    assert isinstance(persistent, dict)
    entries = compile_stats.get("entries", [])
    first_call = [
        entry.get("first_call_wall_ms")
        for entry in entries
        if isinstance(entry, dict)
    ] if isinstance(entries, list) else []
    return {
        "phase": name,
        "outer_wall_ms": outer_wall_ms,
        "cache_root": str(cache_root),
        "artifact_load": {
            "loaded": persistent.get("artifact_loaded"),
            "load_cache_info": persistent.get("load_cache_info"),
            "cold_compile_reason": persistent.get("cold_compile_reason"),
            "quarantine_reason": persistent.get("quarantine_reason"),
        },
        "first_call_wall_ms": first_call,
        "capture_boundaries": [
            mapping
            for mapping in _walk_mappings(result)
            if any(
                key in mapping
                for key in ("capture_ms", "capture_count", "timed_capture_delta")
            )
        ],
        "kernel_or_full_policy_measurements": result.get("measurements"),
        "torch_compile": compile_stats,
        "child_result": result,
    }


def _validate_phase(
    phase: str,
    summary: dict[str, object],
    *,
    expect_loaded: bool,
) -> None:
    result = summary["child_result"]
    assert isinstance(result, dict)
    if result.get("status") != "pass":
        raise RuntimeError(f"{phase} benchmark status is not pass: {result.get('status')!r}")
    stats = summary["torch_compile"]
    assert isinstance(stats, dict)
    if stats.get("compile_abi") != "qwen25_vl_next_token_compile_v4":
        raise RuntimeError(f"{phase} did not use compile ABI v4")
    if int(stats.get("failures", -1)) != 0:
        raise RuntimeError(f"{phase} recorded compile failures")
    persistent = stats.get("persistent_cache")
    if not isinstance(persistent, dict):
        raise RuntimeError(f"{phase} lacks persistent cache stats")
    if not persistent.get("configured") or not persistent.get("launcher_contract_asserted"):
        raise RuntimeError(f"{phase} did not assert the pre-import launcher contract")
    if persistent.get("root") != summary["cache_root"]:
        raise RuntimeError(f"{phase} reported the wrong cache root")
    if persistent.get("fingerprint") != persistent.get("manifest_key"):
        raise RuntimeError(f"{phase} fingerprint/manifest key mismatch")
    if persistent.get("quarantine_reason") is not None:
        raise RuntimeError(f"{phase} quarantined an artifact")
    if not persistent.get("artifact_sha256") or int(persistent.get("artifact_bytes", 0)) <= 0:
        raise RuntimeError(f"{phase} has no validated content-addressed artifact")
    entry_stats = persistent.get("entry_stats")
    if not isinstance(entry_stats, dict) or not entry_stats:
        raise RuntimeError(f"{phase} has no per-execution manifest stats")
    for fingerprint, entry in entry_stats.items():
        if not isinstance(entry, dict) or entry.get("manifest_key") != fingerprint:
            raise RuntimeError(f"{phase} contains an invalid execution entry")
        if entry.get("quarantine_reason") is not None:
            raise RuntimeError(f"{phase} execution entry was quarantined")
        if expect_loaded:
            admission = entry.get("persistent_hit_admission", {})
            if not entry.get("artifact_loaded") or not isinstance(admission, dict) or not admission.get("admitted"):
                raise RuntimeError(f"{phase} did not admit an FX/AOT persistent hit")
            if not (entry.get("artifact_published") or entry.get("artifact_publish_skipped")):
                raise RuntimeError(f"{phase} hit neither published nor legally skipped publish")
        elif not entry.get("artifact_published") or entry.get("artifact_loaded"):
            raise RuntimeError(f"{phase} producer was not a cold published execution")


def _command(
    args: argparse.Namespace,
    *,
    scope: str,
    cache_root: Path,
    output: Path,
) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    script = (
        script_dir / "benchmark_cuda_graph.py"
        if scope == "kernel"
        else script_dir / "benchmark.py"
    )
    command = [
        args.python,
        str(script),
        "--profile",
        args.profile,
        "--checkpoint",
        str(args.checkpoint),
        "--manifest",
        str(args.manifest),
        "--data-root",
        str(args.data_root),
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        "cuda:0",
        "--expected-physical-gpu",
        args.gpu,
        "--attention-backend",
        "torch_sdpa",
        "--compile-backend",
        "inductor",
        "--compile-cache-dir",
        str(cache_root),
        "--compile-text-buckets",
        args.compile_text_buckets,
        "--output",
        str(output),
    ]
    return command


def _run_phase(
    args: argparse.Namespace,
    *,
    scope: str,
    phase: str,
    cache_root: Path,
    output: Path,
    expect_loaded: bool,
) -> dict[str, object]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        _command(args, scope=scope, cache_root=cache_root, output=output),
        cwd=Path(__file__).resolve().parents[1],
        env=_child_environment(cache_root, args.libdevice, args.gpu),
        check=False,
        text=True,
    )
    outer_wall_ms = (time.perf_counter_ns() - started) / 1e6
    if completed.returncode != 0:
        raise RuntimeError(f"{scope}/{phase} exited {completed.returncode}")
    result = json.loads(output.read_text(encoding="utf-8"))
    summary = _phase_summary(
        phase, result, outer_wall_ms=outer_wall_ms, cache_root=cache_root
    )
    _validate_phase(phase, summary, expect_loaded=expect_loaded)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--compile-text-buckets", required=True)
    parser.add_argument("--libdevice", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("kernel", "full-policy", "both"), default="both")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    if "torch" in sys.modules:
        raise RuntimeError("launcher contract violated before child environment setup")
    args = _parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.manifest = args.manifest.resolve(strict=True)
    args.data_root = args.data_root.resolve(strict=True)
    args.libdevice = args.libdevice.resolve(strict=True)
    if not args.libdevice.is_file():
        raise RuntimeError("--libdevice must name a regular file")
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise RuntimeError("--work-dir must not already exist")
    work_dir.mkdir(parents=True, mode=0o700)
    scopes = ("kernel", "full-policy") if args.scope == "both" else (args.scope,)
    report: dict[str, object] = {
        "schema": "qwen25_vl_compile_cache_launcher_v2",
        "bootstrap": {
            "token": f"{BOOTSTRAP_ENV}={BOOTSTRAP_TOKEN}",
            "child_environment": [
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCHINDUCTOR_FX_GRAPH_CACHE=1",
                "TORCHINDUCTOR_AUTOGRAD_CACHE=1",
                "TRITON_LIBDEVICE_PATH",
            ],
            "torch_imported_by_launcher": False,
        },
        "profile": args.profile,
        "compile_text_buckets": args.compile_text_buckets,
        "scopes": {},
    }
    for scope in scopes:
        scope_dir = work_dir / scope
        scope_dir.mkdir(mode=0o700)
        producer_root = scope_dir / "producer-cache"
        fresh_root = scope_dir / "fresh-mega-cache-only"
        _secure_directory(producer_root, empty=True)
        producer = _run_phase(
            args,
            scope=scope,
            phase="producer_empty_cache",
            cache_root=producer_root,
            output=scope_dir / "producer.json",
            expect_loaded=False,
        )
        same_consumer = _run_phase(
            args,
            scope=scope,
            phase="same_directory_consumer",
            cache_root=producer_root,
            output=scope_dir / "same-consumer.json",
            expect_loaded=True,
        )
        _secure_directory(fresh_root, empty=True)
        copied = _copy_mega_cache(producer_root, fresh_root)
        fresh_consumer = _run_phase(
            args,
            scope=scope,
            phase="fresh_directory_mega_cache_consumer",
            cache_root=fresh_root,
            output=scope_dir / "fresh-consumer.json",
            expect_loaded=True,
        )
        producer_fp = producer["torch_compile"]["persistent_cache"]["fingerprint"]
        same_fp = same_consumer["torch_compile"]["persistent_cache"]["fingerprint"]
        fresh_fp = fresh_consumer["torch_compile"]["persistent_cache"]["fingerprint"]
        if not producer_fp or producer_fp != same_fp or producer_fp != fresh_fp:
            raise RuntimeError(f"{scope} execution fingerprint changed across processes")
        report["scopes"][scope] = {
            "copied_mega_cache_artifacts": copied,
            "producer": producer,
            "same_directory_consumer": same_consumer,
            "fresh_directory_consumer": fresh_consumer,
        }
    report["status"] = "pass"
    _atomic_json(args.output.resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
