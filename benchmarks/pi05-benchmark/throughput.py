"""Measure native PI0.5 batching after checking complete actions against B=1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import compare
import numpy as np
import torch
import yaml


def groups(indices: list[int], size: int) -> list[list[int]]:
    """Keep the recorded observation order and form full or final partial batches."""
    return [indices[start : start + size] for start in range(0, len(indices), size)]


def prepare_batch(
    processor: compare.Processor, samples: list, indices: list[int], raw: list, size: int
) -> tuple:
    """Prepare independent observations and pad only the uncounted final slots."""
    prepared = [processor.prepare(samples[index], value) for index, value in zip(indices, raw, strict=True)]
    noises = [compare.noise_for(index, 1, torch.bfloat16) for index in indices]
    while len(prepared) < size:
        prepared.append(prepared[-1])
        noises.append(noises[-1])
    return prepared, torch.cat(noises)


def main() -> None:
    """Run one batch size; numerical mismatches require explicit performance-only opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-numerical-mismatch",
        action="store_true",
        help="measure despite numerical mismatches; still check input parity and finite actions",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["engine"] not in ("embodiinfer", "phyai"):
        raise ValueError("the public C++ prediction APIs expose only B=1")
    if config["dtype"] != "bfloat16" or not config["cuda_graph"] or args.batch_size < 1:
        raise ValueError("this throughput profile requires BF16, CUDA Graphs and a positive batch size")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {**config, "batch_size": args.batch_size, "output": str(args.output)}
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    samples = compare.bench.load_libero(config["dataset"])
    processor = compare.Processor(Path(config["checkpoint"]))
    started = time.perf_counter()
    engine = {"embodiinfer": compare.EmbodiInfer, "phyai": compare.PhyAI}[config["engine"]](config, processor)
    load_seconds = time.perf_counter() - started
    reference_indices = list(compare.bench.uniform_indices(len(samples), 10))
    check_batches = groups(reference_indices, args.batch_size)
    warm_inputs = [
        prepare_batch(
            processor,
            samples,
            indices,
            [compare.bench.read_libero(samples[i]) for i in indices],
            args.batch_size,
        )
        for indices in check_batches
    ]
    checks, rows = [], []
    with torch.inference_mode():
        for iteration in range(max(10, len(warm_inputs))):
            prepared, noise = warm_inputs[iteration % len(warm_inputs)]
            engine.predict(prepared, noise)
        before = engine.runtime()
        for indices, (prepared, noise) in zip(check_batches, warm_inputs, strict=True):
            physical, _ = engine.predict(prepared, noise)
            normalized = processor.normalize_actions(physical).float().cpu().numpy()
            for slot, index in enumerate(indices):
                with np.load(Path(config["reference_dir"]) / f"{index:04d}.npz", allow_pickle=False) as ref:
                    if str(ref["sample_id"]) != samples[index].sample_id:
                        raise ValueError("B=1 reference sample identity differs")
                    np.testing.assert_array_equal(noise[slot : slot + 1].float().numpy(), ref["noise"])
                    for key in ("tokens", "masks"):
                        np.testing.assert_array_equal(prepared[slot][key].numpy(), ref[key])
                    np.testing.assert_array_equal(
                        torch.stack(prepared[slot]["images"]).numpy(), ref["images"]
                    )
                    difference = normalized[slot] - ref["normalized"]
                tolerance = config["sample_tolerances"][samples[index].sample_id]
                maximum, rmse = float(np.abs(difference).max()), float(np.sqrt(np.mean(difference**2)))
                checks.append(
                    {
                        "sample_id": samples[index].sample_id,
                        "max_abs": maximum,
                        "rmse": rmse,
                        "tolerance": tolerance,
                        "passed": bool(
                            np.isfinite(difference).all()
                            and maximum <= tolerance["max_abs"]
                            and rmse <= tolerance["rmse"]
                        ),
                    }
                )
        passed = all(check["passed"] for check in checks)
        print(
            json.dumps({"batch_size": args.batch_size, "action_check_passed": passed, "checks": checks}),
            flush=True,
        )
        if passed or args.allow_numerical_mismatch:
            for batch_number, indices in enumerate(groups(list(range(len(samples))), args.batch_size)):
                raw = [compare.bench.read_libero(samples[index]) for index in indices]
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                prepared, noise = prepare_batch(processor, samples, indices, raw, args.batch_size)
                physical, timing = engine.predict(prepared, noise)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if physical.shape != (args.batch_size, 50, 7) or not torch.isfinite(physical).all():
                    raise ValueError("invalid batched action output")
                rows.append(
                    {
                        "sample_ids": [samples[i].sample_id for i in indices],
                        "observations": len(indices),
                        "e2e_ms": elapsed,
                        **timing,
                    }
                )
                if batch_number % 25 == 0:
                    print(json.dumps({"batch": batch_number, "e2e_ms": elapsed}), flush=True)
    after = engine.runtime()
    if isinstance(engine, (compare.Vvla, compare.PhyAI)) and before != after:
        raise RuntimeError("new graph capture during formal batching; extend warmup")
    observation_count = sum(row["observations"] for row in rows)
    seconds = sum(row["e2e_ms"] for row in rows) / 1000
    pure_seconds = sum(row["pure_inference_ms"] for row in rows) / 1000
    report = {
        "schema": "rlinf_competitor_throughput_v1",
        "model": "pi05",
        "engine": config["engine"],
        "config": config,
        "batch_size": args.batch_size,
        "passed": passed,
        "measurement_complete": observation_count == len(samples),
        "allow_numerical_mismatch": args.allow_numerical_mismatch,
        "checks": checks,
        "selection_sha256": compare.bench.digest_json([sample.sample_id for sample in samples]),
        "observations": observation_count,
        "batches": len(rows),
        "model_load_seconds": load_seconds,
        "mean_batch_e2e_ms": 1000 * seconds / len(rows) if rows else None,
        "mean_batch_pure_inference_ms": 1000 * pure_seconds / len(rows) if rows else None,
        "e2e_observations_per_second": observation_count / seconds if rows else None,
        "pure_observations_per_second": observation_count / pure_seconds if rows else None,
        "timing_boundary": "batch of decoded CPU observations to all CPU physical action chunks; excludes disk/decode and batch-formation waiting",
        "padding": "final batch duplicates its last observation; padded slots are not counted",
        "runtime_before": before,
        "runtime_after": after,
        "environment": compare.bench.provenance(torch.device("cuda:0")),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed and not args.allow_numerical_mismatch:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
