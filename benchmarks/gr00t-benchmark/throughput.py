"""Measure native GR00T N1.7 batching with equal-length task prompts."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import compare
import numpy as np
import torch
import yaml


def collate(prepared: list[dict], indices: list[int], size: int) -> tuple[Any, torch.Tensor]:
    """Batch complete views and equal-length token sequences without text padding."""
    from embodiinfer.policies.gr00t.processor_gr00t import Gr00tBatch

    batches = [item["batch"] for item in prepared]
    noises = [compare.noise_for(index, torch.bfloat16) for index in indices]
    while len(batches) < size:
        batches.append(batches[-1])
        noises.append(noises[-1])
    lengths = {batch.backbone_inputs["input_ids"].shape[1] for batch in batches}
    if len(lengths) != 1:
        raise ValueError("GR00T batching must not introduce new text padding semantics")
    inputs = {
        key: torch.cat([batch.backbone_inputs[key] for batch in batches])
        for key in batches[0].backbone_inputs
    }
    batch = Gr00tBatch(
        inputs,
        torch.cat([item.state for item in batches]),
        torch.cat([item.embodiment_id for item in batches]),
    )
    return batch, torch.cat(noises)


class EmbodiInfer(compare.EmbodiInfer):
    """Reuse the benchmark's optimized policy with the selected engine batch bucket."""

    def __init__(self, config: dict, processor: compare.Processor, profiles: list):
        from embodiinfer.engine.config import EngineConfig
        from embodiinfer.engine.core import EngineCore

        super().__init__(config, processor, [])
        size = config["batch_size"]
        self.core = EngineCore(
            self.policy,
            EngineConfig(
                device="cuda:0",
                dtype="bfloat16",
                max_batch_size=size,
                batch_buckets=(size,),
                use_cuda_graph=True,
                capture_full_loop=True,
                reuse_prefix_kv=True,
            ),
        )

    def predict_batch(self, batch: Any, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Execute every sample's backbone and all four denoising steps."""
        batch = batch.to("cuda:0", torch.bfloat16)
        noise = noise.to("cuda:0", torch.bfloat16)
        prefix, normalized, timing = compare.bench.timed_model(
            lambda: self.policy.encode_prefix(batch),
            lambda prefix: self.policy.decoder.integrate(
                noise, prefix, 4, batch.batch_size, self.core._graphs
            ),
            torch.device("cuda:0"),
        )
        return self.policy.finalize_actions(normalized, prefix).float().cpu(), timing


class PhyAI:
    """Capture native scheduler profiles at the requested batch size."""

    def __init__(self, config: dict, processor: compare.Processor, profiles: list):
        from phyai.engine import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.kernel.config import KernelConfig
        from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
        from phyai.models.gr00t_n17.scheduler_gr00t_n17 import GR00TN17Request

        self.request_class = GR00TN17Request
        requests = [self.request(batch, noise) for batch, noise in profiles]
        self.engine = Engine(
            EngineArgs(
                plugin="gr00t_n17",
                plugin_args=GR00TN17Args(
                    checkpoint_dir=config["checkpoint"],
                    max_batch_size=config["batch_size"],
                    capture_profiles=requests,
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                    kernel=KernelConfig(
                        profile=config.get("kernel_profile"),
                        autotune_cache=config.get("autotune_cache"),
                    ),
                    runtime=RuntimeConfig(
                        use_cuda_graph=True,
                        seed=42,
                        flashinfer_workspace_bytes=config.get(
                            "flashinfer_workspace_bytes", 256 * 1024 * 1024
                        ),
                    ),
                ),
            )
        )

    def request(self, batch: Any, noise: torch.Tensor) -> Any:
        """Transfer the same normalized inputs and explicit per-observation noise."""
        batch = batch.to("cuda:0", torch.bfloat16)
        return self.request_class(
            tensors={**batch.backbone_inputs, "state": batch.state, "embodiment_id": batch.embodiment_id},
            noise=noise.to("cuda:0", torch.bfloat16),
        )

    def predict_batch(self, batch: Any, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Time the complete native scheduler after preparing device inputs."""
        request = self.request(batch, noise)
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        normalized = self.engine.step(request)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return normalized.float().cpu(), {"pure_inference_ms": elapsed}

    def runtime(self) -> dict:
        """Native capture profiles are fixed before processing the dataset."""
        return compare.PhyAI.runtime(self)


def main() -> None:
    """Check B=1 references at the batch shape, then measure all 1,600 observations."""
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
        raise ValueError("throughput requires BF16, CUDA Graphs and a positive batch size")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {**config, "batch_size": args.batch_size, "output": str(args.output)}
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    samples = compare.bench.load_libero(config["dataset"])
    processor = compare.Processor(Path(config["checkpoint"]), config["cosmos_path"], config["engine"])
    task_groups = [
        list(indices)
        for _, indices in itertools.groupby(range(len(samples)), key=lambda i: samples[i].instruction)
    ]
    members = {index: group for group in task_groups for index in group}
    references = compare.bench.uniform_indices(len(samples), 10)
    check_groups = []
    for ordinal, index in enumerate(references):
        group = members[index]
        slot = ordinal % args.batch_size
        start = group.index(index) - slot
        check_groups.append(
            (index, slot, [group[(start + offset) % len(group)] for offset in range(args.batch_size)])
        )
    profiles = []
    for _, _, indices in check_groups:
        prepared = [processor.prepare(samples[i], compare.bench.read_libero(samples[i])) for i in indices]
        profiles.append(collate(prepared, indices, args.batch_size))
    started = time.perf_counter()
    engine = {"embodiinfer": EmbodiInfer, "phyai": PhyAI}[config["engine"]](config, processor, profiles)
    load_seconds = time.perf_counter() - started
    rows, checks = [], []
    with torch.inference_mode():
        for batch, noise in profiles:
            engine.predict_batch(batch, noise)
        before = engine.runtime()
        for (index, slot, _), (batch, noise) in zip(check_groups, profiles, strict=True):
            normalized, _ = engine.predict_batch(batch, noise)
            with np.load(Path(config["reference_dir"]) / f"{index:04d}.npz", allow_pickle=False) as ref:
                if str(ref["sample_id"]) != samples[index].sample_id:
                    raise ValueError("B=1 reference identity differs")
                np.testing.assert_array_equal(noise[slot : slot + 1].float().numpy(), ref["noise"])
                for key, value in batch.backbone_inputs.items():
                    rows_per_observation = ref[key].shape[0]
                    if value.shape[0] != args.batch_size * rows_per_observation:
                        raise ValueError(f"incorrect batched layout for {key}")
                    start = slot * rows_per_observation
                    np.testing.assert_array_equal(
                        value[start : start + rows_per_observation].numpy(), ref[key], err_msg=key
                    )
                np.testing.assert_array_equal(batch.state[slot : slot + 1].numpy(), ref["state"])
                np.testing.assert_array_equal(
                    batch.embodiment_id[slot : slot + 1].numpy(), ref["embodiment_id"]
                )
                difference = normalized[slot, :16, :7].numpy() - ref["normalized"]
            tolerance = config["sample_tolerances"][samples[index].sample_id]
            maximum, rmse = float(np.abs(difference).max()), float(np.sqrt(np.mean(difference**2)))
            checks.append(
                {
                    "sample_id": samples[index].sample_id,
                    "slot": slot,
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
            for group in task_groups:
                for start in range(0, len(group), args.batch_size):
                    indices = group[start : start + args.batch_size]
                    raw = [compare.bench.read_libero(samples[i]) for i in indices]
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    prepared = [
                        processor.prepare(samples[i], value) for i, value in zip(indices, raw, strict=True)
                    ]
                    batch, noise = collate(prepared, indices, args.batch_size)
                    normalized, timing = engine.predict_batch(batch, noise)
                    physical = torch.stack(
                        [processor.restore(row).float() for row in normalized[: len(indices)]]
                    )
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if physical.shape != (len(indices), 16, 7) or not torch.isfinite(physical).all():
                        raise ValueError("invalid batched physical action output")
                    rows.append(
                        {
                            "sample_ids": [samples[i].sample_id for i in indices],
                            "observations": len(indices),
                            "e2e_ms": elapsed,
                            **timing,
                        }
                    )
                    if len(rows) % 25 == 0:
                        print(json.dumps({"batches": len(rows), "e2e_ms": elapsed}), flush=True)
    after = engine.runtime()
    if isinstance(engine, (EmbodiInfer, PhyAI)) and before != after:
        raise RuntimeError("new graph/compilation during formal batching; extend warmup")
    count = sum(row["observations"] for row in rows)
    seconds = sum(row["e2e_ms"] for row in rows) / 1000
    pure_seconds = sum(row["pure_inference_ms"] for row in rows) / 1000
    report = {
        "schema": "rlinf_competitor_throughput_v1",
        "model": "gr00t_n1.7",
        "engine": config["engine"],
        "config": config,
        "batch_size": args.batch_size,
        "passed": passed,
        "measurement_complete": count == len(samples),
        "allow_numerical_mismatch": args.allow_numerical_mismatch,
        "checks": checks,
        "selection_sha256": compare.bench.digest_json([sample.sample_id for sample in samples]),
        "observations": count,
        "batches": len(rows),
        "model_load_seconds": load_seconds,
        "mean_batch_e2e_ms": 1000 * seconds / len(rows) if rows else None,
        "mean_batch_pure_inference_ms": 1000 * pure_seconds / len(rows) if rows else None,
        "e2e_observations_per_second": count / seconds if rows else None,
        "pure_observations_per_second": count / pure_seconds if rows else None,
        "timing_boundary": "decoded CPU observation batch to all CPU physical actions; excludes disk/decode and batch-formation waiting",
        "batching": "preserve dataset order; batch within task only; duplicate uncounted final slots",
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
