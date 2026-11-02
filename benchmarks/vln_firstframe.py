"""Replay 128 frozen first frames through the measured VLN tensor adapters."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def digest(path: str | Path) -> str:
    """Fingerprint a configuration, input or executable without modifying it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    """Atomically publish a finite JSON result inside the fresh output directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def module(name: str, path: Path) -> ModuleType:
    """Import the adapter from the selected inference source checkout."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main() -> None:
    """Warm and measure independent first frames, preserving historical results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--inputs",
        required=True,
        type=Path,
        help="Directory containing cache-manifest.json and the r2r/rxr PNG files",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="A new directory; existing reports are never overwritten"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Inference checkout supplying both model adapters",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = args.source.resolve()
    inputs = args.inputs.resolve()
    output = args.output.resolve()
    if config.get("model") not in ("activevln", "streamvln"):
        parser.error("config model must be activevln or streamvln")
    if type(config.get("batch_size")) is not int or config["batch_size"] not in (1, 2, 4, 8):
        parser.error("config batch_size must be 1, 2, 4 or 8")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config["allocator_config"]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def progress(stage, **values):
        record = dict(stage=stage, time=time.time(), elapsed_s=time.monotonic() - started, **values)
        save(output / "progress.json", record)
        print(json.dumps(record), flush=True)

    model = config["model"]
    benchdir = source / "benchmarks" / f"{model}-benchmark"
    sys.path[:0] = [str(source), str(benchdir), str(benchdir / "orin-nx")]
    import numpy as np
    import torch
    from PIL import Image

    from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest
    from embodiinfer.policies import make_policy

    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    dataset = config["evaluation_dataset"]
    selection = Path(config["navigation_manifest"])
    assert digest(selection) == config["navigation"]["selections"][dataset]["sha256"]
    cachefile = inputs / "cache-manifest.json"
    assert digest(cachefile) == config["firstframe"]["cache_manifest_sha256"]
    episodes = json.loads(selection.read_text())["episodes"]
    samples = json.loads(cachefile.read_text())[dataset]
    assert len(samples) == len(episodes) == 128
    assert len({str(s["episode_id"]) for s in samples}) == 128
    pngs, images = [], []
    for sample, episode in zip(samples, episodes, strict=True):
        assert str(sample["episode_id"]) == str(episode["episode_id"])
        assert sample["instruction"] == episode["instruction"]["instruction_text"]
        path = inputs / sample["image"]
        assert digest(path) == sample["image_sha256"]
        png = path.read_bytes()
        with Image.open(io.BytesIO(png)) as im:
            image = np.array(im.convert("RGB"))
        assert image.shape == (480, 640, 3)
        pngs.append(png)
        images.append(image)
    benchmark = module("benchmark", benchdir / "benchmark.py")
    serve = module("firstframe_serve", benchdir / "serve_batch.py")
    progress("loading", samples=len(samples))
    quantization = None
    batch = config["batch_size"]
    with torch.inference_mode():
        if model == "streamvln":
            policy = make_policy(model, **config["policy"]).to(device).eval()
            runtime = policy.create_batched_runtime(batch_size=batch, **config["batch_runtime"])
            adapter = serve.TensorBatchAdapter(policy, runtime, device, uint8=True)
        else:
            policy = (
                make_policy(
                    model,
                    checkpoint=config["checkpoint"],
                    revision=config["checkpoint_revision"],
                    attention=config["attention"],
                    max_new_tokens=config["max_new_tokens"],
                    max_context=config["max_context"],
                    do_sample=config["do_sample"],
                    repetition_penalty=config["repetition_penalty"],
                    action_space=config["action_space"],
                )
                .to(device=device, dtype=getattr(torch, config["dtype"]))
                .eval()
            )
            if config.get("calibration_path"):
                from calibrated_int8 import convert

                quantization = convert(policy, Path(config["calibration_path"]), config)
                assert quantization["layer_count"] == 204
                gc.collect()
                torch.cuda.empty_cache()
            keys = (
                "query_bucket_size",
                "cuda_graph",
                "fused_ops",
                "split_attention",
                "tree_decode",
                "tree_fp32_projection",
                "tree_repeat_actions",
                "kv_pool_tokens",
                "grammar_tree_budget",
                "grammar_prefix_weight",
                "grammar_local_weight",
                "grammar_refresh",
                "sampling_seed",
                "gpu_preprocess",
                "attention_kernel",
                "incremental_kv_commit",
                "resident_kv",
                "vision_kernel",
                "skip_padded_queries",
                "packed_prefill",
                "packed_prefill_bucket",
                "decode_linear",
            )
            runtime = policy.create_batched_runtime(
                batch_size=batch,
                workspace_tokens=config["graph_workspace_tokens"],
                **{key: config[key] for key in keys if key in config},
            )

            def observation(image, instruction):
                obs = benchmark.make_observation(image, instruction)
                obs.metadata["rgb_uint8"] = image
                return obs

            adapter = serve.TensorBatchAdapter(
                policy,
                runtime,
                SimpleNamespace(make_observation=observation, timed_model=benchmark.timed_model),
                device,
                output / "adapter-evidence.jsonl",
            )
        load_s = time.monotonic() - started
        progress("warming", load_s=load_s)

        def call(index, phase):
            selected = samples[index : index + batch]
            assert len(selected) == batch and not adapter.memories
            requests = []
            for row, sample in enumerate(selected):
                identity = f"{phase}:{dataset}:{sample['episode_id']}"
                requests.append(
                    RawPolicyRequest(
                        session_id=identity,
                        request_id=identity,
                        step_id=0,
                        instruction=sample["instruction"],
                        state={"benchmark_slot": row},
                        images=(
                            RawImage(
                                "observation.images.rgb" + (".0" if model == "streamvln" else ""),
                                "image/png",
                                pngs[index + row],
                            ),
                        ),
                        metadata={"stream_step_id": 0},
                    )
                )
            results = adapter.infer_batch(requests)
            assert len(results) == batch
            times = [float(result.timing["e2e_ms"]) for result in results]
            assert all(math.isfinite(t) and t > 0 and t == times[0] for t in times)
            records = [
                dict(
                    episode_id=s["episode_id"],
                    input_cache_length=0,
                    timing=dict(r.timing),
                    output=dict(r.actions[0].values),
                )
                for s, r in zip(selected, results, strict=True)
            ]
            for request in requests:
                adapter.reset(request.session_id)
            assert not adapter.memories
            return dict(index=index // batch, observations=batch, e2e_ms=times[0], rows=records)

        if model == "streamvln":
            with runtime.startup_capture():
                for index in range(0, 128, batch):
                    call(index, "warmup")
                    if index % (batch * 8) == 0:
                        progress("warming", warmed=index + batch)
            policy.clear_preprocessing_caches()
        else:
            probes = [observation(images[0], sample["instruction"]) for sample in samples]
            lengths = [policy._processor.process_turn(obs, initial=True).input_ids.shape[1] for obs in probes]
            max_context = max(lengths) + config["max_new_tokens"] + 32
            buckets = [2**i for i in range(9, math.ceil(math.log2(max_context)) + 1)]
            save(output / "warmup-shapes.json", dict(prompt_lengths=lengths, context_buckets=buckets))
            runtime.prewarm(probes, context_buckets=buckets)
            del probes
            for index in range(0, 128, batch):
                call(index, "warmup")
                if index % (batch * 8) == 0:
                    progress("warming", warmed=index + batch)
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        runtime.reset_stats()
        torch.cuda.reset_peak_memory_stats(device)
        warmup_s = time.monotonic() - started - load_s
        initial_stats = runtime.stats()
        save(
            output / "ready.json",
            dict(
                load_s=load_s,
                warmup_s=warmup_s,
                runtime=initial_stats,
                quantization=quantization,
                config_sha256=digest(args.config),
                driver_sha256=digest(__file__),
                adapter_sha256=digest(benchdir / "serve_batch.py"),
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                device=torch.cuda.get_device_name(device),
                capability=torch.cuda.get_device_capability(device),
            ),
        )
        measured = []
        measurement_start = time.monotonic()
        with (output / "batches.jsonl").open("w") as stream:
            for index in range(0, 128, batch):
                result = call(index, "measured")
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                measured.append(result)
                if index % (batch * 8) == 0:
                    progress("measuring", measured=index + batch)
        wall_s = time.monotonic() - measurement_start
        latencies = np.array([r["e2e_ms"] for r in measured])
        rows = [row for entry in measured for row in entry["rows"]]
        assert [row["episode_id"] for row in rows] == [sample["episode_id"] for sample in samples]
        summary = dict(
            status="complete",
            profile=config["profile"],
            protocol="first-frame-empty-history-v1",
            model=model,
            dataset=dataset,
            precision=config["precision"],
            batch_size=batch,
            episodes=128,
            frames_per_episode=1,
            model_calls=len(measured),
            batch_mean_ms=float(latencies.mean()),
            batch_p95_ms=float(np.percentile(latencies, 95)),
            amortized_per_image_ms=float(latencies.sum() / 128),
            images_per_second=128000 / float(latencies.sum()),
            model_total_s=float(latencies.sum() / 1000),
            measurement_wall_s=wall_s,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            output_tokens=sum(
                len(row["output"].get("tokens", row["output"].get("token_ids", []))) for row in rows
            ),
            load_s=load_s,
            warmup_s=warmup_s,
            runtime=runtime.stats(),
            source=str(source),
            reference_inference_commit=config["inference_commit"],
            inference_source_patch=config.get("inference_source_patch"),
            cache_manifest_sha256=digest(cachefile),
            selection_sha256=digest(selection),
            config_sha256=digest(args.config),
            driver_sha256=digest(__file__),
            accuracy_evaluated=False,
            timing_scope="Decoded CPU RGB through preprocessing, full action generation and CPU actions; excludes PNG decoding, disk, warmup, transport and simulator.",
            limitation="First frame with empty history; long-history KV reuse and history-feature reuse are not exercised.",
        )
        save(output / "summary.json", summary)
        progress("complete", batch_mean_ms=summary["batch_mean_ms"], batch_p95_ms=summary["batch_p95_ms"])


if __name__ == "__main__":
    main()
