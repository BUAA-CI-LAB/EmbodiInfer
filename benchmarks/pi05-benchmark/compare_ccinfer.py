"""Validate the complete RLinf runtime against a frozen external ccinfer checkout.

The reference checkout is a measurement dependency only. Runtime imports in
EmbodiInfer never depend on it. Use recorded inputs from validate_optimizations.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from validate_optimizations import clear, digest, make_engine, native_batch, run, source_digest

from embodiinfer.policies.pi05 import MlpLayerPrecision, Pi05OptimizationConfig
from embodiinfer.policies.pi05.checkpoints.lerobot import load_lerobot_checkpoint


def tree_digest(root: Path) -> str:
    """Identify the frozen reference's Python and native source bytes."""
    result = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h"):
            result.update(path.relative_to(root).as_posix().encode() + b"\0")
            result.update(path.read_bytes())
    return result.hexdigest()


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Report raw equality and errors without changing the correctness threshold."""
    actual, expected = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(byte_equal=False, actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    equal = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    delta = (actual.float() - expected.float()).abs()
    return dict(
        byte_equal=equal,
        max_abs=float(delta.max()),
        relative_l2=float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ),
        unequal_elements=int((actual != expected).sum()),
    )


def reference_snapshot(provider, observation, noise: torch.Tensor) -> dict:
    """Record all retained prefix K/V and every step using original adapters."""
    call = provider.prepare_prepared(observation, noise=noise)
    binding = provider.bind(call)
    try:
        binding.activate()
        model = provider.policy
        schedule = model.prepare_schedule(1, num_steps=10)
        mask, cache = model.encode_prefix(observation, reuse_rotary=True)
        saved_cache = [(k.transpose(1, 2).cpu(), v.transpose(1, 2).cpu()) for k, v in cache]
        context = model.prepare_action_context(mask)
        actions, velocities = noise, []
        for timestep, modulation in zip(schedule.timesteps, schedule.modulations, strict=True):
            velocity = model.denoise_step(
                actions, timestep, mask, cache, modulations=modulation, context=context
            )
            velocities.append(velocity.cpu())
            actions = actions + (-1.0 / 10) * velocity
        return dict(mask=mask.cpu(), cache=saved_cache, velocities=velocities, actions=actions.cpu())
    finally:
        binding.close()


def candidate_snapshot(engine, batch, noise: torch.Tensor) -> dict:
    """Record the migrated path with graphs disabled for layer/step attribution."""
    policy = engine.policy
    previous = policy.prefix_cuda_graph
    policy.prefix_cuda_graph = False
    try:
        prefix = policy.encode_prefix(batch)
        saved_cache = [(k.cpu(), v.cpu()) for k, v in prefix.kv]
        actions, velocities = noise, []
        with policy._get_optimizations().decode_context(prefix):
            for timestep, dt, modulation in policy._runtime.schedule(noise, 10):
                velocity = policy._denoise_step_impl(actions, timestep, prefix, modulation)
                velocities.append(velocity.cpu())
                actions = actions + dt * velocity
        return dict(
            mask=prefix.prefix_pad_masks.cpu(),
            cache=saved_cache,
            velocities=velocities,
            actions=actions.cpu(),
        )
    finally:
        policy.prefix_cuda_graph = previous


def measure(function: Callable[[], torch.Tensor], warmup: int, iterations: int) -> dict:
    """Time warmed prepared-input requests; capture and CPU preprocessing excluded."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(iterations):
        start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start = time.perf_counter()
        start_event.record()
        function()
        end_event.record()
        end_event.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        gpu.append(start_event.elapsed_time(end_event))
    return dict(
        wall_median_ms=float(np.median(wall)),
        wall_p95_ms=float(np.percentile(wall, 95)),
        gpu_median_ms=float(np.median(gpu)),
        wall_samples_ms=wall,
        gpu_samples_ms=gpu,
    )


def forward_candidate(engine, batch, noise: torch.Tensor) -> torch.Tensor:
    """Use the engine's prefix and diffusion graphs without nested timers."""
    prefix = engine.policy.encode_prefix(batch)
    return engine.policy.decoder.integrate(noise, prefix, 10, 1, engine._graphs)


def install_reference_prefix(provider, choices: tuple[MlpLayerPrecision, ...]) -> None:
    """Wire original generic MLP adapters for ccinfer's experimental FP4 prefix.

    Its production recipe schema exposes action layers only. This measurement
    adapter selects prefix formats through the original MlpAdapter/NormQuantAdapter;
    no kernel, projection or scale calculation is replaced.
    """
    from ccinfer.policy.pi05.runtime import Pi05Runtime
    from ccinfer.policy.pi05.runtime_config import ActionLayerConfig
    from ccinfer.policy.pi05.runtime_mlp import MlpAdapter

    runtime = Pi05Runtime(provider.policy, provider.optimizations)
    provider._runtime = runtime
    for block, choice in zip(provider.policy.llm.layers, choices, strict=True):
        module = block.mlps[0]
        runtime.mlps[id(module)].restore()
        adapter = MlpAdapter(
            module,
            ActionLayerConfig(**asdict(choice)),
            runtime.config,
            runtime.quantizer,
            runtime.fusion,
            runtime.lookup.table,
            prefix=True,
        )
        runtime.mlps[id(module)] = adapter
        adapter.install()


@torch.inference_mode()
def main() -> None:
    """Compare original scales, operators and output bytes on held-out frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ccinfer-source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=("bf16", "fp8", "prefix_nvfp4", "combined"), default=["bf16", "fp8"]
    )
    parser.add_argument(
        "--prefix-calibration", type=Path, help="Frozen original ccinfer calibration report; no recalibration"
    )
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("warmup and iterations must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ccinfer_source.resolve()))
    from ccinfer import Runner, RunnerConfig
    from ccinfer.policy.pi05 import Observation, Pi05, Pi05Config, Pi05Provider, Pi05RuntimeConfig
    from ccinfer.policy.pi05.observation import IMAGE_KEYS

    torch.set_float32_matmul_precision("highest")
    fixture = torch.load(args.inputs, weights_only=False, map_location="cpu")
    indices = args.indices or fixture["validation_indices"]
    batches = {i: native_batch(fixture["observations"][i]) for i in indices}
    observations = {
        i: Observation(
            dict(
                zip(
                    IMAGE_KEYS,
                    [image.permute(0, 2, 3, 1).contiguous() for image in batch.images],
                    strict=True,
                )
            ),
            batch.tokens,
            batch.masks,
            dict(zip(IMAGE_KEYS, batch.img_masks, strict=True)),
        )
        for i, batch in batches.items()
    }
    noises = {
        i: torch.from_numpy(
            np.random.default_rng(1000 + i).standard_normal((1, 10, 32), dtype=np.float32)
        ).cuda()
        for i in indices
    }
    frozen = Pi05RuntimeConfig.from_json(args.recipe)
    migrated = Pi05OptimizationConfig.from_ccinfer_json(args.recipe)
    prefix_choices = ()
    if set(args.modes) & {"prefix_nvfp4", "combined"}:
        if args.prefix_calibration is None:
            parser.error("NVFP4 prefix modes require --prefix-calibration")
        calibration = json.loads(args.prefix_calibration.read_text())
        if (
            calibration["status"] != "complete"
            or calibration["checkpoint_sha256"] != frozen.checkpoint_sha256
        ):
            raise ValueError("Prefix calibration must be complete and match the frozen checkpoint")
        prefix_choices = tuple(
            MlpLayerPrecision("nvfp4", "nvfp4", *calibration["maxima"][f"llm.layers.{i}.mlps.0"])
            for i in range(18)
        )
    result = dict(
        hardware=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        matmul_precision="highest",
        checkpoint_sha256=digest(args.checkpoint / "model.safetensors"),
        inputs_sha256=digest(args.inputs),
        candidate_source_sha256=source_digest(),
        reference_source_sha256=tree_digest(args.ccinfer_source / "ccinfer"),
        original_recipe_sha256=digest(args.recipe),
        original_recipe=json.loads(args.recipe.read_text()),
        batch_size=1,
        horizon=10,
        steps=10,
        indices=indices,
        warmup=args.warmup,
        iterations=args.iterations,
        scope="Prepared GPU tensors through prefix and all ten Euler steps; CPU preprocessing, transfer and robot I/O excluded",
        correctness="Byte equality to matching ccinfer recipe; low precision quality versus BF16 is a separate measurement",
        modes={},
    )
    if args.prefix_calibration is not None:
        result["prefix_calibration_sha256"] = digest(args.prefix_calibration)
    print("Loading identical checkpoint for both implementations", flush=True)
    original = Pi05.from_pretrained(args.checkpoint, config=Pi05Config(action_horizon=10), device="cuda")
    native = load_lerobot_checkpoint(str(args.checkpoint), load_device="cuda", low_cpu_mem_usage=True)
    native._validation_checkpoint = args.checkpoint
    for mode in args.modes:
        original_config = frozen if mode in ("fp8", "combined") else replace(frozen, action_layers=())
        candidate_config = migrated if mode in ("fp8", "combined") else replace(migrated, action_layers=())
        if mode in ("prefix_nvfp4", "combined"):
            candidate_config = replace(candidate_config, prefix_layers=prefix_choices)
        provider = Pi05Provider(original, optimizations=original_config)
        if candidate_config.prefix_layers:
            install_reference_prefix(provider, candidate_config.prefix_layers)
        engine = make_engine(native, 10, 1, candidate_config)
        engine.policy.prefix_cuda_graph = True
        comparisons, reference_outputs, candidate_outputs = [], {}, {}
        print(f"Comparing {mode} prefix K/V and ten velocities", flush=True)
        try:
            first_index = indices[0]
            reference_vision = original.img(
                torch.cat([observations[first_index].images[key] for key in IMAGE_KEYS[:2]])
            )
            candidate_vision = engine.policy._embed_image(torch.cat(batches[first_index].images[:2]))
            components = dict(
                vision=difference(candidate_vision, reference_vision),
                conditioning=[
                    difference(engine.policy._time_condition(t), original._time_condition(t))
                    for t in original.prepare_schedule(1).timesteps
                ],
            )
            print(json.dumps(dict(mode=mode, components=components)), flush=True)
            for index in indices:
                expected = reference_snapshot(provider, observations[index], noises[index])
                actual = candidate_snapshot(engine, batches[index], noises[index])
                reference_outputs[index], candidate_outputs[index] = expected, actual
                report = dict(
                    index=index,
                    mask=difference(actual["mask"], expected["mask"]),
                    cache=[
                        dict(key=difference(a[0], e[0]), value=difference(a[1], e[1]))
                        for a, e in zip(actual["cache"], expected["cache"], strict=True)
                    ],
                    velocities=[
                        difference(a, e)
                        for a, e in zip(actual["velocities"], expected["velocities"], strict=True)
                    ],
                    actions=difference(actual["actions"], expected["actions"]),
                )
                comparisons.append(report)
                print(json.dumps(dict(mode=mode, index=index, actions=report["actions"])), flush=True)
            runner = Runner(provider, RunnerConfig(cuda_graph=True, warmup=args.warmup, cache_size=8))
            graph_comparisons = []
            for index in indices:
                expected = reference_outputs[index]["actions"]
                old_graph = runner.run_prepared(observations[index], noise=noises[index])
                new_graph, _ = run(engine, batches[index], noises[index])
                graph_comparisons.append(
                    dict(
                        index=index,
                        reference_graph_vs_eager=difference(old_graph, expected),
                        candidate_graph_vs_eager=difference(new_graph, candidate_outputs[index]["actions"]),
                        candidate_graph_vs_reference=difference(new_graph, old_graph),
                    )
                )
            index = indices[0]
            changed_noise = noises[index].neg()
            changed_expected = reference_snapshot(provider, observations[index], changed_noise)
            # Binding a diagnostic snapshot restores mask adapters afterwards;
            # cached CUDA Graphs still own their independent static storage.
            changed_old = runner.run_prepared(observations[index], noise=changed_noise)
            changed_new = forward_candidate(engine, batches[index], changed_noise)
            changed_noise_check = dict(
                reference_graph_vs_eager=difference(changed_old, changed_expected["actions"]),
                candidate_graph_vs_reference=difference(changed_new, changed_old),
            )
            if torch.equal(changed_expected["actions"], reference_outputs[index]["actions"]):
                raise AssertionError("Changed-noise control did not change the output")
            old_time = measure(
                partial(runner.run_prepared, observations[index], noise=noises[index]),
                args.warmup,
                args.iterations,
            )

            new_time = measure(
                partial(forward_candidate, engine, batches[index], noises[index]),
                args.warmup,
                args.iterations,
            )
            runner.close()
            result["modes"][mode] = dict(
                config=asdict(candidate_config),
                components=components,
                comparisons=comparisons,
                graph_comparisons=graph_comparisons,
                changed_noise=changed_noise_check,
                reference_latency=old_time,
                candidate_latency=new_time,
                execution="ccinfer full-request CUDA Graph; EmbodiInfer prefix plus full-diffusion CUDA Graphs",
            )
            print(
                json.dumps(
                    dict(
                        mode=mode,
                        reference_ms=old_time["wall_median_ms"],
                        candidate_ms=new_time["wall_median_ms"],
                    )
                ),
                flush=True,
            )
        finally:
            provider.close()
            clear(engine)
        torch.save(
            dict(reference=reference_outputs, candidate=candidate_outputs), args.out / f"{mode}-tensors.pt"
        )
        (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    checks = [
        item
        for mode in result["modes"].values()
        for row in mode["comparisons"]
        for item in [
            row["mask"],
            row["actions"],
            *row["velocities"],
            *[tensor for kv in row["cache"] for tensor in kv.values()],
        ]
    ]
    checks.extend(
        item
        for mode in result["modes"].values()
        for row in mode["graph_comparisons"]
        for key, item in row.items()
        if key != "index"
    )
    checks.extend(item for mode in result["modes"].values() for item in mode["changed_noise"].values())
    checks.extend(
        item
        for mode in result["modes"].values()
        for item in [mode["components"]["vision"], *mode["components"]["conditioning"]]
    )
    result["all_byte_equal"] = all(item["byte_equal"] for item in checks)
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_byte_equal"]:
        raise SystemExit("Migration differs from ccinfer; inspect recorded prefix/step attribution")


if __name__ == "__main__":
    main()
