"""Compare complete PI0.5 inference on the existing LIBERO-10 selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import benchmark as bench
import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer


class Processor:
    """The checkpoint's CPU MEAN_STD transforms, validated against native LeRobot."""

    def __init__(self, checkpoint: Path):
        self.config = json.loads((checkpoint / "config.json").read_text())
        pre = json.loads((checkpoint / "policy_preprocessor.json").read_text())["steps"]
        post = json.loads((checkpoint / "policy_postprocessor.json").read_text())["steps"]
        normalizer = next(s for s in pre if s["registry_name"] == "normalizer_processor")
        restorer = next(s for s in post if s["registry_name"] == "unnormalizer_processor")
        if any(normalizer["config"]["norm_map"][k] != "MEAN_STD" for k in ("STATE", "ACTION")):
            raise ValueError("expected this benchmark's MEAN_STD checkpoint")
        self.input_eps = normalizer["config"]["eps"]
        self.output_eps = restorer["config"]["eps"]
        self.stats = load_file(str(checkpoint / normalizer["state_file"]))
        self.action_stats = load_file(str(checkpoint / restorer["state_file"]))
        token_config = next(s["config"] for s in pre if s["registry_name"] == "tokenizer_processor")
        self.tokenizer = AutoTokenizer.from_pretrained(token_config["tokenizer_name"], local_files_only=True)
        self.tokenizer.padding_side = token_config["padding_side"]
        self.max_length = token_config["max_length"]
        if len(self.tokenizer) != 257152:
            raise ValueError("PI0.5 requires the verified full PaliGemma tokenizer")

    def prepare(self, sample: Any, raw: dict) -> dict:
        """Build deterministic images and state-bearing language tokens on the CPU."""
        state = torch.from_numpy(
            np.concatenate([raw[k] for k in ("ee_pos", "ee_ori", "gripper_states")]).astype(np.float32)
        )
        normalized = (state - self.stats["observation.state.mean"]) / (
            self.stats["observation.state.std"] + self.input_eps
        )
        bins = np.digitize(normalized.numpy(), bins=np.linspace(-1, 1, 257)[:-1]) - 1
        task = sample.instruction.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {task}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self.tokenizer(
            [prompt], padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        images = []
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            image = raw[key]
            if sample.image_convention == "opengl":
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array.transpose(2, 0, 1).copy())[None] * 2 - 1)
        return {
            "images": images,
            "tokens": tokens["input_ids"],
            "masks": tokens["attention_mask"].bool(),
            "state": state,
            "prompt": prompt,
        }

    def restore(self, normalized: torch.Tensor) -> torch.Tensor:
        """Return the same physical 50×7 action block as the native postprocessor."""
        value = normalized.detach().float().cpu()[..., :7]
        return value * (self.action_stats["action.std"] + self.output_eps) + self.action_stats["action.mean"]

    def normalize_actions(self, physical: torch.Tensor) -> torch.Tensor:
        """Map physical output back to the checkpoint's first seven model columns."""
        return (physical - self.action_stats["action.mean"]) / (
            self.action_stats["action.std"] + self.output_eps
        )


def noise_for(index: int, batch: int, dtype: torch.dtype) -> torch.Tensor:
    """Supply identical noise tensors across runtimes, independently of their RNGs."""
    noise = np.random.default_rng(42 + index).standard_normal((batch, 50, 32), dtype=np.float32)
    return torch.from_numpy(noise).to(torch.bfloat16).to(dtype)


class EmbodiInfer:
    """Execute the unchanged EmbodiInfer model and complete Euler loop."""

    def __init__(self, config: dict, processor: Processor):
        from embodiinfer.engine.config import EngineConfig
        from embodiinfer.engine.core import EngineCore
        from embodiinfer.policies import make_policy
        from embodiinfer.policies.pi05.processor_pi05 import make_processor

        self.config = config
        self.warmed_native_layouts = set()
        self.dtype = getattr(torch, config["dtype"])
        self.policy = make_policy(
            "pi05",
            checkpoint=config["checkpoint"],
            attention=config["attention"],
            compile_backend=config["compile_backend"],
            load_device="cuda:0",
            **{
                k: config[k]
                for k in ("native_inference", "prefix_cuda_graph", "denoise_attention", "prefix_attention")
                if k in config
            },
        )
        # FP32 calibration keeps exactly the BF16-rounded weights and inputs.
        if self.dtype == torch.float32:
            self.policy.to(dtype=torch.bfloat16)
        size = config["batch_size"]
        self.core = EngineCore(
            self.policy,
            EngineConfig(
                device="cuda:0",
                dtype=config["dtype"],
                max_batch_size=size,
                batch_buckets=(size,),
                use_cuda_graph=config["cuda_graph"],
                capture_full_loop=config["cuda_graph"],
            ),
        )
        self.native_processor = make_processor(self.policy, config["checkpoint"])
        self.processor = processor

    def validate_processor(self, sample: Any, raw: dict, prepared: dict) -> None:
        """Require exact CPU input parity with the original benchmark processor."""

        images = {}
        for key, target in (
            ("agentview_rgb", "observation.images.image"),
            ("eye_in_hand_rgb", "observation.images.image2"),
        ):
            image = raw[key]
            if sample.image_convention == "opengl":
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            images[target] = torch.from_numpy(self.native_processor.resize_image(image, 224, 224))
        native = self.native_processor.prepare(prepared["state"], images, sample.instruction)
        for actual, expected in zip(prepared["images"], native.images[:2], strict=True):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(prepared["tokens"], native.tokens, rtol=0, atol=0)
        torch.testing.assert_close(prepared["masks"], native.masks, rtol=0, atol=0)
        if any(bool(mask.any()) for mask in native.img_masks[2:]):
            raise ValueError("extra native cameras must be masked")
        if self.policy.native_inference and not self.warmed_native_layouts:
            # Include layouts absent from the ten action-check fixtures, e.g.
            # exactly 48 valid tokens (the dense-attention graph variant).
            for index, item in enumerate(bench.load_libero(self.config["dataset"])):
                inputs = self.processor.prepare(item, bench.read_libero(item))
                length = int(inputs["masks"].sum())
                bucket = next(
                    (n for n in (16, 32, 48, 64, 96, 128, 160, 200) if n >= length), inputs["tokens"].shape[1]
                )
                layout = (bucket, length == bucket)
                if layout not in self.warmed_native_layouts:
                    self.predict([inputs], noise_for(index, 1, self.dtype))
                    self.warmed_native_layouts.add(layout)
                    print(json.dumps({"warmup_native_layout": layout}), flush=True)

    def predict(self, prepared: list[dict], noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Time GPU-ready input through all model steps; output transforms follow."""
        from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

        images = [
            torch.cat([p["images"][i] for p in prepared]).to(torch.bfloat16).to(self.dtype) for i in range(2)
        ]
        size = len(prepared)
        batch = Pi05Batch(
            images + [-torch.ones_like(images[0])],
            [torch.ones(size, dtype=torch.bool)] * 2 + [torch.zeros(size, dtype=torch.bool)],
            torch.cat([p["tokens"] for p in prepared]),
            torch.cat([p["masks"] for p in prepared]),
        )
        batch = batch.to("cuda:0", self.dtype)
        noise = noise.to("cuda:0", self.dtype)
        prefix, normalized, timing = bench.timed_model(
            lambda: self.policy.encode_prefix(batch),
            lambda prefix: self.policy.decoder.integrate(noise, prefix, 10, size, self.core._graphs),
            torch.device("cuda:0"),
        )
        physical = self.processor.restore(self.policy.finalize_actions(normalized, prefix))
        return physical, timing

    def runtime(self) -> dict:
        """Expose observed graph capture and compilation counts."""
        stats = bench.engine_graph_stats(self.core)
        if self.policy.native_inference:
            stats["pi05_native"] = self.policy._runtime.stats()
            stats["warmed_native_layouts"] = sorted(self.warmed_native_layouts)
            if any(name == "flashinfer" or name.startswith("flashinfer.") for name in sys.modules):
                raise RuntimeError("native PI0.5 benchmark must not import FlashInfer")
        return stats


class VlaCpp:
    """Use vla.cpp's persistent in-process C ABI and native complete prediction."""

    def __init__(self, config: dict, processor: Processor):
        source = Path(config["source"])
        sys.path.insert(0, str(source / "bindings/python"))
        os.environ["VLA_LIBRARY"] = config["library"]
        import vla_cpp

        runtime = Path(config["output"]).with_suffix(".runtime.json")
        runtime.write_text(
            json.dumps(
                {
                    "runtime": {
                        "weight_dtype": "bf16",
                        "act_dtype": config["activation_dtype"],
                        "flash_attn": config["flash_attention"],
                        "mm_prec": "f32",
                    }
                }
            )
        )
        self.api = vla_cpp
        self.model = vla_cpp.load(config["gguf"], config_path=str(runtime))
        if (self.model.config.n_suffix, self.model.config.max_action_dim, self.model.config.num_steps) != (
            50,
            32,
            10,
        ):
            raise ValueError("C++ model geometry differs from the PI0.5 protocol")
        self.processor = processor
        self.dtype = getattr(torch, config["dtype"])

    def predict(self, prepared: list[dict], noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Include C++ preprocessing/transfers; do not mislabel its timer as pure model time."""
        if len(prepared) != 1:
            raise ValueError("this vla.cpp ABI supports batch one only")
        p = prepared[0]
        images = [
            ((image[0].to(self.dtype).float().permute(1, 2, 0) + 1) / 2).contiguous().numpy()
            for image in p["images"]
        ]
        tokens = p["tokens"][p["masks"]].tolist()
        actions = self.model.predict(
            images,
            tokens=tokens,
            noise=noise.float().numpy().reshape(-1),
            pixel_format=self.api.PIXEL_F32_RGB_01,
            timing=self.api.TIMING_PHASE,
        )
        native = self.model.last_stats()
        return torch.from_numpy(actions[:, :7])[None], {
            "pure_inference_ms": None,
            "native_predict_ms": native.ms_total,
            "native_vision_ms": native.ms_vision,
            "native_lm_and_denoise_ms": native.ms_inference,
        }

    def runtime(self) -> dict:
        """The public C ABI does not expose CUDA graph capture counters."""
        return {"graph_capture_count": None, "batch_support": [1]}


class PhyAI:
    """Run PhyAI's native vision/prefix/expert scheduler without a serving transport."""

    def __init__(self, config: dict, processor: Processor):
        from phyai.engine import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.kernel.config import KernelConfig
        from phyai.models.pi05.main_pi05 import PI05Args
        from phyai.models.pi05.scheduler_pi05 import PI05Request

        self.dtype = getattr(torch, config["dtype"])
        self.engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=config["canonical_checkpoint"],
                    max_batch_size=config["batch_size"],
                    inputs_image_shape=[[224, 224, 3]] * 2,
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=self.dtype),
                    kernel=KernelConfig(
                        profile=config.get("kernel_profile"),
                        autotune_cache=config.get("autotune_cache"),
                    ),
                    runtime=RuntimeConfig(
                        use_cuda_graph=config["cuda_graph"],
                        seed=42,
                        flashinfer_workspace_bytes=config.get(
                            "flashinfer_workspace_bytes", 256 * 1024 * 1024
                        ),
                    ),
                ),
            )
        )
        self.request = PI05Request
        self.processor = processor

    def predict(self, prepared: list[dict], noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Use the same active views/tokens/noise; pad cameras have no visible keys."""
        images = torch.stack([torch.cat([p["images"][i] for p in prepared]) for i in range(2)], dim=1)
        images = images.to(self.dtype).to("cuda:0")
        tokens = torch.cat([p["tokens"] for p in prepared]).to("cuda:0")
        lengths = torch.cat([p["masks"].sum(1) for p in prepared]).to("cuda:0")
        request = self.request(
            pixel_values=images, input_ids=tokens, lang_lens=lengths, noise=noise.float().to("cuda:0")
        )
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = self.engine.step(request)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        return self.processor.restore(output), {"pure_inference_ms": elapsed}

    def runtime(self) -> dict:
        """Record the scheduler type and the requested native CUDA graph setting."""
        from phyai.kernel.bootstrap import get_kernel_selector

        selector = get_kernel_selector()
        return {
            "scheduler": "PI05Scheduler",
            "kernel_profile": selector.policy.profile,
            "kernel_selections": sorted({item.kernel_id for item in selector._cache.values()}),
            "kernel_cache_entries": len(selector._cache),
            "autotuned_shapes": len(selector._autotune),
        }


class Embodied:
    """Call Embodied.cpp's complete native PI0.5 prediction through a small C bridge."""

    def __init__(self, config: dict, processor: Processor):
        import ctypes as ct

        self.ct = ct
        self.lib = ct.CDLL(config["library"])
        self.float_pointer = ct.POINTER(ct.c_float)
        self.int_pointer = ct.POINTER(ct.c_int32)
        self.lib.benchmark_load.argtypes = [ct.c_char_p] * 3
        self.lib.benchmark_load.restype = ct.c_void_p
        self.lib.benchmark_error.restype = ct.c_char_p
        self.lib.benchmark_predict.argtypes = [
            ct.c_void_p,
            self.float_pointer,
            ct.c_int,
            ct.c_int,
            self.int_pointer,
            ct.c_int,
            ct.c_char_p,
            self.float_pointer,
            self.float_pointer,
            self.float_pointer,
            ct.c_int64,
            self.float_pointer,
        ]
        self.lib.benchmark_predict.restype = ct.c_int
        self.handle = self.lib.benchmark_load(config["gguf"].encode(), config["mmproj"].encode(), b"")
        if not self.handle:
            raise RuntimeError(self.lib.benchmark_error().decode())
        self.dtype = getattr(torch, config["dtype"])

    def predict(self, prepared: list[dict], noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Keep native C++ transfers/postprocessing inside the E2E interval."""
        if len(prepared) != 1:
            raise ValueError("this Embodied.cpp interface supports batch one only")
        p = prepared[0]
        # np.stack preserves the CHW backing layout of transposed HWC views.
        # The public C++ ABI reads interleaved RGB from a contiguous pointer.
        pixels = np.ascontiguousarray(
            np.stack(
                [((im[0].to(self.dtype).float().permute(1, 2, 0) + 1) / 2).numpy() for im in p["images"]]
            ),
            dtype=np.float32,
        )
        tokens = p["tokens"][p["masks"]].numpy().astype(np.int32)
        state = np.pad(p["state"].numpy(), (0, 24))
        noise_array = noise.float().numpy().reshape(-1)
        output, timing = np.empty((50, 32), np.float32), np.zeros(5, np.float32)

        def pointer(array: np.ndarray) -> Any:
            return array.ctypes.data_as(self.float_pointer)

        rc = self.lib.benchmark_predict(
            self.handle,
            pointer(pixels),
            2,
            224,
            tokens.ctypes.data_as(self.int_pointer),
            len(tokens),
            p["prompt"].encode(),
            pointer(state),
            pointer(noise_array),
            pointer(output),
            output.size,
            pointer(timing),
        )
        if rc != 0:
            raise RuntimeError(self.lib.benchmark_error().decode())
        return torch.from_numpy(output[:, :7])[None], {
            "pure_inference_ms": None,
            "native_predict_ms": float(timing[0]),
            "native_vision_ms": float(timing[1]),
            "native_lm_and_denoise_ms": float(timing[2]),
        }

    def runtime(self) -> dict:
        """This upstream interface does not expose graph capture counts."""
        return {"graph_capture_count": None, "batch_support": [1]}


def measurement_gate(config: dict, profile: str, allow_numerical_mismatch: bool) -> dict:
    """Require a matching check; optionally measure while retaining its failed verdict."""
    gate = json.loads(Path(config["gate_report"]).read_text())
    if (
        gate["engine"] != config["engine"]
        or gate["profile_sha256"] != profile
        or gate["mode"] != "check"
        or len(gate["checks"]) != 10
    ):
        raise ValueError("measurement requires this exact profile's ten-observation action check")
    if not gate["passed"] and not allow_numerical_mismatch:
        raise ValueError("action check failed; performance-only runs require --allow-numerical-mismatch")
    return gate


def main() -> None:
    """Export/check paired outputs or measure the unchanged public dataset selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("reference", "calibrate", "check", "measure"), required=True)
    parser.add_argument(
        "--allow-numerical-mismatch",
        action="store_true",
        help="measure despite a failed action check; preserve the failed verdict in the report",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.mode == "calibrate" and (config["engine"], config["dtype"]) != ("embodiinfer", "float32"):
        raise ValueError("calibration uses EmbodiInfer FP32 arithmetic on BF16-rounded weights/inputs")
    if config["batch_size"] != 1:
        raise ValueError("this runner currently requires batch_size=1")
    profile = bench.digest_json({k: v for k, v in config.items() if k not in ("output", "gate_report")})
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    samples = bench.load_libero(config["dataset"])
    selection = bench.digest_json([s.sample_id for s in samples])
    reference_indices = bench.uniform_indices(len(samples), 10)
    indices = reference_indices if args.mode != "measure" else tuple(range(len(samples)))
    if args.mode == "measure":
        gate = measurement_gate(config, profile, args.allow_numerical_mismatch)
    processor = Processor(Path(config["checkpoint"]))
    started = time.perf_counter()
    engine = {"embodiinfer": EmbodiInfer, "vlacpp": VlaCpp, "phyai": PhyAI, "embodied": Embodied}[
        config["engine"]
    ](config, processor)
    load_seconds = time.perf_counter() - started
    reference_dir = Path(config["reference_dir"])
    if args.mode == "reference":
        reference_dir.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    with torch.inference_mode():
        for index in reference_indices:
            sample = samples[index]
            raw = bench.read_libero(sample)
            prepared = processor.prepare(sample, raw)
            if isinstance(engine, EmbodiInfer):
                engine.validate_processor(sample, raw, prepared)
            engine.predict([prepared], noise_for(index, 1, getattr(torch, config["dtype"])))
        before = engine.runtime()
        for index in indices:
            sample = samples[index]
            raw = bench.read_libero(sample)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            prepared = processor.prepare(sample, raw)
            noise = noise_for(index, 1, getattr(torch, config["dtype"]))
            physical, timing = engine.predict([prepared], noise)
            torch.cuda.synchronize()
            e2e_ms = (time.perf_counter_ns() - started) / 1e6
            physical = physical.detach().float().cpu()[0]
            if physical.shape != (50, 7) or not torch.isfinite(physical).all():
                raise ValueError(f"invalid action block at {sample.sample_id}")
            normalized = processor.normalize_actions(physical).numpy()
            fixture = reference_dir / f"{index:04d}.npz"
            if args.mode == "reference":
                if not isinstance(engine, EmbodiInfer):
                    raise ValueError("only EmbodiInfer generates the reference")
                native_post = engine.native_processor.restore_actions(
                    torch.from_numpy(normalized), prepared["state"]
                )
                torch.testing.assert_close(physical, native_post, rtol=1e-6, atol=1e-7)
                np.savez(
                    fixture,
                    actions=physical.numpy(),
                    normalized=normalized,
                    noise=noise.float().numpy(),
                    tokens=prepared["tokens"].numpy(),
                    masks=prepared["masks"].numpy(),
                    images=torch.stack(prepared["images"]).numpy(),
                    sample_id=sample.sample_id,
                )
            elif args.mode in ("check", "calibrate"):
                with np.load(fixture, allow_pickle=False) as ref:
                    if str(ref["sample_id"]) != sample.sample_id:
                        raise ValueError("reference sample identity differs")
                    for key, value in (
                        ("noise", noise.float().numpy()),
                        ("tokens", prepared["tokens"].numpy()),
                        ("masks", prepared["masks"].numpy()),
                        ("images", torch.stack(prepared["images"]).numpy()),
                    ):
                        np.testing.assert_array_equal(value, ref[key], err_msg=key)
                    difference = normalized - ref["normalized"]
                    check = {
                        "sample_id": sample.sample_id,
                        "max_abs": float(np.max(np.abs(difference))),
                        "rmse": float(np.sqrt(np.mean(difference**2))),
                        "physical_max_abs": float(np.max(np.abs(physical.numpy() - ref["actions"]))),
                    }
                    if args.mode == "check":
                        tolerance = config["sample_tolerances"][sample.sample_id]
                        check["passed"] = (
                            check["max_abs"] <= tolerance["max_abs"] and check["rmse"] <= tolerance["rmse"]
                        )
                        check["tolerance"] = tolerance
                    else:
                        check["passed"] = None
                    checks.append(check)
            rows.append({"sample_id": sample.sample_id, "e2e_ms": e2e_ms, **timing})
            if args.mode != "measure" or index % 25 == 0:
                print(
                    json.dumps({"index": index, "e2e_ms": e2e_ms, "check": checks[-1] if checks else None}),
                    flush=True,
                )
    after = engine.runtime()
    if isinstance(engine, (EmbodiInfer, PhyAI)) and before != after:
        raise RuntimeError("new CUDA graph capture occurred during measurement; extend warmup")
    e2e = np.array([row["e2e_ms"] for row in rows])
    pure = [row["pure_inference_ms"] for row in rows if row["pure_inference_ms"] is not None]
    report = {
        "schema": "rlinf_competitor_offline_v1",
        "model": "pi05",
        "engine": config["engine"],
        "mode": args.mode,
        "passed": gate["passed"]
        if args.mode == "measure"
        else args.mode != "calibrate" and all(c["passed"] for c in checks),
        "measurement_complete": args.mode == "measure" and len(rows) == len(samples),
        "allow_numerical_mismatch": args.allow_numerical_mismatch,
        "checks": checks,
        "config": config,
        "profile_sha256": profile,
        "selection_sha256": selection,
        "model_load_seconds": load_seconds,
        "runtime_before": before,
        "runtime_after": after,
        "calls": len(rows),
        "mean_e2e_ms": float(e2e.mean()),
        "p95_e2e_ms": float(np.percentile(e2e, 95)),
        "e2e_observations_per_second": float(len(rows) * 1000 / e2e.sum()),
        "mean_pure_inference_ms": float(np.mean(pure)) if pure else None,
        "environment": bench.provenance(torch.device("cuda:0")),
        "rows": rows,
    }
    if args.mode == "calibrate":
        report["tolerance_contract"] = {
            "reference": "EmbodiInfer BF16 vs FP32 with identical BF16-rounded weights/inputs/noise",
            "comparison": "first 50x7 normalized action columns",
            "factor": 2,
            "max_abs_tolerance": max(1e-4, 2 * max(c["max_abs"] for c in checks)),
            "rmse_tolerance": max(1e-5, 2 * max(c["rmse"] for c in checks)),
            "reason": "admit cross-kernel floating-point differences within twice native BF16/FP32 variation; no task-success claim",
        }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
