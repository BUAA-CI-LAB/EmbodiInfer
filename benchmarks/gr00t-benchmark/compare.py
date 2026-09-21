"""Compare GR00T N1.7 runtimes on the existing LIBERO-10 observation selection."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import benchmark as bench
import numpy as np
import torch
import yaml


def noise_for(index: int, dtype: torch.dtype) -> torch.Tensor:
    """Use identical BF16-rounded Gaussian values in every implementation."""
    values = np.random.default_rng(42 + index).standard_normal((1, 40, 132), dtype=np.float32)
    return torch.from_numpy(values).to(torch.bfloat16).to(dtype)


class Processor(bench.LiberoProcessor):
    """Retain the native LIBERO transforms and the processed camera views."""

    def __init__(self, checkpoint: Path, backbone: str, engine: str):
        super().__init__(checkpoint, backbone)
        self.cpp = engine in ("vlacpp", "embodied")

    def prepare(self, sample: Any, raw: dict) -> dict:
        """Prepare one serial request and retain its image-transform outputs."""
        self.images = []
        batch = self.prepare_cpp(sample, raw) if self.cpp else super().prepare(sample, raw)
        return {
            "batch": batch,
            "images": self.images,
            "instruction": sample.instruction,
            "physical_state": np.concatenate((raw["ee_pos"], raw["ee_ori"], raw["gripper_states"])).astype(
                np.float32
            ),
        }

    def prepare_cpp(self, sample: Any, raw: dict) -> Any:
        """Prepare state and tokens while leaving image patchification to C++."""
        from embodiinfer.policies.gr00t.processor_gr00t import Gr00tBatch

        values = dict(
            zip(
                ("x", "y", "z", "roll", "pitch", "yaw"),
                np.concatenate((raw["ee_pos"], raw["ee_ori"])),
                strict=True,
            )
        )
        values["gripper"] = raw["gripper_states"]
        normalized = []
        for key in self.modalities["state"]["modality_keys"]:
            value = np.atleast_1d(values[key]).astype(np.float32)
            stats = self.statistics["state"][key]
            low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
            valid = ~np.isclose(high, low)
            result = np.zeros_like(value)
            result[valid] = 2 * (value[valid] - low[valid]) / (high[valid] - low[valid]) - 1
            normalized.append(np.clip(result, -1, 1))
        state = torch.zeros(1, 1, self.settings["max_state_dim"], dtype=torch.float32)
        vector = np.concatenate(normalized)
        state[0, 0, : len(vector)] = torch.from_numpy(vector)
        images = [
            self.image(raw[key], sample.image_convention) for key in ("agentview_rgb", "eye_in_hand_rgb")
        ]
        if any(image.shape != (256, 256, 3) for image in images):
            raise ValueError("this C++ comparison profile requires two 256px views")
        instruction = re.sub(r"[^\w\s]", "", sample.instruction.lower())
        message = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": instruction}],
            }
        ]
        prompt = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        image_processor = self.processor.image_processor
        if (image_processor.patch_size, image_processor.merge_size) != (16, 2):
            raise ValueError("unexpected Qwen3-VL image geometry")
        if prompt.count(self.processor.image_token) != 2:
            raise ValueError("unexpected chat-template image placeholders")
        prompt = prompt.replace(self.processor.image_token, self.processor.image_token * 64)
        encoded = dict(self.processor.tokenizer([prompt], return_tensors="pt", padding=True))
        encoded["state"] = state
        return Gr00tBatch.from_backbone_inputs(
            encoded, torch.tensor([self.embodiment_id]), [sample.sample_id]
        )

    def validate_cpp(self, sample: Any, raw: dict, prepared: dict) -> None:
        """Require exact state, token and cropped-image parity before timing C++."""
        self.images = []
        native = super().prepare(sample, raw)
        for key, value in prepared["batch"].backbone_inputs.items():
            torch.testing.assert_close(value, native.backbone_inputs[key], rtol=0, atol=0)
        torch.testing.assert_close(prepared["batch"].state, native.state, rtol=0, atol=0)
        np.testing.assert_array_equal(np.stack(prepared["images"]), np.stack(self.images))

    def image(self, image: Any, convention: str) -> np.ndarray:
        """Reuse each deterministic image transform for the native C++ interface."""
        result = super().image(image, convention)
        self.images.append(result)
        return result


class EmbodiInfer:
    """Run the existing GR00T backbone, action head and four-step Euler loop."""

    def __init__(self, config: dict, processor: Processor, profiles: list[dict]):
        from embodiinfer.engine.config import EngineConfig
        from embodiinfer.engine.core import EngineCore
        from embodiinfer.policies import make_policy

        self.dtype = getattr(torch, config["dtype"])
        self.compile_backend = config["compile_backend"]
        self.policy = make_policy(
            "gr00t",
            checkpoint=config["checkpoint"],
            cosmos_path=config["cosmos_path"],
            attention=config["attention"],
            native_inference=config.get("native_inference", False),
            prefix_cuda_graph=config.get("prefix_cuda_graph", False),
        )
        if self.dtype == torch.float32:
            self.policy.to(dtype=torch.bfloat16)
        self.core = EngineCore(
            self.policy,
            EngineConfig(
                device="cuda:0",
                dtype=config["dtype"],
                max_batch_size=1,
                batch_buckets=(1,),
                use_cuda_graph=config["cuda_graph"],
                capture_full_loop=config["cuda_graph"],
                reuse_prefix_kv=True,
            ),
        )
        if config["compile_backend"] == "inductor":
            bench.compile_inference(self.policy, config, ("denoise_step",))
        elif config["compile_backend"] != "none":
            raise ValueError("unsupported compile backend")

    def predict(self, prepared: dict, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Time device-ready input through the complete model before output restoration."""
        batch = prepared["batch"].to("cuda:0", torch.bfloat16).to("cuda:0", self.dtype)
        noise = noise.to("cuda:0", self.dtype)
        prefix, normalized, timing = bench.timed_model(
            lambda: self.policy.encode_prefix(batch),
            lambda prefix: self.policy.decoder.integrate(noise, prefix, 4, 1, self.core._graphs),
            torch.device("cuda:0"),
        )
        return self.policy.finalize_actions(normalized, prefix)[0].float().cpu(), timing

    def runtime(self) -> dict:
        """Record graph capture and compiler counters for warmup stability checks."""
        return {
            **bench.runtime_details(self.core, self.policy),
            "compile_backend": self.compile_backend,
        }


class PhyAI:
    """Run PhyAI's native N1.7 scheduler using the checkpoint's exact model inputs."""

    def __init__(self, config: dict, processor: Processor, profiles: list[dict]):
        from phyai.engine import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.kernel.config import KernelConfig
        from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
        from phyai.models.gr00t_n17.scheduler_gr00t_n17 import GR00TN17Request

        self.request_class = GR00TN17Request
        self.dtype = getattr(torch, config["dtype"])
        requests = [self.request(p, noise_for(i, self.dtype)) for i, p in enumerate(profiles)]
        self.engine = Engine(
            EngineArgs(
                plugin="gr00t_n17",
                plugin_args=GR00TN17Args(
                    checkpoint_dir=config["checkpoint"], max_batch_size=1, capture_profiles=requests
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

    def request(self, prepared: dict, noise: torch.Tensor) -> Any:
        """Transfer the same backbone/state/embodiment tensors and supplied noise."""
        batch = prepared["batch"].to("cuda:0", torch.bfloat16).to("cuda:0", self.dtype)
        return self.request_class(
            tensors={**batch.backbone_inputs, "state": batch.state, "embodiment_id": batch.embodiment_id},
            noise=noise.to("cuda:0", self.dtype),
        )

    def predict(self, prepared: dict, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Measure all native backbone and denoising work with synchronized wall time."""
        request = self.request(prepared, noise)
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output = self.engine.step(request)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return output[0].float().cpu(), {"pure_inference_ms": elapsed}

    def runtime(self) -> dict:
        """Profile structure is captured by PhyAI at engine construction."""
        from phyai.kernel.bootstrap import get_kernel_selector

        selector = get_kernel_selector()
        return {
            "scheduler": "GR00TN17Scheduler",
            "runtime_capture": False,
            "kernel_profile": selector.policy.profile,
            "kernel_selections": sorted({item.kernel_id for item in selector._cache.values()}),
            "kernel_cache_entries": len(selector._cache),
            "autotuned_shapes": len(selector._autotune),
        }


class VlaCpp:
    """Run native C++ vision, language and action-head inference in one loaded model."""

    def __init__(self, config: dict, processor: Processor, profiles: list[dict]):
        sys.path.insert(0, str(Path(config["source"]) / "bindings/python"))
        os.environ["VLA_LIBRARY"] = config["library"]
        os.environ["VLA_GR00T_EMBODIMENT"] = str(processor.embodiment_id)
        import vla_cpp

        runtime = Path(config["output"]).with_suffix(".runtime.json")
        runtime.write_text(
            json.dumps(
                {
                    "runtime": {
                        "weight_dtype": "bf16",
                        "act_dtype": "f32",
                        "flash_attn": config["flash_attention"],
                        "mm_prec": "f32",
                    }
                }
            )
        )
        self.model = vla_cpp.load(config["gguf"], config_path=str(runtime))
        self.api = vla_cpp
        if (self.model.config.n_suffix, self.model.config.max_action_dim, self.model.config.num_steps) != (
            40,
            132,
            4,
        ):
            raise ValueError("C++ checkpoint geometry differs from GR00T N1.7")

    def predict(self, prepared: dict, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Use complete native prediction, retaining its distinct timing boundary."""
        batch = prepared["batch"]
        images = []
        for image in prepared["images"]:
            normalized = (torch.from_numpy(image.astype(np.float32)) / 255 * 2 - 1).to(torch.bfloat16).float()
            images.append(((normalized + 1) / 2).contiguous().numpy())
        inputs = batch.backbone_inputs
        tokens = inputs["input_ids"][inputs["attention_mask"].bool()].tolist()
        output = self.model.predict(
            images,
            tokens=tokens,
            state=batch.state.to(torch.bfloat16).float().numpy().reshape(-1),
            noise=noise.float().numpy().reshape(-1),
            pixel_format=self.api.PIXEL_F32_RGB_01,
            timing=self.api.TIMING_PHASE,
        )
        stats = self.model.last_stats()
        return torch.from_numpy(output), {
            "pure_inference_ms": None,
            "native_predict_ms": stats.ms_total,
            "native_vision_ms": stats.ms_vision,
            "native_lm_and_denoise_ms": stats.ms_inference,
        }

    def runtime(self) -> dict:
        """The public C ABI does not expose CUDA graph capture counts."""
        return {"graph_capture_count": None, "batch_support": [1]}


class Embodied:
    """Keep Embodied.cpp's native backbone, tokenizer and action head in one process."""

    def __init__(self, config: dict, processor: Processor, profiles: list[dict]):
        import ctypes as ct

        os.environ["VLA_GROOT_WEIGHT_DTYPE"] = "bf16"
        os.environ["VLA_GROOT_FLASH_ATTN"] = "1" if config["flash_attention"] else "0"
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
        self.handle = self.lib.benchmark_load(
            config["gguf"].encode(), config["mmproj"].encode(), config["backbone"].encode()
        )
        if not self.handle:
            raise RuntimeError(self.lib.benchmark_error().decode())
        self.output_horizon = processor.output_horizon
        keys = processor.modalities["action"]["modality_keys"]
        self.action_low = np.concatenate([processor.statistics["action"][key]["q01"] for key in keys])
        self.action_high = np.concatenate([processor.statistics["action"][key]["q99"] for key in keys])

    def predict(self, prepared: dict, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward the recorded views, normalized state and exact supplied noise."""
        import re

        batch = prepared["batch"]
        pixels = np.stack([image.astype(np.float32) / 255 for image in prepared["images"]])
        inputs = batch.backbone_inputs
        tokens = inputs["input_ids"][inputs["attention_mask"].bool()].numpy().astype(np.int32)
        # This native API normalizes physical state and denormalizes actions.
        # Supplying the already normalized VVLA state would normalize twice.
        state = np.pad(prepared["physical_state"], (0, 132 - len(prepared["physical_state"])))
        noise_array = noise.float().numpy().reshape(-1)
        instruction = re.sub(r"[^\w\s]", "", prepared["instruction"].lower())
        output, timing = np.empty((40, 132), np.float32), np.zeros(5, np.float32)

        def pointer(array: np.ndarray) -> Any:
            return array.ctypes.data_as(self.float_pointer)

        rc = self.lib.benchmark_predict(
            self.handle,
            pointer(pixels),
            2,
            pixels.shape[1],
            tokens.ctypes.data_as(self.int_pointer),
            len(tokens),
            instruction.encode(),
            pointer(state),
            pointer(noise_array),
            pointer(output),
            output.size,
            pointer(timing),
        )
        if rc != 0:
            raise RuntimeError(self.lib.benchmark_error().decode())
        return torch.from_numpy(output), {
            "pure_inference_ms": None,
            "native_predict_ms": float(timing[0]),
            "native_vision_ms": float(timing[1]),
            "native_lm_and_denoise_ms": float(timing[2]),
        }

    def restore(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply the public LIBERO horizon and clipping to native physical output."""
        values = actions.detach().float().cpu().numpy()[: self.output_horizon, :7]
        return torch.from_numpy(np.clip(values, self.action_low, self.action_high))

    def compared_actions(self, actions: torch.Tensor) -> np.ndarray:
        """Recover unclipped model columns outside the timed interval for parity."""
        values = actions.detach().float().cpu().numpy()[: self.output_horizon, :7]
        return (2 * (values - self.action_low) / (self.action_high - self.action_low) - 1).astype(np.float32)

    def runtime(self) -> dict:
        """Record that this upstream interface exposes only native phase timing."""
        return {"graph_capture_count": None, "batch_support": [1]}


def fixture_values(prepared: dict, noise: torch.Tensor) -> dict:
    """Serialize every public model input used for paired comparisons."""
    batch = prepared["batch"]
    return {
        **{key: value.cpu().numpy() for key, value in batch.backbone_inputs.items()},
        "state": batch.state.cpu().numpy(),
        "embodiment_id": batch.embodiment_id.cpu().numpy(),
        "noise": noise.float().cpu().numpy(),
        "cropped_images": np.stack(prepared["images"]),
    }


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
    """Export ten cross-task references, check actions, or measure all 1,600 observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("reference", "calibrate", "check", "measure"), required=True)
    parser.add_argument(
        "--allow-numerical-mismatch",
        action="store_true",
        help="measure despite a failed action check; preserve the failed verdict in the report",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.mode == "calibrate" and (config["engine"], config["dtype"]) != ("embodiinfer", "float32"):
        raise ValueError("calibration uses VVLA FP32 arithmetic on BF16-rounded weights/inputs")
    if config["batch_size"] != 1:
        raise ValueError("this runner currently requires batch_size=1")
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = bench.digest_json({k: v for k, v in config.items() if k not in ("output", "gate_report")})
    if args.mode == "measure":
        gate = measurement_gate(config, profile, args.allow_numerical_mismatch)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    samples = bench.load_libero(config["dataset"])
    reference_indices = bench.uniform_indices(len(samples), 10)
    indices = reference_indices if args.mode != "measure" else range(len(samples))
    processor = Processor(Path(config["checkpoint"]), config["cosmos_path"], config["engine"])
    prepared_profiles = [
        processor.prepare(samples[i], bench.read_libero(samples[i])) for i in reference_indices
    ]
    if processor.cpp:
        for index, prepared in zip(reference_indices, prepared_profiles, strict=True):
            processor.validate_cpp(samples[index], bench.read_libero(samples[index]), prepared)
    started = time.perf_counter()
    engine = {"embodiinfer": EmbodiInfer, "phyai": PhyAI, "vlacpp": VlaCpp, "embodied": Embodied}[config["engine"]](
        config, processor, prepared_profiles
    )
    load_seconds = time.perf_counter() - started
    reference = Path(config["reference_dir"])
    if args.mode == "reference":
        reference.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, config["dtype"])
    rows, checks = [], []
    with torch.inference_mode():
        for index, prepared in zip(reference_indices, prepared_profiles, strict=True):
            engine.predict(prepared, noise_for(index, dtype))
        before = engine.runtime()
        for index in indices:
            sample = samples[index]
            raw = bench.read_libero(sample)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            prepared = processor.prepare(sample, raw)
            noise = noise_for(index, dtype)
            normalized, timing = engine.predict(prepared, noise)
            physical = (
                (
                    engine.restore(normalized)
                    if isinstance(engine, Embodied)
                    else processor.restore(normalized)
                )
                .float()
                .cpu()
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if (
                normalized.shape != (40, 132)
                or physical.shape != (16, 7)
                or not torch.isfinite(normalized).all()
            ):
                raise ValueError(f"invalid GR00T output at {sample.sample_id}")
            fixture = reference / f"{index:04d}.npz"
            values = fixture_values(prepared, noise)
            compared = (
                engine.compared_actions(normalized)
                if isinstance(engine, Embodied)
                else normalized.float().cpu().numpy()[:16, :7]
            )
            if args.mode == "reference":
                if not isinstance(engine, EmbodiInfer):
                    raise ValueError("only VVLA generates the reference")
                np.savez(
                    fixture,
                    **values,
                    actions=physical.numpy(),
                    normalized=compared,
                    sample_id=sample.sample_id,
                )
            elif args.mode in ("check", "calibrate"):
                with np.load(fixture, allow_pickle=False) as ref:
                    if str(ref["sample_id"]) != sample.sample_id:
                        raise ValueError("reference sample identity differs")
                    for key, value in values.items():
                        np.testing.assert_array_equal(value, ref[key], err_msg=key)
                    difference = compared - ref["normalized"]
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
            rows.append({"sample_id": sample.sample_id, "e2e_ms": elapsed, **timing})
            if args.mode != "measure" or index % 25 == 0:
                print(
                    json.dumps({"index": index, "e2e_ms": elapsed, "check": checks[-1] if checks else None}),
                    flush=True,
                )
    after = engine.runtime()
    if isinstance(engine, (EmbodiInfer, PhyAI)) and before != after:
        raise RuntimeError("compilation/graph capture changed during measurement; extend warmup")
    e2e = np.asarray([row["e2e_ms"] for row in rows])
    pure = [row["pure_inference_ms"] for row in rows if row["pure_inference_ms"] is not None]
    report = {
        "schema": "rlinf_competitor_offline_v1",
        "model": "gr00t_n1.7",
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
        "environment": bench.provenance(torch.device("cuda:0")),
        "selection_sha256": bench.digest_json([s.sample_id for s in samples]),
        "calls": len(rows),
        "model_load_seconds": load_seconds,
        "runtime_before": before,
        "runtime_after": after,
        "mean_e2e_ms": float(e2e.mean()),
        "p95_e2e_ms": float(np.percentile(e2e, 95)),
        "e2e_observations_per_second": float(len(rows) * 1000 / e2e.sum()),
        "mean_pure_inference_ms": float(np.mean(pure)) if pure else None,
        "rows": rows,
    }
    if args.mode == "calibrate":
        report["tolerance_contract"] = {
            "reference": "VVLA BF16 vs FP32 with identical BF16-rounded weights/inputs/noise",
            "comparison": "first 16x7 normalized action columns before clipping",
            "factor": 2,
            "max_abs_tolerance": max(1e-4, 2 * max(c["max_abs"] for c in checks)),
            "rmse_tolerance": max(1e-5, 2 * max(c["rmse"] for c in checks)),
            "reason": "admit cross-kernel floating-point differences within twice native BF16/FP32 variation; no task-success claim",
        }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
