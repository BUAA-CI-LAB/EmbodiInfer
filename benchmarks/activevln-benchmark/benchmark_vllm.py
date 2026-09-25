"""Compare native vLLM ActiveVLN replay with the decoded-RGB EmbodiInfer boundary.

This optional benchmark runs in its own vLLM environment. It imports only data,
prompt and action helpers from EmbodiInfer, never its neural implementation.
The version pin makes the narrowly scoped instrumentation auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from benchmark import cuda_device, digest_json, load_navigation, provenance, read_rgb, summarize
from PIL import Image
from transformers import AutoProcessor

from embodiinfer.policies.activevln.prompt_activevln import (
    actions_to_tensor,
    parse_r2r_actions,
    render_turn_text,
)


def save(path: Path, value: dict[str, Any]) -> None:
    """Keep a complete diagnostic report if a later request fails."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def unique_update_rules(updates: dict[str, Any]) -> dict[str, list[Any]]:
    """Remove repeated references to the same rule, retaining first-match order.

    vLLM 0.8.5 repeats each matched rule once per historical image. Searching
    the identical object again at a nonmatching token cannot find a new match.
    Distinct objects remain distinct even if they compare equal.
    """
    return {
        modality: list({id(rule): rule for rule in rules}.values()) for modality, rules in updates.items()
    }


class ModelInterval:
    """Start after vision-input H2D, before the first neural operation.

    The end is synchronized after the entire generation loop. Host dispatch and
    per-token stopping checks remain included, as in EmbodiInfer timed_model.
    This is a complete generation wall interval, not one kernel or token latency.
    """

    def __init__(self, model: Any, device: torch.device) -> None:
        self.device = device
        original = model.get_multimodal_embeddings

        def measured_embeddings(**kwargs: Any) -> Any:
            if self.started_ns is None:
                # vLLM already transferred the scheduled image tensors here.
                if not kwargs or any(
                    t.device != device for t in kwargs.values() if isinstance(t, torch.Tensor)
                ):
                    raise RuntimeError("vision interval requires device-ready tensor inputs")
                torch.cuda.synchronize(device)
                self.started_ns = time.perf_counter_ns()
                self.events[0].record()
            self.vision_calls += 1
            return original(**kwargs)

        model.get_multimodal_embeddings = measured_embeddings
        self.reset()

    def reset(self) -> None:
        """Arm one request without retaining any previous timing or input state."""
        self.started_ns: int | None = None
        self.vision_calls = 0
        self.prefill_recorded = False
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    def first_token(self) -> None:
        """Record the end of vision/prefill when the first generated token arrives."""
        if not self.prefill_recorded:
            if self.started_ns is None:
                raise RuntimeError("no vision forward observed; timing would omit part of the model")
            self.events[1].record()
            self.prefill_recorded = True

    def finish(self) -> dict[str, float]:
        """Synchronize the complete vision/prefill/decode sequence exactly once."""
        if self.started_ns is None or not self.prefill_recorded:
            raise RuntimeError("incomplete model interval")
        self.events[2].record()
        self.events[2].synchronize()
        wall = (time.perf_counter_ns() - self.started_ns) / 1e6
        return {
            "pure_inference_ms": wall,
            "prefill_ms": self.events[0].elapsed_time(self.events[1]),
            "decode_ms": self.events[1].elapsed_time(self.events[2]),
            "gpu_inference_ms": self.events[0].elapsed_time(self.events[2]),
        }


class NativeVLLMReplay:
    """Use native vLLM scheduling, attention, graph execution and prefix caching."""

    def __init__(
        self, config: dict[str, Any], device: torch.device, *, deduplicate_placeholder_updates: bool = False
    ) -> None:
        from vllm import EngineArgs, LLMEngine, SamplingParams

        if importlib.metadata.version("vllm") != "0.8.5.post1":
            raise ValueError("this instrumentation requires vLLM 0.8.5.post1")
        if os.environ.get("VLLM_USE_V1") != "1" or os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise ValueError("require VLLM_USE_V1=1 and VLLM_ENABLE_V1_MULTIPROCESSING=0")
        self.config, self.device = config, device
        self.placeholder_rules_before = self.placeholder_rules_after = 0
        if deduplicate_placeholder_updates:
            from vllm.multimodal.processing import BaseMultiModalProcessor

            original_find = BaseMultiModalProcessor._find_mm_placeholders

            def find_unique(processor: Any, updates: Any, tokens: Any, counts: Any) -> Any:
                unique = unique_update_rules(updates)
                self.placeholder_rules_before += sum(map(len, updates.values()))
                self.placeholder_rules_after += sum(map(len, unique.values()))
                return original_find(processor, unique, tokens, counts)

            BaseMultiModalProcessor._find_mm_placeholders = find_unique
        self.processor = AutoProcessor.from_pretrained(
            config["checkpoint"], local_files_only=True, use_fast=False
        )
        self.tokenizer = self.processor.tokenizer
        self.engine_options = {
            "model": config["checkpoint"],
            "tokenizer": config["checkpoint"],
            "dtype": config["dtype"],
            "seed": config["seed"],
            "max_model_len": config["max_context"],
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 1024,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "enforce_eager": False,
            "gpu_memory_utilization": 0.85,
            "limit_mm_per_prompt": {"image": 200, "video": 0},
            "mm_processor_kwargs": {"min_pixels": 1024, "max_pixels": 76800},
            "disable_log_stats": True,
            "generation_config": "vllm",
            "compilation_config": {"level": 3},
        }
        self.engine = LLMEngine.from_engine_args(EngineArgs(**self.engine_options))
        self.core = self.engine.engine_core.engine_core
        self.runner = self.engine.model_executor.driver_worker.worker.model_runner
        self.timer = ModelInterval(self.runner.model, device)
        execute_model = self.runner.execute_model

        def observe_schedule(scheduled: Any, *args: Any, **kwargs: Any) -> Any:
            for request in scheduled.scheduled_new_reqs:
                self.prefix_cached_tokens = request.num_computed_tokens
            count = scheduled.total_num_scheduled_tokens
            self.forward_tokens += count
            if self.runner.use_cuda_graph and count <= self.runner.cudagraph_batch_sizes[-1]:
                self.graph_steps += 1
            else:
                self.non_graph_steps += 1
            return execute_model(scheduled, *args, **kwargs)

        self.runner.execute_model = observe_schedule
        self.sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=config["repetition_penalty"],
            max_tokens=config["max_new_tokens"],
            stop_token_ids=[151645, 151643],
            detokenize=False,
        )
        self.last_processed = None
        process_inputs = self.engine.processor.process_inputs

        def retain_request(*args: Any, **kwargs: Any) -> Any:
            result = process_inputs(*args, **kwargs)
            self.last_processed = result[1]
            return result

        self.engine.processor.process_inputs = retain_request
        self.call_index = 0
        self.history: list[int] = []
        self.images: list[Image.Image] = []

    def reset(self) -> None:
        """Clear episode history and all prefix/preprocessing caches, including warmup."""
        from vllm.multimodal import MULTIMODAL_REGISTRY

        if self.engine.has_unfinished_requests():
            raise RuntimeError("cannot reset a live vLLM request")
        self.engine.reset_prefix_cache()
        MULTIMODAL_REGISTRY._processing_cache._cache.clear()
        self.engine.processor.mm_input_cache_client.mm_cache.clear()
        self.core.mm_input_cache_server.mm_cache.clear()
        self.last_processed = None
        self.history, self.images = [], []

    def call(self, rgb: Any, instruction: str, *, audit_inputs: bool = False) -> dict[str, Any]:
        """Time CPU RGB through its own full-history generation and CPU actions."""
        self.timer.reset()
        self.prefix_cached_tokens = self.forward_tokens = self.graph_steps = self.non_graph_steps = 0
        self.placeholder_rules_before = self.placeholder_rules_after = 0
        torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        initial = not self.images
        turn_text = render_turn_text(self.processor, instruction, initial=initial)
        self.history.extend(self.tokenizer.encode(turn_text, add_special_tokens=False))
        self.images.append(Image.fromarray(rgb))
        identity = f"replay-{self.call_index}"
        self.call_index += 1
        self.engine.add_request(
            identity,
            {"prompt_token_ids": self.history.copy(), "multi_modal_data": {"image": self.images.copy()}},
            self.sampling,
        )
        added_ns = time.perf_counter_ns()
        tokens: list[int] = []
        stop_reason = "max_tokens"
        engine_steps = 0
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            engine_steps += 1
            if not outputs:
                continue
            if len(outputs) != 1 or outputs[0].request_id != identity:
                raise RuntimeError("batch-one request identity changed")
            completion = outputs[0].outputs[0]
            tokens = list(completion.token_ids)
            if not tokens:
                continue
            self.timer.first_token()
            if tokens[-1] in (151645, 151643):
                stop_reason = "eos"
                break
            partial = parse_r2r_actions(self.tokenizer.decode(tokens, skip_special_tokens=True).strip())
            if partial.valid and partial.actions[-1].name == "stop":
                stop_reason = "stop"
                self.engine.abort_request([identity])
                break
        timing = self.timer.finish()
        if self.engine.has_unfinished_requests():
            raise RuntimeError("request remained active after generation finished")
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        parsed = parse_r2r_actions(text)
        actions, mask = actions_to_tensor(parsed)
        self.history.extend(tokens)
        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        processed = self.last_processed
        row = {
            "latency_ms": elapsed_ms,
            "model_timing_ms": timing,
            "cpu_preparation_and_admission_ms": (added_ns - started) / 1e6,
            "token_ids": tokens,
            "generated_tokens": len(tokens),
            "text": text,
            "actions": actions.tolist(),
            "action_slots": actions.shape[0],
            "action_mask": mask.tolist(),
            "output_sha256": hashlib.sha256(actions.numpy().tobytes()).hexdigest(),
            "parsed_valid": parsed.valid,
            "stop_reason": stop_reason,
            "cache_length": len(processed.prompt_token_ids) + len(tokens),
            "prompt_tokens": len(processed.prompt_token_ids),
            "history_images": len(self.images),
            "vision_calls": self.timer.vision_calls,
            "engine_steps": engine_steps,
            "prefix_cached_tokens": self.prefix_cached_tokens,
            "forward_tokens": self.forward_tokens,
            "graph_steps": self.graph_steps,
            "non_graph_steps": self.non_graph_steps,
            "placeholder_rules_before": self.placeholder_rules_before,
            "placeholder_rules_after": self.placeholder_rules_after,
        }
        if audit_inputs:
            # Checks happen after timing and compare the current image and delta
            # tokens, not generated responses from a different numerical engine.
            expected = self.processor(
                text=[turn_text], images=[self.images[-1]], return_tensors="pt", padding=False
            )
            actual = processed.mm_inputs[-1]
            if actual is None:
                key = processed.mm_hashes[-1]
                # Keep both mirrored LRU orders identical even for repeated RGB.
                actual = self.engine.processor.mm_input_cache_client.mm_cache.get(key)
                self.core.mm_input_cache_server.mm_cache.get(key)
                if actual is None:
                    raise RuntimeError("current image missing from the mirrored processor cache")
            actual = actual.data
            row["input_audit"] = {
                "tokens_equal": processed.prompt_token_ids[-expected.input_ids.shape[1] :]
                == expected.input_ids[0].tolist(),
                "pixels_equal": torch.equal(actual["pixel_values"], expected.pixel_values),
                "grid_equal": torch.equal(actual["image_grid_thw"], expected.image_grid_thw),
                "grid": expected.image_grid_thw.tolist(),
            }
            if not all(row["input_audit"][key] for key in ("tokens_equal", "pixels_equal", "grid_equal")):
                raise ValueError(f"vLLM input preprocessing differs: {row['input_audit']}")
        return row


def run(
    config: dict[str, Any],
    output: Path,
    limit: int | None,
    *,
    deduplicate_placeholder_updates: bool = False,
    runtime_class: type = NativeVLLMReplay,
    runtime_options: dict[str, Any] | None = None,
) -> int:
    """Replay every frame without treating predicted STOP as the end of recorded data."""
    if len(config["datasets"]) != 1 or config["batch_size"] != 1 or config["do_sample"]:
        raise ValueError("one split, batch=1 and greedy decoding are required")
    episodes = load_navigation(config["datasets"][0])
    selected = [{"episode_id": ep.episode_id, "frames": len(ep.frames)} for ep in episodes]
    output.parent.mkdir(parents=True, exist_ok=True)
    device = cuda_device(config)
    torch.set_num_threads(config["cpu_threads"])
    report: dict[str, Any] = {
        "schema": "activevln.vllm_replay.v1",
        "status": "running",
        "phase": "loading",
        "reference_config": config,
        "dataset": config["datasets"][0]["name"],
        "selection": selected,
        "selection_sha256": digest_json(selected),
        "rows": [],
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": hashlib.sha256(
            Path(inspect.getfile(runtime_class)).read_bytes()
        ).hexdigest(),
        "deduplicate_placeholder_updates": deduplicate_placeholder_updates,
        "environment": {**provenance(device), "vllm": importlib.metadata.version("vllm")},
        "timing_boundary": {
            "e2e": "Decoded CPU RGB through native full-history input processing, prefix caching, complete generation and parsed CPU action tensor. No disk decoding, HTTP or simulator.",
            "forward": "Synchronized wall interval immediately before vision encoding after vision/text input H2D, through all prefill/decode and host token-stop checks; before final CPU action parsing. Includes scheduler/dispatch between decode steps.",
            "prefill": "CUDA elapsed vision start to first generated token; includes vision, text prefill and first-token sampling.",
            "decode": "CUDA elapsed first token to generation end, including host dispatch and stop checks.",
        },
        "accuracy_scope": "Fixed-frame replay with independently generated history; no navigation SR/SPL claim.",
        "history": "Own complete generated token IDs, including EOS when emitted; no truncation/reset within an episode. STOP ends generation but replay consumes all saved frames.",
    }
    runtime = None
    try:
        save(output, report)
        startup = time.monotonic()
        runtime = runtime_class(
            config,
            device,
            deduplicate_placeholder_updates=deduplicate_placeholder_updates,
            **(runtime_options or {}),
        )
        report["engine_options"] = runtime.engine_options
        report["resolved_compilation"] = str(runtime.engine.vllm_config.compilation_config)
        if hasattr(runtime, "optimization_evidence"):
            report["optimization_evidence"] = runtime.optimization_evidence()
        report["phase"] = "warmup"
        warmup = next(ep for ep in episodes if len(ep.frames) >= config["warmup_calls"])
        for frame in warmup.frames[: config["warmup_calls"]]:
            runtime.call(read_rgb(frame), warmup.instruction)
        runtime.reset()
        report["startup_seconds"] = time.monotonic() - startup
        report["phase"] = "measurement"
        torch.cuda.reset_peak_memory_stats(device)
        save(output, report)
        for ep in episodes:
            runtime.reset()
            for step, frame in enumerate(ep.frames):
                if limit is not None and step >= limit:
                    break
                report["inflight"] = {"episode_id": ep.episode_id, "step": step}
                row = runtime.call(read_rgb(frame), ep.instruction, audit_inputs=(step < 2))
                report["rows"].append({"episode_id": ep.episode_id, "step": step, **row})
            report["metrics"] = summarize(report["rows"])
            save(output, report)
            print(
                json.dumps(
                    {
                        "episode": ep.episode_id,
                        "calls": len(report["rows"]),
                        "e2e_ms": report["metrics"]["latency_ms"]["mean"],
                    }
                ),
                flush=True,
            )
        report.update(status="complete" if limit is None else "partial_probe", phase="finished")
        report.pop("inflight", None)
        code = 0
    except Exception as error:
        report.update(
            status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
            exception={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        code = 2
    finally:
        if report["rows"]:
            report["metrics"] = summarize(report["rows"])
        report["memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        if runtime is not None and hasattr(runtime, "optimization_evidence"):
            report["optimization_evidence"] = runtime.optimization_evidence()
        save(output, report)
        print(
            json.dumps({k: report[k] for k in ("status", "phase", "metrics", "exception") if k in report}),
            flush=True,
        )
        if runtime is not None:
            runtime.engine.engine_core.shutdown()
    return code


def main() -> None:
    """Read an unchanged EmbodiInfer replay config in the isolated vLLM interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument("--deduplicate-placeholder-updates", action="store_true")
    args = parser.parse_args()
    if args.max_steps_per_episode is not None and args.max_steps_per_episode < 1:
        parser.error("max-steps-per-episode must be positive")
    raise SystemExit(
        run(
            json.loads(args.config.read_text()),
            args.output,
            args.max_steps_per_episode,
            deduplicate_placeholder_updates=args.deduplicate_placeholder_updates,
        )
    )


if __name__ == "__main__":
    main()
