"""Measure ActiveVLN with the pinned modern native vLLM optimization stack.

Run in an isolated vLLM 0.30.0 environment. Neural execution, CUDA graphs,
compilation, cache management and speculation remain owned by native vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from benchmark_vllm import ModelInterval, run
from PIL import Image
from transformers import AutoProcessor

from embodiinfer.policies.activevln.prompt_activevln import (
    actions_to_tensor,
    parse_r2r_actions,
    render_turn_text,
)


def stopping_prefix(tokenizer: Any, tokens: list[int], checked: int) -> tuple[list[int], str | None]:
    """Keep the first EOS or complete STOP, including inside a speculative block.

    Tokens after that position may have been verified in the same native model
    invocation. Their compute is timed, but they must not enter episode history.
    """
    for end in range(checked + 1, len(tokens) + 1):
        if tokens[end - 1] in (151645, 151643):
            return tokens[:end], "eos"
        parsed = parse_r2r_actions(tokenizer.decode(tokens[:end], skip_special_tokens=True).strip())
        if parsed.valid and parsed.actions[-1].name == "stop":
            return tokens[:end], "stop"
    return tokens, None


def drain_inflight(engine: Any, core: Any) -> int:
    """Settle async work before ending timing or reusing the next frame's state."""
    steps = 0
    while core.batch_queue:
        if engine.step():
            raise RuntimeError("unexpected public output after the request was stopped")
        steps += 1
    return steps


class EncoderInterval(ModelInterval):
    """Time the native encoder graph after image H2D through complete generation."""

    def __init__(self, manager: Any, device: torch.device) -> None:
        self.device = device
        if manager is None:
            raise RuntimeError("requested vision CUDA graphs did not initialize")
        original = manager.execute

        def measured(mm_kwargs: dict[str, Any]) -> Any:
            pixels = mm_kwargs.get("pixel_values")
            if not isinstance(pixels, torch.Tensor) or pixels.device != device:
                raise RuntimeError("vision graph timing requires device-ready pixels")
            if self.started_ns is None:
                torch.cuda.synchronize(device)
                self.started_ns = time.perf_counter_ns()
                self.events[0].record()
            self.vision_calls += 1
            return original(mm_kwargs)

        manager.execute = measured
        self.reset()


class ModernVLLMReplay:
    """Own full episode histories while using official vision/text optimizations."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        *,
        deduplicate_placeholder_updates: bool = False,
        speculation: str = "ngram_gpu",
        speculative_tokens: int = 16,
        compile_vision: bool = True,
        async_scheduling: bool | None = None,
    ) -> None:
        from vllm import EngineArgs, LLMEngine, SamplingParams

        if importlib.metadata.version("vllm") != "0.30.0":
            raise ValueError("modern instrumentation requires vLLM 0.30.0")
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise ValueError("same-boundary timing requires in-process native vLLM")
        if deduplicate_placeholder_updates:
            raise ValueError("the 0.8.5 frontend patch must not be applied to modern vLLM")
        self.config, self.device = config, device
        self.processor = AutoProcessor.from_pretrained(
            config["checkpoint"], local_files_only=True, use_fast=False
        )
        self.tokenizer = self.processor.tokenizer
        self.speculation = speculation
        batch_size = config.get("batch_size", 1)
        if type(batch_size) is not int or batch_size not in (1, 2, 4):
            raise ValueError("batch_size must be 1, 2 or 4")
        query_width = speculative_tokens + 1 if speculation != "none" else 1
        per_sequence_budget = ((1024 + query_width - 1) // query_width) * query_width
        token_budget = batch_size * per_sequence_budget
        self.engine_options: dict[str, Any] = {
            "model": config["checkpoint"],
            "tokenizer": config["checkpoint"],
            "dtype": config["dtype"],
            "seed": config["seed"],
            "max_model_len": config["max_context"],
            "tensor_parallel_size": 1,
            "max_num_seqs": batch_size,
            "max_num_batched_tokens": token_budget,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "enforce_eager": False,
            "gpu_memory_utilization": 0.85,
            "limit_mm_per_prompt": {"image": 200, "video": 0},
            "mm_processor_kwargs": {"min_pixels": 1024, "max_pixels": 76800, "use_fast": False},
            "mm_device_do_normalize": True,
            "mm_processor_cache_gb": 4,
            "mm_encoder_attn_backend": "FLASH_ATTN",
            "attention_config": {"backend": "FLASH_ATTN"},
            "disable_log_stats": True,
            "generation_config": "vllm",
            "optimization_level": 3,
            "async_scheduling": async_scheduling,
            "compilation_config": {
                "mode": 3,
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "compile_mm_encoder": compile_vision,
                "cudagraph_mm_encoder": True,
                "encoder_cudagraph_token_budgets": sorted(
                    {tokens * n for tokens in (88, 128) for n in range(1, batch_size + 1)}
                ),
                "encoder_cudagraph_max_vision_items_per_batch": batch_size,
                "cudagraph_capture_sizes": sorted(
                    {1, 2, 4, 8, 16, query_width, 24, 32, 64, 96, 128, 192, 256, 512, token_budget}
                    | {query_width * n for n in range(1, batch_size + 1)}
                    | {per_sequence_budget * n for n in range(1, batch_size + 1)}
                ),
                "max_cudagraph_capture_size": token_budget,
            },
        }
        if speculation != "none":
            spec: dict[str, Any] = {"method": speculation, "num_speculative_tokens": speculative_tokens}
            if speculation in ("ngram", "ngram_gpu"):
                spec.update(prompt_lookup_min=2, prompt_lookup_max=8)
            elif speculation == "suffix":
                # Episode histories, not other episodes, supply draft candidates.
                spec.update(suffix_decoding_max_cached_requests=0)
            else:
                raise ValueError(f"unsupported draft method: {speculation}")
            self.engine_options["speculative_config"] = spec
        self.engine = LLMEngine.from_engine_args(EngineArgs(**self.engine_options))
        self.core = self.engine.engine_core.engine_core
        self.runner = self.engine.model_executor.driver_worker.worker.model_runner
        self.runner_v2 = self.engine.vllm_config.use_v2_model_runner
        self.encoder_manager = (
            self.runner.model_state.encoder_runner.cudagraph_manager
            if self.runner_v2
            else self.runner.encoder_cudagraph_manager
        )
        self.timer = EncoderInterval(self.encoder_manager, device)
        # Native modern vLLM transfers uint8 patches, then normalizes on GPU.
        # Audit that real compiled operation against the reference BF16 input,
        # outside all measured calls, retaining the device-normalization option.
        self.audit_normalizer = torch.compile(self.runner.model.visual.input_norm, fullgraph=True)
        self.sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=config["repetition_penalty"],
            max_tokens=config["max_new_tokens"],
            stop_token_ids=[151645, 151643],
            detokenize=False,
        )
        original_inputs = self.engine.input_processor.process_inputs

        def retain_inputs(*args: Any, **kwargs: Any) -> Any:
            request = original_inputs(*args, **kwargs)
            self.last_processed = request
            return request

        self.engine.input_processor.process_inputs = retain_inputs
        original_execute = self.runner.execute_model

        def observe_schedule(scheduled: Any, *args: Any, **kwargs: Any) -> Any:
            for request in scheduled.scheduled_new_reqs:
                self.prefix_cached_tokens = request.num_computed_tokens
            self.forward_tokens += scheduled.total_num_scheduled_tokens
            self.scheduled_draft_tokens += sum(map(len, scheduled.scheduled_spec_decode_tokens.values()))
            return original_execute(scheduled, *args, **kwargs)

        self.runner.execute_model = observe_schedule
        if self.runner_v2:
            from vllm.v1.worker.gpu import model_runner as runner_module

            original_dispatch = runner_module.dispatch_cg_and_sync_dp

            def observe_dispatch(*args: Any, **kwargs: Any) -> Any:
                result = original_dispatch(*args, **kwargs)
                self.graph_modes[result[0].cg_mode.name] += 1
                return result

            runner_module.dispatch_cg_and_sync_dp = observe_dispatch
        else:
            original_dispatch = self.runner._determine_batch_execution_and_padding

            def observe_dispatch(*args: Any, **kwargs: Any) -> Any:
                result = original_dispatch(*args, **kwargs)
                self.graph_modes[result[0].name] += 1
                return result

            self.runner._determine_batch_execution_and_padding = observe_dispatch
        original_update = self.core.scheduler.update_from_output

        def observe_accepted(scheduled: Any, output: Any) -> Any:
            for request_id, drafts in scheduled.scheduled_spec_decode_tokens.items():
                index = output.req_id_to_index.get(request_id)
                if index is not None and output.sampled_token_ids:
                    sampled = output.sampled_token_ids[index]
                    if sampled:
                        self.verified_draft_tokens += len(drafts)
                        self.accepted_draft_tokens += max(
                            len(sampled) - self.core.scheduler.num_sampled_tokens_per_step, 0
                        )
            return original_update(scheduled, output)

        self.core.scheduler.update_from_output = observe_accepted
        self.call_index = self.episode_index = 0
        self.history: list[int] = []
        self.images: list[Image.Image] = []
        self.image_ids: list[str] = []
        self.last_processed = None

    def optimization_evidence(self) -> dict[str, Any]:
        """Record resolved options and actual native encoder graph hit counters."""
        v = self.engine.vllm_config
        return {
            "compilation": str(v.compilation_config),
            "scheduler": str(v.scheduler_config),
            "speculation": str(v.speculative_config),
            "attention": str(v.attention_config),
            "encoder_graph": self.encoder_manager.get_cumulative_stats(),
            "model_runner_v2": self.runner_v2,
            "image_cache_keys": "Immutable per-observation UUIDs, retained with full episode images.",
            "image_normalization": "Native GPU normalization, audited after normalization at BF16 model input.",
            "precision": "BF16 weights/KV; no quantization or pruning.",
        }

    def reset(self) -> None:
        """Discard warmup/episode caches, keeping compiled graphs and native weights."""
        if self.engine.has_unfinished_requests():
            raise RuntimeError("cannot reset a live native request")
        if not self.engine.reset_prefix_cache():
            raise RuntimeError("native prefix cache reset was rejected")
        self.engine.reset_mm_cache()
        self.engine.reset_encoder_cache()
        self.history, self.images, self.image_ids = [], [], []
        self.last_processed = None
        self.episode_index += 1
        if self.speculation == "suffix":
            from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

            self.runner.drafter = SuffixDecodingProposer(self.engine.vllm_config)

    def call(self, rgb: Any, instruction: str, *, audit_inputs: bool = False) -> dict[str, Any]:
        """Measure the unchanged CPU-RGB to CPU-action boundary for one observation."""
        self.timer.reset()
        self.prefix_cached_tokens = self.forward_tokens = self.scheduled_draft_tokens = 0
        self.verified_draft_tokens = self.accepted_draft_tokens = 0
        self.graph_modes: Counter[str] = Counter()
        manager = self.encoder_manager
        hits, misses = manager.graph_hits, manager.graph_misses
        torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        turn_text = render_turn_text(self.processor, instruction, initial=not self.images)
        self.history.extend(self.tokenizer.encode(turn_text, add_special_tokens=False))
        self.images.append(Image.fromarray(rgb))
        self.image_ids.append(f"ep-{self.episode_index}-frame-{len(self.images)}")
        identity = f"replay-{self.call_index}"
        self.call_index += 1
        # Native UUID caching avoids repeatedly hashing immutable historical RGB.
        (prompt,) = self.engine.renderer.render_cmpl(
            [
                {
                    "prompt_token_ids": self.history.copy(),
                    "multi_modal_data": {"image": self.images.copy()},
                    "multi_modal_uuids": {"image": self.image_ids.copy()},
                    # The native Qwen processing-info default otherwise
                    # overrides the engine-level use_fast=False at construction.
                    "mm_processor_kwargs": {"use_fast": False},
                }
            ]
        )
        self.engine.add_request(identity, prompt, self.sampling)
        added_ns = time.perf_counter_ns()
        tokens: list[int] = []
        stop_reason = "max_tokens"
        engine_steps = returned_tokens = accepted_lower_bound = 0
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            engine_steps += 1
            if not outputs:
                continue
            if len(outputs) != 1 or outputs[0].request_id != identity:
                raise RuntimeError("batch-one request identity changed")
            complete = list(outputs[0].outputs[0].token_ids)
            if not complete:
                continue
            self.timer.first_token()
            accepted_lower_bound += max(0, len(complete) - len(tokens) - 1)
            returned_tokens = len(complete)
            tokens, reason = stopping_prefix(self.tokenizer, complete, len(tokens))
            if reason:
                stop_reason = reason
                self.engine.abort_request([identity])
                break
        # In-process has_unfinished_requests() does not include native futures.
        # Abort/EOS can leave a pipelined step; include its settlement in this
        # observation instead of leaking work or cache state into the next one.
        drain_steps = drain_inflight(self.engine, self.core)
        timing = self.timer.finish()
        if self.engine.has_unfinished_requests():
            raise RuntimeError("native request remained active after stopping")
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        parsed = parse_r2r_actions(text)
        actions, mask = actions_to_tensor(parsed)
        self.history.extend(tokens)
        torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        request = self.last_processed
        row = {
            "latency_ms": elapsed,
            "model_timing_ms": timing,
            "cpu_preparation_and_admission_ms": (added_ns - started) / 1e6,
            "token_ids": tokens,
            "generated_tokens": len(tokens),
            "native_returned_tokens": returned_tokens,
            "text": text,
            "actions": actions.tolist(),
            "action_slots": actions.shape[0],
            "action_mask": mask.tolist(),
            "output_sha256": hashlib.sha256(actions.numpy().tobytes()).hexdigest(),
            "parsed_valid": parsed.valid,
            "stop_reason": stop_reason,
            "cache_length": len(request.prompt_token_ids) + len(tokens),
            "prompt_tokens": len(request.prompt_token_ids),
            "history_images": len(self.images),
            "vision_calls": self.timer.vision_calls,
            "vision_graph_hits": manager.graph_hits - hits,
            "vision_graph_misses": manager.graph_misses - misses,
            "engine_steps": engine_steps + drain_steps,
            "drain_steps": drain_steps,
            "prefix_cached_tokens": self.prefix_cached_tokens,
            "forward_tokens": self.forward_tokens,
            "graph_modes": dict(self.graph_modes),
            "graph_steps": sum(n for mode, n in self.graph_modes.items() if mode != "NONE"),
            "non_graph_steps": self.graph_modes["NONE"],
            "scheduled_draft_tokens": self.scheduled_draft_tokens,
            "verified_draft_tokens": self.verified_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "accepted_draft_tokens_lower_bound": accepted_lower_bound,
        }
        if row["vision_graph_misses"] or not row["vision_graph_hits"]:
            raise RuntimeError("vision graph did not cover the current observation")
        if audit_inputs:
            expected = self.processor(
                text=[turn_text], images=[self.images[-1]], return_tensors="pt", padding=False
            )
            expected_raw = self.processor(
                text=[turn_text],
                images=[self.images[-1]],
                return_tensors="pt",
                padding=False,
                do_rescale=False,
                do_normalize=False,
            )
            feature = request.mm_features[-1].data
            if feature is None:
                raise RuntimeError("current observation inputs missing from native request")
            actual = feature.get_data()
            with torch.inference_mode():
                normalized = self.audit_normalizer(
                    actual["pixel_values"].to(self.device), torch.bfloat16
                ).cpu()
            raw_equal = torch.equal(actual["pixel_values"], expected_raw.pixel_values)
            normalized_equal = torch.equal(normalized, expected.pixel_values.to(torch.bfloat16))
            row["input_audit"] = {
                "tokens_equal": request.prompt_token_ids[-expected.input_ids.shape[1] :]
                == expected.input_ids[0].tolist(),
                "pixels_equal": raw_equal and normalized_equal,
                "raw_pixels_equal": raw_equal,
                "normalized_bf16_equal": normalized_equal,
                "native_pixel_dtype": str(actual["pixel_values"].dtype),
                # Native per-item fields omit the batch axis for this grid.
                "grid_equal": torch.equal(actual["image_grid_thw"].reshape(1, 3), expected.image_grid_thw),
                "grid": expected.image_grid_thw.tolist(),
                "pixel_sha256": hashlib.sha256(expected.pixel_values.numpy().tobytes()).hexdigest(),
            }
            if not all(row["input_audit"][key] for key in ("tokens_equal", "pixels_equal", "grid_equal")):
                raise ValueError(f"native preprocessing differs: {row['input_audit']}")
        return row


def main() -> None:
    """Select compatible official optimizations without changing generation semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument(
        "--speculation", choices=("none", "ngram", "ngram_gpu", "suffix"), default="ngram_gpu"
    )
    parser.add_argument("--speculative-tokens", type=int, default=16)
    parser.add_argument("--compile-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.max_steps_per_episode is not None and args.max_steps_per_episode < 1:
        parser.error("max-steps-per-episode must be positive")
    if args.speculative_tokens < 1:
        parser.error("speculative-tokens must be positive")
    raise SystemExit(
        run(
            json.loads(args.config.read_text()),
            args.output,
            args.max_steps_per_episode,
            runtime_class=ModernVLLMReplay,
            runtime_options={
                "speculation": args.speculation,
                "speculative_tokens": args.speculative_tokens,
                "compile_vision": args.compile_vision,
                "async_scheduling": args.async_scheduling,
            },
        )
    )


if __name__ == "__main__":
    main()
