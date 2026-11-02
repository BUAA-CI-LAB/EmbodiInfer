"""R2R first-frame references: native SGLang/vLLM ActiveVLN and official StreamVLN.

Uses the frozen input/config contract of vln_firstframe.py. Each formal run
warms all 128 observations, clears content caches, then measures fresh histories.
Optional engines and official model imports belong in their isolated environments.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image


def digest(path: Path) -> str:
    """Identify an input/config artifact by its exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    """Persist complete audit records without exposing a partially written file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


class NativeVLLM:
    """Reuse the instrumented native vLLM 0.30 batch reference."""

    def __init__(self, config: dict[str, Any], source: Path, **_: Any) -> None:
        sys.path.insert(0, str(source / "benchmarks/activevln-benchmark"))
        from benchmark_vllm_batch import BatchedVLLMReplay

        native = dict(config, max_context=4096)
        self.runner = BatchedVLLMReplay(native, torch.device("cuda:0"))
        self.config = config

    def reset(self) -> None:
        """Discard content caches while retaining compilation and graph state."""
        self.runner.reset()

    def call(self, samples: list[dict[str, Any]], images: list[np.ndarray], identity: str) -> dict[str, Any]:
        """Time real native requests from RGB to CPU action tensors."""
        from benchmark_vllm_batch import ReplaySession

        sessions = [ReplaySession(f"{identity}-{i}", SimpleNamespace(**s)) for i, s in enumerate(samples)]
        rows, stats = self.runner.call_batch(sessions, images, audit_inputs=identity.startswith("warmup-0"))
        return {"e2e_ms": stats["latency_ms"], "rows": rows, "native": stats}

    def evidence(self) -> dict[str, Any]:
        """Expose resolved settings and native graph counters."""
        return {"options": self.runner.engine_options, "runtime": self.runner.optimization_evidence()}

    def close(self) -> None:
        """Release the native engine on process exit."""
        self.runner.engine.engine_core.shutdown()


class NativeSGLang:
    """Run the native SRT scheduler, prefill/verify graphs and ngram drafts."""

    def __init__(self, config: dict[str, Any], **_: Any) -> None:
        os.environ["SGLANG_VIT_ENABLE_CUDA_GRAPH"] = "1"
        from sglang import Engine
        from sglang.srt.sampling.sampling_params import SamplingParams
        from transformers import AutoProcessor

        self.sampling = dict(
            temperature=0,
            top_p=1,
            repetition_penalty=config["repetition_penalty"],
            max_new_tokens=config["max_new_tokens"],
            stop=["stop"],
            no_stop_trim=True,
        )
        SamplingParams(**self.sampling)
        batch = config["batch_size"]
        prefill_budget = 1024 * batch
        self.processor = AutoProcessor.from_pretrained(config["checkpoint"], use_fast=False)
        self.options = dict(
            model_path=config["checkpoint"],
            dtype="bfloat16",
            context_length=4096,
            attention_backend="flashinfer",
            mm_attention_backend="triton_attn",
            mem_fraction_static=0.72,
            max_running_requests=batch,
            max_total_tokens=4096 * batch,
            cuda_graph_max_bs_decode=batch,
            cuda_graph_bs_decode=[n for n in (1, 2, 4, 8) if n <= batch],
            enable_torch_compile=True,
            torch_compile_max_bs=batch,
            chunked_prefill_size=prefill_budget,
            cuda_graph_config={
                "prefill": {
                    "backend": "full",
                    "bs": [n for n in (256, 512, 1024, 2048, 4096, 8192) if n <= prefill_budget],
                    "max_bs": prefill_budget,
                    "full_prefill_max_req": batch,
                    "max_seq_len": 4096,
                }
            },
            mm_process_config={"image": {"min_pixels": 1024, "max_pixels": 76800}},
            mm_preprocess_cache_size_mb=0,
            image_processor_backend="auto",
            mm_feature_transport="cuda_ipc",
            speculative_algorithm="NGRAM",
            speculative_num_draft_tokens=12,
            enable_memory_saver=True,
            random_seed=42,
            log_level="warning",
        )
        self.engine = Engine(**self.options)
        self.config = config

    def reset(self) -> None:
        """Flush native KV and multimodal caches outside the measured interval."""
        result = self.engine.flush_cache()
        if not result.success:
            raise RuntimeError(f"SRT cache flush failed: {result}")

    def call(self, samples: list[dict[str, Any]], images: list[np.ndarray], identity: str) -> dict[str, Any]:
        """Include local SRT process transport and parse complete native responses."""
        from embodiinfer.policies.activevln.prompt_activevln import (
            actions_to_tensor,
            parse_r2r_actions,
            render_turn_text,
        )

        torch.cuda.synchronize()
        start = time.perf_counter()
        prompts = [
            render_turn_text(self.processor, s["instruction"], initial=True, action_space="r2r")
            for s in samples
        ]
        outputs = self.engine.generate(
            prompt=prompts,
            image_data=[[Image.fromarray(im)] for im in images],
            sampling_params=self.sampling,
            cache_salt=[f"{identity}-{i}" for i in range(len(samples))],
        )
        rows = []
        for sample, result in zip(samples, outputs, strict=True):
            parsed = parse_r2r_actions(result["text"])
            actions, mask = actions_to_tensor(parsed)
            rows.append(
                dict(
                    episode_id=sample["episode_id"],
                    text=result["text"],
                    actions=actions.tolist(),
                    action_mask=mask.tolist(),
                    parsed_valid=parsed.valid,
                    generated_tokens=result["meta_info"]["completion_tokens"],
                    native_meta=result["meta_info"],
                )
            )
        torch.cuda.synchronize()
        return dict(e2e_ms=(time.perf_counter() - start) * 1000, rows=rows)

    def evidence(self) -> dict[str, Any]:
        """Return actual SRT configuration rather than inferred optimization flags."""
        return {"options": self.options, "sampling": self.sampling, "runtime": self.engine.get_server_info()}

    def close(self) -> None:
        """Stop the scheduler and its owned worker processes."""
        self.engine.shutdown()


class OfficialStreamVLN:
    """Batch official StreamVLN generate calls with left padding and empty history."""

    def __init__(self, config: dict[str, Any], official_source: Path, source: Path, **_: Any) -> None:
        sys.path.insert(0, str(official_source))
        sys.path.insert(0, str(official_source / "streamvln"))
        from model.stream_video_vln import StreamVLNForCausalLM
        from transformers import AutoConfig, AutoTokenizer

        # Share only the pure prompt/action contract. Importing the policy package
        # would also load the optimized runtime into the official legacy environment.
        prompt_path = source / "embodiinfer/policies/streamvln/prompt.py"
        spec = importlib.util.spec_from_file_location("firstframe_stream_prompt", prompt_path)
        self.prompt = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.prompt
        prompt_source = prompt_path.read_text()
        constants_import = "from ...models.streamvln.backbone import IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX"
        if prompt_source.count(constants_import) != 1:
            raise RuntimeError("prompt contract imports changed; review official dependency isolation")
        prompt_source = prompt_source.replace(
            constants_import, "from utils.utils import IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX", 1
        )
        exec(compile(prompt_source, str(prompt_path), "exec"), self.prompt.__dict__)
        self.prompt_sha256 = digest(prompt_path)
        checkpoint = config["policy"]["checkpoint"]
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
        model_config = AutoConfig.from_pretrained(checkpoint)
        if config.get("official_vision_checkpoint"):
            model_config.mm_vision_tower = config["official_vision_checkpoint"]
        model_config.tokenizer_padding_side = "left"
        model_config.tokenizer_model_max_length = config["policy"]["max_context"]
        self.attention = config.get("official_attention", "flash_attention_2")
        self.model, self.loading = StreamVLNForCausalLM.from_pretrained(
            checkpoint,
            config=model_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attention,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        if self.loading["missing_keys"] or self.loading["unexpected_keys"] or self.loading["mismatched_keys"]:
            raise RuntimeError(f"official checkpoint coverage failure: {self.loading}")
        self.model.requires_grad_(False).to("cuda:0").eval()
        if any(p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise RuntimeError("official model did not load entirely in BF16")
        self.model.reset(1)
        self.processor = self.model.get_vision_tower().image_processor
        self.config = config
        self.forward_shapes: dict[str, int] = {}

        def record_shape(module: Any, args: Any, kwargs: Any) -> None:
            x = kwargs.get("inputs_embeds")
            if x is not None:
                key = str(tuple(x.shape))
                self.forward_shapes[key] = self.forward_shapes.get(key, 0) + 1

        self.model.model.register_forward_pre_hook(record_shape, with_kwargs=True)
        self.source = official_source

    def reset(self) -> None:
        """Reset the official batch cache before every independent first frame."""
        self.model.reset(1)

    def call(self, samples: list[dict[str, Any]], images: list[np.ndarray], identity: str) -> dict[str, Any]:
        """Perform one real B-row generate, including CPU preprocessing and actions."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        tokens = [
            self.prompt.tokenize_multimodal_prompt(
                self.prompt.render_streamvln_prompt(
                    s["instruction"], window_start=True, include_memory=False
                ),
                self.tokenizer,
            )
            for s in samples
        ]
        length = max(len(ids) for ids in tokens)
        ids = torch.full((len(samples), length), self.tokenizer.pad_token_id or 151643, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(tokens):
            ids[i, -len(row) :] = row
            mask[i, -len(row) :] = True
        pixels = self.processor.preprocess([Image.fromarray(im) for im in images], return_tensors="pt")[
            "pixel_values"
        ]
        ids, mask = ids.cuda(), mask.cuda()
        positions = (mask.long().cumsum(-1) - 1).clamp_min_(0)
        outputs = self.model.generate(
            inputs=ids,
            images=pixels[:, None].to("cuda", dtype=torch.bfloat16),
            attention_mask=mask,
            position_ids=positions,
            env_id=0,
            time_ids=[[0]] * len(samples),
            task_type=[0] * len(samples),
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.config["policy"]["max_new_tokens"],
            repetition_penalty=self.config["policy"]["repetition_penalty"],
            use_cache=True,
            num_logits_to_keep=1,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id or 151643,
        )
        sequences = outputs.sequences.cpu().tolist()
        rows = []
        for sample, sequence in zip(samples, sequences, strict=True):
            if self.tokenizer.eos_token_id in sequence:
                sequence = sequence[: sequence.index(self.tokenizer.eos_token_id) + 1]
            text = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
            parsed = self.prompt.parse_symbolic_actions(text)
            rows.append(
                dict(
                    episode_id=sample["episode_id"],
                    token_ids=sequence,
                    text=text,
                    generated_tokens=len(sequence),
                    actions=parsed.actions.tolist(),
                    action_mask=parsed.mask.tolist(),
                    parsed_valid=not parsed.invalid,
                )
            )
        torch.cuda.synchronize()
        return dict(e2e_ms=(time.perf_counter() - start) * 1000, rows=rows)

    def evidence(self) -> dict[str, Any]:
        """Record original source hashes, tensor batch shapes and strict weight coverage."""
        return dict(
            source=str(self.source),
            source_hashes={str(p.relative_to(self.source)): digest(p) for p in self.source.rglob("*.py")},
            loading=self.loading,
            batch_forward_shapes=self.forward_shapes,
            padding_side="left",
            attention=self.attention,
            num_logits_to_keep=1,
            prompt_contract_sha256=self.prompt_sha256,
            generation="unmodified official model.generate",
        )

    def close(self) -> None:
        """Model resources are reclaimed when the isolated process exits."""


@torch.inference_mode()
def main() -> None:
    """Measure 128 fixed first frames with independently reset request histories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["vllm", "sglang", "official"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--official-source", type=Path, help="Official StreamVLN checkout (required for official)"
    )
    args = parser.parse_args()
    if args.engine == "official" and args.official_source is None:
        parser.error("--official-source is required for the official StreamVLN reference")
    sys.path.insert(0, str(args.source))
    config = json.loads(args.config.read_text())
    batch, dataset = config["batch_size"], config["evaluation_dataset"]
    if dataset != "r2r":
        parser.error("this first-frame reference comparison supports R2R only")
    assert config["precision"] == "bf16" and batch in (1, 2, 4, 8)
    assert (config["model"] == "streamvln") == (args.engine == "official")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    manifest_path = args.inputs / "cache-manifest.json"
    assert digest(manifest_path) == config["firstframe"]["cache_manifest_sha256"]
    samples = json.loads(manifest_path.read_text())[dataset]
    assert len(samples) == 128
    selection = Path(config["navigation_manifest"])
    assert digest(selection) == config["navigation"]["selections"][dataset]["sha256"]
    episodes = json.loads(selection.read_text())["episodes"]
    for sample, episode in zip(samples, episodes, strict=True):
        assert str(sample["episode_id"]) == str(episode["episode_id"])
        assert sample["instruction"] == episode["instruction"]["instruction_text"]
    images = []
    for sample in samples:
        path = args.inputs / sample["image"]
        assert digest(path) == sample["image_sha256"]
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB")))
        assert images[-1].shape == (480, 640, 3)
    versions = {}
    for name in ("torch", "transformers", "triton", "vllm", "sglang", "flashinfer-python", "flash-attn"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    started = time.monotonic()
    save(args.output / "progress.json", {"stage": "loading", "versions": versions})
    cls = {"vllm": NativeVLLM, "sglang": NativeSGLang, "official": OfficialStreamVLN}[args.engine]
    runner = cls(config, source=args.source, official_source=args.official_source)
    load_s = time.monotonic() - started
    try:
        for index in range(0, 128, batch):
            runner.reset()
            runner.call(samples[index : index + batch], images[index : index + batch], f"warmup-{index}")
            save(args.output / "progress.json", {"stage": "warming", "observations": index + batch})
        runner.reset()
        torch.cuda.synchronize()
        warmup_s = time.monotonic() - started - load_s
        save(
            args.output / "ready.json",
            dict(
                load_s=load_s,
                warmup_s=warmup_s,
                versions=versions,
                device=torch.cuda.get_device_name(),
                runtime=runner.evidence(),
            ),
        )
        torch.cuda.reset_peak_memory_stats()
        measured = []
        with (args.output / "batches.jsonl").open("w") as stream:
            for index in range(0, 128, batch):
                runner.reset()
                row = runner.call(
                    samples[index : index + batch], images[index : index + batch], f"measured-{index}"
                )
                stream.write(json.dumps(row, default=str) + "\n")
                stream.flush()
                measured.append(row)
                save(args.output / "progress.json", {"stage": "measuring", "observations": index + batch})
        rows = [r for group in measured for r in group["rows"]]
        assert [row["episode_id"] for row in rows] == [s["episode_id"] for s in samples]
        latencies = np.asarray([r["e2e_ms"] for r in measured])
        summary = dict(
            status="complete",
            engine=args.engine,
            model=config["model"],
            dataset=dataset,
            precision="bf16",
            batch_size=batch,
            episodes=128,
            frames_per_episode=1,
            model_calls=len(measured),
            batch_mean_ms=float(latencies.mean()),
            batch_p95_ms=float(np.percentile(latencies, 95)),
            amortized_per_image_ms=float(latencies.sum() / 128),
            images_per_second=128000 / float(latencies.sum()),
            model_total_s=float(latencies.sum() / 1000),
            output_tokens=sum(r["generated_tokens"] for r in rows),
            valid_outputs=sum(r["parsed_valid"] for r in rows),
            load_s=load_s,
            warmup_s=warmup_s,
            versions=versions,
            runtime=runner.evidence(),
            config_sha256=digest(args.config),
            driver_sha256=digest(Path(__file__)),
            cache_manifest_sha256=digest(manifest_path),
            selection_sha256=digest(Path(config["navigation_manifest"])),
            accuracy_evaluated=False,
            timing_scope="Decoded CPU RGB through preprocessing, full action generation and CPU actions; SGLang includes local SRT IPC. Excludes disk, warmup, renderer, HTTP and reset.",
            cache_policy="Flush KV/multimodal caches before every independent batch; keep compiled kernels and graphs.",
            context_capacity=4096 if args.engine != "official" else config["policy"]["max_context"],
        )
        save(args.output / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "runtime"}), flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
