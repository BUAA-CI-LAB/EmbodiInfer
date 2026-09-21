"""Official Low checkpoint loader, processor, and inference runner."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Literal

import torch

from ...engine.parallel import TensorParallelContext, parallelize_qwen25_vl
from ...models.qwen25_vl.history_image_cache import (
    HistoryImageCache,
    HistoryImageCacheEntry,
    HistoryImageCacheKey,
    normalize_history_image_cache_mode,
    prepare_history_images,
)
from ...types import Observation
from .contract import (
    LOW_LEVEL_IMAGE_SIZE,
    LOW_LEVEL_SYSTEM_PROMPT_SHA256,
    R2R_PREPROCESSOR_SHA256,
    QwenR2RLowMemory,
)
from .cuda_graph import QwenR2RLowGraphRuntime
from .processing import tensor_to_pil


class QwenR2RLowRunner:
    def __init__(
        self,
        checkpoint: str,
        profile: str,
        *,
        max_new_tokens: int,
        execute_chunks: int,
        attention_backend: str = "torch_sdpa",
        compile_backend: Literal["none", "inductor"] = "none",
        compile_cache_dir: str | Path | None = None,
        compile_text_buckets: tuple[int, ...] = (),
        history_image_cache: Literal["none", "rgb_bytes"] = "rgb_bytes",
        load_device: str | None = None,
        tensor_parallel_size: int = 1,
        tensor_parallel_group=None,
    ):
        if profile != "low_level":
            raise ValueError("qwen_r2r_low only accepts profile low_level")
        if max_new_tokens != 1 or execute_chunks != 1:
            raise ValueError("qwen_r2r_low is a single-token policy")
        history_image_cache_mode = normalize_history_image_cache_mode(history_image_cache)
        from transformers import (
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
        )

        self.profile = profile
        self.max_new_tokens = max_new_tokens
        self.execute_chunks = execute_chunks
        self.history_image_cache_mode = history_image_cache_mode
        self.history_image_cache_enabled = history_image_cache_mode == "rgb_bytes"
        self.history_image_cache_key = HistoryImageCacheKey(
            profile="low_level",
            kind="low",
            size=LOW_LEVEL_IMAGE_SIZE,
            resize=True,
            processor_sha256=R2R_PREPROCESSOR_SHA256,
        )
        checkpoint_path = Path(checkpoint)
        processor_path = checkpoint_path / "preprocessor_config.upstream.json"
        if not processor_path.is_file():
            processor_path = checkpoint_path / "preprocessor_config.json"
        processor_bytes = processor_path.read_bytes()
        if hashlib.sha256(processor_bytes).hexdigest() != R2R_PREPROCESSOR_SHA256:
            raise ValueError(f"unexpected official processor config: {processor_path}")
        processor_config = json.loads(processor_bytes)
        processor_config.pop("image_processor_type", None)
        processor_config.pop("processor_class", None)
        processor_config["size"] = {
            "shortest_edge": int(processor_config["min_pixels"]),
            "longest_edge": int(processor_config["max_pixels"]),
        }
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        chat_template_path = checkpoint_path / "chat_template.json"
        if chat_template_path.is_file():
            chat_template = json.loads(chat_template_path.read_text(encoding="utf-8"))["chat_template"]
        else:
            chat_template = tokenizer.chat_template
        if not chat_template:
            raise FileNotFoundError(f"chat template is missing from {checkpoint_path}")
        self.processor = Qwen2_5_VLProcessor(
            image_processor=Qwen2VLImageProcessor(**processor_config),
            video_processor=Qwen2VLVideoProcessor(**processor_config),
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype="auto"
        )
        config = self.model.config
        text_config = getattr(config, "text_config", config)
        if config.model_type != "qwen2_5_vl" or int(text_config.hidden_size) != 2048:
            raise ValueError(
                f"Qwen navigation profiles are restricted to Qwen2.5-VL-3B (got model_type={config.model_type!r}, hidden_size={text_config.hidden_size})"
            )
        self.tensor_parallel = TensorParallelContext.from_distributed(
            tensor_parallel_size, tensor_parallel_group
        )
        if self.tensor_parallel.enabled:
            if attention_backend != "torch_sdpa":
                raise ValueError("Qwen tensor parallelism currently requires attention_backend=torch_sdpa")
            if compile_backend != "none":
                raise ValueError("Qwen tensor parallelism currently requires compile_backend=none")
        if load_device is not None:
            self.model.to(load_device)
        parallelize_qwen25_vl(self.model, self.tensor_parallel)
        prompt_path = Path(checkpoint) / "system_prompt.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"official system prompt is missing: {prompt_path}")
        prompt_bytes = prompt_path.read_bytes()
        actual_hash = hashlib.sha256(prompt_bytes).hexdigest()
        if actual_hash != LOW_LEVEL_SYSTEM_PROMPT_SHA256:
            raise ValueError(
                f"unexpected low_level system prompt sha256: {actual_hash}; expected {LOW_LEVEL_SYSTEM_PROMPT_SHA256}"
            )
        self.system_prompt = prompt_bytes.decode("utf-8")
        self.cuda_graph_enabled = False
        self.cuda_graph_requested = False
        self.graph_runtime = QwenR2RLowGraphRuntime(
            self.model,
            profile,
            attention_backend=attention_backend,
            compile_backend=compile_backend,
            compile_cache_dir=compile_cache_dir,
            compile_text_buckets=compile_text_buckets,
            pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None),
            processor_contract={
                "preprocessor_sha256": R2R_PREPROCESSOR_SHA256,
                "system_prompt_sha256": LOW_LEVEL_SYSTEM_PROMPT_SHA256,
                "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
            },
        )

    @property
    def torch_compile_enabled(self) -> bool:
        return self.graph_runtime.torch_compile_enabled

    @property
    def compile_active(self) -> bool:
        return self.torch_compile_enabled

    @property
    def compile_inactive_reason(self) -> str | None:
        return None if self.compile_active else "compile_backend_none"

    @property
    def runtime_mode(self) -> str:
        if self.cuda_graph_enabled:
            return "compiled_manual_cudagraph" if self.torch_compile_enabled else "manual_cudagraph"
        return "compiled" if self.torch_compile_enabled else "eager"

    def configure_cuda_graph(self, enabled: bool) -> bool:
        if enabled and self.tensor_parallel.enabled:
            raise ValueError("Qwen tensor parallelism does not yet support CUDA Graph capture")
        self.cuda_graph_requested = enabled
        device = next(self.model.parameters()).device
        effective = enabled and device.type == "cuda"
        self.cuda_graph_enabled = effective
        return effective

    def _native_inputs(self, encoded: dict[str, torch.Tensor]):
        return self.graph_runtime.native_inputs(encoded)

    def _graph_logits(self, encoded: dict[str, torch.Tensor]):
        return self.graph_runtime.uncaptured_logits(encoded)

    def _manual_graph_key(self, encoded: dict[str, torch.Tensor], inputs: tuple[torch.Tensor, ...]):
        return self.graph_runtime.graph_key(encoded, inputs)

    def _manual_graph_logits(self, encoded: dict[str, torch.Tensor]):
        return self.graph_runtime.captured_logits(encoded)

    def manual_graph_stats(self) -> dict[str, object]:
        return self.graph_runtime.stats()

    def _low_prompt(self, observation: Observation, memory: QwenR2RLowMemory):
        frames = [*memory.frames, observation.images[0]]
        content = [
            {
                "type": "text",
                "text": f"Route Instruction: {observation.instruction}\nCurrent Step: {len(memory.frames)}\nCummulative Distance Traveled: {observation.metadata.get('distance_traveled', 0.0)}\nImages from Previous Steps: ",
            }
        ]
        for _frame in memory.frames:
            content.append({"type": "image"})
        if not memory.frames:
            content[0]["text"] += "[]"
        content.append(
            {
                "type": "text",
                "text": f"\nActions performed at Previous Steps: {list(memory.responses)}\nCurrent image:",
            }
        )
        content.append({"type": "image"})
        possible = ["Left", "Right", "Move", "Stop"]
        if observation.metadata.get("move_possible") is False:
            possible.remove("Move")
        content.append(
            {
                "type": "text",
                "text": f"\nPossible actions: {possible}\nNow predict the next action based on the input you have recived. Answer on the format: Action: (an the action you choose)",
            }
        )
        return (content, frames, ["low"] * len(frames), "Action: ")

    def _render_image(self, frame: torch.Tensor, kind: str):
        if kind == "low":
            return tensor_to_pil(frame, LOW_LEVEL_IMAGE_SIZE, resize=True)
        if kind == "panorama":
            return tensor_to_pil(frame, LOW_LEVEL_IMAGE_SIZE, resize=False)
        if kind == "candidate":
            return tensor_to_pil(frame, LOW_LEVEL_IMAGE_SIZE, resize=False)
        image = tensor_to_pil(frame, LOW_LEVEL_IMAGE_SIZE, resize=False)
        image = image.resize(LOW_LEVEL_IMAGE_SIZE)
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        encoded.seek(0)
        from PIL import Image

        with Image.open(encoded) as decoded:
            return decoded.convert("RGB").copy()

    def new_history_image_cache(self) -> HistoryImageCache:
        return HistoryImageCache.enabled() if self.history_image_cache_enabled else HistoryImageCache()

    def commit_history_image_cache(
        self, memory: QwenR2RLowMemory, entry: HistoryImageCacheEntry
    ) -> HistoryImageCache:
        if not self.history_image_cache_enabled:
            raise RuntimeError("cannot commit an RGB entry while cache mode is none")
        if entry.key != self.history_image_cache_key:
            raise ValueError("low-level history cache entry contract mismatch")
        cache = memory.history_image_cache
        if not cache.enabled_for_session:
            cache = HistoryImageCache.enabled()
        return cache.append_for_frame(frame_index=len(memory.frames), entry=entry)

    def _prepare_batch_impl(
        self,
        observations: list[Observation],
        memories: list[QwenR2RLowMemory],
        *,
        capture_current_entries: bool,
    ) -> tuple[dict[str, torch.Tensor], list[HistoryImageCacheEntry | None]]:
        texts: list[str] = []
        image_batches: list[list] = []
        current_entries: list[HistoryImageCacheEntry | None] = []
        for observation, memory in zip(observations, memories):
            content, frames, kinds, label = self._low_prompt(observation, memory)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": content},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if label:
                text += f"<|im_start|>assistant\n{label}"
            texts.append(text)
            history_count = len(memory.frames)
            rendered, current_entry, _ = prepare_history_images(
                frames,
                kinds,
                history_count=history_count,
                cache=memory.history_image_cache,
                key=self.history_image_cache_key,
                cache_enabled=self.history_image_cache_enabled,
                capture_current_entry=capture_current_entries,
                render=self._render_image,
            )
            image_batches.append(rendered)
            current_entries.append(current_entry)
        encoded = self.processor(text=texts, images=image_batches, padding=True, return_tensors="pt")
        if len(observations) == 1:
            encoded["attention_mask"].fill_(1)
        encoded = self.graph_runtime.bucket_encoded(dict(encoded))
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = int(getattr(text_config, "max_position_embeddings", 0))
        if context_limit and encoded["input_ids"].shape[1] > context_limit:
            raise ValueError(
                f"navigation prompt has {encoded['input_ids'].shape[1]} tokens, above {context_limit}"
            )
        return encoded, current_entries

    def _prepare_batch_with_history_entries(
        self, observations: list[Observation], memories: list[QwenR2RLowMemory]
    ) -> tuple[dict[str, torch.Tensor], list[HistoryImageCacheEntry | None]]:
        return self._prepare_batch_impl(observations, memories, capture_current_entries=True)

    def _prepare_batch(
        self, observations: list[Observation], memories: list[QwenR2RLowMemory]
    ) -> dict[str, torch.Tensor]:
        encoded, _ = self._prepare_batch_impl(observations, memories, capture_current_entries=False)
        return encoded

    def _move_encoded(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in encoded.items()
        }

    def _encode_batch_with_history_entries(
        self, observations: list[Observation], memories: list[QwenR2RLowMemory]
    ) -> tuple[dict[str, torch.Tensor], list[HistoryImageCacheEntry | None]]:
        encoded, current_entries = self._prepare_batch_with_history_entries(observations, memories)
        return self._move_encoded(encoded), current_entries

    def _encode_batch(
        self, observations: list[Observation], memories: list[QwenR2RLowMemory]
    ) -> dict[str, torch.Tensor]:
        return self._move_encoded(self._prepare_batch(observations, memories))

    def _infer_encoded_batch(
        self, encoded: dict[str, torch.Tensor]
    ) -> list[tuple[str, torch.Tensor, list[float]]]:
        with torch.inference_mode():
            if self.cuda_graph_enabled:
                logits = self._manual_graph_logits(encoded)
            elif self.torch_compile_enabled:
                logits = self._graph_logits(encoded)
            else:
                logits = self.model(**encoded, use_cache=False).logits[:, -1]
            tokens = logits.argmax(-1, keepdim=True)
        decoded = self.processor.batch_decode(tokens, skip_special_tokens=True)
        return [(text.strip(), tokens[row], []) for row, text in enumerate(decoded)]

    def infer_batch(
        self, observations: list[Observation], memories: list[QwenR2RLowMemory]
    ) -> list[tuple[str, torch.Tensor, list[float]]]:
        if len(observations) != len(memories) or not observations:
            raise ValueError("observations and memories must be non-empty and aligned")
        return self._infer_encoded_batch(self._encode_batch(observations, memories))

    def infer_batch_with_history_entries(
        self, observations: list[Observation], memories: list[QwenR2RLowMemory]
    ) -> list[tuple[str, torch.Tensor, list[float], HistoryImageCacheEntry | None]]:
        if len(observations) != len(memories) or not observations:
            raise ValueError("observations and memories must be non-empty and aligned")
        encoded, entries = self._encode_batch_with_history_entries(observations, memories)
        rows = self._infer_encoded_batch(encoded)
        return [(*row, entries[index]) for index, row in enumerate(rows)]

    def infer(self, observation: Observation, memory: QwenR2RLowMemory):
        return self.infer_batch([observation], [memory])[0]

    def infer_with_history_entry(self, observation: Observation, memory: QwenR2RLowMemory):
        return self.infer_batch_with_history_entries([observation], [memory])[0]


__all__ = ["QwenR2RLowRunner"]
