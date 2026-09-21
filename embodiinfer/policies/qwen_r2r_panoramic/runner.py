"""Native Qwen2.5-VL-3B R2R panoramic navigation policy."""

from __future__ import annotations

import hashlib
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
)
from ...types import Observation
from .contract import (
    PANORAMIC_IMAGE_SIZE,
    PANORAMIC_SYSTEM_PROMPT_SHA256,
    R2R_PREPROCESSOR_SHA256,
    QwenR2RPanoramicMemory,
)
from .cuda_graph import QwenR2RPanoramicGraphRuntime
from .processing import _QwenR2RPanoramicProcessing


class QwenR2RPanoramicRunner(QwenR2RPanoramicGraphRuntime, _QwenR2RPanoramicProcessing):
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
        if profile != "panoramic":
            raise ValueError("qwen_r2r_panoramic only accepts profile panoramic")
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
            profile="panoramic",
            kind="panorama",
            size=PANORAMIC_IMAGE_SIZE,
            resize=False,
            processor_sha256=R2R_PREPROCESSOR_SHA256,
        )
        checkpoint_path = Path(checkpoint)
        processor_path = checkpoint_path / "preprocessor_config.upstream.json"
        if not processor_path.is_file():
            processor_path = checkpoint_path / "preprocessor_config.json"
        processor_bytes = processor_path.read_bytes()
        actual_processor_hash = hashlib.sha256(processor_bytes).hexdigest()
        if actual_processor_hash != R2R_PREPROCESSOR_SHA256:
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
                "Qwen R2R panoramic is restricted to Qwen2.5-VL-3B "
                f"(got model_type={config.model_type!r}, hidden_size={text_config.hidden_size})"
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
        prompt_path = checkpoint_path / "system_prompt.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"official system prompt is missing: {prompt_path}")
        prompt_bytes = prompt_path.read_bytes()
        actual_prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
        if actual_prompt_hash != PANORAMIC_SYSTEM_PROMPT_SHA256:
            raise ValueError(
                f"unexpected panoramic system prompt sha256: {actual_prompt_hash}; "
                f"expected {PANORAMIC_SYSTEM_PROMPT_SHA256}"
            )
        self.system_prompt = prompt_bytes.decode("utf-8")
        QwenR2RPanoramicGraphRuntime.__init__(
            self,
            self.model,
            profile,
            attention_backend=attention_backend,
            compile_backend=compile_backend,
            compile_cache_dir=compile_cache_dir,
            compile_text_buckets=compile_text_buckets,
            pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None),
            processor_contract={
                "preprocessor_sha256": R2R_PREPROCESSOR_SHA256,
                "system_prompt_sha256": PANORAMIC_SYSTEM_PROMPT_SHA256,
                "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
            },
        )

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

    def new_history_image_cache(self) -> HistoryImageCache:
        return HistoryImageCache.enabled() if self.history_image_cache_enabled else HistoryImageCache()

    def commit_history_image_cache(
        self, memory: QwenR2RPanoramicMemory, entry: HistoryImageCacheEntry
    ) -> HistoryImageCache:
        if not self.history_image_cache_enabled:
            raise RuntimeError("cannot commit an RGB entry while cache mode is none")
        if entry.key != self.history_image_cache_key:
            raise ValueError("panoramic history cache entry contract mismatch")
        cache = memory.history_image_cache
        if not cache.enabled_for_session:
            cache = HistoryImageCache.enabled()
        return cache.append_for_frame(frame_index=len(memory.frames), entry=entry)

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
        self, observations: list[Observation], memories: list[QwenR2RPanoramicMemory]
    ) -> list[tuple[str, torch.Tensor, list[float]]]:
        if len(observations) != len(memories) or not observations:
            raise ValueError("observations and memories must be non-empty and aligned")
        return self._infer_encoded_batch(self._encode_batch(observations, memories))

    def infer_batch_with_history_entries(
        self, observations: list[Observation], memories: list[QwenR2RPanoramicMemory]
    ) -> list[tuple[str, torch.Tensor, list[float], HistoryImageCacheEntry | None]]:
        if len(observations) != len(memories) or not observations:
            raise ValueError("observations and memories must be non-empty and aligned")
        encoded, entries = self._encode_batch_with_history_entries(observations, memories)
        rows = self._infer_encoded_batch(encoded)
        return [(*row, entries[index]) for index, row in enumerate(rows)]

    def infer(self, observation: Observation, memory: QwenR2RPanoramicMemory):
        return self.infer_batch([observation], [memory])[0]

    def infer_with_history_entry(self, observation: Observation, memory: QwenR2RPanoramicMemory):
        return self.infer_batch_with_history_entries([observation], [memory])[0]
