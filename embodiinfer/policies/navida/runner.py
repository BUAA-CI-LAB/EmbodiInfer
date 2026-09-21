"""Public NaViDA runner composed from local processing and generation runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from ...engine.parallel import TensorParallelContext, parallelize_qwen25_vl
from .contract import NAVIDA_SYSTEM_PROMPT
from .cuda_graph import CapturedNaViDADecodeGraph, NaViDAGraphRuntime
from .generation import NaViDAGenerationRuntime
from .processing import NaViDAProcessingRuntime


class NaViDARunner(
    NaViDAProcessingRuntime,
    NaViDAGenerationRuntime,
    NaViDAGraphRuntime,
):
    def __init__(
        self,
        checkpoint: str,
        *,
        max_new_tokens: int,
        execute_chunks: int,
        load_device: str | None = None,
        tensor_parallel_size: int = 1,
        tensor_parallel_group=None,
    ):
        from transformers import (
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
            RepetitionPenaltyLogitsProcessor,
            StaticCache,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
        )

        self.profile = "navida"
        self.max_new_tokens = max_new_tokens
        self.execute_chunks = execute_chunks
        checkpoint_path = Path(checkpoint)
        processor_path = checkpoint_path / "preprocessor_config.upstream.json"
        if not processor_path.is_file():
            processor_path = checkpoint_path / "preprocessor_config.json"
        processor_config = json.loads(processor_path.read_bytes())
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
        self.processor.image_processor.max_pixels = 501760
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype="auto"
        )
        config = self.model.config
        text_config = getattr(config, "text_config", config)
        if config.model_type != "qwen2_5_vl" or int(text_config.hidden_size) != 2048:
            raise ValueError(
                "NaViDA is restricted to a Qwen2.5-VL-3B checkpoint "
                f"(got model_type={config.model_type!r}, hidden_size={text_config.hidden_size})"
            )
        self.tensor_parallel = TensorParallelContext.from_distributed(
            tensor_parallel_size, tensor_parallel_group
        )
        if load_device is not None:
            self.model.to(load_device)
        parallelize_qwen25_vl(self.model, self.tensor_parallel)
        self.system_prompt = NAVIDA_SYSTEM_PROMPT
        self.cuda_graph_enabled = False
        self.cuda_graph_requested = False
        self._static_cache_type = StaticCache
        self._navida_repetition = RepetitionPenaltyLogitsProcessor(1.05)
        self._navida_temperature = TemperatureLogitsWarper(0.2)
        self._navida_top_k = TopKLogitsWarper(50)
        self._navida_graph_cache: dict[tuple, CapturedNaViDADecodeGraph] = {}
        self._navida_graph_captures = 0
        self._navida_graph_replays = 0
        self._navida_graph_lock = Lock()
