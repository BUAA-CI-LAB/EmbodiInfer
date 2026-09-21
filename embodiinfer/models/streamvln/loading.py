"""Direct loader for the published StreamVLN sharded checkpoint."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .backbone import StreamVLNBackbone, StreamVLNProjector

_LOCKED_PROFILE = {
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "num_history": 8,
    "num_future_steps": 4,
    "mm_vision_select_layer": -2,
}


@dataclass(frozen=True)
class StreamVLNCheckpoint:
    backbone: StreamVLNBackbone
    tokenizer: Any
    image_processor: Any
    eos_token_ids: tuple[int, ...]


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported StreamVLN dtype: {name!r}") from exc


def _checkpoint_root(checkpoint: str | Path) -> Path:
    root = Path(checkpoint).expanduser()
    required = (
        root / "config.json",
        root / "generation_config.json",
        root / "model.safetensors.index.json",
        root / "tokenizer.json",
        root / "tokenizer_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("invalid StreamVLN checkpoint; missing: " + ", ".join(missing))
    return root


def _load_checkpoint_weights(
    root: Path,
    llm: torch.nn.Module,
    vision_tower: torch.nn.Module,
    projector: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    expected_llm = set(llm.state_dict())
    expected_vision = set(vision_tower.state_dict())
    expected_projector = set(projector.state_dict())
    seen_llm: set[str] = set()
    seen_vision: set[str] = set()
    seen_projector: set[str] = set()
    unexpected: list[str] = []
    vision_prefix = "model.vision_tower.vision_tower."
    projector_prefix = "model.mm_projector."

    for shard in shards:
        llm_state = {}
        vision_state = {}
        projector_state = {}
        with safe_open(root / shard, framework="pt", device=str(device)) as handle:
            for source_key in handle.keys():  # noqa: SIM118 — safe_open is not an iterable mapping.
                tensor = handle.get_tensor(source_key)
                if tensor.is_floating_point() and tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                if source_key in expected_llm:
                    llm_state[source_key] = tensor
                    seen_llm.add(source_key)
                elif source_key.startswith(vision_prefix):
                    target_key = source_key.removeprefix(vision_prefix)
                    if target_key not in expected_vision:
                        unexpected.append(source_key)
                    else:
                        vision_state[target_key] = tensor
                        seen_vision.add(target_key)
                elif source_key.startswith(projector_prefix):
                    target_key = source_key.removeprefix(projector_prefix)
                    if target_key not in expected_projector:
                        unexpected.append(source_key)
                    else:
                        projector_state[target_key] = tensor
                        seen_projector.add(target_key)
                else:
                    unexpected.append(source_key)
        if llm_state:
            llm.load_state_dict(llm_state, strict=False, assign=True)
        if vision_state:
            vision_tower.load_state_dict(vision_state, strict=False, assign=True)
        if projector_state:
            projector.load_state_dict(projector_state, strict=False, assign=True)
        if device.type == "cuda" and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            # Jetson CUDA and the filesystem page cache share physical DRAM.
            with suppress(OSError), (root / shard).open("rb") as shard_file:
                os.posix_fadvise(shard_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    missing = sorted(
        (expected_llm - seen_llm)
        | {f"vision:{key}" for key in expected_vision - seen_vision}
        | {f"projector:{key}" for key in expected_projector - seen_projector}
    )
    if missing or unexpected:
        raise RuntimeError(
            f"StreamVLN checkpoint key mismatch; missing={missing[:8]}, unexpected={sorted(unexpected)[:8]}"
        )


def _materialize_runtime_buffers(
    llm: torch.nn.Module,
    vision_tower: torch.nn.Module,
    device: torch.device,
) -> None:
    """Create non-persistent buffers omitted from the checkpoint state dict."""

    head_dim = int(
        getattr(
            llm.config,
            "head_dim",
            llm.config.hidden_size // llm.config.num_attention_heads,
        )
    )
    llm.model.rotary_emb.inv_freq = 1.0 / (
        float(llm.config.rope_theta)
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    vision_config = vision_tower.config
    patches_per_side = int(vision_config.image_size) // int(vision_config.patch_size)
    vision_tower.vision_model.embeddings.position_ids = torch.arange(
        patches_per_side * patches_per_side,
        device=device,
        dtype=torch.long,
    ).expand((1, -1))


def load_streamvln_checkpoint(
    checkpoint: str | Path,
    *,
    dtype: str = "bfloat16",
    max_context: int = 32768,
    load_device: str | torch.device = "cpu",
) -> StreamVLNCheckpoint:
    """Build stock weight containers and load every tensor without upstream code."""

    root = _checkpoint_root(checkpoint)
    raw_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    mismatched = {
        key: (raw_config.get(key), value)
        for key, value in _LOCKED_PROFILE.items()
        if raw_config.get(key) != value
    }
    if mismatched:
        raise ValueError(f"checkpoint does not match the locked StreamVLN profile: {mismatched}")

    from transformers import (
        AutoTokenizer,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipImageProcessor,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from transformers.modeling_utils import no_init_weights

    torch_dtype = _torch_dtype(dtype)
    target_device = torch.device(load_device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("StreamVLN load_device requests CUDA, but CUDA is unavailable")
    qwen_config = Qwen2Config(
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_dropout=0.0,
        hidden_act="silu",
        tie_word_embeddings=False,
        use_cache=True,
        use_sliding_window=False,
        bos_token_id=151643,
        eos_token_id=151645,
        pad_token_id=151643,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=26,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        vision_use_head=False,
    )
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        with torch.device("meta"), no_init_weights():
            llm = Qwen2ForCausalLM(qwen_config)
            vision_tower = SiglipVisionModel(vision_config)
            projector = StreamVLNProjector(1152, 3584)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    _load_checkpoint_weights(
        root,
        llm,
        vision_tower,
        projector,
        torch_dtype,
        target_device,
    )
    _materialize_runtime_buffers(llm, vision_tower, target_device)
    tokenizer = AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    # Observation.images is already float in [0, 1], so rescaling by 1/255 a
    # second time would diverge from the published PIL/uint8 preprocessing.
    image_processor = SiglipImageProcessor(
        do_resize=True,
        size={"height": 384, "width": 384},
        resample=3,
        do_rescale=False,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    backbone = StreamVLNBackbone(
        llm,
        vision_tower,
        projector,
        max_context=max_context,
        # The published runtime removes SigLIP's final block, then selects the
        # last remaining hidden state.  This loader constructs that trimmed
        # 26-block tower directly, so the equivalent index is -1 rather than
        # the untrimmed checkpoint configuration's -2.
        vision_feature_layer=-1,
    )
    backbone.to(device=target_device)
    backbone.requires_grad_(False).eval()
    generation = json.loads((root / "generation_config.json").read_text(encoding="utf-8"))
    eos_token_ids = generation.get("eos_token_id", [151645, 151643])
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    return StreamVLNCheckpoint(
        backbone=backbone,
        tokenizer=tokenizer,
        image_processor=image_processor,
        eos_token_ids=tuple(int(token_id) for token_id in eos_token_ids),
    )


__all__ = ["StreamVLNCheckpoint", "load_streamvln_checkpoint"]
