"""Direct loader for the published StreamVLN sharded checkpoint."""

from __future__ import annotations

import json
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...layers.linear import INT8Config
from ..linear import INT8Linear
from .backbone import StreamVLNBackbone, StreamVLNProjector, _PackedLinear

_LOGGER = logging.getLogger(__name__)

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


def _load_int8_checkpoint_weights(
    root: Path,
    llm: torch.nn.Module,
    vision_tower: torch.nn.Module,
    projector: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    config: INT8Config,
) -> tuple[str, ...]:
    """Convert one projection at a time, never materializing the full BF16 model.

    Per-output-channel conversion commutes with QKV/gate-up concatenation.
    Conversion runs on CPU for identical stored weights across CUDA platforms;
    only the final packed tensors move to the serving device. Floating leaves
    and explicitly ignored projections retain the requested checkpoint dtype.
    """
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    groups = (
        (llm, ""),
        (vision_tower, "model.vision_tower.vision_tower."),
        (projector, "model.mm_projector."),
    )
    expected = {prefix + key for module, prefix in groups for key in module.state_dict()}
    if expected != set(index):
        raise RuntimeError(
            f"StreamVLN checkpoint key mismatch; missing={sorted(expected - set(index))[:8]}, "
            f"unexpected={sorted(set(index) - expected)[:8]}"
        )
    loaded: set[str] = set()

    def read(key: str, shape: torch.Size) -> torch.Tensor:
        path = root / index[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            if tensor.shape != shape:
                raise RuntimeError(f"StreamVLN tensor shape mismatch: {key}: {tensor.shape} != {shape}")
            # Own the CPU storage before closing mmap and evicting filesystem pages.
            owned = tensor.to(dtype=dtype, copy=True)
            del tensor
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            with suppress(OSError), path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        loaded.add(key)
        return owned

    quantized = []
    for i, layer in enumerate(llm.model.layers):
        for owner, target, attributes, section in (
            (layer.self_attn, "qkv_proj", ("q_proj", "k_proj", "v_proj"), "self_attn"),
            (layer.self_attn, "o_proj", ("o_proj",), "self_attn"),
            (layer.mlp, "gate_up_proj", ("gate_proj", "up_proj"), "mlp"),
            (layer.mlp, "down_proj", ("down_proj",), "mlp"),
        ):
            name = f"llm.layers.{i}.{section}.{target}"
            ignored = config.is_ignored(name)
            parts = []
            for attribute in attributes:
                original = getattr(owner, attribute)
                prefix = f"model.layers.{i}.{section}.{attribute}"
                state = {
                    key: read(f"{prefix}.{key}", value.shape) for key, value in original.state_dict().items()
                }
                original.load_state_dict(state, assign=True)
                del state
                parts.append(original if ignored else INT8Linear.from_linear(original, config))
                delattr(owner, attribute)
                del original
            if ignored:
                packed = _PackedLinear(*parts) if len(parts) > 1 else parts[0]
            else:
                # Unlike INT8Linear.pack, Qwen QKV has biases. Preserve them.
                bias = (
                    torch.cat([part.bias.detach() for part in parts]) if parts[0].bias is not None else None
                )
                packed = INT8Linear(
                    torch.cat([part.weight.detach() for part in parts], dim=1).contiguous(),
                    torch.cat([part.weight_scale for part in parts]),
                    None if bias is None else torch.nn.Parameter(bias, requires_grad=False),
                    compute_dtype=dtype,
                    backend=config.backend,
                    split_sizes=tuple(part.out_features for part in parts) if len(parts) > 1 else None,
                )
                quantized.append(name)
            setattr(owner, target, packed.to(device))
            del packed, parts
        _LOGGER.info("StreamVLN INT8: loaded transformer layer %d/%d", i + 1, len(llm.model.layers))

    # Assign floating multimodal leaves, embeddings, normalization, and LM head
    # individually. No state dictionary retains a second copy of a full shard.
    for module, prefix in groups:
        for key, parameter in list(module.named_parameters()):
            if not parameter.is_meta:
                continue
            source = prefix + key
            tensor = read(source, parameter.shape).to(device)
            owner_name, _, leaf = key.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            setattr(owner, leaf, torch.nn.Parameter(tensor, requires_grad=False))
    if loaded != expected:
        raise RuntimeError(f"StreamVLN tensors not loaded: {sorted(expected - loaded)[:8]}")
    return tuple(quantized)


def load_streamvln_checkpoint(
    checkpoint: str | Path,
    *,
    dtype: str = "bfloat16",
    max_context: int = 32768,
    load_device: str | torch.device = "cpu",
    int8_config: INT8Config | None = None,
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

    quantized_names = None
    if int8_config is None:
        _load_checkpoint_weights(root, llm, vision_tower, projector, torch_dtype, target_device)
    else:
        quantized_names = _load_int8_checkpoint_weights(
            root, llm, vision_tower, projector, torch_dtype, target_device, int8_config
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
    if quantized_names is not None:
        backbone._projections_fused = True
        backbone.configure_quantized_projections(quantized_names)
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
