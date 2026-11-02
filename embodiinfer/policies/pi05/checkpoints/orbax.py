"""CPU-only Orbax restoration, Pi0.5 tensor conversion and atomic caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)
_CONVERSION_VERSION = 1


def _flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    result = {}
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            result.update(_flatten(value, name))
        else:
            result[name.removesuffix("/value")] = np.asarray(value, dtype=np.float32)
    return result


def convert_orbax_params(params: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Map full OpenPI Pi0.5 tensors to Torch, without quantizing any parameters.

    Layouts follow Physical-Intelligence/openpi's Apache-2.0 JAX-to-PyTorch
    converter (revision 215abfb217dbac7d5f1273282331b9b1866c0479).
    Each source tensor is consumed exactly once; unsupported trees fail closed.
    """
    source = _flatten(params)
    output = {}

    def take(key: str) -> np.ndarray:
        try:
            return source.pop(key)
        except KeyError as error:
            raise ValueError(f"OpenPI checkpoint is missing tensor {key!r}") from error

    def put(key: str, value: np.ndarray) -> None:
        output[key] = torch.from_numpy(np.ascontiguousarray(value))

    base = "paligemma_with_expert.paligemma.model."
    vision = base + "vision_tower.vision_model."
    put(
        vision + "embeddings.patch_embedding.weight",
        take("PaliGemma/img/embedding/kernel").transpose(3, 2, 0, 1),
    )
    put(vision + "embeddings.patch_embedding.bias", take("PaliGemma/img/embedding/bias"))
    position = take("PaliGemma/img/pos_embedding")
    put(vision + "embeddings.position_embedding.weight", position.reshape(-1, position.shape[-1]))
    block = "PaliGemma/img/Transformer/encoderblock/"
    layers = take(block + "LayerNorm_0/scale")
    vision_layers = layers.shape[0]
    for i in range(vision_layers):
        put(f"{vision}encoder.layers.{i}.layer_norm1.weight", layers[i])
    for source_name, target in (
        ("LayerNorm_0/bias", "layer_norm1.bias"),
        ("LayerNorm_1/scale", "layer_norm2.weight"),
        ("LayerNorm_1/bias", "layer_norm2.bias"),
        ("MlpBlock_0/Dense_0/kernel", "mlp.fc1.weight"),
        ("MlpBlock_0/Dense_0/bias", "mlp.fc1.bias"),
        ("MlpBlock_0/Dense_1/kernel", "mlp.fc2.weight"),
        ("MlpBlock_0/Dense_1/bias", "mlp.fc2.bias"),
    ):
        array = take(block + source_name)
        for i in range(vision_layers):
            put(
                f"{vision}encoder.layers.{i}.{target}",
                array[i].T if source_name.endswith("kernel") else array[i],
            )
    for name, target in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "out")):
        kernels = take(block + f"MultiHeadDotProductAttention_0/{name}/kernel")
        biases = take(block + f"MultiHeadDotProductAttention_0/{name}/bias")
        for i in range(vision_layers):
            kernel = kernels[i]
            matrix = (
                kernel.reshape(-1, kernel.shape[-1]) if name == "out" else kernel.reshape(kernel.shape[0], -1)
            )
            put(f"{vision}encoder.layers.{i}.self_attn.{target}_proj.weight", matrix.T)
            put(f"{vision}encoder.layers.{i}.self_attn.{target}_proj.bias", biases[i].reshape(-1))
    for source_name, target in (("scale", "weight"), ("bias", "bias")):
        put(
            vision + "post_layernorm." + target, take("PaliGemma/img/Transformer/encoder_norm/" + source_name)
        )
    put(base + "multi_modal_projector.linear.weight", take("PaliGemma/img/head/kernel").T)
    put(base + "multi_modal_projector.linear.bias", take("PaliGemma/img/head/bias"))
    put(base + "language_model.embed_tokens.weight", take("PaliGemma/llm/embedder/input_embedding"))

    for suffix, tower in (
        ("", base + "language_model."),
        ("_1", "paligemma_with_expert.gemma_expert.model."),
    ):
        block = "PaliGemma/llm/layers/"
        q = take(block + f"attn/q_einsum{suffix}/w")
        kv = take(block + f"attn/kv_einsum{suffix}/w")
        out = take(block + f"attn/attn_vec_einsum{suffix}/w")
        gate = take(block + f"mlp{suffix}/gating_einsum")
        down = take(block + f"mlp{suffix}/linear")
        for i in range(q.shape[0]):
            prefix = f"{tower}layers.{i}."
            put(prefix + "self_attn.q_proj.weight", q[i].transpose(0, 2, 1).reshape(-1, q.shape[-2]))
            for j, name in enumerate(("k", "v")):
                put(
                    prefix + f"self_attn.{name}_proj.weight",
                    kv[i, j].transpose(0, 2, 1).reshape(-1, kv.shape[-2]),
                )
            put(prefix + "self_attn.o_proj.weight", out[i].reshape(-1, out.shape[-1]).T)
            put(prefix + "mlp.gate_proj.weight", gate[i, 0].T)
            put(prefix + "mlp.up_proj.weight", gate[i, 1].T)
            put(prefix + "mlp.down_proj.weight", down[i].T)
        for name, target in (
            ("pre_attention_norm", "input_layernorm"),
            ("pre_ffw_norm", "post_attention_layernorm"),
        ):
            if suffix:
                weight = take(block + name + suffix + "/Dense_0/kernel")
                bias = take(block + name + suffix + "/Dense_0/bias")
                for i in range(q.shape[0]):
                    put(f"{tower}layers.{i}.{target}.dense.weight", weight[i].T)
                    put(f"{tower}layers.{i}.{target}.dense.bias", bias[i])
            else:
                weight = take(block + name + "/scale")
                for i in range(q.shape[0]):
                    put(f"{tower}layers.{i}.{target}.weight", weight[i])
        if suffix:
            put(tower + "norm.dense.weight", take("PaliGemma/llm/final_norm_1/Dense_0/kernel").T)
            put(tower + "norm.dense.bias", take("PaliGemma/llm/final_norm_1/Dense_0/bias"))
        else:
            put(tower + "norm.weight", take("PaliGemma/llm/final_norm/scale"))
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        put(name + ".weight", take(name + "/kernel").T)
        put(name + ".bias", take(name + "/bias"))
    if source:
        raise ValueError(f"unsupported OpenPI checkpoint tensors: {sorted(source)}")
    return output


def cached_orbax_weights(root: Path) -> Path:
    """Restore on CPU once and atomically cache the mapped safetensors file."""
    from filelock import FileLock

    files = sorted(path for path in (root / "params").rglob("*") if path.is_file())
    fingerprint = [
        (str(path.relative_to(root)), str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    ]
    key = hashlib.sha256(json.dumps([_CONVERSION_VERSION, fingerprint]).encode()).hexdigest()
    # Converted safetensors are cached per checkpoint under the user cache
    # dir; the key covers the conversion recipe and the source file inventory,
    # so a reused cache entry is only returned when the input is unchanged.
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "embodiinfer" / "pi05"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{key}.safetensors"
    with FileLock(str(target) + ".lock"):
        if target.is_file():
            return target
        try:
            import jax
            import orbax.checkpoint as ocp
            from safetensors.torch import save_file
        except ImportError as error:
            raise ImportError(
                "Orbax checkpoint preparation requires "
                "`uv sync --python 3.12 --frozen --no-dev --group pi05 --group pi05-openpi`"
            ) from error
        logger.warning("Converting OpenPI Orbax checkpoint on CPU (first load): %s", root)
        with ocp.PyTreeCheckpointer() as checkpointer:
            metadata = checkpointer.metadata((root / "params").resolve())
            item = {"params": metadata["params"]}
            restored = checkpointer.restore(
                (root / "params").resolve(),
                ocp.args.PyTreeRestore(
                    item=item,
                    restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32), item
                    ),
                ),
            )
        tensors = convert_orbax_params(restored["params"])
        del restored
        fd, temporary = tempfile.mkstemp(prefix=f"{key}.", suffix=".tmp", dir=cache)
        os.close(fd)
        try:
            save_file(tensors, temporary, metadata={"format": "pt", "source_format": "openpi-orbax"})
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        logger.warning("Cached converted OpenPI weights: %s", target)
    return target
