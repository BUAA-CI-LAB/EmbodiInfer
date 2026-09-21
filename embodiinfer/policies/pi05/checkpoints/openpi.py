"""OpenPI Pi0.5 metadata and strict loading into the common Torch model."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPiConfig:
    """Inference semantics missing from an OpenPI parameter-only export.

    ``image_keys`` is the ordered model camera layout. ``delta_action_mask``
    selects dimensions relative to the observation at the START of each chunk;
    it does not describe cumulative deltas. Asset paths are checkpoint-relative.
    """

    action_horizon: int
    state_dim: int
    output_action_dim: int
    image_keys: tuple[str, ...]
    tokenizer: str
    norm_stats: str
    use_quantile_norm: bool
    discrete_state_input: bool
    delta_action_mask: tuple[bool, ...]
    action_dim: int = 32
    max_token_len: int = 200
    num_inference_steps: int = 10
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], root: Path) -> OpenPiConfig:
        """Validate explicit training semantics and resolve required assets."""
        data = dict(value)
        for key in ("image_keys", "delta_action_mask"):
            if not isinstance(data.get(key), (tuple, list)):
                raise ValueError(f"OpenPI {key} must be a list")
            data[key] = tuple(data[key])
        config = cls(**data)
        for key in (
            "action_horizon",
            "state_dim",
            "output_action_dim",
            "action_dim",
            "max_token_len",
            "num_inference_steps",
        ):
            number = getattr(config, key)
            if type(number) is not int or number <= 0:
                raise ValueError(f"OpenPI {key} must be a positive integer")
        if config.state_dim > config.action_dim or config.output_action_dim > config.action_dim:
            raise ValueError("OpenPI state/action features exceed the model action_dim")
        if config.dtype not in {"float32", "bfloat16"}:
            raise ValueError("OpenPI dtype must be float32 or bfloat16")
        if (config.paligemma_variant, config.action_expert_variant) != ("gemma_2b", "gemma_300m"):
            raise ValueError("OpenPI loading currently supports gemma_2b + gemma_300m only")
        for key in ("use_quantile_norm", "discrete_state_input"):
            if type(getattr(config, key)) is not bool:
                raise ValueError(f"OpenPI {key} must be a boolean")
        if (
            not config.image_keys
            or any(not isinstance(key, str) or not key.strip() for key in config.image_keys)
            or len(set(config.image_keys)) != len(config.image_keys)
        ):
            raise ValueError("OpenPI image_keys must contain unique nonempty names in model order")
        if any(type(flag) is not bool for flag in config.delta_action_mask) or len(
            config.delta_action_mask
        ) > min(config.state_dim, config.output_action_dim):
            raise ValueError("OpenPI delta_action_mask exceeds state/action features or is not boolean")
        for key in ("tokenizer", "norm_stats"):
            name = getattr(config, key)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"OpenPI {key} must name a local asset")
            path = (root / name).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"OpenPI {key} asset is missing: {path}")
            data[key] = str(path)
        return cls(**data)

    def native_config(self) -> Any:
        """Describe the common Torch modules without using LeRobot processors."""
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        return PI05Config(
            device="cpu",
            dtype=self.dtype,
            paligemma_variant=self.paligemma_variant,
            action_expert_variant=self.action_expert_variant,
            chunk_size=self.action_horizon,
            n_action_steps=self.action_horizon,
            max_state_dim=self.action_dim,
            max_action_dim=self.action_dim,
            num_inference_steps=self.num_inference_steps,
            tokenizer_max_length=self.max_token_len,
            input_features={
                **{
                    key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                    for key in self.image_keys
                },
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.output_action_dim,))
            },
        )


def load_openpi_checkpoint(
    root: Path,
    *,
    load_device: str | None = None,
    checkpoint_config: Mapping[str, Any] | None = None,
    use_orbax: bool,
) -> tuple[Any, OpenPiConfig]:
    """Load either OpenPI storage format with the same explicit processor recipe.

    Orbax tensors are converted and cached before strict Torch loading. Missing
    active parameters fail instead of leaving random model initialization.
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if checkpoint_config is None:
        path = root / "openpi_config.json"
        if not path.is_file():
            raise ValueError(
                "OpenPI weights need openpi_config.json or checkpoint_config with "
                "the training image layout, normalization, tokenizer and action transforms"
            )
        checkpoint_config = json.loads(path.read_text())
    recipe = OpenPiConfig.from_mapping(checkpoint_config, root)
    # Validate processor assets before spending time restoring large tensors.
    from ..processor_pi05 import OpenPiProcessor

    OpenPiProcessor(recipe)
    if use_orbax:
        from .orbax import cached_orbax_weights

        weights = cached_orbax_weights(root)
    else:
        weights = root / "model.safetensors"
    from safetensors.torch import load_file

    model = PI05Policy(recipe.native_config())
    tensors = load_file(str(weights), device="cpu")
    # OpenPI saves the inner model, LeRobot normally prefixes these with model.
    if tensors and all(key.startswith("model.") for key in tensors):
        tensors = {key.removeprefix("model."): tensor for key, tensor in tensors.items()}
    # These vocabulary output heads are not used by Pi0.5's flow decoder.
    # OpenPI Orbax contains neither; do not invent random missing parameters.
    # The prefix input embedding remains required and is checked strictly.
    for tower in ("paligemma", "gemma_expert"):
        getattr(model.model.paligemma_with_expert, tower).lm_head = None
        tensors.pop(f"paligemma_with_expert.{tower}.lm_head.weight", None)
    # Never use LeRobot's broad from_pretrained exception handler for this path:
    # a missing tensor must not leave randomly initialized model parameters.
    model.model.load_state_dict(tensors, strict=True)
    del tensors
    logger.info("Loaded %s checkpoint: %s", "openpi-orbax" if use_orbax else "openpi-pytorch", root)
    return model.eval().to(load_device or "cpu"), recipe
