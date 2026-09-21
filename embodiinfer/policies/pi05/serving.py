"""Model-side serving adapter for pi0.5 (raw HTTP -> raw action chunk)."""

from __future__ import annotations

import io
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from embodiinfer.engine.core import EngineCore
from embodiinfer.engine.serve.contracts import (
    ModelAction,
    ModelResult,
    RawImage,
    RawPolicyRequest,
    ServingAdapter,
)
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch, Pi05Processor, make_processor
from embodiinfer.types import ActionChunk

PI05_ACTION_SPACE = "pi05.action_chunk.v1"


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _feature_width(features: Mapping[str, Any], key: str, name: str) -> int:
    feature = features.get(key)
    shape = getattr(feature, "shape", None)
    if not isinstance(shape, Sequence) or len(shape) != 1:
        raise ValueError(f"pi05 checkpoint has invalid {name} feature {key!r}")
    width = int(shape[0])
    if width <= 0:
        raise ValueError(f"pi05 checkpoint has invalid {name} width")
    return width


def _flatten_numbers(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("state values must be numeric")
    if isinstance(value, Mapping):
        out: list[float] = []
        for item in value.values():
            out.extend(_flatten_numbers(item))
        return tuple(out)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return tuple(out)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("state values must be numeric")
    return (float(value),)


def _image_name(value: str) -> str:
    stem = value.rsplit("/", 1)[-1]
    lowered = stem.lower()
    for suffix in (".jpeg", ".jpg", ".png"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_raw_image(image: RawImage, width: int, height: int, *, processor: Pi05Processor) -> np.ndarray:
    """Decode a wire image; checkpoint-specific resizing belongs to the processor."""
    if image.mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError(f"unsupported image type {image.mime_type!r}")
    if not image.data:
        raise ValueError(f"{image.name} has empty image payload")
    try:
        with Image.open(io.BytesIO(image.data)) as opened:
            if opened.width * opened.height > 2_000_000:
                raise ValueError(f"{image.name} image has too many pixels")
            array = processor.resize_image(opened, width, height)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"{image.name} is not a valid encoded image") from error
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"{image.name} is not an RGB image")
    return array


@dataclass(frozen=True, slots=True)
class Pi05ServingConfig:
    state_fields: tuple[str, ...]
    image_fields: tuple[str, ...]
    return_steps: int
    image_keys: Mapping[str, str] = field(default_factory=dict)
    action_feature_names: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> Pi05ServingConfig:
        cfg = _mapping(value, "adapter_config")
        state_fields = cfg.get("state_fields")
        if not state_fields:
            raise ValueError("state_fields is required for pi05 serving")
        if isinstance(state_fields, (str, bytes)) or not isinstance(state_fields, Sequence):
            raise ValueError("state_fields must be a non-empty string list")
        image_fields = cfg.get("image_fields", ())
        if isinstance(image_fields, (str, bytes)) or not isinstance(image_fields, Sequence):
            raise ValueError("image_fields must be a sequence")
        if not image_fields:
            raise ValueError("image_fields is required for pi05 serving")
        parsed_image_fields = tuple(_identifier(item, "image_field") for item in image_fields)
        image_keys = cfg.get("image_keys", {})
        if not isinstance(image_keys, Mapping):
            raise TypeError("image_keys must be an object")
        parsed_image_keys = {
            _identifier(source, "image_keys source"): _identifier(target, "image_keys target")
            for source, target in image_keys.items()
        }
        unknown_image_keys = sorted(set(parsed_image_keys) - set(parsed_image_fields))
        if unknown_image_keys:
            names = ", ".join(repr(name) for name in unknown_image_keys)
            raise ValueError(f"image_keys contains fields not listed in image_fields: {names}")
        return cls(
            state_fields=tuple(_identifier(value, "state_field") for value in state_fields),
            image_fields=parsed_image_fields,
            return_steps=max(1, int(cfg.get("return_steps", 1))),
            image_keys=parsed_image_keys,
            action_feature_names=tuple(
                str(name)
                for name in cfg.get("action_feature_names", ())
                if isinstance(name, str) and name.strip()
            ),
        )


class Pi05ServingAdapter(ServingAdapter):
    action_space = PI05_ACTION_SPACE

    def __init__(
        self,
        *,
        core: EngineCore,
        checkpoint: str,
        config: Pi05ServingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._core = core
        policy = core.policy
        if not hasattr(policy, "_lerobot"):
            raise TypeError("pi05 serving requires a pi05 policy with LeRobot state")
        self._policy = policy
        self._checkpoint = checkpoint
        self._config = Pi05ServingConfig.from_mapping(config if config is not None else {})
        self._processor = make_processor(policy, checkpoint)
        self._policy_cfg = policy._lerobot.config
        self._native_image_keys = tuple(self._policy_cfg.image_features)
        self._state_dim = _feature_width(self._policy_cfg.input_features, "observation.state", "state")
        self._action_dim = _feature_width(self._policy_cfg.output_features, "action", "action")
        self._max_state_dim = int(self._policy_cfg.max_state_dim)
        self._max_action_dim = int(self._policy_cfg.max_action_dim)
        self._width = int(self._policy_cfg.image_resolution[0])
        self._height = int(self._policy_cfg.image_resolution[1])
        self._lock = threading.Lock()

    def capabilities(self) -> dict[str, Any]:
        return {
            "model": "pi05",
            "action_space": self.action_space,
            "state_fields": list(self._config.state_fields),
            "image_fields": list(self._config.image_fields),
            "image_keys": dict(self._config.image_keys),
            "action_feature_names": list(self._config.action_feature_names),
            "return_steps": self._config.return_steps,
            "action_dim": self._action_dim,
            "state_dim": self._state_dim,
            "max_action_dim": self._max_action_dim,
            "max_state_dim": self._max_state_dim,
            "image_resolution": [self._width, self._height],
            "execution_mode": "serialized_b1",
            "max_batch_size": 1,
        }

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Execute a single request through the same checkpoint transforms."""
        result = self.infer_batch([request])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def infer_batch(self, requests: Sequence[RawPolicyRequest]) -> list[ModelResult | Exception]:
        """Prepare independent requests, execute one tensor batch, scatter results.

        Invalid observations fail only their own row. Each result is restored
        against that request's state, including relative-action checkpoints.
        """
        outcomes: list[ModelResult | Exception] = [RuntimeError("unresolved batch row") for _ in requests]
        with self._lock:
            prepared = []
            rows = []
            for index, request in enumerate(requests):
                try:
                    state_vec = self._state_vector(request.state)
                    images, names = self._image_tensor_stack(request.images)
                    named_images = {name: images[i] for i, name in enumerate(names)}
                    batch = self._processor.prepare(state_vec, named_images, request.instruction)
                    if batch.batch_size != 1:
                        raise ValueError("pi05 processor must prepare one row per request")
                    batch.request_ids = [request.request_id]
                    prepared.append(batch)
                    rows.append((index, state_vec))
                except (TypeError, ValueError) as error:
                    outcomes[index] = error
            if not prepared:
                return outcomes
            chunks = self._core.execute(Pi05Batch.concatenate(prepared))
            if len(chunks) != len(rows):
                raise RuntimeError("pi05 model returned an incorrect number of action chunks")
            for (index, state_vec), chunk in zip(rows, chunks, strict=True):
                try:
                    outcomes[index] = self._result(chunk, state_vec)
                except (TypeError, ValueError, RuntimeError) as error:
                    outcomes[index] = error
        return outcomes

    def _result(self, chunk: ActionChunk, state_vec: torch.Tensor) -> ModelResult:
        rows = max(1, min(int(self._config.return_steps), int(chunk.actions.shape[0])))
        action_rows = self._processor.restore_actions(chunk.actions[:rows], state_vec)
        actions = []
        for index in range(action_rows.shape[0]):
            post = action_rows[index]
            if post.ndim == 2 and post.shape[0] == 1:
                post = post[0]
            if post.ndim != 1:
                raise RuntimeError("pi05 postprocessor returned invalid action shape")
            if not torch.all(torch.isfinite(post)):
                raise RuntimeError("pi05 postprocessor returned non-finite values")
            values = [float(item) for item in post.tolist()]
            actions.append(values)
        return ModelResult(
            action_space=self.action_space,
            actions=(
                ModelAction(
                    kind="action_chunk",
                    values={
                        "data": actions,
                        "feature_names": list(self._config.action_feature_names),
                    },
                ),
            ),
            timing={"policy_ms": float(chunk.latency_ms)},
            policy_revision=str(self._core.policy_version),
        )

    def reset(self, session_id: str) -> None:
        # pi0.5 policy itself is stateless across sessions in deploy mode
        _ = session_id

    def _state_vector(self, state: Mapping[str, Any]) -> torch.Tensor:
        """Read literal feature keys first, retaining legacy nested-path support."""

        values: list[float] = []
        for state_field in self._config.state_fields:
            if state_field in state:
                current = state[state_field]
            else:
                current = state
                for part in state_field.split("."):
                    if not isinstance(current, Mapping) or part not in current:
                        raise ValueError(f"state is missing required field {state_field!r}")
                    current = current[part]
            values.extend(_flatten_numbers(current))
        if len(values) > self._state_dim:
            raise ValueError("state_dim too large for pi05 checkpoint")
        tensor = self._processor.prepare_state(torch.as_tensor(values, dtype=torch.float32))
        if not torch.isfinite(tensor).all():
            raise ValueError("state contains non-finite values")
        return tensor

    def _image_tensor_stack(self, images: tuple[RawImage, ...]) -> tuple[torch.Tensor, tuple[str, ...]]:
        requested: dict[str, RawImage] = {}
        for image in images:
            source_name = _image_name(image.name)
            if source_name in requested:
                raise ValueError(f"duplicate image field {source_name!r}")
            requested[source_name] = image
        native_names = tuple(self._config.image_keys.get(name, name) for name in self._config.image_fields)
        for source_name, native_name in zip(self._config.image_fields, native_names, strict=True):
            if source_name not in requested:
                raise ValueError(f"missing requested image field {source_name!r}")
            if native_name not in self._native_image_keys:
                raise ValueError(
                    f"image field {source_name!r} maps to unsupported pi05 feature {native_name!r}"
                )
        if len(native_names) != len(set(native_names)):
            raise ValueError("multiple image fields map to the same pi05 feature")
        np_arrays = [
            _load_raw_image(
                requested[name],
                width=self._width,
                height=self._height,
                processor=self._processor,
            )
            for name in self._config.image_fields
        ]
        stacked = np.stack(np_arrays, axis=0)
        if stacked.ndim != 4 or stacked.shape[1:] != (3, self._height, self._width):
            raise RuntimeError("internal image layout mismatch")
        return torch.from_numpy(stacked), native_names
