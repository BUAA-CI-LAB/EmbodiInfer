"""Model-side HTTP serving adapter for DM0.5."""

from __future__ import annotations

import hashlib
import io
import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ...engine.core import EngineCore
from ...engine.serve.contracts import (
    ModelAction,
    ModelResult,
    RawImage,
    RawPolicyRequest,
    ServingAdapter,
)
from ...types import Observation
from .modeling_dm05 import DM05Policy
from .processor_dm05 import pack_images

DM05_ACTION_SPACE = "dm05.action_chunk.v1"
DM05_ACTION_REPRESENTATION = "absolute_eef_xyzrpy_with_absolute_gripper"
DM05_STATE_REPRESENTATION = "eef_xyzrpy_gripper"
DM05_VIEW_ORDER = ("cam_global", "cam_side", "cam_arm")
DM05_STATE_DESCRIPTION = ("eef", "eef", "eef", "eef", "eef", "eef", "gripper")
DM05_ACTION_FEATURE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
DM05_HISTORY_FRAMES = 5
DM05_HISTORY_TOKENS_PER_SLOT = 16
DM05_HISTORY_IMAGE_TOKEN = "<unused0>"
DM05_HISTORY_PAD_TOKEN = "<unused1>"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _finite_vector(value: object, name: str, width: int) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError(f"{name} must not contain booleans")
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric sequence") from error
    if result.shape != (width,):
        raise ValueError(f"{name} must have shape [{width}]; got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _load_image(image: RawImage) -> Image.Image:
    if image.mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError(f"unsupported image type {image.mime_type!r}")
    try:
        with Image.open(io.BytesIO(image.data)) as opened:
            if opened.width * opened.height > 4096 * 4096:
                raise ValueError(f"{image.name} image has too many pixels")
            opened.load()
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"{image.name} is not a valid encoded image") from error


@dataclass(frozen=True, slots=True)
class DM05ServingConfig:
    view_order: tuple[str, ...] = DM05_VIEW_ORDER
    state_representation: str = DM05_STATE_REPRESENTATION
    state_description: tuple[str, ...] = DM05_STATE_DESCRIPTION
    action_feature_names: tuple[str, ...] = DM05_ACTION_FEATURE_NAMES
    history_frames: int = DM05_HISTORY_FRAMES
    history_tokens_per_slot: int = DM05_HISTORY_TOKENS_PER_SLOT

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> DM05ServingConfig:
        config = _mapping(value or {}, "adapter_config")

        def names(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = config.get(key, default)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ValueError(f"{key} must be a sequence")
            result = tuple(str(item) for item in raw)
            if not result or any(not item.strip() for item in result):
                raise ValueError(f"{key} must contain non-empty names")
            return result

        history_frames = config.get("history_frames", DM05_HISTORY_FRAMES)
        history_tokens_per_slot = config.get("history_tokens_per_slot", DM05_HISTORY_TOKENS_PER_SLOT)
        if isinstance(history_frames, bool) or not isinstance(history_frames, int):
            raise ValueError("history_frames must be an integer")
        if isinstance(history_tokens_per_slot, bool) or not isinstance(history_tokens_per_slot, int):
            raise ValueError("history_tokens_per_slot must be an integer")
        if history_frames <= 0 or history_tokens_per_slot <= 0:
            raise ValueError("history dimensions must be positive")
        return cls(
            view_order=names("view_order", DM05_VIEW_ORDER),
            state_representation=str(config.get("state_representation", DM05_STATE_REPRESENTATION)),
            state_description=names("state_description", DM05_STATE_DESCRIPTION),
            action_feature_names=names("action_feature_names", DM05_ACTION_FEATURE_NAMES),
            history_frames=history_frames,
            history_tokens_per_slot=history_tokens_per_slot,
        )


class DM05ServingAdapter(ServingAdapter):
    """Convert raw HTTP observations to DM0.5 and return only public actions."""

    action_space = DM05_ACTION_SPACE

    def __init__(
        self,
        *,
        core: EngineCore,
        checkpoint: str | None = None,
        config: DM05ServingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        del checkpoint
        if not isinstance(core.policy, DM05Policy):
            raise TypeError("DM0.5 serving requires a DM05Policy")
        if int(core.policy.config.output_action_dim) != len(DM05_ACTION_FEATURE_NAMES):
            raise ValueError("ARX5 DM0.5 serving requires seven public action dimensions")
        self._core = core
        self._policy = core.policy
        self._config = (
            config if isinstance(config, DM05ServingConfig) else DM05ServingConfig.from_mapping(config)
        )
        if self._config.view_order != DM05_VIEW_ORDER:
            raise ValueError(f"view_order must be {DM05_VIEW_ORDER!r}")
        if self._config.state_representation != DM05_STATE_REPRESENTATION:
            raise ValueError(f"state_representation must be {DM05_STATE_REPRESENTATION!r}")
        if self._config.state_description != DM05_STATE_DESCRIPTION:
            raise ValueError(f"state_description must be {DM05_STATE_DESCRIPTION!r}")
        if self._config.action_feature_names != DM05_ACTION_FEATURE_NAMES:
            raise ValueError(f"action_feature_names must be {DM05_ACTION_FEATURE_NAMES!r}")
        if len(self._config.view_order) != len(self._policy.runtime.image_prompts):
            raise ValueError("view_order must match the DM0.5 checkpoint image prompt count")
        if len(self._config.state_description) != self._policy.config.output_action_dim:
            raise ValueError("state_description must match the public action dimension")
        if len(self._config.action_feature_names) != self._policy.config.output_action_dim:
            raise ValueError("action_feature_names must match the public action dimension")
        if getattr(self._policy.runtime, "is_history", None) is not True:
            raise ValueError("DM0.5 HTTP serving requires a history-enabled policy runtime")
        self._lock = threading.Lock()
        self._history_by_session: dict[str, deque[Image.Image]] = {}

    def capabilities(self) -> dict[str, Any]:
        return {
            "model": "dm05",
            "action_space": self.action_space,
            "view_order": list(self._config.view_order),
            "state_representation": self._config.state_representation,
            "state_description": list(self._config.state_description),
            "action_feature_names": list(self._config.action_feature_names),
            "action_representation": DM05_ACTION_REPRESENTATION,
            "action_horizon": int(self._policy.config.action_horizon),
            "output_action_dim": int(self._policy.config.output_action_dim),
            "internal_action_dim_exposed": False,
            "execution_mode": "serialized_b1",
            "max_batch_size": 1,
            "history_mode": "logical_step",
            "history_frames": self._config.history_frames,
            "history_tokens_per_slot": self._config.history_tokens_per_slot,
        }

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        state = self._state(request.state)
        images = self._images(request.images, request.metadata)
        num_steps = request.metadata.get("num_steps", self._policy.config.default_num_steps)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or not 1 <= num_steps <= 100:
            raise ValueError("metadata.num_steps must be an integer in [1, 100]")
        speed = request.metadata.get("speed", "0.5")
        if isinstance(speed, bool) or not isinstance(speed, (str, int, float)):
            raise ValueError("metadata.speed must be a string or finite number")
        if isinstance(speed, (int, float)) and not math.isfinite(float(speed)):
            raise ValueError("metadata.speed must be finite")
        if isinstance(speed, str) and not speed.strip():
            raise ValueError("metadata.speed must not be empty")
        control_mode = request.metadata.get("control_mode")
        if control_mode is not None and not isinstance(control_mode, str):
            raise ValueError("metadata.control_mode must be a string or null")

        with self._lock:
            history = list(self._history_by_session.get(request.session_id, ()))
            pixels, image_sizes = pack_images(images)
            observation = Observation(
                images=pixels,
                state=state,
                instruction_tokens=torch.empty(0, dtype=torch.long),
                instruction=request.instruction,
                metadata={
                    "image_sizes": image_sizes,
                    "history_images": [image.copy() for image in history],
                    "history_placeholder_text": self._history_placeholder(len(history)),
                    "robot_type": "ARX5",
                    "speed": speed,
                    "control_mode": control_mode,
                    "state_desc": list(self._config.state_description),
                },
            )
            batch = self._policy.collate([observation], [request.request_id])
            chunks = self._core.execute(batch, num_steps=num_steps)
            if len(chunks) != 1:
                raise RuntimeError(f"DM0.5 returned {len(chunks)} action chunks for one request")
            chunk = chunks[0]
            actions = torch.as_tensor(chunk.actions).detach().to(torch.float32).cpu()
            expected = (
                int(self._policy.config.action_horizon),
                int(self._policy.config.output_action_dim),
            )
            if tuple(actions.shape) != expected:
                raise RuntimeError(
                    f"DM0.5 public actions must have shape {expected}; got {tuple(actions.shape)}"
                )
            if not torch.isfinite(actions).all():
                raise RuntimeError("DM0.5 public actions contain NaN or Inf")
            session_history = self._history_by_session.setdefault(
                request.session_id, deque(maxlen=self._config.history_frames)
            )
            session_history.append(images[0].copy())
        timing = {
            str(name): float(value)
            for name, value in (chunk.timing or {"policy_ms": chunk.latency_ms}).items()
        }
        return ModelResult(
            action_space=self.action_space,
            actions=(
                ModelAction(
                    kind="action_chunk",
                    values={
                        "data": actions.tolist(),
                        "feature_names": list(self._config.action_feature_names),
                        "representation": DM05_ACTION_REPRESENTATION,
                        "output_transform_applied": True,
                        "internal_action_dim_exposed": False,
                    },
                ),
            ),
            timing=timing,
            policy_revision=str(self._core.policy_version),
        )

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._history_by_session.pop(session_id, None)

    def _history_placeholder(self, active_frames: int) -> str:
        inactive_frames = self._config.history_frames - active_frames
        return (
            DM05_HISTORY_PAD_TOKEN * (self._config.history_tokens_per_slot * inactive_frames)
            + (DM05_HISTORY_IMAGE_TOKEN * self._config.history_tokens_per_slot + "\n") * active_frames
        )

    def _state(self, raw: Mapping[str, Any]) -> np.ndarray:
        state = _mapping(raw, "state")
        if state.get("representation") != self._config.state_representation:
            raise ValueError(f"state.representation must be {self._config.state_representation!r}")
        if tuple(state.get("state_desc", ())) != self._config.state_description:
            raise ValueError(f"state.state_desc must be {self._config.state_description!r}")
        for name in ("source", "units"):
            value = state.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"state.{name} must be a non-empty string")
        return _finite_vector(state.get("values"), "state.values", len(self._config.state_description))

    def _images(self, raw: tuple[RawImage, ...], metadata: Mapping[str, Any]) -> list[Image.Image]:
        by_name: dict[str, RawImage] = {}
        for image in raw:
            if image.name not in self._config.view_order:
                raise ValueError(f"unknown DM0.5 view {image.name!r}")
            if image.name in by_name:
                raise ValueError(f"duplicate DM0.5 view {image.name!r}")
            by_name[image.name] = image
        missing = [name for name in self._config.view_order if name not in by_name]
        if missing:
            raise ValueError(f"missing DM0.5 views: {missing}")
        digests = {hashlib.sha256(by_name[name].data).digest() for name in self._config.view_order}
        synthetic = metadata.get("synthetic_views")
        if not isinstance(synthetic, bool):
            raise ValueError("metadata.synthetic_views must be an explicit boolean")
        if len(digests) == 1 and not synthetic:
            raise ValueError("identical DM0.5 views require metadata.synthetic_views=true")
        return [_load_image(by_name[name]) for name in self._config.view_order]


__all__ = [
    "DM05_ACTION_FEATURE_NAMES",
    "DM05_ACTION_REPRESENTATION",
    "DM05_ACTION_SPACE",
    "DM05_HISTORY_FRAMES",
    "DM05_HISTORY_IMAGE_TOKEN",
    "DM05_HISTORY_PAD_TOKEN",
    "DM05_HISTORY_TOKENS_PER_SLOT",
    "DM05_STATE_DESCRIPTION",
    "DM05_STATE_REPRESENTATION",
    "DM05_VIEW_ORDER",
    "DM05ServingAdapter",
    "DM05ServingConfig",
]
