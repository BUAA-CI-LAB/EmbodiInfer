"""Pi0.5 batch representation and checkpoint-specific input/output processing.

LeRobot and OpenPI share Pi05Batch and the same decoder. Their image resizing,
tokenization, normalization and action restoration retain their own training
conventions; serving calls the common processor interface.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image

    from .checkpoints.openpi import OpenPiConfig


@dataclass
class Pi05Batch:
    """A collated pi0.5 batch, ready for ``PI05Pytorch`` embedding.

    ``images`` / ``img_masks`` are per-camera lists (one entry per camera slot,
    including zero-padded empty cameras); ``tokens`` / ``masks`` are the language
    token ids and attention mask (state encoded upstream when configured).
    """

    images: list[torch.Tensor]  # each [B, 3, H, W]
    img_masks: list[torch.Tensor]  # each [B] bool
    tokens: torch.Tensor  # [B, L] long
    masks: torch.Tensor  # [B, L] bool
    request_ids: list[str] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> Pi05Batch:
        imgs = [img.to(device, non_blocking=True) for img in self.images]
        if dtype is not None:  # float images cast to model dtype; tokens/masks stay
            imgs = [img.to(dtype) for img in imgs]
        return Pi05Batch(
            images=imgs,
            img_masks=[m.to(device, non_blocking=True) for m in self.img_masks],
            tokens=self.tokens.to(device, non_blocking=True),
            masks=self.masks.to(device, non_blocking=True),
            request_ids=self.request_ids,
        )

    @classmethod
    def concatenate(cls, batches: Sequence[Pi05Batch]) -> Pi05Batch:
        """Join checkpoint-prepared rows without retokenizing or changing masks.

        Camera slots and sequence shapes must match the processor's fixed
        configuration. The single-row path preserves its existing tensors.
        """
        if not batches:
            raise ValueError("cannot concatenate an empty pi05 batch")
        if len(batches) == 1:
            return batches[0]
        cameras = len(batches[0].images)
        if any(len(b.images) != cameras or len(b.img_masks) != cameras for b in batches):
            raise ValueError("pi05 batches have different camera layouts")
        return cls(
            images=[torch.cat([b.images[i] for b in batches]) for i in range(cameras)],
            img_masks=[torch.cat([b.img_masks[i] for b in batches]) for i in range(cameras)],
            tokens=torch.cat([b.tokens for b in batches]),
            masks=torch.cat([b.masks for b in batches]),
            request_ids=[request_id for b in batches for request_id in b.request_ids],
        )

    @classmethod
    def from_lerobot_batch(cls, lerobot_policy, batch: dict) -> Pi05Batch:
        """Build a Pi05Batch from a LeRobot-style batch dict.

        Reuses the loaded policy's ``_preprocess_images`` (SigLIP resize/pad +
        [-1, 1] normalization + empty-camera padding) so the inputs are byte-for-
        byte what ``PI05Policy.predict_action_chunk`` would feed the model. The
        language ``tokens`` / ``masks`` come from the standard LeRobot constants.
        """
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, img_masks = lerobot_policy._preprocess_images(batch)
        return cls(
            images=images,
            img_masks=img_masks,
            tokens=batch[OBS_LANGUAGE_TOKENS],
            masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
        )


class Pi05Processor(Protocol):
    """Checkpoint-specific transforms consumed through one serving interface.

    Images are decoded PIL inputs; resized tensors use CHW layout in [0, 1].
    State preparation validates/pads physical features. Model preprocessing and
    action restoration always receive the state belonging to the same request.
    """

    def prepare_state(self, state: torch.Tensor) -> torch.Tensor:
        """Validate or pad physical state features before model preprocessing."""
        ...

    def resize_image(self, image: Image.Image, width: int, height: int) -> np.ndarray:
        """Return a resized CHW float image in [0, 1]."""
        ...

    def prepare(self, state: torch.Tensor, images: Mapping[str, torch.Tensor], prompt: str) -> Pi05Batch:
        """Build the common model input from one observation."""
        ...

    def restore_actions(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return a physical action chunk using this request's reference state."""
        ...


def make_processor(policy: Any, checkpoint: str) -> Pi05Processor:
    """Choose training semantics once, independently of the weight storage format."""
    config = getattr(policy, "openpi_config", None)
    if config is not None:
        return OpenPiProcessor(config)
    return LeRobotProcessor(policy._lerobot, checkpoint)


class LeRobotProcessor:
    """Wrap the checkpoint's native LeRobot pre/postprocessing pipeline."""

    def __init__(self, policy: Any, checkpoint: str) -> None:
        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
        except ImportError as error:
            raise ImportError("pi05 processor requires `pip install lerobot==0.5.1`") from error

        _ensure_pi05_processor_compatibility()
        self._policy = policy
        self._state_key = OBS_STATE
        self._tokens_key = OBS_LANGUAGE_TOKENS
        self._mask_key = OBS_LANGUAGE_ATTENTION_MASK
        self._state_dim = int(policy.config.input_features[OBS_STATE].shape[0])
        self._action_dim = int(policy.config.output_features["action"].shape[0])
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                **_local_tokenizer_override(checkpoint),
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def prepare_state(self, state: torch.Tensor) -> torch.Tensor:
        """Preserve LeRobot serving's zero padding of omitted trailing features."""
        if state.ndim != 1 or state.numel() > self._state_dim:
            raise ValueError("state_dim too large for pi05 checkpoint")
        return torch.nn.functional.pad(state, (0, self._state_dim - state.numel()))

    def resize_image(self, image: Image.Image, width: int, height: int) -> np.ndarray:
        """Preserve native image geometry before checkpoint normalization."""
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        pixels = torch.from_numpy(array)
        if pixels.shape[:2] != (height, width):
            from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

            pixels = resize_with_pad_torch(pixels.unsqueeze(0), height, width)[0]
        return pixels.permute(2, 0, 1).contiguous().numpy()

    def prepare(self, state: torch.Tensor, images: Mapping[str, torch.Tensor], prompt: str) -> Pi05Batch:
        """Run checkpoint normalization/tokenization and native image preparation."""
        processed = self._preprocessor({self._state_key: state, "task": prompt, **images})
        if not isinstance(processed, Mapping):
            raise TypeError("preprocessed must be an object")
        if self._tokens_key not in processed:
            raise RuntimeError("pi05 preprocessor did not return language tokens")
        if self._mask_key not in processed:
            raise RuntimeError("pi05 preprocessor did not return language masks")
        return Pi05Batch.from_lerobot_batch(self._policy, dict(processed))

    def restore_actions(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Preserve row-wise native postprocessing and the external action width."""
        rows = []
        for row in actions[:, : self._action_dim]:
            post = torch.as_tensor(self._postprocessor(row[None]))
            if post.ndim == 2 and post.shape[0] == 1:
                post = post[0]
            if post.ndim != 1:
                raise RuntimeError("pi05 postprocessor returned invalid action shape")
            rows.append(post)
        return torch.stack(rows)


class OpenPiProcessor:
    """OpenPI transforms driven by checkpoint assets, with no mutable episode state.

    Delta actions are restored against the state passed for the same request,
    not the state of whichever session happened to run its preprocessor last.
    """

    def __init__(self, config: OpenPiConfig) -> None:
        try:
            import sentencepiece
        except ImportError as error:
            raise ImportError(
                "OpenPI tokenization requires "
                "`uv sync --python 3.12 --frozen --no-dev --group pi05 --group pi05-openpi`"
            ) from error
        self.config = config
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=config.tokenizer)
        data = json.loads(Path(config.norm_stats).read_text())
        self.statistics = {}
        keys = ("q01", "q99") if config.use_quantile_norm else ("mean", "std")
        for feature in ("state", "actions"):
            stats = data["norm_stats"][feature]
            arrays = tuple(np.asarray(stats[key], dtype=np.float32) for key in keys)
            if any(array.shape != (config.action_dim,) or not np.isfinite(array).all() for array in arrays):
                raise ValueError(f"OpenPI {feature} statistics must be finite action_dim vectors")
            if config.use_quantile_norm:
                if np.any(arrays[1] < arrays[0]):
                    raise ValueError(f"OpenPI {feature} q99 must be >= q01")
            elif np.any(arrays[1] < 0):
                raise ValueError(f"OpenPI {feature} std must be nonnegative")
            self.statistics[feature] = arrays

    def prepare_state(self, state: torch.Tensor) -> torch.Tensor:
        """Require every physical state feature before internal model padding."""
        if tuple(state.shape) != (self.config.state_dim,):
            raise ValueError("OpenPI observation must provide every checkpoint state feature")
        return state

    def resize_image(self, image: Image.Image, width: int, height: int) -> np.ndarray:
        """Apply OpenPI's antialiased uint8 resize, rounding and letterboxing."""
        pixels = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
        ratio = max(image.width / width, image.height / height)
        resized_height, resized_width = int(image.height / ratio), int(image.width / ratio)
        resized = (
            torch.nn.functional.interpolate(
                pixels[None].float(),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            .round()
            .clamp(0, 255)
        )
        top, left = (height - resized_height) // 2, (width - resized_width) // 2
        padded = torch.nn.functional.pad(
            resized, (left, width - resized_width - left, top, height - resized_height - top)
        )
        return (padded / 255.0).numpy()

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Reproduce OpenPI's separate newline token and optional state prompt."""
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        if self.config.discrete_state_input:
            discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            text = f"Task: {text}, State: {' '.join(map(str, discrete))};\nAction: "
            tokens = self.tokenizer.encode(text, add_bos=True)
        else:
            tokens = self.tokenizer.encode(text, add_bos=True) + self.tokenizer.encode("\n")
        length = min(len(tokens), self.config.max_token_len)
        tokens = tokens[:length] + [0] * (self.config.max_token_len - length)
        mask = [True] * length + [False] * (self.config.max_token_len - length)
        return torch.tensor([tokens], dtype=torch.long), torch.tensor([mask], dtype=torch.bool)

    def prepare(self, state: torch.Tensor, images: Mapping[str, torch.Tensor], prompt: str) -> Pi05Batch:
        """Normalize a single observation and build the common engine batch."""
        if tuple(state.shape) != (self.config.state_dim,) or not torch.isfinite(state).all():
            raise ValueError("OpenPI state does not match checkpoint state_dim")
        padded = np.pad(state.cpu().numpy(), (0, self.config.action_dim - self.config.state_dim))
        first, second = self.statistics["state"]
        normalized = (
            (padded - first) / (second - first + 1e-6) * 2.0 - 1.0
            if self.config.use_quantile_norm
            else (padded - first) / (second + 1e-6)
        )
        tokens, masks = self.tokenize(prompt, normalized)
        tensors = []
        for key in self.config.image_keys:
            if key not in images:
                raise ValueError(f"missing OpenPI camera {key!r}")
            image = images[key]
            if tuple(image.shape) != (3, 224, 224) or not torch.isfinite(image).all():
                raise ValueError(f"invalid OpenPI camera {key!r}")
            tensors.append((image * 2.0 - 1.0).unsqueeze(0))
        return Pi05Batch(tensors, [torch.ones(1, dtype=torch.bool) for _ in tensors], tokens, masks)

    def restore_actions(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Unnormalize a full chunk and restore masked absolute action dimensions."""
        values = actions.detach().float().cpu().numpy()
        if values.ndim != 2 or values.shape[1] != self.config.action_dim or not np.isfinite(values).all():
            raise ValueError("invalid OpenPI action chunk")
        first, second = self.statistics["actions"]
        restored = (
            (values + 1.0) / 2.0 * (second - first + 1e-6) + first
            if self.config.use_quantile_norm
            else values * (second + 1e-6) + first
        )
        count = len(self.config.delta_action_mask)
        restored[:, :count] += np.where(self.config.delta_action_mask, state.cpu().numpy()[:count], 0)
        return torch.from_numpy(restored[:, : self.config.output_action_dim].copy())


def _local_tokenizer_override(checkpoint: str) -> dict[str, dict[str, Any]]:
    """Load the tokenizer without network access for a local checkpoint."""

    root = Path(checkpoint)
    if not root.is_dir():
        return {}

    config_path = root / "policy_preprocessor.json"
    try:
        pipeline = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid local pi05 preprocessor config: {config_path}") from error

    tokenizer_name = None
    for step in pipeline.get("steps", ()):
        if isinstance(step, Mapping) and step.get("registry_name") == "tokenizer_processor":
            config = step.get("config")
            if isinstance(config, Mapping):
                tokenizer_name = config.get("tokenizer_name")
            break
    if not isinstance(tokenizer_name, str) or not tokenizer_name.strip():
        raise ValueError(f"local pi05 preprocessor has no tokenizer name: {config_path}")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception as error:
        raise RuntimeError(
            f"tokenizer {tokenizer_name!r} is not available locally for checkpoint {checkpoint!r}"
        ) from error
    if not set(tokenizer.get_vocab().values()).difference(tokenizer.all_special_ids):
        raise ValueError(
            f"tokenizer {tokenizer_name!r} for checkpoint {checkpoint!r} contains no text tokens; "
            "restore the tokenizer vocabulary in the local Hugging Face cache "
            "and check HF_HOME before starting the service"
        )
    return {"tokenizer_processor": {"tokenizer": tokenizer}}


def _ensure_pi05_processor_compatibility() -> None:
    """Register the processor names serialized by supported pi0.5 checkpoints."""

    importlib.import_module("lerobot.policies.pi05.processor_pi05")
    from lerobot.processor import ProcessorStepRegistry, RelativeActionsProcessorStep

    try:
        ProcessorStepRegistry.get("relative_actions_processor")
    except KeyError:

        class _RelativeActionsProcessorCompatibilityStep(RelativeActionsProcessorStep):
            pass

        ProcessorStepRegistry.register("relative_actions_processor")(
            _RelativeActionsProcessorCompatibilityStep
        )
