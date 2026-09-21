"""DM0.5 adapter using OpenDM's checkpoint loader and exact eager forward."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...types import Observation, validate_observation
from ..base import FlowVLAPolicy, PrefixState, VLAPolicy
from ..factory import register_policy
from .config import DM05PolicyConfig
from .processor_dm05 import (
    DM05Batch,
    image_to_pil,
    install_robochallenge_tokenizer,
    observation_images,
    pad_prefix_tensors,
)

_INSTALL = (
    "DM0.5 requires a dedicated OpenDM environment because its torch/transformers "
    "versions differ from other vvla adapters. Install https://github.com/dexmal/opendm "
    "in that environment, then install vvla with --no-deps."
)

_ROBOCHALLENGE_ROBOTS = frozenset({"ARX5", "UR5", "ALOHA", "W1"})


def _pretrained_source(checkpoint: str) -> str:
    """Return the Hugging Face/OpenDM source for a local checkpoint path.

    OpenDM loads both the model config and weights with ``from_pretrained``.
    A local ``model.safetensors`` file is a valid operator-facing checkpoint
    reference, but Transformers needs its containing directory to discover the
    adjacent ``config.json``. Hub ids and directory paths remain unchanged.
    """

    path = Path(checkpoint).expanduser()
    return str(path.parent) if path.is_file() else checkpoint


def _clone_prefix_cache(cache: Any):
    """Clone a Transformers cache without deepcopying non-leaf tensors."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return copy.deepcopy(cache)
    cloned = copy.copy(cache)
    cloned.layers = []
    for layer in layers:
        cloned_layer = copy.copy(layer)
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                # Preserve autograd when GRPO re-encodes the prefix under
                # torch.enable_grad().  Inference and behavior rollout already
                # run under torch.no_grad(), so their cloned cache remains
                # detached without imposing a model-specific training policy.
                setattr(cloned_layer, name, value.clone())
        cloned.layers.append(cloned_layer)
    return cloned


@dataclass
class DM05Prefix:
    """Read-only VLM KV cache and per-request output conversion context."""

    cache: Any
    input_ids: torch.Tensor
    prefix_len: int
    states: list[np.ndarray]
    meta_data: list[dict[str, Any]]

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    def to(self, device: torch.device | str) -> DM05Prefix:
        # The cache is created after the batch move and therefore is already on
        # the policy device.  Keeping it immutable also avoids a hidden copy in
        # every denoise step.
        return DM05Prefix(self.cache, self.input_ids.to(device), self.prefix_len, self.states, self.meta_data)

    def expand(self, num_samples: int) -> DM05Prefix:
        if num_samples == 1:
            return self
        cache = _clone_prefix_cache(self.cache)
        repeat = getattr(cache, "batch_repeat_interleave", None)
        if repeat is None:
            raise TypeError(f"{type(cache).__name__} cannot expand DM0.5 prefix cache for best-of-N rollout")
        repeat(num_samples)
        return DM05Prefix(
            cache=cache,
            input_ids=self.input_ids.repeat_interleave(num_samples, dim=0),
            prefix_len=self.prefix_len,
            states=[state for state in self.states for _ in range(num_samples)],
            meta_data=[meta for meta in self.meta_data for _ in range(num_samples)],
        )


class DM05Policy(FlowVLAPolicy):
    """OpenDM DM0.5 as a vvla flow policy.

    The VLM prefix is computed once.  Each flow step calls OpenDM's action
    expert helpers with the cached prefix.  Internal state remains 32-wide;
    only the robot-facing 7/14 dimensions enter rollout log-prob and are sliced
    and denormalized once after integration.
    """

    def __init__(
        self,
        config: DM05PolicyConfig,
        *,
        checkpoint: str | None = None,
        norm_stats: str | None = None,
        robot_type: str = "ARX5",
        image_prompts: list[str] | None = None,
        action_mode: str = "relative",
        add_state: bool = True,
        is_history: bool = False,
        n_bins: int = 256,
        model_max_length: int = 1024,
        runtime: Any | None = None,
        llm_attention: str = "sdpa",
        vision_attention: str = "sdpa",
        action_attention: str = "sdpa",
        prompt_style: str = "opendm",
        liger_kernel: bool = False,
    ):
        super().__init__(config)
        if runtime is None:
            runtime = self._load_runtime(
                checkpoint=checkpoint,
                norm_stats=norm_stats,
                robot_type=robot_type,
                image_prompts=image_prompts,
                action_mode=action_mode,
                add_state=add_state,
                is_history=is_history,
                n_bins=n_bins,
                model_max_length=model_max_length,
                llm_attention=llm_attention,
                vision_attention=vision_attention,
                action_attention=action_attention,
                output_action_dim=config.output_action_dim,
                action_horizon=config.action_horizon,
                num_steps=config.default_num_steps,
                liger_kernel=liger_kernel,
                prompt_style=prompt_style,
            )
        self.runtime = runtime
        self.robot_type = robot_type
        self.action_mode = action_mode
        # Register the loaded module exactly once so .to(), state_dict(), and
        # weight synchronization operate on the OpenDM parameters.
        self.model = runtime.model
        try:
            from opendm.model.dm05.dm05_lora import unwrap_dm05_model
        except ImportError:

            def unwrap_dm05_model(model):
                return model

        object.__setattr__(self, "_dm", unwrap_dm05_model(self.model))
        model_cfg = self._dm.model.config
        if int(model_cfg.action_dim) != config.internal_action_dim:
            raise ValueError(
                "DM0.5 internal action dimension disagrees with the checkpoint: "
                f"config={config.internal_action_dim}, checkpoint={model_cfg.action_dim}"
            )
        if int(model_cfg.chunk_size) != config.action_horizon:
            raise ValueError(
                "DM0.5 action horizon disagrees with the checkpoint: "
                f"config={config.action_horizon}, checkpoint={model_cfg.chunk_size}"
            )

    @staticmethod
    def _load_runtime(**kwargs):
        checkpoint = kwargs["checkpoint"]
        norm_stats = kwargs["norm_stats"]
        if checkpoint is None or norm_stats is None:
            raise ValueError("DM0.5 needs both checkpoint and norm_stats")
        pretrained_source = _pretrained_source(checkpoint)
        try:
            from opendm.exp.dm05_exp import DM05InferenceConfig, DM05ModelConfig
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(_INSTALL) from exc

        model_cfg = DM05ModelConfig(
            model_name_or_path=pretrained_source,
            chunk_size=kwargs["action_horizon"],
            llm_attn_implementation=kwargs["llm_attention"],
            vision_attn_implementation=kwargs["vision_attention"],
            action_attn_implementation=kwargs["action_attention"],
            liger_kernel=kwargs["liger_kernel"],
            vlm_gradient_checkpointing=False,
            ae_gradient_checkpointing=False,
        )
        model = DM05Policy._build_base_model(model_cfg)
        infer = DM05InferenceConfig(
            diffusion_steps=kwargs["num_steps"],
            output_action_dim=kwargs.get("output_action_dim", 7),
            image_prompts=kwargs["image_prompts"] or ["Head", "Left wrist", "Right wrist"],
            backend="default",
        )
        infer.default_robot_type = kwargs["robot_type"]
        infer._initialize(
            model=model,
            model_name_or_path=pretrained_source,
            norm_stats_path=norm_stats,
            n_bins=kwargs["n_bins"],
            model_max_length=kwargs["model_max_length"],
            use_absolute_action=kwargs["action_mode"] == "relative",
            add_state=kwargs["add_state"],
            is_history=kwargs["is_history"],
        )
        if kwargs["prompt_style"] == "robochallenge":
            install_robochallenge_tokenizer(infer)
        elif kwargs["prompt_style"] != "opendm":
            raise ValueError("prompt_style must be 'opendm' or 'robochallenge'")
        return infer

    @staticmethod
    def _build_base_model(model_cfg):
        """Load a base checkpoint with attention backends applied before init.

        OpenDM applies ``DM05ModelConfig`` attention choices only after
        ``from_pretrained``.  Table30 checkpoints persist FlashAttention2 in
        their nested vision config, so an explicit SDPA request would otherwise
        fail during model construction when ``flash_attn`` is not installed.
        """
        try:
            from opendm.model.dm05.dm05_arch import (
                DM05Config,
                DM05ForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(_INSTALL) from exc

        config = DM05Config.from_pretrained(model_cfg.model_name_or_path)
        for attr, value in model_cfg._config_overrides().items():
            setattr(config, attr, value)
        config.vlm_config._attn_implementation = model_cfg.llm_attn_implementation
        config.vlm_config.text_config._attn_implementation = model_cfg.llm_attn_implementation
        config.vlm_config.vision_config._attn_implementation = model_cfg.vision_attn_implementation
        model = DM05ForConditionalGeneration.from_pretrained(
            model_cfg.model_name_or_path,
            config=config,
            torch_dtype=model_cfg._torch_dtype(),
        )
        model_cfg._apply_runtime_model_options(model)
        model_cfg._apply_full_model_runtime_optimizations(model)
        return model

    def build_serving_adapter(self, *, core, checkpoint=None, config=None):
        """Build the model-side adapter used by the generic VVLA HTTP service."""

        from .serving import DM05ServingAdapter

        return DM05ServingAdapter(core=core, checkpoint=checkpoint, config=config)

    @property
    def supports_cuda_graph(self) -> bool:
        return True

    def cuda_graph_variant(self, prefix: PrefixState) -> tuple[int, int]:
        if not isinstance(prefix, DM05Prefix):
            raise TypeError(f"DM05Policy expected DM05Prefix; got {type(prefix).__name__}")
        return prefix.prefix_len, int(prefix.input_ids.shape[1])

    def allocate_static_prefix_from_live(
        self,
        prefix: PrefixState,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        variant: object | None,
    ) -> DM05Prefix:
        if not isinstance(prefix, DM05Prefix):
            raise TypeError(f"DM05Policy expected DM05Prefix; got {type(prefix).__name__}")
        if prefix.batch_size != batch_size or variant != self.cuda_graph_variant(prefix):
            raise ValueError("DM0.5 graph prefix layout does not match capture")
        capture_device = torch.device(device)
        if capture_device.type == "cuda" and capture_device.index is None:
            capture_device = torch.device("cuda", torch.cuda.current_device())
        if prefix.input_ids.device != capture_device:
            raise ValueError("DM0.5 graph prefix must already be on the capture device")
        cache = _clone_prefix_cache(prefix.cache)
        layers = getattr(cache, "layers", None)
        if layers is None:
            raise TypeError("DM0.5 graph cache does not expose layers")
        for layer in layers:
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device != capture_device
                    or value.dtype != dtype
                ):
                    raise ValueError(f"DM0.5 graph-cache {name} must match capture device and dtype")
        return DM05Prefix(
            cache, prefix.input_ids.clone(), prefix.prefix_len, list(prefix.states), list(prefix.meta_data)
        )

    def copy_prefix_into(self, dst: PrefixState, src: PrefixState) -> None:
        if not isinstance(dst, DM05Prefix) or not isinstance(src, DM05Prefix):
            raise TypeError("DM0.5 graph prefix copies require DM05Prefix instances")
        if self.cuda_graph_variant(dst) != self.cuda_graph_variant(src):
            raise ValueError("DM0.5 graph prefix layouts differ")
        dst_layers, src_layers = getattr(dst.cache, "layers", None), getattr(src.cache, "layers", None)
        if dst_layers is None or src_layers is None or len(dst_layers) != len(src_layers):
            raise ValueError("DM0.5 graph prefix cache layers differ")
        for dl, sl in zip(dst_layers, src_layers, strict=True):
            for name in ("keys", "values"):
                dv, sv = getattr(dl, name, None), getattr(sl, name, None)
                if (
                    not isinstance(dv, torch.Tensor)
                    or not isinstance(sv, torch.Tensor)
                    or dv.shape != sv.shape
                ):
                    raise ValueError(f"DM0.5 graph-cache {name} shapes differ")
                dv.copy_(sv)
        dst.input_ids.copy_(src.input_ids)

    def flow_schedule(self, num_steps: int) -> list[tuple[float, float]]:
        dt = -1.0 / num_steps
        return [(1.0 + i * dt, dt) for i in range(num_steps)]

    @torch.no_grad()
    def sample_actions(
        self,
        batch: DM05Batch,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the same eager DM0.5 decoder used by ``EngineCore``."""

        steps = num_steps or self.config.default_num_steps
        prefix = self.encode_prefix(batch)
        state = (
            x0
            if x0 is not None
            else self.new_noise(
                batch.batch_size,
                generator=generator,
            )
        )
        expected = (
            batch.batch_size,
            self.config.action_horizon,
            self.config.internal_action_dim,
        )
        if tuple(state.shape) != expected:
            raise ValueError(f"DM0.5 initial flow state must have shape {expected}; got {tuple(state.shape)}")
        return self.decoder.produce_chunk(
            state,
            prefix,
            steps,
            batch.batch_size,
            None,
        )

    def collate(self, observations: list[Observation], request_ids: list[str]) -> DM05Batch:
        """Apply OpenDM preprocessing to shared observations and per-request metadata."""
        if len(observations) != len(request_ids):
            raise ValueError("DM0.5 needs one request_id per observation")
        samples = []
        states = []
        metadata = []
        expected_views = len(self.runtime.image_prompts)
        for observation in observations:
            validate_observation(observation, self.config)
            if len(observation.images) != expected_views:
                raise ValueError(
                    f"DM0.5 expects {expected_views} current camera images; got {len(observation.images)}"
                )
            state = observation.state.detach().cpu().numpy().astype(np.float32, copy=False)
            options = observation.metadata
            robot_type = options.get("robot_type", self.robot_type)
            meta = {
                "robot_type": robot_type,
                "speed": str(options.get("speed", "0.5")),
                "control_mode": options.get("control_mode"),
                "state_desc": options.get("state_desc") or self._default_state_desc(robot_type),
            }
            data = {
                "images": observation_images(observation),
                "history_images": [image_to_pil(image) for image in options.get("history_images", [])],
                "prompt": observation.instruction,
                "state": state,
                "meta_data": meta,
            }
            if options.get("history_placeholder_text") is not None:
                data["history_placeholder_text"] = options["history_placeholder_text"]
            samples.append(self.runtime.input_transform(data))
            states.append(state)
            metadata.append(meta)

        tokenizer = self.runtime.processor.tokenizer
        tensors = pad_prefix_tensors(
            samples,
            pad_token_id=int(tokenizer.pad_token_id),
            padding_side=str(tokenizer.padding_side),
        )
        return DM05Batch(
            **tensors,
            states=states,
            meta_data=metadata,
            request_ids=list(request_ids),
        )

    def _default_state_desc(self, robot_type: str | None = None) -> list[str] | None:
        if self.action_mode != "relative":
            return None
        if (robot_type or self.robot_type).upper() in {"ARX5", "UR5"}:
            return ["eef"] * 6 + ["gripper"]
        if self.config.output_action_dim == 14:
            return ["joint"] * 6 + ["gripper"] + ["joint"] * 6 + ["gripper"]
        return ["joint"] * (self.config.output_action_dim - 1) + ["gripper"]

    def encode_prefix(self, batch: DM05Batch, memory=None) -> DM05Prefix:
        del memory
        cache, prefix_len = self._dm._compute_prefix_cache(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            token_type_ids=batch.token_type_ids,
            history_pixel_values=batch.history_pixel_values,
            history_mask=batch.history_mask,
        )
        return DM05Prefix(cache, batch.input_ids, int(prefix_len), batch.states, batch.meta_data)

    def denoise_step(self, x_t: torch.Tensor, t: torch.Tensor, prefix: DM05Prefix) -> torch.Tensor:
        from opendm.model.dm05.dm05_arch import HISTORY_PAD_TOKEN_ID, make_suffix_attn_mask

        x_t = x_t * self.flow_logprob_mask(x_t)
        suffix_embeds = self._dm.model.action_in_proj(x_t)
        adarms_cond = self._dm._build_adarms_cond(t, suffix_embeds.dtype)
        suffix_len = int(suffix_embeds.shape[1])
        invisible = (HISTORY_PAD_TOKEN_ID,)
        attention_mask = make_suffix_attn_mask(
            input_ids=prefix.input_ids,
            prefix_len=prefix.prefix_len,
            suffix_len=suffix_len,
            batch_size=x_t.shape[0],
            device=x_t.device,
            dtype=suffix_embeds.dtype,
            pad_token_id=self._dm.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible,
        )
        position_ids = self._dm._build_suffix_position_ids(
            prefix.prefix_len,
            suffix_len,
            x_t.device,
            input_ids=prefix.input_ids,
            pad_token_id=self._dm.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible,
        )
        suffix_out = self._dm._suffix_forward(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix.cache,
            adarms_cond=adarms_cond,
        )
        return self._dm.model.action_out_proj(suffix_out)

    def flow_logprob_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Exclude padded columns from the shared flow transition likelihood."""
        mask = torch.zeros((1, 1, state.shape[-1]), device=state.device, dtype=state.dtype)
        mask[..., : self.config.output_action_dim] = 1
        return mask

    def flow_state_shape(self, batch_size: int) -> tuple[int, int, int]:
        """Keep OpenDM's padded state throughout eager, graph, and rollout decoding."""
        return batch_size, self.config.action_horizon, self.config.internal_action_dim

    def finalize_actions(self, actions: torch.Tensor, prefix: PrefixState) -> torch.Tensor:
        """Slice public columns and apply OpenDM's output transform once after decoding."""
        if not isinstance(prefix, DM05Prefix):
            raise TypeError(f"DM05Policy expected DM05Prefix; got {type(prefix).__name__}")
        normalized = actions[..., : self.config.output_action_dim]
        converted = []
        for row, state, meta in zip(normalized, prefix.states, prefix.meta_data, strict=True):
            output = self.runtime.output_transform(
                {
                    "action": row.detach().to(torch.float32).cpu().numpy(),
                    "state": state,
                    "meta_data": meta,
                }
            )["action"]
            converted.append(torch.as_tensor(output, device=actions.device, dtype=actions.dtype))
        result = torch.stack(converted)
        self.validate_output_actions(result)
        return result

    def validate_output_actions(self, actions: torch.Tensor) -> None:
        """Reject internal or malformed action tensors at the DM0.5 boundary."""

        expected = (self.config.action_horizon, self.config.output_action_dim)
        if actions.ndim != 3 or tuple(actions.shape[-2:]) != expected:
            raise ValueError(
                f"DM0.5 must return actions [B, H, A] with (H, A)={expected}; got {tuple(actions.shape)}"
            )


@register_policy("dm05")
def _build_dm05(
    checkpoint: str | None = None,
    norm_stats: str | None = None,
    robot_type: str = "ARX5",
    output_action_dim: int = 7,
    **overrides,
) -> VLAPolicy:
    # The generic HTTP launcher supplies this common loading hint. DM0.5 is
    # moved by EngineCore after OpenDM constructs it, so no model-specific
    # device branch belongs in the model-neutral serving layer.
    overrides.pop("load_device", None)
    policy_keys = {
        "action_horizon",
        "default_num_steps",
        "dtype",
        "internal_action_dim",
    }
    cfg_kwargs = dict(
        name="dm05",
        action_dim=output_action_dim,
        action_horizon=50,
        default_num_steps=10,
        internal_action_dim=32,
        output_action_dim=output_action_dim,
    )
    cfg_kwargs.update({key: overrides.pop(key) for key in list(overrides) if key in policy_keys})
    cfg = DM05PolicyConfig(**cfg_kwargs)
    return DM05Policy(
        cfg,
        checkpoint=checkpoint,
        norm_stats=norm_stats,
        robot_type=robot_type,
        prompt_style=overrides.pop(
            "prompt_style",
            "robochallenge" if robot_type.upper() in _ROBOCHALLENGE_ROBOTS else "opendm",
        ),
        **overrides,
    )
