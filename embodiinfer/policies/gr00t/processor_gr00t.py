"""GR00T N1.7 batch representation + collation.

GR00T's model input is the Qwen3-VL VLM feature-extractor output — ``input_ids``,
``attention_mask``, ``pixel_values``, ``image_grid_thw`` (multi-camera images are
flattened into the token stream as image tokens) — plus the proprio ``state``
and a per-sample ``embodiment_id`` that selects the category-specific
state/action projector weights. ``Gr00tBatch`` carries exactly that, and is what
the GR00T adapter's ``encode_prefix`` consumes.

The heavy obs -> tensor pipeline (image transforms, tokenization, state
normalization) is GR00T's official ``Gr00tN1d7Processor``; it is preprocessing,
not the denoise loop the engine optimizes, so the adapter reuses it as a boundary
(the way pi0.5 reuses LeRobot's ``_preprocess_images``). ``from_backbone_inputs``
builds a batch straight from that processor's collated output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Keys the Qwen3-VL backbone forward consumes (see Qwen3Backbone.forward).
_BACKBONE_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


@dataclass
class Gr00tBatch:
    """A collated GR00T batch, ready for ``Gr00tN1d7`` embedding.

    ``backbone_inputs`` is the Qwen3-VL feature-extractor dict (``input_ids`` /
    ``attention_mask`` long, ``pixel_values`` float, ``image_grid_thw`` long);
    ``state`` is ``[B, state_history_length, max_state_dim]`` float (already
    normalized upstream); ``embodiment_id`` is ``[B]`` int selecting the
    per-embodiment projector weights.
    """

    backbone_inputs: dict[str, torch.Tensor]
    state: torch.Tensor  # [B, state_history_length, max_state_dim]
    embodiment_id: torch.Tensor  # [B] long
    request_ids: list[str] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.state.shape[0]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> Gr00tBatch:
        def move(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device, non_blocking=True)
            # Cast only floating inputs to the model dtype; ids / masks stay integral.
            if dtype is not None and torch.is_floating_point(x):
                x = x.to(dtype)
            return x

        return Gr00tBatch(
            backbone_inputs={k: move(v) for k, v in self.backbone_inputs.items()},
            state=move(self.state),
            embodiment_id=self.embodiment_id.to(device, non_blocking=True),
            request_ids=self.request_ids,
        )

    @classmethod
    def from_backbone_inputs(
        cls,
        collated: dict,
        embodiment_id: torch.Tensor,
        request_ids: list[str] | None = None,
    ) -> Gr00tBatch:
        """Build a ``Gr00tBatch`` from ``Gr00tN1d7DataCollator`` output.

        ``collated`` is the dict the official processor + collator produce (the
        same ``self.model.get_action(**collated)`` is fed). This slices out the
        backbone keys + ``state`` + ``embodiment_id`` so the vvla adapter drives
        its own two-stage forward over byte-identical inputs.
        """
        backbone_inputs = {k: collated[k] for k in _BACKBONE_KEYS if k in collated}
        return cls(
            backbone_inputs=backbone_inputs,
            state=collated["state"],
            embodiment_id=embodiment_id,
            request_ids=list(request_ids) if request_ids is not None else [],
        )
