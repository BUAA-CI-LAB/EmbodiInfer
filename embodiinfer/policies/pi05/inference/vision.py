"""Derived RLinf SigLIP execution over the existing checkpoint parameters."""

from typing import Any

import torch
import torch.nn.functional as F


class SiglipPlan:
    """Own BF16 casts and packed MHA weights without changing checkpoint modules.

    Rebuild with the policy controller after refit or device migration. The FP32
    stem and positions, BF16 encoder/projector, exact GELU and native MHA calls
    preserve the RLinf vision precision contract.
    """

    @torch.no_grad()
    def __init__(self, paligemma: Any) -> None:
        """Prepare immutable weight views/casts before graph capture."""
        vision = paligemma.vision_tower.vision_model
        patch = vision.embeddings.patch_embedding
        self.patch_weight = patch.weight.detach().float()
        self.patch_bias = None if patch.bias is None else patch.bias.detach().float()
        self.stride, self.padding = patch.stride, patch.padding
        self.positions = vision.embeddings.position_embedding.weight.detach().float()[None]
        self.layers = []
        for layer in vision.encoder.layers:
            attn = layer.self_attn
            width = attn.q_proj.weight.shape[0]
            with torch.device("meta"):
                packed = torch.nn.MultiheadAttention(width, attn.num_heads, batch_first=True)
            state = {
                "in_proj_weight": torch.cat(
                    [p.weight.detach() for p in (attn.q_proj, attn.k_proj, attn.v_proj)]
                ).bfloat16(),
                "in_proj_bias": torch.cat(
                    [p.bias.detach() for p in (attn.q_proj, attn.k_proj, attn.v_proj)]
                ).bfloat16(),
                "out_proj.weight": attn.out_proj.weight.detach().bfloat16(),
                "out_proj.bias": attn.out_proj.bias.detach().bfloat16(),
            }
            packed.load_state_dict(state, strict=True, assign=True)
            packed.requires_grad_(False).eval()
            self.layers.append(
                (
                    self._norm(layer.layer_norm1),
                    packed,
                    self._norm(layer.layer_norm2),
                    self._linear(layer.mlp.fc1),
                    self._linear(layer.mlp.fc2),
                )
            )
        self.final_norm = self._norm(vision.post_layernorm)
        self.projector = self._linear(paligemma.multi_modal_projector.linear)

    @staticmethod
    def _linear(module: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (
            module.weight.detach().bfloat16(),
            None if module.bias is None else module.bias.detach().bfloat16(),
        )

    @staticmethod
    def _norm(module: Any) -> tuple:
        return (
            module.normalized_shape,
            module.weight.detach().bfloat16(),
            module.bias.detach().bfloat16(),
            module.eps,
        )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Encode normalized BCHW images using the original MHA fast path."""
        hidden = F.conv2d(
            images.float().contiguous(memory_format=torch.channels_last),
            self.patch_weight,
            self.patch_bias,
            stride=self.stride,
            padding=self.padding,
        )
        hidden = (hidden.flatten(2).transpose(1, 2) + self.positions).bfloat16()
        for norm1, attention, norm2, first, second in self.layers:
            normalized = F.layer_norm(hidden, *norm1)
            attended, _ = attention(normalized, normalized, normalized)
            hidden = hidden + attended
            hidden = hidden + F.linear(F.gelu(F.linear(F.layer_norm(hidden, *norm2), *first)), *second)
        return F.linear(F.layer_norm(hidden, *self.final_norm), *self.projector)
