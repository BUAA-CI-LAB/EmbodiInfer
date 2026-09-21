"""Vendored Prismatic fused vision backbone + projector (OpenVLA-OFT).

A line-for-line port of ``PrismaticVisionBackbone`` / ``PrismaticProjector`` from the
OpenVLA-OFT checkpoint's ``modeling_prismatic.py`` (Apache-2.0). This is a "leaf" run
once per observation inside ``encode_prefix`` — embodiinfer loads the checkpoint's
``vision_backbone.*`` / ``projector.*`` weights into these modules and owns the
multimodal assembly + LLM orchestration itself (see ``modeling_openvla_oft.py``).

For ``dinosiglip-vit-so-224px`` the timm ids are ``[dinov2, siglip]`` so ``featurizer``
is DINOv2 (channels 0:3) and ``fused_featurizer`` is SigLIP (channels 3:6); fused feature
order is ``[DINOv2 ‖ SigLIP]`` (2176 = 1024 + 1152). Patch features come from the
second-to-last ViT block (``get_intermediate_layers(n={num_blocks-2})``, unpacked).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import LayerScale


def unpack_tuple(fn: Callable[[Any], tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        # timm's get_intermediate_layers returns a tuple (older) or list (>=1.0) of layer
        # outputs; unpack the single requested layer either way.
        return result[0] if isinstance(result, (tuple, list)) else result

    return wrapper


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    # HF renames params containing "gamma"; mirror the checkpoint's LayerScale patch so the
    # loaded param name (scale_factor) matches.
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class PrismaticVisionBackbone(nn.Module):
    """Fused (DINOv2 + SigLIP) timm vision backbone; features concatenated on the hidden dim."""

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: list[int],
        timm_model_ids: list[str],
        timm_override_act_layers: list[str | None],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1

        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: str | None) -> nn.Module:
        featurizer = timm.create_model(
            model_id, pretrained=False, num_classes=0, img_size=img_size, act_layer=act_layer
        )
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))
        return featurizer

    def _patch_layer_scales(self) -> None:
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
            return torch.cat([patches, patches_fused], dim=2)

        assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"
        images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)
        all_patches = []
        for img in images:
            img_regular, img_fused = torch.split(img, [3, 3], dim=1)
            patches = self.featurizer(img_regular)
            patches_fused = self.fused_featurizer(img_fused)
            all_patches.append(torch.cat([patches, patches_fused], dim=2))
        return torch.cat(all_patches, dim=1)


class PrismaticProjector(nn.Module):
    """Fused-backbone MLP projector: fc1 -> GELU -> fc2 -> GELU -> fc3 (2176 -> 8704 -> 4096 -> 4096)."""

    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            x = self.fc1(img_patches)
            x = self.act_fn1(x)
            x = self.fc2(x)
        else:
            x = self.fc1(img_patches)
            x = self.act_fn1(x)
            x = self.fc2(x)
            x = self.act_fn2(x)
            x = self.fc3(x)
        return x
