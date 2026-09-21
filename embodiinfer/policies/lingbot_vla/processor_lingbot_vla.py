"""LingBot-VLA batch + processor.

Builds the model-ready batch from raw :class:`~embodiinfer.types.Observation`s with the stock
Qwen2.5-VL ``AutoProcessor``: the ``image_processor`` resizes/patchifies each camera into
``pixel_values`` + ``image_grid_thw`` (Qwen packs every image's merged tokens along dim 0),
the tokenizer emits the prompt ``"<bos>{instruction}\\n"`` right-padded to 72, and the proprio
state is padded to the model's 75-dim action space. No lingbotvla-package processor is used —
the layout matches the checkpoint because it is the same Qwen2.5-VL preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...types import Observation


@dataclass
class LingBotVLABatch:
    """A collated, device-movable LingBot-VLA batch."""

    pixel_values: torch.Tensor  # ViT input (Qwen2.5-VL patchified pixels)
    image_grid_thw: torch.Tensor  # [n_img, 3] grid dims for the ViT
    img_pad_masks: torch.Tensor  # [B, n_img_tokens]
    lang_tokens: torch.Tensor  # [B, 72] long
    lang_pad_masks: torch.Tensor  # [B, 72]
    state: torch.Tensor  # [B, 75] normalised + padded proprio
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return self.lang_tokens.shape[0]

    def to(self, device, dtype: torch.dtype | None = None) -> LingBotVLABatch:
        def mv(x: torch.Tensor, cast: bool) -> torch.Tensor:
            x = x.to(device, non_blocking=True)
            return x.to(dtype) if (cast and dtype is not None) else x

        return LingBotVLABatch(
            pixel_values=mv(self.pixel_values, True),
            image_grid_thw=mv(self.image_grid_thw, False),
            img_pad_masks=mv(self.img_pad_masks, False),
            lang_tokens=mv(self.lang_tokens, False),
            lang_pad_masks=mv(self.lang_pad_masks, False),
            state=mv(self.state, True),
            request_ids=self.request_ids,
        )

    def pad(self, target_batch_size: int) -> LingBotVLABatch:
        b = self.batch_size
        if target_batch_size == b:
            return self
        n = target_batch_size - b

        def rep(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, x[-1:].expand(n, *x.shape[1:])], dim=0)

        return LingBotVLABatch(
            pixel_values=self.pixel_values,  # per-image; batched inside the ViT via grid_thw
            image_grid_thw=self.image_grid_thw,
            img_pad_masks=rep(self.img_pad_masks),
            lang_tokens=rep(self.lang_tokens),
            lang_pad_masks=rep(self.lang_pad_masks),
            state=rep(self.state),
            request_ids=self.request_ids + [f"__pad_{i}" for i in range(n)],
        )

    @classmethod
    def from_observations(
        cls,
        observations: list[Observation],
        request_ids: list[str],
        processor,
        action_dim: int = 75,
        max_lang: int = 72,
    ) -> LingBotVLABatch:
        """Assemble a batch from raw observations via the Qwen2.5-VL processor.

        Images ([num_cam, 3, H, W] in [0,1]) go through the Qwen2.5-VL ``image_processor``
        (resize/patchify -> ``pixel_values`` [total_patches, 1176] + ``image_grid_thw``);
        the prompt ``"<bos>{instruction}\\n"`` is right-padded to ``max_lang=72``; state is
        padded to ``action_dim=75``. Qwen packs all images'
        patches along dim 0, so per-observation image tokens are tracked in ``img_pad_masks``.
        """
        import numpy as np
        from PIL import Image

        pvs, grids, imgmasks, langs, lpads, states = [], [], [], [], [], []
        for o in observations:
            imgs = o.images  # [num_cam, 3, H, W] in [0,1]
            pil = [
                Image.fromarray((imgs[c].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                for c in range(imgs.shape[0])
            ]
            vis = processor.image_processor(images=pil, return_tensors="pt")
            pvs.append(vis["pixel_values"])
            grids.append(vis["image_grid_thw"])
            n_img = int(vis["image_grid_thw"].prod(dim=1).sum() // 4)  # merged tokens
            imgmasks.append(torch.ones(n_img))
            tok = processor.tokenizer(
                f"<bos>{o.instruction or ''}\n",
                return_tensors="pt",
                padding="max_length",
                max_length=max_lang,
                truncation=True,
            )
            langs.append(tok["input_ids"][0])
            lpads.append(tok["attention_mask"][0])
            st = torch.zeros(action_dim)
            st[: o.state.shape[0]] = o.state.flatten()[:action_dim]
            states.append(st)
        # Qwen packs image patches along dim 0; grid_thw stacks per image. img_pad_masks are
        # padded to the max per-obs token count so the batch is rectangular for the VL forward.
        max_tok = max(m.shape[0] for m in imgmasks)
        img_pad = torch.stack([torch.cat([m, torch.zeros(max_tok - m.shape[0])]) for m in imgmasks], dim=0)
        return cls(
            pixel_values=torch.cat(pvs, dim=0),
            image_grid_thw=torch.cat(grids, dim=0),
            img_pad_masks=img_pad,
            lang_tokens=torch.stack(langs, dim=0),
            lang_pad_masks=torch.stack(lpads, dim=0),
            state=torch.stack(states, dim=0),
            request_ids=list(request_ids),
        )
