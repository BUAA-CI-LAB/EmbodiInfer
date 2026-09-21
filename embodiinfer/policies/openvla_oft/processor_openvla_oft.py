"""Input processing for OpenVLA-OFT: Prismatic image transform + Llama tokenisation.

Produces the exact ``input_ids`` (BOS + prompt ending in the space token 29871, left-padded)
and 6-channel ``pixel_values`` ([DINOv2-norm ‖ SigLIP-norm], resize-naive to 224, bicubic)
that the model consumes. The prompt template mirrors RLinf:
``"In: What action should the robot take to {task}?\nOut: "``.

Model-level parity against RLinf is verified by injecting identical ``pixel_values`` /
``input_ids`` into both pipelines; the image transform here is validated separately.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TVF

PROMPT = "In: What action should the robot take to {task}?\nOut: "
SPACE_TOKEN = 29871


@dataclass
class OpenVLAOFTBatch:
    input_ids: torch.Tensor  # [B, L] long, left-padded, last real token = 29871
    attention_mask: torch.Tensor  # [B, L] long
    pixel_values: torch.Tensor  # [B, 6, 224, 224] float
    request_ids: list[str]

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> OpenVLAOFTBatch:
        px = self.pixel_values.to(device, non_blocking=True)
        if dtype is not None:
            px = px.to(dtype)
        return OpenVLAOFTBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            pixel_values=px,
            request_ids=self.request_ids,
        )

    def pad(self, target_batch_size: int) -> OpenVLAOFTBatch:
        b = self.batch_size
        if target_batch_size == b:
            return self
        n = target_batch_size - b

        def rep(x):
            return torch.cat([x, x[-1:].expand(n, *x.shape[1:])], dim=0)

        return OpenVLAOFTBatch(
            rep(self.input_ids),
            rep(self.attention_mask),
            rep(self.pixel_values),
            self.request_ids + [f"__pad_{i}" for i in range(n)],
        )


class OpenVLAOFTProcessor:
    def __init__(self, tokenizer, means, stds, image_size: int = 224, max_length: int = 50):
        self.tokenizer = tokenizer
        # index 0 = primary featurizer (DINOv2), index 1 = fused (SigLIP)
        self.means = [torch.tensor(m, dtype=torch.float32) for m in means]
        self.stds = [torch.tensor(s, dtype=torch.float32) for s in stds]
        self.image_size = image_size
        self.max_length = max_length

    @classmethod
    def from_checkpoint(cls, checkpoint: str, max_length: int = 50) -> OpenVLAOFTProcessor:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
        with open(os.path.join(checkpoint, "preprocessor_config.json")) as f:
            pp = json.load(f)
        return cls(tok, pp["means"], pp["stds"], image_size=pp["input_sizes"][0][-1], max_length=max_length)

    # ---- image: resize-naive to square, per-backbone normalise, 6-channel stack
    def _transform_image(self, img: torch.Tensor) -> torch.Tensor:
        """``img`` [3, H, W] float in [0, 1] -> [6, 224, 224] ([DINOv2-norm ‖ SigLIP-norm])."""
        img = TVF.resize(
            img,
            [self.image_size, self.image_size],
            interpolation=TVF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        chans = [TVF.normalize(img, mean=m.tolist(), std=s.tolist()) for m, s in zip(self.means, self.stds)]
        return torch.cat(chans, dim=0)  # [6, 224, 224]

    # ---- text: prompt -> tokens ending in the space token, left-padded -------
    def _tokenize(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        text = PROMPT.format(task=(instruction or "").lower())
        ids = self.tokenizer(text, add_special_tokens=True).input_ids  # BOS + prompt
        if len(ids) == 0 or ids[-1] != SPACE_TOKEN:
            ids = ids + [SPACE_TOKEN]
        ids = ids[: self.max_length]
        pad = self.max_length - len(ids)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = [pad_id] * pad + ids
        attn = [0] * pad + [1] * len(ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

    def collate(self, observations, request_ids) -> OpenVLAOFTBatch:
        ids, masks, pix = [], [], []
        for o in observations:
            i, m = self._tokenize(o.instruction)
            ids.append(i)
            masks.append(m)
            pix.append(self._transform_image(o.images[0]))  # first (main) camera
        return OpenVLAOFTBatch(
            input_ids=torch.stack(ids),
            attention_mask=torch.stack(masks),
            pixel_values=torch.stack(pix),
            request_ids=list(request_ids),
        )
