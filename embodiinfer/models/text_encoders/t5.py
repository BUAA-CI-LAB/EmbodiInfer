"""T5 text encoder leaf (default ``google-t5/t5-11b`` = T5-XXL, ``d_model=1024``).

Wraps HuggingFace ``T5EncoderModel`` + ``T5TokenizerFast`` as a vendored leaf and
reproduces cosmos-policy's ``CosmosT5TextEncoder.encode_prompts`` op-for-op: tokenize
with ``padding="max_length"`` / ``truncation`` to ``max_length`` (512), run the encoder,
then **zero the positions beyond each prompt's real length**. Returns ``[B, max_length,
1024]`` — the cross-attention context a video DiT conditions on. The encoder is heavy
(~5.5B for the encoder half), so it is **lazily loaded on first ``encode``** — the Cosmos
policy serves LIBERO tasks from precomputed embeddings and only falls back to this for a
cache miss (see ``policies/cosmos`` text handling).
"""

from __future__ import annotations

import torch
from torch import nn


class T5TextEncoder(nn.Module):
    """Encode instruction strings into a ``[B, max_length, d_model]`` cross-attn context."""

    def __init__(self, model_name: str = "google-t5/t5-11b", device: str = "cuda", dtype=torch.bfloat16):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._tok = None
        self._enc = None

    def _lazy_load(self):
        if self._enc is None:
            from transformers import T5EncoderModel, T5TokenizerFast

            self._tok = T5TokenizerFast.from_pretrained(self.model_name)
            self._enc = T5EncoderModel.from_pretrained(self.model_name).to(self.device).to(self.dtype).eval()

    @torch.inference_mode()
    def encode(self, prompts: str | list[str], max_length: int = 512) -> torch.Tensor:
        """``prompts`` -> ``[B, max_length, d_model]``; positions past each prompt's length are 0."""
        self._lazy_load()
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("empty prompt list")
        enc = self._tok.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )
        input_ids = enc.input_ids.to(self.device)
        attn_mask = enc.attention_mask.to(self.device)
        out = self._enc(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()
        for b in range(out.shape[0]):
            out[b][lengths[b] :] = 0
        return out
