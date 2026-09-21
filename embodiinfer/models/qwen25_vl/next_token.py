"""Graph-safe full Qwen2.5-VL vision-to-next-token Torch forward."""

from __future__ import annotations

import torch

from ...layers import get_attention_backend
from .backend import (
    Qwen25VLAttentionSelection,
    normalize_qwen25_vl_attention_backend,
    resolve_qwen25_vl_attention_backend,
)
from .vision import VisionBounds, _Qwen25VLVisionForwardMixin


class Qwen25VLNextTokenForward(_Qwen25VLVisionForwardMixin, torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        attention_backend: str = "torch_sdpa",
        use_text_attention_mask: bool = False,
    ):
        super().__init__()
        self.visual = model.model.visual
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.image_token_id = model.config.image_token_id
        self.use_text_attention_mask = bool(use_text_attention_mask)
        self.window_bounds: VisionBounds = ()
        self.full_bounds: VisionBounds = ()
        self.requested_attention_backend = normalize_qwen25_vl_attention_backend(attention_backend)
        self._attention_selection: Qwen25VLAttentionSelection | None = None
        self._segmented_attention_backend: object | None = None

    def set_vision_bounds(
        self,
        window_bounds: VisionBounds,
        full_bounds: VisionBounds,
    ) -> None:
        if window_bounds != self.window_bounds or full_bounds != self.full_bounds:
            self._attention_selection = None
            self._segmented_attention_backend = None
        self.window_bounds = window_bounds
        self.full_bounds = full_bounds

    def select_attention_backend(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Qwen25VLAttentionSelection:
        attention = self.visual.blocks[0].attn
        num_heads = int(attention.num_heads)
        head_dim = int(attention.qkv.out_features) // (3 * num_heads)
        window_max_length = max((end - start for start, end in self.window_bounds), default=0)
        full_max_length = max((end - start for start, end in self.full_bounds), default=0)
        selection = resolve_qwen25_vl_attention_backend(
            self.requested_attention_backend,
            device=device,
            dtype=dtype,
            num_query_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=head_dim,
            window_max_length=window_max_length,
            full_max_length=full_max_length,
        )
        if self._attention_selection is None or selection.cache_key != self._attention_selection.cache_key:
            self._segmented_attention_backend = (
                get_attention_backend(selection.layers_backend)
                if selection.window_attention == "triton_segmented"
                else None
            )
        self._attention_selection = selection
        return selection

    def attention_backend_cache_key(self) -> tuple[object, ...]:
        if self._attention_selection is None:
            raise RuntimeError("attention backend must be selected before building a graph key")
        return self._attention_selection.cache_key

    def attention_backend_stats(self) -> dict[str, object]:
        if self._attention_selection is None:
            return {
                "requested": self.requested_attention_backend,
                "resolved": None,
                "fallback_reason": None,
                "kernel_abi": None,
                "config": {},
                "layers_backend": None,
                "window_attention": None,
                "full_attention": None,
                "rope": None,
                "triton_version": None,
            }
        return self._attention_selection.as_dict()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        window_index: torch.Tensor,
        reverse_indices: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        backend_selection = self._attention_selection
        if backend_selection is None:
            backend_selection = self.select_attention_backend(
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
        hidden_states = self.visual.patch_embed(pixel_values)
        seq_len = hidden_states.shape[0]
        unit = self.visual.spatial_merge_unit
        hidden_states = hidden_states.reshape(seq_len // unit, unit, -1)
        hidden_states = hidden_states[window_index].reshape(seq_len, -1)
        for layer_num, block in enumerate(self.visual.blocks):
            is_full_attention = layer_num in self.visual.fullatt_block_indexes
            bounds = self.full_bounds if is_full_attention else self.window_bounds
            segment_offsets = cu_seqlens if is_full_attention else cu_window_seqlens
            max_segment_length = max(end - start for start, end in bounds)
            hidden_states = self._vision_block(
                block,
                hidden_states,
                rotary_cos,
                rotary_sin,
                bounds,
                is_full_attention=is_full_attention,
                segment_offsets=segment_offsets,
                max_segment_length=max_segment_length,
                backend_selection=backend_selection,
                segmented_attention_backend=self._segmented_attention_backend,
            )
        image_embeds = self.visual.merger(hidden_states)[reverse_indices]
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
        language_attention_mask = None
        if self.use_text_attention_mask:
            if getattr(self.language_model, "has_sliding_layers", False):
                raise RuntimeError(
                    "bucketed Qwen2.5-VL next-token forward does not support sliding text attention"
                )
            sequence_length = attention_mask.shape[-1]
            causal = torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=attention_mask.device,
            ).tril()
            language_attention_mask = {
                "full_attention": causal[None, None] & attention_mask[:, None, None, :].bool()
            }
        hidden = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=language_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )[0]
        return self.lm_head(hidden[:, -1, :])


__all__ = ["Qwen25VLNextTokenForward"]
