"""Pure-Tensor Qwen2.5-VL vision forward primitives."""

from __future__ import annotations

import torch

from .backend import Qwen25VLAttentionSelection

VisionBounds = tuple[tuple[int, int], ...]


class _Qwen25VLVisionForwardMixin:
    @staticmethod
    def _rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _vision_attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        bounds: VisionBounds,
        *,
        is_full_attention: bool,
        segment_offsets: torch.Tensor | None = None,
        max_segment_length: int | None = None,
        backend_selection: Qwen25VLAttentionSelection | None = None,
        segmented_attention_backend: object | None = None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        query, key, value = (
            attention.qkv(hidden_states)
            .reshape(seq_len, 3, attention.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        attention_plan = (
            backend_selection.full_attention
            if backend_selection is not None and is_full_attention
            else backend_selection.window_attention
            if backend_selection is not None
            else "torch_sdpa"
        )
        rope_plan = backend_selection.rope if backend_selection is not None else "torch"
        if rope_plan == "triton":
            from ...backend.triton import rotate_half_rope

            query, key = rotate_half_rope(
                query.contiguous(),
                key.contiguous(),
                rotary_cos.contiguous(),
                rotary_sin.contiguous(),
            )
        else:
            query_dtype, key_dtype = query.dtype, key.dtype
            cosine = rotary_cos.unsqueeze(-2).float()
            sine = rotary_sin.unsqueeze(-2).float()
            query_float, key_float = query.float(), key.float()
            query = (query_float * cosine + self._rotate_half(query_float) * sine).to(query_dtype)
            key = (key_float * cosine + self._rotate_half(key_float) * sine).to(key_dtype)
        if attention_plan == "triton_segmented":
            if segment_offsets is None or max_segment_length is None:
                raise RuntimeError("Triton vision attention requires segment offsets and maximum length")
            if segmented_attention_backend is None:
                raise RuntimeError("Triton segmented attention backend was not prepared")
            attend_segmented = segmented_attention_backend.attend_segmented
            output = attend_segmented(
                query,
                key,
                value.contiguous(),
                segment_offsets,
                segment_offsets,
                scaling=attention.scaling,
                max_query_length=max_segment_length,
                max_key_length=max_segment_length,
            )
            output = output.reshape(seq_len, -1).contiguous()
            return attention.proj(output)
        if attention_plan != "torch_sdpa":
            raise RuntimeError(f"unsupported Qwen2.5-VL attention plan: {attention_plan}")
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        pieces = [
            torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:end, :],
                key[:, :, start:end, :],
                value[:, :, start:end, :],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=attention.scaling,
            ).transpose(1, 2)
            for start, end in bounds
        ]
        if not pieces:
            raise RuntimeError("vision attention received no non-empty segments")
        output = torch.cat(pieces, dim=1).reshape(seq_len, -1).contiguous()
        return attention.proj(output)

    def _vision_block(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        bounds: VisionBounds,
        *,
        is_full_attention: bool,
        segment_offsets: torch.Tensor | None = None,
        max_segment_length: int | None = None,
        backend_selection: Qwen25VLAttentionSelection | None = None,
        segmented_attention_backend: object | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self._vision_attention(
            block.attn,
            block.norm1(hidden_states),
            rotary_cos,
            rotary_sin,
            bounds,
            is_full_attention=is_full_attention,
            segment_offsets=segment_offsets,
            max_segment_length=max_segment_length,
            backend_selection=backend_selection,
            segmented_attention_backend=segmented_attention_backend,
        )
        return hidden_states + block.mlp(block.norm2(hidden_states))


__all__ = ["VisionBounds"]
