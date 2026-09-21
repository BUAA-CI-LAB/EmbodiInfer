"""GR00T-owned, layout-specific CUDA graphs for complete observation encoding."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .modeling_gr00t import Gr00tPolicy, Gr00tPrefix
    from .processor_gr00t import Gr00tBatch


@dataclass
class PrefixLayout:
    """Image geometry and positions that remain constant for a token layout."""

    vision_position: torch.Tensor
    vision_rope: tuple[torch.Tensor, torch.Tensor]
    vision_lengths: tuple[int, ...]
    image_indices: torch.Tensor
    text_rope: tuple[torch.Tensor, torch.Tensor]
    causal_mask: torch.Tensor | None
    image_mask: torch.Tensor
    text_mask: torch.Tensor


def _vision_attention(attn, hidden: torch.Tensor, layout: PrefixLayout) -> torch.Tensor:
    """Keep the reference per-image SDPA calls while removing device-to-host splits."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    length = hidden.shape[0]
    q, k, v = attn.qkv(hidden).reshape(length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *layout.vision_rope)
    splits = [value.transpose(0, 1).unsqueeze(0).split(layout.vision_lengths, dim=2) for value in (q, k, v)]
    outputs = [
        sdpa_attention_forward(
            attn, q, k, v, attention_mask=None, scaling=attn.scaling, dropout=0.0, is_causal=False
        )[0]
        for q, k, v in zip(*splits, strict=True)
    ]
    return attn.proj(torch.cat(outputs, dim=1).reshape(length, -1).contiguous())


def encode_features(policy: Gr00tPolicy, batch: Gr00tBatch, layout: PrefixLayout) -> Gr00tPrefix:
    """Execute the original vision, DeepStack, language and action-prefix math."""
    from .modeling_gr00t import Gr00tPrefix

    model = policy._backbone.model
    visual = model.visual
    hidden = visual.patch_embed(batch.backbone_inputs["pixel_values"]) + layout.vision_position
    deepstack = []
    for index, block in enumerate(visual.blocks):
        hidden = hidden + _vision_attention(block.attn, block.norm1(hidden), layout)
        hidden = hidden + block.mlp(block.norm2(hidden))
        if index in visual.deepstack_visual_indexes:
            merger = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(index)]
            deepstack.append(merger(hidden))
    images = visual.merger(hidden)
    language = model.language_model
    hidden = language.embed_tokens(batch.backbone_inputs["input_ids"])
    shape = hidden.shape
    hidden = hidden.reshape(-1, shape[-1]).index_copy(0, layout.image_indices, images).view(shape)
    for index, layer in enumerate(language.layers):
        hidden = layer(
            hidden,
            attention_mask=layout.causal_mask,
            position_embeddings=layout.text_rope,
            past_key_values=None,
        )
        if index < len(deepstack):
            flat = hidden.reshape(-1, shape[-1])
            updated = flat.index_select(0, layout.image_indices) + deepstack[index].to(hidden.dtype)
            hidden = flat.index_copy(0, layout.image_indices, updated).view(shape)
    hidden = language.norm(hidden)
    ah = policy._ah
    vl = ah.vl_self_attention(ah.vlln(hidden))
    state = batch.state.view(batch.batch_size, 1, -1)
    state_features = ah.state_encoder(state, batch.embodiment_id)
    image_mask = layout.image_mask if policy._alternate else None
    text_mask = layout.text_mask if policy._alternate else None
    return Gr00tPrefix(
        vl, image_mask, text_mask, state_features, batch.embodiment_id, policy._project_cross_kv(vl)
    )


class Gr00tPrefixRuntime:
    """Bounded graph cache whose outputs remain owned by individual requests."""

    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.graphs: OrderedDict[tuple, tuple] = OrderedDict()
        self.capture_count = 0

    def clear(self) -> None:
        """Discard weight-derived positions and captured graphs after refit or migration."""
        self.graphs.clear()

    def stats(self) -> dict[str, int]:
        """Report captures and resident layouts without changing during ordinary replay."""
        return {"capture_count": self.capture_count, "layouts": len(self.graphs)}

    def _layout(
        self, batch: Gr00tBatch, ids: torch.Tensor, mask: torch.Tensor, grid: torch.Tensor
    ) -> PrefixLayout:
        from transformers.masking_utils import create_causal_mask

        policy = self.policy
        model = policy._backbone.model
        visual, language = model.visual, model.language_model
        device, dtype = batch.state.device, batch.state.dtype
        if visual.config._attn_implementation != "sdpa" or language.config._attn_implementation != "sdpa":
            raise ValueError("GR00T prefix CUDA Graph requires the reference SDPA backbone")
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("GR00T prefix CUDA Graph requires a binary attention_mask")
        image = ids == policy._image_token_id
        lengths = tuple(h * w for t, h, w in grid.tolist() for _ in range(t))
        if sum(lengths) != batch.backbone_inputs["pixel_values"].shape[0]:
            raise ValueError("GR00T image grid and packed pixel count differ")
        count = sum(lengths) // visual.spatial_merge_unit
        indices = image.flatten().nonzero(as_tuple=True)[0].to(device)
        if indices.numel() != count:
            raise ValueError("GR00T image placeholders and patch geometry differ")
        positions = policy._mrope_position_ids(
            {"input_ids": ids, "attention_mask": mask, "image_grid_thw": grid}
        )
        if positions is None:
            raise ValueError("GR00T prefix CUDA Graph requires explicit multimodal positions")
        positions = positions.to(device)
        dummy = torch.empty((*ids.shape, language.config.hidden_size), device=device, dtype=dtype)
        cache_position = torch.arange(ids.shape[1], device=device)
        text_positions = positions[0] if positions.shape[0] == 4 else None
        rope_positions = positions[1:] if positions.shape[0] == 4 else positions
        causal = create_causal_mask(
            config=language.config,
            inputs_embeds=dummy,
            attention_mask=mask.to(device),
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_positions,
        )
        vision_rope = visual.rot_pos_emb(grid)
        vision_rope = torch.cat((vision_rope, vision_rope), dim=-1)
        return PrefixLayout(
            visual.fast_pos_embed_interpolate(grid),
            (vision_rope.cos(), vision_rope.sin()),
            lengths,
            indices,
            language.rotary_emb(dummy, rope_positions),
            causal,
            (image & mask.bool()).to(device)[:, None, None, :],
            (~image & mask.bool()).to(device)[:, None, None, :],
        )

    def encode(self, batch: Gr00tBatch) -> Gr00tPrefix:
        """Copy dynamic observations into a matching graph and return independent tensors."""
        from ...engine.graph import _CAPTURE_LOCK
        from .processor_gr00t import Gr00tBatch

        bi = batch.backbone_inputs
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        if set(bi) != required:
            raise ValueError("GR00T prefix CUDA Graph supports image observations with explicit masks")
        ids = bi["input_ids"].cpu()
        mask = bi["attention_mask"].cpu()
        grid = bi["image_grid_thw"].cpu()
        image = ids == self.policy._image_token_id
        key = (
            tuple(ids.shape),
            tuple(grid.flatten().tolist()),
            tuple(image.flatten().tolist()),
            tuple(mask.flatten().tolist()),
            tuple(batch.state.shape),
            tuple(
                (name, tuple(value.shape), value.dtype, value.device) for name, value in sorted(bi.items())
            ),
            tuple(batch.embodiment_id.shape),
            batch.embodiment_id.dtype,
            batch.state.dtype,
            batch.state.device,
            torch.is_inference_mode_enabled(),
            torch.cuda.current_stream(batch.state.device).cuda_stream,
        )
        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= 16:
                self.graphs.popitem(last=False)
            static = Gr00tBatch(
                {k: v.clone() for k, v in bi.items()},
                batch.state.clone(),
                batch.embodiment_id.clone(),
                list(batch.request_ids),
            )
            layout = self._layout(batch, ids, mask, grid)
            with _CAPTURE_LOCK:
                stream = torch.cuda.Stream(device=batch.state.device)
                stream.wait_stream(torch.cuda.current_stream(batch.state.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        encode_features(self.policy, static, layout)
                torch.cuda.current_stream(batch.state.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                    output = encode_features(self.policy, static, layout)
                # Match the engine capture contract before another replica starts capture.
                torch.cuda.synchronize(batch.state.device)
            entry = (static, graph, output, layout)
            self.graphs[key] = entry
            self.capture_count += 1
        self.graphs.move_to_end(key)
        static, graph, output, _layout = entry
        for name, value in bi.items():
            static.backbone_inputs[name].copy_(value)
        static.state.copy_(batch.state)
        static.embodiment_id.copy_(batch.embodiment_id)
        graph.replay()
        self.policy._cached_prefix_len = output.vl_embeds.shape[1]
        return self.policy.allocate_static_prefix_from_live(
            output,
            batch.batch_size,
            batch.state.device,
            batch.state.dtype,
        )
