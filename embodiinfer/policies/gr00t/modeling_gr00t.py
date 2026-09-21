"""GR00T N1.7 adapter: Isaac-GR00T weights, **fully integrated into vvla**.

GR00T N1.7 is a Qwen3-VL (Cosmos-Reason2-2B) VLM backbone plus a flow-matching
**DiT** action head. This adapter needs **no ``gr00t`` package at runtime**: the
action-head module tree is vendored (:mod:`.modules_gr00t`) and loaded directly
from the checkpoint safetensors; the backbone is the stock transformers
``Qwen3VLForConditionalGeneration`` (which is exactly what gr00t wraps), built
from the Cosmos-Reason2-2B config and filled with the checkpoint's backbone
weights. This keeps GR00T on the single vvla env (torch 2.10 / transformers 5.3 /
diffusers 0.35) instead of gr00t's conflicting pins.

Two-stage split the engine schedules:
  * ``encode_prefix`` (once): run the Qwen3-VL backbone as an eager prefill, take
    ``hidden_states[select_layer]``, apply ``vlln`` + ``vl_self_attention``, embed
    the proprio state — the DiT cross-attends the *output* features, not the
    backbone's internal KV, so the backbone is a clean black-box boundary;
  * ``denoise_step`` (N times): the **embodiinfer-owned** DiT block loop (attention via
    the swappable :class:`AttentionBackend`), cross-attending the cached prefix.

Backbone fidelity (verified on the box): the checkpoint's ``backbone.model.*``
weights load into the stock Qwen3-VL with 0 missing / 0 unexpected. Two
cross-version alignments are needed to reproduce gr00t-native's (transformers
4.57) select-layer features on transformers>=5 — each was found by a per-layer
ground-truth comparison and each was masked by DiT robustness (actions were only
~1.7% off with both bugs present):
  * **pre-/post-final-norm** — gr00t reads ``hidden_states[-1]``, the *pre*-final-
    norm residual after ``select_layer`` layers; tf4.57 returns that there but
    tf>=5 returns the *post*-norm tensor, so ``__init__`` drops the vestigial
    final text norm to recover the pre-norm feature.
  * **multimodal-RoPE positions** — tf>=5's Qwen3-VL needs ``mm_token_type_ids`` to
    place image tokens on their 2D grid positions; the minimal input set GR00T
    feeds it otherwise falls back to sequential 1D positions (wrong vision RoPE),
    so ``encode_prefix`` computes ``position_ids`` via ``get_rope_index``.
With both, the aligned backbone tracks gr00t-native at vl_embeds cos ~0.99.

The denoise attention runs through the swappable :class:`AttentionBackend`;
``sdpa`` matches diffusers' default and is bit-exact against the vendored
diffusers path *within one env* (max|Δ|=0). Against gr00t-native *across*
environments the end-to-end delta collapses onto the DiT floor — max|Δ| ≈ 1.6e-2
(rel 0.4% on the reference obs), equal to feeding the DiT gr00t's own prefix —
i.e. bf16 arithmetic across torch/diffusers versions, the backbone now negligible.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch

from ...layers import get_attention_backend
from ..base import FlowVLAPolicy, VLAPolicy
from ..config import VLAPolicyConfig
from ..factory import register_policy
from .processor_gr00t import Gr00tBatch

_INSTALL = (
    "GR00T needs transformers>=5.3 (Qwen3-VL), diffusers, safetensors — all in the "
    "vvla env. Pass checkpoint=<GR00T-N1.7-3B dir> and cosmos_path=<Cosmos-Reason2-2B dir>."
)


class _BypassedNorm(torch.nn.Module):
    """Identity forward that keeps the wrapped norm's parameters registered.

    Used to skip the backbone's vestigial final text norm (gr00t consumes the
    pre-norm hidden state) without dropping its weight from the state_dict —
    weight-sync protocols that require the exact checkpoint key set still see
    (and harmlessly update) the parameter."""

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        # Re-register the very same Parameter under the same attribute name so
        # the state_dict key ("...norm.weight") is unchanged.
        self.weight = norm.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass
class Gr00tPrefix:
    """Cached VLM features the DiT cross-attends + the state token + cross-attn
    key-padding masks. ``vl_embeds`` (post ``vlln``/``vl_self_attention``) and
    ``state_features`` are constant across denoise steps."""

    vl_embeds: torch.Tensor
    image_key_mask: torch.Tensor | None  # [B,1,1,S] bool, or None (plain DiT)
    text_key_mask: torch.Tensor | None
    state_features: torch.Tensor
    embodiment_id: torch.Tensor
    cross_kv: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None = None

    @property
    def batch_size(self) -> int:
        return self.vl_embeds.shape[0]

    def to(self, device: torch.device | str) -> Gr00tPrefix:
        def mv(x):
            return None if x is None else x.to(device)

        return Gr00tPrefix(
            self.vl_embeds.to(device),
            mv(self.image_key_mask),
            mv(self.text_key_mask),
            self.state_features.to(device),
            self.embodiment_id.to(device),
            None
            if self.cross_kv is None
            else tuple(None if pair is None else tuple(x.to(device) for x in pair) for pair in self.cross_kv),
        )

    def expand(self, num_samples: int) -> Gr00tPrefix:
        if num_samples == 1:
            return self

        def rep(x):
            return None if x is None else x.repeat_interleave(num_samples, dim=0)

        return Gr00tPrefix(
            rep(self.vl_embeds),
            rep(self.image_key_mask),
            rep(self.text_key_mask),
            rep(self.state_features),
            rep(self.embodiment_id),
            None
            if self.cross_kv is None
            else tuple(None if pair is None else tuple(rep(x) for x in pair) for pair in self.cross_kv),
        )


class Gr00tPolicy(FlowVLAPolicy):
    """First-class GR00T N1.7 policy: vendored action head + stock Qwen3-VL backbone,
    embodiinfer-owned denoise loop. No gr00t package at runtime."""

    def __init__(
        self,
        config: VLAPolicyConfig,
        checkpoint: str,
        cosmos_path: str,
        attention: str = "sdpa",
        *,
        native_inference: bool = False,
        prefix_cuda_graph: bool = False,
    ):
        super().__init__(config)
        try:
            from safetensors.torch import load_file
            from transformers import Qwen3VLForConditionalGeneration

            from .modules_gr00t import Gr00tActionHead
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise ImportError(_INSTALL) from exc

        self.attention = attention
        self.native_inference = native_inference
        if prefix_cuda_graph and not native_inference:
            raise ValueError("GR00T prefix_cuda_graph requires native_inference=True")
        self.prefix_cuda_graph = prefix_cuda_graph
        self._attn = get_attention_backend(attention)
        self._cached_prefix_len: int | None = None  # set by encode_prefix, read by allocate_static_prefix
        with open(os.path.join(checkpoint, "config.json")) as f:
            gcfg = json.load(f)
        self._gcfg = gcfg
        self._select_layer = gcfg["select_layer"]
        self._alternate = bool(gcfg.get("use_alternate_vl_dit", False))
        self._attend_text_every_n = int(gcfg.get("attend_text_every_n_blocks", 2))

        dtype = getattr(torch, gcfg.get("model_dtype", "bfloat16"))
        # backbone: stock Qwen3-VL, pruned to select_layer, GR00T weights loaded in.
        # The attention implementation follows the checkpoint config, exactly as
        # gr00t's Qwen3Backbone does (use_flash_attention -> flash_attention_2,
        # sdpa fallback). This is a fidelity requirement, not a speed knob: with
        # right-padded batches the pad-position hidden states are implementation-
        # dependent garbage, and the (unmasked) vl_self_attention mixes them into
        # every token — a trainer recomputing with flash sees uncorrelated
        # features unless the rollout used the same implementation.
        bb_kwargs = {}
        if gcfg.get("use_flash_attention", False):
            try:
                import flash_attn  # noqa: F401

                bb_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                bb_kwargs["attn_implementation"] = "sdpa"
        # The trailing .to(dtype) casts the floating BUFFERS too — notably the
        # rotary inv_freq tables, which gr00t's load_bf16 path materializes in
        # bf16 (the RoPE numerics the checkpoint was trained with). from_pretrained
        # (dtype=...) only casts parameters and would leave fp32 tables, a small
        # position-dependent divergence from the reference forward.
        backbone = (
            Qwen3VLForConditionalGeneration.from_pretrained(cosmos_path, dtype=dtype, **bb_kwargs)
            .to(dtype)
            .eval()
        )
        while len(backbone.model.language_model.layers) > self._select_layer:
            backbone.model.language_model.layers.pop(-1)
        self._image_token_id = backbone.config.image_token_id
        self._backbone = backbone
        self._ah = Gr00tActionHead(gcfg).to(dtype).eval()
        self._interleave = bool(self._ah.model.interleave_self_attention)

        # load checkpoint weights into backbone (strip 'backbone.model.') + action head.
        sd_bb, sd_ah = {}, {}
        for f in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
            for k, v in load_file(f).items():
                if k.startswith("backbone.model."):
                    sd_bb[k[len("backbone.model.") :]] = v
                elif k.startswith("action_head."):
                    sd_ah[k[len("action_head.") :]] = v
        self._backbone.load_state_dict(sd_bb, strict=True)
        self._ah.load_state_dict(sd_ah, strict=True)

        # gr00t takes ``hidden_states[-1]`` as the backbone feature: the PRE-final-norm
        # residual stream after ``select_layer`` layers. gr00t pins transformers 4.57,
        # whose ``output_hidden_states`` returns that pre-norm tensor as the last element;
        # transformers>=5 instead returns the *post*-final-norm tensor there (verified on
        # the box: the two differ at cos~0.62 with a ~1.5e4 massive-activation channel,
        # while the pre-norm tensors match gr00t at cos~0.98). The final text norm is
        # vestigial here — we read a hidden state, never the logits — so drop it to
        # recover the exact pre-norm feature gr00t's action head was trained on, making
        # ``encode_prefix`` transformers-version-independent. (Done after the strict load
        # so the checkpoint's norm weight still resolves.)
        # The norm module is bypassed, not deleted: its weight stays registered
        # (identity forward around it) so the state_dict remains checkpoint-
        # complete — an RL trainer's weight sync, which requires the exact
        # checkpoint key set, then round-trips losslessly.
        self._backbone.model.language_model.norm = _BypassedNorm(self._backbone.model.language_model.norm)

        from .runtime import Gr00tPrefixRuntime

        self._prefix_runtime = Gr00tPrefixRuntime(self)

    def on_refit(self, version: int) -> None:
        """Invalidate positional embeddings and prefix graphs after weight updates."""
        super().on_refit(version)
        self._clear_prefix_runtime()

    def _clear_prefix_runtime(self) -> None:
        runtime = getattr(self, "_prefix_runtime", None)
        if runtime is not None:
            runtime.clear()

    def _apply(self, fn, recurse: bool = True):
        self._clear_prefix_runtime()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True):
        """Keep inference caches out of transitions to and from gradient execution."""
        if mode != self.training:
            self._clear_prefix_runtime()
        return super().train(mode)

    # ---- attention sub-layer (reimplements diffusers AttnProcessor2_0) -------
    def _attend(self, attn, hidden, encoder, mask, projected_kv=None):
        B = hidden.shape[0]
        heads = attn.heads
        context = hidden if encoder is None else encoder
        q = attn.to_q(hidden)
        head_dim = q.shape[-1] // heads
        q = q.view(B, q.shape[1], heads, head_dim).transpose(1, 2)
        if projected_kv is None:
            k, v = attn.to_k(context), attn.to_v(context)
            k = k.view(B, k.shape[1], heads, head_dim).transpose(1, 2)
            v = v.view(B, v.shape[1], heads, head_dim).transpose(1, 2)
        else:
            k, v = projected_kv
        out = self._attn.attend(q, k, v, attn_mask=mask, scaling=None)
        out = out.transpose(1, 2).reshape(B, -1, heads * head_dim).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](out))

    def _block(self, block, h, temb, encoder, mask, projected_kv=None):
        norm_h = block.norm1(h, temb) if block.norm_type == "ada_norm" else block.norm1(h)
        h = self._attend(block.attn1, norm_h, encoder, mask, projected_kv) + h
        return block.ff(block.norm3(h)) + h

    def _cross_mask(self, prefix: Gr00tPrefix, idx: int) -> torch.Tensor | None:
        if not self._alternate:
            return None
        if idx % (2 * self._attend_text_every_n) == 0:
            return prefix.text_key_mask
        return prefix.image_key_mask

    def _mrope_position_ids(self, bi: dict) -> torch.Tensor | None:
        """Qwen3-VL multimodal-RoPE positions for the prefix (image tokens get 2D
        temporal/height/width positions from the grid, not sequential 1D).

        transformers>=5's Qwen3-VL derives these in ``get_rope_index`` from a
        ``mm_token_type_ids`` marker (image tokens -> 1) that its own processor
        emits. GR00T drives the backbone with a minimal input set that lacks that
        marker, so the stock forward falls back to *sequential* 1D positions — the
        wrong RoPE rotation for every vision token. Verified on the box: without
        this, image-token hidden states diverge from gr00t-native (transformers
        4.57, which computes 2D positions internally) at cos ~0.94 by the select
        layer and compound; with it they match at cos ~0.997 and the positions are
        bit-identical to gr00t's (max|Δ|=0). Returns None if the backbone predates
        the ``get_rope_index`` API, in which case the forward positions itself.

        transformers 4.57's ``get_rope_index`` has no ``mm_token_type_ids``
        parameter — that version derives the image-token 2D positions from
        ``input_ids`` internally in the stock forward (the gr00t-native
        reference path), so no explicit positions are needed and we return
        None rather than misbind the tf>=5 argument order."""
        model = self._backbone.model
        if not hasattr(model, "get_rope_index"):
            return None
        import inspect

        if "mm_token_type_ids" not in inspect.signature(model.get_rope_index).parameters:
            return None
        mm_token_type_ids = (bi["input_ids"] == self._image_token_id).to(torch.int32)
        position_ids, _ = model.get_rope_index(
            bi["input_ids"],
            mm_token_type_ids,
            bi.get("image_grid_thw"),
            bi.get("video_grid_thw"),
            bi["attention_mask"],
        )
        return position_ids

    def _native_enabled(self) -> bool:
        return self.native_inference and not self.training and not torch.is_grad_enabled()

    def _project_cross_kv(self, vl: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, ...]:
        """Project observation-specific cross-attention keys/values once per prefix."""
        values = []
        for idx, block in enumerate(self._ah.model.transformer_blocks):
            if self._interleave and idx % 2 == 1:
                values.append(None)
                continue
            attn = block.attn1
            k, v = attn.to_k(vl), attn.to_v(vl)
            shape = (vl.shape[0], vl.shape[1], attn.heads, k.shape[-1] // attn.heads)
            values.append((k.view(shape).transpose(1, 2), v.view(shape).transpose(1, 2)))
        return tuple(values)

    # ---- stage 1: encode prefix (once) --------------------------------------
    def encode_prefix(self, batch: Gr00tBatch) -> Gr00tPrefix:
        if self.prefix_cuda_graph and self._native_enabled() and batch.state.is_cuda:
            return self._prefix_runtime.encode(batch)
        ah = self._ah
        bi = batch.backbone_inputs
        backbone = self._backbone.model if self._native_enabled() else self._backbone
        out = backbone(
            input_ids=bi["input_ids"],
            attention_mask=bi["attention_mask"],
            pixel_values=bi.get("pixel_values"),
            image_grid_thw=bi.get("image_grid_thw"),
            position_ids=self._mrope_position_ids(bi),
            output_hidden_states=True,
        )
        vl = out.hidden_states[self._select_layer]
        vl = ah.vl_self_attention(ah.vlln(vl))
        self._cached_prefix_len = vl.shape[1]  # for allocate_static_prefix (CUDA graph)
        state = batch.state.view(batch.state.shape[0], 1, -1)
        state_features = ah.state_encoder(state, batch.embodiment_id)
        img_mask = txt_mask = None
        if self._alternate:
            attn_m = bi["attention_mask"] == 1
            image = bi["input_ids"] == self._image_token_id
            img_mask = (image & attn_m)[:, None, None, :]
            txt_mask = (~image & attn_m)[:, None, None, :]
        cross_kv = self._project_cross_kv(vl) if self._native_enabled() else None
        return Gr00tPrefix(vl, img_mask, txt_mask, state_features, batch.embodiment_id, cross_kv)

    # ---- stage 2: one denoising step (N times) ------------------------------
    def denoise_step(self, x_t: torch.Tensor, t: torch.Tensor, prefix: Gr00tPrefix) -> torch.Tensor:
        ah, dit = self._ah, self._ah.model
        # GR00T feeds an integer timestep bucket: t_disc = int(t_cont * num_buckets).
        # Discretize in fp32 (engine t is model dtype); default 4 steps -> exact buckets.
        t_disc = (t.to(torch.float32) * self._gcfg["num_timestep_buckets"]).long()

        af = ah.action_encoder(x_t, t_disc, prefix.embodiment_id)
        if self._gcfg.get("add_pos_embed", False):
            pos = torch.arange(af.shape[1], device=x_t.device)
            af = af + ah.position_embedding(pos).unsqueeze(0)
        sa = torch.cat([prefix.state_features, af], dim=1).contiguous()

        temb = dit.timestep_encoder(t_disc)
        h, vl = sa, prefix.vl_embeds
        for idx, block in enumerate(dit.transformer_blocks):
            if self._interleave and idx % 2 == 1:
                h = self._block(block, h, temb, None, None)
            else:
                h = self._block(
                    block,
                    h,
                    temb,
                    vl,
                    self._cross_mask(prefix, idx),
                    None if prefix.cross_kv is None or not self._native_enabled() else prefix.cross_kv[idx],
                )

        shift, scale = dit.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
        h = dit.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        pred = ah.action_decoder(dit.proj_out_2(h), prefix.embodiment_id)
        return pred[:, -self.config.action_horizon :]

    # ---- CUDA-graph capability: static-shape denoise loop -------------------
    @property
    def supports_cuda_graph(self) -> bool:
        """The DiT denoise step runs with identical shapes every iteration (x_t,
        t, and a fixed-shape prefix), so the loop is capturable — GR00T inherits
        the engine's graph capture / replay with no engine change, the second
        first-class policy validating the model-agnostic optimization path. The
        win is in the launch-bound small-batch regime the 32-block DiT * N steps
        otherwise spends on per-kernel Python overhead."""
        return True

    def allocate_static_prefix(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Gr00tPrefix:
        """Zero-filled static-shape :class:`Gr00tPrefix` for graph capture.

        Sizes the VLM feature / state-token / cross-attn mask buffers from the
        checkpoint config and the prefix length cached by the most recent
        ``encode_prefix``; ``copy_prefix_into`` fills them with the live prefix."""
        if self._cached_prefix_len is None:
            raise RuntimeError("allocate_static_prefix needs a prior encode_prefix to know the prefix length")
        s = self._cached_prefix_len
        cross_dim = self._gcfg["backbone_embedding_dim"]
        dm = self._gcfg["diffusion_model_cfg"]
        emb_dim = dm["num_attention_heads"] * dm["attention_head_dim"]
        vl = torch.zeros(batch_size, s, cross_dim, device=device, dtype=dtype)
        state = torch.zeros(batch_size, 1, emb_dim, device=device, dtype=dtype)
        eid = torch.zeros(batch_size, device=device, dtype=torch.long)
        img_mask = txt_mask = None
        if self._alternate:
            # All-valid masks for the capture warmup (an all-masked key row would
            # NaN the cross-attn softmax); copy_prefix_into overwrites them.
            img_mask = torch.ones(batch_size, 1, 1, s, device=device, dtype=torch.bool)
            txt_mask = torch.ones(batch_size, 1, 1, s, device=device, dtype=torch.bool)
        return Gr00tPrefix(vl, img_mask, txt_mask, state, eid)

    def cuda_graph_variant(self, prefix: Gr00tPrefix) -> object | None:
        """Keep layouts and inference/no-grad tensor ownership in separate graphs."""
        return prefix.vl_embeds.shape[1], prefix.cross_kv is not None, torch.is_inference_mode_enabled()

    def allocate_static_prefix_from_live(
        self,
        prefix: Gr00tPrefix,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        variant: object | None = None,
    ) -> Gr00tPrefix:
        """Allocate graph-owned buffers from the current observation's prefix."""
        if prefix.batch_size != batch_size:
            raise ValueError("GR00T prefix batch must match its graph bucket")
        if variant is not None and variant != self.cuda_graph_variant(prefix):
            raise ValueError("GR00T graph shape/variant must match the live prefix")

        def clone(tensor):
            if tensor is None:
                return None
            return tensor.to(
                device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype
            ).clone()

        return Gr00tPrefix(
            clone(prefix.vl_embeds),
            clone(prefix.image_key_mask),
            clone(prefix.text_key_mask),
            clone(prefix.state_features),
            clone(prefix.embodiment_id),
            None
            if prefix.cross_kv is None
            else tuple(None if pair is None else tuple(clone(x) for x in pair) for pair in prefix.cross_kv),
        )

    def copy_prefix_into(self, dst: Gr00tPrefix, src: Gr00tPrefix) -> None:
        """In-place copy the live prefix (VLM features + state token + masks +
        embodiment id) into the static graph buffer."""
        if (dst.cross_kv is None) != (src.cross_kv is None):
            raise ValueError("GR00T graph and live prefix must have the same K/V cache layout")
        if dst.cross_kv is not None:
            for target, value in zip(dst.cross_kv, src.cross_kv, strict=True):
                if (target is None) != (value is None):
                    raise ValueError("GR00T graph cross-attention cache slots differ")
                if target is not None:
                    target[0].copy_(value[0])
                    target[1].copy_(value[1])
        dst.vl_embeds.copy_(src.vl_embeds)
        dst.state_features.copy_(src.state_features)
        dst.embodiment_id.copy_(src.embodiment_id)
        if dst.image_key_mask is not None and src.image_key_mask is not None:
            dst.image_key_mask.copy_(src.image_key_mask)
            dst.text_key_mask.copy_(src.text_key_mask)

    # ---- batch construction --------------------------------------------------
    def collate(self, observations, request_ids):
        raise NotImplementedError(
            "GR00T collate from a raw Observation needs the Qwen3-VL processor "
            "(image transforms + tokenizer + state normalization); build a Gr00tBatch via "
            "Gr00tBatch.from_backbone_inputs(processor_output, embodiment_id) for now."
        )

    def pad(self, batch: Gr00tBatch, target_batch_size: int) -> Gr00tBatch:
        b = batch.batch_size
        if target_batch_size == b:
            return batch
        pad = target_batch_size - b

        def rep(x):
            return torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0)

        # Repeat the last sample. Qwen3-VL packs vision as variable-length tensors;
        # per-row repeat is valid only for batched leading-dim tensors (CUDA-graph
        # bucket padding for packed vision is validated on the box).
        backbone_inputs = {k: (rep(v) if v.shape[0] == b else v) for k, v in batch.backbone_inputs.items()}
        return Gr00tBatch(
            backbone_inputs=backbone_inputs,
            state=rep(batch.state),
            embodiment_id=rep(batch.embodiment_id),
            request_ids=batch.request_ids + [f"__pad_{i}" for i in range(pad)],
        )


@register_policy("gr00t")
def _build_gr00t(
    checkpoint: str | None = None,
    cosmos_path: str | None = None,
    attention: str = "sdpa",
    native_inference: bool = False,
    prefix_cuda_graph: bool = False,
    **overrides,
) -> VLAPolicy:
    if checkpoint is None:
        raise ValueError(
            "gr00t needs a checkpoint dir, e.g. make_policy('gr00t', checkpoint='/models/GR00T-N1.7-3B', "
            "cosmos_path='/models/Cosmos-Reason2-2B')"
        )
    if cosmos_path is None:
        raise ValueError("gr00t needs cosmos_path=<Cosmos-Reason2-2B dir> to build the Qwen3-VL backbone")
    with open(os.path.join(checkpoint, "config.json")) as f:
        gcfg = json.load(f)
    cfg = VLAPolicyConfig(
        name="gr00t_n1.7",
        action_dim=gcfg["max_action_dim"],
        action_horizon=gcfg["action_horizon"],
        default_num_steps=gcfg["num_inference_timesteps"],
        dtype=gcfg.get("model_dtype", "bfloat16"),
        **overrides,
    )
    return Gr00tPolicy(
        cfg,
        checkpoint=checkpoint,
        cosmos_path=cosmos_path,
        attention=attention,
        native_inference=native_inference,
        prefix_cuda_graph=prefix_cuda_graph,
    )
