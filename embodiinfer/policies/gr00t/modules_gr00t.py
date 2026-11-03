"""GR00T N1.7 action-head modules, vendored into embodiinfer.

These are faithful re-implementations of the NVIDIA Isaac-GR00T action-head
building blocks (``gr00t/model/modules/dit.py`` and
``embodiment_conditioned_mlp.py``, Apache-2.0), so embodiinfer can **build the module
tree + load the checkpoint weights without importing the ``gr00t`` package** —
and thus without dragging in gr00t's heavy training/deployment pins (tensorrt,
deepspeed, flash-attn, ...) that would fracture the single embodiinfer env. Only the
inference-relevant structure is kept (training/print/Spark-SDPA paths dropped).

Two roles:
  * **weight container** — the submodule names match the checkpoint keys
    (``transformer_blocks.{i}.attn1.to_q``, ``timestep_encoder.timestep_embedder
    .linear_1``, ``state_encoder.layer1.W`` ...), so a plain ``load_state_dict``
    fills them from the GR00T safetensors;
  * **numerical reference** — each ``forward`` reproduces the official diffusers
    (SDPA) computation, so the embodiinfer-owned denoise loop (attention via the
    swappable :class:`AttentionBackend`) can be parity-checked against it without
    the gr00t package present.

Attention here uses diffusers' ``Attention`` (default ``AttnProcessor2_0`` ->
``F.scaled_dot_product_attention``); the embodiinfer-owned forward in
``modeling_gr00t.py`` re-runs the same projections through the backend instead.
"""

from __future__ import annotations

import torch
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# ---- embodiment-conditioned MLPs (state/action encoders, action decoder) -----
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape (B, T, w) given timesteps (B, T)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """Per-embodiment linear: selects ``W[cat_ids]`` / ``b[cat_ids]`` then bmm."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer per-embodiment MLP (ReLU) — state encoder / action decoder."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(x, cat_ids)), cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Per-embodiment action encoder with sinusoidal timestep encoding."""

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        if not (timesteps.dim() == 1 and timesteps.shape[0] == B):
            raise ValueError("Expected `timesteps` of shape (B,) to replicate across T.")
        timesteps = timesteps.unsqueeze(1).expand(-1, T)
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))
        return self.W3(x, cat_ids)


# ---- DiT (flow-matching action head transformer) -----------------------------
class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    """Timestep-conditioned LayerNorm (scale/shift from ``temb``)."""

    def __init__(self, embedding_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """One DiT block: (AdaLN|LN) -> attn1 (self or cross) -> LN -> FeedForward."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = True,
    ):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        if self.norm_type == "ada_norm":
            norm_h = self.norm1(hidden_states, temb)
        else:
            norm_h = self.norm1(hidden_states)
        attn_out = self.attn1(
            norm_h,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = attn_out + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class DiT(nn.Module):
    """Plain DiT: even blocks cross-attend the VLM prefix, odd blocks self-attend
    (when ``interleave_self_attention``). No cross-attention mask."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        output_dim: int,
        num_layers: int,
        cross_attention_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_type: str = "ada_norm",
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = False,
        final_dropout: bool = True,
        interleave_self_attention: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.interleave_self_attention = interleave_self_attention
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        blocks = []
        for idx in range(num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    norm_elementwise_affine=norm_elementwise_affine,
                    final_dropout=final_dropout,
                    cross_attention_dim=None if use_self_attn else cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        temb = self.timestep_encoder(timestep)
        h = hidden_states.contiguous()
        enc = encoder_hidden_states.contiguous()
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.interleave_self_attention:
                h = block(h, encoder_hidden_states=None, attention_mask=None, temb=temb)
            else:
                h = block(h, encoder_hidden_states=enc, attention_mask=None, temb=temb)
        shift, scale = self.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(h)


class AlternateVLDiT(DiT):
    """DiT whose cross-attention blocks alternate attending image-only vs
    text-only VLM tokens (key-padding mask), every ``attend_text_every_n_blocks``."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self, hidden_states, encoder_hidden_states, timestep, image_mask, backbone_attention_mask, **kwargs
    ):
        temb = self.timestep_encoder(timestep)
        h = hidden_states.contiguous()
        enc = encoder_hidden_states.contiguous()
        image_attn = image_mask & backbone_attention_mask
        text_attn = (~image_mask) & backbone_attention_mask
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                h = block(h, encoder_hidden_states=None, attention_mask=None, temb=temb)
            else:
                mask = text_attn if idx % (2 * self.attend_text_every_n_blocks) == 0 else image_attn
                h = block(h, encoder_hidden_states=enc, attention_mask=mask, temb=temb)
        shift, scale = self.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(h)


class SelfAttentionTransformer(nn.Module):
    """Post-backbone self-attention over VLM tokens (``vl_self_attention``)."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        final_dropout: bool = True,
        **kwargs,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_type="layer_norm",
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class Gr00tActionHead(nn.Module):
    """Vendored container mirroring ``Gr00tN1d7ActionHead``'s submodule layout, so
    the ``action_head.*`` checkpoint keys load straight in.

    Built from the checkpoint ``config.json`` dict. ``input_embedding_dim`` (the
    sa-token / DiT inner dim) is derived from ``num_attention_heads *
    attention_head_dim`` rather than the config field, which the released
    checkpoint stores as ``null``.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        dm = cfg["diffusion_model_cfg"]
        emb_dim = dm["num_attention_heads"] * dm["attention_head_dim"]  # 1536; config field is null
        hid = cfg["hidden_size"]
        ne = cfg["max_num_embodiments"]
        cross_dim = cfg["backbone_embedding_dim"]
        sdim = cfg["max_state_dim"] * cfg.get("state_history_length", 1)

        dit_kwargs = dict(
            num_attention_heads=dm["num_attention_heads"],
            attention_head_dim=dm["attention_head_dim"],
            output_dim=dm["output_dim"],
            num_layers=dm["num_layers"],
            cross_attention_dim=cross_dim,
            dropout=dm.get("dropout", 0.0),
            activation_fn=dm.get("activation_fn", "gelu-approximate"),
            norm_type=dm.get("norm_type", "ada_norm"),
            final_dropout=dm.get("final_dropout", True),
            interleave_self_attention=dm.get("interleave_self_attention", True),
        )
        if cfg.get("use_alternate_vl_dit", False):
            self.model = AlternateVLDiT(
                attend_text_every_n_blocks=cfg.get("attend_text_every_n_blocks", 2), **dit_kwargs
            )
        else:
            self.model = DiT(**dit_kwargs)

        self.state_encoder = CategorySpecificMLP(ne, sdim, hid, emb_dim)
        self.action_encoder = MultiEmbodimentActionEncoder(cfg["max_action_dim"], emb_dim, ne)
        self.action_decoder = CategorySpecificMLP(ne, hid, hid, cfg["max_action_dim"])
        self.vlln = nn.LayerNorm(cross_dim) if cfg.get("use_vlln", True) else nn.Identity()

        vlsa = cfg.get("vl_self_attention_cfg")
        if cfg.get("use_vl_self_attention", False) and vlsa and vlsa.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vlsa)
        else:
            self.vl_self_attention = nn.Identity()

        if cfg.get("add_pos_embed", False):
            self.position_embedding = nn.Embedding(cfg["max_seq_len"], emb_dim)
