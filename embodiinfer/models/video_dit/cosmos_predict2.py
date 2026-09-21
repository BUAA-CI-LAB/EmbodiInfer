"""Self-hosted Cosmos-Predict2 2B DiT (the diffusion denoiser of Cosmos Policy).

Cosmos Policy is a video world-action model: images / proprio / action-chunk /
future-state / value are all packed as latent *frames* in a latent video and a
Cosmos-Predict2 DiT denoises them (EDM-sigma sampler + rectified-flow network
preconditioning). This module re-runs that DiT's forward op-by-op — the loaded
``net.*`` weights are held as plain ``nn.Module`` leaves and the transformer stack
(patchify, per-frame AdaLN-LoRA modulation, joint 3D self-attention with QK-norm +
3D RoPE, text cross-attention, GELU MLP, final unpatchify) is executed here, so the
denoise loop is embodiinfer's to schedule / CUDA-graph and attention goes through
:class:`~embodiinfer.layers.attention.AttentionBackend`. Same "modules load, forward is
self-hosted, vision/VAE stays a leaf" contract as pi0.5 / OpenVLA-OFT / LingBot-VLA.

The DiT is the base ``MinimalV1LVGDiT`` (``MiniTrainDIT``); the ``net.*`` state-dict
maps onto the attribute tree here 1:1 (``_extra_state`` TransformerEngine metadata
and ``accum_*`` counters are dropped, ``pos_embedder`` buffers are recomputed).
Blueprint: source-verified against the upstream implementation on 2026-07-15.

LIBERO Predict2-2B constants: model_channels=2048, num_blocks=28, num_heads=16
(head_dim=128), patch_spatial=2 / patch_temporal=1, in=16(+condition-mask+padding
=18)/out=16, crossattn_emb=1024, mlp_ratio=4, adaln_lora_dim=256, rope3d
(h/w extrapolation 3.0, t 1.0), all Linear bias-free, LayerNorm affine-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ...layers import get_attention_backend
from ...layers.attention import AttentionBackend
from .base import VideoDiT, register_video_dit


@dataclass
class CosmosPredict2DiTConfig:
    """Cosmos-Predict2 DiT hyperparameters (LIBERO 2B defaults)."""

    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    concat_padding_mask: bool = True
    crossattn_emb_channels: int = 1024
    adaln_lora_dim: int = 256
    # rope3d
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    rope_h_extrapolation_ratio: float = 3.0
    rope_w_extrapolation_ratio: float = 3.0
    rope_t_extrapolation_ratio: float = 1.0
    rope_enable_fps_modulation: bool = False
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads


# ---- numeric leaves ---------------------------------------------------------
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """TransformerEngine-style RMSNorm: normalize in fp32, scale by ``weight``, recast."""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q_BHSD: torch.Tensor, k_BHSD: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """NeoX rope over the full head_dim; ``cos``/``sin`` are ``[S, head_dim]`` (from the
    duplicated-half rope angles), broadcast over batch and heads."""
    cos = cos[None, None]  # [1,1,S,D]
    sin = sin[None, None]
    q = (q_BHSD.float() * cos + _rotate_half(q_BHSD.float()) * sin).to(q_BHSD.dtype)
    k = (k_BHSD.float() * cos + _rotate_half(k_BHSD.float()) * sin).to(k_BHSD.dtype)
    return q, k


class _Timesteps(nn.Module):
    """Sinusoidal timestep embedding (cos then sin, fp32), the exact Cosmos ``Timesteps``."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        in_dtype = timesteps_B_T.dtype
        B, T = timesteps_B_T.shape
        timesteps = timesteps_B_T.flatten().float()
        half = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb.to(in_dtype).reshape(B, T, self.num_channels)


class _TimestepEmbedding(nn.Module):
    """``linear_1 -> SiLU -> linear_2`` (AdaLN-LoRA): returns (raw sinusoidal, 3D lora table)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, sample: torch.Tensor):
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        return sample, emb  # (emb_B_T_D = raw sinusoidal, adaln_lora_B_T_3D)


class _PatchProj(nn.Module):
    """Placeholder for the einops ``Rearrange`` (no params) so ``proj.1`` is the Linear —
    matches the checkpoint key ``x_embedder.proj.1.weight``."""

    def forward(self, x):  # never called; patchify is done explicitly in the DiT forward
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, cfg: CosmosPredict2DiTConfig, in_channels: int):
        super().__init__()
        patch_dim = in_channels * cfg.patch_spatial * cfg.patch_spatial * cfg.patch_temporal
        self.ps = cfg.patch_spatial
        self.pt = cfg.patch_temporal
        self.proj = nn.Sequential(_PatchProj(), nn.Linear(patch_dim, cfg.model_channels, bias=False))

    def forward(self, x_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        # b c (t r)(h m)(w n) -> b t h w (c r m n)
        b, c, T, H, W = x_B_C_T_H_W.shape
        r, m, n = self.pt, self.ps, self.ps
        x = x_B_C_T_H_W.reshape(b, c, T // r, r, H // m, m, W // n, n)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, T // r, H // m, W // n, c * r * m * n)
        return self.proj[1](x)


class _Attention(nn.Module):
    """Self- or cross-attention with QK-norm + optional 3D RoPE, via an AttentionBackend."""

    def __init__(
        self, cfg: CosmosPredict2DiTConfig, query_dim: int, context_dim: int | None, backend: AttentionBackend
    ):
        super().__init__()
        self.is_selfattn = context_dim is None
        ctx = query_dim if context_dim is None else context_dim
        inner = cfg.head_dim * cfg.num_heads
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.eps
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = _NormWeight(self.head_dim)  # RMSNorm weight; rms is computed in forward
        self.k_proj = nn.Linear(ctx, inner, bias=False)
        self.k_norm = _NormWeight(self.head_dim)
        self.v_proj = nn.Linear(ctx, inner, bias=False)
        self.output_proj = nn.Linear(inner, query_dim, bias=False)
        self._backend = backend

    def forward(self, x, context=None, cos=None, sin=None):
        b, sq, _ = x.shape
        ctx = x if context is None else context
        sk = ctx.shape[1]
        q = self.q_proj(x).reshape(b, sq, self.n_heads, self.head_dim)
        k = self.k_proj(ctx).reshape(b, sk, self.n_heads, self.head_dim)
        v = self.v_proj(ctx).reshape(b, sk, self.n_heads, self.head_dim)
        q = _rms_norm(q, self.q_norm.weight, self.eps)
        k = _rms_norm(k, self.k_norm.weight, self.eps)
        # [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.is_selfattn and cos is not None:
            q, k = _apply_rope(q, k, cos, sin)
        out = self._backend.attend(q, k, v, attn_mask=None, scaling=None)  # scale defaults to head_dim**-0.5
        out = out.transpose(1, 2).reshape(b, sq, self.n_heads * self.head_dim)
        return self.output_proj(out)


class _NormWeight(nn.Module):
    """Fallback RMSNorm weight holder for torch without ``nn.RMSNorm``."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.activation = nn.GELU()  # exact (erf)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


def _adaln(dim: int, lora_dim: int, n_chunks: int) -> nn.Sequential:
    return nn.Sequential(
        nn.SiLU(),
        nn.Linear(dim, lora_dim, bias=False),
        nn.Linear(lora_dim, n_chunks * dim, bias=False),
    )


class _Block(nn.Module):
    def __init__(self, cfg: CosmosPredict2DiTConfig, backend: AttentionBackend):
        super().__init__()
        d = cfg.model_channels
        self.layer_norm_self_attn = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.eps)
        self.self_attn = _Attention(cfg, d, None, backend)
        self.layer_norm_cross_attn = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.eps)
        self.cross_attn = _Attention(cfg, d, cfg.crossattn_emb_channels, backend)
        self.layer_norm_mlp = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.eps)
        self.mlp = _MLP(d, int(d * cfg.mlp_ratio))
        self.adaln_modulation_self_attn = _adaln(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_cross_attn = _adaln(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_mlp = _adaln(d, cfg.adaln_lora_dim, 3)

    def forward(self, x_BTHWD, emb_BTD, crossattn, cos, sin, adaln_lora_BT3D):
        def mod3(layer):
            s, sc, g = (layer(emb_BTD) + adaln_lora_BT3D).chunk(3, dim=-1)
            r = lambda t: t.reshape(*t.shape[:2], 1, 1, t.shape[-1]).type_as(x_BTHWD)  # noqa: E731
            return r(s), r(sc), r(g)

        B, T, H, W, D = x_BTHWD.shape

        def norm_mod(x, ln, scale, shift):
            return ln(x) * (1 + scale) + shift

        # self-attention (joint over T*H*W)
        sh, sc, g = mod3(self.adaln_modulation_self_attn)
        n = norm_mod(x_BTHWD, self.layer_norm_self_attn, sc, sh).reshape(B, T * H * W, D)
        res = self.self_attn(n, None, cos, sin).reshape(B, T, H, W, D)
        x_BTHWD = x_BTHWD + g * res
        # cross-attention to text
        sh, sc, g = mod3(self.adaln_modulation_cross_attn)
        n = norm_mod(x_BTHWD, self.layer_norm_cross_attn, sc, sh).reshape(B, T * H * W, D)
        res = self.cross_attn(n, crossattn, None, None).reshape(B, T, H, W, D)
        x_BTHWD = res * g + x_BTHWD
        # mlp
        sh, sc, g = mod3(self.adaln_modulation_mlp)
        n = norm_mod(x_BTHWD, self.layer_norm_mlp, sc, sh)
        x_BTHWD = x_BTHWD + g * self.mlp(n)
        return x_BTHWD


class _FinalLayer(nn.Module):
    def __init__(self, cfg: CosmosPredict2DiTConfig):
        super().__init__()
        d = cfg.model_channels
        out = cfg.patch_spatial * cfg.patch_spatial * cfg.patch_temporal * cfg.out_channels
        self.layer_norm = nn.LayerNorm(d, elementwise_affine=False, eps=cfg.eps)
        self.linear = nn.Linear(d, out, bias=False)
        self.adaln_modulation = _adaln(d, cfg.adaln_lora_dim, 2)
        self.hidden = d

    def forward(self, x_BTHWD, emb_BTD, adaln_lora_BT3D):
        shift, scale = (self.adaln_modulation(emb_BTD) + adaln_lora_BT3D[:, :, : 2 * self.hidden]).chunk(
            2, dim=-1
        )
        r = lambda t: t.reshape(*t.shape[:2], 1, 1, t.shape[-1])  # noqa: E731
        x = self.layer_norm(x_BTHWD) * (1 + r(scale)) + r(shift)
        return self.linear(x)


@register_video_dit("cosmos_predict2")
class CosmosPredict2DiT(VideoDiT):
    """The self-hosted Cosmos-Predict2 DiT (``MiniTrainDIT``/``MinimalV1LVGDiT`` family).

    One ``video_dit`` family (distinct from ``wan``): AdaLN-LoRA modulation, head-dim
    QK-norm, GELU MLP. Mirrors the ``net.*`` checkpoint keys 1:1."""

    def __init__(self, cfg: CosmosPredict2DiTConfig | None = None, attention: str = "sdpa"):
        super().__init__()
        self.cfg = cfg or CosmosPredict2DiTConfig()
        c = self.cfg
        backend = get_attention_backend(attention)
        in_ch = c.in_channels + 1 + (1 if c.concat_padding_mask else 0)  # +condition-mask +padding-mask
        self.x_embedder = _PatchEmbed(c, in_ch)
        self.t_embedder = nn.Sequential(
            _Timesteps(c.model_channels), _TimestepEmbedding(c.model_channels, c.model_channels)
        )
        self.t_embedding_norm = _NormWeight(c.model_channels)
        self.blocks = nn.ModuleList([_Block(c, backend) for _ in range(c.num_blocks)])
        self.final_layer = _FinalLayer(c)
        self._rope_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    # ---- 3D RoPE (recomputed; deterministic, not loaded) --------------------
    def _rope_cos_sin(self, T: int, H: int, W: int, device, dtype):
        key = (T, H, W, str(device))
        if key in self._rope_cache:
            return self._rope_cache[key]
        c = self.cfg
        hd = c.head_dim
        dim_h = hd // 6 * 2
        dim_w = dim_h
        dim_t = hd - 2 * dim_h
        spatial_range = torch.arange(0, dim_h, 2, device=device)[: dim_h // 2].float() / dim_h
        temporal_range = torch.arange(0, dim_t, 2, device=device)[: dim_t // 2].float() / dim_t
        h_ntk = c.rope_h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        w_ntk = c.rope_w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        t_ntk = c.rope_t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        h_freqs = 1.0 / ((10000.0 * h_ntk) ** spatial_range)
        w_freqs = 1.0 / ((10000.0 * w_ntk) ** spatial_range)
        t_freqs = 1.0 / ((10000.0 * t_ntk) ** temporal_range)
        seq = torch.arange(max(H, W, T), device=device).float()
        emb_h = torch.outer(seq[:H], h_freqs)  # [H, dim_h/2]
        emb_w = torch.outer(seq[:W], w_freqs)
        emb_t = torch.outer(seq[:T], t_freqs)  # [T, dim_t/2]
        # cat([t,h,w] repeated over the T,H,W grid) * 2 -> [T,H,W,head_dim]
        et = emb_t[:, None, None, :].expand(T, H, W, emb_t.shape[-1])
        eh = emb_h[None, :, None, :].expand(T, H, W, emb_h.shape[-1])
        ew = emb_w[None, None, :, :].expand(T, H, W, emb_w.shape[-1])
        ang = torch.cat([et, eh, ew, et, eh, ew], dim=-1).reshape(T * H * W, hd)  # duplicated halves
        cos, sin = torch.cos(ang), torch.sin(ang)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.cfg
        # 1) concat condition mask (17th ch) then padding mask (18th ch)
        x = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        _, _, T, H, W = x.shape
        if c.concat_padding_mask:
            if padding_mask is None:
                padding_mask = x.new_zeros((x.shape[0], 1, H, W))
            pm = F.interpolate(padding_mask, size=(H, W), mode="nearest")
            x = torch.cat([x, pm.unsqueeze(2).repeat(1, 1, T, 1, 1)], dim=1)
        # 2) patchify -> [B, T, H', W', D]
        x_BTHWD = self.x_embedder(x)
        Tt, Ht, Wt = x_BTHWD.shape[1:4]
        cos, sin = self._rope_cos_sin(Tt, Ht, Wt, x.device, x_BTHWD.dtype)
        # 3) timestep embedding + adaln-lora table
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        emb_BTD, adaln_lora_BT3D = self.t_embedder(timesteps_B_T)
        emb_BTD = _rms_norm(emb_BTD, self.t_embedding_norm.weight, c.eps)
        # 4) transformer blocks
        for block in self.blocks:
            x_BTHWD = block(x_BTHWD, emb_BTD, crossattn_emb, cos, sin, adaln_lora_BT3D)
        # 5) final layer + unpatchify -> [B, out_ch, T, H, W]
        x_BTHWO = self.final_layer(x_BTHWD, emb_BTD, adaln_lora_BT3D)
        return self._unpatchify(x_BTHWO)

    def _unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        B, T, H, W, M = x_B_T_H_W_M.shape
        p1 = p2 = c.patch_spatial
        t = c.patch_temporal
        C = c.out_channels
        x = x_B_T_H_W_M.reshape(B, T, H, W, p1, p2, t, C)
        # B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)
        x = x.permute(0, 7, 1, 6, 2, 4, 3, 5).reshape(B, C, T * t, H * p1, W * p2)
        return x


def load_cosmos_predict2_dit(
    state_dict: dict, cfg: CosmosPredict2DiTConfig | None = None, attention: str = "sdpa"
) -> CosmosPredict2DiT:
    """Build a :class:`CosmosPredict2DiT` and load the ``net.*`` checkpoint into it.

    Strips the ``net.`` prefix, drops TransformerEngine ``_extra_state`` and the
    ``accum_*`` training counters, and skips the recomputed ``pos_embedder`` buffers.
    Verifies every parameter is matched.
    """
    dit = CosmosPredict2DiT(cfg, attention=attention)
    remapped = {}
    for k, v in state_dict.items():
        key = k[4:] if k.startswith("net.") else k
        if key.endswith("_extra_state") or key.startswith("accum_") or key.startswith("pos_embedder."):
            continue
        remapped[key] = v
    missing, unexpected = dit.load_state_dict(remapped, strict=False)
    if missing:
        raise RuntimeError(f"CosmosPredict2DiT missing keys after load: {missing[:8]} ({len(missing)} total)")
    if unexpected:
        raise RuntimeError(f"CosmosPredict2DiT unexpected keys: {unexpected[:8]} ({len(unexpected)} total)")
    return dit
