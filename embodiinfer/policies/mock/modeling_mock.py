"""A synthetic but faithful flow VLA, used to run and benchmark the engine.

It is *not* trained — its purpose is to reproduce the exact compute pattern of
production flow VLAs (pi0.5, GR00T, Cosmos) so that engine optimizations can be
measured honestly on real hardware:

    images+lang+state --(bidirectional VLM backbone)--> multimodal prefix
                       --(project K,V per expert layer)--> cross-attn cache
    x_t, t  --(N x [self-attn + cross-attn(prefix) + MLP])--> velocity field

The two entry points ``encode_prefix`` (once) and ``denoise_step`` (N times)
match the ``VLAPolicy`` contract the engine schedules against.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...layers import AttentionBackend, get_attention_backend
from ...types import BatchedObservation
from ...utils import sinusoidal_time_embedding
from ..base import DenseKVPrefix, FlowVLAPolicy, PrefixState, VLAPolicy
from ..factory import register_policy
from .configuration_mock import MockConfig, preset_config


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn: AttentionBackend):
        super().__init__()
        assert dim % num_heads == 0
        self.nh = num_heads
        self.hd = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn = attn  # pluggable attention kernel (stateless, shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, nh, N, hd]
        o = self.attn.attend(q, k, v)  # bidirectional (no mask) -> [B, nh, N, hd]
        o = o.transpose(1, 2).reshape(B, N, D)
        return self.proj(o)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn: AttentionBackend):
        super().__init__()
        self.nh = num_heads
        self.hd = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn = attn

    def project_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute (and cache) prefix K,V. Called once in ``encode_prefix``."""
        B, P, D = context.shape
        k = self.k_proj(context).view(B, P, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(context).view(B, P, self.nh, self.hd).transpose(1, 2)
        return k, v  # [B, nh, P, hd]

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, D = x.shape
        q = self.q_proj(x).view(B, H, self.nh, self.hd).transpose(1, 2)  # [B,nh,H,hd]
        o = self.attn.attend(q, k, v)
        o = o.transpose(1, 2).reshape(B, H, D)
        return self.out_proj(o)


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class BackboneLayer(nn.Module):
    """Bidirectional VLM backbone block (prenorm)."""

    def __init__(self, dim: int, num_heads: int, ratio: float, attn: AttentionBackend):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, attn)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class ExpertLayer(nn.Module):
    """Action-expert block: self-attn over actions + cross-attn to prefix + MLP."""

    def __init__(self, dim: int, num_heads: int, ratio: float, attn: AttentionBackend):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, attn)
        self.n2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, attn)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ratio)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross_attn(self.n2(x), k, v)
        x = x + self.mlp(self.n3(x))
        return x


class MockFlowVLA(FlowVLAPolicy):
    """A synthetic but faithful flow VLA, used to run and benchmark the engine.

    It is not trained. Its purpose is to reproduce the compute pattern of production flow
    VLAs -- a bidirectional VLM backbone producing a multimodal prefix, then an N-step
    flow-matching expert that cross-attends to that prefix -- so engine mechanisms can be
    measured on real hardware without a checkpoint. It runs on CPU, which is also why it is
    the fixture CI uses for the engine, CUDA-graph, data-parallel, and rollout paths.
    """

    def __init__(self, config: MockConfig, attention: str = "sdpa"):
        super().__init__(config)
        c = config
        D = c.hidden_dim
        self.attention = attention
        attn = get_attention_backend(attention)
        # multimodal encoders
        self.patch_embed = nn.Conv2d(3, D, kernel_size=c.patch_size, stride=c.patch_size)
        self.lang_embed = nn.Embedding(c.vocab_size, D)
        self.state_embed = nn.Linear(c.state_dim, D)
        self.prefix_pos = nn.Parameter(torch.randn(1, c.prefix_len, D) * 0.02)
        # backbone
        self.backbone = nn.ModuleList(
            [BackboneLayer(D, c.num_heads, c.mlp_ratio, attn) for _ in range(c.num_backbone_layers)]
        )
        self.backbone_norm = nn.LayerNorm(D)
        # action head (flow-matching expert)
        self.action_in = nn.Linear(c.action_dim, D)
        self.action_pos = nn.Parameter(torch.randn(1, c.action_horizon, D) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.expert = nn.ModuleList(
            [ExpertLayer(D, c.num_heads, c.mlp_ratio, attn) for _ in range(c.num_expert_layers)]
        )
        self.action_norm = nn.LayerNorm(D)
        self.action_out = nn.Linear(D, c.action_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---- stage 1: encode prefix (once) --------------------------------------
    def encode_prefix(self, batch: BatchedObservation) -> DenseKVPrefix:
        c = self.config
        images = batch.images  # [B, ncam, 3, H, W]
        B, ncam = images.shape[0], images.shape[1]
        x = images.reshape(B * ncam, *images.shape[2:])
        x = self.patch_embed(x)  # [B*ncam, D, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B*ncam, np, D]
        vis = x.reshape(B, ncam * c.num_patches, c.hidden_dim)
        lang = self.lang_embed(batch.instruction_tokens)  # [B, L, D]
        state = self.state_embed(batch.state).unsqueeze(1)  # [B, 1, D]
        prefix = torch.cat([vis, lang, state], dim=1) + self.prefix_pos
        for layer in self.backbone:
            prefix = layer(prefix)
        prefix = self.backbone_norm(prefix)
        # project + cache cross-attn K,V per expert layer
        kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            layer.cross_attn.project_kv(prefix) for layer in self.expert
        ]
        return DenseKVPrefix(kv=kv, batch_size=B)

    # ---- stage 2: one denoising step (N times) ------------------------------
    def denoise_step(self, x_t: torch.Tensor, t: torch.Tensor, prefix: PrefixState) -> torch.Tensor:
        c = self.config
        h = self.action_in(x_t) + self.action_pos  # [B, H, D]
        emb = sinusoidal_time_embedding(t, c.hidden_dim).to(h.dtype)
        temb = self.time_mlp(emb).unsqueeze(1)
        h = h + temb
        for layer, (k, v) in zip(self.expert, prefix.kv):
            h = layer(h, k, v)
        return self.action_out(self.action_norm(h))  # [B, H, A]

    # ---- CUDA-graph capability ----------------------------------------------
    @property
    def supports_cuda_graph(self) -> bool:
        return True

    def allocate_static_prefix(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> DenseKVPrefix:
        c = self.config
        nh, hd = c.num_heads, c.hidden_dim // c.num_heads
        shape = (batch_size, nh, c.prefix_len, hd)
        kv = [
            (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(c.num_expert_layers)
        ]
        return DenseKVPrefix(kv=kv, batch_size=batch_size)

    def copy_prefix_into(self, dst: DenseKVPrefix, src: DenseKVPrefix) -> None:
        for (dk, dv), (sk, sv) in zip(dst.kv, src.kv):
            dk.copy_(sk)
            dv.copy_(sv)


@register_policy("mock_flow_vla")
def _build_mock(preset: str = "small", attention: str = "sdpa", **overrides) -> VLAPolicy:
    return MockFlowVLA(preset_config(preset, **overrides), attention=attention)
