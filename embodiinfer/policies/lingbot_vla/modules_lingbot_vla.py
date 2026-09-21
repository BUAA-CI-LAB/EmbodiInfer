"""Vendored LingBot-VLA action-expert modules (weight holders for option-B loading).

The LingBot-VLA policy must run in an existing env (vvla_env /
pi0.5's env), but the official ``lingbotvla`` package needs LeRobot v3.0 + torch 2.8
(conflicts with vvla_env). So — exactly as GR00T vendors its DiT and OpenVLA-OFT builds
Llama from stock modules — embodiinfer owns no runtime dependency on ``lingbotvla``: the VL
backbone is stock ``Qwen2_5_VLForConditionalGeneration`` (weight holder), and the narrow
Qwen2 action expert is defined here as plain ``nn.Module`` weight holders whose attribute
layout matches the checkpoint keys so ``load_state_dict`` routes ``qwen_expert.model.*``
directly. The forward itself (RoPE/attention/AdaRMSNorm/SwiGLU) is self-hosted in
``modeling_lingbot_vla.py``; these modules only hold parameters.

Checkpoint key layout, per expert layer ``layers.N``:
    input_layernorm.{weight, gamma.{weight,bias}, beta.{weight,bias}}   # AdaRMSNorm
    post_attention_layernorm.{weight, gamma.*, beta.*}
    self_attn.{q_proj[2048,768]+b, k_proj[256,768]+b, v_proj[256,768]+b, o_proj[768,2048]}
    mlp.{gate_proj[2752,768], up_proj[2752,768], down_proj[768,2752]}
and ``layers`` sits under ``ActionExpert`` with a trailing plain ``norm.weight``.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Plain RMSNorm weight holder (``.weight`` read by the self-hosted forward)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps


class AdaRMSNorm(nn.Module):
    """Time-conditioned RMSNorm holder: ``weight`` + ``gamma``/``beta`` Linear(dim->dim).

    The math (``(1 + gamma(cond)) * (weight * RMSNorm(x)) + beta(cond)``, fp32) lives in
    ``modeling_lingbot_vla._ada_rmsnorm``; this only holds the parameters so the checkpoint
    keys ``input_layernorm.{weight, gamma.*, beta.*}`` load in place."""

    def __init__(self, dim: int, cond_dim: int = 768, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Linear(cond_dim, dim)
        self.beta = nn.Linear(cond_dim, dim)
        self.variance_epsilon = eps


class ExpertAttention(nn.Module):
    """Expert self-attention projections (q/k/v have bias, o has none). GQA 16 q / 2 kv heads."""

    def __init__(self, hidden: int = 768, n_heads: int = 16, n_kv: int = 2, head_dim: int = 128):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)


class ExpertMLP(nn.Module):
    """Expert SwiGLU MLP (bias-free)."""

    def __init__(self, hidden: int = 768, intermediate: int = 2752):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class ExpertLayer(nn.Module):
    """One action-expert decoder layer (AdaRMSNorm input/post norms)."""

    def __init__(self, hidden: int = 768, intermediate: int = 2752, cond_dim: int = 768):
        super().__init__()
        self.self_attn = ExpertAttention(hidden)
        self.mlp = ExpertMLP(hidden, intermediate)
        self.input_layernorm = AdaRMSNorm(hidden, cond_dim)
        self.post_attention_layernorm = AdaRMSNorm(hidden, cond_dim)


class ActionExpert(nn.Module):
    """The narrow Qwen2 action expert: ``layers`` (AdaRMSNorm) + a plain final ``norm``.

    Matches checkpoint prefix ``qwen_expert.model.*`` so weights load in place. Holds
    parameters only; the coupled forward is in ``modeling_lingbot_vla``."""

    def __init__(
        self, num_layers: int = 36, hidden: int = 768, intermediate: int = 2752, cond_dim: int = 768
    ):
        super().__init__()
        self.layers = nn.ModuleList(ExpertLayer(hidden, intermediate, cond_dim) for _ in range(num_layers))
        self.norm = RMSNorm(hidden)


class ActionProjections(nn.Module):
    """The pi0-style action/state/time projection heads (checkpoint prefix ``model.*``)."""

    def __init__(self, expert_dim: int = 768, action_dim: int = 75):
        super().__init__()
        self.state_proj = nn.Linear(action_dim, expert_dim)
        self.action_in_proj = nn.Linear(action_dim, expert_dim)
        self.action_out_proj = nn.Linear(expert_dim, action_dim)
        # separate_time_proj=False: action + broadcast-time fused via action_time_mlp
        self.action_time_mlp_in = nn.Linear(2 * expert_dim, expert_dim)
        self.action_time_mlp_out = nn.Linear(expert_dim, expert_dim)
