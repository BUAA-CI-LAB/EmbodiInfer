"""Small real modules exercising the public Pi05 constructor without checkpoints."""

from types import SimpleNamespace

import pytest
import torch

from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05 import Pi05Policy


class _TinyNorm(torch.nn.Module):
    def __init__(self, width, adaptive):
        super().__init__()
        self.eps = 1e-6
        self.dense = torch.nn.Linear(width, width * 3) if adaptive else None
        if not adaptive:
            self.weight = torch.nn.Parameter(torch.zeros(width))


class _TinyTower(torch.nn.Module):
    def __init__(self, adaptive):
        super().__init__()
        width, head_dim = 64, 64
        self.embed_tokens = torch.nn.Embedding(32, width, dtype=torch.bfloat16)
        self.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.input_layernorm = _TinyNorm(width, adaptive)
            layer.post_attention_layernorm = _TinyNorm(width, adaptive)
            layer.self_attn = torch.nn.Module()
            layer.self_attn.head_dim = head_dim
            layer.self_attn.scaling = head_dim**-0.5
            for name, output in (
                ("q_proj", 4 * head_dim),
                ("k_proj", head_dim),
                ("v_proj", head_dim),
            ):
                setattr(
                    layer.self_attn, name, torch.nn.Linear(width, output, bias=False, dtype=torch.bfloat16)
                )
            layer.self_attn.o_proj = torch.nn.Linear(4 * head_dim, width, bias=False, dtype=torch.bfloat16)
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = torch.nn.Linear(width, 256, bias=False, dtype=torch.bfloat16)
            layer.mlp.up_proj = torch.nn.Linear(width, 256, bias=False, dtype=torch.bfloat16)
            layer.mlp.down_proj = torch.nn.Linear(256, width, bias=False, dtype=torch.bfloat16)
            self.layers.append(layer)
        self.norm = _TinyNorm(width, adaptive)
        self.rotary_emb = torch.nn.Module()
        self.rotary_emb.register_buffer(
            "inv_freq", 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        )
        self.rotary_emb.attention_scaling = 1.0


@pytest.fixture
def make_pi05_policy():
    policies = []

    def build(config=None, *, device="cuda"):
        class TinyPolicy(Pi05Policy):
            def _embed_image(self, image):
                return image.mean((1, 2, 3))[:, None, None].expand(-1, 3, 64).contiguous().bfloat16()

        loaded = torch.nn.Module()
        loaded.model = model = torch.nn.Module()
        model.paligemma_with_expert = pwe = torch.nn.Module()
        pwe.paligemma = torch.nn.Module()
        pwe.paligemma.model = torch.nn.Module()
        pwe.paligemma.model.language_model = _TinyTower(False)
        pwe.gemma_expert = torch.nn.Module()
        pwe.gemma_expert.model = _TinyTower(True)
        model.action_in_proj = torch.nn.Linear(8, 64)
        model.action_out_proj = torch.nn.Linear(64, 8)
        model.time_mlp_in = torch.nn.Linear(64, 64)
        model.time_mlp_out = torch.nn.Linear(64, 64)
        model.config = SimpleNamespace(chunk_size=10, min_period=0.004, max_period=4.0)
        policy = (
            TinyPolicy(
                VLAPolicyConfig(action_horizon=10, action_dim=8),
                loaded,
                native_inference=True,
                native_embeddings=True,
                prefix_cuda_graph=True,
                optimizations=config,
            )
            .eval()
            .to(device)
        )
        policies.append(policy)
        return policy

    yield build
    for policy in policies:
        policy._clear_inference_caches()
