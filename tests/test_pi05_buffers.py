"""Meta loading must rebuild checkpoint-omitted buffers, including shared aliases."""

from types import SimpleNamespace

import pytest
import torch

from embodiinfer.policies.pi05.checkpoints.lerobot import _materialize_buffers
from embodiinfer.policies.pi05.embeddings import rope_tables, time_embedding


class Rotary(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        value = 1.0 / config.theta ** (torch.arange(0, 8, 2, device=device).float() / 8)
        self.register_buffer("inv_freq", value, persistent=False)
        self.register_buffer("original_inv_freq", value, persistent=False)


def test_materialize_restores_all_five_nonpersistent_buffers():
    model = torch.nn.Module()
    model.vision = torch.nn.Module()
    model.vision.register_buffer(
        "position_ids", torch.empty(1, 256, device="meta", dtype=torch.long), persistent=False
    )
    for name in ("language", "expert"):
        tower = torch.nn.Module()
        tower.rotary_emb = Rotary(SimpleNamespace(theta=10000.0), device="meta")
        setattr(model, name, tower)
    assert model.state_dict() == {}
    _materialize_buffers(model, torch.device("cpu"))
    assert torch.equal(model.vision.position_ids, torch.arange(256)[None])
    expected = Rotary(SimpleNamespace(theta=10000.0), "cpu").inv_freq
    for tower in (model.language, model.expert):
        for name in ("inv_freq", "original_inv_freq"):
            torch.testing.assert_close(getattr(tower.rotary_emb, name), expected, atol=0, rtol=0)
    assert not any(b.is_meta for b in model.buffers())


def test_unknown_meta_buffer_fails_explicitly():
    model = torch.nn.Module()
    model.child = torch.nn.Module()
    model.child.register_buffer("unknown", torch.empty(2, device="meta"), persistent=False)
    with pytest.raises(RuntimeError, match="unhandled PI0.5 runtime buffer"):
        _materialize_buffers(model, torch.device("cpu"))


def test_native_embeddings_reject_unsupported_variants():
    with pytest.raises(ValueError, match="default RoPE"):
        rope_tables(SimpleNamespace(rope_type="dynamic"), torch.empty(1), torch.empty(1))
    with pytest.raises(ValueError, match="must be even"):
        time_embedding(torch.ones(1), 3, 0.004, 4.0, torch.device("cpu"))
