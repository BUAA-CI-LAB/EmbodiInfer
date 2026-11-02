"""Memory-bounded checkpoint conversion must preserve the existing INT8 payload."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from embodiinfer.layers.linear import INT8Config
from embodiinfer.models.streamvln.backbone import StreamVLNBackbone, _linear
from embodiinfer.models.streamvln.loading import _load_int8_checkpoint_weights
from embodiinfer.policies.streamvln.policy import _quantize_backbone


def _model():
    llm = nn.Module()
    llm.config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=10000,
        rms_norm_eps=1e-6,
    )
    llm.model = nn.Module()
    llm.model.embed_tokens = nn.Embedding(16, 8)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    for name, width in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4)):
        setattr(layer.self_attn, name, nn.Linear(8, width, bias=True))
    layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = nn.Linear(8, 12, bias=False)
    layer.mlp.up_proj = nn.Linear(8, 12, bias=False)
    layer.mlp.down_proj = nn.Linear(12, 8, bias=False)
    layer.input_layernorm = nn.LayerNorm(8)
    llm.model.layers = nn.ModuleList([layer])
    llm.lm_head = nn.Linear(8, 16, bias=False)
    return StreamVLNBackbone(llm, nn.Linear(3, 8), nn.Linear(8, 8), max_context=32).to(torch.bfloat16)


@pytest.mark.parametrize("ignored", [(), ("*.self_attn.qkv_proj", "*.mlp.down_proj")])
def test_streamed_int8_matches_fused_conversion_across_shards(tmp_path, ignored):
    torch.manual_seed(42)
    expected = _model()
    actual = copy.deepcopy(expected).to("meta")
    state = {}
    for module, prefix in (
        (expected.llm, ""),
        (expected.vision_tower, "model.vision_tower.vision_tower."),
        (expected.projector, "model.mm_projector."),
    ):
        state.update({prefix + key: value for key, value in module.state_dict().items()})
    shards = [{}, {}]
    for i, (key, value) in enumerate(state.items()):
        shards[i % 2][key] = value
    index = {}
    for i, shard in enumerate(shards):
        name = f"part-{i}.safetensors"
        save_file(shard, tmp_path / name)
        index.update(dict.fromkeys(shard, name))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    config = INT8Config(backend="torch", ignored_layers=ignored)
    _quantize_backbone(expected, config)
    names = _load_int8_checkpoint_weights(
        tmp_path,
        actual.llm,
        actual.vision_tower,
        actual.projector,
        torch.bfloat16,
        torch.device("cpu"),
        config,
    )
    assert names == expected.quantized_layers
    for left, right in (
        (actual.llm, expected.llm),
        (actual.vision_tower, expected.vision_tower),
        (actual.projector, expected.projector),
    ):
        assert left.state_dict().keys() == right.state_dict().keys()
        for key, value in left.state_dict().items():
            assert torch.equal(value, right.state_dict()[key]), key
    inputs = torch.randn(2, 8, dtype=torch.bfloat16)
    assert torch.equal(
        _linear(actual.llm.model.layers[0].self_attn.qkv_proj, inputs),
        _linear(expected.llm.model.layers[0].self_attn.qkv_proj, inputs),
    )


def test_streamed_loader_rejects_missing_keys_before_materializing(tmp_path):
    model = _model().to("meta")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    with pytest.raises(RuntimeError, match="key mismatch"):
        _load_int8_checkpoint_weights(
            tmp_path,
            model.llm,
            model.vision_tower,
            model.projector,
            torch.bfloat16,
            torch.device("cpu"),
            INT8Config(),
        )
    assert all(parameter.is_meta for parameter in model.parameters())
