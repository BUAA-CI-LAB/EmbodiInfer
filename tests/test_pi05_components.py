"""Small CPU models check EmbodiInfer leaf math against the installed LeRobot stack."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lerobot")
from lerobot.policies.pi_gemma import PiGemmaModel
from transformers import GemmaConfig, SiglipVisionConfig, SiglipVisionModel

from embodiinfer.layers import get_attention_backend
from embodiinfer.policies.base import FlowVLAPolicy
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.embeddings import attention_mask_4d as _mask4d
from embodiinfer.policies.pi05.embeddings import make_attention_mask, time_embedding
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy


def blank_policy():
    policy = Pi05Policy.__new__(Pi05Policy)
    FlowVLAPolicy.__init__(policy, VLAPolicyConfig())
    policy.attention = "eager"
    policy.native_embeddings = True
    policy.compile_backend = "none"
    policy.denoise_attention = "sdpa"
    policy._attn = get_attention_backend("eager")
    policy._vision_attn = get_attention_backend("sdpa")
    return policy


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("conditional", [False, True])
def test_gemma_tower_matches_reference(dtype, conditional):
    cfg = GemmaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        hidden_activation="gelu_pytorch_tanh",
        use_adarms=conditional,
        adarms_cond_dim=32 if conditional else None,
    )
    cfg._attn_implementation = "eager"
    tower = PiGemmaModel(cfg).eval().to(dtype)
    for name, parameter in tower.named_parameters():
        if "norm" in name:
            parameter.data = parameter.data.float()
    x = torch.randn(1, 5, 32)
    positions = torch.arange(5)[None]
    allowed = torch.ones(1, 5, 5, dtype=torch.bool)
    allowed[:, :, -1] = False
    mask = _mask4d(allowed)
    cond = torch.randn(1, 32) if conditional else None
    policy = blank_policy()
    with torch.inference_mode():
        expected = tower(
            inputs_embeds=x, position_ids=positions, attention_mask=mask, adarms_cond=cond, use_cache=False
        ).last_hidden_state
        actual, _ = policy._tower_forward(tower, x, positions, mask, cond)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_siglip_matches_reference_and_does_not_call_vision_forward(monkeypatch):
    vision = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=28,
            patch_size=14,
        )
    ).eval()
    vision.config._attn_implementation = "sdpa"
    projection = torch.nn.Linear(16, 32)
    pg = SimpleNamespace(
        vision_tower=vision,
        multi_modal_projector=SimpleNamespace(linear=projection),
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=32)),
    )
    policy = blank_policy()
    policy._m = SimpleNamespace(paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(model=pg)))
    image = torch.randn(1, 3, 28, 28)
    with torch.inference_mode():
        expected = projection(vision(image).last_hidden_state)
        expected = (expected / 32**0.5) * 32**0.5

        def forbidden(*args, **kwargs):
            raise AssertionError("HF vision forward called")

        monkeypatch.setattr(vision, "forward", forbidden)
        actual = policy._embed_image(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_time_and_padding_masks_match_reference():
    from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding, make_att_2d_masks

    times = torch.tensor([1.0, 0.7, 0.1])
    expected = create_sinusoidal_pos_embedding(times, 32, 0.004, 4.0, device=times.device)
    torch.testing.assert_close(time_embedding(times, 32, 0.004, 4.0, times.device), expected, atol=0, rtol=0)
    pad = torch.tensor([[True, False, True, True], [False, True, False, False]])
    groups = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])
    assert torch.equal(make_attention_mask(pad, groups), make_att_2d_masks(pad, groups))


@pytest.mark.gpu
def test_mixed_pi05_complete_graph_replays_fresh_prefix():
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

    policy = blank_policy()
    policy._config = VLAPolicyConfig(action_dim=4, action_horizon=3, default_num_steps=3)
    policy.native_inference = False
    policy.tensor_parallel = SimpleNamespace(enabled=False)
    policy.prefix_attention = "sdpa"
    policy._cached_prefix_meta = None
    policy._cached_suffix_att = None
    policy._make_att_2d_masks = make_attention_mask
    policy._sinusoidal = time_embedding
    config = GemmaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        hidden_activation="gelu_pytorch_tanh",
        use_adarms=False,
    )
    policy._prefix_tower = PiGemmaModel(config).to(torch.bfloat16)
    config = GemmaConfig(**{**config.to_dict(), "use_adarms": True, "adarms_cond_dim": 32})
    policy._expert_tower = PiGemmaModel(config).to(torch.bfloat16)
    for tower in (policy._prefix_tower, policy._expert_tower):
        for name, parameter in tower.named_parameters():
            if "norm" in name:
                parameter.data = parameter.data.float()
    model = torch.nn.Module()
    model.action_in_proj = torch.nn.Linear(4, 32)
    model.action_out_proj = torch.nn.Linear(32, 4)
    model.time_mlp_in = torch.nn.Linear(32, 32)
    model.time_mlp_out = torch.nn.Linear(32, 32)
    model.config = SimpleNamespace(min_period=0.004, max_period=4.0, chunk_size=3)
    policy._m = model
    core = EngineCore(
        policy,
        EngineConfig(
            device="cuda",
            dtype="auto",
            max_batch_size=1,
            batch_buckets=(1,),
            capture_full_loop=True,
        ),
    )
    previous = None
    with torch.inference_mode():
        for tokens in ([1, 2, 3], [3, 4, 5]):
            batch = Pi05Batch(
                [],
                [],
                torch.tensor([tokens], device="cuda"),
                torch.tensor([[True, True, False]], device="cuda"),
                ["sample"],
            )
            expected = policy.sample_actions(
                batch, generator=torch.Generator(device="cuda").manual_seed(1000)
            )
            actual = core.execute(batch, generator=torch.Generator(device="cuda").manual_seed(1000))[
                0
            ].actions
            torch.testing.assert_close(actual, expected[0].cpu(), atol=0, rtol=0)
            if previous is not None:
                assert not torch.equal(previous, actual)
            previous = actual.clone()
