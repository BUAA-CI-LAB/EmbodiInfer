import sys

import pytest
import torch

from embodiinfer.policies import available_policies, make_policy
from embodiinfer.policies.activevln.cache_activevln import ActiveVLNMemory
from embodiinfer.policies.activevln.modeling_activevln import (
    ACTIVEVLN_COMMIT,
    ACTIVEVLN_REVISION,
    ActiveVLNPolicy,
    apply_mrope,
    build_mrope_position_ids,
    mrope_cos_sin,
    rectangular_causal_mask,
)
from embodiinfer.policies.activevln.prompt_activevln import (
    FORWARD_ACTION,
    LEFT_ACTION,
    PAD_ACTION,
    RIGHT_ACTION,
    STOP_ACTION,
    actions_to_tensor,
    chat_messages,
    parse_r2r_actions,
    render_turn_text,
)


def test_activevln_registered_without_transformers_import():
    assert "activevln" in available_policies()
    assert "transformers" not in sys.modules


def test_activevln_requires_checkpoint_before_heavy_import():
    with pytest.raises(ValueError, match="needs a checkpoint"):
        make_policy("activevln")


def test_activevln_pins_are_immutable():
    assert len(ACTIVEVLN_COMMIT) == 40
    assert len(ACTIVEVLN_REVISION) == 40
    int(ACTIVEVLN_COMMIT, 16)
    int(ACTIVEVLN_REVISION, 16)


def test_r2r_prompt_and_parser_tensor_encoding():
    messages = chat_messages("Walk to the kitchen.", initial=True)
    assert messages[0]["role"] == "system"
    assert messages[1]["content"][1] == {"type": "image"}

    parsed = parse_r2r_actions("move forward 25cm, turn left 30 degrees, turn right 45 degrees")
    assert parsed.valid
    tensor, mask = actions_to_tensor(parsed)
    assert tensor.tolist() == [
        [FORWARD_ACTION, 25.0],
        [LEFT_ACTION, 30.0],
        [RIGHT_ACTION, 45.0],
    ]
    assert mask.tolist() == [True, True, True]

    defaults = parse_r2r_actions("move forward, turn left, turn right")
    assert [action.value for action in defaults.actions] == [25, 15, 15]

    stopped = parse_r2r_actions("stop")
    tensor, mask = actions_to_tensor(stopped)
    assert tensor[0].tolist() == [STOP_ACTION, 0.0]
    assert tensor[1:, 0].tolist() == [PAD_ACTION, PAD_ACTION]
    assert mask.tolist() == [True, False, False]


def test_subsequent_turn_restores_assistant_to_user_newline_boundary():
    class Processor:
        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "<|im_start|>user\nturn<|im_end|>\n<|im_start|>assistant\n"

    assert not render_turn_text(Processor(), "go", initial=True).startswith("\n")
    subsequent = render_turn_text(Processor(), "go", initial=False)
    assert subsequent.startswith("\n<|im_start|>user")
    assert "<|im_start|>system" not in subsequent
    assert "<|vision_start|><|image_pad|><|vision_end|>" in subsequent


def test_r2r_invalid_and_overlong_outputs_remain_visible():
    invalid = parse_r2r_actions("walk ahead, stop")
    assert not invalid.valid
    assert invalid.invalid_fragments == ("walk ahead",)
    assert invalid.actions[0].name == "stop"

    overlong = parse_r2r_actions("stop, move forward 25cm, turn left 15 degrees, turn right 15 degrees")
    assert overlong.truncated
    assert len(overlong.actions) == 3


def test_rectangular_mask_never_allocates_prefix_square():
    mask = rectangular_causal_mask(3, 2, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (1, 1, 2, 5)
    assert torch.equal(mask[0, 0, 0, :4], torch.tensor([0.0, 0.0, 0.0, 0.0]))
    assert mask[0, 0, 0, 4] < -1e20
    assert torch.equal(mask[0, 0, 1], torch.zeros(5))


def test_qwen_image_mrope_positions_and_rotation_shapes():
    ids = torch.tensor([[1, 10, 11, 11, 11, 11, 2]])
    positions, next_position = build_mrope_position_ids(
        ids,
        torch.tensor([[1, 4, 4]]),
        vision_start_token_id=10,
        image_token_id=11,
        spatial_merge_size=2,
    )
    assert positions.shape == (3, 1, 7)
    assert next_position == 5
    assert positions[:, 0, -1].tolist() == [4, 4, 4]

    q = torch.randn(1, 2, 7, 128)
    k = torch.randn(1, 1, 7, 128)
    cos, sin = mrope_cos_sin(positions, 128, 1_000_000.0, q.dtype)
    q_rot, k_rot = apply_mrope(q, k, cos, sin, (16, 24, 24))
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def _chunk(start, length, layers=2):
    kv = []
    for i in range(layers):
        values = torch.arange(start, start + length, dtype=torch.float32) + i * 100
        values = values.view(1, 1, length, 1)
        kv.append((values.clone(), values.clone() + 0.5))
    tokens = torch.arange(start, start + length)[None]
    mask = torch.ones_like(tokens)
    pos = torch.arange(start, start + length)[None, None].expand(3, 1, -1)
    return kv, tokens, mask, pos


def test_activevln_memory_fork_append_does_not_mutate_committed_prefix():
    kv, tokens, mask, pos = _chunk(0, 3)
    committed = ActiveVLNMemory.from_chunk(kv, tokens, mask, pos, max_length=32)
    working = committed.fork()
    kv2, tokens2, mask2, pos2 = _chunk(3, 20)
    working.append_chunk(kv2, tokens2, mask2, pos2)

    assert committed.seq_len == 3
    assert committed.token_ids.tolist() == [[0, 1, 2]]
    assert working.seq_len == 23
    assert working.capacity >= 23
    assert working.token_ids[0, -1].item() == 22


class _Norm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.variance_epsilon = 1e-6


class _Attention(torch.nn.Module):
    def __init__(self, dim=12, heads=2, kv_heads=1):
        super().__init__()
        self.head_dim = dim // heads
        self.q_proj = torch.nn.Linear(dim, dim, bias=True)
        self.k_proj = torch.nn.Linear(dim, kv_heads * self.head_dim, bias=True)
        self.v_proj = torch.nn.Linear(dim, kv_heads * self.head_dim, bias=True)
        self.o_proj = torch.nn.Linear(dim, dim, bias=False)


class _MLP(torch.nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, dim * 2, bias=False)
        self.up_proj = torch.nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = torch.nn.Linear(dim * 2, dim, bias=False)
        self.act_fn = torch.nn.SiLU()


class _Layer(torch.nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.input_layernorm = _Norm(dim)
        self.self_attn = _Attention(dim)
        self.post_attention_layernorm = _Norm(dim)
        self.mlp = _MLP(dim)


class _Text(torch.nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, dim)
        self.layers = torch.nn.ModuleList([_Layer(dim), _Layer(dim)])
        self.norm = _Norm(dim)


class _Config:
    torch_dtype = "float32"
    image_token_id = 20
    vision_start_token_id = 19

    class text_config:
        rope_scaling = {"mrope_section": [1, 1, 1]}
        rope_theta = 1_000_000.0


class _GenerationConfig:
    eos_token_id = [2]


class _Qwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.generation_config = _GenerationConfig()
        self.model = _Text()
        self.lm_head = torch.nn.Linear(12, 32, bias=False)


class _Processor:
    tokenizer = None


def test_tiny_self_hosted_forward_incremental_matches_full_last_token():
    torch.manual_seed(0)
    policy = ActiveVLNPolicy(_Qwen(), _Processor(), do_sample=False, max_new_tokens=2)
    hidden = torch.randn(1, 4, 12)
    positions = torch.arange(4)[None, None].expand(3, 1, -1)

    full, _ = policy._forward_chunk(hidden, positions, None)
    first, kv = policy._forward_chunk(hidden[:, :3], positions[:, :, :3], None)
    assert first.shape == (1, 3, 12)
    last, _ = policy._forward_chunk(hidden[:, 3:], positions[:, :, 3:], kv)

    torch.testing.assert_close(last[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)
