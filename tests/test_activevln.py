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


@torch.inference_mode()
def test_packed_memory_bulk_append_grow_and_branches_preserve_visible_values():
    kv, tokens, mask, pos = _chunk(0, 3)
    packed = torch.stack([torch.stack(pair) for pair in kv])
    committed = ActiveVLNMemory.from_chunk(packed, tokens, mask, pos, max_length=64)
    assert committed.packed_kv is not None
    snapshot = committed.packed_kv.clone()
    working = committed.fork(extra_capacity=20).to("cpu")
    assert working.capacity == 23 and working.length == 3
    kv2, tokens2, mask2, pos2 = _chunk(3, 20)
    packed2 = torch.stack([torch.stack(pair) for pair in kv2])
    working.append_chunk(packed2, tokens2, mask2, pos2)
    assert working.capacity == 23 and working.length == 23
    torch.testing.assert_close(working.packed_kv, torch.cat([packed, packed2], dim=-2), rtol=0, atol=0)
    # Force growth through the list fallback after a packed append.
    kv3, tokens3, mask3, pos3 = _chunk(23, 4)
    working.append_chunk(kv3, tokens3, mask3, pos3)
    for index, (key, value) in enumerate(working.visible_kv()):
        for slot, actual in enumerate((key, value)):
            expected = torch.cat([kv[index][slot], kv2[index][slot], kv3[index][slot]], dim=2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    branches = working.expand(2)
    branches.branches[0].layers[0].key.zero_()
    torch.testing.assert_close(committed.packed_kv, snapshot, rtol=0, atol=0)
    torch.testing.assert_close(branches.branches[1].packed_kv, working.packed_kv, rtol=0, atol=0)
    with pytest.raises(ValueError, match="nonnegative Python integer"):
        working.fork(extra_capacity=-1)


@torch.inference_mode()
def test_packed_memory_rejects_context_overflow_without_modifying_visible_prefix():
    kv, tokens, mask, pos = _chunk(0, 3)
    memory = ActiveVLNMemory.from_chunk(kv, tokens, mask, pos, max_length=4)
    snapshot = memory.packed_kv.clone()
    kv2, tokens2, mask2, pos2 = _chunk(3, 2)
    with pytest.raises(ValueError, match="exceeds max length"):
        memory.append_chunk(kv2, tokens2, mask2, pos2)
    assert memory.length == 3
    torch.testing.assert_close(memory.packed_kv, snapshot, rtol=0, atol=0)


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


@torch.inference_mode()
def test_graph_text_workspace_matches_causal_reference_with_padding():
    """Dummy query/key slots must not enter real-token attention or corrupt past KV."""
    from embodiinfer.policies.activevln.cuda_graph import ActiveVLNGraphRuntime

    torch.manual_seed(19)
    policy = ActiveVLNPolicy(_Qwen(), _Processor(), do_sample=False)
    runtime = ActiveVLNGraphRuntime.__new__(ActiveVLNGraphRuntime)
    runtime.split_attention = False
    runtime.tree_fp32_projection = False
    runtime.policy = policy
    runtime.fused_ops = False
    runtime.device = torch.device("cpu")
    runtime.storage = torch.zeros(2, 2, 1, 1, 32, 6)
    prefix = torch.randn(1, 3, 12)
    prefix_pos = torch.arange(3)[None, None].expand(3, 1, -1)
    _, prefix_kv = policy._forward_chunk(prefix, prefix_pos, None)
    for index, (key, value) in enumerate(prefix_kv):
        runtime.storage[index, 0, :, :, :3].copy_(key)
        runtime.storage[index, 1, :, :, :3].copy_(value)
    saved_prefix = runtime.storage[..., :3, :].clone()
    for query in (1, 3):
        hidden = torch.randn(1, query, 12)
        # Positions intentionally differ from the KV offset, as with image mRoPE.
        positions = (torch.arange(query) + 8)[None, None].expand(3, 1, -1)
        reference, reference_kv = policy._forward_chunk(hidden, positions, prefix_kv)
        padded = torch.nn.functional.pad(hidden, (0, 0, 0, 5 - query))
        padded_pos = torch.nn.functional.pad(positions, (0, 5 - query))
        actual = runtime._text_forward(padded, padded_pos, torch.tensor([3]), 16)
        torch.testing.assert_close(actual[:, :query], reference, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(runtime.storage[..., :3, :], saved_prefix, atol=0, rtol=0)
        for index, (key, value) in enumerate(reference_kv):
            torch.testing.assert_close(runtime.storage[index, 0, :, :, 3 : 3 + query], key)
            torch.testing.assert_close(runtime.storage[index, 1, :, :, 3 : 3 + query], value)


@torch.inference_mode()
def test_tree_attention_matches_each_independent_path_and_preserves_prefix():
    from embodiinfer.policies.activevln.cuda_graph import ActiveVLNGraphRuntime
    from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree

    torch.manual_seed(39)
    policy = ActiveVLNPolicy(_Qwen(), _Processor(), do_sample=False).eval()
    runtime = ActiveVLNGraphRuntime.__new__(ActiveVLNGraphRuntime)
    runtime.split_attention = False
    runtime.tree_fp32_projection = False
    runtime.policy, runtime.fused_ops, runtime.device = policy, False, torch.device("cpu")
    runtime.storage = torch.zeros(2, 2, 1, 1, 64, 6)
    prefix = torch.randn(1, 3, 12)
    prefix_pos = torch.arange(3)[None, None].expand(3, 1, -1)
    _, prefix_kv = policy._forward_chunk(prefix, prefix_pos, None)
    for index, (key, value) in enumerate(prefix_kv):
        runtime.storage[index, 0, :, :, :3].copy_(key)
        runtime.storage[index, 1, :, :, :3].copy_(value)
    saved_prefix = runtime.storage[..., :3, :].clone()
    paths = [[4, 5, 6, 7], [4, 8, 9], [10, 5, 9, 7]]
    tree = ActionTokenTree.from_paths("test", paths, "cpu")
    positions = (tree.depths + 11)[None, None].expand(3, 1, -1)
    actual = runtime._text_forward(
        policy._text.embed_tokens(tree.token_ids), positions, torch.tensor([3]), 32, tree.ancestors
    )
    for path in paths:
        nodes, parent = [], -1
        for token in path:
            parent = tree.children[(parent, token)]
            nodes.append(parent)
        ids = torch.tensor([path])
        pos = (torch.arange(len(path)) + 11)[None, None].expand(3, 1, -1)
        expected, kv = policy._forward_chunk(policy._text.embed_tokens(ids), pos, prefix_kv)
        torch.testing.assert_close(actual[:, nodes], expected, rtol=1e-5, atol=1e-6)
        for index, pair in enumerate(kv):
            for slot, expected_kv in enumerate(pair):
                torch.testing.assert_close(
                    runtime.storage[index, slot, :, :, [3 + n for n in nodes]],
                    expected_kv,
                    rtol=1e-5,
                    atol=1e-6,
                )
    torch.testing.assert_close(runtime.storage[..., :3, :], saved_prefix, rtol=0, atol=0)


def test_tree_repetition_penalty_uses_only_prefix_and_own_ancestors():
    from embodiinfer.policies.activevln.modeling_activevln import _apply_repetition_penalty
    from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree, tree_greedy_scores

    tree = ActionTokenTree.from_paths("test", [[3, 4, 3], [3, 7], [8, 4]], "cpu")
    kv, tokens, mask, pos = _chunk(0, 2)
    memory = ActiveVLNMemory.from_chunk(kv, tokens, mask, pos)
    logits = torch.randn(tree.token_ids.shape[1], 12, generator=torch.Generator().manual_seed(48))
    selected, logprobs = tree_greedy_scores(logits, memory, tree, 1.05)
    for index in range(tree.token_ids.shape[1]):
        history = torch.cat((tokens, tree.path_ids[index, tree.path_valid[index]][None]), dim=1)
        reference = _apply_repetition_penalty(logits[index : index + 1], history, 1.05)
        expected = reference.argmax(dim=-1, keepdim=True)
        assert selected[index].item() == expected.item()
        torch.testing.assert_close(
            logprobs[index], torch.log_softmax(reference, -1).gather(1, expected)[0, 0], rtol=0, atol=0
        )


def _cpu_tree_runtime(policy, tree):
    from threading import RLock
    from types import MethodType

    from embodiinfer.policies.activevln.cuda_graph import ActiveVLNGraphRuntime

    runtime = ActiveVLNGraphRuntime.__new__(ActiveVLNGraphRuntime)
    runtime.split_attention = False
    runtime.tree_fp32_projection = False
    runtime.policy, runtime.fused_ops, runtime.device = policy, False, torch.device("cpu")
    runtime.storage = torch.zeros(2, 2, 1, 1, 128, 6)
    runtime._resident_memory, runtime._resident_length = None, 0
    runtime.lock, runtime.action_trees = RLock(), (tree,)
    runtime.reset_stats()

    def forward(self, hidden, positions, memory, *, tree=None):
        self._restore_memory(memory)
        length = memory.length
        output = self._text_forward(
            hidden, positions, torch.tensor([length]), 128, None if tree is None else tree.ancestors
        )
        self._resident_length = length + hidden.shape[1]
        if tree is not None:
            self.counters["tree_replays"] += 1
        return output.clone(), self.storage[..., length : length + hidden.shape[1], :]

    runtime.forward = MethodType(forward, runtime)
    return runtime


@pytest.mark.parametrize("mismatch", [False, True])
@torch.inference_mode()
def test_tree_generation_matches_serial_tokens_logprobs_and_committed_cache(mismatch):
    from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPrefix
    from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "unknown " + " ".join(map(str, ids))

    processor = _Processor()
    processor.tokenizer = Tokenizer()
    torch.manual_seed(55)
    policy = ActiveVLNPolicy(_Qwen(), processor, do_sample=False, max_new_tokens=7).eval()
    policy.eos_token_ids = (99,)
    tokens = torch.tensor([[1, 3]])
    positions = torch.arange(2)[None, None].expand(3, 1, -1)
    hidden, kv = policy._forward_chunk(policy._text.embed_tokens(tokens), positions, None)
    memory = ActiveVLNMemory.from_chunk(kv, tokens, torch.ones_like(tokens), positions, max_length=128)
    next_logits = policy._lm_head(hidden[:, -1])
    expected = policy.decoder.generate_tokens(ActiveVLNPrefix(memory.fork(), next_logits))
    path = expected.token_ids[0].tolist()
    if mismatch:
        path[3] = (path[3] + 1) % 32
    # Extra proposed nodes must not be committed beyond the response budget.
    tree = ActionTokenTree.from_paths("test", [path + [4, 6, 8], [path[0], 31, 30]], "cpu")
    runtime = _cpu_tree_runtime(policy, tree)
    policy._inference_runtime = runtime
    actual = policy.decoder.generate_tokens(ActiveVLNPrefix(memory.fork(), next_logits))
    assert actual.stop_reason == expected.stop_reason == "max_tokens"
    torch.testing.assert_close(actual.token_ids, expected.token_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.token_logprobs, expected.token_logprobs, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.memory.packed_kv, expected.memory.packed_kv, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(runtime.storage[..., : actual.memory.length, :], actual.memory.packed_kv)
    assert memory.length == 2 and actual.memory.length == 9
    assert runtime.counters["tree_accepted_tokens"] >= 3
    if mismatch:
        assert runtime.counters["tree_fallback_tokens"] > 0


@torch.inference_mode()
def test_cancelled_tree_verification_keeps_working_and_committed_prefixes_unchanged():
    from embodiinfer.exceptions import SessionCancelledError
    from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPrefix
    from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "unknown"

    processor = _Processor()
    processor.tokenizer = Tokenizer()
    policy = ActiveVLNPolicy(_Qwen(), processor, do_sample=False, max_new_tokens=4).eval()
    policy.eos_token_ids = (99,)
    kv, tokens, mask, positions = _chunk(0, 2)
    kv = [(key.expand(1, 1, 2, 6), value.expand(1, 1, 2, 6)) for key, value in kv]
    committed = ActiveVLNMemory.from_chunk(kv, tokens, mask, positions, max_length=128)
    memory = committed.fork()
    snapshot = committed.packed_kv.clone()
    logits = torch.full((1, 32), -100.0)
    logits[0, 4] = 100
    tree = ActionTokenTree.from_paths("test", [[4, 5, 6]], "cpu")
    runtime = _cpu_tree_runtime(policy, tree)
    policy._inference_runtime = runtime
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks == 2

    with pytest.raises(SessionCancelledError):
        policy.decoder.generate_tokens(ActiveVLNPrefix(memory, logits), cancelled=cancelled)
    assert memory.length == committed.length == 2
    torch.testing.assert_close(committed.packed_kv, snapshot, rtol=0, atol=0)
    torch.testing.assert_close(memory.packed_kv, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("script", "limit", "reason"),
    [([3, 4, 5, 2], 10, "eos"), ([3, 4, 5, 6, 7], 10, "stop"), ([3, 4, 5, 6, 7], 3, "max_tokens")],
)
@torch.inference_mode()
def test_tree_terminal_nodes_preserve_serial_eos_stop_and_budget(script, limit, reason):
    from threading import RLock

    from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPrefix, _ActiveVLNDecoder
    from embodiinfer.policies.activevln.speculation_activevln import ActionTokenTree

    class Tokenizer:
        def decode(self, ids, **kwargs):
            words = {2: "", 3: "move", 4: " forward", 5: " 25cm", 6: ",", 7: " stop"}
            return "".join(words.get(i, "unknown") for i in ids)

    class Policy:
        do_sample = False
        repetition_penalty = 1.05
        eos_token_ids = (2,)
        max_new_tokens = limit
        tokenizer = Tokenizer()
        _inference_graphs = None

        def logits(self, generated):
            next_id = script[len(generated)] if len(generated) < len(script) else 2
            if generated != script[: len(generated)]:
                next_id = 9
            logits = torch.full((1, 12), -20.0)
            logits[0, next_id] = 20
            return logits

        def append_token(self, memory, token):
            kv = token.float().reshape(1, 1, 1, 1, 1, 1).expand(2, 2, 1, 1, 1, 1)
            positions = torch.full((3, 1, 1), memory.next_position)
            memory.append_chunk(
                kv, token, torch.ones_like(token), positions, next_position=memory.next_position + 1
            )
            return memory, self.logits(memory.token_ids[0, 1:].tolist())

    policy = Policy()
    decoder = _ActiveVLNDecoder(policy)
    tree = ActionTokenTree.from_paths("terminal", [[3, 4, 5, 2], [3, 4, 5, 6, 7, 2]], "cpu")

    class Runtime:
        action_trees = (tree,)
        lock = RLock()
        counters = {"tree_fallback_tokens": 0, "tree_accepted_tokens": 0}
        storage = torch.zeros(2, 2, 1, 1, 64, 1)

        def verify_tree(self, candidate, memory):
            history = memory.token_ids[0, 1:].tolist()
            rows = [
                policy.logits(history + candidate.path_ids[i, candidate.path_valid[i]].tolist())
                for i in range(candidate.token_ids.shape[1])
            ]
            kv = candidate.token_ids.float().reshape(1, 1, 1, 1, -1, 1).expand(2, 2, 1, 1, -1, 1)
            positions = (candidate.depths + memory.next_position)[None, None].expand(3, 1, -1)
            return torch.cat(rows)[None], kv, positions

        def bind_memory(self, memory):
            pass

    kv, tokens, mask, pos = _chunk(0, 1)
    memory = ActiveVLNMemory.from_chunk(kv, tokens, mask, pos, max_length=64)
    expected = decoder.generate_tokens(ActiveVLNPrefix(memory.fork(), policy.logits([])))
    runtime = Runtime()
    policy._inference_graphs = runtime
    actual = decoder.generate_tokens(ActiveVLNPrefix(memory.fork(), policy.logits([])))
    assert actual.stop_reason == expected.stop_reason == reason
    torch.testing.assert_close(actual.token_ids, expected.token_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.token_logprobs, expected.token_logprobs, rtol=0, atol=0)
    torch.testing.assert_close(actual.memory.packed_kv, expected.memory.packed_kv, rtol=0, atol=0)
    assert runtime.counters["tree_accepted_tokens"] == min(len(script), limit)


def test_memory_rotary_metadata_survives_fork_and_invalidates_on_untracked_append():
    kv, tokens, mask, positions = _chunk(0, 3)
    # Image rotary coordinates need not advance once per physical cache slot.
    positions = positions.clone()
    positions[:, :, 1:] = 0
    memory = ActiveVLNMemory.from_chunk(kv, tokens, mask, positions, next_position=1)
    assert memory.seq_len == 3 and memory.next_position == 1
    fork = memory.fork().to("cpu")
    assert fork.next_position == 1
    kv2, tokens2, mask2, positions2 = _chunk(4, 1)
    fork.append_chunk(kv2, tokens2, mask2, positions2)
    assert fork.next_position == 5
    assert memory.next_position == 1
    with pytest.raises(ValueError, match="nonnegative Python integer"):
        fork.append_chunk(kv2, tokens2, mask2, positions2, next_position=-1)
    assert fork.seq_len == 4


def test_differentiable_append_bypasses_inference_graph_workspace():
    class ForbiddenRuntime:
        def forward(self, *args):
            raise AssertionError("training entered the inference graph")

    torch.manual_seed(21)
    policy = ActiveVLNPolicy(_Qwen(), _Processor(), do_sample=False).eval()
    policy._inference_runtime = ForbiddenRuntime()
    with torch.enable_grad():
        tokens = torch.tensor([[1, 3]])
        positions = torch.arange(2)[None, None].expand(3, 1, -1)
        _, kv = policy._forward_chunk(policy._text.embed_tokens(tokens), positions, None)
        memory = ActiveVLNMemory.from_chunk(kv, tokens, torch.ones_like(tokens), positions)
        _, logits = policy.append_token(memory, torch.tensor([[4]]))
        logits.sum().backward()
    assert policy._text.layers[0].self_attn.q_proj.weight.grad is not None
    assert torch.isfinite(policy._text.layers[0].self_attn.q_proj.weight.grad).all()


def test_differentiable_append_detaches_shared_cache_storage_before_writing():
    torch.manual_seed(25)
    policy = ActiveVLNPolicy(_Qwen(), _Processor(), do_sample=False).eval()
    with torch.no_grad():
        tokens = torch.tensor([[1, 3]])
        positions = torch.arange(2)[None, None].expand(3, 1, -1)
        _, kv = policy._forward_chunk(policy._text.embed_tokens(tokens), positions, None)
        memory = ActiveVLNMemory.from_chunk(kv, tokens, torch.ones_like(tokens), positions)
        assert memory.packed_kv is not None
    with torch.enable_grad():
        _, logits = policy.append_token(memory, torch.tensor([[4]]))
        assert memory.packed_kv is None
        logits.sum().backward()
    gradient = policy._text.layers[0].self_attn.q_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
