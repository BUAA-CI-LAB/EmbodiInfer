from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from embodiinfer.models.streamvln import StreamVLNCache
from embodiinfer.models.streamvln.optimized import StreamVLNTextOptions, forward_text
from embodiinfer.policies.streamvln.tree import DraftTree, StreamVLNTreeDrafter, commit_tree_path, tree_scores


def test_tree_visibility_and_depths_exclude_siblings_and_padding():
    tree = DraftTree.from_parents([4, 2, 3, 5], [-1, 0, 0, 1], 8)
    assert tree.depths.tolist() == [[0, 1, 1, 2, 0, 0, 0, 0]]
    assert tree.ancestors[3].nonzero().flatten().tolist() == [0, 1, 3]
    assert tree.children == {(0, 2): 1, (0, 3): 2, (1, 5): 3}
    assert not tree.ancestors[:4, 4:].any()
    assert set(tree.path_tokens[3].tolist()) == {4, 2, 5}
    with pytest.raises(ValueError, match="precede"):
        DraftTree.from_parents([1, 2], [-1, 2], 2)
    with pytest.raises(ValueError, match="duplicate"):
        DraftTree.from_parents([1, 2, 2], [-1, 0, 0], 4)


def test_drafter_proposes_envelope_branches_eos_and_keeps_vocab_unrestricted():
    drafter = StreamVLNTreeDrafter((10, 11, 12), (1, 2, 3, 4), (20,), refresh=1)
    tree = drafter.tree([10], 31, 32)
    assert tree is not None
    assert tree.children[(0, 11)] == 1
    assert tree.children[(1, 12)] == 2
    assert all((2, action) in tree.children for action in (1, 2, 3, 4))
    first = tree.children[(2, 1)]
    assert (first, 20) in tree.children
    assert drafter.tree([99], 31, 32) is None
    assert drafter.tree([10, 11, 12, 20], 31, 32) is None
    # Learned proposals use only previous outputs, never restrict selection.
    for _ in range(4):
        drafter.observe([10, 11, 12, 3, 3, 3, 3, 20])
    learned = drafter.tree([10, 11, 12], 3, 4)
    assert learned.tokens[0, 1].item() == 3


@pytest.mark.parametrize("refresh", [1, 4, 64])
def test_drafter_refresh_uses_committed_counts_at_the_declared_interval(refresh):
    drafter = StreamVLNTreeDrafter((10, 11, 12), (1, 2, 3, 4), (20,), refresh=refresh)
    initial = drafter.tree([10, 11, 12], 3, 4)
    for _ in range(refresh - 1):
        drafter.observe([10, 11, 12, 3, 3, 3, 3, 20])
        assert drafter.tree([10, 11, 12], 3, 4) is initial
    drafter.observe([10, 11, 12, 3, 3, 3, 3, 20])
    refreshed = drafter.tree([10, 11, 12], 3, 4)
    assert refreshed is not initial
    assert refreshed.tokens[0, 1].item() == 3


def test_tree_scores_use_ancestor_penalties_only_and_allow_non_grammar_tokens():
    logits = torch.tensor([[3.0, 4.0, 2.0, 1.0], [3.0, 4.0, 5.0, 6.0]])
    seen = torch.tensor([True, False, False, False])
    paths = torch.tensor([[1, 1], [1, 2]])
    actual = tree_scores(logits, seen, paths, 2.0)
    expected = torch.tensor([[1.5, 2.0, 2.0, 1.0], [1.5, 2.0, 2.5, 6.0]])
    ids = expected.argmax(-1)
    assert actual[:, 0].long().tolist() == [1, 3]
    assert torch.equal(actual[:, 1], expected.log_softmax(-1).gather(1, ids[:, None])[:, 0])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_tree_commit_gathers_before_overlapping_writes_and_preserves_prefix(device):
    storage = torch.arange(2 * 2 * 16 * 4, dtype=torch.float32, device=device).reshape(2, 1, 2, 16, 4)
    cache = StreamVLNCache(storage.clone(), -storage.clone(), 3)
    advanced = commit_tree_path(cache, [0, 2, 5, 6])
    assert advanced.seq_len == 7 and cache.seq_len == 3
    assert torch.equal(advanced.key_storage[..., :3, :], storage[..., :3, :])
    assert torch.equal(advanced.key_storage[..., 3:7, :], storage[..., [3, 5, 8, 9], :])
    assert torch.equal(advanced.value_storage[..., 3:7, :], -storage[..., [3, 5, 8, 9], :])
    with pytest.raises(ValueError):
        commit_tree_path(cache, [-1])


def _small_backbone(max_context=256):
    transformers = pytest.importorskip("transformers")
    from embodiinfer.models.streamvln import StreamVLNBackbone

    torch.manual_seed(42)
    config = transformers.Qwen2Config(
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=256,
        use_sliding_window=False,
    )
    llm = transformers.Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    return StreamVLNBackbone(llm, nn.Identity(), nn.Identity(), max_context=max_context).cuda().eval()


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["gqa", "split", "sdpa"])
@pytest.mark.parametrize("projection", ["cublas", "decode", "fp32"])
def test_batched_tree_backbone_isolates_sibling_branches(kernel, projection):
    with torch.inference_mode():
        backbone = _small_backbone()
        backbone.text_options = StreamVLNTextOptions(
            True,
            True,
            kernel != "sdpa",
            "gqa" if kernel == "sdpa" else kernel,
            projection == "decode",
            projection == "fp32",
        )
        tree = DraftTree.from_parents([10, 1, 2, 3], [-1, 0, 0, 1], 4)
        tokens = tree.tokens.cuda()
        cache = backbone._allocate_cache(backbone.llm.model.embed_tokens(tokens))
        cache.key_storage.zero_()
        cache.value_storage.zero_()
        position = torch.tensor([5], device="cuda")

        def run(ids):
            return forward_text(
                backbone,
                backbone.llm.model.embed_tokens(ids),
                cache,
                position,
                key_bucket=256,
                depths=tree.depths.cuda(),
                ancestors=tree.ancestors.cuda(),
            )

        original = run(tokens).clone()
        changed = tokens.clone()
        changed[0, 2] = 51
        actual = run(changed)
        assert torch.equal(original[:, [0, 1, 3]], actual[:, [0, 1, 3]])
        assert not torch.equal(original[:, 2], actual[:, 2])


@pytest.mark.gpu
def test_tree_graph_replays_dynamic_trees_and_rejects_changed_cache():
    from embodiinfer.backend.triton import GreedyWorkspace
    from embodiinfer.policies.streamvln.tree import _TreeCall

    with torch.inference_mode():
        backbone = _small_backbone()
        backbone.text_options = StreamVLNTextOptions(True, True, True, "gqa", True)
        tree = DraftTree.from_parents([10, 1, 2, 3], [-1, 0, 0, 1], 4)
        cache = backbone._allocate_cache(backbone.llm.model.embed_tokens(tree.tokens.cuda()))
        cache.key_storage.zero_()
        cache.value_storage.zero_()
        workspace = GreedyWorkspace.allocate(64, torch.device("cuda"))
        workspace.reset()
        decoder = SimpleNamespace(policy=SimpleNamespace(backbone=backbone, repetition_penalty=1.05))
        eager = _TreeCall(decoder, cache, workspace, 4, 256, capture=False)
        graph = _TreeCall(decoder, cache, workspace, 4, 256, capture=True)
        for start, tokens in ((5, [10, 1, 2, 3]), (9, [11, 4, 5, 6])):
            candidate = DraftTree.from_parents(tokens, [-1, 0, 0, 1], 4)
            expected = eager.replay(candidate, cache.advance(start)).clone()
            actual = graph.replay(candidate, cache.advance(start))
            assert torch.equal(actual, expected)
        assert graph.matches(cache, workspace)
        assert not graph.matches(
            StreamVLNCache(cache.key_storage.clone(), cache.value_storage.clone(), 0), workspace
        )
        backbone.llm.lm_head.weight = nn.Parameter(backbone.llm.lm_head.weight.clone())
        assert not graph.matches(cache, workspace)
        with pytest.raises(ValueError, match="profile changed"):
            graph.replay(tree, cache)


def _cpu_staged_call(monkeypatch):
    from embodiinfer.policies.streamvln.tree import _TreeCall

    backbone = nn.Linear(1, 1)
    backbone.text_options = StreamVLNTextOptions()
    backbone.register_buffer("quantized_weight", torch.ones(8, dtype=torch.int8))
    call = _TreeCall.__new__(_TreeCall)
    call.decoder = SimpleNamespace(policy=SimpleNamespace(backbone=backbone, repetition_penalty=1.05))
    call.workspace = SimpleNamespace(penalized=torch.zeros(64, dtype=torch.bool))
    storage = torch.zeros(1, 1, 1, 16, 4)
    call.cache = StreamVLNCache(storage, storage.clone(), 0)
    call.tokens = torch.zeros(1, 4, dtype=torch.long)
    call.depths = torch.zeros_like(call.tokens)
    call.ancestors = torch.eye(4, dtype=torch.bool)
    call.paths = torch.zeros(4, 4, dtype=torch.long)
    call.position = torch.zeros(1, dtype=torch.long)
    call.options = backbone.text_options
    call.penalty = 1.05
    call.stream = 123
    call.owns_staging_cache = True
    call.parameters = call._parameter_signature()
    call.graph = SimpleNamespace(replay=lambda: None)
    call.output = torch.zeros(4, 2)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=123))
    return call


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("capacity", [12, 16, 24])
def test_tree_staging_preserves_caller_prefix_and_copies_only_verified_nodes(monkeypatch, fail, capacity):
    call = _cpu_staged_call(monkeypatch)
    original = torch.arange(capacity * 4, dtype=torch.float32).reshape(1, 1, 1, capacity, 4)
    cache = StreamVLNCache(original.clone(), -original.clone(), 5)
    tree = DraftTree.from_parents([10, 1, 2, 3], [-1, 0, 0, 1], 4)

    def replay():
        assert torch.equal(call.cache.key_storage[..., :5, :], original[..., :5, :])
        call.cache.key_storage.fill_(200)
        call.cache.value_storage.fill_(-200)
        if fail:
            raise RuntimeError("simulated graph failure")

    call.graph.replay = replay
    if fail:
        with pytest.raises(RuntimeError, match="simulated graph failure"):
            call.replay(tree, cache)
        assert torch.equal(cache.key_storage, original)
        assert torch.equal(cache.value_storage, -original)
    else:
        assert call.replay(tree, cache) is call.output
        for actual, before, value in (
            (cache.key_storage, original, 200),
            (cache.value_storage, -original, -200),
        ):
            assert torch.equal(actual[..., :5, :], before[..., :5, :])
            assert torch.equal(actual[..., 9:, :], before[..., 9:, :])
            assert (actual[..., 5:9, :] == value).all()


def test_tree_staging_rejects_borrowed_storage_and_replaced_quantized_buffers(monkeypatch):
    call = _cpu_staged_call(monkeypatch)
    cache = StreamVLNCache(call.cache.key_storage.clone(), call.cache.value_storage.clone(), 0)
    assert call.can_stage(cache, call.workspace)
    call.owns_staging_cache = False
    assert not call.can_stage(cache, call.workspace)
    call.owns_staging_cache = True
    call.decoder.policy.backbone.quantized_weight = call.decoder.policy.backbone.quantized_weight.clone()
    assert not call.can_stage(cache, call.workspace)


@pytest.mark.gpu
@pytest.mark.parametrize("compact", [False, True])
def test_int8_tree_graph_replays_fresh_window_caches_with_exact_kv_and_scores(compact):
    from embodiinfer.backend.triton import GreedyWorkspace
    from embodiinfer.layers.linear import INT8Config
    from embodiinfer.policies.streamvln.policy import _quantize_backbone
    from embodiinfer.policies.streamvln.tree import StreamVLNTreeRuntime, _TreeCall

    with torch.inference_mode():
        backbone = _small_backbone(max_context=4096 if compact else 256)
        _quantize_backbone(backbone, INT8Config(backend="native"))
        backbone.text_options = StreamVLNTextOptions(True, True, True, "gqa", True)
        backbone.kv_cache_initial_capacity = 512 if compact else 0
        policy = SimpleNamespace(
            backbone=backbone,
            repetition_penalty=1.05,
            processor=SimpleNamespace(response_prefix_ids=(10,)),
            action_token_ids=(1, 2, 3, 4),
            eos_token_ids=(20,),
            tree_refresh=64,
        )
        decoder = SimpleNamespace(policy=policy, _startup_capture_active=True)
        runtime = StreamVLNTreeRuntime(decoder)
        workspace = GreedyWorkspace.allocate(64, torch.device("cuda"))
        workspace.reset()
        held_caches = []
        for index, (start, tokens) in enumerate(((5, [10, 1, 2, 3]), (9, [11, 4, 5, 6]), (3, [12, 6, 7, 8]))):
            start = (330, 590, 1200)[index] if compact else start
            tree = DraftTree.from_parents(tokens, [-1, 0, 0, 1], 4)
            embeds = backbone.llm.model.embed_tokens(tree.tokens.cuda())
            cache = backbone._allocate_cache(embeds, capacity=(512 << index) if compact else None).advance(
                start
            )
            cache.key_storage.normal_()
            cache.value_storage.normal_()
            before = (cache.key_storage.clone(), cache.value_storage.clone())
            expected_cache = backbone._allocate_cache(embeds).advance(start)
            expected_cache.key_storage[..., : cache.capacity, :].copy_(cache.key_storage)
            expected_cache.value_storage[..., : cache.capacity, :].copy_(cache.value_storage)
            eager = _TreeCall(decoder, expected_cache, workspace, 4, backbone.max_context, capture=False)
            expected = eager.replay(tree, expected_cache).clone()
            actual = runtime.verify(tree, cache, workspace)
            assert torch.equal(actual, expected)
            for value, reference in zip((cache.key_storage, cache.value_storage), before, strict=True):
                assert torch.equal(value[..., :start, :], reference[..., :start, :])
                assert torch.equal(value[..., start + 4 :, :], reference[..., start + 4 :, :])
            assert torch.equal(cache.key_storage, expected_cache.key_storage[..., : cache.capacity, :])
            assert torch.equal(cache.value_storage, expected_cache.value_storage[..., : cache.capacity, :])
            for previous, snapshot in held_caches:
                assert torch.equal(previous.key_storage, snapshot[0])
                assert torch.equal(previous.value_storage, snapshot[1])
            held_caches.append((cache, (cache.key_storage.clone(), cache.value_storage.clone())))
            decoder._startup_capture_active = False
        assert len(runtime.graphs) == 1
        assert runtime.counters["tree_replays"] == 3
        assert runtime.counters["tree_staged_replays"] == 3
        assert runtime.counters["tree_eager_fallbacks"] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("fused", [False, True])
def test_siglip_selected_layer_adaptation_matches_reference(fused, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from embodiinfer.models.streamvln.optimized import encode_siglip

    torch.manual_seed(42)
    config = transformers.SiglipVisionConfig(
        hidden_size=144,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
    )
    with torch.inference_mode():
        vision = transformers.SiglipVisionModel(config).to(device="cuda", dtype=torch.bfloat16).eval()
        pixels = torch.randn(2, 3, 32, 32, device="cuda", dtype=torch.bfloat16)
        backbone = SimpleNamespace(vision_tower=vision, vision_feature_layer=-2)
        if fused:
            # Validate the adapter against HF with the same attention operator.
            # FP32-probability attention is a deliberate different numerical
            # profile from SDPA, whose probabilities can be rounded to BF16.
            from types import MethodType

            from embodiinfer.backend.triton.vision_attention import segment_attention

            def attend(module, hidden_states, attention_mask=None, **kwargs):
                b, t, _ = hidden_states.shape
                h, d = module.num_heads, module.head_dim
                q, k, v = [
                    p(hidden_states).reshape(-1, h, d).contiguous()
                    for p in (module.q_proj, module.k_proj, module.v_proj)
                ]
                offsets = torch.arange(b + 1, device=q.device, dtype=torch.int32) * t
                output = segment_attention(q, k, v, offsets, t, scale=module.scale)
                return module.out_proj(output.reshape(b, t, -1)), None

            for layer in vision.vision_model.encoder.layers:
                monkeypatch.setattr(layer.self_attn, "forward", MethodType(attend, layer.self_attn))
        expected = vision(pixels, output_hidden_states=True).hidden_states[-2]
        actual = encode_siglip(backbone, pixels, fused=fused)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [64, 72, 128])
def test_segment_attention_matches_fp32_probability_math(dim):
    from embodiinfer.backend.triton.vision_attention import segment_attention

    torch.manual_seed(42)
    q, k, v = [torch.randn(163, 2, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    offsets = torch.tensor([0, 17, 163], device="cuda", dtype=torch.int32)
    actual = segment_attention(q, k, v, offsets, 146)
    for start, end in ((0, 17), (17, 163)):
        qd, kd, vd = [x[start:end].double().transpose(0, 1) for x in (q, k, v)]
        exact = ((qd @ kd.transpose(-1, -2)) * dim**-0.5).softmax(-1) @ vd
        error = (actual[start:end].double().transpose(0, 1) - exact).abs()
        # One BF16 output rounding plus FP32 softmax/reduction slack. This
        # tests the operator's stated contract independently of HF SDPA.
        assert (error <= exact.abs() * 2**-8 + 2e-5).all()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "sequence,budget",
    [
        ([10, 11, 12, 1, 2, 3, 20], 16),
        ([10, 11, 12, 2, 20], 16),
        ([25, 26, 20], 16),
        ([10, 11, 12, 4, 4, 4, 4, 20], 5),
    ],
)
def test_tree_generation_eos_fallback_budget_and_cancellation(sequence, budget):
    from embodiinfer.backend.triton import GreedyWorkspace
    from embodiinfer.exceptions import SessionCancelledError
    from embodiinfer.policies.streamvln.policy import StreamVLNMemory, StreamVLNPrefix
    from embodiinfer.policies.streamvln.tree import generate_tree_tokens

    with torch.inference_mode():
        storage = torch.zeros((1, 1, 1, 64, 1), device="cuda")
        cache = StreamVLNCache(storage, storage.clone(), 3)
        workspace = GreedyWorkspace.allocate(32, torch.device("cuda"))
        workspace.reset()

        def next_pair(index):
            return [sequence[min(index, len(sequence) - 1)], -0.25]

        class Backbone:
            def decode_token_graph(self, cache, tokens, position):
                cache.key_storage[..., cache.seq_len, 0] = tokens[0, 0]
                cache.value_storage[..., cache.seq_len, 0] = -tokens[0, 0]
                return cache.seq_len - 3 + 1

        class Runtime:
            drafter = StreamVLNTreeDrafter((10, 11, 12), (1, 2, 3, 4), (20,), refresh=1)
            counters = dict.fromkeys(
                ("tree_blocks", "tree_candidates", "tree_accepted_tokens", "tree_fallback_tokens"), 0
            )

            def verify(self, tree, cache, workspace):
                start = cache.seq_len
                width = tree.tokens.shape[1]
                cache.key_storage[..., start : start + width, 0] = tree.tokens.to("cuda")
                cache.value_storage[..., start : start + width, 0] = -tree.tokens.to("cuda")
                return torch.tensor(
                    [next_pair(start - 3 + depth + 1) for depth in tree.depths[0].tolist()], device="cuda"
                )

        class Decoder:
            policy = SimpleNamespace(
                backbone=Backbone(), max_new_tokens=budget, tree_budget=31, eos_token_ids={20}
            )
            _tree_runtime = Runtime()

            def _get_workspace(self, hidden):
                workspace.reset()
                return workspace

            def _select_token(self, hidden, workspace, _):
                token, score = next_pair(hidden)
                return torch.tensor([[token]], device="cuda"), torch.tensor([[score]], device="cuda")

            def _get_cuda_graph(self, *args):
                return None

        decoder = Decoder()
        prefix = StreamVLNPrefix(StreamVLNMemory(cache, (), 1, "go"), 0)
        actual = generate_tree_tokens(decoder, prefix)
        assert actual.token_ids.tolist() == [sequence[:budget]]
        assert actual.cache.seq_len == 3 + min(budget, len(sequence))
        assert actual.stop_reason == ("eos" if len(sequence) <= budget else "max_tokens")
        assert cache.seq_len == 3
        assert torch.equal(
            actual.cache.key_storage[0, 0, 0, 3 : actual.cache.seq_len, 0].long(), actual.token_ids[0]
        )
        saved = cache.key_storage[..., :3, :].clone()
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls == 2

        with pytest.raises(SessionCancelledError):
            generate_tree_tokens(decoder, prefix, cancelled=cancelled)
        assert cache.seq_len == 3 and torch.equal(cache.key_storage[..., :3, :], saved)
