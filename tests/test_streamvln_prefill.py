"""Quantized prefill graphs must retain eager arithmetic and private session KV."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from embodiinfer.models.streamvln import StreamVLNBackbone
from embodiinfer.models.streamvln.optimized import StreamVLNTextOptions
from embodiinfer.models.streamvln.prefill import StreamVLNQuantizedPrefillRunner, StreamVLNVisionRunner


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["sdpa", "fused"])
def test_explicit_siglip_vision_graphs_preserve_eager_bf16_rounding(kernel):
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
        backbone = SimpleNamespace(vision_tower=vision, vision_feature_layer=-2, vision_kernel=kernel)
        backbone._encode_frames_eager = lambda pixels: encode_siglip(
            backbone, pixels, fused=kernel == "fused"
        )
        runner = StreamVLNVisionRunner(backbone)
        runner.begin_startup_capture()
        for index, count in enumerate((1, 2, 1, 2, 1)):
            pixels = torch.randn(count, 3, 32, 32, device="cuda", dtype=torch.bfloat16)
            expected = backbone._encode_frames_eager(pixels)
            actual = runner.run(pixels)
            assert torch.equal(actual, expected)
            if index == 1:
                runner.finish_startup_capture()
        assert runner.graph_replays == 5 and len(runner.graphs) == 2
        pixels = torch.randn(3, 3, 32, 32, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(runner.run(pixels), backbone._encode_frames_eager(pixels))
        assert runner.eager_fallbacks == 1 and len(runner.graphs) == 2


@pytest.mark.parametrize("fail", [False, True])
def test_staged_prefill_does_not_publish_prefix_writes_or_failed_suffix(monkeypatch, fail):
    llm = nn.Module()
    llm.anchor = nn.Parameter(torch.zeros(1))
    llm.config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    llm.model = nn.Module()
    llm.model.layers = nn.ModuleList([nn.Identity()])
    backbone = StreamVLNBackbone(llm, nn.Identity(), nn.Identity(), max_context=32)
    inputs = torch.zeros(1, 3, 4)
    committed = backbone._allocate_cache(inputs).advance(5)
    committed.key_storage.fill_(7)
    committed.value_storage.fill_(-7)
    staging = backbone._allocate_cache(inputs)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=17))
    monkeypatch.setattr(backbone, "text_graph_workspace", lambda _: (staging, (0, 1)))

    def forward(embeddings, cache, position, key_bucket):
        assert torch.equal(cache.key_storage[..., :5, :], committed.key_storage[..., :5, :])
        cache.key_storage.fill_(100)
        cache.value_storage.fill_(-100)
        if fail:
            raise RuntimeError("simulated prefill failure")
        return embeddings + 1

    class Capture:
        def __init__(self, call, *args):
            self.call = call

        def replay(self, inputs, past_length):
            return self.call(inputs, torch.tensor([past_length]))

    monkeypatch.setattr(backbone, "prefill_embeddings_graph", forward)
    monkeypatch.setattr("embodiinfer.models.streamvln.prefill._CapturedPrefillCall", Capture)
    runner = StreamVLNQuantizedPrefillRunner(backbone)
    runner.begin_startup_capture()
    if fail:
        with pytest.raises(RuntimeError, match="simulated prefill failure"):
            runner.run(inputs, committed)
        assert (committed.key_storage == 7).all() and (committed.value_storage == -7).all()
        assert runner.graph_replays == 0
    else:
        result, hidden = runner.run(inputs, committed)
        assert result.seq_len == 8 and committed.seq_len == 5
        assert torch.equal(hidden, inputs + 1)
        assert (result.key_storage[..., 5:8, :] == 100).all()
        assert (result.value_storage[..., 5:8, :] == -100).all()
        assert (result.key_storage[..., :5, :] == 7).all()
        assert (result.key_storage[..., 8:, :] == 7).all()
        assert runner.graph_replays == 1


@pytest.mark.gpu
def test_quantized_prefill_and_tree_share_staging_with_exact_eager_outputs():
    transformers = pytest.importorskip("transformers")
    from embodiinfer.backend.triton import GreedyWorkspace
    from embodiinfer.layers.linear import INT8Config
    from embodiinfer.policies.streamvln.policy import _quantize_backbone
    from embodiinfer.policies.streamvln.tree import DraftTree, StreamVLNTreeRuntime, _TreeCall

    with torch.inference_mode():
        torch.manual_seed(42)
        config = transformers.Qwen2Config(
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=4096,
            use_sliding_window=False,
        )
        llm = transformers.Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
        backbone = StreamVLNBackbone(llm, nn.Identity(), nn.Identity(), max_context=4096).cuda().eval()
        _quantize_backbone(backbone, INT8Config(backend="native"))
        backbone.text_options = StreamVLNTextOptions(True, True, True, "gqa", True)
        backbone.kv_cache_initial_capacity = 512
        backbone.cache_generation_reserve = 16
        backbone.quantized_prefill_graphs = True
        backbone.configure_prefill_optimizations(True)
        runner = StreamVLNQuantizedPrefillRunner(backbone)
        runner.begin_startup_capture()
        policy = SimpleNamespace(
            backbone=backbone,
            repetition_penalty=1.05,
            processor=SimpleNamespace(response_prefix_ids=(10,)),
            action_token_ids=(1, 2, 3, 4),
            eos_token_ids=(20,),
            tree_refresh=64,
        )
        decoder = SimpleNamespace(policy=policy, _startup_capture_active=True)
        trees = StreamVLNTreeRuntime(decoder)
        workspace = GreedyWorkspace.allocate(64, torch.device("cuda"))
        workspace.reset()
        held = []
        for index, start in enumerate((5, 600, 17, 680, 70)):
            inputs = torch.randn(1, 32, 512, device="cuda", dtype=torch.bfloat16)
            cache = backbone._allocate_cache(inputs, capacity=512 if start < 512 else 1024).advance(start)
            cache.key_storage.normal_()
            cache.value_storage.normal_()
            before = (cache.key_storage.clone(), cache.value_storage.clone())
            reference = backbone._allocate_cache(inputs).advance(start)
            reference.key_storage[..., : cache.capacity, :].copy_(cache.key_storage)
            reference.value_storage[..., : cache.capacity, :].copy_(cache.value_storage)
            reference, expected_hidden = backbone._forward_embeddings_eager(
                inputs, reference, project_logits=False, return_hidden_states=True
            )
            actual, hidden = runner.run(inputs, cache)
            assert torch.equal(hidden, expected_hidden)
            assert torch.equal(
                actual.key_storage[..., : actual.seq_len, :],
                reference.key_storage[..., : reference.seq_len, :],
            )
            assert torch.equal(
                actual.value_storage[..., : actual.seq_len, :],
                reference.value_storage[..., : reference.seq_len, :],
            )
            for value, original in zip((cache.key_storage, cache.value_storage), before, strict=True):
                assert torch.equal(value[..., :start, :], original[..., :start, :])
            tree = DraftTree.from_parents([10, 1, 2, 3], [-1, 0, 0, 1], 4)
            eager = _TreeCall(decoder, reference, workspace, 4, backbone.max_context, capture=False)
            expected_scores = eager.replay(tree, reference).clone()
            scores = trees.verify(tree, actual, workspace)
            assert torch.equal(scores, expected_scores)
            assert torch.equal(
                actual.key_storage[..., : actual.seq_len + 4, :],
                reference.key_storage[..., : reference.seq_len + 4, :],
            )
            assert torch.equal(
                actual.value_storage[..., : actual.seq_len + 4, :],
                reference.value_storage[..., : reference.seq_len + 4, :],
            )
            assert (
                next(iter(trees.graphs.values())).cache.key_storage.data_ptr()
                == runner.cache.key_storage.data_ptr()
            )
            # Serial off-grammar fallback keeps the maximum-context attention
            # plan even when its physical session cache is smaller.
            fallback_position = torch.tensor([actual.seq_len + 4], device="cuda")
            fallback_token = torch.tensor([[53]], device="cuda")
            expected_fallback = backbone.decode_token_graph(reference, fallback_token, fallback_position)
            actual_fallback = backbone.decode_token_graph(actual, fallback_token, fallback_position)
            assert torch.equal(actual_fallback, expected_fallback)
            for previous, snapshot in held:
                assert torch.equal(previous.key_storage, snapshot[0])
                assert torch.equal(previous.value_storage, snapshot[1])
            held.append((actual, (actual.key_storage.clone(), actual.value_storage.clone())))
            if index == 1:
                runner.finish_startup_capture()
                decoder._startup_capture_active = False
        assert runner.graph_replays == 5 and len(runner.graphs) == 2
        assert trees.counters["tree_replays"] == 5 and len(trees.graphs) == 1
        # A frozen unseen shape falls back without overwriting any prior session.
        inputs = torch.randn(1, 41, 512, device="cuda", dtype=torch.bfloat16)
        fresh, _ = runner.run(inputs, None)
        assert fresh.seq_len == 41 and runner.eager_fallbacks == 1 and len(runner.graphs) == 2
        for previous, snapshot in held:
            assert torch.equal(previous.key_storage, snapshot[0])
        with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="stream changed"):
            runner.run(inputs, None)
        backbone._rope_cosine = backbone._rope_cosine.clone()
        with pytest.raises(RuntimeError, match="parameters"):
            runner.run(inputs, None)
