from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from embodiinfer.backend.triton.image_resize import PillowBicubicResize
from embodiinfer.exceptions import SessionCancelledError
from embodiinfer.models.streamvln import StreamVLNCache
from embodiinfer.policies.streamvln.image_gpu import StreamVLNImageTransform
from embodiinfer.policies.streamvln.policy import (
    StreamVLNDecoder,
    StreamVLNMemory,
    StreamVLNPolicy,
    StreamVLNPrefix,
)
from embodiinfer.policies.streamvln.speculation import score_repeated_candidates


@pytest.mark.gpu
@pytest.mark.parametrize("width,position", [(2, 0), (4, 61)])
def test_candidate_backbone_preserves_serial_hidden_and_kv(width, position):
    transformers = pytest.importorskip("transformers")
    from embodiinfer.models.streamvln.backbone import StreamVLNBackbone

    config = transformers.Qwen2Config(
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=256,
        use_sliding_window=False,
    )
    torch.manual_seed(42)
    with torch.inference_mode():
        llm = transformers.Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
        model = StreamVLNBackbone(llm, nn.Identity(), nn.Identity(), max_context=256).cuda().eval()
        tokens = torch.full((1, width), 4, device="cuda", dtype=torch.long)
        cache = model._allocate_cache(llm.model.embed_tokens(tokens))
        cache.key_storage.zero_()
        cache.value_storage.zero_()
        start = torch.tensor([position], device="cuda")
        serial = torch.stack(
            [model.decode_token_graph(cache, tokens[:, i : i + 1], start + i).clone() for i in range(width)],
            dim=1,
        )
        keys, values = cache.key_storage.clone(), cache.value_storage.clone()
        actual = model.decode_tokens_graph(cache, tokens, start)
        assert torch.equal(actual, serial)
        assert torch.equal(cache.key_storage, keys)
        assert torch.equal(cache.value_storage, values)


@pytest.mark.gpu
@pytest.mark.parametrize("width", [128, 3584])
def test_multirow_norm_retains_serial_unrounded_residual_semantics(width):
    from embodiinfer.backend.triton.norm import add_rms_norm, rms_norm, supports_rms_norm

    generator = torch.Generator(device="cuda").manual_seed(42)
    residual = torch.randn((1, 4, width), device="cuda", dtype=torch.bfloat16, generator=generator)
    update = torch.randn(residual.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((width,), device="cuda", dtype=torch.bfloat16, generator=generator)
    assert not supports_rms_norm(residual, weight)  # Existing prefill dispatch is unchanged.
    expected_norm = torch.cat([rms_norm(residual[:, i : i + 1], weight, 1e-6) for i in range(4)], dim=1)
    assert torch.equal(rms_norm(residual, weight, 1e-6, allow_multiple_rows=True), expected_norm)
    expected = [add_rms_norm(residual[:, i : i + 1], update[:, i : i + 1], weight, 1e-6) for i in range(4)]
    actual = add_rms_norm(residual, update, weight, 1e-6, allow_multiple_rows=True)
    for index in range(2):
        assert torch.equal(actual[index], torch.cat([pair[index] for pair in expected], dim=1))


@pytest.mark.gpu
def test_candidate_graph_replay_tracks_position_and_rejects_stale_storage():
    from embodiinfer.policies.streamvln.cuda_graph import StreamVLNVerificationGraph

    class Backbone:
        def decode_tokens_graph(self, cache, tokens, position):
            indices = position + torch.arange(tokens.shape[1], device=tokens.device)
            values = tokens.to(cache.key_storage.dtype).reshape(1, 1, 1, -1, 1)
            cache.key_storage.index_copy_(3, indices, values)
            cache.value_storage.index_copy_(3, indices, values + 1)
            return tokens[..., None].float() + indices[None, :, None]

    storage = torch.zeros((1, 1, 1, 16, 1), device="cuda")
    cache = StreamVLNCache(storage, storage.clone(), 2)
    with torch.inference_mode():
        graph = StreamVLNVerificationGraph(Backbone(), cache, 4)
        for start, ids in ((2, [2, 2, 2, 2]), (4, [3, 3, 3, 3])):
            tokens = torch.tensor([ids], device="cuda")
            hidden = graph.replay(cache.advance(start), tokens)
            expected = tokens[..., None] + torch.arange(start, start + 4, device="cuda")[None, :, None]
            assert torch.equal(hidden, expected)
            assert torch.equal(storage[0, 0, 0, start : start + 4, 0], tokens[0])
        with pytest.raises(ValueError, match="storage or token shape"):
            graph.replay(StreamVLNCache(storage.clone(), storage.clone(), 2), tokens)
        with pytest.raises(ValueError, match="storage or token shape"):
            graph.replay(cache, tokens[:, :2])
        with pytest.raises(RuntimeError, match="capacity"):
            graph.replay(cache.advance(14), tokens)


@pytest.mark.gpu
@pytest.mark.parametrize("width,position", [(2, 0), (4, 61), (8, 127)])
def test_parallel_verification_attention_matches_serial_reductions(width, position):
    from embodiinfer.backend.triton.attention import gqa_decode_graph, gqa_verify_graph

    generator = torch.Generator(device="cuda").manual_seed(42)
    queries = torch.randn((2, 8, width, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    keys = torch.randn((2, 2, 256, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    values = torch.randn(keys.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    start = torch.tensor([position], device="cuda")
    expected = torch.cat(
        [gqa_decode_graph(queries[:, :, i : i + 1], keys, values, start + i) for i in range(width)], dim=2
    )
    actual = gqa_verify_graph(queries, keys, values, start)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "shape,size", [((13, 17), (9, 21)), ((9, 21), (13, 17)), ((16, 16), (16, 16)), ((13, 17), (13, 9))]
)
def test_pillow_resize_matches_integer_reference(device, shape, size):
    images = np.random.default_rng(42).integers(0, 256, (2, *shape, 3), dtype=np.uint8)
    actual = PillowBicubicResize(*size)(torch.from_numpy(images).to(device)).cpu()
    expected = np.stack(
        [np.asarray(Image.fromarray(image).resize(size[::-1], Image.Resampling.BICUBIC)) for image in images]
    )
    assert np.array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("kind", ["rgb", "float", "binary", "near_zero", "black", "white"])
def test_siglip_device_pixels_are_bit_exact(device, kind):
    transformers = pytest.importorskip("transformers")
    processor = transformers.SiglipImageProcessor(
        size={"height": 16, "width": 24},
        do_rescale=False,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    generator = torch.Generator().manual_seed(42)
    frame = torch.randint(0, 256, (3, 19, 27), generator=generator).float() / 255
    if kind == "float":
        frame = torch.rand(frame.shape, generator=generator)
    elif kind == "binary":
        frame = (frame > 0.5).float()
    elif kind == "near_zero":
        frame = frame * 1e-9
    elif kind in ("black", "white"):
        frame.fill_(0 if kind == "black" else 1)
    expected = processor.preprocess(frame, return_tensors="pt")["pixel_values"][0]
    actual = StreamVLNImageTransform(processor)(frame, device=torch.device(device)).cpu()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("kind", ["all_bytes", "binary", "black", "white"])
def test_uint8_observations_preserve_float_reference_pixels(device, kind):
    from embodiinfer.policies.streamvln.processing import make_streamvln_observation

    transformers = pytest.importorskip("transformers")
    processor = transformers.SiglipImageProcessor(size={"height": 19, "width": 13}, do_rescale=False)
    frame = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16).expand(3, -1, -1)
    if kind == "binary":
        frame = (frame > 127).to(torch.uint8) * 255
    elif kind in ("black", "white"):
        frame = torch.full_like(frame, 0 if kind == "black" else 255)
    observation = make_streamvln_observation(frame, "go", uint8=True)
    assert observation.images.dtype == torch.uint8
    reference = make_streamvln_observation(frame, "go")
    expected = processor.preprocess(reference.images[0], return_tensors="pt")["pixel_values"][0]
    actual = StreamVLNImageTransform(processor)(observation.images[0], device=torch.device(device)).cpu()
    assert torch.equal(actual, expected)


def test_device_pixels_reject_invalid_frames_and_processor():
    transformers = pytest.importorskip("transformers")
    processor = transformers.SiglipImageProcessor(do_rescale=False)
    transform = StreamVLNImageTransform(processor)
    for value in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            transform(torch.full((3, 8, 8), value), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="CHW"):
        transform(torch.zeros(8, 8, 3), device=torch.device("cpu"))
    processor.do_rescale = True
    with pytest.raises(ValueError, match="slow bicubic"):
        StreamVLNImageTransform(processor)


class _ScriptedBackbone:
    def __init__(self, sequence):
        self.sequence = sequence
        self.llm = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=lambda ids: ids[..., None].float()),
            lm_head=nn.Linear(1, 8, bias=False),
        )
        self.widths = []

    def scores(self, position):
        scores = torch.full((1, 8), -3.0)
        scores[0, self.sequence[min(position, len(self.sequence) - 1)]] = 10.0
        return scores

    def project_logits(self, hidden):
        return hidden

    def forward_embeddings(self, inputs, cache, **kwargs):
        width = inputs.shape[1]
        self.widths.append(width)
        start = cache.seq_len
        cache.key_storage[..., start : start + width, :] = inputs
        states = torch.cat([self.scores(start + i + 1) for i in range(width)], dim=0)[None]
        return cache.advance(start + width), states

    def append_token(self, cache, token):
        advanced, states = self.forward_embeddings(token[..., None].float(), cache)
        return advanced, states[:, -1]


def _decoder(sequence, speculation, *, budget=16, capacity=32):
    backbone = _ScriptedBackbone(sequence)
    policy = SimpleNamespace(
        backbone=backbone,
        append_token=backbone.append_token,
        decode_block_size=4,
        max_new_tokens=budget,
        speculative_tokens=speculation,
        repetition_penalty=1.05,
        cuda_graph=False,
        fast_action_decode=False,
        action_token_ids={2, 3},
        eos_token_ids={7},
    )
    storage = torch.full((1, 1, 1, capacity, 1), -1.0)
    cache = StreamVLNCache(storage, storage.clone(), 0)
    prefix = StreamVLNPrefix(StreamVLNMemory(cache, (), 0, "walk"), backbone.scores(0))
    return StreamVLNDecoder(policy), prefix


@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("sequence", [(2, 2, 2, 2, 7), (1, 2, 2, 3, 3, 7), (7,), (2, 7), (2, 3, 2, 3, 7)])
@pytest.mark.parametrize("budget", [1, 3, 16])
def test_speculation_matches_serial_tokens_penalties_and_committed_kv(width, sequence, budget):
    reference, original = _decoder(sequence, 0, budget=budget)
    candidate, prefix = _decoder(sequence, width, budget=budget)
    expected = reference.generate_tokens(original)
    actual = candidate.generate_tokens(prefix)
    assert torch.equal(actual.token_ids, expected.token_ids)
    torch.testing.assert_close(actual.token_logprobs, expected.token_logprobs, rtol=0, atol=0)
    assert actual.stop_reason == expected.stop_reason
    assert actual.cache.seq_len == expected.cache.seq_len
    assert torch.equal(
        actual.cache.key_storage[..., : actual.cache.seq_len, :],
        expected.cache.key_storage[..., : expected.cache.seq_len, :],
    )
    assert prefix.memory.cache.seq_len == 0


def test_speculation_capacity_and_cancellation_do_not_commit_tails():
    decoder, prefix = _decoder((2, 2, 2, 7), 4, capacity=2)
    with pytest.raises(RuntimeError, match="capacity"):
        decoder.generate_tokens(prefix)
    assert prefix.memory.cache.seq_len == 0
    decoder, prefix = _decoder((2, 2, 2, 7), 4)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 3  # Cancel after the candidate forward, before accepting it.

    with pytest.raises(SessionCancelledError):
        decoder.generate_tokens(prefix, cancelled=cancelled)
    assert prefix.memory.cache.seq_len == 0
    assert decoder.speculation_stats["speculative_accepted_tokens"] == 0


def test_candidate_scoring_keeps_full_vocabulary_and_penalizes_once():
    logits = torch.tensor([[2.0, -2.0, 1.5, 1.9], [-1.0, -2.0, -3.0, -4.0]])
    seen = torch.tensor([True, True, False, False])
    tokens, scores = score_repeated_candidates(logits, seen, 2.0)
    expected = torch.tensor([[1.0, -4.0, 1.5, 1.9], [-2.0, -4.0, -3.0, -4.0]])
    assert tokens.tolist() == [3, 0]
    assert torch.equal(scores, expected.log_softmax(-1).gather(1, tokens[:, None])[:, 0])


@pytest.mark.gpu
def test_device_history_owns_vision_output_and_preserves_sparse_selection():
    source = torch.arange(24, dtype=torch.float32, device="cuda").reshape(6, 4)
    expected = source.clone()
    feature = StreamVLNPolicy._retain_history_feature(SimpleNamespace(gpu_history_features=True), source)
    assert feature.data_ptr() != source.data_ptr()
    source.zero_()  # Simulate the next vision graph replay.
    assert torch.equal(feature, expected)
    selected = StreamVLNPolicy._selected_history_features((feature, None, feature), (0, 2), source)
    assert torch.equal(selected, torch.stack((expected, expected)))
    memory = StreamVLNMemory(None, (), 3, "walk", frame_features=(feature, None))
    moved = memory.to("cpu", torch.float64)
    assert moved.frame_features[0].device.type == "cpu"
    assert moved.frame_features[0].dtype == torch.float64
    assert moved.frame_features[1] is None
    with pytest.raises(RuntimeError, match="discarded"):
        StreamVLNPolicy._selected_history_features((feature, None), (1,), source)


def test_speculative_prefix_failure_rolls_back_and_discards_penalty_state(monkeypatch):
    decoder, base = _decoder((2, 2, 7), 4)
    prefix = StreamVLNPrefix(
        StreamVLNMemory(base.memory.cache.advance(1), (), 0, "walk"),
        decoder.policy.backbone.scores(1),
        response_prefix_ids=torch.tensor([[1]]),
        response_prefix_hiddens=torch.stack((base.next_hidden, decoder.policy.backbone.scores(1)), dim=1),
        reference_cache_length=0,
    )

    def reject(prefix, workspace, *, cancelled):
        workspace.penalized.fill_(True)
        return None

    monkeypatch.setattr(decoder, "_verify_response_prefix", reject)
    actual = decoder.generate_tokens(prefix)
    reference, reference_prefix = _decoder((2, 2, 7), 0)
    expected = reference.generate_tokens(reference_prefix)
    assert torch.equal(actual.token_ids, expected.token_ids)
    assert torch.equal(actual.token_logprobs, expected.token_logprobs)
    assert actual.cache.seq_len == expected.cache.seq_len


def test_speculative_prefix_acceptance_preserves_token_budget(monkeypatch):
    decoder, base = _decoder((1, 2, 2, 7), 4, budget=3)
    cache = base.memory.cache.advance(1)
    cache.key_storage[..., 0, :] = 1
    first_score = base.next_hidden.log_softmax(-1)[:, 1:2]
    prefix = StreamVLNPrefix(
        StreamVLNMemory(cache, (), 0, "walk"),
        decoder.policy.backbone.scores(1),
        response_prefix_ids=torch.tensor([[1]]),
        reference_cache_length=0,
    )

    def accept(prefix, workspace, *, cancelled):
        workspace.penalized[1] = True
        return [torch.tensor([[1]])], [first_score]

    monkeypatch.setattr(decoder, "_verify_response_prefix", accept)
    actual = decoder.generate_tokens(prefix)
    reference, reference_prefix = _decoder((1, 2, 2, 7), 0, budget=3)
    expected = reference.generate_tokens(reference_prefix)
    assert torch.equal(actual.token_ids, expected.token_ids)
    assert torch.equal(actual.token_logprobs, expected.token_logprobs)
    assert actual.cache.seq_len == expected.cache.seq_len == 3
    assert actual.stop_reason == "max_tokens"
    assert decoder.speculation_stats["speculative_prefix_accepts"] == 1
