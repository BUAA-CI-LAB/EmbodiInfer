from __future__ import annotations

import gc
import os

import pytest
import torch

from embodiinfer import Observation
from embodiinfer.backend.triton import triton_capability
from embodiinfer.models.qwen25_vl.history_image_cache import (
    HistoryImageCache,
    HistoryImageCacheEntry,
)
from embodiinfer.policies.navida.modeling_navida import (
    NAVIDA_CHECKPOINT_REVISION,
    NAVIDA_SOURCE_REVISION,
    NaViDAMemory,
    NaViDARunner,
    _navida_generation_kwargs,
    _parse_navida_actions,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    QwenVLNMemory as LowLevelMemory,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    _parse_low_level_action,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    _Qwen25VLRunner as LowLevelRunner,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicMemory as PanoramicMemory,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicRunner as PanoramicRunner,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    parse_panoramic_action,
)


def _next_token_signature(runner, logits, profile, candidate_count):
    token_ids = logits.argmax(-1, keepdim=True)
    text = runner.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    if profile == "low_level":
        actions = _parse_low_level_action(text)
    else:
        actions = parse_panoramic_action(text, candidate_count)
    return token_ids.detach().cpu(), text, actions.tolist()


def _top50_overlap(left, right):
    left_ids = set(left.topk(50, dim=-1).indices.detach().cpu().reshape(-1).tolist())
    right_ids = set(right.topk(50, dim=-1).indices.detach().cpu().reshape(-1).tolist())
    return len(left_ids & right_ids)


INDUCTOR_TOP50_REQUIRED = 45
BUCKET_TOP50_REQUIRED = 45


class _GpuRunnerRegistry:
    def __init__(self):
        self._runners = []

    def __call__(self, runner):
        self._runners.append(runner)
        return runner

    def release(self, runner) -> None:
        self._runners.remove(runner)

    def clear(self) -> None:
        self._runners.clear()


@pytest.fixture
def gpu_runner_registry():
    registry = _GpuRunnerRegistry()
    try:
        yield registry
    finally:
        # The call phase has finished. Drop fixture-owned graph/model references
        # before flushing asynchronous CUDA and process-global compiler caches.
        registry.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compiler_reset = getattr(getattr(torch, "compiler", None), "reset", None)
        if callable(compiler_reset):
            compiler_reset()
        else:
            dynamo_reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
            if callable(dynamo_reset):
                dynamo_reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _real_navigation_observation(profile: str) -> Observation:
    generator = torch.Generator().manual_seed(20260820)
    if profile == "low_level":
        return Observation(
            torch.rand(1, 3, 240, 320, generator=generator),
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            instruction="walk to the door",
        )
    panorama = torch.rand(1, 3, 240, 960, generator=generator)
    candidates = torch.rand(2, 3, 240, 320, generator=generator)
    return Observation(
        torch.cat((panorama, torch.nn.functional.pad(candidates, (0, 640))), dim=0),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        instruction="walk to the door",
        metadata={
            "candidate_images": list(candidates),
            "candidates": [
                {"relative_angle": -30, "distance": 1.0},
                {"relative_angle": 30, "distance": 2.0},
            ],
        },
    )


@pytest.mark.parametrize(
    ("profile", "environment"),
    [
        ("low_level", "EMBODIINFER_QWEN_R2R_LOW_CHECKPOINT"),
        ("panoramic", "EMBODIINFER_QWEN_R2R_PANORAMIC_CHECKPOINT"),
    ],
)
def test_real_history_cache_processor_parity(profile, environment, gpu_runner_registry):
    checkpoint = os.environ.get(environment)
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip(f"requires CUDA and {environment}")
    observation = _real_navigation_observation(profile)
    runner_class = LowLevelRunner if profile == "low_level" else PanoramicRunner
    memory_class = LowLevelMemory if profile == "low_level" else PanoramicMemory
    runner = gpu_runner_registry(
        runner_class(
            checkpoint,
            profile,
            max_new_tokens=1,
            execute_chunks=1,
            attention_backend="torch_sdpa",
            compile_backend="none",
            history_image_cache="rgb_bytes",
        )
    )
    runner.model.to(device="cuda:0", dtype=torch.bfloat16).eval()
    assert runner.history_image_cache_mode == "rgb_bytes"
    assert runner.new_history_image_cache().enabled_for_session is True
    history_frame = torch.linspace(
        0.0,
        1.0,
        observation.images[0].numel(),
        dtype=torch.float32,
    ).reshape_as(observation.images[0])
    assert not torch.equal(history_frame, observation.images[0])
    history_kind = "low" if profile == "low_level" else "panorama"
    history_entry = HistoryImageCacheEntry.from_pil(
        runner._render_image(history_frame, history_kind),
        runner.history_image_cache_key,
    )
    response = "Move" if profile == "low_level" else "0"
    cached_memory = memory_class(
        frames=(history_frame,),
        responses=(response,),
        history_image_cache=HistoryImageCache.enabled().append_for_frame(
            frame_index=0,
            entry=history_entry,
        ),
    )
    uncached_memory = memory_class(
        frames=(history_frame,),
        responses=(response,),
        history_image_cache=HistoryImageCache.enabled(),
    )
    assert uncached_memory.history_image_cache.get(0, runner.history_image_cache_key) is None
    assert cached_memory.history_image_cache.get(0, runner.history_image_cache_key) is history_entry

    original_render = runner._render_image
    render_counts = {"low": 0, "panorama": 0, "candidate": 0}

    def counted_render(frame, kind):
        render_counts[kind] += 1
        return original_render(frame, kind)

    runner._render_image = counted_render
    uncached_processor = runner._prepare_batch([observation], [uncached_memory])
    for name in render_counts:
        render_counts[name] = 0
    cached_processor, current_entries = runner._prepare_batch_with_history_entries(
        [observation], [cached_memory]
    )
    runner._render_image = original_render
    assert set(cached_processor) == set(uncached_processor)
    for name in cached_processor:
        torch.testing.assert_close(
            cached_processor[name],
            uncached_processor[name],
            rtol=0,
            atol=0,
        )
    assert current_entries[0] is not None
    assert render_counts[history_kind] == 1
    if profile == "panoramic":
        assert render_counts["candidate"] == len(observation.metadata["candidate_images"])
        assert current_entries[0].key.kind == "panorama"
    else:
        assert render_counts["candidate"] == 0


@pytest.mark.parametrize("attention_backend", ["torch_sdpa", "triton"])
@pytest.mark.parametrize("compile_backend", ["none", "inductor"])
@pytest.mark.parametrize(
    ("profile", "environment"),
    [
        ("low_level", "EMBODIINFER_QWEN_R2R_LOW_CHECKPOINT"),
        ("panoramic", "EMBODIINFER_QWEN_R2R_PANORAMIC_CHECKPOINT"),
    ],
)
def test_native_next_token_forward_matches_huggingface(
    profile, environment, attention_backend, compile_backend, gpu_runner_registry
):
    checkpoint = os.environ.get(environment)
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip(f"requires CUDA and {environment}")
    if attention_backend == "triton":
        capability = triton_capability(torch.device("cuda:0"))
        if not capability.available:
            pytest.skip(f"requires Triton attention capability: {capability.reason}")
    if compile_backend == "inductor" and attention_backend != "torch_sdpa":
        pytest.skip("torch.compile is intentionally restricted to torch_sdpa")
    observation = _real_navigation_observation(profile)
    runner_class = LowLevelRunner if profile == "low_level" else PanoramicRunner
    memory_class = LowLevelMemory if profile == "low_level" else PanoramicMemory
    runner = gpu_runner_registry(
        runner_class(
            checkpoint,
            profile,
            max_new_tokens=1,
            execute_chunks=1,
            attention_backend=attention_backend,
            compile_backend=compile_backend,
            history_image_cache="none",
        )
    )
    runner.model.to(device="cuda:0", dtype=torch.bfloat16).eval()
    encoded = runner._encode_batch([observation], [memory_class()])
    with torch.inference_mode():
        reference = runner.model(**encoded, use_cache=False).logits[:, -1].float()
        raw_native = None
        if compile_backend == "inductor":
            raw_native = (
                runner.graph_runtime.forward(*runner.graph_runtime.native_inputs(encoded))
                if hasattr(runner, "graph_runtime")
                else runner.graph_decoder(*runner._native_inputs(encoded))
            ).float()
        native = runner._graph_logits(encoded).float()
        graphed = runner._manual_graph_logits(encoded).float()
    if raw_native is not None:
        torch.testing.assert_close(raw_native, reference, rtol=0.02, atol=0.05)
        assert raw_native.argmax(-1).tolist() == reference.argmax(-1).tolist()
        # BF16 Inductor changes reduction grouping. Keep exact semantic gates
        # and use a principled 90% top-50 distribution guard.
        torch.testing.assert_close(raw_native, native, rtol=0.06, atol=0.30)
        assert _top50_overlap(raw_native, native) >= INDUCTOR_TOP50_REQUIRED
    assert native.argmax(-1).tolist() == reference.argmax(-1).tolist()
    if attention_backend == "triton":
        # Reduction differences accumulate across all 32 vision layers in the
        # hybrid plan; 48/50 preserves a 96% boundary-set gate while exact
        # token/action gates remain unchanged below.
        torch.testing.assert_close(native, reference, rtol=0.06, atol=0.30)
        assert _top50_overlap(native, reference) >= 48
    elif compile_backend == "inductor":
        torch.testing.assert_close(native, reference, rtol=0.06, atol=0.30)
        assert _top50_overlap(native, reference) >= INDUCTOR_TOP50_REQUIRED
    else:
        torch.testing.assert_close(native, reference, rtol=0.02, atol=0.05)
    assert graphed.argmax(-1).tolist() == reference.argmax(-1).tolist()
    if attention_backend == "triton":
        torch.testing.assert_close(graphed, native, rtol=0.02, atol=0.05)
    elif compile_backend == "inductor":
        torch.testing.assert_close(graphed, reference, rtol=0.06, atol=0.30)
        assert _top50_overlap(graphed, reference) >= INDUCTOR_TOP50_REQUIRED
        if raw_native is not None:
            torch.testing.assert_close(graphed, raw_native, rtol=0.06, atol=0.30)
            assert _top50_overlap(graphed, raw_native) >= INDUCTOR_TOP50_REQUIRED
        torch.testing.assert_close(graphed, native, rtol=0.02, atol=0.05)
    else:
        torch.testing.assert_close(graphed, reference, rtol=0.02, atol=0.05)
    candidate_count = len(observation.metadata.get("candidates", ()))
    reference_signature = _next_token_signature(runner, reference, profile, candidate_count)
    assert _next_token_signature(runner, native, profile, candidate_count) == reference_signature
    assert _next_token_signature(runner, graphed, profile, candidate_count) == reference_signature
    if raw_native is not None:
        assert _next_token_signature(runner, raw_native, profile, candidate_count) == reference_signature

    changed = observation.images.clone()
    changed.mul_(0.5)
    changed_observation = Observation(
        changed,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        instruction=observation.instruction,
        metadata=observation.metadata,
    )
    changed_encoded = runner._encode_batch([changed_observation], [memory_class()])
    with torch.inference_mode():
        changed_reference = runner.model(**changed_encoded, use_cache=False).logits[:, -1].float()
        changed_raw_native = None
        if compile_backend == "inductor":
            changed_raw_native = (
                runner.graph_runtime.forward(*runner.graph_runtime.native_inputs(changed_encoded))
                if hasattr(runner, "graph_runtime")
                else runner.graph_decoder(*runner._native_inputs(changed_encoded))
            ).float()
        changed_native = runner._graph_logits(changed_encoded).float()
        changed_graphed = runner._manual_graph_logits(changed_encoded).float()
    if changed_raw_native is not None:
        torch.testing.assert_close(changed_raw_native, changed_reference, rtol=0.02, atol=0.05)
        assert changed_raw_native.argmax(-1).tolist() == changed_reference.argmax(-1).tolist()
        torch.testing.assert_close(changed_raw_native, changed_native, rtol=0.06, atol=0.30)
        assert _top50_overlap(changed_raw_native, changed_native) >= INDUCTOR_TOP50_REQUIRED
    assert changed_native.argmax(-1).tolist() == changed_reference.argmax(-1).tolist()
    if attention_backend == "triton":
        torch.testing.assert_close(changed_native, changed_reference, rtol=0.06, atol=0.30)
        assert _top50_overlap(changed_native, changed_reference) >= 48
    elif compile_backend == "inductor":
        torch.testing.assert_close(changed_native, changed_reference, rtol=0.06, atol=0.30)
        assert _top50_overlap(changed_native, changed_reference) >= INDUCTOR_TOP50_REQUIRED
    else:
        torch.testing.assert_close(changed_native, changed_reference, rtol=0.02, atol=0.05)
    assert changed_graphed.argmax(-1).tolist() == changed_reference.argmax(-1).tolist()
    if attention_backend == "triton":
        torch.testing.assert_close(changed_graphed, changed_native, rtol=0.02, atol=0.05)
    elif compile_backend == "inductor":
        torch.testing.assert_close(changed_graphed, changed_reference, rtol=0.06, atol=0.30)
        assert _top50_overlap(changed_graphed, changed_reference) >= INDUCTOR_TOP50_REQUIRED
        if changed_raw_native is not None:
            torch.testing.assert_close(changed_graphed, changed_raw_native, rtol=0.06, atol=0.30)
            assert _top50_overlap(changed_graphed, changed_raw_native) >= INDUCTOR_TOP50_REQUIRED
        torch.testing.assert_close(changed_graphed, changed_native, rtol=0.02, atol=0.05)
    else:
        torch.testing.assert_close(changed_graphed, changed_reference, rtol=0.02, atol=0.05)
    changed_reference_signature = _next_token_signature(runner, changed_reference, profile, candidate_count)
    assert (
        _next_token_signature(runner, changed_native, profile, candidate_count) == changed_reference_signature
    )
    assert (
        _next_token_signature(runner, changed_graphed, profile, candidate_count)
        == changed_reference_signature
    )
    if changed_raw_native is not None:
        assert (
            _next_token_signature(runner, changed_raw_native, profile, candidate_count)
            == changed_reference_signature
        )
    assert runner.manual_graph_stats()["capture_count"] == 1
    assert runner.manual_graph_stats()["replay_count"] == 2
    backend_stats = runner.manual_graph_stats()["attention_backend"]
    assert backend_stats["requested"] == attention_backend
    expected_resolved = "torch_sdpa" if attention_backend == "torch_sdpa" else "triton_hybrid"
    assert backend_stats["resolved"] == expected_resolved
    assert backend_stats["fallback_reason"] is None
    if attention_backend == "triton":
        assert backend_stats["config"]["window_attention"] == "triton_segmented"
        assert backend_stats["config"]["full_attention"] == "torch_sdpa"
        assert backend_stats["config"]["rope"] == "triton"
    compile_stats = runner.manual_graph_stats()["torch_compile"]
    assert compile_stats["requested"] == compile_backend
    assert compile_stats["resolved"] == ("inductor" if compile_backend == "inductor" else "eager")
    assert compile_stats["fallback_reason"] is None
    assert compile_stats["failures"] == 0
    if compile_backend == "inductor":
        assert compile_stats["backend"] == "inductor"
        assert compile_stats["fullgraph"] is True
        assert compile_stats["dynamic"] is False
        assert compile_stats["inductor_cudagraphs"] is False
        assert compile_stats["attempts"] >= 1
        assert compile_stats["cache_entries"] >= 1
        assert all(entry["first_call_wall_ms"] > 0.0 for entry in compile_stats["entries"])


def test_navida_official_oracle_and_full_policy_cuda_graph(
    gpu_runner_registry,
):
    checkpoint = os.environ.get("EMBODIINFER_NAVIDA_CHECKPOINT")
    reference_path = os.environ.get("EMBODIINFER_NAVIDA_REFERENCE")
    if checkpoint is None or reference_path is None or not torch.cuda.is_available():
        pytest.skip("requires CUDA, EMBODIINFER_NAVIDA_CHECKPOINT, and EMBODIINFER_NAVIDA_REFERENCE")
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    assert reference["source_revision"] == NAVIDA_SOURCE_REVISION
    assert reference["checkpoint_revision"] == NAVIDA_CHECKPOINT_REVISION
    source = reference["source_image"]
    observation = Observation(
        source.permute(2, 0, 1).float().div(255).unsqueeze(0),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        instruction=reference["instruction"],
    )
    runner = gpu_runner_registry(
        NaViDARunner(
            checkpoint,
            max_new_tokens=512,
            execute_chunks=2,
        )
    )
    runner.model.to(device="cuda:0", dtype=torch.bfloat16).eval()
    prepared = runner._prepare_batch([observation], [NaViDAMemory()])
    for key, expected in reference["processor"].items():
        assert torch.equal(prepared[key], expected), key
    encoded = runner._encode_batch([observation], [NaViDAMemory()])

    with torch.inference_mode():
        first_logits = runner.model(**encoded, use_cache=True).logits[:, -1].float().cpu()
    assert first_logits.argmax(-1).tolist() == reference["first_logits"].argmax(-1).tolist()
    torch.testing.assert_close(first_logits, reference["first_logits"], rtol=0.02, atol=0.05)

    seed = int(reference["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        eager = runner.model.generate(
            **encoded,
            **_navida_generation_kwargs(512),
            use_model_defaults=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    eager_tokens = eager.sequences[:, encoded["input_ids"].shape[1] :]
    assert torch.equal(eager_tokens.cpu(), reference["generated_token_ids"])
    eager_text = runner.processor.batch_decode(eager_tokens, skip_special_tokens=True)[0].strip()
    assert eager_text == reference["decoded_text"]
    assert _parse_navida_actions(eager_text).tolist() == reference["actions"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runner.configure_cuda_graph(True)
    graph_tokens, graph_logits = runner._navida_graph_generate(encoded, return_scores=True)
    assert torch.equal(graph_tokens, eager_tokens)
    assert torch.equal(graph_tokens.cpu(), reference["generated_token_ids"])
    assert len(graph_logits) == len(eager.logits)
    for actual, expected in zip(graph_logits, eager.logits, strict=True):
        assert actual.argmax(-1).tolist() == expected.argmax(-1).tolist()
        expected = expected.float()
        actual_top = actual.topk(50, dim=-1).indices
        expected_top = expected.topk(50, dim=-1).indices
        overlap = torch.isin(actual_top, expected_top).sum(dim=-1)
        assert int(overlap.min().item()) >= 49
        union = torch.unique(torch.cat((actual_top[0], expected_top[0])))
        torch.testing.assert_close(actual[:, union], expected[:, union], rtol=0.06, atol=0.30)
    reference_scores = reference["generation_scores"]
    assert len(eager.scores) == reference_scores.shape[0]
    for actual, expected in zip(eager.scores, reference_scores, strict=True):
        actual = actual.float().cpu()
        common = torch.isfinite(actual) & torch.isfinite(expected)
        assert int(common.sum().item()) >= 49
        assert actual.argmax(-1).tolist() == expected.argmax(-1).tolist()
        torch.testing.assert_close(actual[common], expected[common], rtol=0.02, atol=0.05)
    graph_text = runner.processor.batch_decode(graph_tokens, skip_special_tokens=True)[0].strip()
    assert graph_text == reference["decoded_text"]
    assert _parse_navida_actions(graph_text).tolist() == reference["actions"]
    stats = runner.manual_graph_stats()["navida_decode"]
    assert stats["capture_count"] == 1
    assert stats["replay_count"] == max(0, graph_tokens.shape[1] - 1)

    changed = 255 - source
    changed_observation = Observation(
        changed.permute(2, 0, 1).float().div(255).unsqueeze(0),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        instruction=reference["instruction"],
    )
    changed_encoded = runner._encode_batch([changed_observation], [NaViDAMemory()])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _, changed_logits = runner._navida_graph_generate(changed_encoded, return_scores=True)
    assert not torch.equal(changed_logits[0], graph_logits[0])
    assert runner.manual_graph_stats()["navida_decode"]["capture_count"] == 1

    history = tuple(
        torch.roll(source, shifts=index + 1, dims=1).permute(2, 0, 1).float().div(255) for index in range(8)
    )
    history_memory = NaViDAMemory(frames=history)
    history_encoded = runner._encode_batch([observation], [history_memory])
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    with torch.inference_mode():
        history_eager = runner.model.generate(
            **history_encoded,
            **_navida_generation_kwargs(512),
            use_model_defaults=True,
        )
    history_eager_tokens = history_eager[:, history_encoded["input_ids"].shape[1] :]
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    history_graph_tokens, _ = runner._navida_graph_generate(history_encoded)
    assert torch.equal(history_graph_tokens, history_eager_tokens)
    history_text = runner.processor.batch_decode(history_graph_tokens, skip_special_tokens=True)[0].strip()
    assert _parse_navida_actions(history_text).shape[0] <= 6
    assert runner.manual_graph_stats()["navida_decode"]["capture_count"] == 2


@pytest.mark.parametrize(
    ("profile", "environment"),
    [
        ("low_level", "EMBODIINFER_QWEN_R2R_LOW_CHECKPOINT"),
        ("panoramic", "EMBODIINFER_QWEN_R2R_PANORAMIC_CHECKPOINT"),
    ],
)
def test_inductor_text_bucket_matches_exact_text_shape(profile, environment, gpu_runner_registry):
    checkpoint = os.environ.get(environment)
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip(f"requires CUDA and {environment}")
    generator = torch.Generator().manual_seed(20260820)
    if profile == "low_level":
        observation = Observation(
            torch.rand(1, 3, 240, 320, generator=generator),
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            instruction="walk to the door",
        )
    else:
        panorama = torch.rand(1, 3, 240, 960, generator=generator)
        candidates = torch.rand(2, 3, 240, 320, generator=generator)
        observation = Observation(
            torch.cat((panorama, torch.nn.functional.pad(candidates, (0, 640))), dim=0),
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            instruction="walk to the door",
            metadata={
                "candidate_images": list(candidates),
                "candidates": [
                    {"relative_angle": -30, "distance": 1.0},
                    {"relative_angle": 30, "distance": 2.0},
                ],
            },
        )
    runner_class = LowLevelRunner if profile == "low_level" else PanoramicRunner
    memory_class = LowLevelMemory if profile == "low_level" else PanoramicMemory

    def make_runner(text_buckets):
        return gpu_runner_registry(
            runner_class(
                checkpoint,
                profile,
                max_new_tokens=1,
                execute_chunks=1,
                attention_backend="torch_sdpa",
                compile_backend="inductor",
                compile_text_buckets=text_buckets,
            )
        )

    def runtime_for(runner):
        return runner.graph_runtime if profile == "low_level" else runner

    def native_inputs(runner, encoded):
        runtime = runtime_for(runner)
        return runtime.native_inputs(encoded) if profile == "low_level" else runtime._native_inputs(encoded)

    def raw_logits(runner, encoded):
        runtime = runtime_for(runner)
        inputs = native_inputs(runner, encoded)
        forward = runtime.forward if profile == "low_level" else runtime.graph_decoder
        return forward(*inputs)

    candidate_count = len(observation.metadata.get("candidates", ()))
    unbucketed_runner = make_runner(())
    unbucketed_runner.model.to(device="cuda:0", dtype=torch.bfloat16).eval()
    unbucketed = unbucketed_runner._encode_batch([observation], [memory_class()])
    source_length = int(unbucketed["input_ids"].shape[1])
    target_length = ((source_length + 63) // 64) * 64
    if target_length == source_length:
        target_length += 64
    unbucketed_position_ids = native_inputs(unbucketed_runner, unbucketed)[3].detach().cpu().clone()
    assert unbucketed_position_ids.shape[0] == 3
    unbucketed_encoded = {
        name: unbucketed[name].detach().cpu().clone()
        for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }
    with torch.inference_mode():
        hf_unbucketed = unbucketed_runner.model(**unbucketed, use_cache=False).logits[:, -1].float().cpu()
        native_unbucketed = raw_logits(unbucketed_runner, unbucketed).float().cpu()
        compiled_unbucketed = unbucketed_runner._graph_logits(unbucketed).float().cpu()
        captured_unbucketed = unbucketed_runner._manual_graph_logits(unbucketed).float().cpu()
    unbucketed_signature = _next_token_signature(unbucketed_runner, hf_unbucketed, profile, candidate_count)
    unbucketed_stats = unbucketed_runner.manual_graph_stats()["torch_compile"]
    assert unbucketed_stats["bucket_ids"] == [["qwen25_vl_exact_text_shape_v1", source_length]]
    assert unbucketed_stats["failures"] == 0
    torch.testing.assert_close(native_unbucketed, hf_unbucketed, rtol=0.02, atol=0.05)
    assert native_unbucketed.argmax(-1).tolist() == hf_unbucketed.argmax(-1).tolist()
    assert (
        _next_token_signature(
            unbucketed_runner,
            native_unbucketed,
            profile,
            candidate_count,
        )
        == unbucketed_signature
    )
    for actual in (compiled_unbucketed, captured_unbucketed):
        assert actual.argmax(-1).tolist() == hf_unbucketed.argmax(-1).tolist()
        torch.testing.assert_close(actual, hf_unbucketed, rtol=0.06, atol=0.30)
        assert _top50_overlap(actual, hf_unbucketed) >= INDUCTOR_TOP50_REQUIRED
        assert (
            _next_token_signature(unbucketed_runner, actual, profile, candidate_count) == unbucketed_signature
        )
    torch.testing.assert_close(compiled_unbucketed, captured_unbucketed, rtol=0.02, atol=0.05)
    gpu_runner_registry.release(unbucketed_runner)
    del unbucketed, unbucketed_runner
    gc.collect()
    torch.cuda.empty_cache()

    runner = make_runner((target_length,))
    runner.model.to(device="cuda:0", dtype=torch.bfloat16).eval()
    bucketed = runner._encode_batch([observation], [memory_class()])
    bucket_runtime = runtime_for(runner)
    assert bucket_runtime.compile_text_buckets == (target_length,)
    assert bucket_runtime.use_text_attention_mask is True
    bucketed_position_ids = native_inputs(runner, bucketed)[3].detach().cpu()
    assert bucketed_position_ids.shape[0] == 3
    assert torch.equal(bucketed_position_ids[..., -source_length:], unbucketed_position_ids)
    assert bucketed["input_ids"].shape[1] == target_length
    assert torch.equal(
        bucketed["input_ids"][:, -source_length:].cpu(),
        unbucketed_encoded["input_ids"],
    )
    assert not bool(bucketed["attention_mask"][:, :-source_length].any().item())
    assert torch.equal(
        bucketed["attention_mask"][:, -source_length:].cpu(),
        unbucketed_encoded["attention_mask"],
    )
    for name in ("pixel_values", "image_grid_thw"):
        assert torch.equal(bucketed[name].cpu(), unbucketed_encoded[name])

    with torch.inference_mode():
        hf_bucketed = runner.model(**bucketed, use_cache=False).logits[:, -1].float().cpu()
        native_bucketed = raw_logits(runner, bucketed).float().cpu()
        compiled_bucketed = runner._graph_logits(bucketed).float().cpu()
        captured_bucketed = runner._manual_graph_logits(bucketed).float().cpu()
    torch.testing.assert_close(native_bucketed, hf_bucketed, rtol=0.02, atol=0.05)
    assert native_bucketed.argmax(-1).tolist() == hf_bucketed.argmax(-1).tolist()
    torch.testing.assert_close(compiled_bucketed, native_bucketed, rtol=0.06, atol=0.30)
    assert _top50_overlap(compiled_bucketed, native_bucketed) >= INDUCTOR_TOP50_REQUIRED
    assert compiled_bucketed.argmax(-1).tolist() == native_bucketed.argmax(-1).tolist()
    torch.testing.assert_close(compiled_bucketed, captured_bucketed, rtol=0.02, atol=0.05)
    assert compiled_bucketed.argmax(-1).tolist() == captured_bucketed.argmax(-1).tolist()
    for exact, bucketed_actual in (
        (hf_unbucketed, hf_bucketed),
        (native_unbucketed, native_bucketed),
        (compiled_unbucketed, compiled_bucketed),
        (captured_unbucketed, captured_bucketed),
    ):
        assert bucketed_actual.argmax(-1).tolist() == exact.argmax(-1).tolist()
        torch.testing.assert_close(bucketed_actual, exact, rtol=0.06, atol=0.30)
        assert _top50_overlap(bucketed_actual, exact) >= BUCKET_TOP50_REQUIRED
    for actual in (
        hf_bucketed,
        native_bucketed,
        compiled_bucketed,
        captured_bucketed,
    ):
        assert _next_token_signature(runner, actual, profile, candidate_count) == unbucketed_signature
    stats = runner.manual_graph_stats()["torch_compile"]
    assert stats["bucket_ids"] == [["qwen25_vl_left_masked_text_v3", target_length]]
    assert stats["persistent_cache"]["configured"] is False
    assert stats["failures"] == 0
