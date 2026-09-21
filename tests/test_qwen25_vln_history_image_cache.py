from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from PIL import Image

from embodiinfer.engine.session import SessionStore
from embodiinfer.exceptions import StaleSessionError
from embodiinfer.models.qwen25_vl.history_image_cache import (
    QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES,
    HistoryImageCache,
    HistoryImageCacheEntry,
    HistoryImageCacheKey,
    normalize_history_image_cache_mode,
    prepare_history_images,
)
from embodiinfer.policies.qwen_r2r_low import policy as low_policy_module
from embodiinfer.policies.qwen_r2r_low import runner as low_runner_module
from embodiinfer.policies.qwen_r2r_low.contract import (
    LOW_LEVEL_IMAGE_SIZE,
    QwenR2RLowMemory,
    parse_low_action,
)
from embodiinfer.policies.qwen_r2r_low.contract import (
    R2R_PREPROCESSOR_SHA256 as LOW_PROCESSOR_SHA256,
)
from embodiinfer.policies.qwen_r2r_low.processing import tensor_to_pil
from embodiinfer.policies.qwen_r2r_low.runner import QwenR2RLowRunner
from embodiinfer.policies.qwen_r2r_panoramic import policy as panoramic_policy_module
from embodiinfer.policies.qwen_r2r_panoramic import runner as panoramic_runner_module
from embodiinfer.policies.qwen_r2r_panoramic.contract import (
    CANDIDATE_IMAGE_SIZE,
    PANORAMIC_IMAGE_SIZE,
    QwenR2RPanoramicMemory,
    parse_panoramic_action,
)
from embodiinfer.policies.qwen_r2r_panoramic.contract import (
    R2R_PREPROCESSOR_SHA256 as PANORAMIC_PROCESSOR_SHA256,
)
from embodiinfer.policies.qwen_r2r_panoramic.processing import _pil
from embodiinfer.policies.qwen_r2r_panoramic.runner import QwenR2RPanoramicRunner
from embodiinfer.types import SessionKey


def _key(profile: str = "low") -> HistoryImageCacheKey:
    if profile == "low":
        return HistoryImageCacheKey(
            profile="low_level",
            kind="low",
            size=LOW_LEVEL_IMAGE_SIZE,
            resize=True,
            processor_sha256=LOW_PROCESSOR_SHA256,
        )
    return HistoryImageCacheKey(
        profile="panoramic",
        kind="panorama",
        size=PANORAMIC_IMAGE_SIZE,
        resize=False,
        processor_sha256=PANORAMIC_PROCESSOR_SHA256,
    )


def _entry(key: HistoryImageCacheKey, value: int) -> HistoryImageCacheEntry:
    image = Image.new("RGB", key.size, color=(value, value + 1, value + 2))
    return HistoryImageCacheEntry.from_pil(image, key)


def _frame(height: int, width: int, value: float) -> torch.Tensor:
    return torch.full((3, height, width), value, dtype=torch.float32)


def _processor_encoding(
    images: list[Image.Image],
) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
    encoded = []
    for image in images:
        tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).contiguous()
        encoded.append(
            (
                str(tensor.dtype),
                tuple(tensor.shape),
                tensor.numpy().tobytes(),
            )
        )
    return tuple(encoded)


def _session_key(name: str) -> SessionKey:
    return SessionKey(env_id=name, episode_id=1, rollout_id=0)


def test_post_resize_rgb_bytes_are_exact_and_immutable() -> None:
    key = _key()
    image = tensor_to_pil(_frame(120, 160, 0.25), LOW_LEVEL_IMAGE_SIZE, resize=True)
    entry = HistoryImageCacheEntry.from_pil(image, key)
    assert entry.rgb_bytes == image.tobytes()
    assert len(entry.rgb_bytes) == 320 * 240 * 3
    assert entry.to_pil().tobytes() == image.tobytes()
    with pytest.raises(FrozenInstanceError):
        entry.rgb_bytes = b""


def test_formal_keys_reject_candidates_and_malformed_processor_hash() -> None:
    with pytest.raises(ValueError, match="candidate"):
        HistoryImageCacheKey(
            profile="panoramic",
            kind="candidate",
            size=CANDIDATE_IMAGE_SIZE,
            resize=False,
            processor_sha256=PANORAMIC_PROCESSOR_SHA256,
        )
    with pytest.raises(ValueError, match="64 lowercase"):
        HistoryImageCacheKey(
            profile="low_level",
            kind="low",
            size=LOW_LEVEL_IMAGE_SIZE,
            resize=True,
            processor_sha256="not-a-checksum",
        )


def test_recent_ring_is_bounded_and_uses_absolute_frame_indices() -> None:
    key = _key()
    one = _entry(key, 1)
    cache = HistoryImageCache.enabled(limit_bytes=2 * one.nbytes)
    for frame_index in range(5):
        cache = cache.append_for_frame(
            frame_index=frame_index,
            entry=_entry(key, frame_index + 1),
        )
    assert cache.base_frame_index == 3
    assert len(cache.entries) == 2
    assert cache.bytes_used == 2 * one.nbytes
    assert cache.get(2, key) is None
    assert cache.get(3, key) is not None
    assert cache.get(4, key) is not None
    assert cache.as_dict()["end_frame_index"] == 5


def test_16_mib_bounds_match_low_and_panoramic_payloads() -> None:
    for profile, expected in (("low", 72), ("panoramic", 24)):
        key = _key(profile)
        entry = _entry(key, 1)
        cache = HistoryImageCache.enabled()
        for frame_index in range(expected + 3):
            cache = cache.append_for_frame(frame_index=frame_index, entry=entry)
        assert len(cache.entries) == expected
        assert cache.bytes_used <= QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES
        assert cache.base_frame_index == 3


@pytest.mark.parametrize(
    "memory_type,profile,height,width",
    [
        (QwenR2RLowMemory, "low", 240, 320),
        (QwenR2RPanoramicMemory, "panoramic", 240, 960),
    ],
)
def test_memory_b1_expand_compact_to_and_suffix_validation(memory_type, profile, height, width) -> None:
    key = _key(profile)
    cache = HistoryImageCache.enabled().append_for_frame(frame_index=0, entry=_entry(key, 1))
    memory = memory_type(
        frames=(_frame(height, width, 0.1),),
        responses=("Move",),
        history_image_cache=cache,
    )
    assert memory.expand(1) is memory
    assert memory.compact() is memory
    assert memory.compact([0]) is memory
    assert memory.compact(torch.tensor([0])) is memory
    assert memory.to("cpu") is memory
    with pytest.raises(ValueError, match="B1"):
        memory.expand(2)
    with pytest.raises(ValueError, match=r"compact\(\[0\]\)"):
        memory.compact([])
    with pytest.raises(ValueError, match="contiguous suffix"):
        memory_type(
            frames=(
                _frame(height, width, 0.1),
                _frame(height, width, 0.2),
            ),
            responses=("Move", "Move"),
            history_image_cache=cache,
        )
    cold = memory_type(
        frames=(
            _frame(height, width, 0.1),
            _frame(height, width, 0.2),
        ),
        responses=("Move", "Move"),
    )
    assert cold.history_image_cache.entries == ()


def test_low_cached_and_uncached_processor_tensors_are_exact() -> None:
    key = _key("low")
    history = _frame(240, 320, 0.2)
    current = _frame(240, 320, 0.7)
    cache = HistoryImageCache.enabled().append_for_frame(
        frame_index=0,
        entry=HistoryImageCacheEntry.from_pil(tensor_to_pil(history, LOW_LEVEL_IMAGE_SIZE, resize=True), key),
    )

    def render(frame, kind):
        assert kind == "low"
        return tensor_to_pil(frame, LOW_LEVEL_IMAGE_SIZE, resize=True)

    uncached, no_entry, uncached_stats = prepare_history_images(
        (history, current),
        ("low", "low"),
        history_count=1,
        cache=HistoryImageCache(),
        key=key,
        cache_enabled=False,
        render=render,
    )
    cached, current_entry, cached_stats = prepare_history_images(
        (history, current),
        ("low", "low"),
        history_count=1,
        cache=cache,
        key=key,
        cache_enabled=True,
        render=render,
    )
    assert _processor_encoding(cached) == _processor_encoding(uncached)
    assert no_entry is None
    assert current_entry is not None
    assert uncached_stats.history_misses == 0
    assert uncached_stats.history_renders == 1
    assert cached_stats.history_hits == 1
    assert cached_stats.history_renders == 0
    assert cached_stats.current_renders == 1


def test_panoramic_cached_and_uncached_tensors_and_render_counts_are_exact() -> None:
    key = _key("panoramic")
    history = _frame(240, 960, 0.1)
    current = _frame(240, 960, 0.4)
    candidates = (
        _frame(240, 320, 0.6),
        _frame(240, 320, 0.8),
    )
    cache = HistoryImageCache.enabled().append_for_frame(
        frame_index=0,
        entry=HistoryImageCacheEntry.from_pil(_pil(history, PANORAMIC_IMAGE_SIZE, resize=False), key),
    )
    calls: list[str] = []

    def render(frame, kind):
        calls.append(kind)
        size = PANORAMIC_IMAGE_SIZE if kind == "panorama" else CANDIDATE_IMAGE_SIZE
        return _pil(frame, size, resize=False)

    frames = (history, current, *candidates)
    kinds = ("panorama", "panorama", "candidate", "candidate")
    uncached, no_entry, _ = prepare_history_images(
        frames,
        kinds,
        history_count=1,
        cache=HistoryImageCache(),
        key=key,
        cache_enabled=False,
        render=render,
    )
    calls.clear()
    cached, current_entry, stats = prepare_history_images(
        frames,
        kinds,
        history_count=1,
        cache=cache,
        key=key,
        cache_enabled=True,
        render=render,
    )
    assert _processor_encoding(cached) == _processor_encoding(uncached)
    assert no_entry is None
    assert calls.count("panorama") == 1
    assert calls.count("candidate") == 2
    assert current_entry is not None and current_entry.key.kind == "panorama"
    assert stats.history_hits == 1
    assert stats.history_renders == 0
    assert stats.current_renders == 1
    assert stats.candidate_renders == 2


@pytest.mark.parametrize(
    "runner_type,profile",
    [
        (QwenR2RLowRunner, "low"),
        (QwenR2RPanoramicRunner, "panoramic"),
    ],
)
def test_legacy_runner_api_stays_dict_and_three_tuple(runner_type, profile) -> None:
    runner = object.__new__(runner_type)
    encoded = {"input_ids": torch.ones((1, 1), dtype=torch.long)}
    entry = _entry(_key(profile), 5)
    capture_flags: list[bool] = []

    def prepare_impl(_observations, _memories, *, capture_current_entries):
        capture_flags.append(capture_current_entries)
        return encoded, [entry if capture_current_entries else None]

    runner._prepare_batch_impl = prepare_impl
    assert runner._prepare_batch([object()], [object()]) is encoded
    aware_encoded, entries = runner._prepare_batch_with_history_entries([object()], [object()])
    assert aware_encoded is encoded
    assert entries == [entry]
    assert capture_flags == [False, True]

    runner._encode_batch = lambda _observations, _memories: encoded
    runner._encode_batch_with_history_entries = lambda _observations, _memories: (encoded, [entry])
    runner._infer_encoded_batch = lambda _encoded: [("Move", torch.tensor([1]), [])]
    legacy = runner.infer_batch([object()], [object()])
    aware = runner.infer_batch_with_history_entries([object()], [object()])
    assert len(legacy[0]) == 3
    assert len(aware[0]) == 4
    assert aware[0][3] is entry


def test_legacy_missing_entry_cold_advances_resident_suffix() -> None:
    key = _key("low")
    cache = HistoryImageCache.enabled().append_for_frame(frame_index=0, entry=_entry(key, 1))
    advanced = cache.advance_without_entry(frame_index=1)
    assert advanced.entries == ()
    assert advanced.base_frame_index == 2
    assert advanced.limit_bytes == cache.limit_bytes


@pytest.mark.parametrize(
    "runner_type,memory_type,profile,text",
    [
        (QwenR2RLowRunner, QwenR2RLowMemory, "low", "Move"),
        (
            QwenR2RPanoramicRunner,
            QwenR2RPanoramicMemory,
            "panoramic",
            "0",
        ),
    ],
)
def test_public_infer_with_entry_commit_advances_suffix_with_exact_parity(
    runner_type, memory_type, profile, text
) -> None:
    runner = object.__new__(runner_type)
    runner.history_image_cache_mode = "rgb_bytes"
    runner.history_image_cache_enabled = True
    runner.history_image_cache_key = _key(profile)
    entry = _entry(runner.history_image_cache_key, 11)
    encoded = {"input_ids": torch.ones((1, 1), dtype=torch.long)}
    token = torch.tensor([7], dtype=torch.long)
    runner._encode_batch = lambda _observations, _memories: encoded
    runner._encode_batch_with_history_entries = lambda _observations, _memories: (encoded, [entry])
    runner._infer_encoded_batch = lambda _encoded: [(text, token, [0.0])]
    memory = memory_type(history_image_cache=HistoryImageCache.enabled())
    observation = object()

    legacy = runner.infer(observation, memory)
    aware = runner.infer_with_history_entry(observation, memory)
    assert legacy[0] == aware[0] == text
    assert torch.equal(legacy[1], aware[1])
    assert legacy[2] == aware[2]
    legacy_action = parse_low_action(legacy[0]) if profile == "low" else parse_panoramic_action(legacy[0], 2)
    aware_action = parse_low_action(aware[0]) if profile == "low" else parse_panoramic_action(aware[0], 2)
    torch.testing.assert_close(legacy_action, aware_action, rtol=0, atol=0)

    committed_cache = runner.commit_history_image_cache(memory, aware[3])
    height, width = (240, 320) if profile == "low" else (240, 960)
    next_memory = memory_type(
        frames=(_frame(height, width, 0.25),),
        responses=(text,),
        history_image_cache=committed_cache,
    )
    assert next_memory.history_image_cache.base_frame_index == 0
    assert len(next_memory.history_image_cache.entries) == 1
    assert next_memory.history_image_cache.get(0, runner.history_image_cache_key) is entry


@pytest.mark.parametrize(
    "module,runner_type,profile",
    [
        (low_runner_module, QwenR2RLowRunner, "low_level"),
        (
            panoramic_runner_module,
            QwenR2RPanoramicRunner,
            "panoramic",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["none", "rgb_bytes"])
def test_runner_constructor_receives_both_public_cache_modes_before_loading(
    monkeypatch, tmp_path, module, runner_type, profile, mode
) -> None:
    class ConstructorProbe(Exception):
        pass

    seen = []

    def stop_after_resolution(value):
        seen.append(value)
        raise ConstructorProbe

    monkeypatch.setattr(module, "normalize_history_image_cache_mode", stop_after_resolution)
    with pytest.raises(ConstructorProbe):
        runner_type(
            str(tmp_path),
            profile,
            max_new_tokens=1,
            execute_chunks=1,
            history_image_cache=mode,
        )
    assert seen == [mode]


@pytest.mark.parametrize(
    "module,builder,runner_name,profile",
    [
        (
            low_policy_module,
            low_policy_module.build_qwen_r2r_low,
            "QwenR2RLowRunner",
            "low_level",
        ),
        (
            panoramic_policy_module,
            panoramic_policy_module.build_qwen_r2r_panoramic,
            "QwenR2RPanoramicRunner",
            "panoramic",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["none", "rgb_bytes"])
def test_public_factory_forwards_cache_mode(
    monkeypatch, tmp_path, module, builder, runner_name, profile, mode
) -> None:
    captured = []

    class FakeRunner:
        def __init__(self, checkpoint, resolved_profile, **kwargs):
            captured.append((checkpoint, resolved_profile, kwargs["history_image_cache"]))
            self.history_image_cache_mode = kwargs["history_image_cache"]
            self.model = torch.nn.Identity()

    monkeypatch.setattr(module, runner_name, FakeRunner)
    policy = builder(
        checkpoint=tmp_path,
        history_image_cache=mode,
    )
    assert captured == [(tmp_path, profile, mode)]
    assert policy.runner.history_image_cache_mode == mode


@pytest.mark.parametrize(
    "runner_type,profile",
    [
        (QwenR2RLowRunner, "low_level"),
        (QwenR2RPanoramicRunner, "panoramic"),
    ],
)
@pytest.mark.parametrize("invalid", [True, False, "auto", "", None])
def test_runner_constructor_rejects_invalid_cache_modes_before_loading(
    tmp_path, runner_type, profile, invalid
) -> None:
    with pytest.raises(ValueError, match="none.*rgb_bytes"):
        runner_type(
            str(tmp_path),
            profile,
            max_new_tokens=1,
            execute_chunks=1,
            history_image_cache=invalid,
        )


def test_speculative_memory_transactions_rollback_cancel_reset_and_isolate() -> None:
    key = _key("low")
    committed_cache = HistoryImageCache.enabled().append_for_frame(frame_index=0, entry=_entry(key, 1))
    committed = QwenR2RLowMemory(
        frames=(_frame(240, 320, 0.1),),
        responses=("Move",),
        history_image_cache=committed_cache,
    )
    speculative = QwenR2RLowMemory(
        frames=(*committed.frames, _frame(240, 320, 0.2)),
        responses=(*committed.responses, "Turn"),
        history_image_cache=committed_cache.append_for_frame(frame_index=1, entry=_entry(key, 2)),
    )
    first_key = _session_key("cache-a")
    second_key = _session_key("cache-b")
    store = SessionStore()
    store.checkout(first_key).commit(committed)
    store.checkout(second_key).commit(committed)

    rolled_back = store.checkout(first_key)
    rolled_back.rollback()
    assert store.committed(first_key) is committed
    assert store.committed(second_key) is committed

    cancelled = store.checkout(first_key)
    store.cancel([first_key])
    assert cancelled.cancelled()
    with pytest.raises(StaleSessionError):
        cancelled.commit(speculative)
    assert store.committed(first_key) is committed
    assert store.committed(second_key) is committed

    store.checkout(second_key).commit(speculative)
    assert store.committed(first_key) is committed
    assert store.committed(second_key) is speculative
    store.reset([first_key])
    assert store.committed(first_key) is None
    assert store.committed(second_key) is speculative


def test_history_cache_modes_are_explicit_and_fail_loud() -> None:
    assert normalize_history_image_cache_mode("none") == "none"
    assert normalize_history_image_cache_mode("rgb_bytes") == "rgb_bytes"
    for invalid in (False, True, "auto", "", None):
        with pytest.raises(ValueError, match="none.*rgb_bytes"):
            normalize_history_image_cache_mode(invalid)


def test_payload_hash_is_stable_for_reference_artifacts() -> None:
    entry = _entry(_key(), 7)
    assert hashlib.sha256(entry.rgb_bytes).hexdigest() == hashlib.sha256(entry.to_pil().tobytes()).hexdigest()
