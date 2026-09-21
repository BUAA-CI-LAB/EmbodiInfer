"""Immutable recurrent post-resize RGB cache for Qwen navigation policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar, cast

from PIL import Image

QWEN25_VL_HISTORY_IMAGE_CACHE_ABI = "qwen25_vl_history_image_cache_v2"
QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES = 16 * 1024 * 1024
HistoryImageCacheMode = Literal["none", "rgb_bytes"]

_FrameT = TypeVar("_FrameT")


def normalize_history_image_cache_mode(value: object) -> HistoryImageCacheMode:
    if value not in ("none", "rgb_bytes") or not isinstance(value, str):
        raise ValueError("history_image_cache must be exactly 'none' or 'rgb_bytes'")
    return cast(HistoryImageCacheMode, value)


@dataclass(frozen=True)
class HistoryImageCacheKey:
    profile: str
    kind: str
    size: tuple[int, int]
    resize: bool
    processor_sha256: str
    abi: str = QWEN25_VL_HISTORY_IMAGE_CACHE_ABI

    def __post_init__(self) -> None:
        if self.abi != QWEN25_VL_HISTORY_IMAGE_CACHE_ABI:
            raise ValueError("unsupported history image cache ABI")
        formal = {
            ("low_level", "low"): ((320, 240), True),
            ("panoramic", "panorama"): ((960, 240), False),
        }
        expected = formal.get((self.profile, self.kind))
        if expected is None:
            raise ValueError(
                "history image cache only accepts low-level frames or panoramas; "
                "candidate images are never cacheable"
            )
        if self.size != expected[0] or self.resize is not expected[1]:
            raise ValueError("history image cache geometry contract mismatch")
        if len(self.processor_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.processor_sha256
        ):
            raise ValueError("processor_sha256 must be 64 lowercase hexadecimal digits")

    @property
    def payload_bytes(self) -> int:
        width, height = self.size
        return width * height * 3

    def as_dict(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "profile": self.profile,
            "kind": self.kind,
            "size": list(self.size),
            "resize": self.resize,
            "processor_sha256": self.processor_sha256,
        }


@dataclass(frozen=True)
class HistoryImageCacheEntry:
    key: HistoryImageCacheKey
    rgb_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.rgb_bytes) is not bytes:
            raise TypeError("history image cache payload must be immutable bytes")
        if len(self.rgb_bytes) != self.key.payload_bytes:
            raise ValueError("history image cache payload byte count mismatch")

    @classmethod
    def from_pil(
        cls,
        image: Image.Image,
        key: HistoryImageCacheKey,
    ) -> HistoryImageCacheEntry:
        if image.mode != "RGB":
            raise ValueError("history image cache requires post-conversion RGB")
        if image.size != key.size:
            raise ValueError("history image cache requires exact post-resize dimensions")
        return cls(key=key, rgb_bytes=image.tobytes())

    @property
    def nbytes(self) -> int:
        return len(self.rgb_bytes)

    def to_pil(self) -> Image.Image:
        return Image.frombytes("RGB", self.key.size, self.rgb_bytes)


@dataclass(frozen=True)
class HistoryImageCache:
    entries: tuple[HistoryImageCacheEntry, ...] = ()
    base_frame_index: int = 0
    bytes_used: int = 0
    limit_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.base_frame_index) is not int or self.base_frame_index < 0:
            raise ValueError("base_frame_index must be a non-negative integer")
        if (
            type(self.limit_bytes) is not int
            or self.limit_bytes < 0
            or self.limit_bytes > QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES
        ):
            raise ValueError("history image cache limit must be between 0 and 16 MiB")
        if self.entries and self.limit_bytes == 0:
            raise ValueError("a disabled history image cache cannot contain entries")
        actual_bytes = sum(entry.nbytes for entry in self.entries)
        if self.bytes_used != actual_bytes:
            raise ValueError("history image cache bytes_used does not match payloads")
        if actual_bytes > self.limit_bytes:
            raise ValueError("history image cache exceeds its session byte limit")

    @property
    def enabled_for_session(self) -> bool:
        return self.limit_bytes > 0

    @classmethod
    def enabled(
        cls,
        limit_bytes: int = QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES,
    ) -> HistoryImageCache:
        if type(limit_bytes) is not int or not (0 < limit_bytes <= QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES):
            raise ValueError("enabled history image cache needs a 1..16 MiB limit")
        return cls(limit_bytes=limit_bytes)

    def get(
        self,
        frame_index: int,
        key: HistoryImageCacheKey,
    ) -> HistoryImageCacheEntry | None:
        if type(frame_index) is not int or frame_index < self.base_frame_index:
            return None
        offset = frame_index - self.base_frame_index
        if offset < 0 or offset >= len(self.entries):
            return None
        entry = self.entries[offset]
        return entry if entry.key == key else None

    def append_for_frame(
        self,
        *,
        frame_index: int,
        entry: HistoryImageCacheEntry,
    ) -> HistoryImageCache:
        if not self.enabled_for_session:
            return self
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self.entries and frame_index != self.base_frame_index + len(self.entries):
            raise ValueError("history cache appends must be contiguous")
        entries = (*self.entries, entry)
        base = self.base_frame_index if self.entries else frame_index
        bytes_used = self.bytes_used + entry.nbytes
        evicted = 0
        while entries and bytes_used > self.limit_bytes:
            bytes_used -= entries[0].nbytes
            entries = entries[1:]
            evicted += 1
        return HistoryImageCache(
            entries=entries,
            base_frame_index=base + evicted,
            bytes_used=bytes_used,
            limit_bytes=self.limit_bytes,
        )

    def advance_without_entry(self, *, frame_index: int) -> HistoryImageCache:
        """Commit a raw frame while dropping a no-longer-contiguous RGB suffix."""
        if not self.enabled_for_session:
            return self
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self.entries and frame_index != self.base_frame_index + len(self.entries):
            raise ValueError("cold history cache advances must be contiguous")
        return HistoryImageCache(
            base_frame_index=frame_index + 1,
            limit_bytes=self.limit_bytes,
        )

    def validate_history(
        self,
        history_length: int,
        expected_key: HistoryImageCacheKey,
    ) -> None:
        if type(history_length) is not int or history_length < 0:
            raise ValueError("history_length must be a non-negative integer")
        if not 0 <= self.base_frame_index <= history_length:
            raise ValueError("history cache base_frame_index is outside frame history")
        if not self.entries:
            return
        if self.base_frame_index + len(self.entries) != history_length:
            raise ValueError("resident history entries must form a contiguous suffix")
        if any(entry.key != expected_key for entry in self.entries):
            raise ValueError("resident history entry has the wrong formal cache key")

    def as_dict(self) -> dict[str, object]:
        return {
            "abi": QWEN25_VL_HISTORY_IMAGE_CACHE_ABI,
            "enabled": self.enabled_for_session,
            "limit_bytes": self.limit_bytes,
            "bytes_used": self.bytes_used,
            "resident_entries": len(self.entries),
            "base_frame_index": self.base_frame_index,
            "end_frame_index": self.base_frame_index + len(self.entries),
        }


@dataclass(frozen=True)
class HistoryImagePreparationStats:
    history_hits: int = 0
    history_misses: int = 0
    history_renders: int = 0
    current_renders: int = 0
    candidate_renders: int = 0


def prepare_history_images(
    frames: Sequence[_FrameT],
    kinds: Sequence[str],
    *,
    history_count: int,
    cache: HistoryImageCache,
    key: HistoryImageCacheKey,
    cache_enabled: bool,
    capture_current_entry: bool = True,
    render: Callable[[_FrameT, str], Image.Image],
) -> tuple[
    list[Image.Image],
    HistoryImageCacheEntry | None,
    HistoryImagePreparationStats,
]:
    if len(frames) != len(kinds):
        raise ValueError("history image frames and kinds must be aligned")
    if not 0 <= history_count < len(frames):
        raise ValueError("prepared image sequence must contain one current frame")
    rendered: list[Image.Image] = []
    current_entry: HistoryImageCacheEntry | None = None
    hits = 0
    misses = 0
    history_renders = 0
    current_renders = 0
    candidate_renders = 0
    for index, (frame, kind) in enumerate(zip(frames, kinds, strict=True)):
        cached = None
        if index < history_count and kind == key.kind and cache_enabled:
            cached = cache.get(index, key)
            if cached is None:
                misses += 1
            else:
                hits += 1
        if cached is not None:
            image = cached.to_pil()
        else:
            image = render(frame, kind)
            if index < history_count:
                history_renders += 1
            elif index == history_count and kind == key.kind:
                current_renders += 1
            elif index >= history_count:
                candidate_renders += 1
        if cache_enabled and capture_current_entry and index == history_count and kind == key.kind:
            current_entry = HistoryImageCacheEntry.from_pil(image, key)
        rendered.append(image)
    if cache_enabled and capture_current_entry and current_entry is None:
        raise ValueError("cacheable current frame is missing from prepared images")
    return (
        rendered,
        current_entry,
        HistoryImagePreparationStats(
            history_hits=hits,
            history_misses=misses,
            history_renders=history_renders,
            current_renders=current_renders,
            candidate_renders=candidate_renders,
        ),
    )


__all__ = [
    "HistoryImageCache",
    "HistoryImageCacheEntry",
    "HistoryImageCacheKey",
    "HistoryImageCacheMode",
    "HistoryImagePreparationStats",
    "QWEN25_VL_HISTORY_IMAGE_CACHE_ABI",
    "QWEN25_VL_HISTORY_IMAGE_CACHE_LIMIT_BYTES",
    "normalize_history_image_cache_mode",
    "prepare_history_images",
]
