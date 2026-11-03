"""Device-side camera frame preparation, so graph-captured requests can start from raw uint8 frames."""

import ctypes
import math

import torch

from .native import build_library

_LIBRARY: ctypes.CDLL | None = None
# Pillow keeps 8-bit resampling taps as 22-bit fixed point.
_PRECISION_BITS = 22


def _library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        library = build_library("camera_frames", strict=True).library
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        library.cc_normalize_frames.argtypes = [pointer, pointer, integer, integer, pointer]
        library.cc_normalize_frames.restype = integer
        library.cc_resize_frames.argtypes = (
            [pointer] * 5 + [integer] + [pointer] * 2 + [integer] * 10 + [pointer]
        )
        library.cc_resize_frames.restype = integer
        _LIBRARY = library
    return _LIBRARY


def _check_frames(frames: torch.Tensor, images: torch.Tensor) -> None:
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[3] != 3 or not frames.is_contiguous():
        raise ValueError("frames must be contiguous uint8 [cameras, height, width, 3]")
    if (
        images.dtype != torch.float32
        or images.ndim != 4
        or tuple(images.shape[:2]) != (frames.shape[0], 3)
        or not images.is_contiguous()
    ):
        raise ValueError("images must be contiguous float32 [cameras, 3, height, width]")
    if frames.device.type != "cuda" or images.device != frames.device:
        raise ValueError("frames and images must share a CUDA device")


def normalize_frames(frames: torch.Tensor, images: torch.Tensor) -> None:
    """Write ``(frames / 255) * 2 - 1`` into float32 ``images`` [cameras, 3, H, W] on the current stream.

    ``frames`` is contiguous uint8 [cameras, H, W, 3] on the same CUDA device. Every step is rounded
    as on the host, so the result matches a float32 ``frames / 255 * 2 - 1`` bit for bit.
    """
    _check_frames(frames, images)
    cameras, height, width, _ = frames.shape
    if tuple(images.shape[2:]) != (height, width):
        raise ValueError("images must have the frame height and width")
    stream = torch.cuda.current_stream(frames.device).cuda_stream
    status = _library().cc_normalize_frames(
        frames.data_ptr(), images.data_ptr(), cameras, height * width, stream
    )
    if status:
        raise RuntimeError(f"Camera frame normalization launch failed: {status}")


def _pillow_bilinear_taps(source: int, target: int) -> tuple[list[int], list[int], int]:
    """Pillow's BILINEAR taps for one axis: (first, count) per output and fixed-point weights.

    Mirrors ``precompute_coeffs`` and ``normalize_coeffs_8bpc`` in Pillow's Resample.c, including
    its double-precision expression order, so the integer weights are identical.
    """
    scale = source / target
    filter_scale = max(scale, 1.0)
    support = filter_scale
    inverse = 1.0 / filter_scale
    taps = int(math.ceil(support)) * 2 + 1
    bounds, weights = [], []
    for index in range(target):
        center = (index + 0.5) * scale
        first = max(int(center - support + 0.5), 0)
        count = min(int(center + support + 0.5), source) - first
        raw = [max(0.0, 1.0 - abs((offset + first - center + 0.5) * inverse)) for offset in range(count)]
        total = sum(raw)
        if total != 0.0:
            raw = [value / total for value in raw]
        fixed = [
            int(-0.5 + value * (1 << _PRECISION_BITS))
            if value < 0
            else int(0.5 + value * (1 << _PRECISION_BITS))
            for value in raw
        ]
        bounds += [first, count]
        weights += fixed + [0] * (taps - count)
    return bounds, weights, taps


class FrameResize:
    """OpenPI's ``resize_with_pad`` on the device, bit-exact with Pillow, followed by normalization.

    Frames of ``source`` (height, width) are scaled by one ratio to fit ``target``, resampled with
    Pillow's two-pass 8-bit bilinear filter, centered on a black canvas and written as float32
    ``(x / 255) * 2 - 1``. The fixed-point taps are uploaded once, so launches are graph-capturable.
    """

    def __init__(self, source: tuple[int, int], target: tuple[int, int], device: torch.device) -> None:
        source_height, source_width = source
        self.canvas = target
        ratio = max(source_width / target[1], source_height / target[0])
        self.height, self.width = int(source_height / ratio), int(source_width / ratio)
        self.top, self.left = (target[0] - self.height) // 2, (target[1] - self.width) // 2
        self.source = source
        row_bounds, row_weights, self.row_taps = _pillow_bilinear_taps(source_width, self.width)
        column_bounds, column_weights, self.column_taps = _pillow_bilinear_taps(source_height, self.height)
        tables = [row_bounds, row_weights, column_bounds, column_weights]
        self.row_bounds, self.row_weights, self.column_bounds, self.column_weights = (
            torch.tensor(table, dtype=torch.int32, device=device) for table in tables
        )

    def rows_buffer(self, cameras: int, device: torch.device) -> torch.Tensor:
        """Scratch for the horizontal pass, [cameras, source height, resized width, 3] uint8."""
        return torch.empty((cameras, self.source[0], self.width, 3), dtype=torch.uint8, device=device)

    def __call__(self, frames: torch.Tensor, rows: torch.Tensor, images: torch.Tensor) -> None:
        """Resize ``frames`` through ``rows`` into ``images`` [cameras, 3, *target] on the current stream."""
        _check_frames(frames, images)
        cameras = frames.shape[0]
        if tuple(frames.shape[1:3]) != self.source or tuple(images.shape[2:]) != self.canvas:
            raise ValueError("frames and images must match the planned source and target sizes")
        if rows.dtype != torch.uint8 or tuple(rows.shape) != (cameras, self.source[0], self.width, 3):
            raise ValueError("rows must be the uint8 scratch from rows_buffer")
        stream = torch.cuda.current_stream(frames.device).cuda_stream
        status = _library().cc_resize_frames(
            frames.data_ptr(),
            rows.data_ptr(),
            images.data_ptr(),
            self.row_bounds.data_ptr(),
            self.row_weights.data_ptr(),
            self.row_taps,
            self.column_bounds.data_ptr(),
            self.column_weights.data_ptr(),
            self.column_taps,
            cameras,
            self.source[0],
            self.source[1],
            self.height,
            self.width,
            self.canvas[0],
            self.canvas[1],
            self.top,
            self.left,
            stream,
        )
        if status:
            raise RuntimeError(f"Camera frame resize launch failed: {status}")
