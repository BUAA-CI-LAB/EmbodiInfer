"""Side-stream prefetch of upcoming read-only weights into the device L2 cache.

Hopper and later issue TMA bulk prefetches; Ampere/Ada hint each 128-byte line, where two
256-thread CTAs land a 4090 expert layer in time without crowding the attention it overlaps.
"""

import ctypes
from collections.abc import Sequence

import torch

from .native import build_library

_U64 = ctypes.c_ulonglong
_MAX_SEGMENTS = 16
_DISCARD: ctypes.CDLL | None = None


def l2_discard(tensors: Sequence[torch.Tensor], *, ctas: int = 32) -> None:
    """Drop dead tensors from L2 on the current stream without writing them back.

    Their contents become undefined, so only pass tensors that every later reader rewrites first.
    """
    global _DISCARD
    if not tensors:
        return
    if _DISCARD is None:
        _DISCARD = build_library("l2_prefetch").library
        pointer = ctypes.POINTER(_U64)
        _DISCARD.cc_l2_discard.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _DISCARD.cc_l2_discard.restype = ctypes.c_int
    stream = torch.cuda.current_stream(tensors[0].device).cuda_stream
    for start in range(0, len(tensors), _MAX_SEGMENTS):
        chunk = tensors[start : start + _MAX_SEGMENTS]
        pointers = (_U64 * len(chunk))(*(t.data_ptr() for t in chunk))
        sizes = (_U64 * len(chunk))(*(t.numel() * t.element_size() for t in chunk))
        status = _DISCARD.cc_l2_discard(pointers, sizes, len(chunk), ctas, stream)
        if status:
            raise RuntimeError(f"L2 discard launch failed: {status}")


class L2Prefetcher:
    """Fork bulk L2 prefetches onto a private stream and join them before the caller returns.

    Prefetches only read their segments, so outputs are unchanged; they overlap the DRAM
    transfer of the next layer's weights with the current layer's compute.
    """

    def __init__(self, device: torch.device, *, ctas: int = 2) -> None:
        """Load the prefetch kernel and create the private stream outside graph capture."""
        self.device = torch.device(device)
        self.library = build_library("l2_prefetch").library
        pointer = ctypes.POINTER(_U64)
        self.library.cc_l2_bulk_prefetch.argtypes = [
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.cc_l2_bulk_prefetch.restype = ctypes.c_int
        self.stream = torch.cuda.Stream(self.device)
        self.ctas = ctas
        self.fork_event = torch.cuda.Event()
        self.done_event = torch.cuda.Event()
        self.pending = False

    @staticmethod
    def segments(tensors: Sequence[torch.Tensor]) -> tuple:
        """Pack device pointers and byte sizes once for repeated launches."""
        pointers = (_U64 * len(tensors))(*(t.data_ptr() for t in tensors))
        sizes = (_U64 * len(tensors))(*(t.numel() * t.element_size() for t in tensors))
        return pointers, sizes, len(tensors)

    def fork(self, segments: tuple) -> None:
        """Start one prefetch after the work already queued on the current stream."""
        self.fork_event.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(self.fork_event)
        pointers, sizes, count = segments
        status = self.library.cc_l2_bulk_prefetch(pointers, sizes, count, self.ctas, self.stream.cuda_stream)
        if status:
            raise RuntimeError(f"L2 bulk prefetch launch failed: {status}")
        self.pending = True

    def join(self) -> None:
        """Make the current stream wait for every queued prefetch."""
        if not self.pending:
            return
        self.done_event.record(self.stream)
        torch.cuda.current_stream(self.device).wait_event(self.done_event)
        self.pending = False
