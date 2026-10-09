"""Native warp mask/softmax with query-major output strides."""

import ctypes

import torch

from .native import build_library


class Layout(ctypes.Structure):
    """C ABI for score/mask strides and flattened Gemma head dimensions."""

    _fields_ = [("strides", ctypes.c_int64 * 13), ("dimensions", ctypes.c_int * 6)]


class CudaSoftmax:
    """Preserve sequential per-lane sums, XOR warp sums and RN probability division."""

    def __init__(self) -> None:
        """Load the cached library for the current CUDA device."""
        built = build_library("softmax", strict=True, specific=False, native_fp4=False)
        self.library, self.compiler, self.command = built.library, built.compiler, built.command
        pointer = ctypes.c_void_p
        self.library.cc_mask_softmax.argtypes = [pointer] * 3 + [
            Layout,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
        ]
        self.library.cc_mask_softmax.restype = ctypes.c_int

    def __call__(self, logits, mask, output, warps, *, query_rows=False):
        """Launch on the current Torch stream, including during CUDA Graph capture."""
        if logits.shape[-1] > 2048:
            raise ValueError("CUDA warp softmax supports at most 2048 keys")
        batch, heads, groups, queries, keys = logits.shape
        layout = Layout(
            (ctypes.c_int64 * 13)(*logits.stride(), *mask.stride(), *output.stride()),
            (ctypes.c_int * 6)(
                heads, groups, queries, keys, batch * heads * groups * queries, int(query_rows)
            ),
        )
        status = self.library.cc_mask_softmax(
            logits.data_ptr(),
            mask.data_ptr(),
            output.data_ptr(),
            layout,
            warps,
            int(output.dtype == torch.bfloat16),
            torch.cuda.current_stream(logits.device).cuda_stream,
        )
        if status:
            raise RuntimeError(f"CUDA mask/softmax/cast launch failed: {status}")
