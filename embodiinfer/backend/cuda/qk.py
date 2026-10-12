"""Instance-local Tensor Core QK with FP32 storage and explicit stream dispatch."""

from __future__ import annotations

import ctypes
import threading

import torch

from .native import build_library


class TensorCoreQk:
    """B1 MQA prefix QK without process-wide matmul precision changes.

    One policy serializes its attention dispatches and owns this cuBLAS handle.
    TF32 changes accumulation order relative to strict FP32; inputs must already
    contain the reference's BF16-rounded query scaling and rotary results.
    """

    def __init__(self) -> None:
        """Create a private TF32 handle outside capture on the selected device."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.library = build_library("qk", specific=False, libraries=("cublas",)).library
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.library.ei_qk_create.argtypes = [ctypes.POINTER(ptr)]
        self.library.ei_qk_destroy.argtypes = [ptr]
        self.library.ei_qk_run.argtypes = [ptr] * 4 + [integer] * 3 + [ptr, ctypes.c_size_t, ptr]
        for name in ("ei_qk_create", "ei_qk_destroy", "ei_qk_run"):
            getattr(self.library, name).restype = integer
        self.handle = ptr()
        self.lock = threading.Lock()
        status = self.library.ei_qk_create(ctypes.byref(self.handle))
        if status:
            raise RuntimeError(f"TF32 QK handle creation failed: {status}")

    def __del__(self) -> None:
        """Release the native handle on its owning device when initialized."""
        if getattr(self, "handle", None) and torch.cuda.is_initialized():
            with torch.cuda.device(self.device):
                self.library.ei_qk_destroy(self.handle)

    def __call__(self, query: torch.Tensor, key: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
        """Project with caller-owned scratch whose lifetime covers its graph replay."""
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("TF32 QK requires BHSD queries and BS1D keys")
        batch, heads, tokens, width = query.shape
        if (
            batch != 1
            or min(heads, tokens, width) <= 0
            or heads * tokens * max(width, tokens) >= 2**31
            or key.shape != (1, tokens, 1, width)
            or any(
                x.device != self.device or x.dtype != torch.float32 or not x.is_contiguous()
                for x in (query, key)
            )
        ):
            raise ValueError("TF32 QK requires contiguous B1 MQA prefix storage on its bound device")
        if (
            workspace.device != self.device
            or workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() < 4 * 1024**2
        ):
            raise ValueError("TF32 QK requires at least 4 MiB of private same-device workspace")
        scores = torch.empty((1, 1, heads, tokens, tokens), device=self.device, dtype=torch.float32)
        with self.lock, torch.cuda.device(self.device):
            status = self.library.ei_qk_run(
                self.handle,
                query.data_ptr(),
                key.data_ptr(),
                scores.data_ptr(),
                tokens,
                heads,
                width,
                workspace.data_ptr(),
                workspace.numel(),
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        if status:
            raise RuntimeError(f"TF32 QK dispatch failed: {status}")
        return scores
