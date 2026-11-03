"""CUTLASS swap-AB NVFP4 projection for a few hundred tokens on Blackwell."""

import ctypes

import torch

from .cutlass_geglu import available, cutlass_root
from .native import build_library

_LIBRARIES: dict[int, ctypes.CDLL] = {}

__all__ = ["available", "cutlass_swap_linear"]


def _library(device: torch.device) -> ctypes.CDLL:
    index = device.index if device.index is not None else torch.cuda.current_device()
    library = _LIBRARIES.get(index)
    if library is None:
        root = cutlass_root()
        if root is None:
            raise RuntimeError(
                "CUTLASS projection requires EMBODIINFER_CUTLASS_DIR to point at CUTLASS headers"
            )
        with torch.cuda.device(index):
            library = build_library(
                "cutlass_projection",
                specific=True,
                flags=(
                    f"-I{root / 'include'}",
                    f"-I{root / 'tools' / 'util' / 'include'}",
                    "--expt-relaxed-constexpr",
                ),
            ).library
        library.cc_cutlass_swap_linear.argtypes = (
            [ctypes.c_int] * 3
            + [ctypes.c_void_p] * 6
            + [
                ctypes.c_int,
                ctypes.c_void_p,
            ]
        )
        library.cc_cutlass_swap_linear.restype = ctypes.c_int
        _LIBRARIES[index] = library
    return library


def cutlass_swap_linear(
    x_packed: torch.Tensor,
    x_blocked: torch.Tensor,
    weight: torch.Tensor,
    weight_blocked: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """out = bf16(alpha * x @ weight^T) for NVFP4 x [tokens, K/2] and weight [n_out, K/2] (programmatic launch)."""
    tokens, half_k = x_packed.shape
    status = _library(x_packed.device).cc_cutlass_swap_linear(
        tokens,
        weight.shape[0],
        half_k * 2,
        x_packed.data_ptr(),
        x_blocked.data_ptr(),
        weight.data_ptr(),
        weight_blocked.data_ptr(),
        alpha.data_ptr(),
        out.data_ptr(),
        1,
        torch.cuda.current_stream(x_packed.device).cuda_stream,
    )
    if status != 0:
        raise RuntimeError(f"CUTLASS swap-AB projection launch failed with status {status}")
