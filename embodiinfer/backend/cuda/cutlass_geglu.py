"""CUTLASS NVFP4 prefix GeGLU for Blackwell over 8-row-interleaved gate/up weights."""

import ctypes
import os
from pathlib import Path

import torch

from .native import build_library

_LIBRARIES: dict[int, ctypes.CDLL] = {}


def cutlass_root() -> Path | None:
    """CUTLASS checkout named by EMBODIINFER_CUTLASS_DIR, if it provides the C++ headers."""
    configured = os.environ.get("EMBODIINFER_CUTLASS_DIR")
    if not configured:
        return None
    root = Path(configured).expanduser()
    return root if (root / "include" / "cutlass" / "cutlass.h").is_file() else None


def available(device: torch.device) -> bool:
    """CUTLASS headers are configured and the device has Blackwell block-scaled tensor cores."""
    return cutlass_root() is not None and torch.cuda.get_device_capability(device)[0] >= 10


def _library(device: torch.device) -> ctypes.CDLL:
    index = device.index if device.index is not None else torch.cuda.current_device()
    library = _LIBRARIES.get(index)
    if library is None:
        root = cutlass_root()
        if root is None:
            raise RuntimeError("CUTLASS GeGLU requires EMBODIINFER_CUTLASS_DIR to point at CUTLASS headers")
        with torch.cuda.device(index):
            library = build_library(
                "cutlass_geglu",
                specific=True,
                flags=(
                    f"-I{root / 'include'}",
                    f"-I{root / 'tools' / 'util' / 'include'}",
                    "--expt-relaxed-constexpr",
                ),
            ).library
        library.cc_cutlass_geglu_fp4.argtypes = (
            [ctypes.c_int] * 3 + [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_void_p]
        )
        library.cc_cutlass_geglu_fp4.restype = ctypes.c_int
        _LIBRARIES[index] = library
    return library


def _plain_scales(blocked: torch.Tensor, rows: int, scale_columns: int) -> torch.Tensor:
    """128x4 interleaved block scales -> row-major [rows, scale_columns] bytes."""
    tiles = blocked.view(torch.uint8).reshape(rows // 128, scale_columns // 4, 32, 4, 4)
    return tiles.permute(0, 3, 2, 1, 4).reshape(rows, scale_columns)


def interleave_gate_up(
    gate: torch.Tensor, gate_blocked: torch.Tensor, up: torch.Tensor, up_blocked: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows [gate 0..7, up 0..7, gate 8..15, up 8..15, ...] and their block scales in the 128x4 layout."""
    rows, half_k = gate.shape
    if up.shape != gate.shape or rows % 128 or (half_k * 2) % 64:
        raise ValueError("Interleaved GeGLU weights require matching [F, K/2] gate/up with F % 128 == 0")
    scale_columns = half_k * 2 // 16
    weights = torch.stack((gate.view(rows // 8, 8, half_k), up.view(rows // 8, 8, half_k)), dim=1)
    weights = weights.reshape(2 * rows, half_k).contiguous()
    plain = torch.stack(
        (
            _plain_scales(gate_blocked, rows, scale_columns).view(rows // 8, 8, scale_columns),
            _plain_scales(up_blocked, rows, scale_columns).view(rows // 8, 8, scale_columns),
        ),
        dim=1,
    ).reshape(2 * rows, scale_columns)
    tiles = plain.reshape(2 * rows // 128, 4, 32, scale_columns // 4, 4).permute(0, 3, 2, 1, 4).contiguous()
    return weights, tiles.reshape(-1).view(torch.float8_e4m3fn)


def cutlass_geglu_fp4(
    a_packed: torch.Tensor,
    a_blocked: torch.Tensor,
    weights: torch.Tensor,
    weights_blocked: torch.Tensor,
    gate_scale: torch.Tensor,
    up_scale: torch.Tensor,
    inv_scale: torch.Tensor,
    out: torch.Tensor,
    out_blocked: torch.Tensor,
) -> None:
    """Launch the fused GEMM + GeGLU + NVFP4 kernel on the current stream (programmatic launch).

    The output is written L2 evict_last for the Down GEMM that reads it next; release it with
    l2_prefetch.l2_discard once no kernel reads it any more.
    """
    rows, half_k = a_packed.shape
    features = weights.shape[0] // 2
    status = _library(a_packed.device).cc_cutlass_geglu_fp4(
        rows,
        half_k * 2,
        features,
        a_packed.data_ptr(),
        a_blocked.data_ptr(),
        weights.data_ptr(),
        weights_blocked.data_ptr(),
        gate_scale.data_ptr(),
        up_scale.data_ptr(),
        inv_scale.data_ptr(),
        out.data_ptr(),
        out_blocked.data_ptr(),
        1,
        torch.cuda.current_stream(a_packed.device).cuda_stream,
    )
    if status != 0:
        raise RuntimeError(f"CUTLASS GeGLU launch failed with status {status}")
