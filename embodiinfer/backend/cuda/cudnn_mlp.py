"""cuDNN 9.20 runtime-fused NVFP4 and FP8 MLP and Linear epilogues."""

import ctypes

import torch

from .native import build_library

_LIBRARIES: dict[int, ctypes.CDLL] = {}


def _get_library(device: torch.device) -> ctypes.CDLL:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    lib = _LIBRARIES.get(idx)
    if lib is None:
        with torch.cuda.device(idx):
            built = build_library(
                "cudnn_mlp",
                strict=False,
                specific=True,
                native_fp4=False,
                libraries=("cudnn",),
            )
        lib = built.library
        vp = ctypes.c_void_p
        i64 = ctypes.c_int64
        ci = ctypes.c_int
        lib.cc_cudnn_dual_geglu.argtypes = [i64, i64, i64, ci] + [vp] * 12
        lib.cc_cudnn_dual_geglu.restype = ci
        lib.cc_cudnn_fp8_bias_gelu.argtypes = [i64, i64, i64] + [vp] * 7
        lib.cc_cudnn_fp8_bias_gelu.restype = ci
        lib.cc_cudnn_nvfp4_linear.argtypes = [i64, i64, i64] + [vp] * 8
        lib.cc_cudnn_nvfp4_linear.restype = ci
        lib.cc_cudnn_fp8_linear.argtypes = [i64, i64, i64] + [vp] * 6
        lib.cc_cudnn_fp8_linear.restype = ci
        lib.cc_cudnn_fp4_bias_gelu.argtypes = [i64, i64, i64] + [vp] * 10
        lib.cc_cudnn_fp4_bias_gelu.restype = ci
        lib.cc_cudnn_packed_geglu_fp4.argtypes = [i64, i64, i64] + [vp] * 5 + [ctypes.c_float] + [vp] * 6
        lib.cc_cudnn_packed_geglu_fp4.restype = ci
        sz = ctypes.c_size_t
        lib.cc_l2_prefetch_fork.argtypes = [vp, sz, vp, sz, vp, sz, vp, sz, ci, vp]
        lib.cc_l2_prefetch_fork.restype = ci
        lib.cc_l2_prefetch_join.argtypes = [vp]
        lib.cc_l2_prefetch_join.restype = ci
        _LIBRARIES[idx] = lib
    return lib


def cudnn_dual_geglu(
    a_packed: torch.Tensor,
    a_blocked: torch.Tensor,
    wg_packed: torch.Tensor,
    wg_blocked: torch.Tensor,
    wu_packed: torch.Tensor,
    wu_blocked: torch.Tensor,
    g_scale: torch.Tensor,
    u_scale: torch.Tensor,
    inv_scale: torch.Tensor | None,
    out: torch.Tensor,
    out_blocked: torch.Tensor | None,
    out_bits: int,
) -> None:
    """Launch the single-node cuDNN 9.20 DualMatmul + GeGLU + Quantize graph."""
    rows = a_packed.shape[0]
    k = a_packed.shape[1] * 2
    n = wg_packed.shape[0]
    lib = _get_library(a_packed.device)
    stream = torch.cuda.current_stream(a_packed.device).cuda_stream
    status = lib.cc_cudnn_dual_geglu(
        rows,
        k,
        n,
        out_bits,
        a_packed.data_ptr(),
        a_blocked.data_ptr(),
        wg_packed.data_ptr(),
        wg_blocked.data_ptr(),
        wu_packed.data_ptr(),
        wu_blocked.data_ptr(),
        g_scale.data_ptr(),
        u_scale.data_ptr(),
        inv_scale.data_ptr() if inv_scale is not None else None,
        out.data_ptr(),
        out_blocked.data_ptr() if out_blocked is not None else None,
        stream,
    )
    if status != 0:
        raise RuntimeError(f"cuDNN dual GeGLU execution failed with status {status}")


def cudnn_fp8_bias_gelu(
    a_fp8: torch.Tensor,
    w_fp8_t: torch.Tensor,
    comb_scale: torch.Tensor,
    bias_bf16: torch.Tensor,
    inv_out_scale: torch.Tensor,
    out_fp8: torch.Tensor,
) -> None:
    """Launch the single-node cuDNN 9.20 FP8 Matmul + Bias + Tanh-GELU + Quantize graph."""
    rows, k = a_fp8.shape
    n = w_fp8_t.shape[1]
    lib = _get_library(a_fp8.device)
    stream = torch.cuda.current_stream(a_fp8.device).cuda_stream
    status = lib.cc_cudnn_fp8_bias_gelu(
        rows,
        k,
        n,
        a_fp8.data_ptr(),
        w_fp8_t.data_ptr(),
        comb_scale.data_ptr(),
        bias_bf16.data_ptr(),
        inv_out_scale.data_ptr(),
        out_fp8.data_ptr(),
        stream,
    )
    if status != 0:
        raise RuntimeError(f"cuDNN FP8 bias+GELU execution failed with status {status}")


def cudnn_nvfp4_linear(
    a_packed: torch.Tensor,
    a_blocked: torch.Tensor,
    w_packed: torch.Tensor,
    w_blocked: torch.Tensor,
    comb_scale: torch.Tensor,
    bias_bf16: torch.Tensor | None,
    out_bf16: torch.Tensor,
) -> None:
    """Launch the single-node cuDNN 9.20 NVFP4 Matmul (+ optional BF16 bias) -> BF16 graph."""
    rows = a_packed.shape[0]
    k = a_packed.shape[1] * 2
    n = w_packed.shape[0]
    lib = _get_library(a_packed.device)
    stream = torch.cuda.current_stream(a_packed.device).cuda_stream
    status = lib.cc_cudnn_nvfp4_linear(
        rows,
        k,
        n,
        a_packed.data_ptr(),
        a_blocked.data_ptr(),
        w_packed.data_ptr(),
        w_blocked.data_ptr(),
        comb_scale.data_ptr(),
        bias_bf16.data_ptr() if bias_bf16 is not None else None,
        out_bf16.data_ptr(),
        stream,
    )
    if status != 0:
        raise RuntimeError(f"cuDNN NVFP4 linear execution failed with status {status}")


def cudnn_fp8_linear(
    a_fp8: torch.Tensor,
    w_fp8_t: torch.Tensor,
    comb_scale: torch.Tensor,
    bias_bf16: torch.Tensor | None,
    out_bf16: torch.Tensor,
) -> None:
    """Launch the single-node cuDNN 9.20 FP8 Matmul (+ optional BF16 bias) -> BF16 graph."""
    rows, k = a_fp8.shape
    n = w_fp8_t.shape[1]
    lib = _get_library(a_fp8.device)
    stream = torch.cuda.current_stream(a_fp8.device).cuda_stream
    status = lib.cc_cudnn_fp8_linear(
        rows,
        k,
        n,
        a_fp8.data_ptr(),
        w_fp8_t.data_ptr(),
        comb_scale.data_ptr(),
        bias_bf16.data_ptr() if bias_bf16 is not None else None,
        out_bf16.data_ptr(),
        stream,
    )
    if status != 0:
        raise RuntimeError(f"cuDNN FP8 linear execution failed with status {status}")


def cudnn_fp4_bias_gelu(
    a_packed: torch.Tensor,
    a_blocked: torch.Tensor,
    w_packed: torch.Tensor,
    w_blocked: torch.Tensor,
    comb_scale: torch.Tensor,
    bias_bf16: torch.Tensor,
    inv_out_scale: torch.Tensor,
    out_packed: torch.Tensor,
    out_blocked: torch.Tensor,
) -> None:
    """Launch the single-node cuDNN 9.20 NVFP4 Matmul + Bias + Tanh-GELU + Quantize -> NVFP4 graph."""
    rows = a_packed.shape[0]
    k = a_packed.shape[1] * 2
    n = w_packed.shape[0]
    lib = _get_library(a_packed.device)
    stream = torch.cuda.current_stream(a_packed.device).cuda_stream
    status = lib.cc_cudnn_fp4_bias_gelu(
        rows,
        k,
        n,
        a_packed.data_ptr(),
        a_blocked.data_ptr(),
        w_packed.data_ptr(),
        w_blocked.data_ptr(),
        comb_scale.data_ptr(),
        bias_bf16.data_ptr(),
        inv_out_scale.data_ptr(),
        out_packed.data_ptr(),
        out_blocked.data_ptr(),
        stream,
    )
    if status != 0:
        raise RuntimeError(f"cuDNN FP4 bias+GELU execution failed with status {status}")


def cudnn_packed_geglu_fp4(
    a_packed: torch.Tensor,
    a_blocked: torch.Tensor,
    w_packed: torch.Tensor,
    wb_packed: torch.Tensor,
    g_scale: torch.Tensor,
    u_ratio: float,
    packed_bf16: torch.Tensor,
    out_packed: torch.Tensor,
    out_blocked: torch.Tensor,
    down_global: torch.Tensor,
    table: torch.Tensor,
) -> None:
    """Launch cuDNN packed [2*N, K] NVFP4 GEMM + lookup_fp4_packed_vec8 epilogue."""
    rows = a_packed.shape[0]
    k = a_packed.shape[1] * 2
    n = w_packed.shape[0] // 2
    lib = _get_library(a_packed.device)
    stream = torch.cuda.current_stream(a_packed.device).cuda_stream
    status = lib.cc_cudnn_packed_geglu_fp4(
        rows,
        k,
        n,
        a_packed.data_ptr(),
        a_blocked.data_ptr(),
        w_packed.data_ptr(),
        wb_packed.data_ptr(),
        g_scale.data_ptr(),
        u_ratio,
        packed_bf16.data_ptr(),
        out_packed.data_ptr(),
        out_blocked.data_ptr(),
        down_global.data_ptr(),
        table.data_ptr(),
        stream,
    )
    if status != 0:
        raise RuntimeError(f"cuDNN packed GeGLU FP4 execution failed with status {status}")


def l2_prefetch_fork(
    t0: torch.Tensor | None = None,
    t1: torch.Tensor | None = None,
    t2: torch.Tensor | None = None,
    t3: torch.Tensor | None = None,
    *,
    nctas: int = 2,
    device: torch.device,
) -> None:
    """Fork a non-blocking hardware L2 prefetch on the side stream."""
    lib = _get_library(device)
    stream = torch.cuda.current_stream(device).cuda_stream
    p0 = t0.data_ptr() if t0 is not None else None
    n0 = t0.numel() * t0.element_size() if t0 is not None else 0
    p1 = t1.data_ptr() if t1 is not None else None
    n1 = t1.numel() * t1.element_size() if t1 is not None else 0
    p2 = t2.data_ptr() if t2 is not None else None
    n2 = t2.numel() * t2.element_size() if t2 is not None else 0
    p3 = t3.data_ptr() if t3 is not None else None
    n3 = t3.numel() * t3.element_size() if t3 is not None else 0
    lib.cc_l2_prefetch_fork(p0, n0, p1, n1, p2, n2, p3, n3, nctas, stream)


def l2_prefetch_join(device: torch.device) -> None:
    """Join the side prefetch stream back onto the active stream once at graph end."""
    lib = _get_library(device)
    stream = torch.cuda.current_stream(device).cuda_stream
    lib.cc_l2_prefetch_join(stream)
