"""Weight-streaming linear layers for decode-sized row counts.

With a few dozen rows a projection only streams its weight. cuBLAS leaves
10-35% of RTX 4090 bandwidth unused there (split-K or wide tiles); these
tiles read the weight once at close to peak bandwidth. Every output
accumulates its full depth in FP32 in a fixed order, so results depend on
neither the tiles nor the row count, nor on whether projections share a
launch. They can differ from cuBLAS by output rounding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .rounded_ops import rounded_swiglu

try:
    import triton
    import triton.language as tl

    from .rounded_ops import _silu_fp32
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _linear(
        X,
        W,
        Bias,
        Y,
        M,
        N,
        K,
        stride_x,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_m, tiles_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
        first_n = pid // (GROUP * tiles_m) * GROUP
        size = tl.minimum(tiles_n - first_n, GROUP)
        tile_m = pid % (GROUP * tiles_m) // size
        tile_n = first_n + pid % (GROUP * tiles_m) % size
        rows = tile_m * BM + tl.arange(0, BM)
        columns = tile_n * BN + tl.arange(0, BN)
        depth = tl.arange(0, BK)
        x = X + rows[:, None] * stride_x + depth[None, :]
        w = W + columns[None, :] * K + depth[:, None]
        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(x, mask=rows[:, None] < M, other=0.0)
            b = tl.load(w, mask=columns[None, :] < N, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            x += BK
            w += BK
        if HAS_BIAS:
            accumulator += tl.load(Bias + columns, mask=columns < N, other=0.0).to(tl.float32)[None, :]
        tl.store(
            Y + rows[:, None] * N + columns[None, :],
            accumulator.to(Y.dtype.element_ty),
            mask=(rows[:, None] < M) & (columns[None, :] < N),
        )

    @triton.jit
    def _qkv(
        X,
        Wq,
        Wk,
        Wv,
        Bq,
        Bk,
        Bv,
        Yq,
        Yk,
        Yv,
        M,
        NQ,
        NKV,
        K,
        stride_x,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        tiles_m = tl.cdiv(M, BM)
        tile_m = tl.program_id(0) % tiles_m
        start = tl.program_id(0) // tiles_m * BN
        if start < NQ:
            W, B, Y, N, first = Wq, Bq, Yq, NQ, start
        elif start < NQ + NKV:
            W, B, Y, N, first = Wk, Bk, Yk, NKV, start - NQ
        else:
            W, B, Y, N, first = Wv, Bv, Yv, NKV, start - NQ - NKV
        rows = tile_m * BM + tl.arange(0, BM)
        columns = first + tl.arange(0, BN)
        depth = tl.arange(0, BK)
        x = X + rows[:, None] * stride_x + depth[None, :]
        w = W + columns[None, :] * K + depth[:, None]
        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(x, mask=rows[:, None] < M, other=0.0)
            accumulator = tl.dot(a, tl.load(w), accumulator)
            x += BK
            w += BK
        if HAS_BIAS:
            accumulator += tl.load(B + columns).to(tl.float32)[None, :]
        tl.store(
            Y + rows[:, None] * N + columns[None, :],
            accumulator.to(Y.dtype.element_ty),
            mask=rows[:, None] < M,
        )

    @triton.jit
    def _gated(X, Wg, Wu, Y, M, N, K, stride_x, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        tiles_m = tl.cdiv(M, BM)
        rows = tl.program_id(0) % tiles_m * BM + tl.arange(0, BM)
        columns = tl.program_id(0) // tiles_m * BN + tl.arange(0, BN)
        depth = tl.arange(0, BK)
        x = X + rows[:, None] * stride_x + depth[None, :]
        offsets = columns[None, :] * K + depth[:, None]
        gate = tl.zeros((BM, BN), dtype=tl.float32)
        up = tl.zeros((BM, BN), dtype=tl.float32)
        for start in range(0, K, BK):
            a = tl.load(x, mask=rows[:, None] < M, other=0.0)
            gate = tl.dot(a, tl.load(Wg + offsets + start), gate)
            up = tl.dot(a, tl.load(Wu + offsets + start), up)
            x += BK
        # rounded_swiglu of the two rounded projections.
        gate = gate.to(Y.dtype.element_ty).to(tl.float32)
        up = up.to(Y.dtype.element_ty).to(tl.float32)
        activated = _silu_fp32(gate).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(
            Y + rows[:, None] * N + columns[None, :],
            (activated * up).to(Y.dtype.element_ty),
            mask=rows[:, None] < M,
        )


# (most rows, tiles) tables, timed for every tile against cuBLAS at
# Qwen2.5-VL-3B text shapes on an RTX 4090 with L2-cold weights. Single
# projections take (BM, BN, BK, GROUP, warps, stages) per weight class;
# larger row counts and narrow outputs stay on cuBLAS. The grouped QKV and
# gated kernels take (BM, BN, BK, warps, stages).
_TILES = {
    "square": (
        (16, (16, 32, 128, 8, 4, 4)),
        (48, (16, 32, 128, 1, 4, 3)),
        (96, (32, 64, 128, 1, 8, 4)),
        (128, (32, 64, 64, 8, 8, 5)),
    ),
    "wide": (
        (16, (16, 32, 128, 1, 4, 4)),
        (48, (16, 64, 128, 8, 8, 4)),
        (128, (64, 64, 64, 8, 8, 3)),
    ),
    "deep": (
        (16, (16, 32, 128, 8, 4, 4)),
        (48, (32, 32, 128, 8, 8, 4)),
        (96, (32, 64, 128, 1, 8, 4)),
    ),
}
_QKV_TILES = ((48, (16, 32, 64, 4, 4)), (96, (32, 64, 64, 8, 5)), (192, (64, 64, 128, 8, 3)))
_GATED_TILES = ((48, (16, 32, 128, 4, 3)), (192, (64, 32, 64, 4, 3)))
_MIN_WIDTH = 1024


def decode_tiles(rows: int, width: int, depth: int) -> tuple[int, int, int, int, int, int] | None:
    """Tiles for ``rows`` inputs to a ``width`` x ``depth`` weight, or None to keep cuBLAS."""
    if rows < 1 or width < _MIN_WIDTH:
        return None
    kind = "square" if width == depth else "wide" if width > depth else "deep"
    for limit, tiles in _TILES[kind]:
        if rows <= limit:
            return tiles if depth % tiles[2] == 0 else None
    return None


def _rows_tiles(table, rows: int, widths: Sequence[int], depth: int) -> tuple[int, ...] | None:
    for limit, tiles in table:
        if rows <= limit:
            fits = depth % tiles[2] == 0 and all(width % tiles[1] == 0 for width in widths)
            return tiles if fits else None
    return None


def _streamable(inputs: torch.Tensor, *tensors: torch.Tensor | None) -> bool:
    """CUDA FP16/BF16 inference operands that the kernels can read in place."""
    present = [tensor for tensor in tensors if tensor is not None]
    return (
        triton is not None
        and inputs.is_cuda
        and inputs.dtype in (torch.float16, torch.bfloat16)
        and all(t.dtype == inputs.dtype and t.device == inputs.device and t.is_contiguous() for t in present)
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (inputs, *present)))
    )


def _rows(inputs: torch.Tensor, depth: int) -> torch.Tensor:
    flat = inputs.reshape(-1, depth)
    return flat if flat.stride(-1) == 1 else flat.contiguous()


def decode_linear(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``F.linear`` for inference, through Triton where ``decode_tiles`` beats cuBLAS.

    Triton takes CUDA FP16/BF16 inputs, a contiguous weight of the same dtype
    and an optional contiguous bias, which is added to the FP32 accumulator
    before the single output rounding, as in cuBLAS's bias epilogue.
    """
    depth = weight.shape[-1]
    rows = inputs.numel() // depth if inputs.shape[-1:] == (depth,) else 0
    tiles = decode_tiles(rows, weight.shape[0], depth) if weight.ndim == 2 else None
    if (
        tiles is None
        or not _streamable(inputs, weight, bias)
        or (bias is not None and bias.shape != weight.shape[:1])
    ):
        return F.linear(inputs, weight, bias)
    flat, width = _rows(inputs, depth), weight.shape[0]
    output = torch.empty((*inputs.shape[:-1], width), device=inputs.device, dtype=inputs.dtype)
    block_m, block_n, block_k, group, warps, stages = tiles
    _linear[(triton.cdiv(rows, block_m) * triton.cdiv(width, block_n),)](
        flat,
        weight,
        weight if bias is None else bias,
        output,
        rows,
        width,
        depth,
        flat.stride(0),
        HAS_BIAS=bias is not None,
        BM=block_m,
        BN=block_n,
        BK=block_k,
        GROUP=group,
        num_warps=warps,
        num_stages=stages,
    )
    return output


def decode_qkv(
    inputs: torch.Tensor,
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor | None],
) -> list[torch.Tensor]:
    """Query, key and value projections of the same inputs.

    Up to 192 rows one launch writes all three, so the narrow key and value
    projections stream alongside the query instead of running as two
    underfilled kernels. Each output equals ``decode_linear``'s Triton result.
    Other inputs take three ``decode_linear`` calls.
    """
    (query, key, value), depth = weights, weights[0].shape[-1]
    rows = inputs.numel() // depth if inputs.shape[-1:] == (depth,) else 0
    tiles = _rows_tiles(_QKV_TILES, rows, (query.shape[0], key.shape[0]), depth) if rows else None
    if (
        tiles is None
        or any(weight.ndim != 2 or weight.shape[1] != depth for weight in weights)
        or key.shape != value.shape
        or not _streamable(inputs, *weights, *biases)
        or len({bias is None for bias in biases}) != 1
        or any(bias is not None and bias.shape != weight.shape[:1] for weight, bias in zip(weights, biases))
    ):
        return [decode_linear(inputs, weight, bias) for weight, bias in zip(weights, biases, strict=True)]
    flat = _rows(inputs, depth)
    outputs = [
        torch.empty((*inputs.shape[:-1], weight.shape[0]), device=inputs.device, dtype=inputs.dtype)
        for weight in weights
    ]
    block_m, block_n, block_k, warps, stages = tiles
    has_bias = biases[0] is not None
    _qkv[(triton.cdiv(rows, block_m) * ((query.shape[0] + 2 * key.shape[0]) // block_n),)](
        flat,
        *weights,
        *(biases if has_bias else weights),
        *outputs,
        rows,
        query.shape[0],
        key.shape[0],
        depth,
        flat.stride(0),
        HAS_BIAS=has_bias,
        BM=block_m,
        BN=block_n,
        BK=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return outputs


def decode_swiglu(inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``rounded_swiglu`` of two bias-free projections of the same inputs.

    Up to 192 rows one kernel computes both projections and applies the
    activation to its tiles, equal to ``rounded_swiglu`` of the two Triton
    projections. Other inputs take two ``decode_linear`` calls.
    """
    depth = gate.shape[-1]
    rows = inputs.numel() // depth if inputs.shape[-1:] == (depth,) else 0
    tiles = _rows_tiles(_GATED_TILES, rows, gate.shape[:1], depth) if rows and gate.ndim == 2 else None
    if tiles is None or up.shape != gate.shape or not _streamable(inputs, gate, up):
        return rounded_swiglu(decode_linear(inputs, gate), decode_linear(inputs, up))
    flat, width = _rows(inputs, depth), gate.shape[0]
    output = torch.empty((*inputs.shape[:-1], width), device=inputs.device, dtype=inputs.dtype)
    block_m, block_n, block_k, warps, stages = tiles
    _gated[(triton.cdiv(rows, block_m) * (width // block_n),)](
        flat,
        gate,
        up,
        output,
        rows,
        width,
        depth,
        flat.stride(0),
        BM=block_m,
        BN=block_n,
        BK=block_k,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    return output


__all__ = ("decode_linear", "decode_qkv", "decode_swiglu", "decode_tiles")
