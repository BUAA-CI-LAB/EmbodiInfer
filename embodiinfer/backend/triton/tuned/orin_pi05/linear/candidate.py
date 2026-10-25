"""Fixed split-K BF16 projections for the two captured Orin layouts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_split_k(
    X,
    W,
    Partials,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Write disjoint FP32 output planes for contiguous, exhaustive K slices."""
    tl.static_assert(SPLIT_K == 2 or SPLIT_K == 8)
    tl.static_assert(K % (SPLIT_K * BLOCK_K) == 0)
    tl.static_assert(N % BLOCK_N == 0)

    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = N // BLOCK_N
    pid = tl.program_id(0)
    split = pid // (tiles_m * tiles_n)
    tile = pid % (tiles_m * tiles_n)
    tile_m = tile % tiles_m
    tile_n = tile // tiles_m

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = split * (K // SPLIT_K) + tl.arange(0, BLOCK_K)
    x_ptrs = X + rows[:, None] * K + ks[None, :]
    w_ptrs = W + cols[None, :] * K + ks[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # The fixed stage count pipelines this regular loop without full unrolling.
    for _ in range(K // (SPLIT_K * BLOCK_K)):
        a = tl.load(x_ptrs, mask=rows[:, None] < M, other=0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    offsets = split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(Partials + offsets, acc, mask=rows[:, None] < M)


@triton.jit
def _reduce_partials(
    Partials,
    Y,
    ELEMENTS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum two or eight FP32 planes in a fixed tree and round to BF16 once."""
    tl.static_assert(SPLIT_K == 2 or SPLIT_K == 8)

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < ELEMENTS
    p0 = tl.load(Partials + offsets, mask=valid, other=0)
    p1 = tl.load(Partials + ELEMENTS + offsets, mask=valid, other=0)
    if SPLIT_K == 8:
        p2 = tl.load(Partials + 2 * ELEMENTS + offsets, mask=valid, other=0)
        p3 = tl.load(Partials + 3 * ELEMENTS + offsets, mask=valid, other=0)
        p4 = tl.load(Partials + 4 * ELEMENTS + offsets, mask=valid, other=0)
        p5 = tl.load(Partials + 5 * ELEMENTS + offsets, mask=valid, other=0)
        p6 = tl.load(Partials + 6 * ELEMENTS + offsets, mask=valid, other=0)
        p7 = tl.load(Partials + 7 * ELEMENTS + offsets, mask=valid, other=0)
        total = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
    else:
        total = p0 + p1
    tl.store(Y + offsets, total.to(tl.bfloat16), mask=valid)


def run(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Return a fresh BF16 ``x0 @ x1.T`` for either captured batch-one layout.

    Supports contiguous, zero-offset CUDA BF16 shapes
    ``(1, 672, 16384)/(2048, 16384)`` and
    ``(1, 50, 4096)/(1024, 4096)``. The result has shape ``(1, M, N)``,
    strides ``(M*N, N, 1)``, and no alias with either read-only input.
    Unsupported metadata raises ValueError. Both kernels use the input
    device's current stream; the caller's device and stream are preserved.
    """
    if not isinstance(x0, torch.Tensor) or not isinstance(x1, torch.Tensor):
        raise ValueError("Both linear inputs must be tensors.")
    if x0.layout != torch.strided or x1.layout != torch.strided:
        raise ValueError("Only dense strided inputs are supported.")
    if x0.ndim != 3 or x1.ndim != 2:
        raise ValueError("Linear inputs must have ranks three and two.")
    if x0.dtype != torch.bfloat16 or x1.dtype != torch.bfloat16:
        raise ValueError("Both linear inputs must have dtype bfloat16.")
    if x0.device.type != "cuda" or x1.device != x0.device:
        raise ValueError("Both linear inputs must be on the same CUDA device.")

    if tuple(x0.shape) == (1, 672, 16384) and tuple(x1.shape) == (2048, 16384):
        m, n, k = 672, 2048, 16384
        block_m, block_n, block_k, split_k, warps = 128, 128, 64, 8, 8
    elif tuple(x0.shape) == (1, 50, 4096) and tuple(x1.shape) == (1024, 4096):
        m, n, k = 50, 1024, 4096
        block_m, block_n, block_k, split_k, warps = 64, 64, 128, 2, 4
    else:
        raise ValueError("Unsupported linear shapes; expected a captured workload.")

    if tuple(x0.stride()) != (m * k, k, 1) or tuple(x1.stride()) != (k, 1):
        raise ValueError("Linear input strides must match the captured layouts.")
    if x0.storage_offset() != 0 or x1.storage_offset() != 0:
        raise ValueError("Linear inputs must have zero storage offsets.")

    # Device context restores the caller's device without changing any stream.
    with torch.cuda.device(x0.device):
        output = torch.empty((1, m, n), dtype=torch.bfloat16, device=x0.device)
        partials = torch.empty((split_k, m, n), dtype=torch.float32, device=x0.device)
        grid = (split_k * triton.cdiv(m, block_m) * (n // block_n),)
        _linear_split_k[grid](
            x0,
            x1,
            partials,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            num_warps=warps,
            num_stages=2,
        )
        _reduce_partials[(triton.cdiv(m * n, 1024),)](
            partials,
            output,
            ELEMENTS=m * n,
            SPLIT_K=split_k,
            BLOCK=1024,
            num_warps=4,
            num_stages=1,
        )
    return output
