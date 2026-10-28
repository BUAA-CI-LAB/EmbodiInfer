"""Shape-specialized BF16 ``linear`` for the recorded pi05 projections.

The recorded operator is ``F.linear(x, weight, None)`` with contiguous BF16
``x`` of shape ``[1, M, K]`` and contiguous BF16 ``weight`` of shape
``[N, K]``.  The result is contiguous BF16 ``[1, M, N]``.  Recorded calls:

* ``M=672, K=16384, N=2048`` (36 weighted calls),
* ``M=712, K=16384, N=2048`` (36),
* ``M=50,  K=4096,  N=1024`` (720).

The incumbent already replaced the wide shapes with a ``256x128x64`` NT GEMM
using 16 warps and three stages.  That CTA needs 144 KiB of dynamic shared
memory, so only one CTA is resident per SM, and its grid of 48 CTAs occupies
four waves on this device's 14 SMs with only six CTAs in the last wave.

This version keeps the incumbent's tile, warps, stages, K reduction granularity
and tile visit order, and splits only the K loop in two.  The grid becomes
96 CTAs of half the duration: seven waves with 12 CTAs in the last one, i.e.
98 % useful wave slots instead of 86 %.  Each split writes an FP32 partial
``[SPLIT_K, M, N]``; a second kernel adds the two partials in split order and
rounds once to BF16.  This is a reassociation of the same BF16-products /
FP32-accumulation arithmetic, within the frozen FP64 tolerance contract.

The hot ``M=50`` call carries 720 of the 792 recorded calls.  Its production
cuBLAS kernel is already near the device memory roof, so it deliberately keeps
the unchanged ``F.linear(x, weight, None)`` path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn.functional import linear as _torch_linear

#: Rows per program.  Three tiles cover both recorded wide row counts.
BLOCK_M = 256
#: Columns per program.  There are sixteen N tiles for N = 2048.
BLOCK_N = 128
#: K tile.  Both operands are K-contiguous, giving 128 B per row.
BLOCK_K = 64
#: 256x128 accumulators over 512 threads = 64 FP32 registers per thread.
NUM_WARPS = 16
#: Three stages of (256 + 128) * 64 * 2 = 144 KiB of dynamic shared memory.
NUM_STAGES = 3
#: Split K into two exact 8192-element halves for K = 16384.
SPLIT_K = 2
#: Reduction tile.  Both recorded outputs contain an integral number of these.
REDUCE_BLOCK = 4096
#: Enough warps to saturate the small bandwidth-bound reduction.
REDUCE_WARPS = 8
#: Largest row count admitted by the specialized family (four M tiles).
MAX_SPECIALIZED_ROWS = 4 * BLOCK_M


@triton.jit
def _splitk_linear_kernel(
    x_ptr,
    w_ptr,
    partial_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Write one FP32 partial of ``x[0:M, :] @ weight[:, :].T``.

    Program order is split-plane first, then M, then N: all N tiles of one M row
    are adjacent so the CTAs resident on the 14 SMs share that row's streaming
    x window through L2.  ``N`` and ``K`` are constexpr and exactly
    tile-divisible, so neither axis needs a mask.  Rows ``M .. BLOCK_M-1`` of
    the final M tile load as zero and are never stored.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N
    programs_per_split = num_pid_m * num_pid_n

    pid_k = pid // programs_per_split
    local_pid = pid % programs_per_split
    pid_m = local_pid // num_pid_n
    pid_n = local_pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    k_start = pid_k * (K // SPLIT_K)
    x_ptrs = x_ptr + offs_m[:, None] * K + k_start + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + k_start + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in tl.range(0, K // SPLIT_K, BLOCK_K):
        # A is [M, K] row-major and the weight tile is [N, K] with K contiguous,
        # i.e. both operands are K-major, the native mma.sync row.col layout.
        a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    partial_ptrs = (
        partial_ptr
        + pid_k * M * N
        + offs_m[:, None] * N
        + offs_n[None, :]
    )
    tl.store(partial_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def _reduce_splitk_kernel(
    partial_ptr,
    out_ptr,
    total,
    BLOCK: tl.constexpr,
):
    """Add the two FP32 split partials in order and round once to BF16."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    first = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    second = tl.load(partial_ptr + total + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (first + second).to(out_ptr.dtype.element_ty), mask=mask)


def _use_wide_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the layout is one of the specialized wide BF16 projections.

    The two recorded wide shapes qualify.  The hot 50-row projection and every
    unexpected layout keep the production ``F.linear`` path.
    """
    if x.dim() < 2 or weight.dim() != 2:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if not x.is_cuda or not weight.is_cuda:
        return False
    if not x.is_contiguous() or not weight.is_contiguous():
        return False
    n, k = weight.shape
    if x.shape[-1] != k or k == 0:
        return False
    rows = x.numel() // k
    if rows <= 2 * BLOCK_N or rows > MAX_SPECIALIZED_ROWS:
        return False
    return n % BLOCK_N == 0 and k % (SPLIT_K * BLOCK_K) == 0


def run(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Registry ``linear``: ``x0 @ x1.T`` in ``x0.dtype`` with recorded layouts."""
    if not _use_wide_kernel(x0, x1):
        return _torch_linear(x0, x1, None)

    k = x1.shape[1]
    n = x1.shape[0]
    rows = x0.numel() // k

    partial = torch.empty((SPLIT_K, rows, n), dtype=torch.float32, device=x0.device)
    out = torch.empty((rows, n), dtype=x0.dtype, device=x0.device)

    grid = (SPLIT_K * triton.cdiv(rows, BLOCK_M) * (n // BLOCK_N),)
    _splitk_linear_kernel[grid](
        x0,
        x1,
        partial,
        rows,
        n,
        k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    total = rows * n
    _reduce_splitk_kernel[(triton.cdiv(total, REDUCE_BLOCK),)](
        partial,
        out,
        total,
        BLOCK=REDUCE_BLOCK,
        num_warps=REDUCE_WARPS,
    )
    return out.view(*x0.shape[:-1], n)


__all__ = ["run"]
