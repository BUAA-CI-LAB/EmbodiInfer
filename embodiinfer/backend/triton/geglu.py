"""Paired BF16 GEMMs fused with GEGLU and optional device-local lookup."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ...layers.launch import GemmTile
from ...layers.registry import OperatorCapabilities


@triton.jit
def _dual_geglu(
    inputs,
    gate_weight,
    up_weight,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + inner
        values = tl.load(
            inputs + row[:, None] * K + k[None, :], (row[:, None] < M) & (k[None, :] < K), other=0
        )
        g = tl.load(
            gate_weight + k[:, None] * N + column[None, :],
            (k[:, None] < K) & (column[None, :] < N),
            other=0,
        )
        u = tl.load(
            up_weight + k[:, None] * N + column[None, :],
            (k[:, None] < K) & (column[None, :] < N),
            other=0,
        )
        gate = tl.dot(values, g, gate)
        up = tl.dot(values, u, up)
    # Keep the reference BF16 boundaries at each GEMM, GELU and product.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    activated = 0.5 * gate * (1.0 + libdevice.erf(gate * 0.7071067811865476))
    hidden = activated.to(tl.bfloat16).to(tl.float32) * up
    tl.store(
        output + row[:, None] * N + column[None, :],
        hidden.to(tl.bfloat16),
        (row[:, None] < M) & (column[None, :] < N),
    )


def dual_geglu(inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, tile: GemmTile) -> torch.Tensor:
    """Compute paired projections and GEGLU without materializing gate/up outputs.

    This kernel retains BF16 conversion points but uses a different GEMM
    reduction schedule. Callers must measure numerical drift and task quality
    before deployment. Only contiguous CUDA BF16 [M,K] and [K,N] are supported.
    """
    if inputs.ndim != 2 or gate.ndim != 2 or gate.shape != up.shape:
        raise ValueError("Expected input [M,K] and equally shaped gate/up [K,N]")
    if inputs.shape[1] != gate.shape[0]:
        raise ValueError("Projection dimensions differ")
    if any(
        value.dtype != torch.bfloat16
        or not value.is_cuda
        or not value.is_contiguous()
        or value.device != inputs.device
        for value in (inputs, gate, up)
    ):
        raise ValueError("Expected contiguous BF16 tensors on one CUDA device")
    rows, width = inputs.shape
    if rows > 16 and tile.m == 16:
        tile = GemmTile(64, tile.n, tile.k, tile.warps, tile.stages)
    columns = gate.shape[1]
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=inputs.device)
    _dual_geglu[(triton.cdiv(rows, tile.m), triton.cdiv(columns, tile.n))](
        inputs,
        gate,
        up,
        output,
        rows,
        columns,
        width,
        tile.m,
        tile.n,
        tile.k,
        num_warps=tile.warps,
        num_stages=tile.stages,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _packed_geglu_lut(
    packed,
    table,
    output,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    base = packed + row * (2 * N)
    gate = tl.load(base + cols, mask=mask, other=0.0)
    up = tl.load(base + N + cols, mask=mask, other=0.0)
    gate_bits = gate.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    activated = tl.load(table + gate_bits.to(tl.int32), mask=mask, other=0).to(tl.bfloat16, bitcast=True)
    hidden = activated.to(tl.float32) * up.to(tl.bfloat16).to(tl.float32)
    tl.store(output + row * N + cols, hidden.to(tl.bfloat16), mask=mask)


def packed_geglu_lut(packed: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Apply exact table-lookup GeGLU to a contiguous [M, 2*N] packed projection."""
    rows, total = packed.shape
    inter = total // 2
    output = torch.empty((rows, inter), device=packed.device, dtype=packed.dtype)
    block = 1024
    _packed_geglu_lut[(rows, triton.cdiv(inter, block))](
        packed, table, output, N=inter, BLOCK=block, num_warps=4, enable_fp_fusion=False
    )
    return output


@triton.jit
def _dual_geglu_lut(
    inputs,
    gate_weight,
    up_weight,
    table,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + inner
        values = tl.load(
            inputs + row[:, None] * K + k[None, :], (row[:, None] < M) & (k[None, :] < K), other=0
        )
        g = tl.load(
            gate_weight + k[:, None] * N + column[None, :],
            (k[:, None] < K) & (column[None, :] < N),
            other=0,
        )
        u = tl.load(
            up_weight + k[:, None] * N + column[None, :],
            (k[:, None] < K) & (column[None, :] < N),
            other=0,
        )
        gate = tl.dot(values, g, gate)
        up = tl.dot(values, u, up)
    # Keep the reference BF16 boundaries at each GEMM, GELU and product.
    gate_bits = gate.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    activated = tl.load(table + gate_bits.to(tl.int32)).to(tl.bfloat16, bitcast=True)
    hidden = activated.to(tl.float32) * up.to(tl.bfloat16).to(tl.float32)
    tl.store(
        output + row[:, None] * N + column[None, :],
        hidden.to(tl.bfloat16),
        (row[:, None] < M) & (column[None, :] < N),
    )


def lookup_geglu(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    table: torch.Tensor,
    tile: GemmTile,
) -> torch.Tensor:
    """Fuse paired projections with the original CUDA GELU rounding table.

    Args:
        inputs: Contiguous CUDA BF16 [M,K] normalized activations.
        gate: Contiguous CUDA BF16 [K,N] gate weight.
        up: Contiguous CUDA BF16 [K,N] up weight.
        table: Contiguous CUDA uint16 [65536] table generated on this device.
        tile: Fixed launch configuration chosen before capture.

    Returns:
        BF16 [M,N] activations. The table preserves GELU for each rounded gate;
        GEMM reduction may still differ from native GEMMs.
    """
    if inputs.ndim != 2 or gate.ndim != 2 or gate.shape != up.shape:
        raise ValueError("Expected input [M,K] and gate/up [K,N]")
    if inputs.shape[1] != gate.shape[0] or any(
        not value.is_cuda
        or not value.is_contiguous()
        or value.dtype != torch.bfloat16
        or value.device != inputs.device
        for value in (inputs, gate, up)
    ):
        raise ValueError("Require matching contiguous CUDA BF16 matrices")
    if (
        table.shape != (65536,)
        or table.dtype != torch.uint16
        or table.device != inputs.device
        or not table.is_contiguous()
    ):
        raise ValueError("Require the device-local CUDA GELU table")
    rows, width = inputs.shape
    columns = gate.shape[1]
    output = torch.empty((rows, columns), device=inputs.device, dtype=inputs.dtype)
    _dual_geglu_lut[(triton.cdiv(rows, tile.m), triton.cdiv(columns, tile.n))](
        inputs,
        gate,
        up,
        table,
        output,
        rows,
        columns,
        width,
        tile.m,
        tile.n,
        tile.k,
        num_warps=tile.warps,
        num_stages=tile.stages,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _tail_geglu_lut(
    inputs,
    gate,
    up,
    table,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TM: tl.constexpr,
):
    full: tl.constexpr = M // BM * BM
    if tl.program_id(0) < full // BM:
        _dual_geglu_lut(inputs, gate, up, table, output, M, N, K, BM, BN, BK)
    else:
        # Preserve the callee's row program id while moving its base to the tail.
        shift: tl.constexpr = full - (full // BM) * TM
        tail_limit: tl.constexpr = (full // BM) * TM + M - full
        _dual_geglu_lut(
            inputs + shift * K,
            gate,
            up,
            table,
            output + shift * N,
            tail_limit,
            N,
            K,
            TM,
            BN,
            BK,
        )


def prefix_geglu(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    table: torch.Tensor,
    tile: GemmTile,
    tail_m: int | None = None,
) -> torch.Tensor:
    """Use one prefix launch, shrinking the last row tile when it covers the tail."""
    remainder = inputs.shape[0] % tile.m
    if tail_m is not None and 0 < remainder <= tail_m:
        if remainder <= 16:
            tail_m = 16
        elif remainder <= 32:
            tail_m = min(tail_m, 32)
    if tail_m is None or remainder == 0 or remainder > tail_m:
        return lookup_geglu(inputs, gate, up, table, tile)
    if tail_m < 16 or tail_m >= tile.m or tail_m & (tail_m - 1):
        raise ValueError("Prefix tail rows must be a power of two smaller than the full tile")
    if inputs.ndim != 2 or gate.ndim != 2 or gate.shape != up.shape:
        raise ValueError("Expected input [M,K] and gate/up [K,N]")
    if inputs.shape[1] != gate.shape[0] or any(
        not value.is_cuda
        or not value.is_contiguous()
        or value.dtype != torch.bfloat16
        or value.device != inputs.device
        for value in (inputs, gate, up)
    ):
        raise ValueError("Require matching contiguous CUDA BF16 matrices")
    if table.shape != (65536,) or table.dtype != torch.uint16 or table.device != inputs.device:
        raise ValueError("Require the device-local GELU table")
    rows, width = inputs.shape
    columns = gate.shape[1]
    output = torch.empty((rows, columns), dtype=inputs.dtype, device=inputs.device)
    _tail_geglu_lut[(triton.cdiv(rows, tile.m), triton.cdiv(columns, tile.n))](
        inputs,
        gate,
        up,
        table,
        output,
        rows,
        columns,
        width,
        tile.m,
        tile.n,
        tile.k,
        tail_m,
        num_warps=tile.warps,
        num_stages=tile.stages,
        enable_fp_fusion=False,
    )
    return output


class PairedGelu:
    """Prepared Triton lookup projections with explicit launch and GELU semantics."""

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "paired_gemm_bf16_lookup", rank=2, minimum_sm=(8, 0)
    )

    def __init__(
        self,
        *,
        tile: GemmTile | None = None,
        tail_m: int | None = None,
        approximate: str = "tanh",
        profile: tuple[int, int] | None = None,
        large_m: bool = False,
    ) -> None:
        """Prepare a device-local rounding table before graph capture."""
        if tile is None:
            if profile != torch.cuda.get_device_capability():
                raise ValueError("Paired GELU launch profile must match the current CUDA device")
            if profile == (11, 0):
                tile = GemmTile(128, 128, 64, 8, 4) if large_m else GemmTile(16, 64, 64, 4, 3)
                tail_m = 64 if large_m else None
            elif profile == (12, 1):
                tile = GemmTile(64, 64, 64, 4, 3) if large_m else GemmTile(16, 32, 32, 4, 3)
            elif profile == (8, 9):
                tile = GemmTile(64, 64, 64, 8, 3) if large_m else GemmTile(16, 64, 64, 4, 3)
                tail_m = 16 if large_m else None
                self.use_native_large = large_m
                self.fast_sm89 = not large_m
            else:
                raise ValueError("No measured paired GELU launch profile; supply an explicit tile")
        if approximate not in ("none", "tanh"):
            raise ValueError("GELU approximation must be none or tanh")
        if tail_m is not None and (tail_m < 16 or tail_m >= tile.m or tail_m & (tail_m - 1)):
            raise ValueError("Prefix tail rows must be a power of two smaller than the full tile")
        self.tile, self.tail_m = tile, tail_m
        self.arithmetic = False
        codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
        self.table = torch.nn.functional.gelu(codes.view(torch.bfloat16), approximate=approximate).view(
            torch.uint16
        )

    def plan(self, gate_weight: torch.Tensor, up_weight: torch.Tensor) -> "PairedGeluPlan":
        """Convert original [output,input] weights to the kernel's [input,output] layout."""
        return PairedGeluPlan(self, gate_weight, up_weight)


class PairedGeluPlan:
    """Immutable kernel weights; invocation returns an independent BF16 activation."""

    def __init__(self, backend: PairedGelu, gate: torch.Tensor, up: torch.Tensor) -> None:
        """Pack matching BF16 projection weights without changing Parameters."""
        if (
            gate.ndim != 2
            or gate.shape != up.shape
            or gate.device != backend.table.device
            or up.device != gate.device
            or any(value.dtype != torch.bfloat16 for value in (gate, up))
        ):
            raise ValueError("Paired projections require matching CUDA BF16 [output,input] weights")
        self.backend = backend
        if getattr(backend, "use_native_large", False):
            self.packed_w = torch.cat((gate.detach(), up.detach()), dim=0).contiguous()
            self.gate = self.up = None
        elif getattr(backend, "fast_sm89", False):
            self.packed_w = None
            self.gate, self.up = (value.detach().contiguous() for value in (gate, up))
        else:
            self.packed_w = None
            self.gate, self.up = (value.detach().t().contiguous() for value in (gate, up))

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use the prepared tile and optional smaller tail for the current matrix."""
        if self.packed_w is not None:
            return packed_geglu_lut(torch.nn.functional.linear(inputs, self.packed_w), self.backend.table)
        if getattr(self.backend, "fast_sm89", False):
            return dual_geglu_fast(inputs, self.gate, self.up, self.backend.tile)
        if self.backend.arithmetic:
            return dual_geglu(inputs, self.gate, self.up, self.backend.tile)
        return prefix_geglu(
            inputs, self.gate, self.up, self.backend.table, self.backend.tile, self.backend.tail_m
        )


class ExactPairedGelu(PairedGelu):
    """Exact arithmetic for small M, with the table/tail kernel for large M."""

    def __init__(
        self,
        *,
        approximate: str = "none",
        large_m: bool = False,
        tile: GemmTile | None = None,
        tail_m: int | None = None,
        profile: tuple[int, int] | None = None,
    ) -> None:
        """Select the original arithmetic/table split before preparing weights."""
        if approximate != "none":
            raise ValueError("Exact paired GEMMs require exact GELU")
        super().__init__(approximate=approximate, large_m=large_m, tile=tile, tail_m=tail_m, profile=profile)
        self.arithmetic = not large_m


@triton.jit
def _linear_bf16_nk(X, W, Y, M, N, K, stride_x, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tiles_m = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    rows = (pid % tiles_m) * BM + tl.arange(0, BM)
    cols = (pid // tiles_m) * BN + tl.arange(0, BN)
    depth = tl.arange(0, BK)
    x = X + rows[:, None] * stride_x + depth[None, :]
    w = W + cols[None, :] * K + depth[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x, mask=rows[:, None] < M, other=0.0)
        b = tl.load(w, mask=cols[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc)
        x += BK
        w += BK
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        acc.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _gelu_gemm_bf16_nk(X, Wg, Wu, Y, M, N, K, stride_x, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tiles_m = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    rows = (pid % tiles_m) * BM + tl.arange(0, BM)
    cols = (pid // tiles_m) * BN + tl.arange(0, BN)
    depth = tl.arange(0, BK)
    x = X + rows[:, None] * stride_x + depth[None, :]
    offsets = cols[None, :] * K + depth[:, None]
    gate = tl.zeros((BM, BN), dtype=tl.float32)
    up = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        a = tl.load(x, mask=rows[:, None] < M, other=0.0)
        gate = tl.dot(a, tl.load(Wg + offsets + start), gate)
        up = tl.dot(a, tl.load(Wu + offsets + start), up)
        x += BK
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    act = (g * tl.sigmoid(2.0 * inner)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + rows[:, None] * N + cols[None, :], (act * u).to(tl.bfloat16), mask=rows[:, None] < M)


@triton.jit
def _linear_fp8_nk(
    X, W, Y, ScaleX, ScaleW, M, N, K, stride_x, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    tiles_m = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    rows = (pid % tiles_m) * BM + tl.arange(0, BM)
    cols = (pid // tiles_m) * BN + tl.arange(0, BN)
    depth = tl.arange(0, BK)
    x_ptrs = X + rows[:, None] * stride_x + depth[None, :]
    w_ptrs = W + cols[None, :] * K + depth[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x_ptrs, mask=rows[:, None] < M, other=0.0)
        b = tl.load(w_ptrs, mask=cols[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        x_ptrs += BK
        w_ptrs += BK
    scale = tl.load(ScaleX) * tl.load(ScaleW)
    y = (acc * scale).to(tl.bfloat16)
    tl.store(Y + rows[:, None] * N + cols[None, :], y, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _gelu_gemm_fp8_nk(
    X,
    Wg,
    Wu,
    Y,
    ScaleX,
    ScaleG,
    ScaleU,
    InvScaleOut,
    M,
    N,
    K,
    stride_x,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tiles_m = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    rows = (pid % tiles_m) * BM + tl.arange(0, BM)
    cols = (pid // tiles_m) * BN + tl.arange(0, BN)
    depth = tl.arange(0, BK)
    x_ptrs = X + rows[:, None] * stride_x + depth[None, :]
    offsets = cols[None, :] * K + depth[:, None]
    gate = tl.zeros((BM, BN), dtype=tl.float32)
    up = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        a = tl.load(x_ptrs, mask=rows[:, None] < M, other=0.0)
        bg = tl.load(Wg + offsets + start, mask=cols[None, :] < N, other=0.0)
        bu = tl.load(Wu + offsets + start, mask=cols[None, :] < N, other=0.0)
        gate = tl.dot(a, bg, gate, out_dtype=tl.float32)
        up = tl.dot(a, bu, up, out_dtype=tl.float32)
        x_ptrs += BK
    sx = tl.load(ScaleX)
    sg = tl.load(ScaleG)
    su = tl.load(ScaleU)
    g = (gate * (sx * sg)).to(tl.bfloat16).to(tl.float32)
    u = (up * (sx * su)).to(tl.bfloat16).to(tl.float32)
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    act = (g * tl.sigmoid(2.0 * inner)).to(tl.bfloat16).to(tl.float32)
    val = (act * u).to(tl.bfloat16).to(tl.float32)
    inv_so = tl.load(InvScaleOut)
    q = tl.clamp(val * inv_so, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Y + rows[:, None] * N + cols[None, :], q, mask=rows[:, None] < M)


def dual_geglu_fast(
    inputs: torch.Tensor, gate_nk: torch.Tensor, up_nk: torch.Tensor, tile: GemmTile
) -> torch.Tensor:
    """Small-M BF16 paired gate/up + GeGLU using [N, K] coalesced weight streaming."""
    rows, width = inputs.shape
    columns = gate_nk.shape[0]
    output = torch.empty((rows, columns), device=inputs.device, dtype=inputs.dtype)
    bm, bn, bk, warps, stages = (16, 32, 64, 4, 3) if rows <= 16 else (64, 16, 64, 4, 3)
    _gelu_gemm_bf16_nk[(triton.cdiv(rows, bm) * triton.cdiv(columns, bn),)](
        inputs,
        gate_nk,
        up_nk,
        output,
        rows,
        columns,
        width,
        inputs.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return output


def triton_linear(inputs: torch.Tensor, weight_nk: torch.Tensor) -> torch.Tensor:
    """Small-M BF16 projection using [N, K] coalesced weight-streaming Triton tiles."""
    rows, width = inputs.shape
    columns = weight_nk.shape[0]
    out = torch.empty((rows, columns), device=inputs.device, dtype=torch.bfloat16)
    if rows <= 16:
        if width <= 1024:
            bm, bn, bk, warps, stages = 16, 16, 64, 4, 4
        elif width <= 2048:
            bm, bn, bk, warps, stages = 16, 32, 128, 4, 5
        else:
            bm, bn, bk, warps, stages = 16, 32, 256, 4, 4
    else:
        bm, bn, bk = 32, 32, 128
        warps, stages = (4, 3) if width <= 1024 else (8, 5)
    _linear_bf16_nk[(triton.cdiv(rows, bm) * triton.cdiv(columns, bn),)](
        inputs,
        weight_nk,
        out,
        rows,
        columns,
        width,
        inputs.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def triton_linear_fp8(
    inputs_fp8: torch.Tensor,
    weight_nk_fp8: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
) -> torch.Tensor:
    """Small-M FP8 projection using [N, K] coalesced weight-streaming Triton tiles."""
    rows, width = inputs_fp8.shape
    columns = weight_nk_fp8.shape[0]
    out = torch.empty((rows, columns), device=inputs_fp8.device, dtype=torch.bfloat16)
    if rows <= 16:
        # The expert's 1024-wide projections read L2-prefetched weights; 16-column tiles give them
        # 64 CTAs instead of 32. The K loop, and with it every output's rounding, is unchanged.
        bm, bn, bk, warps = 16, 16 if columns <= 1024 else 32, 256, 4
        stages = 4 if width <= 1024 else 5
    else:
        bm, bn = 32, 32
        bk = 128 if width <= 1024 else 256
        warps = 4 if width <= 1024 else 8
        stages = 4 if width <= 2048 else 5
    _linear_fp8_nk[(triton.cdiv(rows, bm) * triton.cdiv(columns, bn),)](
        inputs_fp8,
        weight_nk_fp8,
        out,
        scale_x,
        scale_w,
        rows,
        columns,
        width,
        inputs_fp8.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _qkv_rope_fp8_nk(
    X,
    W,
    ScaleX,
    ScaleW,
    Cos,
    Sin,
    Q,
    Kc,
    Vc,
    M,
    K,
    stride_x,
    kv_rows,
    kv_offset,
    q_scale,
    T: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    # Each program owns 16 rotary pairs of one Q/K head (columns c and c + D/2) or 32 V columns, so the
    # rotation happens in registers. Every column keeps _linear_fp8_nk's K loop and BF16 rounding.
    HALF: tl.constexpr = D // 2
    CHUNKS: tl.constexpr = D // 32
    ROTARY: tl.constexpr = (QH + KVH) * CHUNKS
    tiles_m = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    group = pid // tiles_m
    rows = (pid % tiles_m) * BM + tl.arange(0, BM)
    pair = tl.arange(0, 16)
    head = group // CHUNKS
    chunk = group % CHUNKS
    if group < ROTARY:
        offset = chunk * 16
        lo = head * D + offset + pair
        hi = lo + HALF
    else:
        offset = chunk * 32
        lo = head * D + offset + pair
        hi = lo + 16
    depth = tl.arange(0, BK)
    x_ptrs = X + rows[:, None] * stride_x + depth[None, :]
    lo_ptrs = W + lo[None, :] * K + depth[:, None]
    hi_ptrs = W + hi[None, :] * K + depth[:, None]
    acc_lo = tl.zeros((BM, 16), dtype=tl.float32)
    acc_hi = tl.zeros((BM, 16), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x_ptrs, mask=rows[:, None] < M, other=0.0)
        acc_lo = tl.dot(a, tl.load(lo_ptrs), acc_lo, out_dtype=tl.float32)
        acc_hi = tl.dot(a, tl.load(hi_ptrs), acc_hi, out_dtype=tl.float32)
        x_ptrs += BK
        lo_ptrs += BK
        hi_ptrs += BK
    scale = tl.load(ScaleX) * tl.load(ScaleW)
    first = (acc_lo * scale).to(tl.bfloat16)
    second = (acc_hi * scale).to(tl.bfloat16)
    live = rows[:, None] < M
    step = rows % T
    batch = rows // T
    columns = offset + pair
    if group < ROTARY:
        # split_qkv_rope_vec4 is built without FMA contraction: both products round before the sum.
        cosine = tl.load(Cos + rows[:, None] * HALF + columns[None, :], mask=live, other=0.0)
        sine = tl.load(Sin + rows[:, None] * HALF + columns[None, :], mask=live, other=0.0)
        x1 = first.to(tl.float32)
        x2 = second.to(tl.float32)
        first = libdevice.sub_rn(libdevice.mul_rn(x1, cosine), libdevice.mul_rn(x2, sine)).to(tl.bfloat16)
        second = libdevice.add_rn(libdevice.mul_rn(x2, cosine), libdevice.mul_rn(x1, sine)).to(tl.bfloat16)
        if head < QH:
            first = libdevice.mul_rn(first.to(tl.float32), q_scale).to(tl.bfloat16)
            second = libdevice.mul_rn(second.to(tl.float32), q_scale).to(tl.bfloat16)
            destination = Q + ((batch * QH + head) * T + step)[:, None] * D + columns[None, :]
        else:
            destination = (
                Kc + ((batch * KVH + head - QH) * kv_rows + kv_offset + step)[:, None] * D + columns[None, :]
            )
        tl.store(destination, first, mask=live)
        tl.store(destination + HALF, second, mask=live)
    else:
        destination = (
            Vc
            + ((batch * KVH + head - QH - KVH) * kv_rows + kv_offset + step)[:, None] * D
            + columns[None, :]
        )
        tl.store(destination, first, mask=live)
        tl.store(destination + 16, second, mask=live)


def triton_qkv_rope_fp8(
    inputs_fp8: torch.Tensor,
    weight_nk_fp8: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kv_offset: int,
    q_scale: float,
) -> None:
    """Small-M packed FP8 QKV projection with split_qkv_rope_vec4's rotation, written to BHSD Q/K/V.

    q is [B, QH, T, D]; k and v are [B, KVH, rows, D] with this step's tokens at kv_offset. Outputs
    match triton_linear_fp8 followed by split_qkv_rope_vec4 bit for bit.
    """
    rows, width = inputs_fp8.shape
    batch, q_heads, tokens, head_dim = q.shape
    kv_heads = k.shape[1]
    if (
        rows > 16
        or rows != batch * tokens
        or head_dim % 32
        or weight_nk_fp8.shape != ((q_heads + 2 * kv_heads) * head_dim, width)
        or not (
            weight_nk_fp8.is_contiguous() and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        )
        or k.shape != v.shape
        or cos.numel() != rows * head_dim // 2
        or sin.shape != cos.shape
        or cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or not (cos.is_contiguous() and sin.is_contiguous())
    ):
        raise ValueError(
            "Fused QKV RoPE needs at most 16 rows, contiguous BHSD outputs and [rows, D/2] FP32 tables"
        )
    groups = (q_heads + 2 * kv_heads) * head_dim // 32
    _qkv_rope_fp8_nk[(triton.cdiv(rows, 16) * groups,)](
        inputs_fp8,
        weight_nk_fp8,
        scale_x,
        scale_w,
        cos,
        sin,
        q,
        k,
        v,
        rows,
        width,
        inputs_fp8.stride(0),
        k.shape[2],
        kv_offset,
        q_scale,
        T=tokens,
        QH=q_heads,
        KVH=kv_heads,
        D=head_dim,
        BM=16,
        BK=256,
        num_warps=4,
        num_stages=4 if width <= 1024 else 5,
    )


def dual_geglu_fp8(
    x_fp8: torch.Tensor,
    g_nk_fp8: torch.Tensor,
    u_nk_fp8: torch.Tensor,
    scale_x: torch.Tensor,
    scale_g: torch.Tensor,
    scale_u: torch.Tensor,
    inv_out: torch.Tensor,
    out_fp8: torch.Tensor,
    rows: int,
) -> torch.Tensor:
    """Small-M FP8 paired gate/up + GeGLU + FP8 quantization using [N, K] coalesced tiles."""
    width = x_fp8.shape[1]
    columns = g_nk_fp8.shape[0]
    bm, bn, bk, warps, stages = (
        ((16, 64, 128, 4, 3) if x_fp8.shape[1] == 1024 else (16, 32, 64, 4, 3))
        if rows <= 16
        else (64, 64, 64, 4, 3)
    )
    _gelu_gemm_fp8_nk[(triton.cdiv(rows, bm) * triton.cdiv(columns, bn),)](
        x_fp8,
        g_nk_fp8,
        u_nk_fp8,
        out_fp8,
        scale_x,
        scale_g,
        scale_u,
        inv_out,
        rows,
        columns,
        width,
        x_fp8.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out_fp8


@triton.jit
def _dual_geglu_fp8_lut_kn(
    inputs,
    gate_weight,
    up_weight,
    table,
    output,
    ScaleX,
    ScaleG,
    ScaleU,
    InvScaleOut,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + inner
        a = tl.load(inputs + row[:, None] * K + k[None, :], (row[:, None] < M) & (k[None, :] < K), other=0.0)
        g = tl.load(
            gate_weight + k[:, None] * N + col[None, :], (k[:, None] < K) & (col[None, :] < N), other=0.0
        )
        u = tl.load(
            up_weight + k[:, None] * N + col[None, :], (k[:, None] < K) & (col[None, :] < N), other=0.0
        )
        gate = tl.dot(a, g, gate, out_dtype=tl.float32)
        up = tl.dot(a, u, up, out_dtype=tl.float32)
    sx = tl.load(ScaleX)
    sg = tl.load(ScaleG)
    su = tl.load(ScaleU)
    g_bf16 = (gate * (sx * sg)).to(tl.bfloat16)
    u_fp32 = (up * (sx * su)).to(tl.bfloat16).to(tl.float32)
    gate_bits = g_bf16.to(tl.uint16, bitcast=True)
    activated = tl.load(table + gate_bits.to(tl.int32)).to(tl.bfloat16, bitcast=True)
    hidden = (activated.to(tl.float32) * u_fp32).to(tl.bfloat16).to(tl.float32)
    inv_so = tl.load(InvScaleOut)
    q = tl.clamp(hidden * inv_so, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(output + row[:, None] * N + col[None, :], q, (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _tail_geglu_fp8_lut_kn(
    inputs,
    gate,
    up,
    table,
    output,
    ScaleX,
    ScaleG,
    ScaleU,
    InvScaleOut,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TM: tl.constexpr,
):
    full: tl.constexpr = M // BM * BM
    if tl.program_id(0) < full // BM:
        _dual_geglu_fp8_lut_kn(
            inputs, gate, up, table, output, ScaleX, ScaleG, ScaleU, InvScaleOut, M, N, K, BM, BN, BK
        )
    else:
        shift: tl.constexpr = full - (full // BM) * TM
        tail_limit: tl.constexpr = (full // BM) * TM + M - full
        _dual_geglu_fp8_lut_kn(
            inputs + shift * K,
            gate,
            up,
            table,
            output + shift * N,
            ScaleX,
            ScaleG,
            ScaleU,
            InvScaleOut,
            tail_limit,
            N,
            K,
            TM,
            BN,
            BK,
        )


def prefix_geglu_fp8(
    x_fp8: torch.Tensor,
    g_kn_fp8: torch.Tensor,
    u_kn_fp8: torch.Tensor,
    table: torch.Tensor,
    scale_x: torch.Tensor,
    scale_g: torch.Tensor,
    scale_u: torch.Tensor,
    inv_out: torch.Tensor,
    out_fp8: torch.Tensor,
    rows: int,
) -> torch.Tensor:
    # Large-M FP8 paired gate/up + lookup GeGLU + FP8 quantization using [K, N] tiles.
    width, columns = g_kn_fp8.shape
    bm, bn, bk, warps, stages = 128, 256, 64, 8, 4
    remainder = rows % bm
    if 32 < remainder <= 64:
        blocks = rows // bm + 1
        _tail_geglu_fp8_lut_kn[(blocks, triton.cdiv(columns, bn))](
            x_fp8,
            g_kn_fp8,
            u_kn_fp8,
            table,
            out_fp8,
            scale_x,
            scale_g,
            scale_u,
            inv_out,
            rows,
            columns,
            width,
            BM=bm,
            BN=bn,
            BK=bk,
            TM=64,
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=False,
        )
    else:
        _dual_geglu_fp8_lut_kn[(triton.cdiv(rows, bm), triton.cdiv(columns, bn))](
            x_fp8,
            g_kn_fp8,
            u_kn_fp8,
            table,
            out_fp8,
            scale_x,
            scale_g,
            scale_u,
            inv_out,
            rows,
            columns,
            width,
            BM=bm,
            BN=bn,
            BK=bk,
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=False,
        )
    return out_fp8
