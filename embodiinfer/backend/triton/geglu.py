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
def _dual_geglu_fast(
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
    inner_val = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    activated = gate * tl.sigmoid(2.0 * inner_val)
    hidden = activated * up
    tl.store(
        output + row[:, None] * N + column[None, :],
        hidden.to(tl.bfloat16),
        (row[:, None] < M) & (column[None, :] < N),
    )


def dual_geglu_fast(
    inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, tile: GemmTile
) -> torch.Tensor:
    rows, width = inputs.shape
    columns = gate.shape[1]
    output = torch.empty((rows, columns), device=inputs.device, dtype=inputs.dtype)
    bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    _dual_geglu_fast[(triton.cdiv(rows, bm), triton.cdiv(columns, tile.n))](
        inputs,
        gate,
        up,
        output,
        rows,
        columns,
        width,
        bm,
        tile.n,
        tile.k,
        num_warps=tile.warps,
        num_stages=tile.stages,
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
        else:
            self.packed_w = None
            self.gate, self.up = (value.detach().t().contiguous() for value in (gate, up))

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use the prepared tile and optional smaller tail for the current matrix."""
        if self.packed_w is not None:
            if not hasattr(self.backend, "_packed_fusion"):
                from ..cuda.activation import GeluMulFusion
                from ..cuda.quantization import CudaQuantizer

                self.backend._packed_fusion = GeluMulFusion(CudaQuantizer(), approximate="none")
                self.backend._packed_plans = {}
            packed = torch.nn.functional.linear(inputs, self.packed_w)
            inter = packed.shape[1] // 2
            key = (
                packed.shape[0],
                inter,
                packed.device,
                torch.cuda.current_stream(packed.device).cuda_stream,
            )
            if key not in self.backend._packed_plans:
                dummy = torch.empty((packed.shape[0], inter), device=packed.device, dtype=torch.bfloat16)
                self.backend._packed_plans[key] = self.backend._packed_fusion.plan(dummy, 16, None)
            return self.backend._packed_plans[key].encode_packed(packed).output
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
def _linear_bf16_kernel(
    x,
    w,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + rk
        xm = (rm[:, None] < M) & (k[None, :] < K)
        wm = (k[:, None] < K) & (rn[None, :] < N)
        a = tl.load(x + rm[:, None] * K + k[None, :], mask=xm, other=0.0)
        b = tl.load(w + k[:, None] * N + rn[None, :], mask=wm, other=0.0)
        acc = tl.dot(a, b, acc)
    om = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=om)


@triton.jit
def _linear_fp8_kernel(
    x_fp8,
    w_fp8,
    scale_x_ptr,
    scale_w_ptr,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + rk
        xm = (rm[:, None] < M) & (k[None, :] < K)
        wm = (k[:, None] < K) & (rn[None, :] < N)
        a = tl.load(x_fp8 + rm[:, None] * K + k[None, :], mask=xm, other=0.0)
        b = tl.load(w_fp8 + k[:, None] * N + rn[None, :], mask=wm, other=0.0)
        acc = tl.dot(a, b, acc)
    sx = tl.load(scale_x_ptr).to(tl.float32)
    sw = tl.load(scale_w_ptr).to(tl.float32)
    acc = acc * (sx * sw)
    om = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=om)


def triton_linear(inputs: torch.Tensor, weight_kn: torch.Tensor) -> torch.Tensor:
    """Small-M BF16 projection using weight-streaming Triton tiles."""
    rows, width = inputs.shape
    columns = weight_kn.shape[1]
    out = torch.empty((rows, columns), device=inputs.device, dtype=torch.bfloat16)
    bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    bn = 64
    bk = 64
    _linear_bf16_kernel[(triton.cdiv(rows, bm), triton.cdiv(columns, bn))](
        inputs,
        weight_kn,
        out,
        M=rows,
        N=columns,
        K=width,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=4,
        num_stages=3,
    )
    return out


def triton_linear_fp8(
    inputs_fp8: torch.Tensor,
    weight_kn_fp8: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
) -> torch.Tensor:
    """Small-M FP8 projection using weight-streaming Triton tiles."""
    rows, width = inputs_fp8.shape
    columns = weight_kn_fp8.shape[1]
    out = torch.empty((rows, columns), device=inputs_fp8.device, dtype=torch.bfloat16)
    bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    bn = 64
    bk = 128 if width >= 2048 else 64
    _linear_fp8_kernel[(triton.cdiv(rows, bm), triton.cdiv(columns, bn))](
        inputs_fp8,
        weight_kn_fp8,
        scale_x,
        scale_w,
        out,
        M=rows,
        N=columns,
        K=width,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _dual_geglu_fp8_kernel(
    x_fp8,
    g_fp8,
    u_fp8,
    scale_x_ptr,
    scale_g_ptr,
    scale_u_ptr,
    inv_out_ptr,
    out_fp8,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc_g = tl.zeros((BM, BN), tl.float32)
    acc_u = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + rk
        xm = (rm[:, None] < M) & (k[None, :] < K)
        wm = (k[:, None] < K) & (rn[None, :] < N)
        a = tl.load(x_fp8 + rm[:, None] * K + k[None, :], mask=xm, other=0.0)
        bg = tl.load(g_fp8 + k[:, None] * N + rn[None, :], mask=wm, other=0.0)
        bu = tl.load(u_fp8 + k[:, None] * N + rn[None, :], mask=wm, other=0.0)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
    sx = tl.load(scale_x_ptr).to(tl.float32)
    sg = tl.load(scale_g_ptr).to(tl.float32)
    su = tl.load(scale_u_ptr).to(tl.float32)
    inv_out = tl.load(inv_out_ptr).to(tl.float32)
    gate = (acc_g * (sx * sg)).to(tl.bfloat16).to(tl.float32)
    up = (acc_u * (sx * su)).to(tl.bfloat16).to(tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    act = (gate * tl.sigmoid(2.0 * inner)).to(tl.bfloat16).to(tl.float32)
    prod = (act * up).to(tl.bfloat16).to(tl.float32)
    q = tl.clamp(prod * inv_out, -448.0, 448.0).to(tl.float8e4nv)
    om = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_fp8 + rm[:, None] * N + rn[None, :], q, mask=om)


def dual_geglu_fp8(
    x_fp8: torch.Tensor,
    g_kn_fp8: torch.Tensor,
    u_kn_fp8: torch.Tensor,
    scale_x: torch.Tensor,
    scale_g: torch.Tensor,
    scale_u: torch.Tensor,
    inv_out: torch.Tensor,
    out_fp8: torch.Tensor,
    rows: int,
) -> torch.Tensor:
    width = x_fp8.shape[1]
    columns = g_kn_fp8.shape[1]
    bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    bn = 64
    bk = 64
    _dual_geglu_fp8_kernel[(triton.cdiv(rows, bm), triton.cdiv(columns, bn))](
        x_fp8,
        g_kn_fp8,
        u_kn_fp8,
        scale_x,
        scale_g,
        scale_u,
        inv_out,
        out_fp8,
        M=rows,
        N=columns,
        K=width,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=4,
        num_stages=3,
    )
    return out_fp8
