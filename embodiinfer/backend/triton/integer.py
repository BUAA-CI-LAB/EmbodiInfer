"""Native INT8 Tensor Core projections with exact rounded BF16 GeGLU fusion."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch
import triton
import triton.language as tl

from ...layers.quantization import WorkspaceKey
from ...layers.registry import OperatorCapabilities
from ..cuda.rotation_quantization import RotatedInt8Quantizer


@lru_cache
def _gelu_table(device: torch.device) -> torch.Tensor:
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(device)
    return torch.nn.functional.gelu(bits, approximate="none")


@triton.jit
def _projection(
    Q,
    W,
    XS,
    WS,
    Y,
    LUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GATE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    rawn = tl.arange(0, BN)
    cols = pn * (BN // 2) + (rawn % (BN // 2)) + (rawn // (BN // 2)) * (N // 2) if GATE else pn * BN + rawn
    kk = tl.arange(0, BK)
    accum = tl.full((BM, BN), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        inner = block * BK + kk
        a = tl.load(Q + rows[:, None] * K + inner[None, :], (rows[:, None] < M) & (inner[None, :] < K), 0)
        b = tl.load(W + cols[None, :] * K + inner[:, None], (cols[None, :] < N) & (inner[:, None] < K), 0)
        accum = tl.dot(a, b, accum)
    ws = tl.load(WS + cols, cols < N, 0)
    xs = tl.load(XS + rows, rows < M, 0)
    result = (accum.to(tl.float32) * ws[None, :] * xs[:, None]).to(tl.bfloat16).to(tl.float32)
    if GATE:
        gate, up = tl.split(tl.permute(tl.reshape(result, (BM, 2, BN // 2)), (0, 2, 1)))
        bits = gate.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
        gelu = tl.load(LUT + bits).to(tl.float32)
        out = (gelu * up).to(tl.bfloat16)
        oc = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(
            Y + rows[:, None] * (N // 2) + oc[None, :], out, (rows[:, None] < M) & (oc[None, :] < N // 2)
        )
    else:
        tl.store(
            Y + rows[:, None] * N + cols[None, :],
            result.to(tl.bfloat16),
            (rows[:, None] < M) & (cols[None, :] < N),
        )


class RotatedInt8Backend:
    """Frozen W8/per-token A8 execution with a calibrated H128 input transform."""

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "h128_bf16_round_int8_int32_bf16", minimum_sm=(8, 0)
    )

    def __init__(self) -> None:
        """Prepare the device-local encoder and exhaustive BF16 exact-GELU table."""
        self.quantizer = RotatedInt8Quantizer()
        self.device = self.quantizer.device
        self.table = _gelu_table(self.device)

    def plan(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        divisor: torch.Tensor,
        *,
        workspace_key: Callable[[torch.device], WorkspaceKey],
        fused_gelu: bool = False,
    ) -> RotatedInt8Plan:
        """Validate calibrated CPU tensors and retain their codes/scales exactly."""
        return RotatedInt8Plan(self, codes, scales, divisor, workspace_key, fused_gelu)


class RotatedInt8Plan:
    """Shared immutable weights and separate shape/stream/graph activation buffers."""

    def __init__(
        self,
        backend: RotatedInt8Backend,
        codes: torch.Tensor,
        scales: torch.Tensor,
        divisor: torch.Tensor,
        workspace_key: Callable[[torch.device], WorkspaceKey],
        fused_gelu: bool,
    ) -> None:
        """Reject unsupported widths and malformed packs before GPU placement."""
        if codes.ndim != 2 or codes.dtype != torch.int8 or codes.device.type != "cpu":
            raise ValueError("Rotated INT8 weights require a CPU signed INT8 matrix")
        n, k = codes.shape
        if k not in (2048, 16384) or n <= 0 or n % (2 if fused_gelu else 1):
            raise ValueError("Rotated INT8 requires K=2048 or 16384 and complete gate/up pairs")
        if any(
            x.dtype != torch.float32
            or x.device.type != "cpu"
            or x.shape != (length,)
            or not bool(torch.isfinite(x).all())
            or not bool((x > 0).all())
            for x, length in ((scales, n), (divisor, k))
        ):
            raise ValueError("Rotated INT8 requires finite positive FP32 channel scales/divisors")
        self.backend, self.workspace_key, self.fused_gelu = backend, workspace_key, fused_gelu
        self.n, self.k, self.device = n, k, backend.device
        with torch.cuda.device(self.device):
            self.codes = codes.to(self.device).contiguous()
            self.scales = scales.to(self.device).contiguous()
            self.divisor = divisor.to(self.device).contiguous()
        self.buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Include channel rotation/quantization and the fused integer GEMM epilogue."""
        if (
            inputs.ndim < 2
            or inputs.shape[-1] != self.k
            or inputs.dtype != torch.bfloat16
            or (inputs.device != self.device or not inputs.is_contiguous())
        ):
            raise ValueError("Rotated INT8 requires contiguous BF16 inputs on its bound device")
        rows = inputs.numel() // self.k
        if rows <= 0 or rows * max(self.n, self.k) >= 2**31:
            raise ValueError("Rotated INT8 matrix exceeds its indexing bounds")
        key = rows, self.workspace_key(self.device)
        with torch.cuda.device(self.device):
            if key not in self.buffers:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm rotated INT8 execution scopes before capture")
                self.buffers[key] = (
                    torch.empty((rows, self.k), device=self.device, dtype=torch.int8),
                    torch.empty(rows, device=self.device, dtype=torch.float32),
                    torch.empty(
                        (rows, self.n // 2 if self.fused_gelu else self.n),
                        device=self.device,
                        dtype=torch.bfloat16,
                    ),
                )
            codes, scales, output = self.buffers[key]
            self.backend.quantizer(inputs, self.divisor, codes, scales)
            bm, bn, bk, warps, stages = (128, 128, 64, 4, 3) if self.fused_gelu else (128, 256, 128, 8, 3)
            grid = triton.cdiv(rows, bm), triton.cdiv(self.n, bn)
            _projection[grid](
                codes,
                self.codes,
                scales,
                self.scales,
                output,
                self.backend.table,
                rows,
                self.n,
                self.k,
                self.fused_gelu,
                bm,
                bn,
                bk,
                num_warps=warps,
                num_stages=stages,
                enable_fp_fusion=True,
            )
        return output.reshape(*inputs.shape[:-1], output.shape[-1])

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release only the completed graph's quantization and output storage."""
        for key in tuple(self.buffers):
            if key[-1] == scope:
                del self.buffers[key]
