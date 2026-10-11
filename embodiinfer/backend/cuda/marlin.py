"""BF16/group64 Marlin projections with optional rounded GeGLU epilogue.

Packing derives from IST-DASLab/Marlin, Apache-2.0,
Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at).
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable

import torch

from ...layers.quantization import WorkspaceKey
from ...layers.registry import OperatorCapabilities
from .native import build_library


def _permutations() -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for i in range(32):
        fragment = [
            16 * row + i // 4 + 8 * block
            for block in (0, 1)
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1)
        ]
        for j in range(4):
            values.extend(p + 256 * j for p in fragment)
    weight = torch.tensor(values).reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].flatten()
    scales = torch.tensor([i + 8 * j for i in range(8) for j in range(8)])
    return weight, scales


class MarlinBackend:
    """Prepare bias-free W4A16 projections; activations stay in BF16."""

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "symmetric_group64_bf16_weights", minimum_sm=(8, 0)
    )

    def __init__(self) -> None:
        """Bind packed weights to the registry's guarded CUDA device."""
        self.device = torch.device("cuda", torch.cuda.current_device())

    def plan(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        group_size: int,
        workspace_key: Callable[[torch.device], WorkspaceKey],
        fused_gelu: bool = False,
        large_m: bool = False,
        blocks: int | None = None,
    ) -> MarlinPlan:
        """Pack frozen codes on CPU and transfer to this backend's device."""
        return MarlinPlan(self.device, codes, scales, group_size, workspace_key, fused_gelu, large_m, blocks)


class MarlinPlan:
    """Own one packed projection and isolated synchronization locks.

    Gate/up codes are concatenated gate then up; the fused variant pairs rows
    within a fixed N256 tile before packing. No online rounding, scaling or
    activation permutation is performed. Returned outputs are ordinary tensors.
    """

    def __init__(
        self,
        device: torch.device,
        codes: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        workspace_key: Callable[[torch.device], WorkspaceKey],
        fused_gelu: bool,
        large_m: bool,
        blocks: int | None,
    ) -> None:
        """Validate the decoder layout before loading a packaged CUDA kernel."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Pack Marlin weights before CUDA Graph capture")
        if codes.ndim != 2 or codes.dtype != torch.uint8 or group_size != 64:
            raise ValueError("Marlin requires natural-column uint4 codes and group64 scales")
        n, k = codes.shape
        if n <= 0 or k <= 0 or n % 256 or k % 128 or n * k >= 2**31:
            raise ValueError("Marlin requires positive N256/K128 dimensions")
        if scales.shape != (n, k // 64) or scales.dtype != torch.bfloat16:
            raise ValueError("Marlin requires BF16 scales with shape [N,K/64]")
        codes, scales = codes.cpu(), scales.cpu()
        if bool((codes > 15).any()) or not bool((torch.isfinite(scales) & (scales > 0)).all()):
            raise ValueError("Marlin codes must be uint4 and scales finite and positive")
        if fused_gelu and large_m:
            if n % 512:
                raise ValueError("Paired GeGLU requires complete gate/up N256 tiles")
            gate = torch.arange(n // 2).reshape(-1, 128)
            order = torch.stack((gate, gate + n // 2), dim=1).flatten()
            codes, scales = codes[order], scales[order]
        perm, scale_perm = _permutations()
        q = codes.int().t().contiguous()
        q = q.reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n * 16)
        q = q.reshape(-1, perm.numel())[:, perm].reshape_as(q)
        packed = torch.zeros((q.shape[0], q.shape[1] // 8), dtype=torch.int32)
        for i in range(8):
            packed |= q[:, i::8] << (4 * i)
        scales = scales.t().contiguous()
        scales = scales.reshape(-1, scale_perm.numel())[:, scale_perm].reshape(-1, n)
        self.device, self.n, self.k = device, n, k
        self.fused_gelu, self.workspace_key = fused_gelu, workspace_key
        self.fused_gemm = fused_gelu and large_m
        if blocks is not None and (type(blocks) is not int or blocks <= 0):
            raise ValueError("Marlin block count must be positive")
        self.blocks = blocks if blocks is not None else -1
        self.locks: dict[WorkspaceKey, torch.Tensor] = {}
        with torch.cuda.device(device):
            self.values = packed.to(device)
            self.scales = scales.to(device).contiguous()
            name = "marlin_geglu" if self.fused_gemm else "marlin_gate" if large_m else "marlin"
            self.library = build_library(name, specific=False, flags=("--expt-relaxed-constexpr",)).library
        self.library.ei_marlin.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        self.library.ei_marlin.restype = ctypes.c_int
        self.library.ei_geglu.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        self.library.ei_geglu.restype = ctypes.c_int

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Drop locks only after the caller has completed CUDA work and graph use."""
        self.locks.pop(scope, None)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Launch on the input device's current stream with scope-private locks."""
        if (
            inputs.device != self.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
            or inputs.ndim < 2
            or inputs.shape[-1] != self.k
            or inputs.numel() == 0
            or inputs.numel() >= 2**31
        ):
            raise ValueError("Marlin requires contiguous nonempty same-device BF16 inputs ending in K")
        key = self.workspace_key(self.device)
        if key not in self.locks:
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm Marlin execution scopes before graph capture")
                self.locks[key] = torch.zeros(self.n // 128 * 16, device=self.device, dtype=torch.int32)
        columns = self.n // 2 if self.fused_gemm else self.n
        output = inputs.new_empty((*inputs.shape[:-1], columns))
        with torch.cuda.device(self.device):
            status = self.library.ei_marlin(
                inputs.data_ptr(),
                self.values.data_ptr(),
                output.data_ptr(),
                self.scales.data_ptr(),
                self.locks[key].data_ptr(),
                inputs.numel() // self.k,
                self.n,
                self.k,
                64,
                -1,
                256 if self.fused_gemm else -1,
                self.blocks,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        if status:
            raise RuntimeError(f"Marlin CUDA launch failed: {status}")
        if self.fused_gelu and not self.fused_gemm:
            hidden = inputs.new_empty((*inputs.shape[:-1], self.n // 2))
            with torch.cuda.device(self.device):
                status = self.library.ei_geglu(
                    output.data_ptr(),
                    hidden.data_ptr(),
                    inputs.numel() // self.k,
                    self.n // 2,
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            if status:
                raise RuntimeError(f"Marlin GeGLU launch failed: {status}")
            return hidden
        return output
