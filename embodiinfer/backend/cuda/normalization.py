"""RMSNorm and residual kernels retaining explicit BF16 rounding."""

import ctypes
from typing import Literal

import torch

from ...layers.registry import OperatorCapabilities
from .native import build_library

Mode = Literal["pointwise", "reduce"]


class NormFusion:
    """Compile standalone CUDA kernels without changing the reference model."""

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "torch_rmsnorm_bf16_rounding", rank=3, minimum_sm=(8, 0)
    )

    def __init__(self) -> None:
        """Load the cached library for the current CUDA device."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        built = build_library("norm", strict=True, specific=False, native_fp4=False)
        self.library, self.compiler, self.command = built.library, built.compiler, built.command
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.library.cc_norm.argtypes = [pointer] * 6 + [integer] * 4 + [pointer]
        self.library.cc_residual_square.argtypes = [pointer] * 5 + [integer] * 3 + [pointer]
        self.library.cc_residual_norm.argtypes = [pointer] * 8 + [integer] * 3 + [pointer]
        for name in ("cc_norm", "cc_residual_square", "cc_residual_norm"):
            getattr(self.library, name).restype = integer

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise RuntimeError(f"CUDA norm fusion launch failed with error {status}")

    def plan(
        self, inputs: torch.Tensor, *, adaptive: bool, eps: float = 1e-6, mode: Mode = "pointwise"
    ) -> "NormPlan":
        """Allocate one fixed-shape workspace before capture or timing."""
        if eps != 1e-6:
            raise ValueError("CUDA RMSNorm currently requires eps=1e-6")
        return NormPlan(self, inputs, adaptive=adaptive, mode=mode)


class NormPlan:
    """Reusable BF16 workspace; returned tensors are overwritten by subsequent calls."""

    def __init__(self, backend: NormFusion, inputs: torch.Tensor, *, adaptive: bool, mode: Mode) -> None:
        """Allocate outputs and retain Torch mean reduction for the strict kernel."""
        if mode not in ("pointwise", "reduce"):
            raise ValueError("Norm fusion mode must be pointwise or reduce")
        if (
            inputs.device != backend.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
            or inputs.ndim != 3
            or min(inputs.shape) <= 0
        ):
            raise ValueError("Norm fusion requires nonempty contiguous CUDA BF16 [B,T,D] inputs")
        self.backend, self.shape, self.mode, self.adaptive = backend, inputs.shape, mode, adaptive
        self.output = torch.empty_like(inputs)
        self.residual = torch.empty_like(inputs)
        self.gate = (
            torch.empty((inputs.shape[0], 1, inputs.shape[2]), device=inputs.device, dtype=inputs.dtype)
            if adaptive
            else None
        )
        self.square = torch.empty_like(inputs, dtype=torch.float32) if mode == "pointwise" else None
        self.mean = (
            torch.empty((*inputs.shape[:2], 1), device=inputs.device, dtype=torch.float32)
            if mode == "pointwise"
            else None
        )

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if (
            inputs.shape != self.shape
            or inputs.device != self.output.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
        ):
            raise ValueError("Norm fusion input shape, layout, dtype, or device changed")

    def _validate_affine(self, scale: torch.Tensor | None, modulation: torch.Tensor | None) -> None:
        expected = (self.shape[0], self.shape[2] * 3) if self.adaptive else (self.shape[2],)
        affine = modulation if self.adaptive else scale
        if (
            affine is None
            or affine.shape != expected
            or affine.dtype != torch.float32
            or affine.device != self.output.device
            or not affine.is_contiguous()
        ):
            raise ValueError("Norm fusion requires contiguous FP32 scale or raw modulation")

    def normalize(
        self,
        inputs: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply FP32 RMS statistics and affine operations, then round output and gate.

        Pointwise mode uses Torch's original squared-value mean reduction. Reduce
        mode changes the FP32 reduction tree and requires separate tolerance tests.
        """
        self._validate_inputs(inputs)
        self._validate_affine(scale, modulation)
        if self.mode == "pointwise":
            torch.square(inputs.float(), out=self.square)
            torch.mean(self.square, dim=-1, keepdim=True, out=self.mean)
        return self._normalize_with_mean(inputs, scale, modulation)

    def _normalize_with_mean(
        self, inputs: torch.Tensor, scale: torch.Tensor | None, modulation: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, tokens, columns = self.shape
        with torch.cuda.device(inputs.device):
            self.backend._check(
                self.backend.library.cc_norm(
                    inputs.data_ptr(),
                    self.mean.data_ptr() if self.mean is not None else None,
                    scale.data_ptr() if scale is not None else None,
                    modulation.data_ptr() if modulation is not None else None,
                    self.output.data_ptr(),
                    self.gate.data_ptr() if self.gate is not None else None,
                    batch * tokens,
                    columns,
                    tokens,
                    int(self.mode == "reduce"),
                    torch.cuda.current_stream(inputs.device).cuda_stream,
                )
            )
        return self.output, self.gate

    def residual_normalize(
        self,
        inputs: torch.Tensor,
        update: torch.Tensor,
        residual_gate: torch.Tensor | None,
        *,
        scale: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Fuse residual rounding with the following norm, retaining the residual tensor.

        The gated product rounds to BF16 before the sum rounds to BF16. Statistics
        use that rounded sum. Outputs share this workspace and are ephemeral.
        """
        self._validate_inputs(inputs)
        self._validate_inputs(update)
        self._validate_affine(scale, modulation)
        if residual_gate is not None and (
            residual_gate.shape != (self.shape[0], 1, self.shape[2])
            or residual_gate.dtype != torch.bfloat16
            or residual_gate.device != inputs.device
            or not residual_gate.is_contiguous()
        ):
            raise ValueError("Residual gate must be contiguous CUDA BF16 [B,1,D]")
        batch, tokens, columns = self.shape
        pointers = [
            inputs.data_ptr(),
            update.data_ptr(),
            residual_gate.data_ptr() if residual_gate is not None else None,
        ]
        with torch.cuda.device(inputs.device):
            stream = torch.cuda.current_stream(inputs.device).cuda_stream
            if self.mode == "pointwise":
                self.backend._check(
                    self.backend.library.cc_residual_square(
                        *pointers,
                        self.residual.data_ptr(),
                        self.square.data_ptr(),
                        batch * tokens,
                        columns,
                        tokens,
                        stream,
                    )
                )
                torch.mean(self.square, dim=-1, keepdim=True, out=self.mean)
                self._normalize_with_mean(self.residual, scale, modulation)
            else:
                self.backend._check(
                    self.backend.library.cc_residual_norm(
                        *pointers,
                        scale.data_ptr() if scale is not None else None,
                        modulation.data_ptr() if modulation is not None else None,
                        self.residual.data_ptr(),
                        self.output.data_ptr(),
                        self.gate.data_ptr() if self.gate is not None else None,
                        batch * tokens,
                        columns,
                        tokens,
                        stream,
                    )
                )
        return self.residual, self.output, self.gate
