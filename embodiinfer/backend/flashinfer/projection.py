"""BF16 cuBLASLt projection through FlashInfer with caller-owned scratch storage."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ...layers.quantization import ActivationQuantizer, EncodedActivation, Precision, WorkspaceKey
from ...layers.registry import OperatorCapabilities, OperatorRequest
from .runtime import load_module


class FlashInferCublasLtBackend:
    """Bias-free BF16 linear plans using FlashInfer's cuBLASLt implementation.

    The pinned runner API accepts an explicit workspace, unlike mm_bf16's
    process-global buffer. Scratch can be shared by sequential projections in
    one scope, but never by independently replayed graphs or CUDA streams.
    """

    capabilities = OperatorCapabilities(
        ("cuda",),
        (torch.bfloat16,),
        "bf16_fp32_accumulation",
        minimum_sm=(8, 0),
    )

    def __init__(self, *, workspace_bytes: int = 32 * 1024 * 1024) -> None:
        """Build the pinned cuBLASLt runner outside capture; defer tensor storage."""
        if type(workspace_bytes) is not int or workspace_bytes <= 0:
            raise ValueError("FlashInfer GEMM workspace_bytes must be a positive integer")
        module = load_module("flashinfer.gemm.gemm_base").get_mm_bf16_cublaslt_module()
        self._runner_factory = module.cublaslt_bf16_gemm_runner
        self._workspace_bytes = workspace_bytes
        self._workspaces: dict[tuple, torch.Tensor] = {}

    def plan(
        self,
        weight: torch.Tensor,
        maximum: float,
        precision: Precision,
        quantizer: ActivationQuantizer | None,
        workspace_key: Callable[[torch.device], WorkspaceKey],
    ) -> FlashInferProjectionPlan:
        """Keep BF16 [N,K] weights; reject encoded formats and malformed layouts."""
        self.capabilities.check(OperatorRequest(weight.device, weight.dtype))
        if precision != "bf16" or quantizer is not None:
            raise ValueError("FlashInfer cuBLASLt projection supports unquantized BF16 only")
        if weight.ndim != 2 or min(weight.shape) <= 0 or not weight.is_contiguous():
            raise ValueError("FlashInfer projection requires nonempty contiguous [N,K] weights")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare FlashInfer projection weights before capture")
        return FlashInferProjectionPlan(self, weight, maximum, workspace_key)

    def _workspace(self, device: torch.device, scope: WorkspaceKey) -> torch.Tensor:
        key = (device, scope)
        if key not in self._workspaces:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm FlashInfer GEMM storage in this execution scope before capture")
            self._workspaces[key] = torch.empty(self._workspace_bytes, dtype=torch.uint8, device=device)
        return self._workspaces[key]

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release shared scratch after every operation in this scope has completed."""
        for key in tuple(self._workspaces):
            if key[-1] == scope:
                del self._workspaces[key]

    def clear(self) -> None:
        """Release all scratch after all plans and referencing graphs are discarded."""
        self._workspaces.clear()


class FlashInferProjectionPlan:
    """Frozen BF16 weight and fixed-address matrices for each warmed row count."""

    precision: Precision = "bf16"

    def __init__(
        self,
        backend: FlashInferCublasLtBackend,
        weight: torch.Tensor,
        maximum: float,
        workspace_key: Callable[[torch.device], WorkspaceKey],
    ) -> None:
        """Retain weights and the scoped allocator; defer per-shape heuristics."""
        self.maximum = maximum
        self._backend, self._weight, self._workspace_key = backend, weight.detach(), workspace_key
        self._plans: dict[tuple, tuple[torch.Tensor, torch.Tensor, Any]] = {}

    def encode(self, inputs: torch.Tensor) -> None:
        """BF16 projections do not encode activations."""
        return None

    def apply(self, inputs: torch.Tensor, encoded: EncodedActivation | None) -> torch.Tensor:
        """Compute inputs @ weight.T, preserving leading dimensions and BF16 dtype."""
        if encoded is not None:
            raise ValueError("FlashInfer BF16 projection does not accept encoded activations")
        if (
            inputs.ndim < 2
            or min(inputs.shape) <= 0
            or inputs.shape[-1] != self._weight.shape[1]
            or inputs.dtype != torch.bfloat16
            or inputs.device != self._weight.device
        ):
            raise ValueError("FlashInfer projection input shape, dtype or device does not match its weight")
        with torch.cuda.device(inputs.device):
            matrix = inputs.reshape(-1, inputs.shape[-1])
            scope = self._workspace_key(inputs.device)
            key = (matrix.shape[0], scope)
            workspace = self._backend._workspace(inputs.device, scope)
            if key not in self._plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm FlashInfer GEMM rows in this scope before capture")
                # Stable aligned inputs keep the warmed cuBLASLt heuristic valid
                # even when a later caller supplies a strided or offset view.
                source = torch.empty(matrix.shape, dtype=inputs.dtype, device=inputs.device)
                output = source.new_empty((matrix.shape[0], self._weight.shape[0]))
                runner = self._backend._runner_factory()
                source.copy_(matrix)
                runner.forward([source, self._weight.t(), None, False, output, workspace], tactic=-1)
                self._plans[key] = source, output, runner
                return output.view(*inputs.shape[:-1], self._weight.shape[0])
            source, output, runner = self._plans[key]
            source.copy_(matrix)
            runner.forward([source, self._weight.t(), None, False, output, workspace], tactic=-1)
            return output.view(*inputs.shape[:-1], self._weight.shape[0])

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Drop completed graph matrices and their shared scratch allocation."""
        for key in tuple(self._plans):
            if key[-1] == scope:
                del self._plans[key]
        self._backend.release_scope(scope)
