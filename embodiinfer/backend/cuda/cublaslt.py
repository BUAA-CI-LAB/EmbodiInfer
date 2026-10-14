"""Prepared cuBLASLt BF16 projections with explicit algorithm identities."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ...layers.quantization import ActivationQuantizer, EncodedActivation, Precision, WorkspaceKey
from ...layers.registry import OperatorCapabilities
from .native import build_library


@dataclass(frozen=True)
class LtAlgorithm:
    """Library configuration identity; stages/tile are cuBLASLt enum values."""

    identifier: int = 6
    tile: int = 24
    stages: int = 9
    split_k: int = 1
    reduction: int = 0
    swizzle: int = 1
    custom: int = 0

    @property
    def values(self) -> tuple[int, ...]:
        """Return ABI attribute order independent of heuristic rank."""
        return (
            self.identifier,
            self.tile,
            self.stages,
            self.split_k,
            self.reduction,
            self.swizzle,
            self.custom,
        )


class CublasLtBackend:
    """Prepare BF16/FP32-accumulation GEMMs outside capture.

    The default identity is the measured Ampere 256x128/K32 configuration.
    Unsupported library versions/configurations fail explicitly. No timing or
    autotuning occurs during inference; deployments select a validated recipe.
    """

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "bf16_fp32_accumulation", minimum_sm=(8, 0)
    )

    def __init__(self, algorithm: LtAlgorithm | None = None, library_version: int = 120604) -> None:
        """Load the optional CUDA binding and check the library used by this recipe."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.algorithm = algorithm or LtAlgorithm()
        self.library = build_library("cublaslt", specific=False, libraries=("cublasLt",)).library
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.library.ei_lt_version.restype = ctypes.c_size_t
        self.library.ei_lt_create.argtypes = [integer] * 3 + [ctypes.c_size_t, integer, ctypes.POINTER(ptr)]
        self.library.ei_lt_count.argtypes = [ptr]
        self.library.ei_lt_info.argtypes = [
            ptr,
            integer,
            ctypes.POINTER(integer),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.ei_lt_run.argtypes = [ptr, integer] + [ptr] * 4 + [ctypes.c_size_t, ptr]
        self.library.ei_lt_destroy.argtypes = [ptr]
        self.library.ei_lt_destroy.restype = None
        for name in ("ei_lt_create", "ei_lt_count", "ei_lt_info", "ei_lt_run"):
            getattr(self.library, name).restype = integer
        self.version = self.library.ei_lt_version()
        if self.version != library_version:
            raise RuntimeError(f"cuBLASLt recipe requires version {library_version}, found {self.version}")

    def plan(
        self,
        weight: torch.Tensor,
        maximum: float,
        precision: Precision,
        quantizer: ActivationQuantizer | None,
        workspace_key: Callable[[torch.device], WorkspaceKey],
    ) -> LtProjection:
        """Retain a bias-free BF16 weight and create scoped GEMM plans lazily."""
        if (
            precision != "bf16"
            or weight.device != self.device
            or weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or not weight.is_contiguous()
        ):
            raise ValueError("cuBLASLt requires a contiguous same-device BF16 [N,K] weight")
        return LtProjection(self, weight.detach(), workspace_key)


class LtProjection:
    """Own descriptors and outputs per shape/execution; no duplicated weights."""

    precision: Precision = "bf16"
    maximum: float = 1.0

    def __init__(self, backend: CublasLtBackend, weight: torch.Tensor, workspace_key: Callable) -> None:
        """Bind descriptor lifetimes to the policy's execution scope."""
        self.backend, self.weight, self.workspace_key = backend, weight, workspace_key
        self.plans: dict[tuple, LtExecution] = {}

    def encode(self, inputs: torch.Tensor) -> None:
        """BF16 projections do not encode activations."""
        return None

    def apply(self, inputs: torch.Tensor, encoded: EncodedActivation | None) -> torch.Tensor:
        """Enqueue on the guarded input stream; return scope-owned ephemeral output."""
        n, k = self.weight.shape
        if (
            encoded is not None
            or inputs.device != self.weight.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
            or inputs.shape[-1] != k
            or inputs.numel() == 0
        ):
            raise ValueError("cuBLASLt projection input dtype, device, layout or width changed")
        key = (tuple(inputs.shape), self.workspace_key(inputs.device))
        with torch.cuda.device(inputs.device):
            if key not in self.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm cuBLASLt shapes/scopes before graph capture")
                self.plans[key] = LtExecution(self.backend, inputs, n, k)
            execution = self.plans[key]
            status = self.backend.library.ei_lt_run(
                execution.pointer,
                execution.index,
                self.weight.data_ptr(),
                inputs.data_ptr(),
                execution.output.data_ptr(),
                execution.workspace.data_ptr(),
                execution.workspace.numel(),
                torch.cuda.current_stream(inputs.device).cuda_stream,
            )
        if status:
            raise RuntimeError(f"cuBLASLt launch failed: {status}")
        return execution.output

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Destroy descriptors after graph destruction and CUDA synchronization."""
        for key in tuple(self.plans):
            if key[-1] == scope:
                self.plans.pop(key).close()

    def close(self) -> None:
        """Release all completed execution scopes, including eager streams."""
        for execution in self.plans.values():
            execution.close()
        self.plans.clear()


class LtExecution:
    """Native descriptors and buffers whose lifetimes cover graph replay."""

    def __init__(self, backend: CublasLtBackend, inputs: torch.Tensor, n: int, k: int) -> None:
        """Select the frozen algorithm identity rather than its heuristic index."""
        self.library, self.device = backend.library, backend.device
        self.pointer = ctypes.c_void_p()
        m = inputs.numel() // k
        status = self.library.ei_lt_create(m, n, k, 0, 64, ctypes.byref(self.pointer))
        if status:
            raise RuntimeError(f"cuBLASLt plan creation failed: {status}")
        try:
            self.index = -1
            for index in range(self.library.ei_lt_count(self.pointer)):
                values = (ctypes.c_int * 7)()
                workspace, waves = ctypes.c_size_t(), ctypes.c_float()
                status = self.library.ei_lt_info(
                    self.pointer, index, values, ctypes.byref(workspace), ctypes.byref(waves)
                )
                if status == 0 and tuple(values) == backend.algorithm.values and workspace.value == 0:
                    self.index = index
                    break
            if self.index < 0:
                raise ValueError(f"Selected cuBLASLt configuration unavailable for M/N/K={m}/{n}/{k}")
            self.output = inputs.new_empty((*inputs.shape[:-1], n))
            self.workspace = torch.empty(256, device=inputs.device, dtype=torch.uint8)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Free CPU descriptors on their construction device after CUDA work ends."""
        if self.pointer:
            with torch.cuda.device(self.device):
                self.library.ei_lt_destroy(self.pointer)
            self.pointer = ctypes.c_void_p()
