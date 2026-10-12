"""Strict FP32 attention with fused RoPE/cache writes and copy-free layouts."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import torch

from ...layers.registry import OperatorCapabilities
from .native import build_library

if TYPE_CHECKING:
    from .qk import TensorCoreQk


class StrictRotaryAttention:
    """B1/MQA attention preserving Torch's persistent FP32 softmax reduction.

    Q is stored head-major and probabilities token-major in the existing fusions,
    avoiding the two layout clones in the following einsums. FP32 QK, finite
    masking, BF16 prescaling/probabilities and the value GEMM remain unchanged.
    """

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "fp32_scores_bf16_query_probability", minimum_sm=(8, 0)
    )
    qk: TensorCoreQk | None = None

    def __init__(self) -> None:
        """Resolve packaged kernels on the registry's current device."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.library = build_library("attention_prepare", strict=True, specific=False).library
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.library.ei_prepare_qk.argtypes = [ptr] * 9 + [integer] * 6 + [ctypes.c_float, ptr]
        self.library.ei_load_prefix.argtypes = [ptr] * 5 + [integer] * 4 + [ptr]
        self.library.ei_masked_softmax.argtypes = [ptr] * 3 + [integer] * 4 + [ptr]
        for name in ("ei_prepare_qk", "ei_load_prefix", "ei_masked_softmax"):
            getattr(self.library, name).restype = integer

    def plan(
        self, query_shape: tuple[int, int, int, int], *, kv_heads: int, prefix_length: int = 0
    ) -> StrictAttentionPlan:
        """Prepare bounded B1/MQA workspaces outside capture."""
        return StrictAttentionPlan(self, query_shape, kv_heads, prefix_length)

    @staticmethod
    def check(status: int) -> None:
        """Report launch errors without synchronizing the caller's stream."""
        if status:
            raise RuntimeError(f"Strict attention CUDA launch failed: {status}")


class StrictAttentionPlan:
    """Private cache and query storage for one layer and execution scope."""

    def __init__(self, backend: StrictRotaryAttention, shape: tuple, kv_heads: int, prefix: int) -> None:
        """Reject unsupported indexing/reduction sizes before allocating buffers."""
        batch, tokens, heads, width = shape
        columns = tokens + prefix
        if (
            batch != 1
            or kv_heads != 1
            or min(tokens, heads, width) <= 0
            or width % 2
            or prefix < 0
            or not 128 < columns <= 1024
            or batch * heads * tokens * width >= 2**31
        ):
            raise ValueError("Strict attention requires B1, one KV head, even D and 129..1024 keys")
        self.backend, self.shape, self.prefix = backend, shape, prefix
        self.key_shape = (batch, tokens, kv_heads, width)
        self.loaded = prefix == 0
        with torch.cuda.device(backend.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm attention layouts before capture")
            self.query = torch.empty(
                (batch, heads, tokens, width), device=backend.device, dtype=torch.float32
            )
            full = (batch, columns, kv_heads, width)
            self.key = torch.empty(full, device=backend.device, dtype=torch.bfloat16)
            self.key32 = torch.empty(full, device=backend.device, dtype=torch.float32)
            self.value = torch.empty_like(self.key)
            self.qk_workspace = (
                torch.empty(4 * 1024**2, device=backend.device, dtype=torch.uint8)
                if prefix == 0 and backend.qk is not None
                else None
            )

    def _validate(self, tensor: torch.Tensor, shape: tuple, dtype: torch.dtype) -> None:
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != self.backend.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("Strict attention tensor shape, dtype, device or layout changed")

    def load_prefix(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Fuse prefix copies and FP32 conversion once per request."""
        shape = (1, self.prefix, 1, self.shape[-1])
        if self.prefix == 0:
            raise ValueError("This attention plan has no cached prefix")
        self._validate(key, shape, torch.bfloat16)
        self._validate(value, shape, torch.bfloat16)
        with torch.cuda.device(self.backend.device):
            self.backend.check(
                self.backend.library.ei_load_prefix(
                    key.data_ptr(),
                    value.data_ptr(),
                    self.key.data_ptr(),
                    self.key32.data_ptr(),
                    self.value.data_ptr(),
                    1,
                    self.prefix,
                    self.shape[1],
                    self.shape[-1],
                    torch.cuda.current_stream(self.backend.device).cuda_stream,
                )
            )
        self.loaded = True

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sine: torch.Tensor,
        cosine: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preserve the reference GEMMs while fusing their layout preparation."""
        if not self.loaded:
            raise RuntimeError("Load the current request's prefix before attention")
        self._validate(query, self.shape, torch.bfloat16)
        for tensor in (key, value):
            self._validate(tensor, self.key_shape, torch.bfloat16)
        batch, tokens, heads, width = self.shape
        for tensor in (sine, cosine):
            self._validate(tensor, (batch, tokens, 1, width // 2), torch.float32)
        columns = self.key.shape[1]
        if mask is not None:
            self._validate(mask, (batch, tokens, columns), torch.bool)
        with torch.cuda.device(self.backend.device):
            stream = torch.cuda.current_stream(self.backend.device).cuda_stream
            self.backend.check(
                self.backend.library.ei_prepare_qk(
                    query.data_ptr(),
                    key.data_ptr(),
                    value.data_ptr(),
                    sine.data_ptr(),
                    cosine.data_ptr(),
                    self.query.data_ptr(),
                    self.key.data_ptr(),
                    self.key32.data_ptr(),
                    self.value.data_ptr() if self.prefix else None,
                    batch,
                    tokens,
                    heads,
                    1,
                    width,
                    self.prefix,
                    width**-0.5,
                    stream,
                )
            )
            values = self.value if self.prefix else value
            q = self.query.permute(0, 2, 1, 3).reshape(batch, tokens, 1, heads, width)
            scores = (
                self.backend.qk(self.query, self.key32, self.qk_workspace)
                if self.prefix == 0 and self.backend.qk is not None
                else torch.einsum("BTKGH,BSKH->BKGTS", q, self.key32)
            )
            storage = torch.empty(
                (batch, 1, tokens, heads, columns), device=query.device, dtype=torch.bfloat16
            )
            self.backend.check(
                self.backend.library.ei_masked_softmax(
                    scores.data_ptr(),
                    None if mask is None else mask.data_ptr(),
                    storage.data_ptr(),
                    batch,
                    tokens,
                    heads,
                    columns,
                    stream,
                )
            )
            probabilities = storage.permute(0, 1, 3, 2, 4)
            output = torch.einsum("BKGTS,BSKH->BTKGH", probabilities, values)
        return output.reshape(batch, tokens, heads, width), self.key, values


class TensorCorePrefixAttention(StrictRotaryAttention):
    """Use private TF32 QK for prefix attention; keep action QK and softmax strict."""

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "tf32_prefix_fp32_softmax_bf16_probability", minimum_sm=(8, 0)
    )

    def __init__(self) -> None:
        """Prepare the strict layout kernels and a private cuBLAS math mode."""
        from .qk import TensorCoreQk

        super().__init__()
        self.qk = TensorCoreQk()
