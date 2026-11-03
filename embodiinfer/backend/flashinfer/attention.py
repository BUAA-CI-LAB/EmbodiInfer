"""Explicit FA2 attention with graph-scoped ragged plans and dynamic Boolean masks."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from ...layers.quantization import WorkspaceKey
from .runtime import availability, load_module


def _stream_scope(device: torch.device) -> WorkspaceKey:
    return "stream", torch.cuda.current_stream(device).cuda_stream


class FlashInferFA2Attention:
    """BF16 BHSD GQA through noncausal FA2 prefill, including multi-token queries.

    Head widths 64/128/256 are supported. Boolean masks broadcast to [B,1,Q,K];
    True means allowed. Additive biases and dropout are rejected. Fully masked
    query rows return zero, matching Torch SDPA's Boolean mask convention.
    Output storage is ephemeral until the next same-shape call in this scope.
    """

    name = "flashinfer_fa2"

    def __init__(self, *, workspace_bytes: int = 32 * 1024 * 1024) -> None:
        """Resolve the optional wrapper and defer shape compilation until warmup."""
        if type(workspace_bytes) is not int or workspace_bytes <= 0:
            raise ValueError("FlashInfer attention workspace_bytes must be a positive integer")
        self._wrapper = load_module("flashinfer.prefill").BatchPrefillWithRaggedKVCacheWrapper
        self._workspace_bytes = workspace_bytes
        self._scope: Callable[[torch.device], WorkspaceKey] = _stream_scope
        self._plans: dict[tuple, _AttentionPlan] = {}

    @staticmethod
    def capability() -> tuple[bool, str | None]:
        """Report optional installation/SM80+ support without compiling kernels."""
        return availability()

    def bind_workspace_key(self, workspace_key: Callable[[torch.device], WorkspaceKey]) -> None:
        """Bind the owning runtime's graph/stream lifetimes before first use."""
        if self._plans:
            raise RuntimeError("Bind attention execution scopes before warming any plans")
        self._scope = workspace_key

    def release_scope(self, scope: WorkspaceKey) -> None:
        """Release plans only after their graph and CUDA work have completed."""
        for key in tuple(self._plans):
            if key[-1] == scope:
                del self._plans[key]

    def clear(self) -> None:
        """Release all plans after all referencing graphs have been destroyed."""
        self._plans.clear()

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Apply noncausal GQA with explicit optional masks and no K/V head expansion."""
        self._validate(q, k, v, attn_mask, scaling, dropout_p)
        scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
        with torch.cuda.device(q.device):
            key = (
                tuple(q.shape),
                tuple(k.shape),
                q.device,
                scale,
                attn_mask is not None,
                self._scope(q.device),
            )
            if key not in self._plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm FlashInfer attention shape/mask/scale in this scope before capture"
                    )
                plan = _AttentionPlan(
                    self._wrapper, q, k, attn_mask is not None, scale, self._workspace_bytes
                )
                output = plan.run(q, k, v, attn_mask)
                self._plans[key] = plan
                return output
            return self._plans[key].run(q, k, v, attn_mask)

    @staticmethod
    def _validate(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        scaling: float | None,
        dropout_p: float,
    ) -> None:
        if any(t.ndim != 4 or min(t.shape) <= 0 for t in (q, k, v)):
            raise ValueError("FlashInfer FA2 requires nonempty BHSD Q/K/V")
        if (
            q.device.type != "cuda"
            or any(t.device != q.device or t.dtype != torch.bfloat16 for t in (q, k, v))
            or k.shape != v.shape
            or q.shape[0] != k.shape[0]
            or q.shape[-1] != k.shape[-1]
            or q.shape[1] % k.shape[1]
        ):
            raise ValueError("FlashInfer FA2 requires same-device BF16 Q/K/V with compatible GQA dimensions")
        if q.shape[-1] not in (64, 128, 256):
            raise ValueError("FlashInfer FA2 head width must be 64, 128 or 256; width 72 is unsupported")
        if dropout_p != 0:
            raise ValueError("FlashInfer FA2 inference does not support dropout")
        if scaling is not None and (
            not isinstance(scaling, (float, int)) or not math.isfinite(scaling) or scaling <= 0
        ):
            raise ValueError("FlashInfer attention scaling must be finite and positive")
        if mask is not None:
            target = (q.shape[0], 1, q.shape[2], k.shape[2])
            if mask.dtype != torch.bool or mask.device != q.device:
                raise ValueError(
                    "FlashInfer FA2 accepts Boolean allowed masks, not additive attention biases"
                )
            try:
                valid = torch.broadcast_shapes(mask.shape, target) == target
            except RuntimeError:
                valid = False
            if not valid:
                raise ValueError("FlashInfer attention mask must broadcast to [B,1,Q,K]")


class _AttentionPlan:
    def __init__(
        self, wrapper: Any, q: torch.Tensor, k: torch.Tensor, masked: bool, scale: float, workspace_bytes: int
    ) -> None:
        batch, heads, qlen, width = q.shape
        kv_heads, klen = k.shape[1:3]
        self._batch, self._qlen, self._klen = batch, qlen, klen
        self.q = q.new_empty((batch * qlen, heads, width))
        self.k = k.new_empty((batch * klen, kv_heads, width))
        self.v = torch.empty_like(self.k)
        self.output = torch.empty_like(self.q)
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
        self.q_indptr = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
        self.kv_indptr = torch.empty_like(self.q_indptr)
        self.packed_mask = self.mask_indptr = self.mask_bits = self.bit_weights = None
        if masked:
            byte_length = (qlen * klen + 7) // 8
            self.packed_mask = torch.empty(batch * byte_length, dtype=torch.uint8, device=q.device)
            self.mask_indptr = torch.empty_like(self.q_indptr)
            self.mask_bits = torch.zeros((batch, byte_length * 8), dtype=torch.bool, device=q.device)
            self.bit_weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=q.device)
        self.wrapper = wrapper(
            self.workspace,
            kv_layout="NHD",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buf=self.q_indptr,
            kv_indptr_buf=self.kv_indptr,
            custom_mask_buf=self.packed_mask,
            mask_indptr_buf=self.mask_indptr,
        )
        indptr = torch.arange(batch + 1, dtype=torch.int32, device="cpu")
        # Let the upstream planner generate byte offsets for each packed mask
        # segment. Supplying packed_custom_mask directly in this pinned release
        # leaves bit offsets in mask_indptr for ragged multi-request attention.
        self.wrapper.plan(
            indptr * qlen,
            indptr * klen,
            heads,
            kv_heads,
            width,
            custom_mask=torch.ones(batch * qlen * klen, dtype=torch.bool, device=q.device)
            if masked
            else None,
            causal=False,
            pos_encoding_mode="NONE",
            use_fp16_qk_reduction=False,
            sm_scale=scale,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            non_blocking=False,
        )

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch, qlen, klen = self._batch, self._qlen, self._klen
        for destination, source in ((self.q, q), (self.k, k), (self.v, v)):
            destination.view(batch, -1, source.shape[1], source.shape[-1]).copy_(source.transpose(1, 2))
        valid_rows = None
        if mask is not None:
            allowed = mask.broadcast_to((batch, 1, qlen, klen))[:, 0]
            valid_rows = allowed.any(dim=-1)
            bits = self.mask_bits[:, : qlen * klen].view(batch, qlen, klen)
            bits.copy_(allowed)
            # Avoid undefined all-masked softmax rows, then return SDPA's zeros.
            bits[:, :, 0].logical_or_(~valid_rows)
            packed = (self.mask_bits.view(batch, -1, 8) * self.bit_weights).sum(dim=-1)
            self.packed_mask.copy_(packed.reshape(-1))
        self.wrapper.run(self.q, self.k, self.v, out=self.output, enable_pdl=False)
        output = self.output.view(batch, qlen, q.shape[1], q.shape[-1])
        if valid_rows is not None:
            output.masked_fill_(~valid_rows[:, :, None, None], 0)
        return output.transpose(1, 2)
