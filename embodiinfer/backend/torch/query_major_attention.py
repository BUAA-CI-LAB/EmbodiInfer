"""Registered attention variants using ccinfer's query-major layout."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .attention_layout import query_major_qk


class QueryMajorAttention:
    """BF16 Q/K with FP32 scores and fused, query-major BF16 probabilities.

    Unlike eager BF16 attention, scores accumulate and scale in FP32. Softmax
    reduction also differs. This backend requires an explicit numerical gate.
    Masks must be Boolean validity masks; arbitrary additive biases are unsupported.
    """

    name = "query_major"

    def __init__(self) -> None:
        """Resolve probability kernels before graph capture."""
        from ...layers.launch import SoftmaxTile
        from ..triton.softmax import MaskSoftmax

        self.prefix = MaskSoftmax(SoftmaxTile(4, 4, True), pv_layout=True, query_rows=True)
        self.action = MaskSoftmax(SoftmaxTile(1, 4), pv_layout=True, query_rows=True)

    @staticmethod
    def capability() -> tuple[bool, str | None]:
        """Require a supported Triton CUDA target."""
        from ..triton.capability import triton_capability

        result = triton_capability()
        return result.available, result.reason

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Compute BHSD attention without a group-major Q/probability copy."""
        if dropout_p:
            raise ValueError("Query-major inference attention does not implement dropout")
        if any(t.dtype != torch.bfloat16 or not t.is_cuda or t.device != q.device for t in (q, k, v)):
            raise ValueError("Query-major attention requires BF16 tensors on one CUDA device")
        batch, heads, queries, width = q.shape
        kv_heads, keys = k.shape[1:3]
        if k.shape != v.shape or heads % kv_heads:
            raise ValueError("Query and K/V head layouts are incompatible")
        query = q.transpose(1, 2).reshape(batch, queries, kv_heads, heads // kv_heads, width)
        logits = query_major_qk(query, k.transpose(1, 2))
        logits = logits * (width**-0.5 if scaling is None else scaling)
        if attn_mask is None:
            mask = torch.ones(batch, queries, keys, device=q.device, dtype=torch.bool)
        else:
            if attn_mask.dtype != torch.bool or attn_mask.ndim != 4 or attn_mask.shape[1] != 1:
                raise ValueError("Query-major attention requires a Boolean [B,1,Q,K] validity mask")
            mask = attn_mask.expand(batch, 1, queries, keys)[:, 0].contiguous()
        probability = (self.action if queries <= 16 else self.prefix)(logits, mask, q.dtype)
        output = torch.einsum("BKGTS,BSKH->BTKGH", probability, v.transpose(1, 2))
        return output.reshape(batch, queries, heads, width).transpose(1, 2)


class QueryMajorCudaAttention(QueryMajorAttention):
    """Query-major attention with ccinfer's Spark CUDA prefix softmax launch."""

    name = "query_major_cuda"

    def __init__(self) -> None:
        """Use one row per warp and CUDA's exponential for prefix probabilities."""
        from ...layers.launch import SoftmaxTile
        from ..triton.softmax import MaskSoftmax

        super().__init__()
        self.prefix = MaskSoftmax(SoftmaxTile(4, 4, True, "cuda"), pv_layout=True, query_rows=True)


class FoldedFlashAttention:
    """Mask-free FlashAttention, folding small MQA queries into one query head."""

    name = "folded_flash"

    @staticmethod
    def capability() -> tuple[bool, str | None]:
        """Require CUDA and a FlashAttention-capable architecture."""
        if not torch.cuda.is_available():
            return False, "CUDA is unavailable"
        available = torch.cuda.get_device_capability() >= (8, 0)
        return available, None if available else "FlashAttention requires SM80 or newer"

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Preserve the caller's SDPA scale while changing small-query head layout."""
        if attn_mask is not None or dropout_p:
            raise ValueError("Folded FlashAttention requires a compact all-valid inference prefix")
        batch, heads, queries, width = q.shape
        folded = queries <= 16 and k.shape[1] == 1
        query = q.transpose(1, 2).reshape(batch, 1, queries * heads, width) if folded else q
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output = F.scaled_dot_product_attention(
                query, k, v, scale=scaling, enable_gqa=query.shape[1] != k.shape[1]
            )
        return output.reshape(batch, queries, heads, width).transpose(1, 2) if folded else output
