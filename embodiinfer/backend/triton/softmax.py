"""Mask, FP32 softmax and BF16 stores without probability copies."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ...layers.launch import SoftmaxTile


@triton.jit
def mask_softmax_rows(
    SCORES,
    MASK,
    OUTPUT,
    SB: tl.constexpr,
    SH: tl.constexpr,
    SG: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    MB: tl.constexpr,
    MQ: tl.constexpr,
    MK: tl.constexpr,
    OB: tl.constexpr,
    OH: tl.constexpr,
    OG: tl.constexpr,
    OQ: tl.constexpr,
    OK: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    QUERIES: tl.constexpr,
    KEYS: tl.constexpr,
    ROWS: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    ACCURATE_EXP: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
):
    """Reduce rows in FP32 and store the requested probability dtype directly."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    columns = tl.arange(0, BK)
    if QUERY_ROWS:
        query = rows // GROUPS % QUERIES
        group = rows % GROUPS
    else:
        query = rows % QUERIES
        group = rows // QUERIES % GROUPS
    head = rows // (QUERIES * GROUPS) % HEADS
    batch = rows // (QUERIES * GROUPS * HEADS)
    valid = (rows[:, None] < ROWS) & (columns[None, :] < KEYS)
    offsets = batch * SB + head * SH + group * SG + query * SQ
    scores = tl.load(SCORES + offsets[:, None] + columns[None, :] * SK, valid, other=0.0)
    allowed = tl.load(
        MASK + (batch * MB + query * MQ)[:, None] + columns[None, :] * MK,
        valid,
        other=False,
    )
    # The finite sentinel deliberately makes completely masked rows uniform.
    scores = tl.where(allowed, scores, -2.3819763e38)
    scores = tl.where(columns[None, :] < KEYS, scores, -float("inf"))
    shifted = scores - tl.max(scores, axis=1)[:, None]
    numerator = libdevice.exp(shifted) if ACCURATE_EXP else tl.exp(shifted)
    denominator = tl.sum(numerator, axis=1)
    # A shared RN reciprocal avoids repeated RN division. The multiply changes
    # FP32 rounding relative to division and is an explicit approximate path.
    inverse = tl.div_rn(1.0, denominator)
    probabilities = numerator * inverse[:, None]
    # tl.store performs the final RN cast; no global FP32 probability tensor.
    output_offsets = batch * OB + head * OH + group * OG + query * OQ
    tl.store(OUTPUT + output_offsets[:, None] + columns[None, :] * OK, probabilities, valid)


class MaskSoftmax:
    """Launch a selected tile with strided FP32 scores and Boolean [B,Q,K] masks."""

    def __init__(
        self,
        tile: SoftmaxTile | None = None,
        *,
        pv_layout: bool = False,
        query_rows: bool = False,
    ) -> None:
        """Own only compiled-kernel evidence; each call returns independent storage."""
        self.tile = SoftmaxTile() if tile is None else tile
        self.pv_layout = pv_layout
        self.query_rows = query_rows
        self.cuda = None
        if self.tile.backend == "cuda":
            from ..cuda.softmax import CudaSoftmax

            self.cuda = CudaSoftmax()
        elif self.tile.backend != "triton":
            raise ValueError(f"Unknown probability backend: {self.tile.backend}")

    def __call__(
        self, logits: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Fuse masking, max/exp/sum/divide and the output cast in one CUDA launch.

        Reduction order and optionally exp implementation differ from Torch.
        This is an approximate implementation, including with accurate exp.
        The mask is read on every call, so its values may change during replay.
        """
        if not logits.is_cuda or logits.dtype != torch.float32 or logits.ndim != 5:
            raise ValueError("Fusion requires CUDA FP32 [B,KV_heads,groups,Q,K] scores")
        batch, heads, groups, queries, keys = logits.shape
        if (
            mask.device != logits.device
            or mask.dtype != torch.bool
            or mask.shape != (batch, queries, keys)
            or dtype not in (torch.bfloat16, torch.float32)
            or min(logits.shape) < 1
        ):
            raise ValueError("Expected nonempty scores, matching Boolean mask and BF16/FP32 output")
        if self.pv_layout:
            # The logical contract stays [B,K,G,T,S]. Native einsum PV flattens
            # T,G, so write that physical order directly instead of copying P.
            output = torch.empty(
                (batch, heads, queries, groups, keys), dtype=dtype, device=logits.device
            ).transpose(2, 3)
        else:
            output = torch.empty(logits.shape, dtype=dtype, device=logits.device)
        if self.cuda is not None:
            self.cuda(logits, mask, output, self.tile.warps, query_rows=self.query_rows)
            return output
        rows = batch * heads * groups * queries
        mask_softmax_rows[(triton.cdiv(rows, self.tile.rows),)](
            logits,
            mask,
            output,
            *logits.stride(),
            *mask.stride(),
            *output.stride(),
            heads,
            groups,
            queries,
            keys,
            rows,
            self.tile.rows,
            triton.next_power_of_2(keys),
            self.tile.accurate_exp,
            self.query_rows,
            num_warps=self.tile.warps,
            enable_fp_fusion=False,
        )
        return output
