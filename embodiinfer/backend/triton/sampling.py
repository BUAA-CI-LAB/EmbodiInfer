"""Full-vocabulary greedy projection and sampling kernels owned by EmbodiInfer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


LOGIT_BLOCK_SIZE = 1024
LM_HEAD_BLOCK_SIZE = 32


if triton is not None:

    @triton.jit
    def _mark_penalized_kernel(penalized, token, vocab_size):
        token_id = tl.load(token)
        tl.store(
            penalized + token_id,
            1,
            mask=(token_id >= 0) & (token_id < vocab_size),
        )

    @triton.jit
    def _greedy_blocks_kernel(
        logits,
        penalized,
        block_maxima,
        block_sums,
        block_tokens,
        vocab_size,
        penalty,
        BLOCK_SIZE: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        scores = tl.load(logits + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        apply_penalty = tl.load(penalized + offsets, mask=mask, other=0).to(tl.int1)
        adjusted = tl.where(scores < 0.0, scores * penalty, scores / penalty)
        scores = tl.where(apply_penalty, adjusted, scores)
        maximum = tl.max(scores, axis=0)
        probabilities = tl.exp(scores - maximum)
        local_token = tl.argmax(scores, axis=0, tie_break_left=True)
        tl.store(block_maxima + block, maximum)
        tl.store(block_sums + block, tl.sum(probabilities, axis=0))
        tl.store(block_tokens + block, block * BLOCK_SIZE + local_token)

    @triton.jit
    def _lm_head_blocks_kernel(
        hidden,
        weight,
        penalized,
        block_maxima,
        block_sums,
        block_tokens,
        vocab_size,
        penalty,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets_v = block * BLOCK_V + tl.arange(0, BLOCK_V)
        vocab_mask = offsets_v < vocab_size
        accumulator = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for start_d in range(0, HIDDEN_SIZE, BLOCK_D):
            offsets_d = start_d + tl.arange(0, BLOCK_D)
            hidden_mask = offsets_d < HIDDEN_SIZE
            hidden_values = tl.load(
                hidden + offsets_d,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            weights = tl.load(
                weight + offsets_v[:, None] * HIDDEN_SIZE + offsets_d[None, :],
                mask=vocab_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(weights * hidden_values[None, :], axis=1)
        scores = accumulator.to(tl.bfloat16).to(tl.float32)
        scores = tl.where(vocab_mask, scores, -float("inf"))
        apply_penalty = tl.load(
            penalized + offsets_v,
            mask=vocab_mask,
            other=0,
        ).to(tl.int1)
        adjusted = tl.where(scores < 0.0, scores * penalty, scores / penalty)
        scores = tl.where(apply_penalty, adjusted, scores)
        maximum = tl.max(scores, axis=0)
        probabilities = tl.exp(scores - maximum)
        local_token = tl.argmax(scores, axis=0, tie_break_left=True)
        tl.store(block_maxima + block, maximum)
        tl.store(block_sums + block, tl.sum(probabilities, axis=0))
        tl.store(block_tokens + block, block * BLOCK_V + local_token)

    @triton.jit
    def _greedy_reduce_kernel(
        block_maxima,
        block_sums,
        block_tokens,
        token,
        logprob,
        num_blocks,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_blocks
        maxima = tl.load(block_maxima + offsets, mask=mask, other=-float("inf"))
        sums = tl.load(block_sums + offsets, mask=mask, other=0.0)
        tokens = tl.load(block_tokens + offsets, mask=mask, other=0)
        maximum = tl.max(maxima, axis=0)
        total = tl.sum(sums * tl.exp(maxima - maximum), axis=0)
        winning_block = tl.argmax(maxima, axis=0, tie_break_left=True)
        winning_token = tl.sum(tl.where(offsets == winning_block, tokens, 0), axis=0)
        tl.store(token, winning_token)
        tl.store(logprob, -tl.log(total))


@dataclass
class GreedyWorkspace:
    vocab_size: int
    block_maxima: torch.Tensor
    block_sums: torch.Tensor
    block_tokens: torch.Tensor
    token: torch.Tensor
    logprob: torch.Tensor
    penalized: torch.Tensor

    @classmethod
    def allocate(cls, vocab_size: int, device: torch.device) -> GreedyWorkspace:
        max_blocks = (vocab_size + LM_HEAD_BLOCK_SIZE - 1) // LM_HEAD_BLOCK_SIZE
        return cls(
            vocab_size=vocab_size,
            block_maxima=torch.empty(max_blocks, dtype=torch.float32, device=device),
            block_sums=torch.empty(max_blocks, dtype=torch.float32, device=device),
            block_tokens=torch.empty(max_blocks, dtype=torch.int32, device=device),
            token=torch.empty((1, 1), dtype=torch.long, device=device),
            logprob=torch.empty((1, 1), dtype=torch.float32, device=device),
            penalized=torch.zeros(vocab_size, dtype=torch.bool, device=device),
        )

    def reset(self) -> None:
        self.penalized.zero_()


def supports_fused_greedy(logits: torch.Tensor, workspace: GreedyWorkspace) -> bool:
    return (
        triton is not None
        and logits.is_cuda
        and logits.ndim == 2
        and logits.shape == (1, workspace.vocab_size)
        and logits.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and logits.is_contiguous()
        and workspace.penalized.device == logits.device
    )


def mark_penalized(workspace: GreedyWorkspace, token: torch.Tensor) -> None:
    """Mark one generated token in graph-safe repetition-penalty state."""
    if (
        triton is None
        or not token.is_cuda
        or token.numel() != 1
        or token.dtype != torch.long
        or token.device != workspace.penalized.device
    ):
        raise ValueError("unsupported token for EmbodiInfer Triton repetition state")
    _mark_penalized_kernel[(1,)](
        workspace.penalized,
        token,
        workspace.vocab_size,
        num_warps=1,
    )


def supports_fused_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    workspace: GreedyWorkspace,
) -> bool:
    return (
        triton is not None
        and hidden.is_cuda
        and hidden.dtype == weight.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.shape[0] == 1
        and weight.ndim == 2
        and weight.shape == (workspace.vocab_size, hidden.shape[-1])
        and hidden.is_contiguous()
        and weight.is_contiguous()
        and workspace.penalized.device == hidden.device
    )


def _reduce_workspace(workspace: GreedyWorkspace, num_blocks: int) -> None:
    _greedy_reduce_kernel[(1,)](
        workspace.block_maxima,
        workspace.block_sums,
        workspace.block_tokens,
        workspace.token,
        workspace.logprob,
        num_blocks,
        BLOCK_SIZE=triton.next_power_of_2(num_blocks),
        num_warps=8,
    )


def fused_greedy(
    logits: torch.Tensor,
    workspace: GreedyWorkspace,
    repetition_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not supports_fused_greedy(logits, workspace):
        raise ValueError("unsupported input for EmbodiInfer Triton greedy sampling")
    num_blocks = triton.cdiv(workspace.vocab_size, LOGIT_BLOCK_SIZE)
    _greedy_blocks_kernel[(num_blocks,)](
        logits,
        workspace.penalized,
        workspace.block_maxima,
        workspace.block_sums,
        workspace.block_tokens,
        workspace.vocab_size,
        repetition_penalty,
        BLOCK_SIZE=LOGIT_BLOCK_SIZE,
        num_warps=8,
    )
    _reduce_workspace(workspace, num_blocks)
    return workspace.token, workspace.logprob


def fused_lm_head_greedy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    workspace: GreedyWorkspace,
    repetition_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not supports_fused_lm_head(hidden, weight, workspace):
        raise ValueError("unsupported input for EmbodiInfer Triton LM-head greedy sampling")
    num_blocks = triton.cdiv(workspace.vocab_size, LM_HEAD_BLOCK_SIZE)
    _lm_head_blocks_kernel[(num_blocks,)](
        hidden,
        weight,
        workspace.penalized,
        workspace.block_maxima,
        workspace.block_sums,
        workspace.block_tokens,
        workspace.vocab_size,
        repetition_penalty,
        HIDDEN_SIZE=hidden.shape[-1],
        BLOCK_V=LM_HEAD_BLOCK_SIZE,
        BLOCK_D=128,
        num_warps=4,
    )
    _reduce_workspace(workspace, num_blocks)
    return workspace.token, workspace.logprob


__all__ = [
    "GreedyWorkspace",
    "fused_greedy",
    "fused_lm_head_greedy",
    "mark_penalized",
    "supports_fused_greedy",
    "supports_fused_lm_head",
]
