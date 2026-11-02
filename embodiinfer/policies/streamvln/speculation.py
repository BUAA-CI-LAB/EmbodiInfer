"""Full-vocabulary verification of repeated StreamVLN action proposals."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from ...backend.triton import GreedyWorkspace
from ...exceptions import SessionCancelledError
from ...models.streamvln import StreamVLNCache
from .cuda_graph import StreamVLNDecodeGraph

if TYPE_CHECKING:
    from .policy import StreamVLNDecoder


def score_repeated_candidates(
    logits: torch.Tensor, penalized: torch.Tensor, penalty: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each row from the full vocabulary with a shared repetition history.

    All proposals repeat the same root, already marked in ``penalized``. Thus
    each causal candidate prefix has the same set of seen response tokens.
    Scores never include rejected siblings or tokens from an earlier turn.
    """
    scores = logits.float()
    if penalty != 1.0:
        scores = torch.where(
            penalized[None], torch.where(scores < 0, scores * penalty, scores / penalty), scores
        )
    tokens = scores.argmax(dim=-1)
    logprobs = scores.log_softmax(dim=-1).gather(1, tokens[:, None])[:, 0]
    return tokens, logprobs


def decode_speculative(
    decoder: StreamVLNDecoder,
    cache: StreamVLNCache,
    hidden: torch.Tensor,
    workspace: GreedyWorkspace,
    *,
    cancelled: Callable[[], bool] | None = None,
    initial_tokens: list[torch.Tensor] | None = None,
    initial_logprobs: list[torch.Tensor] | None = None,
) -> tuple[StreamVLNCache, list[torch.Tensor], list[torch.Tensor], str]:
    """Commit only verified repeated arrows; decode other roots serially.

    The root is the model's existing greedy choice. One causal forward scores
    up to ``speculative_tokens`` repeated roots; stop accepting at the first
    disagreement, and carry that row's prediction into the next iteration.
    Rejected physical KV tails are overwritten from the accepted logical end.
    CUDA verification retains serial projection shapes and decode arithmetic;
    the surrounding independent row operators are batched and graph-captured.
    """
    policy = decoder.policy
    generated = list(initial_tokens or ())
    logprobs = list(initial_logprobs or ())
    count = sum(part.shape[1] for part in generated)
    reason = "max_tokens"

    def check_cancel() -> None:
        if cancelled is not None and cancelled():
            raise SessionCancelledError("StreamVLN generation was cancelled")

    check_cancel()
    token, logprob = decoder._select_token(hidden, workspace, generated)
    token, logprob = token.clone(), logprob.clone()
    while count < policy.max_new_tokens:
        check_cancel()
        available = cache.capacity - cache.seq_len
        if available <= 0:
            raise RuntimeError("StreamVLN cache capacity exceeded")
        root = int(token.item())
        width = min(policy.speculative_tokens, policy.max_new_tokens - count, available)
        is_eos = root in policy.eos_token_ids
        if root not in policy.action_token_ids or is_eos or width < 2:
            decoder.speculation_stats["speculative_fallback_tokens"] += 1
            need_next = not is_eos and count + 1 < policy.max_new_tokens
            graph = None
            if policy.cuda_graph and token.is_cuda and StreamVLNDecodeGraph.is_available():
                graph = decoder._get_cuda_graph(cache, workspace, 1, need_next)
            if graph is not None:
                cache, ids, scores, next_token, next_logprob = graph.replay(cache, token, logprob, workspace)
            else:
                ids, scores = token.clone(), logprob.clone()
                workspace.penalized[token.reshape(-1)] = True
                if policy.cuda_graph and token.is_cuda and StreamVLNDecodeGraph.is_available():
                    position = torch.tensor([cache.seq_len], dtype=torch.long, device=token.device)
                    next_hidden = policy.backbone.decode_token_graph(cache, token, position)
                    cache = cache.advance(cache.seq_len + 1)
                else:
                    cache, next_hidden = policy.append_token(cache, token)
                next_token = next_logprob = None
                if need_next:
                    next_token, next_logprob = decoder._select_token(
                        next_hidden, workspace, [*generated, ids]
                    )
            check_cancel()
            generated.append(ids)
            logprobs.append(scores)
            count += 1
            if is_eos:
                reason = "eos"
                break
            if need_next:
                if next_token is None or next_logprob is None:
                    raise RuntimeError("StreamVLN serial fallback lost its next token")
                token, logprob = next_token.clone(), next_logprob.clone()
            continue

        draft = token.expand(1, width).contiguous()
        start = cache.seq_len
        if policy.cuda_graph and token.is_cuda and StreamVLNDecodeGraph.is_available():
            states = decoder.verify_candidate_tokens(cache, draft)
            advanced = cache.advance(start + width)
        else:
            embeddings = policy.backbone.llm.model.embed_tokens(draft)
            advanced, states = policy.backbone.forward_embeddings(
                embeddings, cache, project_logits=False, return_hidden_states=True
            )
        workspace.penalized[token.reshape(-1)] = True
        logits = torch.cat([policy.backbone.project_logits(states[:, i]) for i in range(width)], dim=0)
        predictions, scores = score_repeated_candidates(
            logits, workspace.penalized, policy.repetition_penalty
        )
        choices = predictions.tolist()
        check_cancel()
        accepted = 1
        while accepted < width and choices[accepted - 1] == root:
            accepted += 1
        generated.append(draft[:, :accepted].clone())
        logprobs.append(torch.cat((logprob.reshape(1, 1), scores[: accepted - 1].reshape(1, -1)), dim=1))
        cache = advanced.advance(start + accepted)
        count += accepted
        decoder.speculation_stats["speculative_blocks"] += 1
        decoder.speculation_stats["speculative_proposed_tokens"] += width
        decoder.speculation_stats["speculative_accepted_tokens"] += accepted
        token = predictions[accepted - 1].reshape(1, 1).clone()
        logprob = scores[accepted - 1].reshape(1, 1).clone()
    return cache, generated, logprobs, reason
