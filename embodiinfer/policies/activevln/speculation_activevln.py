"""Optional greedy verification of public ActiveVLN action-phrase candidates.

Candidates only propose work to batch. Every accepted edge must equal the
model's full-vocabulary argmax with the original repetition penalty. Uncovered
tokens continue through ordinary autoregressive decoding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...exceptions import SessionCancelledError
from .prompt_activevln import parse_r2r_actions

if TYPE_CHECKING:
    from .cache_activevln import ActiveVLNMemory
    from .cuda_graph import ActiveVLNGraphRuntime
    from .modeling_activevln import ActiveVLNGeneration, ActiveVLNPrefix, _ActiveVLNDecoder


@dataclass(frozen=True)
class ActionTokenTree:
    """One fixed token trie with ancestor-only attention and rotary depths."""

    name: str
    token_ids: torch.Tensor  # [1, nodes]
    depths: torch.Tensor  # [nodes], zero-based
    ancestors: torch.Tensor  # [nodes, nodes], includes each node itself
    path_ids: torch.Tensor  # [nodes, maximum depth + 1]
    path_valid: torch.Tensor
    children: dict[tuple[int, int], int]  # (parent index, token) -> node; root is -1

    @classmethod
    def from_paths(
        cls, name: str, paths: Sequence[Sequence[int]], device: torch.device | str
    ) -> ActionTokenTree:
        """Construct a trie from proposals; no path is a constraint on generation."""
        children: dict[tuple[int, int], int] = {}
        node_ids: list[int] = []
        routes: list[list[int]] = []
        for path in paths:
            parent = -1
            for token in path:
                if type(token) is not int or token < 0:
                    raise ValueError("action-tree tokens must be nonnegative Python integers")
                edge = (parent, token)
                if edge not in children:
                    index = len(node_ids)
                    children[edge] = index
                    node_ids.append(token)
                    routes.append(([] if parent < 0 else routes[parent]) + [index])
                parent = children[edge]
        if not node_ids:
            raise ValueError("an action tree must contain at least one token")
        size, depth = len(node_ids), max(map(len, routes))
        ancestors = torch.zeros(size, size, dtype=torch.bool)
        path_ids = torch.zeros(size, depth, dtype=torch.long)
        path_valid = torch.zeros_like(path_ids, dtype=torch.bool)
        for index, route in enumerate(routes):
            ancestors[index, route] = True
            path_ids[index, : len(route)] = torch.tensor([node_ids[i] for i in route])
            path_valid[index, : len(route)] = True
        return cls(
            name,
            torch.tensor([node_ids], dtype=torch.long, device=device),
            torch.tensor([len(route) - 1 for route in routes], dtype=torch.long, device=device),
            ancestors.to(device),
            path_ids.to(device),
            path_valid.to(device),
            children,
        )


def build_action_trees(
    tokenizer,
    eos_token_ids: Sequence[int],
    device: torch.device,
    *,
    repeat_actions: int = 1,
    partition_roots: bool = False,
) -> tuple[ActionTokenTree, ...]:
    """Propose public phrases and optional repeats, with comma/EOS at every action.

    Repeats only enlarge the candidate tree. They neither require a repeated
    action nor change the original token budget or early-stop checks. Root
    partitioning removes proposals whose first token already disagrees with the
    full-vocabulary model choice; it does not need another forward.
    """
    if type(repeat_actions) is not int or not 1 <= repeat_actions <= 3:
        raise ValueError("repeat_actions must be a Python integer from 1 to 3")
    phrases = [f"move forward {distance}cm" for distance in (25, 50, 75)]
    phrases += [
        f"turn {direction} {angle} degrees" for direction in ("left", "right") for angle in (15, 30, 45)
    ]
    phrases.append("stop")
    trees = []
    for name, space in (("initial", ""), ("continuation", " ")):
        paths = []
        for phrase in phrases:
            for count in range(1, (1 if phrase == "stop" else repeat_actions) + 1):
                text = space + ", ".join([phrase] * count)
                # Encode complete strings so punctuation merges follow this tokenizer.
                paths.append(tokenizer.encode(text + ",", add_special_tokens=False))
                phrase_ids = tokenizer.encode(text, add_special_tokens=False)
                paths.extend([*phrase_ids, eos] for eos in eos_token_ids)
        if partition_roots:
            for root in sorted({path[0] for path in paths}):
                trees.append(
                    ActionTokenTree.from_paths(
                        f"{name}-{root}", [path for path in paths if path[0] == root], device
                    )
                )
        else:
            trees.append(ActionTokenTree.from_paths(name, paths, device))
    return tuple(trees)


def tree_greedy_scores(
    logits: torch.Tensor, memory: ActiveVLNMemory, tree: ActionTokenTree, penalty: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the unchanged full-vocabulary greedy distribution to every ancestor path."""
    from .modeling_activevln import _apply_repetition_penalty

    if memory.length < 1:
        raise ValueError("tree verification requires a nonempty encoded prefix")
    prefix = memory.token_ids.expand(tree.token_ids.shape[1], -1)
    # Padding repeats an already-present prefix token, avoiding an extra penalty
    # for an arbitrary pad ID. Repeated scatter indices receive identical scores.
    path = torch.where(tree.path_valid, tree.path_ids, prefix[0, 0])
    history = torch.cat((prefix, path), dim=1)
    effective = _apply_repetition_penalty(logits, history, penalty)
    chosen = effective.argmax(dim=-1)
    logprobs = torch.log_softmax(effective, dim=-1).gather(1, chosen[:, None])[:, 0]
    return chosen, logprobs


def _stop_reason(policy, ids: list[int]) -> str | None:
    if ids[-1] in policy.eos_token_ids:
        return "eos"
    partial = parse_r2r_actions(policy.tokenizer.decode(ids, skip_special_tokens=True).strip())
    return "stop" if partial.valid and partial.actions[-1].name == "stop" else None


def generate_tree_tokens(
    decoder: _ActiveVLNDecoder,
    prefix: ActiveVLNPrefix,
    runtime: ActiveVLNGraphRuntime,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> ActiveVLNGeneration:
    """Verify candidate paths, committing only accepted tokens to private working memory.

    Sampling and gradient-enabled execution never enter this optional path.
    Cancellation can discard a verification block without writing committed state.
    """
    from .modeling_activevln import ActiveVLNGeneration

    policy = decoder.policy
    if policy.do_sample or torch.is_grad_enabled():
        raise ValueError("action-tree verification requires greedy inference")
    memory, logits = prefix.memory, prefix.next_logits
    output: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    ids: list[int] = []
    reason = None

    def check_cancel() -> None:
        if cancelled is not None and cancelled():
            raise SessionCancelledError("ActiveVLN generation was cancelled")

    while len(ids) < policy.max_new_tokens and reason is None:
        check_cancel()
        token, logprob = decoder._select(logits, memory, None)
        first_id = int(token.item())
        tree = next((t for t in runtime.action_trees if (-1, first_id) in t.children), None)
        # Terminal singleton responses do not benefit from tree verification.
        if _stop_reason(policy, ids + [first_id]) is not None:
            tree = None
        with runtime.lock:
            result = None if tree is None else runtime.verify_tree(tree, memory)
            if result is None:
                runtime.counters["tree_fallback_tokens"] += 1
                memory, logits = policy.append_token(memory, token)
                output.append(token)
                logprobs.append(logprob)
                ids.append(first_id)
                reason = _stop_reason(policy, ids)
                continue

            node_logits, packed_kv, positions = result
            chosen, node_logprobs = tree_greedy_scores(
                node_logits[0], memory, tree, policy.repetition_penalty
            )
            chosen_ids = chosen.tolist()
            accepted = []
            node = tree.children[(-1, first_id)]
            while True:
                check_cancel()
                accepted.append(node)
                output.append(token)
                logprobs.append(logprob)
                ids.append(first_id)
                reason = _stop_reason(policy, ids)
                if reason is not None or len(ids) >= policy.max_new_tokens:
                    break
                next_id = chosen_ids[node]
                child = tree.children.get((node, next_id))
                if child is None:
                    break
                token = chosen[node].reshape(1, 1)
                logprob = node_logprobs[node].reshape(1)
                first_id, node = next_id, child

            indices = torch.tensor(accepted, dtype=torch.long, device=packed_kv.device)
            selected_kv = packed_kv.index_select(-2, indices)
            selected_tokens = tree.token_ids.index_select(1, indices)
            selected_positions = positions.index_select(2, indices)
            old_length, next_position = memory.length, memory.next_position
            check_cancel()
            memory.append_chunk(
                selected_kv,
                selected_tokens,
                torch.ones_like(selected_tokens),
                selected_positions,
                next_position=next_position + len(accepted),
            )
            # Physical tree order differs from the linear accepted history.
            # Compact a separate gathered tensor before marking workspace resident.
            runtime.storage[..., old_length : memory.length, :].copy_(selected_kv)
            runtime.bind_memory(memory)
            runtime.counters["tree_accepted_tokens"] += len(accepted)
            logits = node_logits[:, accepted[-1]]

    return ActiveVLNGeneration(
        memory, torch.cat(output, dim=1), torch.stack(logprobs, dim=1), reason or "max_tokens"
    )
