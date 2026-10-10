"""Grammar-trie tree speculation for batched ActiveVLN, exact under sampling.

Candidates come from the complete response grammar: one to three actions whose
numeric arguments are spelled digit by digit, joined by ", " and closed by EOS.
Counts of the policy's own earlier responses rank the grammar's continuations,
and best-first expansion keeps a fixed node budget below the pending token.

Verification draws one token at every node from the unchanged decoding
distribution (repetition penalty over that node's prefix and path, then
temperature and top-p when sampling) and follows the draw while it names a
child. The first draw that names no child becomes the next pending token, so
every emitted token is distributed as in ordinary autoregressive decoding.
Greedy decoding is the zero-temperature case of the same rule.
"""

from __future__ import annotations

import heapq
import itertools
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...exceptions import SessionCancelledError
from .modeling_activevln import _top_p_filter
from .prompt_activevln import SYSTEM_PROMPTS
from .speculation_activevln import _stop_reason

if TYPE_CHECKING:
    from .batching_activevln import ActiveVLNBatchedRuntime, ActiveVLNBatchPrefix
    from .modeling_activevln import ActiveVLNGeneration

_START, _FORWARD, _DIRECTION, _SPACE, _DIGIT, _END, _DONE = range(7)
_NUCLEUS_CANDIDATES = 64
_SAMPLE_ROWS = 256


class ActionGrammar:
    """Token automaton of complete responses, checked against the tokenizer."""

    def __init__(self, tokenizer, eos_token_id: int, *, max_actions: int = 3, max_digits: int = 3) -> None:
        def single(text: str) -> int:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"the action grammar expects {text!r} to be one token")
            return ids[0]

        words = ("turn", "move", "stop")
        self.eos, self.max_actions, self.max_digits = int(eos_token_id), max_actions, max_digits
        self.starts = tuple({single(space + word): word for word in words} for space in ("", " "))
        # Continuation actions start with a space-prefixed token; statistics share one key.
        self.canonical = {single(" " + word): single(word) for word in words}
        self.directions = (single(" left"), single(" right"))
        self.forward, self.space, self.comma = single(" forward"), single(" "), single(",")
        self.digits = tuple(single(str(value)) for value in range(10))
        self.units = {"turn": single(" degrees"), "move": single("cm")}
        self.initial = (_START, 0, None, 0)
        self._edges: dict[tuple, dict[int, tuple]] = {}
        for text, closed in (
            ("turn left 15 degrees, move forward 25cm, stop", False),
            ("move forward 75cm, turn right 100 degrees, turn left 5 degrees", True),
        ):
            state = self.initial
            for token in tokenizer.encode(text, add_special_tokens=False) + ([self.eos] if closed else []):
                state = None if state is None else self.edges(state).get(token)
            if state is None or state[0] != _DONE:
                raise ValueError(f"the tokenizer splits {text!r} differently from the action grammar")

    def edges(self, state: tuple) -> dict[int, tuple]:
        """Legal next tokens mapped to their states."""
        cached = self._edges.get(state)
        if cached is not None:
            return cached
        phase, index, kind, digits = state
        edges: dict[int, tuple] = {}
        if phase == _START:
            for token, word in self.starts[index > 0].items():
                if word == "turn":
                    edges[token] = (_DIRECTION, index, "turn", 0)
                elif word == "move":
                    edges[token] = (_FORWARD, index, "move", 0)
                else:
                    edges[token] = (_DONE, index, None, 0)
        elif phase == _FORWARD:
            edges[self.forward] = (_SPACE, index, kind, 0)
        elif phase == _DIRECTION:
            edges.update(dict.fromkeys(self.directions, (_SPACE, index, kind, 0)))
        elif phase == _SPACE:
            edges[self.space] = (_DIGIT, index, kind, 0)
        elif phase == _DIGIT:
            if digits < self.max_digits:
                for value, token in enumerate(self.digits):
                    if digits or value:
                        edges[token] = (_DIGIT, index, kind, digits + 1)
            if digits:
                edges[self.units[kind]] = (_END, index, None, 0)
        elif phase == _END:
            if index + 1 < self.max_actions:
                edges[self.comma] = (_START, index + 1, None, 0)
            edges[self.eos] = (_DONE, index, None, 0)
        self._edges[state] = edges
        return edges


def prompt_action_phrases(action_space: str) -> list[str]:
    """Actions listed by the public system prompt, used only to seed the prior."""
    listed = re.search(r"choose one action from: (.*?)\. \n", SYSTEM_PROMPTS[action_space])
    if listed is None:
        raise ValueError(f"no action list in the {action_space!r} system prompt")
    return [part.strip().removeprefix("or ").strip() for part in listed.group(1).split(",")]


class _Count:
    __slots__ = ("count", "children")

    def __init__(self) -> None:
        self.count = 0
        self.children: dict[int, _Count] = {}


@dataclass(frozen=True)
class DraftTree:
    """Pending token at index 0 followed by best-first grammar continuations."""

    token_ids: np.ndarray  # [nodes]
    depths: np.ndarray  # [nodes], zero-based
    ancestors: np.ndarray  # [nodes, nodes], includes each node itself
    path_ids: np.ndarray  # [nodes, maximum depth + 1]
    path_valid: np.ndarray
    children: dict[tuple[int, int], int]  # (parent index, token) -> node

    @property
    def size(self) -> int:
        return len(self.token_ids)

    @classmethod
    def from_parents(cls, tokens: list[int], parents: list[int]) -> DraftTree:
        size = len(tokens)
        ancestors = np.zeros((size, size), dtype=bool)
        for index, parent in enumerate(parents):
            if parent >= 0:
                ancestors[index] = ancestors[parent]
            ancestors[index, index] = True
        depths = ancestors.sum(1) - 1
        token_ids = np.asarray(tokens, dtype=np.int64)
        path_ids = np.zeros((size, int(depths.max()) + 1), dtype=np.int64)
        path_valid = np.zeros_like(path_ids, dtype=bool)
        for index in range(size):
            # Parents precede children, so ascending ancestor indices run root to node.
            route = np.flatnonzero(ancestors[index])
            path_ids[index, : len(route)] = token_ids[route]
            path_valid[index, : len(route)] = True
        children = {
            (parent, token): index
            for index, (parent, token) in enumerate(zip(parents, tokens))
            if parent >= 0
        }
        return cls(token_ids, depths, ancestors, path_ids, path_valid, children)


class GrammarDrafter:
    """Budgeted best-first grammar trees ranked by the policy's earlier responses.

    The prior interpolates counts of the identical response prefix with counts
    of the same position inside an action (shared across action slots), which
    in turn fall back to a uniform choice among grammar-legal tokens. Trees are
    cached per response prefix and rebuilt after ``refresh`` observed responses.
    """

    def __init__(
        self,
        grammar: ActionGrammar,
        *,
        prefix_weight: float = 1.0,
        local_weight: float = 1.0,
        refresh: int = 64,
        seed_phrases: Sequence[Sequence[int]] = (),
    ) -> None:
        if not prefix_weight > 0 or not local_weight > 0:
            raise ValueError("grammar prior weights must be positive")
        if type(refresh) is not int or refresh < 1:
            raise ValueError("grammar tree refresh must be a positive Python integer")
        self.grammar = grammar
        self.prefix_weight, self.local_weight, self.refresh = prefix_weight, local_weight, refresh
        self.root = _Count()
        self.local: dict[tuple, dict[int, int]] = {}
        self.local_total: dict[tuple, int] = {}
        self._cache: dict[tuple, DraftTree | None] = {}
        self._unrefreshed = 0
        self.observed = self.grammatical = self.builds = self.cache_hits = 0
        reverse = {initial: continuation for continuation, initial in grammar.canonical.items()}
        for ids in seed_phrases:
            for index in range(grammar.max_actions):
                tokens = list(ids)
                if index and tokens:
                    tokens[0] = reverse.get(tokens[0], tokens[0])
                state, key = (_START, index, None, 0), (index,)
                for token in tokens:
                    step = self._advance(state, key, token)
                    if step is None:
                        break
                    self._bump(key, token)
                    state, key = step
                else:
                    if state[0] == _END:
                        for token in grammar.edges(state):
                            self._bump(key, token)

    def stats(self) -> dict[str, object]:
        return {
            "observed_responses": self.observed,
            "grammatical_responses": self.grammatical,
            "tree_builds": self.builds,
            "tree_cache_hits": self.cache_hits,
            "prefix_weight": self.prefix_weight,
            "local_weight": self.local_weight,
            "refresh": self.refresh,
        }

    def _advance(self, state: tuple, key: tuple, token: int) -> tuple[tuple, tuple | None] | None:
        following = self.grammar.edges(state).get(token)
        if following is None:
            return None
        if following[0] == _DONE:
            return following, None
        if following[0] == _START:
            return following, (following[1],)
        return following, key + (self.grammar.canonical.get(token, token),)

    def _bump(self, key: tuple, token: int) -> None:
        counts = self.local.setdefault(key, {})
        canonical = self.grammar.canonical.get(token, token)
        counts[canonical] = counts.get(canonical, 0) + 1
        self.local_total[key] = self.local_total.get(key, 0) + 1

    def _walk(
        self, token_ids: Sequence[int], update: bool
    ) -> tuple[tuple | None, tuple | None, _Count | None]:
        """Grammar state, local key and prefix-count node after ``token_ids``.

        An off-grammar token loses the prefix node; the walk resynchronizes at
        the next comma, the only unambiguous action boundary.
        """
        grammar = self.grammar
        state, key, node, commas = grammar.initial, (0,), self.root, 0
        for token in token_ids:
            step = None if state is None else self._advance(state, key, token)
            if token == grammar.comma:
                commas += 1
            if step is None:
                node = None
                if token == grammar.comma and commas < grammar.max_actions:
                    state, key = (_START, commas, None, 0), (commas,)
                else:
                    state = key = None
                continue
            if update:
                self._bump(key, token)
            if node is not None:
                child = node.children.get(token)
                if child is None and update:
                    child = node.children[token] = _Count()
                if child is not None and update:
                    child.count += 1
                node = child
            state, key = step
        return state, key, node

    def observe(self, token_ids: Sequence[int]) -> None:
        """Count one finished response; cached trees refresh periodically."""
        state, _, node = self._walk(token_ids, update=True)
        self.observed += 1
        self.grammatical += node is not None and state is not None and state[0] == _DONE
        self._unrefreshed += 1
        if self._unrefreshed >= self.refresh:
            self._cache.clear()
            self._unrefreshed = 0

    def _ranked(self, state: tuple, key: tuple, node: _Count | None) -> list[tuple]:
        edges = self.grammar.edges(state)
        local = self.local.get(key, {})
        local_total = self.local_total.get(key, 0)
        children = {} if node is None else node.children
        prefix_total = sum(child.count for token, child in children.items() if token in edges)
        uniform = self.local_weight / len(edges)
        ranked = []
        for token, following in edges.items():
            canonical = self.grammar.canonical.get(token, token)
            backoff = (local.get(canonical, 0) + uniform) / (local_total + self.local_weight)
            child = children.get(token)
            count = 0 if child is None else child.count
            probability = (count + self.prefix_weight * backoff) / (prefix_total + self.prefix_weight)
            if following[0] == _DONE:
                next_key = None
            elif following[0] == _START:
                next_key = (following[1],)
            else:
                next_key = key + (canonical,)
            ranked.append((math.log(probability), token, following, next_key, child))
        ranked.sort(key=lambda item: -item[0])
        return ranked

    def tree(self, token_ids: Sequence[int], budget: int) -> DraftTree | None:
        """Up to ``budget`` most probable continuations below the last token.

        Returns None when that token is off-grammar or completes the response.
        """
        cache_key = (budget, *token_ids)
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]
        self.builds += 1
        state, key, node = self._walk(token_ids, update=False)
        tree = None
        if state is not None and state[0] != _DONE and budget > 0:
            tokens, parents = [int(token_ids[-1])], [-1]
            heap: list[tuple] = []
            order = itertools.count()

            def expand(index: int, score: float, state: tuple, key: tuple, node: _Count | None) -> None:
                ranked = self._ranked(state, key, node)
                if ranked:
                    heapq.heappush(heap, (-(score + ranked[0][0]), next(order), index, score, ranked, 0))

            expand(0, 0.0, state, key, node)
            while heap and len(tokens) <= budget:
                _, _, parent, score, ranked, rank = heapq.heappop(heap)
                logprob, token, following, next_key, child = ranked[rank]
                index = len(tokens)
                tokens.append(token)
                parents.append(parent)
                # Siblings enter lazily, so the heap holds O(budget) entries.
                if rank + 1 < len(ranked):
                    sibling = ranked[rank + 1][0]
                    heapq.heappush(heap, (-(score + sibling), next(order), parent, score, ranked, rank + 1))
                if following[0] != _DONE:
                    expand(index, score + logprob, following, next_key, child)
            if len(tokens) > 1:
                tree = DraftTree.from_parents(tokens, parents)
        self._cache[cache_key] = tree
        return tree


def select_tokens(
    effective: torch.Tensor, policy, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one token per row of penalized logits under the policy's decoding rule.

    Greedy takes the argmax. Sampling draws exactly from softmax(logits / T)
    restricted to the policy's top-p nucleus, by a Gumbel race. The nucleus is
    located among the largest 64 logits, with a full sort when those fall short.
    Scores are log-probabilities under the same distribution.
    """
    shape = effective.shape[:-1]
    flat = effective.reshape(-1, effective.shape[-1])
    if not policy.do_sample:
        tokens = flat.argmax(-1)
        scores = (flat.gather(1, tokens[:, None])[:, 0] - torch.logsumexp(flat, -1)).float()
        return tokens.view(shape), scores.view(shape)
    tokens = torch.empty(flat.shape[0], device=flat.device, dtype=torch.long)
    scores = torch.empty(flat.shape[0], device=flat.device, dtype=torch.float32)
    for start in range(0, flat.shape[0], _SAMPLE_ROWS):
        rows = slice(start, start + _SAMPLE_ROWS)
        tokens[rows], scores[rows] = _sample(flat[rows].float() / policy.temperature, policy.top_p, generator)
    return tokens.view(shape), scores.view(shape)


def _gumbel(like: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    uniform = torch.rand(like.shape, device=like.device, generator=generator)
    return -torch.log(-torch.log(uniform.clamp_(min=torch.finfo(torch.float32).tiny)))


def _sample(scaled: torch.Tensor, top_p: float, generator) -> tuple[torch.Tensor, torch.Tensor]:
    if top_p >= 1.0:
        tokens = (scaled + _gumbel(scaled, generator)).argmax(-1)
        return tokens, scaled.gather(1, tokens[:, None])[:, 0] - torch.logsumexp(scaled, -1)
    values, indices = scaled.topk(min(_NUCLEUS_CANDIDATES, scaled.shape[-1]), dim=-1)
    cumulative = (values - torch.logsumexp(scaled, -1, keepdim=True)).exp().cumsum(-1)
    # Same boundary as _top_p_filter: keep tokens until the mass first exceeds top_p.
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    nucleus = values.masked_fill(remove, -float("inf"))
    choice = (nucleus + _gumbel(nucleus, generator)).argmax(-1, keepdim=True)
    tokens = indices.gather(1, choice)[:, 0]
    scores = values.gather(1, choice)[:, 0] - torch.logsumexp(nucleus, -1)
    short = cumulative[:, -1] <= top_p
    if bool(short.any()):
        for row in short.nonzero()[:, 0].tolist():
            filtered = _top_p_filter(scaled[row : row + 1], top_p)
            token = (filtered + _gumbel(filtered, generator)).argmax(-1)
            tokens[row] = token[0]
            scores[row] = filtered[0, token[0]] - torch.logsumexp(filtered[0], -1)
    return tokens, scores


def _padded(sets: Sequence[set[int]], pads: Sequence[int]) -> torch.Tensor:
    width = max(len(values) for values in sets)
    array = np.empty((len(sets), width), dtype=np.int64)
    for row, (values, pad) in enumerate(zip(sets, pads, strict=True)):
        array[row].fill(pad)
        array[row, : len(values)] = np.fromiter(values, dtype=np.int64, count=len(values))
    return torch.from_numpy(array)


def _penalize(logits: torch.Tensor, seen: torch.Tensor, paths: torch.Tensor, penalty: float) -> torch.Tensor:
    """Penalize, in place, each token of every node's prefix and ancestor path once.

    Padding repeats a prefix token, whose identical penalized score is simply
    written again; siblings never penalize each other.
    """
    if penalty == 1.0:
        return logits
    indices = torch.cat((seen[:, None].expand(-1, logits.shape[1], -1), paths), dim=2)
    values = logits.gather(2, indices)
    logits.scatter_(2, indices, torch.where(values < 0, values * penalty, values / penalty))
    return logits


def generate_batched_grammar_tokens(
    runtime: ActiveVLNBatchedRuntime,
    prefix: ActiveVLNBatchPrefix,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[ActiveVLNGeneration, ...]:
    """Decode all live rows with budgeted grammar trees, one tensor forward per block.

    Rows choose trees from their own response prefixes. A row whose pending
    token is terminal or off-grammar joins the same forward as a singleton.
    Only accepted nodes enter each row's linear KV history.
    """
    policy, device = runtime.policy, runtime.device
    drafter, budget, generator = (
        runtime.grammar_drafter,
        runtime.grammar_tree_budget,
        runtime.sampling_generator,
    )
    count, batch = len(prefix.prepared.turns), runtime.batch_size
    sizes = runtime._tree_query_sizes()
    penalty = policy.repetition_penalty
    lengths, coordinates = list(prefix.lengths), list(prefix.positions)
    seen = [set(torch.unique(history).tolist()) for history in prefix.token_history]
    pads = [int(history[0]) for history in prefix.token_history]
    tokens: list[list[int]] = [[] for _ in range(count)]
    scores: list[list[torch.Tensor]] = [[] for _ in range(count)]
    reasons, active = ["max_tokens"] * count, [True] * count

    def check_cancel() -> None:
        if cancelled is not None and cancelled():
            raise SessionCancelledError("batched ActiveVLN grammar-tree generation was cancelled")

    first = torch.tensor(pads, device=device)[:, None, None]
    effective = _penalize(prefix.logits[:, None].clone(), _padded(seen, pads).to(device), first, penalty)
    pending, pending_scores = select_tokens(effective[:, 0], policy, generator)
    pending_ids, pending_scores = pending.tolist(), list(pending_scores.unbind())

    while any(active):
        check_cancel()
        if any(lengths[row] >= policy.max_context for row in range(count) if active[row]):
            raise ValueError("batched generation exceeds the model context limit")
        trees: list[DraftTree | None] = [None] * count
        for row in range(count):
            if not active[row]:
                continue
            candidate = tokens[row] + [pending_ids[row]]
            if len(candidate) >= policy.max_new_tokens or _stop_reason(policy, candidate) is not None:
                continue
            tree = drafter.tree(candidate, budget)
            if tree is not None and lengths[row] + tree.size > min(
                policy.max_context, runtime._row_capacities[row]
            ):
                tree = None
            trees[row] = tree
        need = max((tree.size for tree in trees if tree is not None), default=1)
        query = 1 if need == 1 else next((size for size in sizes if size >= need), need)
        depth = max((tree.path_ids.shape[1] for tree in trees if tree is not None), default=1)
        input_ids = np.zeros((batch, query), dtype=np.int64)
        positions = np.zeros((batch, query), dtype=np.int64)
        ancestors = np.broadcast_to(np.eye(query, dtype=bool), (batch, query, query)).copy()
        paths = np.empty((count, query, depth), dtype=np.int64)
        write_lengths, offsets = [0] * batch, [0] * batch
        for row in range(count):
            paths[row].fill(pads[row])
            if not active[row]:
                continue
            offsets[row] = lengths[row]
            tree = trees[row]
            if tree is None:
                input_ids[row, 0] = paths[row, 0, 0] = pending_ids[row]
                positions[row, 0] = coordinates[row]
                write_lengths[row] = 1
            else:
                size, tree_depth = tree.path_ids.shape
                input_ids[row, :size] = tree.token_ids
                positions[row, :size] = coordinates[row] + tree.depths
                ancestors[row, :size, :size] = tree.ancestors
                paths[row, :size, :tree_depth] = np.where(tree.path_valid, tree.path_ids, pads[row])
                write_lengths[row] = size
        is_tree = query > 1
        hidden = runtime._forward(
            policy._text.embed_tokens(torch.from_numpy(input_ids).to(device)),
            torch.from_numpy(positions).to(device)[None].expand(3, -1, -1).contiguous(),
            offsets,
            ancestors=torch.from_numpy(ancestors).to(device) if is_tree else None,
            write_lengths=write_lengths,
        )
        logits = runtime._tree_logits(hidden[:count]) if is_tree else policy._lm_head(hidden[:count])
        effective = _penalize(
            logits, _padded(seen, pads).to(device), torch.from_numpy(paths).to(device), penalty
        )
        chosen, chosen_scores = select_tokens(effective, policy, generator)
        chosen_ids = chosen.tolist()
        if is_tree:
            verified = [tree for tree in trees if tree is not None]
            runtime.counters["tree_verified_rows"] += len(verified)
            runtime.counters["tree_candidate_nodes"] += sum(tree.size for tree in verified)
            runtime.counters["tree_padding_nodes"] += batch * query - sum(write_lengths)
        for row in range(count):
            if not active[row]:
                continue
            check_cancel()
            tree = trees[row]
            token, score, node = pending_ids[row], pending_scores[row], 0
            accepted = []
            while True:
                accepted.append(node)
                tokens[row].append(token)
                scores[row].append(score)
                reason = _stop_reason(policy, tokens[row])
                if reason is not None or len(tokens[row]) >= policy.max_new_tokens:
                    reasons[row], active[row] = reason or "max_tokens", False
                    break
                drawn = chosen_ids[row][node]
                child = None if tree is None else tree.children.get((node, drawn))
                if child is None:
                    pending_ids[row], pending_scores[row] = drawn, chosen_scores[row, node]
                    break
                token, score, node = drawn, chosen_scores[row, node], child
            if tree is None:
                runtime.counters["tree_fallback_tokens"] += 1
            else:
                runtime.counters["tree_accepted_tokens"] += len(accepted)
                if accepted != list(range(len(accepted))):
                    # Physical tree order differs from the linear accepted history.
                    index = torch.tensor(accepted, device=device)
                    compacted = runtime._row_kv(row, lengths[row], lengths[row] + tree.size).index_select(
                        -2, index
                    )
                    runtime._row_kv(row, lengths[row], lengths[row] + len(accepted)).copy_(compacted)
            lengths[row] += len(accepted)
            coordinates[row] += len(accepted)
            seen[row].update(tokens[row][-len(accepted) :])
    for row in range(count):
        drafter.observe(tokens[row])
    committed = [list(torch.tensor(row_tokens, device=device).unbind()) for row_tokens in tokens]
    return runtime._finish_generation(prefix, committed, scores, lengths, coordinates, reasons)
