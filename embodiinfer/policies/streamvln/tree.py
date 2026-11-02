"""Budgeted full-vocabulary grammar-tree verification for StreamVLN."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...exceptions import SessionCancelledError
from ...models.streamvln import StreamVLNCache
from ...models.streamvln.optimized import forward_text, project

if TYPE_CHECKING:
    from ...backend.triton import GreedyWorkspace
    from .policy import StreamVLNDecoder, StreamVLNGeneration, StreamVLNPrefix


@dataclass(frozen=True)
class DraftTree:
    """Topological nodes with rotary depths, ancestor visibility and child edges."""

    tokens: torch.Tensor
    depths: torch.Tensor
    ancestors: torch.Tensor
    path_tokens: torch.Tensor
    children: dict[tuple[int, int], int]
    node_count: int

    @classmethod
    def from_parents(cls, tokens: Sequence[int], parents: Sequence[int], width: int) -> DraftTree:
        """Pad a well-formed tree; padding is invisible to every real node."""
        if not tokens or len(tokens) != len(parents) or not len(tokens) <= width or parents[0] != -1:
            raise ValueError("invalid tree node layout")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("tree tokens must be nonnegative integers")
        ids = np.full(width, int(tokens[0]), dtype=np.int64)
        ids[: len(tokens)] = tokens
        ancestors = np.eye(width, dtype=bool)
        children = {}
        for index, parent in enumerate(parents):
            if index and not 0 <= parent < index:
                raise ValueError("tree parents must precede their children")
            if parent >= 0:
                edge = (parent, int(tokens[index]))
                if edge in children:
                    raise ValueError("duplicate tree edge")
                children[edge] = index
                ancestors[index] |= ancestors[parent]
        depths = ancestors.sum(1) - 1
        paths = np.where(ancestors, ids[None], ids[0])
        tensors = [torch.from_numpy(array.copy()) for array in (ids[None], depths[None], ancestors, paths)]
        if torch.cuda.is_available():
            tensors = [tensor.pin_memory() for tensor in tensors]
        return cls(*tensors, children, len(tokens))


class StreamVLNTreeDrafter:
    """Rank public grammar continuations using only the policy's prior responses.

    The deterministic assistant envelope is proposed, then one to four symbolic
    actions and EOS alternatives. An off-grammar prediction always falls back
    to ordinary decoding; this grammar never restricts the model vocabulary.
    """

    def __init__(
        self, prefix_ids: Sequence[int], actions: Sequence[int], eos: Sequence[int], *, refresh: int = 64
    ) -> None:
        if not prefix_ids or not actions or not eos or type(refresh) is not int or refresh < 1:
            raise ValueError("tree drafting requires envelope, action, EOS tokens and positive refresh")
        self.prefix = tuple(prefix_ids)
        self.actions, self.eos = tuple(sorted(actions)), tuple(sorted(eos))
        self.refresh = refresh
        self.counts: dict[tuple[int, ...], Counter[int]] = {}
        self.local: dict[tuple[int, int], Counter[int]] = {}
        self.cache: OrderedDict[tuple, DraftTree | None] = OrderedDict()
        self.observed = 0

    def _state(self, ids: Sequence[int]) -> tuple[int, int] | None:
        prefix_length = min(len(ids), len(self.prefix))
        if tuple(ids[:prefix_length]) != self.prefix[:prefix_length]:
            return None
        if len(ids) < len(self.prefix):
            return (len(ids), 0)
        actions = ids[len(self.prefix) :]
        if any(token not in self.actions for token in actions) or len(actions) > 4:
            return None
        return (len(self.prefix), len(actions))

    def _edges(self, ids: Sequence[int]) -> tuple[int, ...]:
        state = self._state(ids)
        if state is None:
            return ()
        prefix, count = state
        if prefix < len(self.prefix):
            return (self.prefix[prefix],)
        return (() if count == 4 else self.actions) + (self.eos if count else ())

    def observe(self, ids: Sequence[int]) -> None:
        """Update bounded proposal statistics only from a completed model response."""
        for index, token in enumerate(ids):
            prefix = tuple(ids[:index])
            state = self._state(prefix)
            if state is None or token not in self._edges(prefix):
                break
            self.counts.setdefault(prefix, Counter())[int(token)] += 1
            self.local.setdefault(state, Counter())[int(token)] += 1
        self.observed += 1
        if self.observed % self.refresh == 0:
            self.cache.clear()

    def tree(self, pending_ids: Sequence[int], budget: int, width: int) -> DraftTree | None:
        """Build at most budget continuations below an already selected root."""
        key = (tuple(pending_ids), budget, width)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        tokens, parents = [int(pending_ids[-1])], [-1]
        heap, order = [], itertools.count()

        def expand(parent: int, path: tuple[int, ...], log_probability: float) -> None:
            edges = self._edges(path)
            if not edges:
                return
            counts = self.counts.get(path, Counter())
            local = self.local.get(self._state(path), Counter())
            local_total = sum(local[token] for token in edges)
            total = sum(counts[token] for token in edges)
            for token in edges:
                prior = (local[token] + 1 / len(edges)) / (local_total + 1)
                score = log_probability + math.log((counts[token] + prior) / (total + 1))
                heapq.heappush(heap, (-score, next(order), parent, (*path, token)))

        expand(0, tuple(pending_ids), 0.0)
        while heap and len(tokens) < min(width, budget + 1):
            score, _, parent, path = heapq.heappop(heap)
            index = len(tokens)
            tokens.append(path[-1])
            parents.append(parent)
            expand(index, path, -score)
        tree = DraftTree.from_parents(tokens, parents, width) if len(tokens) > 1 else None
        self.cache[key] = tree
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return tree


def tree_scores(
    logits: torch.Tensor, seen: torch.Tensor, paths: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Pack full-vocabulary argmax/log-prob into one small device-to-host result."""
    penalized = seen[None].expand(logits.shape[0], -1).clone()
    penalized.scatter_(1, paths, True)
    values = logits.float()
    if penalty != 1.0:
        values = torch.where(penalized, torch.where(values < 0, values * penalty, values / penalty), values)
    ids = values.argmax(-1)
    scores = values.log_softmax(-1).gather(1, ids[:, None])[:, 0]
    return torch.stack((ids.float(), scores), dim=-1)


class _TreeCall:
    def __init__(
        self,
        decoder: StreamVLNDecoder,
        cache: StreamVLNCache,
        workspace: GreedyWorkspace,
        width: int,
        key_bucket: int,
        *,
        capture: bool,
        pool=None,
    ) -> None:
        self.decoder, self.cache, self.workspace = decoder, cache, workspace
        self.owns_staging_cache = bool(capture and getattr(decoder.policy.backbone, "quantized_layers", ()))
        if self.owns_staging_cache:
            # The graph must not borrow a committed session's prefix when it
            # stages other windows or episodes into the captured addresses.
            backbone = decoder.policy.backbone
            if backbone.quantized_prefill_graphs:
                self.cache, pool = backbone.text_graph_workspace(cache)
            else:
                shape = list(cache.key_storage.shape)
                if backbone.kv_cache_initial_capacity:
                    shape[-2] = backbone.max_context
                self.cache = StreamVLNCache(
                    cache.key_storage.new_empty(shape), cache.value_storage.new_empty(shape), cache.seq_len
                )
            for target, source in (
                (self.cache.key_storage, cache.key_storage),
                (self.cache.value_storage, cache.value_storage),
            ):
                target[..., : cache.seq_len, :].copy_(source[..., : cache.seq_len, :])
        device = cache.key_storage.device
        self.tokens = torch.zeros((1, width), dtype=torch.long, device=device)
        self.depths = torch.zeros_like(self.tokens)
        self.ancestors = torch.eye(width, dtype=torch.bool, device=device)
        self.paths = torch.zeros((width, width), dtype=torch.long, device=device)
        self.position = torch.tensor([cache.seq_len], dtype=torch.long, device=device)
        self.key_bucket, self.graph = key_bucket, None
        self.stream = torch.cuda.current_stream(device).cuda_stream
        decoder.policy.backbone.fuse_projections()
        self.options = decoder.policy.backbone.text_options
        self.parameters = self._parameter_signature()
        self.penalty = decoder.policy.repetition_penalty
        if capture:
            current = torch.cuda.current_stream(device)
            warmup = torch.cuda.Stream(device=device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    self.run()
            current.wait_stream(warmup)
            current.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=pool):
                self.output = self.run()

    def run(self) -> torch.Tensor:
        backbone = self.decoder.policy.backbone
        hidden = forward_text(
            backbone,
            backbone.llm.model.embed_tokens(self.tokens),
            self.cache,
            self.position,
            key_bucket=min(self.key_bucket, self.cache.capacity),
            attention_plan_bucket=(self.key_bucket if backbone.kv_cache_initial_capacity else None),
            depths=self.depths,
            ancestors=self.ancestors,
        )
        logits = project(backbone.llm.lm_head, hidden[0], backbone.text_options, tree=True)
        return tree_scores(
            logits, self.workspace.penalized, self.paths, self.decoder.policy.repetition_penalty
        )

    def matches(self, cache: StreamVLNCache, workspace: GreedyWorkspace) -> bool:
        return (
            cache.key_storage.data_ptr() == self.cache.key_storage.data_ptr()
            and cache.value_storage.data_ptr() == self.cache.value_storage.data_ptr()
            and self._state_matches(workspace)
        )

    def _state_matches(self, workspace: GreedyWorkspace) -> bool:
        return (
            workspace.penalized.data_ptr() == self.workspace.penalized.data_ptr()
            and self.parameters == self._parameter_signature()
            and self.penalty == self.decoder.policy.repetition_penalty
        )

    def can_stage(self, cache: StreamVLNCache, workspace: GreedyWorkspace) -> bool:
        """Whether a fresh window can use this graph through owned KV staging."""
        return (
            self.graph is not None
            and self.owns_staging_cache
            and self._state_matches(workspace)
            and self.decoder.policy.backbone.text_options == self.options
            and all(
                (source.shape[:-2], source.shape[-1], source.dtype, source.device)
                == (target.shape[:-2], target.shape[-1], target.dtype, target.device)
                and source.is_contiguous()
                and target.is_contiguous()
                and cache.seq_len + self.tokens.shape[1] <= self.cache.capacity
                for source, target in (
                    (cache.key_storage, self.cache.key_storage),
                    (cache.value_storage, self.cache.value_storage),
                )
            )
        )

    def _parameter_signature(self) -> tuple:
        return tuple(
            (p.data_ptr(), None if p.is_inference() else p._version)
            for p in itertools.chain(
                self.decoder.policy.backbone.parameters(), self.decoder.policy.backbone.buffers()
            )
        )

    def replay(self, tree: DraftTree, cache: StreamVLNCache) -> torch.Tensor:
        staged = not self.matches(cache, self.workspace)
        if (
            (staged and not self.can_stage(cache, self.workspace))
            or tree.tokens.shape != self.tokens.shape
            or self.decoder.policy.backbone.text_options != self.options
        ):
            raise ValueError("tree graph cache, shape or profile changed")
        if torch.cuda.current_stream(self.tokens.device).cuda_stream != self.stream:
            raise RuntimeError("tree graph belongs to another CUDA stream")
        if cache.seq_len + self.tokens.shape[1] > cache.capacity:
            raise RuntimeError("tree graph exceeded KV capacity")
        for dst, src in (
            (self.tokens, tree.tokens),
            (self.depths, tree.depths),
            (self.ancestors, tree.ancestors),
            (self.paths, tree.path_tokens),
        ):
            dst.copy_(src, non_blocking=True)
        self.position.fill_(cache.seq_len)
        if self.graph is None:
            return self.run()
        if staged:
            # Quantized eager prefill owns a fresh cache at window boundaries.
            # Stage its committed prefix into the captured storage; only the
            # newly verified nodes are copied back. The caller's prefix stays
            # unchanged even if graph execution fails.
            for target, source in (
                (self.cache.key_storage, cache.key_storage),
                (self.cache.value_storage, cache.value_storage),
            ):
                target[..., : cache.seq_len, :].copy_(source[..., : cache.seq_len, :])
        self.graph.replay()
        if staged:
            end = cache.seq_len + self.tokens.shape[1]
            for target, source in (
                (cache.key_storage, self.cache.key_storage),
                (cache.value_storage, self.cache.value_storage),
            ):
                target[..., cache.seq_len : end, :].copy_(source[..., cache.seq_len : end, :])
        return self.output


class StreamVLNTreeRuntime:
    """Own startup-captured tree calls and the policy's bounded proposal cache."""

    def __init__(self, decoder: StreamVLNDecoder) -> None:
        policy = decoder.policy
        self.decoder = decoder
        self.drafter = StreamVLNTreeDrafter(
            policy.processor.response_prefix_ids,
            policy.action_token_ids,
            policy.eos_token_ids,
            refresh=policy.tree_refresh,
        )
        self.graphs: dict[tuple[int, int], _TreeCall] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.counters = dict.fromkeys(
            (
                "tree_blocks",
                "tree_candidates",
                "tree_accepted_tokens",
                "tree_fallback_tokens",
                "tree_replays",
                "tree_staged_replays",
                "tree_eager_fallbacks",
            ),
            0,
        )

    def verify(self, tree: DraftTree, cache: StreamVLNCache, workspace: GreedyWorkspace) -> torch.Tensor:
        """Replay known graph shapes; never capture during measured inference."""
        width = tree.tokens.shape[1]
        if cache.seq_len + width > cache.capacity:
            raise RuntimeError("StreamVLN tree exceeded KV capacity")
        key_bucket = min(cache.capacity, max(512, 1 << (cache.seq_len + width - 1).bit_length()))
        # Grouped-query attention partitions the actual context. Its executable
        # is independent of the upper bound, so one graph covers all positions.
        if (
            self.decoder.policy.backbone.text_options.split_attention
            and self.decoder.policy.backbone.text_options.attention_kernel == "gqa"
        ):
            key_bucket = cache.capacity
            if self.decoder.policy.backbone.kv_cache_initial_capacity:
                key_bucket = self.decoder.policy.backbone.max_context
        key = (width, key_bucket)
        call = self.graphs.get(key)
        if call is None or not (call.matches(cache, workspace) or call.can_stage(cache, workspace)):
            capture = self.decoder._startup_capture_active
            call = _TreeCall(
                self.decoder, cache, workspace, width, key_bucket, capture=capture, pool=self.pool
            )
            if capture:
                self.graphs[key] = call
        self.counters["tree_replays" if call.graph is not None else "tree_eager_fallbacks"] += 1
        if call.graph is not None and not call.matches(cache, workspace):
            self.counters["tree_staged_replays"] += 1
        return call.replay(tree, cache)


def commit_tree_path(cache: StreamVLNCache, path: Sequence[int]) -> StreamVLNCache:
    """Compact only accepted nodes; gather first so overlapping writes are safe."""
    if not path or any(index < 0 or cache.seq_len + index >= cache.capacity for index in path):
        raise ValueError("invalid accepted tree path")
    if len(path) > cache.capacity - cache.seq_len:
        raise RuntimeError("accepted tree path exceeds cache capacity")
    if list(path) != list(range(len(path))):
        source = torch.tensor(path, device=cache.key_storage.device) + cache.seq_len
        destination = torch.arange(len(path), device=source.device) + cache.seq_len
        for storage in (cache.key_storage, cache.value_storage):
            selected = storage.index_select(3, source)
            storage.index_copy_(3, destination, selected)
    return cache.advance(cache.seq_len + len(path))


def generate_tree_tokens(
    decoder: StreamVLNDecoder, prefix: StreamVLNPrefix, *, cancelled: Callable[[], bool] | None = None
) -> StreamVLNGeneration:
    """Verify batched grammar branches and commit the exact selected causal path."""
    from .policy import StreamVLNGeneration

    policy, cache = decoder.policy, prefix.memory.cache
    if cache is None or not cache.key_storage.is_cuda:
        raise ValueError("StreamVLN trees require an encoded CUDA prefix")
    if prefix.reference_cache_length is not None:
        raise ValueError("tree decoding requires its own unextended response prefix")
    workspace = decoder._get_workspace(prefix.next_hidden)
    if decoder._tree_runtime is None:
        decoder._tree_runtime = StreamVLNTreeRuntime(decoder)
    runtime = decoder._tree_runtime
    token, score = decoder._select_token(prefix.next_hidden, workspace, [])
    root = int(token.item())
    root_score = float(score.item())
    ids, logprobs = [], []
    reason = "max_tokens"

    def check_cancel() -> None:
        if cancelled is not None and cancelled():
            raise SessionCancelledError("StreamVLN tree generation was cancelled")

    while len(ids) < policy.max_new_tokens:
        check_cancel()
        available = cache.capacity - cache.seq_len
        if available < 1:
            raise RuntimeError("StreamVLN cache capacity exceeded")
        width = min(1 << policy.tree_budget.bit_length(), available)
        tree = (
            None
            if root in policy.eos_token_ids or width < 2
            else runtime.drafter.tree([*ids, root], min(policy.tree_budget, width - 1), width)
        )
        if tree is None:
            runtime.counters["tree_fallback_tokens"] += 1
            token = torch.tensor([[root]], device=cache.key_storage.device, dtype=torch.long)
            score = torch.tensor([[root_score]], device=token.device)
            need_next = root not in policy.eos_token_ids and len(ids) + 1 < policy.max_new_tokens
            graph = decoder._get_cuda_graph(cache, workspace, 1, need_next)
            if graph is not None:
                cache, _, _, next_token, next_score = graph.replay(cache, token, score, workspace)
            else:
                workspace.penalized[root] = True
                position = torch.tensor([cache.seq_len], device=token.device)
                hidden = policy.backbone.decode_token_graph(cache, token, position)
                cache = cache.advance(cache.seq_len + 1)
                next_token, next_score = (
                    decoder._select_token(hidden, workspace, []) if need_next else (None, None)
                )
            check_cancel()
            ids.append(root)
            logprobs.append(root_score)
            if root in policy.eos_token_ids:
                reason = "eos"
                break
            if need_next:
                root, root_score = int(next_token.item()), float(next_score.item())
            continue
        packed = runtime.verify(tree, cache, workspace).cpu().tolist()
        check_cancel()
        path, accepted, accepted_scores = [0], [root], [root_score]
        node = 0
        while len(ids) + len(accepted) < policy.max_new_tokens and accepted[-1] not in policy.eos_token_ids:
            predicted, predicted_score = packed[node]
            child = tree.children.get((node, int(predicted)))
            if child is None:
                break
            path.append(child)
            accepted.append(int(predicted))
            accepted_scores.append(float(predicted_score))
            node = child
        cache = commit_tree_path(cache, path)
        workspace.penalized[torch.tensor(accepted, device=cache.key_storage.device)] = True
        ids.extend(accepted)
        logprobs.extend(accepted_scores)
        runtime.counters["tree_blocks"] += 1
        runtime.counters["tree_candidates"] += tree.node_count
        runtime.counters["tree_accepted_tokens"] += len(accepted)
        if accepted[-1] in policy.eos_token_ids:
            reason = "eos"
            break
        root, root_score = int(packed[node][0]), float(packed[node][1])
    runtime.drafter.observe(ids)
    return StreamVLNGeneration(
        cache,
        torch.tensor([ids], device=cache.key_storage.device),
        torch.tensor([logprobs], device=cache.key_storage.device),
        reason,
        (),
        False,
    )
