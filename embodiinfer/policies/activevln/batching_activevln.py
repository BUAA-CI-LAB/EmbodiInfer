"""True batched, greedy ActiveVLN inference with private per-episode memories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING
from weakref import ref

import torch

from ...exceptions import SessionCancelledError
from ...types import Observation
from .cache_activevln import ActiveVLNMemory, _layer_views
from .cuda_graph import ActiveVLNGraphRuntime, _TextGraph
from .modeling_activevln import ActiveVLNGeneration, PreparedActiveVLNTurn
from .processor_activevln import ProcessedTurn
from .prompt_activevln import parse_r2r_actions

if TYPE_CHECKING:
    from .modeling_activevln import ActiveVLNPolicy


@dataclass(frozen=True)
class PreparedActiveVLNBatch:
    """Device-ready observations in caller order, before any model forward."""

    turns: tuple[PreparedActiveVLNTurn, ...]


@dataclass
class ActiveVLNBatchPrefix:
    """Exclusive workspace transaction; another prefill invalidates this prefix."""

    prepared: PreparedActiveVLNBatch
    logits: torch.Tensor
    lengths: list[int]
    positions: list[int]
    token_history: list[torch.Tensor]
    position_history: list[torch.Tensor]
    generation: int
    owner: object


@dataclass
class _BatchGraphBuffers:
    hidden: torch.Tensor
    positions: torch.Tensor
    offsets: torch.Tensor
    output: torch.Tensor
    ancestors: torch.Tensor | None


class ActiveVLNBatchedRuntime(ActiveVLNGraphRuntime):
    """Share weights and execute vision/text/decode across independent batch rows.

    This inference-only API owns execution scratch, never committed episode KV.
    CPU execution is an unfused correctness reference. CUDA graphs are optional
    and captured only by ``prewarm``; missing shapes use counted batched eager
    execution. The generic recurrent engine retains its separate B=1 contract.
    """

    def __init__(
        self,
        policy: ActiveVLNPolicy,
        *,
        batch_size: int,
        workspace_tokens: int,
        query_bucket_size: int = 32,
        cuda_graph: bool = False,
        fused_ops: bool = False,
        split_attention: bool = False,
        tree_decode: bool = False,
        tree_fp32_projection: bool = False,
        tree_repeat_actions: int = 1,
        kv_pool_tokens: int | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive Python integer")
        if type(query_bucket_size) is not int or query_bucket_size < 1:
            raise ValueError("query_bucket_size must be a positive Python integer")
        if type(workspace_tokens) is not int or not 1 <= workspace_tokens <= policy.max_context:
            raise ValueError("workspace_tokens must fit the model context limit")
        if policy.training or torch.is_grad_enabled() or policy.do_sample:
            raise ValueError("batched ActiveVLN requires greedy eval() in inference_mode()/no_grad()")
        if policy._inference_runtime is not None:
            raise ValueError("clear the single-row graph runtime before creating a batched runtime")
        parameter = next(policy.parameters())
        if kv_pool_tokens is not None and (
            type(kv_pool_tokens) is not int
            or not workspace_tokens <= kv_pool_tokens <= batch_size * policy.max_context
        ):
            raise ValueError("kv_pool_tokens must cover workspace_tokens and fit the batch context limit")
        if kv_pool_tokens is not None and parameter.is_cuda and not split_attention:
            raise ValueError("CUDA shared KV pools require split_attention")
        if tree_fp32_projection and not tree_decode:
            raise ValueError("tree_fp32_projection requires tree_decode")
        self.pooled_kv = kv_pool_tokens is not None
        self.workspace_limit = workspace_tokens
        allocated_tokens = workspace_tokens if kv_pool_tokens is None else kv_pool_tokens
        self.batch_size = batch_size
        self.ragged_batch = True
        self.use_cuda_graph = cuda_graph
        self._generation = 0
        self._owner = object()
        self._pending_generation: int | None = None
        self.capacity_growths = 0
        if parameter.device.type == "cuda":
            super().__init__(
                policy,
                batch_size=1 if self.pooled_kv else batch_size,
                workspace_tokens=min(allocated_tokens, policy.max_context),
                query_bucket_size=query_bucket_size,
                fused_ops=fused_ops,
                split_attention=split_attention,
                tree_decode=tree_decode,
                tree_fp32_projection=tree_fp32_projection,
                tree_repeat_actions=tree_repeat_actions,
            )
            if allocated_tokens > self.storage.shape[-2]:
                self._replace_storage(allocated_tokens)
        else:
            if cuda_graph or fused_ops or split_attention:
                raise ValueError("graph/fused/split batch execution requires CUDA BF16")
            self.policy, self.device = policy, parameter.device
            self.weight_pointer = parameter.data_ptr()
            self.query_bucket_size = query_bucket_size
            self.fused_ops = self.split_attention = self.tree_fp32_projection = False
            self.tree_repeat_actions = tree_repeat_actions
            self.lock = RLock()
            self.capture_enabled = False
            self.pool = None
            self.text_graphs, self.tree_graphs, self.vision_graphs = {}, {}, {}
            self.action_trees = ()
            if tree_decode:
                from .speculation_activevln import build_action_trees

                self.action_trees = build_action_trees(
                    policy.tokenizer,
                    policy.eos_token_ids,
                    self.device,
                    repeat_actions=tree_repeat_actions,
                    partition_roots=True,
                )
            if tree_fp32_projection:
                raise ValueError("FP32 tree projection requires CUDA BF16")
            attention = policy._text.layers[0].self_attn
            heads = attention.k_proj.out_features // attention.head_dim
            self.storage = parameter.new_zeros(
                len(policy._text.layers),
                2,
                1 if self.pooled_kv else batch_size,
                heads,
                allocated_tokens,
                attention.head_dim,
            )
            self.reset_stats()
        self._graph_buffers: dict[tuple[int, bool], _BatchGraphBuffers] = {}
        self._write_lengths = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        self._row_starts = [0] * batch_size
        self._row_capacities = [0 if self.pooled_kv else allocated_tokens] * batch_size
        self._cache_layout = None
        if self.pooled_kv:
            self._cache_layout = (
                torch.zeros(batch_size, device=self.device, dtype=torch.long),
                torch.full((batch_size,), allocated_tokens, device=self.device, dtype=torch.long),
            )
        self._resident_rows = [None] * batch_size

    def reset_stats(self) -> None:
        """Reset graph/tree and cache reuse counts without dropping prepared graphs."""
        super().reset_stats()
        self.counters.update(
            memory_resident_hits=0,
            pool_relocations=0,
            tree_verified_rows=0,
            tree_candidate_nodes=0,
            tree_padding_nodes=0,
            prefill_actual_tokens=0,
            prefill_padding_tokens=0,
        )

    def _validate(self) -> None:
        parameter = next(self.policy.parameters())
        if self.policy.training or torch.is_grad_enabled() or self.policy.do_sample:
            raise RuntimeError("batched ActiveVLN execution requires greedy inference mode")
        if parameter.device != self.device or parameter.data_ptr() != self.weight_pointer:
            raise RuntimeError("batched ActiveVLN weights moved; rebuild the runtime")
        if self.device.type == "cuda" and torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("batched ActiveVLN must use its startup CUDA stream")

    def stats(self) -> dict[str, object]:
        """Disclose actual tensor batching, graph coverage and workspace growth."""
        return {
            **super().stats(),
            "batch_size": self.batch_size,
            "batching_mode": "padded_tensor_batch",
            "cuda_graph_requested": self.use_cuda_graph,
            "capacity_growths": self.capacity_growths,
            "tree_decode": bool(self.action_trees),
            "workspace_tokens": self.workspace_limit if self.pooled_kv else self.storage.shape[-2],
            "kv_pool_tokens": self.storage.shape[-2] if self.pooled_kv else None,
            "shared_graph_buffers": True,
            "text_graph_executables": len({id(entry.graph) for entry in self.text_graphs.values()}),
            "tree_graph_executables": len({id(entry.graph) for entry in self.tree_graphs.values()}),
        }

    def prepare(
        self,
        observations: Sequence[Observation],
        memories: Sequence[ActiveVLNMemory | None] | None = None,
    ) -> PreparedActiveVLNBatch:
        """Preprocess/H2D each observation without invoking the model or mutating KV."""
        self._validate()
        if not 1 <= len(observations) <= self.batch_size:
            raise ValueError("observation count must fit the configured batch size")
        memories = [None] * len(observations) if memories is None else list(memories)
        if len(memories) != len(observations):
            raise ValueError("memories must align with observations")
        turns = []
        for row, (observation, memory) in enumerate(zip(observations, memories, strict=True)):
            if memory is not None and memory.token_ids_buffer.device != self.device:
                raise ValueError("batch memories must already be on the model device")
            batch = self.policy.collate([observation], [f"batch-row-{row}"])
            turn = self.policy.prepare_prefix(batch, memory)
            past = 0 if memory is None else memory.seq_len
            if past + turn.turn.input_ids.shape[1] > self.policy.max_context:
                raise ValueError("batched turn exceeds the model context limit")
            turns.append(turn)
        return PreparedActiveVLNBatch(tuple(turns))

    def _query_bucket(self, length: int) -> int:
        bucket = (
            1
            if length == 1
            else (length + self.query_bucket_size - 1) // self.query_bucket_size * self.query_bucket_size
        )
        return min(bucket, self.policy.max_context)

    def _replace_storage(self, capacity: int) -> None:
        shape = (*self.storage.shape[:-2], capacity, self.storage.shape[-1])
        # Prefixes live in separately owned memories, so scratch can be released
        # before replacement; holding both copies would create an artificial OOM.
        self.storage = next(self.policy.parameters()).new_empty(0)
        self.storage = next(self.policy.parameters()).new_zeros(shape)

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.storage.shape[-2]:
            return
        limit = self.batch_size * self.policy.max_context if self.pooled_kv else self.policy.max_context
        if required > limit:
            raise ValueError("batched query exceeds the model context limit")
        capacity = min(limit, 1 << (required - 1).bit_length())
        self.text_graphs.clear()
        self.tree_graphs.clear()
        self._graph_buffers.clear()
        self._resident_rows = [None] * self.batch_size
        self._replace_storage(capacity)
        if not self.pooled_kv:
            self._row_capacities = [capacity] * self.batch_size
        self.capacity_growths += 1

    def _plan_rows(self, past: Sequence[int], query: int, count: int) -> None:
        tree_size = max((t.token_ids.shape[1] for t in self.action_trees), default=0)
        required = [
            min(self.policy.max_context, n + query + self.policy.max_new_tokens + tree_size)
            if row < count
            else 1
            for row, n in enumerate(past)
        ]
        if not self.pooled_kv:
            self._ensure_capacity(max(required))
            return
        if all(n <= capacity for n, capacity in zip(required, self._row_capacities, strict=True)):
            return
        alignment = min(2048, self.policy.max_context)
        capacities = [
            min(self.policy.max_context, (n + alignment - 1) // alignment * alignment) for n in required
        ]
        self._ensure_capacity(sum(capacities))
        starts, offset = [], 0
        for capacity in capacities:
            starts.append(offset)
            offset += capacity
        self._row_starts, self._row_capacities = starts, capacities
        self._cache_layout[0].copy_(torch.tensor(starts, device=self.device))
        self._cache_layout[1].copy_(torch.tensor(capacities, device=self.device))
        self._resident_rows = [None] * self.batch_size
        self.counters["pool_relocations"] += 1

    def _row_kv(self, row: int, start: int, end: int) -> torch.Tensor:
        if self.pooled_kv:
            base = self._row_starts[row]
            return self.storage[:, :, :, :, base + start : base + end]
        return self.storage[:, :, row : row + 1, :, start:end]

    def _capture_batch(self, query: int, key_bucket: int, *, tree: bool = False) -> _TextGraph:
        buffers = self._graph_buffers.get((query, tree))
        if buffers is None:
            parameter = next(self.policy.parameters())
            hidden = parameter.new_zeros(
                self.batch_size, query, self.policy._text.embed_tokens.weight.shape[1]
            )
            buffers = _BatchGraphBuffers(
                hidden,
                torch.zeros(3, self.batch_size, query, device=self.device, dtype=torch.long),
                torch.zeros(self.batch_size, device=self.device, dtype=torch.long),
                torch.empty_like(hidden),
                torch.eye(query, device=self.device, dtype=torch.bool)[None].repeat(self.batch_size, 1, 1)
                if tree
                else None,
            )
            self._graph_buffers[(query, tree)] = buffers
        buffers.offsets.fill_(key_bucket - query)
        self._write_lengths.zero_()

        def forward():
            result = self._text_forward(
                buffers.hidden, buffers.positions, buffers.offsets, key_bucket, buffers.ancestors
            )
            buffers.output.copy_(result)
            return buffers.output

        graph, output = self._capture(forward, self.device, self.pool)
        return _TextGraph(graph, buffers.hidden, buffers.positions, buffers.offsets, output)

    def prewarm(self, observations: Sequence[Observation], *, context_buckets: Sequence[int]) -> None:
        """Capture supplied input shapes before measurement, without retaining answers."""
        self._validate()
        if not self.use_cuda_graph:
            return
        if self._pending_generation is not None:
            raise RuntimeError("cannot prewarm during a pending generation")
        if not observations or not context_buckets:
            raise ValueError("prewarm requires observations and context buckets")
        capacity = min(self.workspace_limit, self.storage.shape[-2])
        if any(type(k) is not int or not 1 <= k <= capacity for k in context_buckets):
            raise ValueError("prewarm context buckets must fit the workspace")
        queries = {1}
        dtype = next(self.policy.parameters()).dtype
        with self.lock:
            # Capture may be called again after inference. Dummy reads must use
            # the whole pool, rather than a previous episode's short segment.
            if self.pooled_kv:
                self._cache_layout[0].zero_()
                self._cache_layout[1].fill_(self.storage.shape[-2])
                self._row_starts = [0] * self.batch_size
                self._row_capacities = [0] * self.batch_size
            self._resident_rows = [None] * self.batch_size
            self.capture_enabled = True
            try:
                for observation in observations:
                    for initial in (True, False):
                        turn = self.policy._processor.process_turn(observation, initial=initial)
                        queries.add(self._query_bucket(turn.input_ids.shape[1]))
                        # The image encoder accepts a packed list of images;
                        # capture occupancy 1..B, including tail batches.
                        for count in range(1, self.batch_size + 1):
                            packed = ProcessedTurn(
                                turn.input_ids,
                                turn.attention_mask,
                                turn.pixel_values.repeat(count, 1),
                                turn.image_grid_thw.repeat(count, 1),
                                "",
                                "",
                            ).to(self.device, dtype)
                            self.vision(packed)
                for query in sorted(queries):
                    self._capture_contexts(query, context_buckets)
                for query in sorted({t.token_ids.shape[1] for t in self.action_trees}):
                    self._capture_contexts(query, context_buckets, tree=True)
            finally:
                self.capture_enabled = False
                self._pending_generation = None
            self.storage.zero_()
            self._resident_rows = [None] * self.batch_size

    def _capture_contexts(self, query: int, context_buckets: Sequence[int], *, tree: bool = False) -> None:
        graphs = self.tree_graphs if tree else self.text_graphs
        keys = sorted({key for key in context_buckets if query <= key})
        if self.split_attention:
            from ...backend.triton.split_attention import split_kv_partitions

            attention = self.policy._text.layers[0].self_attn
            heads = attention.q_proj.out_features // attention.head_dim
            single = [key for key in keys if split_kv_partitions(self.batch_size, heads, query, key) == 1]
            if single:
                # With one partition, end=min(capacity, past+query) is dynamic
                # and independent of every sufficient key bound. Keep multi-
                # partition graphs distinct to preserve their reduction order.
                largest = max(single)
                entry = graphs.get((query, largest))
                if entry is None:
                    entry = self._capture_batch(query, largest, tree=tree)
                for key in single:
                    graphs[(query, key)] = entry
        for key in keys:
            if (query, key) not in graphs:
                graphs[(query, key)] = self._capture_batch(query, key, tree=tree)

    def _forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        past: Sequence[int],
        *,
        ancestors: torch.Tensor | None = None,
        write_lengths: Sequence[int] | None = None,
    ) -> torch.Tensor:
        query = hidden.shape[1]
        extent = max(past) + query
        limit = self.policy.max_context if self.pooled_kv else self.storage.shape[-2]
        key_bucket = min(limit, max(512, 1 << (extent - 1).bit_length()))
        offsets = torch.tensor(past, device=self.device, dtype=torch.long)
        self._write_lengths.copy_(
            torch.tensor(
                [query] * self.batch_size if write_lengths is None else write_lengths, device=self.device
            )
        )
        stage = "tree" if ancestors is not None else "decode" if query == 1 else "prefill"
        graphs = self.tree_graphs if ancestors is not None else self.text_graphs
        entry = graphs.get((query, key_bucket)) if self.use_cuda_graph else None
        if entry is None:
            self.counters[f"{stage}_fallbacks"] += 1
            return self._text_forward(hidden, positions, offsets, key_bucket, ancestors)
        entry.hidden.copy_(hidden)
        entry.positions.copy_(positions)
        entry.cache_position.copy_(offsets)
        if ancestors is not None:
            self._graph_buffers[(query, True)].ancestors.copy_(ancestors)
        entry.graph.replay()
        self.counters[f"{stage}_replays"] += 1
        return entry.output.clone()

    def prefill(self, prepared: PreparedActiveVLNBatch) -> ActiveVLNBatchPrefix:
        """Run one packed vision call and one batched text prefill, starting a transaction."""
        self._validate()
        with self.lock:
            self._generation += 1
            self._pending_generation = None
            rows = prepared.turns
            if not 1 <= len(rows) <= self.batch_size:
                raise ValueError("prepared batch does not fit this runtime")
            query = self._query_bucket(max(row.turn.input_ids.shape[1] for row in rows))
            past = [0 if row.memory is None else row.memory.seq_len for row in rows]
            past += [0] * (self.batch_size - len(rows))
            self._plan_rows(past, query, len(rows))
            token_ids = torch.zeros(self.batch_size, query, device=self.device, dtype=torch.long)
            positions = torch.zeros(3, self.batch_size, query, device=self.device, dtype=torch.long)
            lengths, next_positions, histories, position_histories = [], [], [], []
            for index, row in enumerate(rows):
                size = row.turn.input_ids.shape[1]
                token_ids[index, :size] = row.turn.input_ids[0]
                positions[:, index, :size] = row.positions[:, 0]
                memory = row.memory
                ids, pos = row.turn.input_ids[0], row.positions[:, 0]
                if memory is not None:
                    resident = self._resident_rows[index]
                    if resident is not None and resident[0]() is memory and resident[1] == memory.seq_len:
                        self.counters["memory_resident_hits"] += 1
                    else:
                        destination = self._row_kv(index, 0, past[index])
                        if memory.packed_kv is not None:
                            destination.copy_(memory.packed_kv)
                        else:
                            for layer, (key, value) in enumerate(memory.visible_kv()):
                                destination[layer, 0].copy_(key)
                                destination[layer, 1].copy_(value)
                        self.counters["memory_restores"] += 1
                    ids = torch.cat((memory.token_ids[0], ids))
                    pos = torch.cat((memory.position_ids[:, 0], pos), dim=1)
                self._resident_rows[index] = None
                histories.append(ids)
                position_histories.append(pos)
                lengths.append(past[index] + size)
                next_positions.append(row.next_position)
            combined = ProcessedTurn(
                token_ids,
                torch.ones_like(token_ids),
                torch.cat([row.turn.pixel_values for row in rows]),
                torch.cat([row.turn.image_grid_thw for row in rows]),
                "",
                "",
            )
            hidden = self.policy._text.embed_tokens(token_ids)
            image_features = self.vision(combined) if self.use_cuda_graph else None
            if image_features is None:
                image_features = self.policy._visual(combined.pixel_values, grid_thw=combined.image_grid_thw)
            image_features = getattr(image_features, "pooler_output", image_features)
            if isinstance(image_features, (tuple, list)):
                image_features = image_features[0]
            image_mask = token_ids == self.policy.qwen.config.image_token_id
            if int(image_mask.sum()) != image_features.shape[0]:
                raise ValueError("packed images do not align with batched image tokens")
            hidden = hidden.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(hidden), image_features.to(hidden)
            )
            hidden = self._forward(
                hidden,
                positions,
                past,
                write_lengths=[row.turn.input_ids.shape[1] for row in rows]
                + [0] * (self.batch_size - len(rows)),
            )
            actual_tokens = sum(row.turn.input_ids.shape[1] for row in rows)
            self.counters["prefill_actual_tokens"] += actual_tokens
            self.counters["prefill_padding_tokens"] += self.batch_size * query - actual_tokens
            final_indices = [row.turn.input_ids.shape[1] - 1 for row in rows]
            selected = hidden[torch.arange(len(rows), device=self.device), final_indices]
            logits = self.policy._lm_head(selected)
            self._pending_generation = self._generation
            return ActiveVLNBatchPrefix(
                prepared,
                logits,
                lengths,
                next_positions,
                histories,
                position_histories,
                self._generation,
                self._owner,
            )

    def generate(
        self, prefix: ActiveVLNBatchPrefix, *, cancelled: Callable[[], bool] | None = None
    ) -> tuple[ActiveVLNGeneration, ...]:
        """Decode all live rows together; snapshot private histories only after success."""
        self._validate()
        with self.lock:
            if prefix.owner is not self._owner or self._pending_generation != prefix.generation:
                raise RuntimeError("batched prefix is stale or already consumed")
            try:
                if self.action_trees:
                    from .speculation_activevln import generate_batched_tree_tokens

                    return generate_batched_tree_tokens(self, prefix, cancelled=cancelled)
                return self._generate(prefix, cancelled)
            except BaseException:
                self._resident_rows = [None] * self.batch_size
                raise
            finally:
                self._pending_generation = None

    def _generate(
        self, prefix: ActiveVLNBatchPrefix, cancelled: Callable[[], bool] | None
    ) -> tuple[ActiveVLNGeneration, ...]:
        count = len(prefix.prepared.turns)
        lengths, coordinates = list(prefix.lengths), list(prefix.positions)
        logits = prefix.logits
        seen = torch.zeros_like(logits, dtype=torch.bool)
        for row, history in enumerate(prefix.token_history):
            seen[row].scatter_(0, history, True)
        active = [True] * count
        tokens: list[list[torch.Tensor]] = [[] for _ in range(count)]
        scores: list[list[torch.Tensor]] = [[] for _ in range(count)]
        reasons = ["max_tokens"] * count
        for _ in range(self.policy.max_new_tokens):
            if cancelled is not None and cancelled():
                raise SessionCancelledError("batched ActiveVLN generation was cancelled")
            if any(lengths[row] >= self.policy.max_context for row in range(count) if active[row]):
                raise ValueError("batched generation exceeds the model context limit")
            penalty = self.policy.repetition_penalty
            effective = torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)
            token = effective.argmax(-1, keepdim=True)
            logprob = torch.log_softmax(effective, -1).gather(1, token)
            seen.scatter_(1, token, True)
            ids = torch.zeros(self.batch_size, 1, device=self.device, dtype=torch.long)
            ids[:count] = token
            position = torch.zeros(3, self.batch_size, 1, device=self.device, dtype=torch.long)
            for row in range(count):
                position[:, row, 0] = coordinates[row] if active[row] else 0
            hidden = self._forward(
                self.policy._text.embed_tokens(ids),
                position,
                [lengths[row] if active[row] else 0 for row in range(count)]
                + [0] * (self.batch_size - count),
                write_lengths=[int(value) for value in active] + [0] * (self.batch_size - count),
            )
            logits = self.policy._lm_head(hidden[:count, -1])
            token_values = token[:, 0].tolist()
            for row in range(count):
                if not active[row]:
                    continue
                tokens[row].append(token[row, 0].clone())
                scores[row].append(logprob[row, 0].clone())
                lengths[row] += 1
                coordinates[row] += 1
                if token_values[row] in self.policy.eos_token_ids:
                    reasons[row], active[row] = "eos", False
                else:
                    text = self.policy.tokenizer.decode(
                        torch.stack(tokens[row]).tolist(), skip_special_tokens=True
                    ).strip()
                    parsed = parse_r2r_actions(text)
                    if parsed.valid and parsed.actions[-1].name == "stop":
                        reasons[row], active[row] = "stop", False
            if not any(active):
                break
        return self._finish_generation(prefix, tokens, scores, lengths, coordinates, reasons)

    def _finish_generation(self, prefix, tokens, scores, lengths, coordinates, reasons):
        count = len(prefix.prepared.turns)
        generations = []
        for row in range(count):
            response = torch.stack(tokens[row])
            response_scores = torch.stack(scores[row])[None]
            ids = torch.cat((prefix.token_history[row], response))
            positions = torch.cat(
                (
                    prefix.position_history[row],
                    torch.arange(prefix.positions[row], coordinates[row], device=self.device)[None].expand(
                        3, -1
                    ),
                ),
                dim=1,
            )
            packed = self._row_kv(row, 0, lengths[row]).clone().contiguous()
            old = prefix.prepared.turns[row].memory
            memory = ActiveVLNMemory(
                layers=_layer_views(packed),
                token_ids_buffer=ids,
                attention_buffer=torch.ones_like(ids),
                position_buffer=positions,
                length=lengths[row],
                max_length=self.policy.max_context,
                prompt_hashes=(() if old is None else old.prompt_hashes)
                + (prefix.prepared.turns[row].turn.prompt_sha256,),
                _next_position=coordinates[row],
                _packed_kv=packed,
            )
            self._resident_rows[row] = (ref(memory), memory.seq_len)
            generations.append(ActiveVLNGeneration(memory, response[None], response_scores, reasons[row]))
        return tuple(generations)
