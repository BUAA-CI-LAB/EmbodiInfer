"""Tensor-batched ActiveVLN inference with explicit episode KV ownership."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING
from weakref import ref

import numpy as np
import torch

from ...exceptions import SessionCancelledError
from ...types import Observation
from .cache_activevln import ActiveVLNMemory, _layer_views, _ResidentKVLease
from .cuda_graph import ActiveVLNGraphRuntime, _TextGraph
from .modeling_activevln import ActiveVLNGeneration, HostActiveVLNTurn, PreparedActiveVLNTurn
from .processor_activevln import ProcessedTurn

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


@dataclass
class _PackedGraphBuffers:
    """Inputs of every packed prefill graph; each graph uses leading slices."""

    hidden: torch.Tensor
    positions: torch.Tensor
    offsets: torch.Tensor
    starts: torch.Tensor
    extent: torch.Tensor
    slots: torch.Tensor
    blocks: torch.Tensor
    output: torch.Tensor


class ActiveVLNBatchedRuntime(ActiveVLNGraphRuntime):
    """Share weights and execute vision/text/decode across independent batch rows.

    By default, execution scratch is separate from committed episode KV.
    Opt-in ``resident_kv`` lends one fixed arena to an exclusive batch-one session;
    returned memories and tensor views are borrowed until the session is reset.
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
        grammar_tree_budget: int = 0,
        grammar_prefix_weight: float = 1.0,
        grammar_local_weight: float = 1.0,
        grammar_refresh: int = 64,
        sampling_seed: int = 0,
        gpu_preprocess: bool = False,
        attention_kernel: str = "split",
        incremental_kv_commit: bool = False,
        resident_kv: bool = False,
        vision_kernel: str = "hf",
        skip_padded_queries: bool = False,
        packed_prefill: bool = False,
        packed_prefill_bucket: int = 64,
        decode_linear: bool = False,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive Python integer")
        if type(resident_kv) is not bool:
            raise ValueError("resident_kv must be a boolean")
        if resident_kv and (batch_size != 1 or kv_pool_tokens is None):
            raise ValueError("resident_kv requires batch_size=1 and an explicit kv_pool_tokens capacity")
        if decode_linear and tree_fp32_projection:
            raise ValueError("decode_linear and tree_fp32_projection are alternative projections")
        if packed_prefill and (kv_pool_tokens is None or attention_kernel != "gqa" or not cuda_graph):
            raise ValueError("packed prefill requires CUDA graphs, a shared KV pool and the GQA kernel")
        if type(packed_prefill_bucket) is not int or packed_prefill_bucket < 1:
            raise ValueError("packed_prefill_bucket must be a positive Python integer")
        if attention_kernel not in ("split", "gqa"):
            raise ValueError("attention_kernel must be 'split' or 'gqa'")
        if attention_kernel == "gqa" and not split_attention:
            raise ValueError("the GQA attention kernel replaces split attention")
        if skip_padded_queries and attention_kernel != "gqa":
            raise ValueError("skipping padded queries requires the GQA attention kernel")
        if type(query_bucket_size) is not int or query_bucket_size < 1:
            raise ValueError("query_bucket_size must be a positive Python integer")
        if type(workspace_tokens) is not int or not 1 <= workspace_tokens <= policy.max_context:
            raise ValueError("workspace_tokens must fit the model context limit")
        if policy.training or torch.is_grad_enabled():
            raise ValueError("batched ActiveVLN requires eval() in inference_mode()/no_grad()")
        if type(grammar_tree_budget) is not int or grammar_tree_budget < 0:
            raise ValueError("grammar_tree_budget must be a nonnegative Python integer")
        if grammar_tree_budget and tree_decode:
            raise ValueError("phrase trees and grammar trees are alternative tree modes")
        if policy.do_sample and tree_decode:
            raise ValueError("phrase-tree verification is greedy-only; sampled decoding uses grammar trees")
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
        if gpu_preprocess and not parameter.is_cuda:
            raise ValueError("gpu_preprocess requires a CUDA policy")
        if decode_linear and parameter.is_cuda:
            from ...backend.triton.capability import require_triton

            require_triton(parameter.device)
        self.decode_linear = decode_linear
        self.pooled_kv = kv_pool_tokens is not None
        self.attention_kernel = attention_kernel
        self.skip_padded_queries = skip_padded_queries
        self.packed_prefill = packed_prefill
        self.packed_prefill_bucket = packed_prefill_bucket
        self.packed_graphs: dict[tuple[int, int], _TextGraph] = {}
        self._packed_keys: dict[int, int] = {}
        self._packed_buffers: _PackedGraphBuffers | None = None
        self.incremental_kv_commit = incremental_kv_commit
        self.resident_kv = resident_kv
        self.gpu_preprocess = gpu_preprocess
        policy.set_gpu_preprocess(gpu_preprocess)
        self.workspace_limit = workspace_tokens
        # Packed prefill writes bucket-padding tokens' KV to slots no row reads.
        self._spare_tokens = packed_prefill_bucket if packed_prefill else 0
        self._spare_start = 0
        allocated_tokens = workspace_tokens if kv_pool_tokens is None else kv_pool_tokens + self._spare_tokens
        self.batch_size = batch_size
        self.ragged_batch = True
        self.use_cuda_graph = cuda_graph
        self._generation = 0
        self._owner = object()
        self._pending_generation: int | None = None
        self.capacity_growths = 0
        self.grammar_tree_budget = grammar_tree_budget
        self.grammar_drafter = self.grammar_stops = None
        if grammar_tree_budget:
            from .grammar_tree_activevln import (
                ActionGrammar,
                GrammarDrafter,
                StopReasons,
                prompt_action_phrases,
            )

            self.grammar_stops = StopReasons(policy)
            self.grammar_drafter = GrammarDrafter(
                ActionGrammar(policy.tokenizer, policy.eos_token_ids[0]),
                prefix_weight=grammar_prefix_weight,
                local_weight=grammar_local_weight,
                refresh=grammar_refresh,
                seed_phrases=[
                    policy.tokenizer.encode(phrase, add_special_tokens=False)
                    for phrase in prompt_action_phrases(policy.action_space)
                ],
            )
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
                vision_kernel=vision_kernel,
            )
            if allocated_tokens > self.storage.shape[-2]:
                self._replace_storage(allocated_tokens)
        else:
            if cuda_graph or fused_ops or split_attention or decode_linear or vision_kernel != "hf":
                raise ValueError("graph/fused/split batch execution requires CUDA BF16")
            self.policy, self.device = policy, parameter.device
            self.vision_kernel = vision_kernel
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
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(sampling_seed)
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
        self._owned_rows: list[tuple[torch.Tensor, object, list] | None] = [None] * batch_size
        self._single_kv_lease: _ResidentKVLease | None = None
        self._single_kv_memory: ref[ActiveVLNMemory] | None = None
        if self.resident_kv:
            self._set_resident_layout()

    def _set_resident_layout(self) -> None:
        """Keep the session region and packed-padding region disjoint at fixed addresses."""
        capacity = self.storage.shape[-2] - self._spare_tokens
        self._row_starts = [0]
        self._row_capacities = [capacity]
        self._spare_start = capacity
        self._cache_layout[0].zero_()
        self._cache_layout[1].fill_(capacity)

    def reset_resident_kv(self) -> None:
        """End the exclusive session and invalidate its borrowed memories and prefixes.

        The arena and captured graph addresses are retained. Call only after a
        request has drained; raw tensor views must not be used after reset. Use
        ``memory.fork()`` first to retain an independently owned history. This
        method is a no-op when resident KV is disabled.
        """
        if not self.resident_kv:
            return
        with self.lock:
            if self._single_kv_lease is not None:
                self._single_kv_lease.valid = False
            self._single_kv_lease = self._single_kv_memory = None
            self._pending_generation = None
            self._generation += 1
            self._resident_rows = [None]

    def _check_resident_memory(self, memory: ActiveVLNMemory | None) -> None:
        """Reject interleaved sessions and stale continuations before writing the arena."""
        if not self.resident_kv:
            return
        if memory is not None:
            memory._check_lease()
        current = None if self._single_kv_memory is None else self._single_kv_memory()
        if current is not None:
            if memory is not current:
                raise RuntimeError(
                    "resident KV requires the latest memory; reset before starting another session"
                )
        elif memory is not None and memory._resident_lease is not None:
            raise RuntimeError("cannot import a stale or foreign resident KV memory; import a private fork")

    def _begin_resident_turn(self, memory: ActiveVLNMemory | None) -> None:
        self._check_resident_memory(memory)
        current = None if self._single_kv_memory is None else self._single_kv_memory()
        if current is None:
            if self._single_kv_lease is not None:
                self._single_kv_lease.valid = False
            self._single_kv_lease = _ResidentKVLease()
            self._single_kv_memory = None
            self._resident_rows = [None]

    def reset_stats(self) -> None:
        """Reset graph/tree and cache reuse counts without dropping prepared graphs."""
        super().reset_stats()
        self.counters.update(
            memory_resident_hits=0,
            owned_kv_allocations=0,
            owned_kv_reuses=0,
            pool_relocations=0,
            tree_verified_rows=0,
            tree_candidate_nodes=0,
            tree_padding_nodes=0,
            prefill_actual_tokens=0,
            prefill_padding_tokens=0,
            prefill_packed=0,
            resident_kv_commits=0,
        )

    def _validate(self) -> None:
        parameter = next(self.policy.parameters())
        if self.policy.training or torch.is_grad_enabled():
            raise RuntimeError("batched ActiveVLN execution requires inference mode")
        if self.policy.do_sample and self.action_trees:
            raise RuntimeError("phrase-tree verification is greedy-only")
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
            "grammar_tree_budget": self.grammar_tree_budget,
            "grammar_prior": None if self.grammar_drafter is None else self.grammar_drafter.stats(),
            "tree_query_sizes": list(self._tree_query_sizes()),
            "do_sample": self.policy.do_sample,
            "temperature": self.policy.temperature if self.policy.do_sample else None,
            "top_p": self.policy.top_p if self.policy.do_sample else None,
            "workspace_tokens": self.workspace_limit if self.pooled_kv else self.storage.shape[-2],
            "kv_pool_tokens": self.storage.shape[-2] if self.pooled_kv else None,
            "shared_graph_buffers": True,
            "attention_kernel": self.attention_kernel,
            "skip_padded_queries": self.skip_padded_queries,
            "packed_prefill": self.packed_prefill,
            "packed_prefill_bucket": self.packed_prefill_bucket if self.packed_prefill else None,
            "decode_linear": self.decode_linear,
            "incremental_kv_commit": self.incremental_kv_commit,
            "resident_kv": self.resident_kv,
            "resident_kv_capacity": self.storage.shape[-2] - self._spare_tokens if self.resident_kv else None,
            "gpu_preprocess": self.gpu_preprocess,
            "text_graph_executables": len({id(entry.graph) for entry in self.text_graphs.values()}),
            "tree_graph_executables": len({id(entry.graph) for entry in self.tree_graphs.values()}),
            "packed_graph_executables": len(self.packed_graphs),
        }

    def prepare_host(
        self, observations: Sequence[Observation], initial: Sequence[bool]
    ) -> tuple[HostActiveVLNTurn, ...]:
        """CPU preprocessing only.

        Touches neither the device nor any memory, so callers may run it for
        the next batch while the current batch is on the GPU.
        """
        if not 1 <= len(observations) <= self.batch_size or len(initial) != len(observations):
            raise ValueError("observations and initial flags must align and fit the batch size")
        return tuple(
            self.policy.prepare_host(observation, initial=bool(first))
            for observation, first in zip(observations, initial, strict=True)
        )

    def prepare(
        self,
        observations: Sequence[Observation] | None,
        memories: Sequence[ActiveVLNMemory | None] | None = None,
        *,
        host_turns: Sequence[HostActiveVLNTurn] | None = None,
    ) -> PreparedActiveVLNBatch:
        """Preprocess/H2D each observation without invoking the model or mutating KV.

        ``host_turns`` replaces ``observations`` with already preprocessed rows.
        """
        self._validate()
        if (observations is None) == (host_turns is None):
            raise ValueError("pass exactly one of observations and host_turns")
        count = len(host_turns) if observations is None else len(observations)
        if not 1 <= count <= self.batch_size:
            raise ValueError("observation count must fit the configured batch size")
        memories = [None] * count if memories is None else list(memories)
        if len(memories) != count:
            raise ValueError("memories must align with observations")
        if self.resident_kv:
            self._check_resident_memory(memories[0])
        for memory in memories:
            if memory is not None and memory.token_ids_buffer.device != self.device:
                raise ValueError("batch memories must already be on the model device")
        if host_turns is None:
            host_turns = self.prepare_host(observations, [memory is None for memory in memories])
        turns = self.policy.prepare_devices(host_turns, memories)
        for turn, memory in zip(turns, memories, strict=True):
            past = 0 if memory is None else memory.seq_len
            if past + turn.turn.input_ids.shape[1] > self.policy.max_context:
                raise ValueError("batched turn exceeds the model context limit")
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
        if self.resident_kv:
            raise ValueError("resident KV has fixed capacity; rebuild the runtime with a larger arena")
        limit = (
            self.batch_size * self.policy.max_context + self._spare_tokens
            if self.pooled_kv
            else self.policy.max_context
        )
        if required > limit:
            raise ValueError("batched query exceeds the model context limit")
        capacity = min(limit, 1 << (required - 1).bit_length())
        self.text_graphs.clear()
        self.tree_graphs.clear()
        self.packed_graphs.clear()
        self._graph_buffers.clear()
        self._resident_rows = [None] * self.batch_size
        self._replace_storage(capacity)
        if not self.pooled_kv:
            self._row_capacities = [capacity] * self.batch_size
        self.capacity_growths += 1

    def _tree_query_sizes(self) -> tuple[int, ...]:
        """Tree forward widths with captured graphs; smaller trees pad to the next one."""
        if self.grammar_tree_budget:
            budget = self.grammar_tree_budget
            return tuple(
                sorted({1 + min(budget, n) for n in (2, 8, 32, 128, 512) if n < budget} | {1 + budget})
            )
        return tuple(sorted({t.token_ids.shape[1] for t in self.action_trees}))

    def _plan_rows(self, past: Sequence[int], query: int, count: int) -> None:
        tree_size = max(self._tree_query_sizes(), default=0)
        required = [
            min(self.policy.max_context, n + query + self.policy.max_new_tokens + tree_size)
            if row < count
            else 1
            for row, n in enumerate(past)
        ]
        if self.resident_kv:
            capacity = self.storage.shape[-2] - self._spare_tokens
            if required[0] > capacity:
                raise ValueError(
                    f"resident KV capacity {capacity} cannot fit turn budget {required[0]}; "
                    "history was not truncated and no KV was written"
                )
            return
        if not self.pooled_kv:
            self._ensure_capacity(max(required))
            return
        if all(n <= capacity for n, capacity in zip(required, self._row_capacities, strict=True)):
            return
        alignment = min(2048, self.policy.max_context)
        capacities = [
            min(self.policy.max_context, (n + alignment - 1) // alignment * alignment) for n in required
        ]
        self._ensure_capacity(sum(capacities) + self._spare_tokens)
        starts, offset = [], 0
        for capacity in capacities:
            starts.append(offset)
            offset += capacity
        self._row_starts, self._row_capacities = starts, capacities
        self._spare_start = offset
        self._cache_layout[0].copy_(torch.tensor(starts, device=self.device))
        self._cache_layout[1].copy_(torch.tensor(capacities, device=self.device))
        self._resident_rows = [None] * self.batch_size
        self.counters["pool_relocations"] += 1

    def _row_kv(self, row: int, start: int, end: int) -> torch.Tensor:
        if self.pooled_kv:
            base = self._row_starts[row]
            return self.storage[:, :, :, :, base + start : base + end]
        return self.storage[:, :, row : row + 1, :, start:end]

    def _upload(self, *arrays: np.ndarray) -> list[torch.Tensor]:
        """Device copies of same-dtype host arrays, through one non-blocking transfer.

        The pinned staging block stays reserved until the copy has run, so the
        caller may reuse its arrays at once.
        """
        host = torch.from_numpy(np.concatenate([array.ravel() for array in arrays]))
        if self.device.type == "cuda":
            host = host.pin_memory()
        flat = host.to(self.device, non_blocking=True)
        parts = flat.split([array.size for array in arrays])
        return [part.view(array.shape) for part, array in zip(parts, arrays, strict=True)]

    def _keep_tree_nodes(self, kept: Sequence[tuple[int, int, Sequence[int]]]) -> None:
        """Move accepted tree nodes into linear history order.

        Each entry is (row, start, nodes): the KV of slots start + nodes[i]
        moves to start + i. Every slot is read before any is written.
        """
        if not kept:
            return
        if self.pooled_kv:
            sources = [self._row_starts[row] + start + np.asarray(nodes) for row, start, nodes in kept]
            targets = [self._row_starts[row] + start + np.arange(len(nodes)) for row, start, nodes in kept]
            source, target = self._upload(
                np.concatenate(sources).astype(np.int64), np.concatenate(targets).astype(np.int64)
            )
            self.storage.index_copy_(-2, target, self.storage.index_select(-2, source))
            return
        for row, start, nodes in kept:
            (index,) = self._upload(np.asarray(nodes, dtype=np.int64))
            moved = self._row_kv(row, start, start + max(nodes) + 1).index_select(-2, index)
            self._row_kv(row, start, start + len(nodes)).copy_(moved)

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

    def _query_group(self) -> int:
        attention = self.policy._text.layers[0].self_attn
        return attention.q_proj.out_features // attention.k_proj.out_features

    def _packs(self, query: int) -> bool:
        """Whether prefills padded to ``query`` run packed, matching the padded attention plan."""
        from ...backend.triton.gqa_split_attention import PACKED_MIN_ROWS

        return self.packed_prefill and query * self._query_group() >= PACKED_MIN_ROWS

    def _packed_programs(self, tokens: int) -> int:
        from ...backend.triton.gqa_split_attention import PACKED_BLOCK_M

        return -(-tokens // (PACKED_BLOCK_M // self._query_group())) + self.batch_size

    def _capture_packed(self, tokens: int, key_bucket: int) -> _TextGraph:
        buffers = self._packed_buffers
        hidden, positions = buffers.hidden[:, :tokens], buffers.positions[:, :, :tokens]
        output = buffers.output[:, :tokens]
        layout = (
            buffers.blocks[: self._packed_programs(tokens)],
            buffers.starts,
            buffers.extent,
            buffers.slots[:tokens],
        )
        self._write_lengths.zero_()

        def forward():
            output.copy_(self._text_forward(hidden, positions, buffers.offsets, key_bucket, packed=layout))
            return output

        graph, output = self._capture(forward, self.device, self.pool)
        return _TextGraph(graph, hidden, positions, buffers.offsets, output)

    def _capture_packed_contexts(self, longest: int, context_buckets: Sequence[int]) -> None:
        """Packed prefill graphs for every bucketed token count up to B turns of ``longest``.

        Graphs with the same KV partition count share the largest key bound:
        a sufficient bound never enters a packed launch's partitions.
        """
        from ...backend.triton.gqa_split_attention import PACKED_MIN_ROWS, gqa_launch_config

        bucket = self.packed_prefill_bucket
        most = -(-self.batch_size * longest // bucket) * bucket
        if self._packed_buffers is None or self._packed_buffers.hidden.shape[1] < most:
            self.packed_graphs.clear()
            hidden = next(self.policy.parameters()).new_zeros(
                1, most, self.policy._text.embed_tokens.weight.shape[1]
            )
            rows = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
            self._packed_buffers = _PackedGraphBuffers(
                hidden,
                torch.zeros(3, 1, most, device=self.device, dtype=torch.long),
                rows,
                torch.zeros_like(rows),
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.zeros(most, device=self.device, dtype=torch.long),
                torch.full((self._packed_programs(most),), -1, device=self.device, dtype=torch.long),
                torch.empty_like(hidden),
            )
        for key in context_buckets:
            splits = gqa_launch_config(PACKED_MIN_ROWS, key)[4]
            self._packed_keys[splits] = max(key, self._packed_keys.get(splits, 0))
        for key in self._packed_keys.values():
            for tokens in range(bucket, most + 1, bucket):
                if (tokens, key) not in self.packed_graphs:
                    self.packed_graphs[(tokens, key)] = self._capture_packed(tokens, key)

    def _packed_graph(self, sizes: Sequence[int], query: int, past: Sequence[int]) -> _TextGraph | None:
        if not self._packs(query):
            return None
        from ...backend.triton.gqa_split_attention import PACKED_MIN_ROWS, gqa_launch_config

        key_bucket = self._key_bucket(max(past) + query)
        key = self._packed_keys.get(gqa_launch_config(PACKED_MIN_ROWS, key_bucket)[4], 0)
        bucket = self.packed_prefill_bucket
        tokens = -(-sum(sizes) // bucket) * bucket
        return self.packed_graphs.get((tokens, key)) if key >= key_bucket else None

    def _packed_forward(
        self,
        entry: _TextGraph,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        past: Sequence[int],
        sizes: Sequence[int],
        query: int,
    ) -> torch.Tensor:
        """Replay a packed prefill graph on back-to-back turns; returns each row's last hidden state."""
        from ...backend.triton.gqa_split_attention import packed_query_blocks

        tokens, total = entry.hidden.shape[1], hidden.shape[0]
        programs = self._packed_programs(tokens)
        bounds = np.cumsum([0, *sizes])
        pad = [0] * (self.batch_size - len(sizes))
        slots = [self._row_starts[row] + past[row] + np.arange(size) for row, size in enumerate(sizes)]
        slots.append(self._spare_start + np.arange(tokens - total))
        offsets, lengths, starts, extent, slots, blocks, ends = self._upload(
            np.asarray(past, dtype=np.int64),
            np.asarray([*sizes, *pad], dtype=np.int64),
            np.asarray([*bounds[:-1], *pad], dtype=np.int64),
            np.asarray([query], dtype=np.int64),
            np.concatenate(slots).astype(np.int64),
            np.asarray(packed_query_blocks(sizes, self._query_group(), programs), dtype=np.int64),
            bounds[1:].astype(np.int64) - 1,
        )
        buffers = self._packed_buffers
        self._write_lengths.copy_(lengths)
        buffers.offsets.copy_(offsets)
        buffers.starts.copy_(starts)
        buffers.extent.copy_(extent)
        buffers.slots[:tokens].copy_(slots)
        buffers.blocks[:programs].copy_(blocks)
        entry.hidden[0, :total].copy_(hidden)
        entry.positions[:, 0, :total].copy_(positions)
        entry.graph.replay()
        self.counters["prefill_replays"] += 1
        self.counters["prefill_packed"] += 1
        self.counters["prefill_padding_tokens"] += tokens - total
        return entry.output[0].index_select(0, ends)

    def prewarm(self, observations: Sequence[Observation], *, context_buckets: Sequence[int]) -> None:
        """Capture supplied input shapes before measurement, without retaining answers."""
        self._validate()
        if self.resident_kv and self._single_kv_lease is not None:
            raise RuntimeError("reset_resident_kv before prewarming; capture overwrites the resident arena")
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
                    if not self._packs(query):
                        self._capture_contexts(query, context_buckets)
                if any(self._packs(query) for query in queries):
                    self._capture_packed_contexts(max(queries), context_buckets)
                for query in self._tree_query_sizes():
                    self._capture_contexts(query, context_buckets, tree=True)
            finally:
                self.capture_enabled = False
                self._pending_generation = None
                if self.resident_kv:
                    self._set_resident_layout()
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

    def _key_bucket(self, extent: int) -> int:
        """Round an attention bound within the allocated scratch storage."""
        limit = self.storage.shape[-2]
        if self.pooled_kv:
            limit = min(limit, self.policy.max_context)
        return min(limit, max(512, 1 << (extent - 1).bit_length()))

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
        key_bucket = self._key_bucket(max(past) + query)
        written = [query] * self.batch_size if write_lengths is None else write_lengths
        offsets, lengths = self._upload(np.asarray(past, dtype=np.int64), np.asarray(written, dtype=np.int64))
        self._write_lengths.copy_(lengths)
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
            if self.resident_kv:
                self._check_resident_memory(rows[0].memory)
            sizes = [row.turn.input_ids.shape[1] for row in rows]
            query = self._query_bucket(max(sizes))
            past = [0 if row.memory is None else row.memory.seq_len for row in rows]
            past += [0] * (self.batch_size - len(rows))
            self._plan_rows(past, query, len(rows))
            if self.resident_kv:
                self._begin_resident_turn(rows[0].memory)
            packed = self._packed_graph(sizes, query, past)
            if packed is None:
                token_ids = torch.zeros(self.batch_size, query, device=self.device, dtype=torch.long)
                positions = torch.zeros(3, self.batch_size, query, device=self.device, dtype=torch.long)
            else:
                token_ids = torch.cat([row.turn.input_ids[0] for row in rows])
                positions = torch.cat([row.positions[:, 0] for row in rows], dim=1)
            lengths, next_positions, histories, position_histories = [], [], [], []
            for index, row in enumerate(rows):
                size = sizes[index]
                if packed is None:
                    token_ids[index, :size] = row.turn.input_ids[0]
                    positions[:, index, :size] = row.positions[:, 0]
                memory = row.memory
                ids, pos = row.turn.input_ids[0], row.positions[:, 0]
                if memory is not None:
                    resident = self._resident_rows[index]
                    if (self.resident_kv and memory._resident_lease is self._single_kv_lease) or (
                        resident is not None and resident[0]() is memory and resident[1] == memory.seq_len
                    ):
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
            self.counters["prefill_actual_tokens"] += sum(sizes)
            if packed is None:
                hidden = self._forward(
                    hidden, positions, past, write_lengths=sizes + [0] * (self.batch_size - len(rows))
                )
                self.counters["prefill_padding_tokens"] += self.batch_size * query - sum(sizes)
                selected = hidden[torch.arange(len(rows), device=self.device), [size - 1 for size in sizes]]
            else:
                selected = self._packed_forward(packed, hidden, positions, past, sizes, query)
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
        """Decode all live rows together and publish committed histories only after success."""
        self._validate()
        with self.lock:
            if prefix.owner is not self._owner or self._pending_generation != prefix.generation:
                raise RuntimeError("batched prefix is stale or already consumed")
            try:
                if self.grammar_tree_budget:
                    from .grammar_tree_activevln import generate_batched_grammar_tokens

                    return generate_batched_grammar_tokens(self, prefix, cancelled=cancelled)
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
            if self.policy.do_sample:
                from .grammar_tree_activevln import select_tokens

                token, logprob = select_tokens(effective, self.policy, self.sampling_generator)
                token, logprob = token[:, None], logprob[:, None]
            else:
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
                    parsed = self.policy.parse_actions(text)
                    if parsed.valid and parsed.actions[-1].name == "stop":
                        reasons[row], active[row] = "stop", False
            if not any(active):
                break
        return self._finish_generation(prefix, tokens, scores, lengths, coordinates, reasons)

    _OWNED_KV_ALIGNMENT = 4096

    def _commit_row_kv(self, row: int, length: int, memory: ActiveVLNMemory | None) -> torch.Tensor:
        """Return the resident arena or copy scratch into this row's private KV.

        The buffer replaces the previous ``_row_kv(...).clone().contiguous()``
        per-call allocation. Reuse requires ownership identity: a buffer is
        topped up in place only when the caller supplies exactly the memory it
        was last returned with (rows can be reordered between calls), or when
        its previous owner has been dropped and the slot starts a fresh episode
        (``memory is None``). A different live owner always gets a fresh buffer
        so earlier memories keep their committed histories. Capacity grows by a
        fixed token alignment instead of a full-history reallocation.
        """
        if self.resident_kv:
            if length > self._row_capacities[0]:
                raise ValueError("resident KV commit exceeds its fixed capacity")
            # The uncommitted suffix is already in its final location. Metadata
            # publication below is the transaction's only commit operation.
            return self.storage
        source = self._row_kv(row, 0, length)
        entry = self._owned_rows[row]
        if entry is not None:
            owned, owner, _ = entry
            holder = owner()
            reusable = owned.shape[-2] >= length and (
                (memory is not None and holder is memory) or (memory is None and holder is None)
            )
            if reusable:
                self.counters["owned_kv_reuses"] += 1
                # The previous commit of this same memory already holds its
                # prefix; scratch rows were restored from or match it.
                kept = memory.seq_len if self.incremental_kv_commit and memory is not None else 0
                owned[..., kept:length, :].copy_(source[..., kept:, :])
                return owned
            if holder is None:
                # Let the dead buffer go before allocating its replacement.
                self._owned_rows[row] = None
        alignment = self._OWNED_KV_ALIGNMENT
        capacity = min(
            self.policy.max_context,
            max(alignment, ((length + alignment - 1) // alignment) * alignment),
        )
        capacity = max(capacity, length)
        owned = source.new_empty((*source.shape[:-2], capacity, source.shape[-1]))
        self.counters["owned_kv_allocations"] += 1
        owned[..., :length, :].copy_(source)
        return owned

    def _finish_generation(self, prefix, tokens, scores, lengths, coordinates, reasons):
        """Commit each row; tokens and scores are 1-D tensors or lists of scalar tensors per row."""
        count = len(prefix.prepared.turns)
        generations = []
        for row in range(count):
            response, response_scores = (
                values if isinstance(values, torch.Tensor) else torch.stack(values)
                for values in (tokens[row], scores[row])
            )
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
            old = prefix.prepared.turns[row].memory
            packed = self._commit_row_kv(row, lengths[row], old)
            entry = self._owned_rows[row]
            layers = list(entry[2]) if entry is not None and entry[0] is packed else _layer_views(packed)
            memory = ActiveVLNMemory(
                layers=layers,
                token_ids_buffer=ids,
                attention_buffer=torch.ones_like(ids),
                position_buffer=positions,
                length=lengths[row],
                max_length=self.policy.max_context,
                prompt_hashes=(() if old is None else old.prompt_hashes)
                + (prefix.prepared.turns[row].turn.prompt_sha256,),
                _next_position=coordinates[row],
                _packed_kv=packed,
                _resident_lease=self._single_kv_lease if self.resident_kv else None,
            )
            generations.append(
                ActiveVLNGeneration(memory, response[None], response_scores[None], reasons[row])
            )
            if self.resident_kv:
                self._single_kv_memory = ref(memory)
                self.counters["resident_kv_commits"] += 1
            else:
                self._owned_rows[row] = (packed, ref(memory), layers)
            self._resident_rows[row] = (ref(memory), memory.seq_len)
        return tuple(generations)
