"""Explicit tensor batching with private episode memories and pooled graph scratch."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from itertools import chain
from threading import Lock
from typing import TYPE_CHECKING

import torch

from ...backend.triton.gqa_split_attention import PACKED_BLOCK_M, packed_query_blocks
from ...exceptions import SessionCancelledError
from ...models.streamvln import StreamVLNCache
from ...models.streamvln.optimized import forward_text, project
from ...models.streamvln.prefill import _CapturedTensorCall
from ...types import Observation
from .policy import StreamVLNGeneration, StreamVLNMemory, StreamVLNPrefix, StreamVLNPreparedPrefix
from .tree import DraftTree, StreamVLNTreeDrafter, commit_tree_path, tree_scores

if TYPE_CHECKING:
    from ..decoder import DecodeResult
    from .policy import StreamVLNPolicy


class _GraphInputArena:
    """Share input storage across serial graphs while keeping captured addresses stable.

    A larger shape allocates a power-of-two block rather than resizing a block
    already referenced by a graph. Smaller shapes reuse an existing block.
    Inputs with different names, dtypes or devices never alias each other.
    The owning runtime must serialize all captures and replays on its stream.
    """

    def __init__(self) -> None:
        self.blocks: dict[tuple[str, str, torch.dtype, torch.device], dict[int, torch.Tensor]] = {}

    def bind(self, inputs: dict[str, torch.Tensor], *, namespace: str = "inputs") -> dict[str, torch.Tensor]:
        """Return contiguous fixed-address views initialized from this call's inputs."""
        result = {}
        for name, value in inputs.items():
            blocks = self.blocks.setdefault((namespace, name, value.dtype, value.device), {})
            required = value.numel()
            capacity = min((size for size in blocks if size >= required), default=0)
            if not capacity:
                capacity = 1 << (max(1, required) - 1).bit_length()
                blocks[capacity] = torch.empty(capacity, device=value.device, dtype=value.dtype)
            view = blocks[capacity][:required].view(value.shape)
            view.copy_(value)
            result[name] = view
        return result

    @property
    def allocated_bytes(self) -> int:
        """Count every retained block, including smaller blocks owned by older graphs."""
        return sum(
            block.numel() * block.element_size()
            for blocks in self.blocks.values()
            for block in blocks.values()
        )


class _Call:
    """Fixed-address tensor inputs; graph creation is controlled by the owner."""

    def __init__(
        self,
        run,
        inputs: dict[str, torch.Tensor],
        *,
        capture: bool,
        pool,
        arena: _GraphInputArena | None = None,
    ) -> None:
        self.inputs = (
            arena.bind(inputs)
            if capture and arena is not None
            else {key: value.clone() for key, value in inputs.items()}
        )
        self.run, self.graph = run, None
        if capture:
            device = next(iter(inputs.values())).device
            current = torch.cuda.current_stream(device)
            warmup = torch.cuda.Stream(device=device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    probe = self.run(self.inputs)
            current.wait_stream(warmup)
            current.synchronize()
            if arena is not None:
                self.output = arena.bind({"result": probe}, namespace="outputs")["result"]
            del probe
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=pool):
                if arena is None:
                    self.output = self.run(self.inputs)
                else:
                    # The temporary forward result dies inside capture, so the
                    # graph allocator can reuse it across different shapes.
                    self.output.copy_(self.run(self.inputs))

    def replay(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        for key, source in inputs.items():
            self.inputs[key].copy_(source)
        if self.graph is None:
            return self.run(self.inputs)
        self.graph.replay()
        return self.output


class StreamVLNBatchedRuntime:
    """Run independent episodes in one vision/text/tree batch.

    This explicit inference API does not change the generic engine's B=1
    contract. Memories are private and can be reordered or forked. A resident
    scratch arena avoids reimporting the previous successful rows; failed or
    cancelled calls invalidate that shortcut. Packed prefill projects only live
    tokens. Graphs capture only inside ``startup_capture`` and report misses.
    One runtime belongs to one policy configuration, device and CUDA stream.
    """

    def __init__(
        self,
        policy: StreamVLNPolicy,
        *,
        batch_size: int,
        workspace_tokens: int | None = None,
        kv_pool_tokens: int | None = None,
        packed_prefill: bool = True,
        packed_prefill_bucket: int = 1,
        incremental_kv_commit: bool = True,
        shared_graph_buffers: bool = False,
    ) -> None:
        parameter = next(policy.backbone.parameters())
        options = policy.backbone.text_options
        if policy.training or torch.is_grad_enabled():
            raise ValueError("batched StreamVLN requires eval() and inference_mode()/no_grad()")
        if not parameter.is_cuda or parameter.dtype != torch.bfloat16:
            raise ValueError("batched StreamVLN requires CUDA BF16")
        if not (options.enabled and options.split_attention and options.attention_kernel == "gqa"):
            raise ValueError("batched StreamVLN requires the optimized GQA split-attention path")
        if not policy.cache_history_features or policy.speculative_tokens:
            raise ValueError("batched StreamVLN requires feature history and serial or tree decoding")
        if policy.fast_action_decode and not policy.tree_decode:
            raise ValueError("batched serial decoding requires fast_action_decode=False")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(packed_prefill_bucket) is not int or packed_prefill_bucket < 1:
            raise ValueError("packed_prefill_bucket must be a positive integer")
        if packed_prefill_bucket > 1 and not packed_prefill:
            raise ValueError("prefill buckets require packed_prefill=True")
        if type(shared_graph_buffers) is not bool:
            raise ValueError("shared_graph_buffers must be a boolean")
        capacity = policy.backbone.max_context if workspace_tokens is None else workspace_tokens
        if type(capacity) is not int or not 1 <= capacity <= policy.backbone.max_context:
            raise ValueError("workspace_tokens must fit the model context")
        pool_tokens = batch_size * capacity if kv_pool_tokens is None else kv_pool_tokens
        if type(pool_tokens) is not int or not capacity <= pool_tokens <= batch_size * capacity:
            raise ValueError("kv_pool_tokens must cover one workspace and fit the batch context capacity")
        policy.backbone.fuse_projections()
        self.policy, self.batch_size, self.capacity = policy, batch_size, capacity
        self.device, self.dtype = parameter.device, parameter.dtype
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self.parameters = self._parameter_signature()
        self.profile = self._profile()
        self.packed_prefill, self.incremental_kv_commit = packed_prefill, incremental_kv_commit
        self.packed_prefill_bucket = packed_prefill_bucket
        self.pool_tokens, self.dynamic_pool = pool_tokens, kv_pool_tokens is not None
        shape = (
            len(policy.backbone.llm.model.layers),
            1,
            policy.backbone.num_kv_heads,
            pool_tokens + packed_prefill_bucket - 1,
            policy.backbone.head_dim,
        )
        self.cache = StreamVLNCache(parameter.new_zeros(shape), parameter.new_zeros(shape), 0)
        self.starts = torch.arange(batch_size, device=self.device) * capacity
        self.capacities = torch.full((batch_size,), capacity, device=self.device, dtype=torch.long)
        self._row_starts, self._row_capacities = (
            list(range(0, batch_size * capacity, capacity)),
            [capacity] * batch_size,
        )
        if self.dynamic_pool:
            self._row_capacities = [0] * batch_size
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.vision_graphs = {}, {}
        self.graph_inputs = _GraphInputArena() if shared_graph_buffers else None
        self.capture_active, self.capture_frozen = False, False
        self._lock = Lock()
        self.resident: list[StreamVLNMemory | None] = [None] * batch_size
        self.drafter = StreamVLNTreeDrafter(
            policy.processor.response_prefix_ids,
            policy.action_token_ids,
            policy.eos_token_ids,
            refresh=policy.tree_refresh,
        )
        self.counters = dict.fromkeys(
            (
                "text_replays",
                "text_eager_fallbacks",
                "vision_replays",
                "vision_eager_fallbacks",
                "prefill_live_tokens",
                "prefill_padding_tokens",
                "prefill_rows",
                "imported_kv_tokens",
                "committed_kv_tokens",
                "tree_blocks",
                "tree_candidates",
                "tree_accepted_tokens",
                "serial_tokens",
                "pool_relocations",
            ),
            0,
        )

    def _profile(self) -> tuple:
        p = self.policy
        return (
            p.backbone.text_options,
            p.backbone.vision_kernel,
            p.tree_decode,
            p.tree_budget,
            p.tree_refresh,
            p.cuda_graph,
            p.max_new_tokens,
            p.repetition_penalty,
        )

    def _parameter_signature(self) -> tuple:
        result = []
        for parameter in chain(self.policy.parameters(), self.policy.buffers()):
            # Inference-created tensors have no version counter; storage and
            # profile ownership still apply. Refit must rebuild this runtime.
            version = None if parameter.is_inference() else parameter._version
            result.append((parameter.data_ptr(), version))
        return tuple(result)

    def _encode_primitive_frames(
        self, prepared: Sequence[StreamVLNPreparedPrefix]
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor | None, ...]]]:
        """Encode every retained primitive frame without creating extra text turns.

        Bound each vision invocation to the runtime's episode capacity. Outputs
        must own storage before another vision replay can reuse graph scratch.
        Intermediate features fill only this turn's private history container.
        """
        current = [
            feature.clone() for feature in self._vision(torch.cat([t.current_pixels for t in prepared]))
        ]
        histories = [list(turn.committed.frame_features) for turn in prepared]
        intermediate = [
            (row, step, pixels)
            for row, turn in enumerate(prepared)
            for step, pixels in turn.intermediate_pixels
        ]
        for start in range(0, len(intermediate), self.batch_size):
            chunk = intermediate[start : start + self.batch_size]
            encoded = self._vision(torch.cat([pixels for _, _, pixels in chunk]))
            for (row, step, _), feature in zip(chunk, encoded, strict=True):
                histories[row][step] = self.policy._retain_history_feature(feature)
        return current, [tuple(history) for history in histories]

    def _validate_owner(self) -> None:
        if self.policy.training or torch.is_grad_enabled():
            raise ValueError("batched StreamVLN is inference-only")
        if self._profile() != self.profile or self.parameters != self._parameter_signature():
            raise RuntimeError("StreamVLN batch policy parameters or configuration changed")
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("StreamVLN batch runtime belongs to another CUDA stream")

    @contextmanager
    def startup_capture(self) -> Iterator[None]:
        """Warm up representative calls, then freeze capture before measurement."""
        self._validate_owner()
        if self.capture_active or self.capture_frozen:
            raise RuntimeError("batch startup capture is already active or frozen")
        self.capture_active = True
        try:
            yield
        except BaseException:
            self.graphs.clear()
            self.vision_graphs.clear()
            if self.graph_inputs is not None:
                self.graph_inputs.blocks.clear()
            self.resident = [None] * self.batch_size
            raise
        else:
            self.capture_frozen = True
        finally:
            self.capture_active = False

    def stats(self) -> dict[str, int | bool]:
        """Report actual graph coverage and useful token work."""
        return dict(
            self.counters,
            tree_refresh=self.drafter.refresh,
            text_graphs=len(self.graphs),
            vision_graphs=len(self.vision_graphs),
            capture_frozen=self.capture_frozen,
            workspace_tokens=self.capacity,
            kv_pool_tokens=self.pool_tokens,
            packed_prefill_bucket=self.packed_prefill_bucket,
            shared_graph_buffers=self.graph_inputs is not None,
            graph_buffer_allocated_bytes=0
            if self.graph_inputs is None
            else self.graph_inputs.allocated_bytes,
        )

    def reset_stats(self) -> None:
        """Clear measurement counters without changing graphs or episode state."""
        for key in self.counters:
            self.counters[key] = 0

    def prewarm_prefill(self, shapes: Sequence[Sequence[int]]) -> None:
        """Capture declared per-row token lengths during startup without decoding.

        Shapes describe projected multimodal tokens, not tokenizer IDs. Zero
        embeddings select graph shapes only; no generated answer or episode
        history is retained. Existing memories remain private, and scratch
        residency is invalidated because these probes overwrite its contents.
        Unknown measured shapes continue to use the counted eager fallback.
        """
        self._validate_owner()
        if not self.capture_active or not self.packed_prefill:
            raise RuntimeError("prefill prewarm requires packed execution inside startup_capture")
        self.resident = [None] * self.batch_size
        for shape in shapes:
            sizes = list(shape)
            if not 1 <= len(sizes) <= self.batch_size or any(
                type(size) is not int or not 1 <= size <= self.capacity - self.policy.max_new_tokens
                for size in sizes
            ):
                raise ValueError("prefill warmup lengths must fit each row and its generation reserve")
            self._plan_rows([size + self.policy.max_new_tokens for size in sizes])
            embeddings = [
                torch.zeros(1, size, self.policy.backbone.hidden_size, device=self.device, dtype=self.dtype)
                for size in sizes
            ]
            self._prefill(embeddings, [0] * len(sizes))

    def _row(self, index: int, length: int = 0) -> StreamVLNCache:
        start, capacity = self._row_starts[index], self._row_capacities[index]
        return StreamVLNCache(
            self.cache.key_storage[..., start : start + capacity, :],
            self.cache.value_storage[..., start : start + capacity, :],
            length,
        )

    def _plan_rows(self, required: list[int]) -> None:
        if not self.dynamic_pool:
            return
        required = required + [1] * (self.batch_size - len(required))
        if all(n <= available for n, available in zip(required, self._row_capacities, strict=True)):
            return
        capacities = [min(self.capacity, -(-n // 256) * 256) for n in required]
        if sum(capacities) > self.pool_tokens:
            capacities = required
        if sum(capacities) > self.pool_tokens:
            raise ValueError("pooled KV cannot fit the batch; no history is truncated")
        starts = [sum(capacities[:i]) for i in range(self.batch_size)]
        self.starts.copy_(torch.tensor(starts, device=self.device))
        self.capacities.copy_(torch.tensor(capacities, device=self.device))
        self._row_starts, self._row_capacities = starts, capacities
        # Committed memories own their prefix, so relocation requires no
        # overlapping in-arena move. Reimport them after updating the layout.
        self.resident = [None] * self.batch_size
        self.counters["pool_relocations"] += 1

    def _vision(self, pixels: torch.Tensor) -> torch.Tensor:
        key = tuple(pixels.shape)
        call = self.vision_graphs.get(key)
        if call is None and self.capture_active:
            call = _CapturedTensorCall(
                self.policy.backbone._encode_frames_eager,
                pixels,
                self.pool,
                compile_call=self.policy.backbone.vision_kernel == "hf",
            )
            self.vision_graphs[key] = call
        self.counters["vision_replays" if call is not None else "vision_eager_fallbacks"] += 1
        return call.replay(pixels) if call is not None else self.policy.backbone._encode_frames_eager(pixels)

    def _text(self, inputs: dict[str, torch.Tensor], *, packed: bool, tree: bool) -> torch.Tensor:
        backbone = self.policy.backbone

        def run(values):
            kwargs = dict(
                query_lengths=values["lengths"],
                cache_starts=self.starts[:count],
                cache_capacities=self.capacities[:count],
            )
            if packed:
                kwargs.update(
                    packed=tuple(values[k] for k in ("blocks", "starts", "extent", "slots")),
                    rotary_positions=values["rotary"],
                )
                if "live" in values:
                    kwargs.update(
                        packed_live_tokens=values["live"], packed_padding=self.packed_prefill_bucket - 1
                    )
            elif tree:
                kwargs.update(depths=values["depths"], ancestors=values["ancestors"])
            hidden = forward_text(
                backbone, values["hidden"], self.cache, values["position"], key_bucket=self.capacity, **kwargs
            )
            if "seen" not in values:
                return hidden
            logits = project(backbone.llm.lm_head, hidden, backbone.text_options, tree=tree)
            # Each row has a separate penalty history; no sibling/episode leakage.
            return torch.stack(
                [
                    tree_scores(
                        logits[i], values["seen"][i], values["paths"][i], self.policy.repetition_penalty
                    )
                    for i in range(count)
                ]
            )

        count = inputs["position"].numel()
        key = (packed, tree, tuple((k, tuple(v.shape), v.dtype) for k, v in inputs.items()))
        call = self.graphs.get(key)
        if call is None:
            call = _Call(run, inputs, capture=self.capture_active, pool=self.pool, arena=self.graph_inputs)
            if self.capture_active:
                self.graphs[key] = call
        self.counters["text_replays" if call.graph is not None else "text_eager_fallbacks"] += 1
        return call.replay(inputs)

    def _prefill(self, embeddings: list[torch.Tensor], positions: list[int]) -> torch.Tensor:
        sizes = [e.shape[1] for e in embeddings]
        self.counters["prefill_live_tokens"] += sum(sizes)
        self.counters["prefill_rows"] += len(sizes)
        inputs = dict(
            position=torch.tensor(positions, device=self.device),
            lengths=torch.tensor(sizes, device=self.device),
        )
        if self.packed_prefill:
            starts = [sum(sizes[:i]) for i in range(len(sizes))]
            group = self.policy.backbone.num_heads // self.policy.backbone.num_kv_heads
            programs = sum(-(-n // (PACKED_BLOCK_M // group)) for n in sizes)
            total = sum(sizes)
            bucket = self.packed_prefill_bucket
            tokens = -(-total // bucket) * bucket
            padding = tokens - total
            self.counters["prefill_padding_tokens"] += padding
            if bucket > 1:
                # Each row can contribute one incomplete attention program.
                # Constant table capacity lets different ragged lengths reuse
                # the same token-bucket graph without changing live work.
                programs = -(-tokens // (PACKED_BLOCK_M // group)) + len(sizes) - 1
            hidden = torch.cat(embeddings, 1)
            slots = torch.cat(
                [
                    torch.arange(n, device=self.device) + self._row_starts[i] + positions[i]
                    for i, n in enumerate(sizes)
                ]
            )
            rotary = torch.cat([torch.arange(n, device=self.device) + p for n, p in zip(sizes, positions)])
            if bucket > 1:
                hidden = torch.cat((hidden, hidden.new_zeros(1, padding, hidden.shape[-1])), 1)
                slots = torch.cat((slots, torch.arange(padding, device=self.device) + self.pool_tokens))
                rotary = torch.cat((rotary, torch.zeros(padding, device=self.device, dtype=torch.long)))
                inputs["live"] = torch.tensor([total], device=self.device)
            inputs.update(
                hidden=hidden,
                starts=torch.tensor(starts, device=self.device),
                extent=torch.tensor([max(sizes)], device=self.device),
                blocks=torch.tensor(packed_query_blocks(sizes, group, programs), device=self.device),
                slots=slots,
                rotary=rotary[None],
            )
            hidden = self._text(inputs, packed=True, tree=False)
            return hidden[0, torch.tensor([s + n - 1 for s, n in zip(starts, sizes)], device=self.device)]
        hidden = embeddings[0].new_zeros(len(sizes), max(sizes), embeddings[0].shape[-1])
        for i, embedding in enumerate(embeddings):
            hidden[i, : sizes[i]].copy_(embedding[0])
        inputs["hidden"] = hidden
        hidden = self._text(inputs, packed=False, tree=False)
        return hidden[torch.arange(len(sizes), device=self.device), inputs["lengths"] - 1]

    def _decode(self, initial: torch.Tensor, positions: list[int], check_cancel) -> list[StreamVLNGeneration]:
        p, count = self.policy, len(positions)
        logits = p.backbone.project_logits(initial).float()
        roots = logits.argmax(-1)
        scores = logits.log_softmax(-1).gather(1, roots[:, None])[:, 0]
        pending = torch.stack((roots.float(), scores), -1).cpu().tolist()
        seen = torch.zeros_like(logits, dtype=torch.bool)
        ids, logs, done = [[] for _ in positions], [[] for _ in positions], [False] * count
        reasons = ["max_tokens"] * count
        while not all(done):
            check_cancel()
            width = (
                min(
                    1 << p.tree_budget.bit_length(),
                    min(
                        self._row_capacities[i] - n for i, (n, d) in enumerate(zip(positions, done)) if not d
                    ),
                )
                if p.tree_decode
                else 1
            )
            if width < 1:
                raise RuntimeError("batched StreamVLN decode exceeded KV capacity")
            trees, lengths = [], []
            for i, (token, _) in enumerate(pending):
                tree = None
                if p.tree_decode and not done[i] and int(token) not in p.eos_token_ids:
                    tree = self.drafter.tree([*ids[i], int(token)], min(p.tree_budget, width - 1), width)
                if tree is None:
                    tree = DraftTree.from_parents([int(token)], [-1], width)
                trees.append(tree)
                lengths.append(0 if done[i] else tree.node_count)
            tokens = torch.cat([t.tokens for t in trees]).to(self.device, non_blocking=True)
            inputs = dict(
                hidden=p.backbone.llm.model.embed_tokens(tokens),
                position=torch.tensor([0 if d else n for n, d in zip(positions, done)], device=self.device),
                lengths=torch.tensor(lengths, device=self.device),
                seen=seen,
                paths=torch.stack([t.path_tokens for t in trees]).to(self.device, non_blocking=True),
            )
            if p.tree_decode:
                inputs.update(
                    depths=torch.cat([t.depths for t in trees]).to(self.device, non_blocking=True),
                    ancestors=torch.stack([t.ancestors for t in trees]).to(self.device, non_blocking=True),
                )
            predictions = self._text(inputs, packed=False, tree=p.tree_decode).cpu().tolist()
            check_cancel()
            for i, tree in enumerate(trees):
                if done[i]:
                    continue
                path, accepted, probabilities = [0], [int(pending[i][0])], [float(pending[i][1])]
                node = 0
                while len(ids[i]) + len(accepted) < p.max_new_tokens and accepted[-1] not in p.eos_token_ids:
                    token, score = predictions[i][node]
                    child = tree.children.get((node, int(token)))
                    if child is None:
                        break
                    path.append(child)
                    accepted.append(int(token))
                    probabilities.append(float(score))
                    node = child
                commit_tree_path(self._row(i, positions[i]), path)
                positions[i] += len(path)
                ids[i].extend(accepted)
                logs[i].extend(probabilities)
                seen[i, torch.tensor(accepted, device=self.device)] = True
                if tree.node_count > 1:
                    self.counters["tree_blocks"] += 1
                    self.counters["tree_candidates"] += tree.node_count
                    self.counters["tree_accepted_tokens"] += len(path)
                else:
                    self.counters["serial_tokens"] += 1
                if accepted[-1] in p.eos_token_ids:
                    reasons[i], done[i] = "eos", True
                elif len(ids[i]) == p.max_new_tokens:
                    done[i] = True
                pending[i] = predictions[i][node]
        # Proposal statistics are committed by infer only after every row succeeds.
        return [
            StreamVLNGeneration(
                self._row(i, positions[i]),
                torch.tensor([ids[i]], device=self.device),
                torch.tensor([logs[i]], device=self.device),
                reasons[i],
                (),
                False,
            )
            for i in range(count)
        ]

    def _commit(self, cache: StreamVLNCache, old: StreamVLNCache | None) -> StreamVLNCache:
        start = old.seq_len if old is not None and self.incremental_kv_commit else 0
        if old is not None and self.incremental_kv_commit and old.capacity >= cache.seq_len:
            destination = old
        else:
            capacity = min(self.capacity, 1 << (cache.seq_len - 1).bit_length())
            shape = (*cache.key_storage.shape[:-2], capacity, cache.key_storage.shape[-1])
            destination = StreamVLNCache(
                cache.key_storage.new_empty(shape), cache.value_storage.new_empty(shape), 0
            )
            start = 0
        for source, target in (
            (cache.key_storage, destination.key_storage),
            (cache.value_storage, destination.value_storage),
        ):
            target[..., start : cache.seq_len, :].copy_(source[..., start : cache.seq_len, :])
        self.counters["committed_kv_tokens"] += cache.seq_len - start
        return destination.advance(cache.seq_len)

    @torch.inference_mode()
    def infer(
        self,
        observations: Sequence[Observation],
        request_ids: Sequence[str],
        memories: Sequence[StreamVLNMemory | None] | None = None,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[DecodeResult, ...]:
        """Process one turn per episode; cancellation leaves input prefixes intact."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("StreamVLN batched runtime already has an in-flight call")
        try:
            self._validate_owner()
            count = len(observations)
            if (
                not 1 <= count <= self.batch_size
                or len(request_ids) != count
                or len(set(request_ids)) != count
            ):
                raise ValueError("observations require unique aligned IDs within batch capacity")
            memories = [None] * count if memories is None else list(memories)
            if len(memories) != count:
                raise ValueError("memories must align with observations")

            def check_cancel():
                if cancelled is not None and cancelled():
                    raise SessionCancelledError("batched StreamVLN inference was cancelled")

            check_cancel()
            p = self.policy
            prepared = [
                p.prepare_prefix(p.collate([o], [name]), memory)
                for o, name, memory in zip(observations, request_ids, memories, strict=True)
            ]
            features, histories = self._encode_primitive_frames(prepared)
            embeddings, prefixes, positions = [], [], []
            for i, turn in enumerate(prepared):
                old = turn.committed
                history = p._selected_history_features(
                    histories[i], turn.inputs.slow_frame_indices, features[i]
                )
                embedding = p.backbone.prepare_multimodal_feature_embeddings(
                    turn.input_ids,
                    features[i],
                    history,
                    image_position=turn.inputs.image_position + turn.position_shift,
                    memory_position=None
                    if turn.inputs.memory_position is None
                    else turn.inputs.memory_position + turn.position_shift,
                )
                start = 0 if turn.base_cache is None else turn.base_cache.seq_len
                if start + embedding.shape[1] + p.max_new_tokens > self.capacity:
                    raise ValueError(
                        "batched turn and token budget exceed workspace; no history is truncated"
                    )
                embeddings.append(embedding)
                positions.append(start)
                retained = (
                    p._retain_history_feature(features[i])
                    if p.processor.retains_history_feature(old.step_id)
                    else None
                )
                working = StreamVLNMemory(
                    None,
                    (),
                    old.step_id + 1,
                    old.instruction,
                    histories[i] + (retained,),
                    turn.inputs.slow_frame_indices or old.slow_frame_indices,
                )
                prefixes.append(working)
            self._plan_rows(
                [n + e.shape[1] + p.max_new_tokens for n, e in zip(positions, embeddings, strict=True)]
            )
            previous, self.resident = self.resident, [None] * self.batch_size
            for i, turn in enumerate(prepared):
                if turn.base_cache is not None and previous[i] is not memories[i]:
                    cache = turn.base_cache
                    if cache.key_storage.device != self.device:
                        raise ValueError("batch memories must be on the policy device")
                    for target, source in (
                        (self._row(i).key_storage, cache.key_storage),
                        (self._row(i).value_storage, cache.value_storage),
                    ):
                        target[..., : cache.seq_len, :].copy_(source[..., : cache.seq_len, :])
                    self.counters["imported_kv_tokens"] += cache.seq_len
            hidden = self._prefill(embeddings, positions)
            positions = [n + e.shape[1] for n, e in zip(positions, embeddings, strict=True)]
            generations = self._decode(hidden, positions, check_cancel)
            check_cancel()
            results = []
            for i, generation in enumerate(generations):
                # Only the uniquely consumed latest memory may lend unused
                # capacity. Reordered/forked older memories get fresh storage.
                reusable = previous[i] is memories[i] and sum(m is memories[i] for m in memories) == 1
                cache = self._commit(generation.cache, prepared[i].base_cache if reusable else None)
                prefix = StreamVLNPrefix(prefixes[i], hidden[i : i + 1])
                results.append(p.decoder.finalize_generation(prefix, replace(generation, cache=cache)))
            for i, result in enumerate(results):
                self.resident[i] = result.next_memory
                self.drafter.observe(result.traces[0].token_ids.tolist())
            return tuple(results)
        except BaseException:
            self.resident = [None] * self.batch_size
            raise
        finally:
            self._lock.release()
