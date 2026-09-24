"""Policy-local, fixed-storage CUDA graphs for ActiveVLN inference.

Graph storage is a disposable execution workspace. Public recurrent memories
remain separately owned, so capture/replay never writes a committed prefix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .cache_activevln import ActiveVLNMemory
    from .modeling_activevln import ActiveVLNPolicy
    from .processor_activevln import ProcessedTurn
    from .speculation_activevln import ActionTokenTree


def plan_graph_shapes(
    initial_queries: Sequence[int],
    recurring_queries: Sequence[int],
    context_buckets: Sequence[int],
    *,
    query_bucket_size: int,
    max_context: int,
) -> list[tuple[int, int]]:
    """Plan startup-only shapes from input lengths, without assuming response lengths.

    Initial turns have no history. Recurring turns and single-token decoding
    cover the explicitly requested context buckets; longer contexts still use
    the unchanged eager fallback instead of truncating memory.
    """
    if type(query_bucket_size) is not int or query_bucket_size < 1:
        raise ValueError("query_bucket_size must be a positive Python integer")
    if type(max_context) is not int or max_context < 1:
        raise ValueError("max_context must be a positive Python integer")
    for key in context_buckets:
        if (
            type(key) is not int
            or key < min(512, max_context)
            or key > max_context
            or (key != max_context and key & (key - 1))
        ):
            raise ValueError("context buckets must match runtime powers of two or the context limit")
    buckets = sorted(set(context_buckets))
    shapes = {(1, key) for key in buckets}
    for queries, initial in ((initial_queries, True), (recurring_queries, False)):
        for query in queries:
            if type(query) is not int or query < 1:
                raise ValueError("query lengths must be positive Python integers")
            padded = (
                1
                if query == 1
                else ((query + query_bucket_size - 1) // query_bucket_size) * query_bucket_size
            )
            if padded > max_context:
                continue
            keys = [min(max_context, max(512, 1 << (padded - 1).bit_length()))] if initial else buckets
            shapes.update((padded, key) for key in keys if padded <= key)
    return sorted(shapes)


@dataclass
class _TextGraph:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    positions: torch.Tensor
    cache_position: torch.Tensor
    output: torch.Tensor


@dataclass
class _VisionGraph:
    graph: torch.cuda.CUDAGraph
    pixels: torch.Tensor
    output: torch.Tensor
    constants: tuple[torch.Tensor, ...]


class ActiveVLNGraphRuntime:
    """Capture only during explicit startup, then replay or report eager fallback.

    Both the rotary position and KV insertion position are runtime inputs. The
    latter must not be substituted for the former in this multimodal model.
    The caller holds ``lock`` until returned KV views have been copied into its
    private working memory and ``bind_memory`` has been called.
    """

    def __init__(
        self,
        policy: ActiveVLNPolicy,
        *,
        query_bucket_size: int = 32,
        fused_ops: bool = False,
        split_attention: bool = False,
        tree_decode: bool = False,
        tree_fp32_projection: bool = False,
        tree_repeat_actions: int = 1,
        workspace_tokens: int | None = None,
    ) -> None:
        if query_bucket_size < 1:
            raise ValueError("query_bucket_size must be positive")
        if type(tree_repeat_actions) is not int or not 1 <= tree_repeat_actions <= 3:
            raise ValueError("tree_repeat_actions must be a Python integer from 1 to 3")
        capacity = policy.max_context if workspace_tokens is None else workspace_tokens
        if type(capacity) is not int or not 1 <= capacity <= policy.max_context:
            raise ValueError(
                "workspace_tokens must be a positive Python integer within the model context limit"
            )
        parameter = next(policy.parameters())
        if parameter.device.type != "cuda" or parameter.dtype != torch.bfloat16:
            raise ValueError("ActiveVLN graphs require a CUDA BF16 policy")
        if policy.training or torch.is_grad_enabled():
            raise ValueError("ActiveVLN graphs require eval() and inference_mode()/no_grad()")
        if tree_decode and policy.do_sample:
            raise ValueError("ActiveVLN tree_decode requires greedy decoding")
        self.policy = policy
        self.lock = RLock()
        self.query_bucket_size = query_bucket_size
        self.fused_ops = fused_ops
        self.split_attention = split_attention
        self.tree_fp32_projection = tree_fp32_projection
        if tree_fp32_projection and not tree_decode:
            raise ValueError("tree_fp32_projection requires tree_decode")
        self.tree_repeat_actions = tree_repeat_actions
        if fused_ops and getattr(policy._text_config, "hidden_act", "silu") not in ("silu", "swish"):
            raise ValueError("ActiveVLN fused MLP requires SwiGLU")
        self.device = parameter.device
        self.stream = torch.cuda.current_stream(self.device)
        self.weight_pointer = parameter.data_ptr()
        self.capture_enabled = False
        self.pool = torch.cuda.graph_pool_handle()
        config = policy._text_config
        layers = len(policy._text.layers)
        heads = int(config.num_key_value_heads)
        width = int(policy._text.layers[0].self_attn.head_dim)
        self.storage = torch.zeros(
            (layers, 2, 1, heads, capacity, width),
            dtype=parameter.dtype,
            device=self.device,
        )
        self.text_graphs: dict[tuple[int, int], _TextGraph] = {}
        self.tree_graphs: dict[tuple[str, int], _TextGraph] = {}
        self.action_trees: tuple[ActionTokenTree, ...] = ()
        if tree_decode:
            from .speculation_activevln import build_action_trees

            self.action_trees = build_action_trees(
                policy.tokenizer,
                policy.eos_token_ids,
                self.device,
                repeat_actions=tree_repeat_actions,
                partition_roots=True,
            )
        self.vision_graphs: dict[tuple, _VisionGraph] = {}
        self._resident_memory: ActiveVLNMemory | None = None
        self._resident_length = 0
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset measured replay/fallback counters without discarding captures."""
        self.counters = dict.fromkeys(
            (
                "vision_replays",
                "vision_fallbacks",
                "prefill_replays",
                "prefill_fallbacks",
                "decode_replays",
                "decode_fallbacks",
                "memory_restores",
                "tree_replays",
                "tree_fallbacks",
                "tree_accepted_tokens",
                "tree_fallback_tokens",
            ),
            0,
        )

    def stats(self) -> dict[str, object]:
        """Expose actual capture coverage and runtime execution counts."""
        return {
            **self.counters,
            "vision_graphs": len(self.vision_graphs),
            "text_graphs": len(self.text_graphs),
            "text_shapes": [list(key) for key in sorted(self.text_graphs)],
            "tree_decode": bool(self.action_trees),
            "tree_repeat_actions": self.tree_repeat_actions,
            "tree_fp32_projection": self.tree_fp32_projection,
            "tree_nodes": {tree.name: tree.token_ids.shape[1] for tree in self.action_trees},
            "tree_graphs": len(self.tree_graphs),
            "tree_shapes": [list(key) for key in sorted(self.tree_graphs)],
            "capture_enabled": self.capture_enabled,
            "query_bucket_size": self.query_bucket_size,
            "fused_ops": self.fused_ops,
            "split_attention": self.split_attention,
            "workspace_mib": self.storage.numel() * self.storage.element_size() / 2**20,
            "workspace_tokens": self.storage.shape[-2],
            "model_max_context": self.policy.max_context,
        }

    def _validate(self) -> None:
        if self.policy.training or torch.is_grad_enabled():
            raise RuntimeError("captured ActiveVLN execution is inference-only")
        parameter = next(self.policy.parameters())
        if parameter.device != self.device or parameter.data_ptr() != self.weight_pointer:
            raise RuntimeError("ActiveVLN weights moved after graph initialization; rebuild the runtime")
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("ActiveVLN graphs must execute on their startup CUDA stream")

    def bind_memory(self, memory: ActiveVLNMemory) -> None:
        """Associate workspace contents with a successfully appended working state."""
        self._resident_memory = memory
        self._resident_length = memory.length

    def _restore_memory(self, memory: ActiveVLNMemory | None) -> None:
        if memory is None:
            self._resident_memory = None
            self._resident_length = 0
            return
        if memory is self._resident_memory and memory.length == self._resident_length:
            return
        # A failed restore must never leave the previous resident marker valid.
        self._resident_memory = None
        if memory.packed_kv is not None:
            self.storage[..., : memory.length, :].copy_(memory.packed_kv)
        else:
            for index, (key, value) in enumerate(memory.visible_kv()):
                self.storage[index, 0, :, :, : memory.length].copy_(key)
                self.storage[index, 1, :, :, : memory.length].copy_(value)
        self.counters["memory_restores"] += 1
        self._resident_memory = memory
        self._resident_length = memory.length

    @staticmethod
    def _capture(callback, device: torch.device, pool) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        current = torch.cuda.current_stream(device)
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                callback()
        current.wait_stream(warmup)
        current.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
            output = callback()
        return graph, output

    def vision(self, turn: ProcessedTurn) -> torch.Tensor | None:
        """Replay the pinned dense-mask vision forward with cached shape constants."""
        self._validate()
        grid = turn.image_grid_thw.detach().cpu()
        key = (tuple(grid.reshape(-1).tolist()), tuple(turn.pixel_values.shape))
        entry = self.vision_graphs.get(key)
        if entry is None:
            if not self.capture_enabled:
                self.counters["vision_fallbacks"] += 1
                return None
            visual = self.policy._visual
            rotary = visual.rot_pos_emb(grid)
            window_index, cu_window = visual.get_window_index(grid)
            window_index = window_index.to(self.device)
            reverse = torch.argsort(window_index)
            unit = visual.spatial_merge_unit
            size = turn.pixel_values.shape[0]
            rotary = rotary.reshape(size // unit, unit, -1)[window_index].reshape(size, -1)
            rotary = torch.cat((rotary, rotary), dim=-1)
            cos, sin = rotary.cos(), rotary.sin()
            # CPU scalar slice boundaries avoid device synchronization in the
            # pinned 4.51.3 block's construction of its dense attention mask.
            cu_window = torch.unique_consecutive(torch.tensor(cu_window, dtype=torch.int32))
            cu_full = (grid[:, 1] * grid[:, 2]).repeat_interleave(grid[:, 0]).cumsum(0, dtype=torch.int32)
            cu_full = torch.nn.functional.pad(cu_full, (1, 0))
            pixels = turn.pixel_values.clone()

            def forward() -> torch.Tensor:
                hidden = visual.patch_embed(pixels)
                hidden = hidden.reshape(size // unit, unit, -1)[window_index].reshape(size, -1)
                for index, block in enumerate(visual.blocks):
                    bounds = cu_full if index in visual.fullatt_block_indexes else cu_window
                    hidden = block(hidden, cu_seqlens=bounds, position_embeddings=(cos, sin))
                return visual.merger(hidden)[reverse]

            graph, output = self._capture(forward, self.device, self.pool)
            # CUDA Graph records addresses, not Python ownership of inputs
            # allocated outside capture. Keep all layout/rotary buffers alive.
            entry = _VisionGraph(graph, pixels, output, (window_index, reverse, cos, sin, cu_window, cu_full))
            self.vision_graphs[key] = entry
        entry.pixels.copy_(turn.pixel_values)
        entry.graph.replay()
        self.counters["vision_replays"] += 1
        # Text graphs share a graph pool, so consume an independent output.
        return entry.output.clone()

    def _text_forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_position: torch.Tensor,
        key_bucket: int,
        ancestors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .modeling_activevln import _mlp, _rmsnorm, apply_mrope, mrope_cos_sin

        policy = self.policy
        text = policy._text
        head_dim = text.layers[0].self_attn.head_dim
        sections = tuple(policy._text_config.rope_scaling["mrope_section"])
        cos, sin = mrope_cos_sin(positions, head_dim, float(policy._text_config.rope_theta), hidden.dtype)
        norm = _rmsnorm
        mlp = _mlp
        rotary = partial(apply_mrope, cos=cos, sin=sin, sections=sections)
        if self.fused_ops:
            from ...backend.triton.rounded_ops import rounded_rms_norm, rounded_rope, rounded_swiglu

            cos_table = torch.cat(
                [part[i % 3] for i, part in enumerate(cos.split(list(sections) * 2, dim=-1))], dim=-1
            ).reshape(-1, head_dim)
            sin_table = torch.cat(
                [part[i % 3] for i, part in enumerate(sin.split(list(sections) * 2, dim=-1))], dim=-1
            ).reshape(-1, head_dim)

            def norm(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
                return rounded_rms_norm(x, module.weight, module.variance_epsilon)

            def mlp(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
                return module.down_proj(rounded_swiglu(module.gate_proj(x), module.up_proj(x)))

            rotary = partial(rounded_rope, cosine=cos_table, sine=sin_table)

        def project(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            return module(x)

        if ancestors is not None and self.tree_fp32_projection:
            from ...backend.torch.linear import linear_fp32_output

            def project(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
                return linear_fp32_output(x, module.weight, module.bias)

            def mlp(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
                gate, up = project(module.gate_proj, x), project(module.up_proj, x)
                activated = rounded_swiglu(gate, up) if self.fused_ops else module.act_fn(gate) * up
                return project(module.down_proj, activated)

        query_length = hidden.shape[1]
        indices = cache_position + torch.arange(query_length, device=self.device)
        mask = None
        if not self.split_attention:
            key_indices = torch.arange(key_bucket, device=self.device)
            if ancestors is None:
                allowed = key_indices[None, :] <= indices[:, None]
            else:
                local_keys = key_indices - cache_position
                in_tree = (local_keys >= 0) & (local_keys < query_length)
                allowed = (key_indices[None, :] < cache_position) | (
                    ancestors[:, local_keys.clamp(0, query_length - 1)] & in_tree[None, :]
                )
            mask = torch.zeros((query_length, key_bucket), device=self.device, dtype=hidden.dtype)
            mask = mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[None, None]
        for index, layer in enumerate(text.layers):
            residual = hidden
            h = norm(layer.input_layernorm, hidden)
            attention = layer.self_attn
            q = project(attention.q_proj, h).view(1, query_length, -1, head_dim).transpose(1, 2)
            k = project(attention.k_proj, h).view(1, query_length, -1, head_dim).transpose(1, 2)
            v = project(attention.v_proj, h).view(1, query_length, -1, head_dim).transpose(1, 2)
            q, k = rotary(q, k)
            key_cache, value_cache = self.storage[index].unbind(0)
            key_cache.index_copy_(2, indices, k)
            value_cache.index_copy_(2, indices, v)
            if self.split_attention:
                from ...backend.triton.split_attention import split_kv_attention

                out = split_kv_attention(
                    q, key_cache, value_cache, cache_position, key_bucket, ancestors=ancestors
                )
            else:
                out = policy._attn.attend(
                    q,
                    key_cache[:, :, :key_bucket],
                    value_cache[:, :, :key_bucket],
                    attn_mask=mask,
                    scaling=head_dim**-0.5,
                )
            hidden = residual + project(attention.o_proj, out.transpose(1, 2).reshape(1, query_length, -1))
            hidden = hidden + mlp(layer.mlp, norm(layer.post_attention_layernorm, hidden))
        return norm(text.norm, hidden)

    def _capture_text(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        length: int,
        bucket: int,
        key_bucket: int,
        tree: ActionTokenTree | None = None,
    ) -> _TextGraph:
        query = hidden.shape[1]
        static_hidden = hidden.new_zeros((1, bucket, hidden.shape[-1]))
        static_positions = positions.new_zeros((3, 1, bucket))
        static_hidden[:, :query].copy_(hidden)
        static_positions[:, :, :query].copy_(positions)
        cache_position = torch.tensor([length], device=self.device, dtype=torch.long)
        graph, output = self._capture(
            lambda: self._text_forward(
                static_hidden,
                static_positions,
                cache_position,
                key_bucket,
                None if tree is None else tree.ancestors,
            ),
            self.device,
            self.pool,
        )
        return _TextGraph(graph, static_hidden, static_positions, cache_position, output)

    def prewarm_vision(self, turn: ProcessedTurn) -> None:
        """Capture a processed CPU observation's vision shape during explicit startup."""
        self._validate()
        if not self.capture_enabled:
            raise RuntimeError("ActiveVLN shape prewarming is only allowed during startup capture")
        grid = turn.image_grid_thw.detach().cpu()
        key = (tuple(grid.reshape(-1).tolist()), tuple(turn.pixel_values.shape))
        if key not in self.vision_graphs:
            self.vision(turn.to(self.device, next(self.policy.parameters()).dtype))

    def prewarm_text(
        self,
        initial_queries: Sequence[int],
        recurring_queries: Sequence[int],
        context_buckets: Sequence[int],
    ) -> list[tuple[int, int]]:
        """Capture input-shape coverage before timing, using disposable synthetic KV.

        This does not generate actions or change any session's public memory.
        The workspace resident marker is invalidated even if capture fails.
        """
        self._validate()
        if not self.capture_enabled:
            raise RuntimeError("ActiveVLN shape prewarming is only allowed during startup capture")
        shapes = plan_graph_shapes(
            initial_queries,
            recurring_queries,
            context_buckets,
            query_bucket_size=self.query_bucket_size,
            max_context=self.storage.shape[-2],
        )
        parameter = next(self.policy.parameters())
        width = self.policy._text.embed_tokens.weight.shape[1]
        with self.lock:
            self._resident_memory, self._resident_length = None, 0
            self.storage.zero_()
            try:
                for query, key in shapes:
                    if (query, key) in self.text_graphs:
                        continue
                    hidden = parameter.new_zeros((1, query, width))
                    positions = torch.arange(query, device=self.device)[None, None].expand(3, 1, -1)
                    entry = self._capture_text(hidden, positions, key - query, query, key)
                    self.text_graphs[(query, key)] = entry
                    self.policy._lm_head(entry.output[:, -1])
                for tree in self.action_trees:
                    query = tree.token_ids.shape[1]
                    for key in sorted(set(context_buckets)):
                        if query > key or (tree.name, key) in self.tree_graphs:
                            continue
                        hidden = self.policy._text.embed_tokens(tree.token_ids)
                        positions = tree.depths[None, None].expand(3, 1, -1)
                        entry = self._capture_text(hidden, positions, key - query, query, key, tree)
                        self.tree_graphs[(tree.name, key)] = entry
                        self._tree_logits(entry.output)
            finally:
                self._resident_memory, self._resident_length = None, 0
        return shapes

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        memory: ActiveVLNMemory | None,
        *,
        tree: ActionTokenTree | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Run vision-independent text prefill/decode against private workspace KV."""
        self._validate()
        length = 0 if memory is None else memory.length
        query = hidden.shape[1]
        stage = "tree" if tree is not None else "decode" if query == 1 else "prefill"
        bucket = (
            query
            if tree is not None or query == 1
            else ((query + self.query_bucket_size - 1) // self.query_bucket_size) * self.query_bucket_size
        )
        # Execution scratch capacity is independent of the public memory limit.
        # Longer histories retain normal eager execution without truncation.
        capacity = self.storage.shape[-2]
        if length + bucket > capacity:
            self.counters[f"{stage}_fallbacks"] += 1
            self._resident_memory = None
            return None
        key_bucket = min(capacity, max(512, 1 << (length + bucket - 1).bit_length()))
        key = (bucket, key_bucket) if tree is None else (tree.name, key_bucket)
        graphs = self.text_graphs if tree is None else self.tree_graphs
        entry = graphs.get(key)
        if entry is None and not self.capture_enabled:
            self.counters[f"{stage}_fallbacks"] += 1
            self._resident_memory = None
            return None
        self._restore_memory(memory)
        if entry is None:
            entry = self._capture_text(hidden, positions, length, bucket, key_bucket, tree)
            graphs[key] = entry
        entry.hidden.zero_()
        entry.hidden[:, :query].copy_(hidden)
        entry.positions.zero_()
        entry.positions[:, :, :query].copy_(positions)
        entry.cache_position.fill_(length)
        entry.graph.replay()
        self.counters[f"{stage}_replays"] += 1
        self._resident_length = length + query
        kv = self.storage[..., length : length + query, :]
        return entry.output[:, :query].clone(), kv

    def verify_tree(
        self, tree: ActionTokenTree, memory: ActiveVLNMemory
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Compute all candidate paths without appending any node to public memory.

        The caller holds the runtime lock through selection and compaction of
        the accepted path. Sibling nodes share the prefix, never each other's KV.
        """
        positions = (tree.depths + memory.next_position)[None, None].expand(3, 1, -1)
        hidden = self.policy._text.embed_tokens(tree.token_ids)
        result = self.forward(hidden, positions, memory, tree=tree)
        if result is None:
            return None
        output, packed_kv = result
        return self._tree_logits(output), packed_kv, positions

    def _tree_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tree_fp32_projection:
            from ...backend.torch.linear import linear_fp32_output

            head = self.policy._lm_head
            return linear_fp32_output(hidden, head.weight, head.bias)
        return self.policy._lm_head(hidden)
