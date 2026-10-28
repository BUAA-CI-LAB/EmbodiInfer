"""Measured INT8-aware text projection dispatch for the batch-one experiment."""

from __future__ import annotations

from functools import partial

import torch

from embodiinfer.models.linear import QuantizedLinear


def _text_forward(
    self,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    cache_position: torch.Tensor,
    key_bucket: int,
    ancestors: torch.Tensor | None = None,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``packed`` = (blocks, starts, extent, slots) marks ``hidden`` as one row of
    back-to-back turns for pooled GQA attention: the first three are the
    kernel's packed query layout and ``slots`` each token's pooled KV index."""
    from embodiinfer.policies.activevln.modeling_activevln import (
        _mlp,
        _rmsnorm,
        apply_mrope,
        mrope_cos_sin,
    )

    policy = self.policy
    text = policy._text
    head_dim = text.layers[0].self_attn.head_dim
    sections = tuple(policy._text_config.rope_scaling["mrope_section"])
    cos, sin = mrope_cos_sin(positions, head_dim, float(policy._text_config.rope_theta), hidden.dtype)
    norm = _rmsnorm
    mlp = _mlp
    rotary = partial(apply_mrope, cos=cos, sin=sin, sections=sections)
    if self.fused_ops:
        from embodiinfer.backend.triton.rounded_ops import rounded_rms_norm, rounded_rope, rounded_swiglu

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

    project_qkv = None
    if ancestors is not None and self.tree_fp32_projection:
        from embodiinfer.backend.torch.linear import linear_fp32_output

        def project(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            return linear_fp32_output(x, module.weight, module.bias)

        def mlp(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            gate, up = project(module.gate_proj, x), project(module.up_proj, x)
            activated = rounded_swiglu(gate, up) if self.fused_ops else module.act_fn(gate) * up
            return project(module.down_proj, activated)

    elif getattr(self, "decode_linear", False):
        from embodiinfer.backend.triton.decode_linear import decode_linear, decode_qkv, decode_swiglu

        def project(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            if isinstance(module, QuantizedLinear):
                return module(x)
            return decode_linear(x, module.weight, module.bias)

        def project_qkv(attention: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
            projections = (attention.q_proj, attention.k_proj, attention.v_proj)
            if any(isinstance(projection, QuantizedLinear) for projection in projections):
                return [project(projection, x) for projection in projections]
            weights = [projection.weight for projection in projections]
            return decode_qkv(x, weights, [projection.bias for projection in projections])

        def mlp(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            if self.fused_ops and not any(
                isinstance(p, QuantizedLinear) for p in (module.gate_proj, module.up_proj)
            ):
                activated = decode_swiglu(x, module.gate_proj.weight, module.up_proj.weight)
            else:
                gate, up = project(module.gate_proj, x), project(module.up_proj, x)
                activated = rounded_swiglu(gate, up) if self.fused_ops else module.act_fn(gate) * up
            return project(module.down_proj, activated)

    batch, query_length = hidden.shape[:2]
    layout = getattr(self, "_cache_layout", None)
    write_lengths = getattr(self, "_write_lengths", None)
    indices = cache_position[:, None] + torch.arange(query_length, device=self.device)[None, :]
    mask = None
    if not self.split_attention:
        key_indices = torch.arange(key_bucket, device=self.device)
        if ancestors is None:
            allowed = key_indices[None, None, :] <= indices[:, :, None]
        else:
            local_keys = key_indices[None, :] - cache_position[:, None]
            in_tree = (local_keys >= 0) & (local_keys < query_length)
            masks = ancestors.expand(batch, -1, -1) if ancestors.ndim == 2 else ancestors
            edges = masks.gather(
                2, local_keys.clamp(0, query_length - 1)[:, None].expand(-1, query_length, -1)
            )
            allowed = (key_indices[None, None, :] < cache_position[:, None, None]) | (
                edges & in_tree[:, None, :]
            )
        mask = torch.zeros((batch, query_length, key_bucket), device=self.device, dtype=hidden.dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[:, None]
    for index, layer in enumerate(text.layers):
        residual = hidden
        h = norm(layer.input_layernorm, hidden)
        attention = layer.self_attn
        if project_qkv is None:
            q = project(attention.q_proj, h).view(batch, query_length, -1, head_dim).transpose(1, 2)
            k = project(attention.k_proj, h).view(batch, query_length, -1, head_dim).transpose(1, 2)
            v = project(attention.v_proj, h).view(batch, query_length, -1, head_dim).transpose(1, 2)
        else:
            q, k, v = (
                part.view(batch, query_length, -1, head_dim).transpose(1, 2)
                for part in project_qkv(attention, h)
            )
        q, k = rotary(q, k)
        key_cache, value_cache = self.storage[index].unbind(0)
        if packed is not None:
            key_cache.index_copy_(2, packed[3], k)
            value_cache.index_copy_(2, packed[3], v)
        elif batch == 1 and not getattr(self, "ragged_batch", False):
            key_cache.index_copy_(2, indices[0], k)
            value_cache.index_copy_(2, indices[0], v)
        elif hidden.is_cuda:
            from embodiinfer.backend.triton.split_attention import write_batched_kv

            write_batched_kv(
                k,
                v,
                key_cache,
                value_cache,
                cache_position,
                cache_starts=None if layout is None else layout[0],
                cache_capacities=None if layout is None else layout[1],
                write_lengths=write_lengths,
            )
        else:
            for row, offset in enumerate(cache_position.tolist()):
                base = 0 if layout is None else int(layout[0][row])
                capacity = key_cache.shape[-2] if layout is None else int(layout[1][row])
                count = min(
                    query_length if write_lengths is None else int(write_lengths[row]), capacity - offset
                )
                slot = row if layout is None else 0
                key_cache[slot, :, base + offset : base + offset + count].copy_(k[row, :, :count])
                value_cache[slot, :, base + offset : base + offset + count].copy_(v[row, :, :count])
        if self.split_attention:
            extra = {}
            if getattr(self, "attention_kernel", "split") == "gqa":
                from embodiinfer.backend.triton.gqa_split_attention import (
                    gqa_split_kv_attention as split_kv_attention,
                )

                if getattr(self, "skip_padded_queries", False) or packed is not None:
                    extra["query_lengths"] = write_lengths
                if packed is not None:
                    extra["packed"] = packed[:3]
            else:
                from embodiinfer.backend.triton.split_attention import split_kv_attention

            out = split_kv_attention(
                q,
                key_cache,
                value_cache,
                cache_position,
                key_bucket,
                ancestors=ancestors,
                cache_starts=None if layout is None else layout[0],
                cache_capacities=None if layout is None else layout[1],
                **extra,
            )
        else:
            if layout is not None:
                # CPU correctness reference; accelerated pooled execution uses split-KV.
                starts, capacities = (x.tolist() for x in layout)

                def gather_rows(cache, starts=starts, capacities=capacities):
                    return torch.stack(
                        [
                            torch.nn.functional.pad(
                                cache[0, :, start : start + min(capacity, key_bucket)],
                                (0, 0, 0, max(0, key_bucket - capacity)),
                            )
                            for start, capacity in zip(starts, capacities, strict=True)
                        ]
                    )

                key_cache, value_cache = gather_rows(key_cache), gather_rows(value_cache)
            out = policy._attn.attend(
                q,
                key_cache[:, :, :key_bucket],
                value_cache[:, :, :key_bucket],
                attn_mask=mask,
                scaling=head_dim**-0.5,
            )
        hidden = residual + project(attention.o_proj, out.transpose(1, 2).reshape(batch, query_length, -1))
        hidden = hidden + mlp(layer.mlp, norm(layer.post_attention_layernorm, hidden))
    return norm(text.norm, hidden)
