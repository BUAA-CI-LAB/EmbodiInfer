"""Private text decode graphs with unchanged, unpadded SDPA attention."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from ...exceptions import SessionCancelledError
from .modeling_activevln import _mlp, _mrope_config, _rmsnorm, apply_mrope

if TYPE_CHECKING:
    from .cache_activevln import ActiveVLNMemory
    from .modeling_activevln import ActiveVLNPolicy


class _Captured:
    def __init__(self, forward: Callable[[], tuple[torch.Tensor, ...]]) -> None:
        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = forward()

    def replay(self) -> tuple[torch.Tensor, ...]:
        self.graph.replay()
        return self.outputs


class ActiveVLNTextGraph:
    """Capture constant-shape decode work around dynamic-length attention.

    This borrows the policy's parameter tensors and owns only graph workspaces.
    It never reads the navigation graph cache. The caller must serialize use on
    the capture stream, and recreate the object after refit or device changes.
    Only BF16 CUDA, batch one and the SDPA policy backend are supported.
    """

    def __init__(self, policy: ActiveVLNPolicy) -> None:
        parameter = next(policy.parameters())
        if parameter.device.type != "cuda" or parameter.dtype != torch.bfloat16:
            raise ValueError("text decode graphs require a BF16 CUDA policy")
        if policy._attn.name != "sdpa":
            raise ValueError("text decode graphs require the SDPA policy backend")
        self.policy = policy
        self._version = policy.policy_version
        self._text = policy._text
        self._device = parameter.device
        self._stream = torch.cuda.current_stream(self._device)
        attention = self._text.layers[0].self_attn
        self._head_dim = attention.head_dim
        self._sections, _ = _mrope_config(policy._text_config)
        self._hidden = torch.zeros(
            (1, 1, policy._text_config.hidden_size), device=self._device, dtype=parameter.dtype
        )
        self._positions = torch.zeros((3, 1, 1), device=self._device, dtype=torch.long)
        self._attended = torch.zeros(
            (1, attention.q_proj.out_features // self._head_dim, 1, self._head_dim),
            device=self._device,
            dtype=parameter.dtype,
        )
        self._graphs: list[_Captured] = []
        with torch.cuda.device(self._device), torch.inference_mode():
            self._initial = _Captured(self._initial_forward)
            hidden, _, _, _, cos, sin = self._initial.outputs
            for index in range(len(self._text.layers)):
                captured = _Captured(self._after_attention(index, hidden, cos, sin))
                self._graphs.append(captured)
                hidden = captured.outputs[0]
            torch.cuda.synchronize(self._device)

    def _project(
        self, layer: torch.nn.Module, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = _rmsnorm(layer.input_layernorm, hidden)
        attention = layer.self_attn
        q = attention.q_proj(h).view(1, 1, -1, self._head_dim).transpose(1, 2)
        k = attention.k_proj(h).view(1, 1, -1, self._head_dim).transpose(1, 2)
        v = attention.v_proj(h).view(1, 1, -1, self._head_dim).transpose(1, 2)
        q, k = apply_mrope(q, k, cos, sin, self._sections)
        return q, k, v

    def _initial_forward(self) -> tuple[torch.Tensor, ...]:
        cos, sin = self._text.rotary_emb(self._hidden, self._positions)
        q, k, v = self._project(self._text.layers[0], self._hidden, cos, sin)
        return self._hidden, q, k, v, cos, sin

    def _after_attention(
        self, index: int, residual: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Callable[[], tuple[torch.Tensor, ...]]:
        layer = self._text.layers[index]

        def forward() -> tuple[torch.Tensor, ...]:
            hidden = residual + layer.self_attn.o_proj(self._attended.transpose(1, 2).reshape(1, 1, -1))
            hidden = hidden + _mlp(layer.mlp, _rmsnorm(layer.post_attention_layernorm, hidden))
            if index + 1 == len(self._text.layers):
                return (_rmsnorm(self._text.norm, hidden),)
            return (hidden, *self._project(self._text.layers[index + 1], hidden, cos, sin))

        return forward

    def append_token(
        self, memory: ActiveVLNMemory, token: torch.Tensor
    ) -> tuple[ActiveVLNMemory, torch.Tensor]:
        """Append one text token and copy temporary graph KV into private memory.

        The caller holds the execution gate and inference mode. Attention sees
        the exact prefix length, matching the eager text reference's kernel
        dispatch; it receives neither padded keys nor a synthetic mask.
        """
        if self.policy.policy_version != self._version:
            raise SessionCancelledError("text decode graphs belong to a stale policy version")
        if torch.cuda.current_stream(self._device) != self._stream:
            raise ValueError("text decode graphs must use their capture stream")
        if token.shape != (1, 1) or memory.token_ids_buffer.device != self._device:
            raise ValueError("text decode graphs require one token and memory on the capture device")
        token = token.to(self._device)
        position = torch.full((3, 1, 1), memory.next_position, device=self._device, dtype=torch.long)
        self._hidden.copy_(self._text.embed_tokens(token))
        self._positions.copy_(position)
        _, q, k, v, _, _ = self._initial.replay()
        prefix_kv = memory.visible_kv()
        collected = []
        for index, graph in enumerate(self._graphs):
            collected.append((k, v))
            pk, pv = prefix_kv[index]
            full_k, full_v = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
            out = self.policy._reference_text_attn.attend(
                q, full_k, full_v, attn_mask=None, scaling=self._head_dim**-0.5
            )
            self._attended.copy_(out)
            result = graph.replay()
            if len(result) == 4:
                hidden, q, k, v = result
            else:
                (hidden,) = result
        memory.append_chunk(
            collected, token, torch.ones_like(token), position, next_position=memory.next_position + 1
        )
        return memory, self.policy._lm_head(hidden[:, -1])
