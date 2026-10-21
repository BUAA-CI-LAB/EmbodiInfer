"""Correctness-first growable KV memory for a single ActiveVLN session."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

_KVChunk = list[tuple[torch.Tensor, torch.Tensor]] | torch.Tensor


def _pairs(kv: _KVChunk) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(kv, torch.Tensor):
        if kv.ndim != 6 or kv.shape[1] != 2:
            raise ValueError("packed ActiveVLN KV must have shape [layers,2,batch,heads,Q,width]")
        return [(layer[0], layer[1]) for layer in kv.unbind(0)]
    return kv


def _layer_views(storage: torch.Tensor) -> list[_LayerBuffer]:
    return [_LayerBuffer(key, value) for key, value in _pairs(storage)]


@dataclass
class _LayerBuffer:
    key: torch.Tensor  # [1, n_kv, capacity, head_dim]
    value: torch.Tensor

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    def to(self, device: torch.device | str) -> _LayerBuffer:
        return _LayerBuffer(self.key.to(device), self.value.to(device))


@dataclass
class ActiveVLNMemory:
    layers: list[_LayerBuffer]
    token_ids_buffer: torch.Tensor  # [capacity]
    attention_buffer: torch.Tensor  # [capacity]
    position_buffer: torch.Tensor  # [3, capacity]
    length: int
    max_length: int = 32768
    prompt_hashes: tuple[str, ...] = field(default_factory=tuple)
    _next_position: int | None = None
    _packed_kv: torch.Tensor | None = field(default=None, repr=False)

    @property
    def seq_len(self) -> int:
        return self.length

    @property
    def capacity(self) -> int:
        return self.token_ids_buffer.shape[0]

    @property
    def next_position(self) -> int:
        if self.length == 0:
            return 0
        if self._next_position is not None:
            return self._next_position
        return int(self.position_buffer[:, : self.length].max().item()) + 1

    @property
    def token_ids(self) -> torch.Tensor:
        return self.token_ids_buffer[: self.length][None]

    @property
    def attention_mask(self) -> torch.Tensor:
        return self.attention_buffer[: self.length][None]

    @property
    def position_ids(self) -> torch.Tensor:
        return self.position_buffer[:, : self.length, None].transpose(1, 2)

    def visible_kv(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(layer.key[:, :, : self.length], layer.value[:, :, : self.length]) for layer in self.layers]

    @property
    def packed_kv(self) -> torch.Tensor | None:
        """Visible inference KV, when layers share one allocation; never includes spare slots."""
        return None if self._packed_kv is None else self._packed_kv[..., : self.length, :]

    def to(self, device: torch.device | str) -> ActiveVLNMemory:
        packed = None if self._packed_kv is None else self._packed_kv.to(device)
        return ActiveVLNMemory(
            layers=[layer.to(device) for layer in self.layers] if packed is None else _layer_views(packed),
            token_ids_buffer=self.token_ids_buffer.to(device),
            attention_buffer=self.attention_buffer.to(device),
            position_buffer=self.position_buffer.to(device),
            length=self.length,
            max_length=self.max_length,
            prompt_hashes=self.prompt_hashes,
            _next_position=self._next_position,
            _packed_kv=packed,
        )

    def fork(self, *, extra_capacity: int = 0) -> ActiveVLNMemory:
        """Copy the visible prefix into private buffers, reserving room for a known turn.

        The optional spare capacity is bounded by the context limit and changes
        allocation only; visible length and committed state remain unchanged.
        """
        if type(extra_capacity) is not int or extra_capacity < 0:
            raise ValueError("extra_capacity must be a nonnegative Python integer")
        capacity = min(self.max_length, max(16, self.length + extra_capacity))
        layers = []
        packed = None
        if self._packed_kv is not None and not torch.is_grad_enabled():
            shape = (*self._packed_kv.shape[:-2], capacity, self._packed_kv.shape[-1])
            packed = self._packed_kv.new_empty(shape)
            packed[..., : self.length, :].copy_(self.packed_kv)
            layers = _layer_views(packed)
        else:
            for key, value in self.visible_kv():
                k = torch.empty(
                    key.shape[0], key.shape[1], capacity, key.shape[3], device=key.device, dtype=key.dtype
                )
                v = torch.empty_like(k)
                if self.length:
                    k[:, :, : self.length].copy_(key)
                    v[:, :, : self.length].copy_(value)
                layers.append(_LayerBuffer(k, v))
        tokens = torch.empty(capacity, device=self.token_ids_buffer.device, dtype=self.token_ids_buffer.dtype)
        attention = torch.empty(
            capacity, device=self.attention_buffer.device, dtype=self.attention_buffer.dtype
        )
        positions = torch.empty(
            3, capacity, device=self.position_buffer.device, dtype=self.position_buffer.dtype
        )
        if self.length:
            tokens[: self.length].copy_(self.token_ids_buffer[: self.length])
            attention[: self.length].copy_(self.attention_buffer[: self.length])
            positions[:, : self.length].copy_(self.position_buffer[:, : self.length])
        return ActiveVLNMemory(
            layers,
            tokens,
            attention,
            positions,
            self.length,
            self.max_length,
            self.prompt_hashes,
            self._next_position,
            packed,
        )

    def expand(self, num_samples: int) -> BranchedActiveVLNMemory | ActiveVLNMemory:
        """L1 prefix sharing: clone the already-computed KV into isolated branches.

        This saves backbone/prefill compute, not KV storage.  Each branch owns
        its buffers so token appends are transactionally independent.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if num_samples == 1:
            return self.fork()
        return BranchedActiveVLNMemory(tuple(self.fork() for _ in range(num_samples)))

    def compact(self, keep: Sequence[int] | torch.Tensor | None = None) -> ActiveVLNMemory:
        """Append-only ActiveVLN has no token compaction in the first version."""
        if keep is not None:
            indices = keep.tolist() if isinstance(keep, torch.Tensor) else list(keep)
            if indices not in ([], [0]):
                raise IndexError("a single ActiveVLN memory only has row 0")
        return self

    @classmethod
    def from_chunk(
        cls,
        kv: _KVChunk,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        max_length: int = 32768,
        prompt_hash: str | None = None,
        next_position: int | None = None,
    ) -> ActiveVLNMemory:
        memory = cls._allocate(kv, token_ids, attention_mask, position_ids, max_length=max_length)
        memory.append_chunk(
            kv, token_ids, attention_mask, position_ids, prompt_hash=prompt_hash, next_position=next_position
        )
        return memory

    @classmethod
    def _allocate(
        cls,
        kv,
        token_ids,
        attention_mask,
        position_ids,
        *,
        max_length,
    ) -> ActiveVLNMemory:
        q_len = token_ids.shape[-1]
        capacity = min(max_length, max(16, 1 << max(0, q_len - 1).bit_length()))
        layers = []
        packed = None
        pairs = _pairs(kv)
        if pairs and not torch.is_grad_enabled():
            reference = pairs[0][0]
            if all(
                x.shape == reference.shape and x.dtype == reference.dtype and x.device == reference.device
                for pair in pairs
                for x in pair
            ):
                packed = reference.new_empty(
                    (len(pairs), 2, reference.shape[0], reference.shape[1], capacity, reference.shape[3])
                )
                layers = _layer_views(packed)
        for key, value in pairs if packed is None else []:
            k = torch.empty(
                key.shape[0], key.shape[1], capacity, key.shape[3], device=key.device, dtype=key.dtype
            )
            v = torch.empty(
                value.shape[0],
                value.shape[1],
                capacity,
                value.shape[3],
                device=value.device,
                dtype=value.dtype,
            )
            layers.append(_LayerBuffer(k, v))
        device = token_ids.device
        return cls(
            layers=layers,
            token_ids_buffer=torch.empty(capacity, device=device, dtype=token_ids.dtype),
            attention_buffer=torch.empty(capacity, device=device, dtype=attention_mask.dtype),
            position_buffer=torch.empty(3, capacity, device=device, dtype=position_ids.dtype),
            length=0,
            max_length=max_length,
            _packed_kv=packed,
        )

    def _reserve(self, required: int) -> None:
        if required > self.max_length:
            raise ValueError(f"ActiveVLN context exceeds max length {self.max_length}: {required}")
        if required <= self.capacity:
            return
        capacity = min(self.max_length, max(required, self.capacity * 2))
        if self._packed_kv is not None:
            old = self._packed_kv
            packed = old.new_empty((*old.shape[:-2], capacity, old.shape[-1]))
            packed[..., : self.length, :].copy_(self.packed_kv)
            self._packed_kv = packed
            self.layers = _layer_views(packed)
        for layer in self.layers if self._packed_kv is None else []:
            key = torch.empty(
                layer.key.shape[0],
                layer.key.shape[1],
                capacity,
                layer.key.shape[3],
                device=layer.key.device,
                dtype=layer.key.dtype,
            )
            value = torch.empty(
                layer.value.shape[0],
                layer.value.shape[1],
                capacity,
                layer.value.shape[3],
                device=layer.value.device,
                dtype=layer.value.dtype,
            )
            key[:, :, : self.length].copy_(layer.key[:, :, : self.length])
            value[:, :, : self.length].copy_(layer.value[:, :, : self.length])
            layer.key, layer.value = key, value
        tokens = torch.empty(capacity, device=self.token_ids_buffer.device, dtype=self.token_ids_buffer.dtype)
        attention = torch.empty(
            capacity, device=self.attention_buffer.device, dtype=self.attention_buffer.dtype
        )
        positions = torch.empty(
            3, capacity, device=self.position_buffer.device, dtype=self.position_buffer.dtype
        )
        tokens[: self.length].copy_(self.token_ids_buffer[: self.length])
        attention[: self.length].copy_(self.attention_buffer[: self.length])
        positions[:, : self.length].copy_(self.position_buffer[:, : self.length])
        self.token_ids_buffer, self.attention_buffer, self.position_buffer = tokens, attention, positions

    def append_chunk(
        self,
        kv: _KVChunk,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        prompt_hash: str | None = None,
        next_position: int | None = None,
    ) -> None:
        """Append a private chunk; optional next_position describes the complete history.

        Position-building callers can supply their already known CPU maximum to
        avoid a GPU reduction/synchronization for every generated token. Calls
        without it invalidate the cached value and retain the reference lookup.
        """
        if next_position is not None and (type(next_position) is not int or next_position < 0):
            raise ValueError("next_position must be a nonnegative Python integer")
        q_len = token_ids.shape[-1]
        if len(kv) != len(self.layers):
            raise ValueError("ActiveVLN layer-cache count changed during append")
        if position_ids.shape != (3, 1, q_len):
            raise ValueError(f"position_ids must be [3,1,Q], got {tuple(position_ids.shape)}")
        if isinstance(kv, torch.Tensor) and (kv.ndim != 6 or kv.shape[1] != 2 or kv.shape[-2] != q_len):
            raise ValueError("packed ActiveVLN KV must have shape [layers,2,batch,heads,Q,width]")
        required = self.length + q_len
        if self._packed_kv is not None and torch.is_grad_enabled():
            # Independent layer buffers avoid shared autograd version counters.
            # Leave any prefix views already saved for backward untouched.
            self.layers = [_LayerBuffer(layer.key.clone(), layer.value.clone()) for layer in self.layers]
            self._packed_kv = None
        self._reserve(required)
        start, end = self.length, required
        if self._packed_kv is not None and isinstance(kv, torch.Tensor):
            self._packed_kv[..., start:end, :].copy_(kv)
        else:
            for layer, (key, value) in zip(self.layers, _pairs(kv)):
                layer.key[:, :, start:end].copy_(key)
                layer.value[:, :, start:end].copy_(value)
        self.token_ids_buffer[start:end].copy_(token_ids[0])
        self.attention_buffer[start:end].copy_(attention_mask[0])
        self.position_buffer[:, start:end].copy_(position_ids[:, 0])
        self.length = end
        self._next_position = next_position
        if prompt_hash is not None:
            self.prompt_hashes += (prompt_hash,)


@dataclass(frozen=True)
class BranchedActiveVLNMemory:
    """A serial-ragged group of independent ActiveVLN memories.

    This is intentionally a control-plane container, not a claim of a padded
    KV batch.  Decode consumes its rows serially today, which is the exact
    correctness anchor for a future length-bucketed kernel.
    """

    branches: tuple[ActiveVLNMemory, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("branched memory cannot be empty")

    @property
    def seq_len(self) -> int:
        return max(branch.seq_len for branch in self.branches)

    @property
    def batch_size(self) -> int:
        return len(self.branches)

    @property
    def seq_lens(self) -> tuple[int, ...]:
        return tuple(branch.seq_len for branch in self.branches)

    def to(self, device: torch.device | str) -> BranchedActiveVLNMemory:
        return BranchedActiveVLNMemory(tuple(branch.to(device) for branch in self.branches))

    def expand(self, num_samples: int) -> BranchedActiveVLNMemory:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        return BranchedActiveVLNMemory(
            tuple(branch.fork() for branch in self.branches for _ in range(num_samples))
        )

    def compact(
        self, keep: Sequence[int] | torch.Tensor | None = None
    ) -> BranchedActiveVLNMemory | ActiveVLNMemory:
        if keep is None:
            return self
        indices = keep.tolist() if isinstance(keep, torch.Tensor) else list(keep)
        selected = tuple(self.branches[int(index)] for index in indices)
        if not selected:
            raise ValueError("cannot compact all ActiveVLN branches")
        return selected[0] if len(selected) == 1 else BranchedActiveVLNMemory(selected)
