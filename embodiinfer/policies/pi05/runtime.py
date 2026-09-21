"""PI0.5 inference caches and graphs.

The engine keeps its public FlowDecoder contract. Schedule-specific AdaRMS
projections and compact camera/text layouts remain local to this policy.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from ...models.schedulers.flow import euler_step
from ..decoder import FlowDecoder
from .processor_pi05 import Pi05Batch

if TYPE_CHECKING:
    from ...engine.graph import GraphManager
    from .modeling_pi05 import Pi05Policy, Pi05Prefix


def _compact_layout(batch: Pi05Batch) -> tuple[Pi05Batch, bool]:
    """Remove only globally masked cameras and trailing language padding.

    Mixed-camera batches retain a view whenever any row uses it. Token order,
    holes, and left padding are preserved, so cumulative RoPE positions agree.
    Host mask inspection happens before graph capture and is included in timing.
    """
    if not batch.images or len(batch.images) != len(batch.img_masks):
        raise ValueError("PI0.5 requires matching nonempty images and camera masks")
    if batch.tokens.ndim != 2 or batch.masks.shape != batch.tokens.shape:
        raise ValueError("PI0.5 language tokens and masks must share [batch, length]")
    if batch.masks.dtype != torch.bool or any(mask.dtype != torch.bool for mask in batch.img_masks):
        raise ValueError("PI0.5 camera and language masks must be boolean")
    size = batch.batch_size
    if size < 1 or batch.tokens.shape[1] < 1:
        raise ValueError("PI0.5 requires a nonempty batch and language sequence")
    if any(mask.device != batch.masks.device for mask in batch.img_masks):
        raise ValueError("PI0.5 camera and language masks must share a device")
    for image, mask in zip(batch.images, batch.img_masks, strict=True):
        if image.ndim != 4 or image.shape[0] != size or mask.shape != (size,):
            raise ValueError(
                "PI0.5 camera tensors must be [batch, channels, height, width] with [batch] masks"
            )
    # One small transfer replaces a synchronization for each individual mask.
    host_masks = torch.cat([m[:, None] for m in batch.img_masks] + [batch.masks], dim=1).detach().cpu()
    cameras = host_masks[:, : len(batch.images)]
    cpu_mask = host_masks[:, len(batch.images) :]
    active = [i for i in range(len(batch.images)) if bool(cameras[:, i].any())]
    if not active:
        raise ValueError("PI0.5 requires at least one active camera in the batch")
    visible = host_masks.any(dim=1)
    if not bool(visible.all()):
        raise ValueError("every PI0.5 observation must contain at least one valid prefix token")
    positions = torch.arange(1, cpu_mask.shape[1] + 1)
    length = int((cpu_mask * positions).max())
    bucket = next((n for n in (16, 32, 48, 64, 96, 128, 160, 200) if n >= length), batch.tokens.shape[1])
    bucket = min(bucket, batch.tokens.shape[1])
    compacted = Pi05Batch(
        [batch.images[i] for i in active],
        [batch.img_masks[i] for i in active],
        batch.tokens[:, :bucket].contiguous(),
        batch.masks[:, :bucket].contiguous(),
        batch.request_ids,
    )
    all_valid = bool(cameras[:, active].all()) and bool(cpu_mask[:, :bucket].all())
    return compacted, all_valid


def compact_batch(batch: Pi05Batch) -> Pi05Batch:
    """Remove globally masked cameras and trailing padding while preserving valid positions."""
    return _compact_layout(batch)[0]


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tensor.shape, tensor.dtype, tensor.device, tensor.stride()


def _stream_key(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _clone_prefix(prefix: Pi05Prefix) -> Pi05Prefix:
    from .modeling_pi05 import Pi05Prefix

    return Pi05Prefix(
        [(k.clone(), v.clone()) for k, v in prefix.kv],
        prefix.prefix_pad_masks.clone(),
        None if prefix.last_hidden is None else prefix.last_hidden.clone(),
        prefix.all_valid,
    )


class _PrefixGraph:
    def __init__(self, policy: Pi05Policy, batch: Pi05Batch, all_valid: bool):
        self.batch = Pi05Batch(
            [x.clone() for x in batch.images],
            [x.clone() for x in batch.img_masks],
            batch.tokens.clone(),
            batch.masks.clone(),
        )
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=batch.tokens.device)
        stream.wait_stream(torch.cuda.current_stream(batch.tokens.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                policy._encode_prefix_impl(self.batch, all_valid=all_valid)
        torch.cuda.current_stream(batch.tokens.device).wait_stream(stream)
        with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="thread_local"):
            self.output = policy._encode_prefix_impl(self.batch, all_valid=all_valid)

    def run(self, batch: Pi05Batch) -> Pi05Prefix:
        for dst, src in zip(
            self.batch.images + self.batch.img_masks, batch.images + batch.img_masks, strict=True
        ):
            dst.copy_(src)
        self.batch.tokens.copy_(batch.tokens)
        self.batch.masks.copy_(batch.masks)
        self.graph.replay()
        return _clone_prefix(self.output)


class _LoopGraph:
    def __init__(self, runtime: Pi05Runtime, state: torch.Tensor, prefix: Pi05Prefix, steps: int):
        self.state = torch.empty_like(state)
        self.prefix = _clone_prefix(prefix)
        self.graph = torch.cuda.CUDAGraph()
        self.state.copy_(state)
        stream = torch.cuda.Stream(device=state.device)
        stream.wait_stream(torch.cuda.current_stream(state.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                runtime.integrate(self.state, self.prefix, steps)
        torch.cuda.current_stream(state.device).wait_stream(stream)
        with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="thread_local"):
            self.output = runtime.integrate(self.state, self.prefix, steps)

    def run(self, state: torch.Tensor, prefix: Pi05Prefix) -> torch.Tensor:
        self.state.copy_(state)
        self.prefix.prefix_pad_masks.copy_(prefix.prefix_pad_masks)
        for (dk, dv), (sk, sv) in zip(self.prefix.kv, prefix.kv, strict=True):
            dk.copy_(sk)
            dv.copy_(sv)
        self.graph.replay()
        return self.output.clone()


class Pi05Runtime:
    """Bounded inference caches; ownership and invalidation follow one policy."""

    def __init__(self, policy: Pi05Policy):
        self.policy = policy
        self.prefix_graphs: OrderedDict[tuple, _PrefixGraph] = OrderedDict()
        self.loop_graphs: OrderedDict[tuple, _LoopGraph] = OrderedDict()
        self.schedules: OrderedDict[tuple, Any] = OrderedDict()
        self.lock = threading.RLock()
        self.capture_count = 0

    def clear(self) -> None:
        """Discard weight-derived schedules and graphs after refit or migration."""
        with self.lock:
            self.prefix_graphs.clear()
            self.loop_graphs.clear()
            self.schedules.clear()

    @staticmethod
    def _remember(cache: OrderedDict, key: tuple, value: Any, limit: int = 8) -> Any:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)
        return value

    def encode(self, batch: Pi05Batch, return_hidden: bool) -> Pi05Prefix:
        """Encode the compact valid layout, optionally through a prefix graph."""
        with self.lock:
            batch, all_valid = _compact_layout(batch)
            if not self.policy.prefix_cuda_graph or return_hidden or batch.tokens.device.type != "cuda":
                return self.policy._encode_prefix_impl(batch, return_hidden, all_valid=all_valid)
            key = (
                tuple(_tensor_key(x) for x in batch.images + batch.img_masks),
                _tensor_key(batch.tokens),
                _tensor_key(batch.masks),
                all_valid,
                _stream_key(batch.tokens.device),
            )
            graph = self.prefix_graphs.get(key)
            if graph is None:
                graph = self._remember(self.prefix_graphs, key, _PrefixGraph(self.policy, batch, all_valid))
                self.capture_count += 1
            return graph.run(batch)

    def schedule(self, state: torch.Tensor, steps: int) -> tuple:
        """Cache the exact per-step, per-batch time projections used by eager."""
        if steps < 1:
            raise ValueError("PI0.5 requires a positive number of denoise steps")
        key = (steps, state.shape[0], state.dtype, state.device, _stream_key(state.device))
        cached = self.schedules.get(key)
        if cached is None:
            policy = self.policy
            rows = []
            norms = [
                n
                for layer in policy._expert_tower.layers
                for n in (layer.input_layernorm, layer.post_attention_layernorm)
            ]
            norms.append(policy._expert_tower.norm)
            for time, dt in policy.flow_schedule(steps):
                t = torch.full((state.shape[0],), time, device=state.device, dtype=state.dtype)
                condition = policy._time_condition(t)
                rows.append((t, dt, tuple(norm.dense(condition) for norm in norms)))
            cached = self._remember(self.schedules, key, tuple(rows))
        return cached

    def integrate(self, state: torch.Tensor, prefix: Pi05Prefix, steps: int) -> torch.Tensor:
        """Run every Euler step with immutable, step-indexed AdaRMS projections."""
        x = state
        for t, dt, modulations in self.schedule(state, steps):
            velocity = self.policy._denoise_step_impl(x, t, prefix, modulations)
            x = euler_step(x, velocity, dt)
        return x

    def decode(self, state: torch.Tensor, prefix: Pi05Prefix, steps: int, use_graph: bool) -> torch.Tensor:
        """Execute the complete schedule; graph replay returns independent storage."""
        with self.lock:
            if not use_graph:
                return self.integrate(state, prefix, steps)
            key = (
                _tensor_key(state),
                steps,
                _tensor_key(prefix.prefix_pad_masks),
                prefix.all_valid,
                _stream_key(state.device),
            )
            graph = self.loop_graphs.get(key)
            if graph is None:
                graph = self._remember(self.loop_graphs, key, _LoopGraph(self, state, prefix, steps))
                self.capture_count += 1
            return graph.run(state, prefix)

    def stats(self) -> dict[str, int]:
        """Expose actual cache/capture counts for benchmark warmup verification."""
        return dict(
            prefix_graphs=len(self.prefix_graphs),
            loop_graphs=len(self.loop_graphs),
            schedules=len(self.schedules),
            capture_count=self.capture_count,
        )


class Pi05FlowDecoder(FlowDecoder):
    """Specialize deterministic inference while inheriting the public RL paths."""

    def integrate(
        self,
        state: torch.Tensor | None,
        prefix: Pi05Prefix,
        num_steps: int,
        bucket: int,
        graphs: GraphManager | None,
    ) -> torch.Tensor:
        """Return model-space actions using cached schedules for CUDA eval without autograd."""
        policy = self.policy
        if state is None:
            raise ValueError("Flow decoding requires an initial state")
        if not policy._native_enabled() or state.device.type != "cuda":
            return super().integrate(state, prefix, num_steps, bucket, graphs)
        if state.shape[0] != bucket:
            raise ValueError("PI0.5 state batch must match the decode bucket")
        policy._prepare_native_attention()
        return policy._runtime.decode(state, prefix, num_steps, graphs is not None and graphs.full_loop)
