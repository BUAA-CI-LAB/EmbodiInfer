"""PI0.5 inference caches and graphs.

The engine keeps its public ActionDecoder contract. Schedule-specific AdaRMS
projections and compact camera/text layouts remain local to this policy.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from ....backend.torch.request_graph import (
    CUDA_GRAPH_CAPTURE_LOCK,
    CapturedRequest,
    cuda_execution_signature,
)
from ....models.schedulers.flow import euler_step
from ...decoder import DecodeResult, FlowActionDecoder
from ..processor_pi05 import Pi05Batch

if TYPE_CHECKING:
    from ....engine.graph import GraphManager
    from ..modeling_pi05 import Pi05Policy, Pi05Prefix


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


def _compact_all_valid(batch: Pi05Batch) -> Pi05Batch:
    """Gather B1 valid prompt tokens before mask-free FlashAttention capture."""
    if batch.batch_size != 1:
        raise ValueError("Mask-free Pi05 prefix compaction requires B1")
    columns = batch.masks[0].nonzero().flatten()
    return Pi05Batch(
        batch.images,
        batch.img_masks,
        batch.tokens.index_select(1, columns),
        batch.masks.index_select(1, columns),
        batch.request_ids,
    )


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tensor.shape, tensor.dtype, tensor.device, tensor.stride()


def _stream_key(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _clone_prefix(prefix: Pi05Prefix) -> Pi05Prefix:
    from ..modeling_pi05 import Pi05Prefix

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
        getter = getattr(policy, "_get_optimizations", None)
        self.operators = getter() if getter is not None else None
        self.scope = object()
        self.device = batch.tokens.device
        context = nullcontext() if self.operators is None else self.operators.execution_scope(self.scope)
        try:
            with context:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        policy._encode_prefix_impl(self.batch, all_valid=all_valid)
                torch.cuda.current_stream(batch.tokens.device).wait_stream(stream)
                with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="thread_local"):
                    self.output = policy._encode_prefix_impl(self.batch, all_valid=all_valid)

        except BaseException:
            if self.operators is not None:
                torch.cuda.synchronize(self.device)
                self.graph.reset()
                self.operators.release_scope(self.scope)
            raise

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
        getter = getattr(runtime.policy, "_get_optimizations", None)
        self.operators = getter() if getter is not None else None
        self.scope = object()
        self.device = state.device
        context = nullcontext() if self.operators is None else self.operators.execution_scope(self.scope)
        try:
            with context:
                with torch.cuda.stream(stream):
                    # Graph inputs must outlive eviction from the shared schedule LRU.
                    self.schedule = runtime.schedule(self.state, steps)
                    for _ in range(3):
                        runtime.integrate(self.state, self.prefix, steps)
                torch.cuda.current_stream(state.device).wait_stream(stream)
                with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="thread_local"):
                    self.output = runtime.integrate(self.state, self.prefix, steps)
                self.suffix_attention = getattr(runtime.policy, "_cached_suffix_att", None)

        except BaseException:
            if self.operators is not None:
                torch.cuda.synchronize(self.device)
                self.graph.reset()
                self.operators.release_scope(self.scope)
            raise

    def run(self, state: torch.Tensor, prefix: Pi05Prefix) -> torch.Tensor:
        self.state.copy_(state)
        self.prefix.prefix_pad_masks.copy_(prefix.prefix_pad_masks)
        for (dk, dv), (sk, sv) in zip(self.prefix.kv, prefix.kv, strict=True):
            dk.copy_(sk)
            dv.copy_(sv)
        self.graph.replay()
        return self.output.clone()


class _RequestGraph:
    """Pi05 adapter: keep the prefix private inside one captured request."""

    def __init__(
        self, runtime: Pi05Runtime, batch: Pi05Batch, state: torch.Tensor, steps: int, all_valid: bool
    ):
        self.batch = Pi05Batch(
            [x.clone() for x in batch.images],
            [x.clone() for x in batch.img_masks],
            batch.tokens.clone(),
            batch.masks.clone(),
        )
        self.state = state.clone()
        self.device = state.device
        self.operators = runtime.policy._get_optimizations()
        self.scope = object()

        def forward() -> torch.Tensor:
            # Capture keeps addresses, not Python owners of cached graph inputs.
            self.schedule = runtime.schedule(self.state, steps)
            self.prefix = runtime.policy._encode_prefix_impl(self.batch, all_valid=all_valid)
            actions = runtime.integrate(self.state, self.prefix, steps)
            self.suffix_attention = getattr(runtime.policy, "_cached_suffix_att", None)
            return runtime.policy.finalize_actions(actions, self.prefix)

        operators, scope = self.operators, self.scope
        self.captured = CapturedRequest(
            forward,
            self.device,
            execution_context=(
                nullcontext if self.operators is None else lambda: self.operators.execution_scope(self.scope)
            ),
            release=None if operators is None else lambda: operators.release_scope(scope),
        )

    def run(self, batch: Pi05Batch, state: torch.Tensor) -> torch.Tensor:
        def update() -> None:
            for dst, src in zip(
                self.batch.images + self.batch.img_masks, batch.images + batch.img_masks, strict=True
            ):
                dst.copy_(src)
            self.batch.tokens.copy_(batch.tokens)
            self.batch.masks.copy_(batch.masks)
            self.state.copy_(state)

        return self.captured.run(update)

    def close(self, *, synchronize: bool = True) -> None:
        self.captured.close(synchronize=synchronize)


def _close_graph(graph: Any, *, synchronize: bool = True) -> None:
    if isinstance(graph, _RequestGraph):
        graph.close(synchronize=synchronize)
        return
    cuda_graph = getattr(graph, "graph", None)
    if cuda_graph is None:
        return
    if synchronize:
        torch.cuda.synchronize(graph.device)
    cuda_graph.reset()
    operators = getattr(graph, "operators", None)
    if operators is not None:
        operators.release_scope(graph.scope)


class Pi05Runtime:
    """Bounded inference caches; ownership and invalidation follow one policy."""

    def __init__(self, policy: Pi05Policy):
        self.policy = policy
        self.prefix_graphs: OrderedDict[tuple, _PrefixGraph] = OrderedDict()
        self.loop_graphs: OrderedDict[tuple, _LoopGraph] = OrderedDict()
        self.request_graphs: OrderedDict[tuple, _RequestGraph] = OrderedDict()
        self.schedules: OrderedDict[tuple, Any] = OrderedDict()
        self.lock = threading.RLock()
        self.capture_count = 0

    def clear(self) -> None:
        """Discard weight-derived schedules and graphs after refit or migration."""
        with self.lock:
            operators = getattr(self.policy, "_fused_ops", None)
            graphs = (*self.prefix_graphs.values(), *self.loop_graphs.values(), *self.request_graphs.values())
            devices = {graph.device for graph in graphs}
            if operators is not None:
                # ctypes launches do not participate in Torch's stream recording.
                devices.add(self.policy._m.action_in_proj.weight.device)
            for device in devices:
                torch.cuda.synchronize(device)
            for graph in graphs:
                _close_graph(graph, synchronize=False)
            self.prefix_graphs.clear()
            self.loop_graphs.clear()
            self.request_graphs.clear()
            self.schedules.clear()

    @staticmethod
    def _remember(cache: OrderedDict, key: tuple, value: Any, limit: int = 8) -> Any:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            _, evicted = cache.popitem(last=False)
            _close_graph(evicted)
        return value

    def _prepare_batch(self, batch: Pi05Batch, return_hidden: bool = False) -> tuple[Pi05Batch, bool]:
        """Use the same camera/token compaction for separate and combined execution."""
        batch, all_valid = _compact_layout(batch)
        config = getattr(self.policy, "_optimization_config", None)
        if (
            config is not None
            and (config.compact_prefix or config.attention == "folded_flash")
            and not return_hidden
        ):
            columns = batch.masks.any(dim=0).nonzero().flatten()
            batch = Pi05Batch(
                batch.images,
                batch.img_masks,
                batch.tokens.index_select(1, columns),
                batch.masks.index_select(1, columns),
                batch.request_ids,
            )
            all_valid = bool(batch.masks.all()) and all(bool(mask.all()) for mask in batch.img_masks)
            if config.attention == "folded_flash" and not all_valid:
                raise ValueError("Folded FlashAttention requires all-valid compact prefix masks")
        return batch, all_valid

    def encode(self, batch: Pi05Batch, return_hidden: bool) -> Pi05Prefix:
        """Encode the compact valid layout, optionally through a prefix graph."""
        with self.lock:
            batch, all_valid = self._prepare_batch(batch, return_hidden)
            if not self.policy.prefix_cuda_graph or return_hidden or batch.tokens.device.type != "cuda":
                return self.policy._encode_prefix_impl(batch, return_hidden, all_valid=all_valid)
            key = (
                tuple(_tensor_key(x) for x in batch.images + batch.img_masks),
                _tensor_key(batch.tokens),
                _tensor_key(batch.masks),
                all_valid,
                _stream_key(batch.tokens.device),
                cuda_execution_signature(),
            )
            graph = self.prefix_graphs.get(key)
            if graph is None:
                with CUDA_GRAPH_CAPTURE_LOCK:
                    graph = self._remember(
                        self.prefix_graphs, key, _PrefixGraph(self.policy, batch, all_valid)
                    )
                self.capture_count += 1
            return graph.run(batch)

    def schedule(self, state: torch.Tensor, steps: int) -> tuple:
        """Cache the exact per-step, per-batch time projections used by eager."""
        if steps < 1:
            raise ValueError("PI0.5 requires a positive number of denoise steps")
        key = (
            steps,
            state.shape[0],
            state.dtype,
            state.device,
            _stream_key(state.device),
            cuda_execution_signature(),
        )
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
        getter = getattr(self.policy, "_get_optimizations", None)
        operators = getter() if getter is not None else None
        if operators is not None:
            operators.validate_request(state.shape[0], steps)
        context = nullcontext() if operators is None else operators.decode_context(prefix)
        x = state
        device_context = torch.cuda.device(state.device) if state.is_cuda else nullcontext()
        with device_context, context:
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
                cuda_execution_signature(),
            )
            graph = self.loop_graphs.get(key)
            if graph is None:
                with CUDA_GRAPH_CAPTURE_LOCK:
                    graph = self._remember(self.loop_graphs, key, _LoopGraph(self, state, prefix, steps))
                self.capture_count += 1
            return graph.run(state, prefix)

    def request(self, batch: Pi05Batch, state: torch.Tensor, steps: int) -> torch.Tensor:
        """Replay prefix plus the unchanged flow schedule with a private shared prefix.

        The input signature includes every tensor's layout, the active stream,
        step count and policy version. No prefix escapes, so overlapping requests
        cannot observe or overwrite another request's K/V. Public two-stage calls
        still return owned prefix tensors and work independently of this cache.
        """
        if not self.policy._native_enabled() or state.device.type != "cuda":
            raise ValueError("Combined Pi05 requests require native CUDA eval/no-grad execution")
        if state.ndim != 3 or state.shape[0] != batch.batch_size or steps < 1:
            raise ValueError("Pi05 request state must be [batch, horizon, action_dim] with positive steps")
        if tuple(state.shape) != self.policy.flow_state_shape(batch.batch_size):
            raise ValueError(
                "Pi05 request state must match the policy's configured horizon and action dimension"
            )
        if state.device != batch.tokens.device:
            raise ValueError("Pi05 request state and batch must share a CUDA device")
        with self.lock:
            self.policy._prepare_native_attention()
            batch, all_valid = self._prepare_batch(batch)
            key = (
                tuple(_tensor_key(x) for x in batch.images + batch.img_masks),
                _tensor_key(batch.tokens),
                _tensor_key(batch.masks),
                _tensor_key(state),
                all_valid,
                steps,
                self.policy.policy_version,
                _stream_key(state.device),
                cuda_execution_signature(),
            )
            graph = self.request_graphs.get(key)
            if graph is None:
                graph = self._remember(
                    self.request_graphs, key, _RequestGraph(self, batch, state, steps, all_valid)
                )
                self.capture_count += 1
            else:
                self.request_graphs.move_to_end(key)
            return graph.run(batch, state)

    def stats(self) -> dict[str, int]:
        """Expose actual cache/capture counts for benchmark warmup verification."""
        return dict(
            prefix_graphs=len(self.prefix_graphs),
            loop_graphs=len(self.loop_graphs),
            request_graphs=len(self.request_graphs),
            schedules=len(self.schedules),
            capture_count=self.capture_count,
        )


class Pi05FlowDecoder(FlowActionDecoder):
    """Expose deterministic inference without the policy-gradient rollout capability."""

    def supports_request_execution(self, graphs: GraphManager | None) -> bool:
        """Opt in only for CUDA native inference with a full-loop graph manager."""
        return (
            getattr(self.policy, "request_cuda_graph", False)
            and self.policy._native_enabled()
            and graphs is not None
            and graphs.full_loop
        )

    def decode_request(
        self,
        state: torch.Tensor | None,
        batch: Pi05Batch,
        num_steps: int,
        bucket: int,
        graphs: GraphManager | None,
    ) -> DecodeResult:
        """Run an opt-in whole request without exporting its intermediate prefix."""
        if not self.supports_request_execution(graphs):
            raise ValueError("Pi05 whole-request execution is not enabled for this mode")
        if state is None or state.shape[0] != bucket or batch.batch_size != bucket:
            raise ValueError("Pi05 request batch and initial state must match the decode bucket")
        return DecodeResult(actions=self.policy._runtime.request(batch, state, num_steps))

    def integrate(
        self,
        state: torch.Tensor | None,
        prefix: Pi05Prefix,
        num_steps: int,
        bucket: int,
        graphs: GraphManager | None,
        *,
        guidance=None,
    ) -> torch.Tensor:
        """Return model-space actions using cached schedules for CUDA eval without autograd.

        RTC guidance deliberately takes the generic eager path: soft RTC needs a
        Jacobian-vector product through the velocity field and cannot be served
        by the native cached schedule or a captured loop.
        """
        policy = self.policy
        if state is None:
            raise ValueError("Flow decoding requires an initial state")
        if guidance is not None and guidance.enabled:
            return super().integrate(state, prefix, num_steps, bucket, graphs, guidance=guidance)
        if not policy._native_enabled() or state.device.type != "cuda":
            return super().integrate(state, prefix, num_steps, bucket, graphs)
        if state.shape[0] != bucket:
            raise ValueError("PI0.5 state batch must match the decode bucket")
        policy._prepare_native_attention()
        return policy._runtime.decode(state, prefix, num_steps, graphs is not None and graphs.full_loop)
