"""Static-shape CUDA-graph capture of the action-decode step.

Whatever the paradigm, the decode runs with *identical shapes* every prediction — a flow
``denoise_step`` (x_t: [B, H, A], t: [B], fixed-shape prefix) executed N times, or a
single-pass policy's fixed decode over a padded prefix (OpenVLA-OFT). This is the ideal
CUDA-graph target: capture once, replay, and the per-kernel Python/launch overhead —
which dominates at the batch sizes typical of VLA serving — is amortized away.
(Cosmos's diffusion per-step DiT graph is the same idea, currently captured in the Cosmos
policy; unifying it here is tracked as future work.)

Capture granularities provided:

  * :class:`ForwardGraph` — a single fixed-shape forward for a non-flow policy
    (OpenVLA-OFT's action decode); no per-step ``x_t``/``t``.

For the flow denoise loop, two granularities:

  * :class:`DenoiseGraph` captures a *single* denoise step; the engine drives the
    N-step loop in Python, replaying once per step and doing the Euler update
    itself. This removes the *intra-step* launch overhead.
  * :class:`LoopGraph` captures the *whole* N-step integration (denoise + Euler +
    time progression) as one graph. Because the flow schedule is static — the
    ``(t, dt)`` sequence is value-independent, only ``x`` is carried along — the
    entire loop is capturable, and one replay per prediction removes the
    *inter-step* host overhead too. See docs/proposals/0001-static-loop-capture.md.

The prefix has a policy-specific layout, so a graph does not allocate it
directly: it asks the policy for a static-shape prefix buffer
(``allocate_static_prefix``) and, once per generation, for an in-place copy of
the live prefix into that buffer (``copy_prefix_into`` via ``set_prefix``). The
prefix is constant across the N denoise steps, so it is copied once — not per
replay. Only policies that report ``supports_cuda_graph`` reach this module.

Prefill (``encode_prefix``) is left eager: it runs once per observation and its
cost is compute, not launch overhead.

Captures run with ``capture_error_mode="thread_local"``: the default (global)
mode invalidates CUDA calls issued by OTHER threads for the duration of the
capture, which corrupts hosts that run background communication threads (e.g.
an RL trainer's collective send/recv loops in the same process). Thread-local
capture restricts the recording to the capturing thread only.
"""

from __future__ import annotations

import threading

import torch

from ..models.schedulers.flow import euler_step, sde_coefficients, sde_transition_mean
from ..policies.base import PrefixState, VLAPolicy

# PyTorch/CUDA graph capture has process-global runtime state even when captures
# target different devices and use ``capture_error_mode="thread_local"``.  Two
# replicas may replay concurrently, but their one-time lazy captures must not
# enter capture_begin at the same instant (cudaErrorIllegalState on dual-GPU
# in-process DP).
_CAPTURE_LOCK = threading.Lock()


def _flow_state_shape(policy: VLAPolicy, batch: int) -> tuple[int, int, int]:
    """Resolve decoder-owned internal state shape, preserving config fallback."""
    shape = policy.decoder.state_shape(batch)
    if shape is None:
        c = policy.config
        shape = (batch, c.action_horizon, c.action_dim)
    shape = tuple(int(dim) for dim in shape)
    if len(shape) != 3 or shape[0] != batch or any(dim <= 0 for dim in shape):
        raise ValueError(f"flow decoder state_shape must be [B,H,A], got {shape}")
    return shape


def _build_time_schedule(
    policy: VLAPolicy, num_steps: int, batch: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, list[float]]:
    """Materialize the static ``(t, dt)`` schedule for whole-loop capture.

    ``flow_schedule`` is deterministic and value-independent, so all N time
    tensors can be built once before capture: ``t_all[k]`` is the ``[batch]``
    time tensor for step ``k`` and ``dts[k]`` its Euler coefficient. Returns
    ``(t_all, dts)``. Pure tensor construction (no CUDA), so it is CPU-testable.
    """
    schedule = policy.flow_schedule(num_steps)
    t_all = torch.empty(num_steps, batch, device=device, dtype=dtype)
    dts: list[float] = []
    for k, (t_val, dt) in enumerate(schedule):
        t_all[k].fill_(t_val)
        dts.append(dt)
    return t_all, dts


class DenoiseGraph:
    """A captured denoise_step for one fixed (bucket) batch size."""

    def __init__(
        self,
        policy: VLAPolicy,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        graph_variant: object | None = None,
        prefix: PrefixState | None = None,
    ):
        self.policy = policy
        self.batch = batch
        self.device = device
        self.dtype = dtype
        # static input buffers (action + time are engine-contract shapes)
        self._x = torch.zeros(_flow_state_shape(policy, batch), device=device, dtype=dtype)
        self._t = torch.zeros(batch, device=device, dtype=dtype)
        # the prefix buffer is policy-specific; let the policy allocate it
        self._prefix = (
            policy.allocate_static_prefix_for_variant(batch, device, dtype, graph_variant)
            if prefix is None
            else policy.allocate_static_prefix_from_live(prefix, batch, device, dtype, graph_variant)
        )
        self._graph = torch.cuda.CUDAGraph()
        self._out = None
        self._capture()

    def _capture(self) -> None:
        # warmup on a private stream so allocator state is graph-safe
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.policy.denoise_step(self._x, self._t, self._prefix)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self._graph, stream=s, capture_error_mode="thread_local"):
            self._out = self.policy.denoise_step(self._x, self._t, self._prefix)

    def set_prefix(self, prefix: PrefixState) -> None:
        """Copy the live prefix into the static buffer once for this generation.

        The prefix is constant across the denoise loop, so this runs once before
        the N replays rather than inside :meth:`run` — saving the per-step prefix
        copies (which for pi0.5 are one per expert layer)."""
        self.policy.copy_prefix_into(self._prefix, prefix)

    @torch.no_grad()
    def run(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self._x.copy_(x)
        self._t.copy_(t)
        self._graph.replay()
        return self._out


class ForwardGraph:
    """A captured single forward for a non-flow policy (OpenVLA-OFT's action decode).

    Single-pass policies have no ``x_t``/``t`` — the whole action chunk is a function of
    the cached prefix alone. This captures ``policy._decode(static_prefix)``: the engine
    copies the live prefix into the static buffer (``set_prefix``) and replays. The decode
    is a fixed-shape callable (the prompt is padded, so the prefix length is static), which
    is exactly what makes it CUDA-graph-capturable. Analogous to :class:`DenoiseGraph` but
    with no per-step input.
    """

    def __init__(self, policy: VLAPolicy, batch: int, device: torch.device, dtype: torch.dtype):
        self.policy = policy
        self.batch = batch
        self.device = device
        self.dtype = dtype
        self._prefix = policy.allocate_static_prefix(batch, device, dtype)
        self._graph = torch.cuda.CUDAGraph()
        self._out = None
        self._capture()

    def _capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.policy._decode(self._prefix)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self._graph, stream=s, capture_error_mode="thread_local"):
            self._out = self.policy._decode(self._prefix)

    def set_prefix(self, prefix: PrefixState) -> None:
        self.policy.copy_prefix_into(self._prefix, prefix)

    @torch.no_grad()
    def run(self) -> torch.Tensor:
        self._graph.replay()
        return self._out.clone()


class LoopGraph:
    """A captured whole N-step flow integration for one (bucket, num_steps).

    The flow schedule is static, so the entire loop — ``denoise_step``, the Euler
    update, and the time progression — is captured once and replayed once per
    prediction, replacing the per-step Python-driven loop. Only the initial noise
    ``x0`` (via :meth:`run`) and the prefix (via :meth:`set_prefix`) vary between
    predictions; the ``(t, dt)`` schedule is baked into the graph.
    """

    def __init__(
        self,
        policy: VLAPolicy,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        num_steps: int,
        graph_variant: object | None = None,
        prefix: PrefixState | None = None,
    ):
        self.policy = policy
        self.batch = batch
        self.device = device
        self.dtype = dtype
        self.num_steps = num_steps
        self._x = torch.zeros(_flow_state_shape(policy, batch), device=device, dtype=dtype)
        self._prefix = (
            policy.allocate_static_prefix_for_variant(batch, device, dtype, graph_variant)
            if prefix is None
            else policy.allocate_static_prefix_from_live(prefix, batch, device, dtype, graph_variant)
        )
        self._t_all, self._dts = _build_time_schedule(policy, num_steps, batch, device, dtype)
        self._graph = torch.cuda.CUDAGraph()
        self._out = None
        self._capture()

    def _integrate(self) -> torch.Tensor:
        # unrolled N-step Euler integration; identical op sequence to the engine's
        # eager loop, so capture is bit-exact against it.
        x = self._x
        for k in range(self.num_steps):
            v = self.policy.denoise_step(x, self._t_all[k], self._prefix)
            x = euler_step(x, v, self._dts[k])
        return x

    def _capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self._integrate()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self._graph, stream=s, capture_error_mode="thread_local"):
            self._out = self._integrate()

    def set_prefix(self, prefix: PrefixState) -> None:
        """Copy the live prefix into the static buffer once for this generation."""
        self.policy.copy_prefix_into(self._prefix, prefix)

    @torch.no_grad()
    def run(self, x0: torch.Tensor) -> torch.Tensor:
        """Replay the whole loop from initial noise ``x0``; returns ``x_N``."""
        self._x.copy_(x0)
        self._graph.replay()
        return self._out


class SdeLoopGraph:
    """A captured whole N-step *stochastic* (SDE) flow integration for RL rollout.

    Extends :class:`LoopGraph` with the rollout surface an on-policy RL trainer
    consumes — per-step transition noise, the visited trajectory, and the
    per-step velocities — while keeping the whole loop a single replay:

      * ``eps`` is drawn OUTSIDE the graph into a static ``[N, B, H, A]`` buffer
        (:meth:`sample` draws it, or takes it explicitly). The graph stays
        deterministic given its buffers, so replay parity is checkable
        bit-exactly against an eager reference consuming the same noise.
      * the per-step noise scale and drift correction are read from ``[N]``
        device buffers (see :func:`~embodiinfer.models.schedulers.flow.sde_coefficients`), so the mixed ODE/SDE
        pattern of flow-SDE trainers — noise on one randomly selected step per
        call — changes per call WITHOUT re-capture: ODE steps multiply the
        noise/correction by zero, keeping the captured op sequence identical.
      * the trajectory ``[B, N+1, H, A]`` and velocities ``[B, N, H, A]`` are
        recorded into static output buffers; :meth:`sample` returns clones (the
        buffers are reused by the next call).

    The transition mean is the score-corrected rectified-flow form
    ``x + v*dt - c_corr * (x + gamma*v)`` with ``gamma = t_noise - t`` derived
    from the schedule's integration direction (see :func:`~embodiinfer.models.schedulers.flow.sde_coefficients`),
    so one graph class serves both time conventions (pi0.5's 1 -> 0 and
    GR00T's 0 -> 1); with ``c_corr == 0`` it reduces to plain
    Euler-Maruyama / ODE.
    """

    def __init__(
        self,
        policy: VLAPolicy,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        num_steps: int,
        prefix_dtype: torch.dtype | None = None,
        action_shape: tuple[int, int] | None = None,
    ):
        self.policy = policy
        self.batch = batch
        self.device = device
        self.dtype = dtype
        self.num_steps = num_steps
        c = policy.config
        # The flow-state shape defaults to the policy config, but a trainer may
        # integrate over a different horizon than the checkpoint's chunk size
        # (e.g. RLinf pi0.5 samples a 10-step horizon against a 50-chunk
        # config); the denoise step handles the suffix length dynamically.
        H, A = action_shape if action_shape is not None else (c.action_horizon, c.action_dim)
        self._x = torch.zeros(batch, H, A, device=device, dtype=dtype)
        self._eps = torch.zeros(num_steps, batch, H, A, device=device, dtype=dtype)
        self._c_noise = torch.zeros(num_steps, device=device, dtype=dtype)
        self._c_corr = torch.zeros(num_steps, device=device, dtype=dtype)
        self._gamma = torch.zeros(num_steps, device=device, dtype=dtype)
        self._traj = torch.zeros(batch, num_steps + 1, H, A, device=device, dtype=dtype)
        self._vel = torch.zeros(batch, num_steps, H, A, device=device, dtype=dtype)
        # The prefix buffer dtype may differ from the flow-state dtype (e.g. a
        # bf16 backbone whose KV cache is bf16 while x/chains stay fp32).
        self._prefix = policy.allocate_static_prefix(
            batch, device, prefix_dtype if prefix_dtype is not None else dtype
        )
        self._schedule = policy.flow_schedule(num_steps)
        self._t_all, _ = _build_time_schedule(policy, num_steps, batch, device, dtype)
        self._graph = torch.cuda.CUDAGraph()
        self._capture()

    def _integrate_sde(self) -> None:
        x = self._x
        self._traj[:, 0].copy_(x)
        for k, (_t_val, dt) in enumerate(self._schedule):
            v = self.policy.denoise_step(x, self._t_all[k], self._prefix)
            self._vel[:, k].copy_(v)
            mean = sde_transition_mean(x, v, dt, self._c_corr[k], self._gamma[k])
            x = mean + self._c_noise[k] * self._eps[k]
            self._traj[:, k + 1].copy_(x)

    def _capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._integrate_sde()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self._graph, stream=s, capture_error_mode="thread_local"):
            self._integrate_sde()

    def set_prefix(self, prefix: PrefixState) -> None:
        """Copy the live prefix into the static buffer once for this generation."""
        self.policy.copy_prefix_into(self._prefix, prefix)

    @torch.no_grad()
    def sample(
        self,
        prefix: PrefixState,
        x0: torch.Tensor,
        sigmas: list[float],
        eps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One rollout generation: replay the captured loop from noise ``x0``.

        Args:
            prefix: live prefix state for this batch (copied into the static
                buffer).
            x0: ``[B, H, A]`` initial noise.
            sigmas: per-step noise scale ``sigma(t_k)``, length ``num_steps``
                (0.0 entries are deterministic ODE steps).
            eps: optional pre-drawn transition noise ``[N, B, H, A]``; drawn
                from ``generator`` (or the default RNG) when omitted. Only
                steps with ``sigma > 0`` consume it.
        Returns:
            ``(actions, trajectory, velocities)`` — ``[B, H, A]``,
            ``[B, N+1, H, A]``, ``[B, N, H, A]``; clones, safe to retain.
        """
        c_noise, c_corr, gamma = sde_coefficients(self._schedule, sigmas)
        self._c_noise.copy_(torch.tensor(c_noise, dtype=self.dtype))
        self._c_corr.copy_(torch.tensor(c_corr, dtype=self.dtype))
        self._gamma.copy_(torch.tensor(gamma, dtype=self.dtype))
        if eps is not None:
            self._eps.copy_(eps)
        else:
            torch.randn(self._eps.shape, generator=generator, out=self._eps)
        self.set_prefix(prefix)
        self._x.copy_(x0)
        self._graph.replay()
        traj = self._traj.clone()
        return traj[:, -1], traj, self._vel.clone()


class CosmosDenoiseGraph:
    """A captured per-step DiT denoise for Cosmos (diffusion WAM).

    Analogous to :class:`DenoiseGraph` but the staged inputs are ``(x, sigma)`` — the
    diffusion sampler's current latent + noise level — and the captured callable is the
    policy's preconditioned ``denoise`` (DiT x0-prediction with frame replacement). The
    prefix (gt_frames / mask / crossattn / padding) is copied into static buffers once per
    generation (:meth:`set_prefix`), so one capture replays for every sampler step; the
    2ab solver's host-side float64 arithmetic stays outside the graph (only the DiT forward
    is captured). Reproduces the eager result bit-exactly (same static buffers).
    """

    def __init__(self, policy: VLAPolicy, batch: int, device: torch.device, dtype: torch.dtype):
        self.policy = policy
        # the latent x + noise level are staged per step; both are fp32 (solver dtype)
        self._x = torch.zeros(
            batch,
            policy.latent_ch,
            policy.state_t,
            policy.latent_hw,
            policy.latent_hw,
            device=device,
            dtype=torch.float32,
        )
        self._sigma = torch.zeros(batch, device=device, dtype=torch.float32)
        self._prefix = policy.allocate_static_prefix(batch, device, dtype)
        self._graph = torch.cuda.CUDAGraph()
        self._out = None
        self._capture()

    def _capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warmup (also populates the DiT RoPE cache)
                _ = self.policy.denoise(self._x, self._sigma, self._prefix)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self._graph, stream=s, capture_error_mode="thread_local"):
            self._out = self.policy.denoise(self._x, self._sigma, self._prefix)

    def set_prefix(self, prefix: PrefixState) -> None:
        self.policy.copy_prefix_into(self._prefix, prefix)

    @torch.no_grad()
    def run(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        self._x.copy_(x)
        self._sigma.copy_(sigma)
        self._graph.replay()
        return self._out.clone()


_AnyGraph = DenoiseGraph | LoopGraph | ForwardGraph | CosmosDenoiseGraph


class GraphManager:
    """Lazily captures and caches graphs, keyed by ``(bucket, num_steps)``.

    The graph class is chosen from the policy's declared ``cuda_graph_kind`` (metadata the
    policy exposes — the engine never sniffs the policy's methods), mirroring vllm-omni's
    "model declares structure, the runner owns capture":

      * ``"flow"``           -> :class:`DenoiseGraph` (per-step) or :class:`LoopGraph`
                                (whole loop, ``full_loop``); keyed on ``num_steps`` only in
                                full-loop mode (the single-step graph is ``num_steps``-agnostic).
      * ``"single_forward"`` -> :class:`ForwardGraph` (OpenVLA-OFT's one decode).
      * ``"diffusion_step"`` -> :class:`CosmosDenoiseGraph` (Cosmos's per-step DiT).
    """

    def __init__(self, policy: VLAPolicy, device: torch.device, dtype: torch.dtype, full_loop: bool = False):
        self.policy = policy
        self.device = device
        self.dtype = dtype
        self.full_loop = full_loop
        self._graphs: dict[tuple[int, int | None, object | None], _AnyGraph] = {}

    def get(
        self,
        bucket: int,
        num_steps: int,
        graph_variant: object | None = None,
        prefix: PrefixState | None = None,
    ) -> _AnyGraph:
        kind = self.policy.cuda_graph_kind
        if kind == "single_forward":
            key = (bucket, None, graph_variant)
            factory = lambda: ForwardGraph(self.policy, bucket, self.device, self.dtype)  # noqa: E731
        elif kind == "diffusion_step":
            key = (bucket, None, graph_variant)  # one per-step DiT graph, replayed every sampler step
            factory = lambda: CosmosDenoiseGraph(self.policy, bucket, self.device, self.dtype)  # noqa: E731
        elif kind == "flow":
            if self.full_loop:
                key = (bucket, num_steps, graph_variant)
                factory = lambda: LoopGraph(  # noqa: E731
                    self.policy,
                    bucket,
                    self.device,
                    self.dtype,
                    num_steps,
                    graph_variant,
                    prefix,
                )
            else:
                key = (bucket, None, graph_variant)
                factory = lambda: DenoiseGraph(  # noqa: E731
                    self.policy,
                    bucket,
                    self.device,
                    self.dtype,
                    graph_variant,
                    prefix,
                )
        else:
            raise ValueError(
                f"{type(self.policy).__name__}.cuda_graph_kind={kind!r} is not a known graph kind "
                "('flow' / 'single_forward' / 'diffusion_step')"
            )
        if key not in self._graphs:
            with _CAPTURE_LOCK:
                # Another caller for this manager may have populated the key
                # while this thread waited for a different replica's capture.
                if key not in self._graphs:
                    self._graphs[key] = factory()
                    # Capture finalization is asynchronous on some PyTorch/CUDA
                    # combinations.  Finish it before another device enters its
                    # own capture; replay remains fully concurrent afterwards.
                    torch.cuda.synchronize(self.device)
        return self._graphs[key]
