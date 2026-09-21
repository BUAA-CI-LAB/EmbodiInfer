# Architecture

EmbodiInfer separates an inference engine from model adapters through a small set of
contracts. The engine schedules batching, CUDA graphs, parallelism, rollout, and
sessions; each adapter describes how one model encodes observations and decodes
actions.

## From observations to actions

A flow-matching VLA maps an observation `o = (images, language, state)` to a future
action chunk `a ∈ R^(H×A)`. Its computation has two stages.

1. **Multimodal prefix encoding.** Images, language, and proprioceptive state are
   encoded by a vision encoder and a VLM backbone into a feature sequence
   `c ∈ R^(P×D)`. This stage is compute-bound.
2. **Denoising.** An action head starts from noise and integrates a velocity field
   `v_θ(x_t, t | c)` over `N` steps — deterministic Euler, or a stochastic SDE form —
   attending to the fixed prefix `c` at every step.

This structure offers three opportunities to reduce inference cost:

- **Shapes and control flow are static.** `N` is fixed, every step has the same tensor
  shapes, and the time schedule can be precomputed. The denoising loop is therefore an
  ideal CUDA-graph target: capture once, replay thereafter. That removes host-side
  kernel launch overhead, which is a significant share of the cost at small batch.
- **The prefix is constant within one prediction.** The observation does not change
  while the `N` steps run, so the prefix and its attention K/V are computed once and
  reused across all `N` steps, and across the candidates of a best-of-N plan.
- **Requests are driven by the environment clock.** Parallel simulators produce
  observations at nearly the same moment, which suits request-level batching. In
  asynchronous rollout, requests arrive spread out, so prefill and the previous batch's
  denoising can overlap on separate CUDA streams.

The engine mechanisms follow directly: request-level batching across environments,
CUDA-graph capture of the denoising loop, prefix K/V reuse, prefill/denoise stream
pipelining, and a weight-sync and log-probability surface for RL.

## Relationship to applications and trainers

EmbodiInfer owns model execution. Applications supply observations and consume
actions; an RL trainer additionally supplies weight updates and evaluates its
training objective.

For deployment, use the [network service](serving.md). For an embedded engine or
trainer integration, use the [Python API](api.md). The
[benchmark guide](benchmark.md) records comparisons with reference implementations
under specific environments and numerical conditions.

## Engine contracts

`VLAPolicy` is the engine-visible contract for a model: a multimodal prefix encode
(`encode_prefix`), observation collation and padding, the CUDA-graph prefix pipeline,
and a `decoder`.

`ActionDecoder` is the serving contract: `init_state` and `produce_chunk`, plus a
structured `decode` default wrapper. `RLDecoder` is the narrower on-policy capability
layer for decoders that also support generic policy-gradient rollout: behavior
sampling and differentiable recomputation of log-probability.

Decode strategies are policy objects, not engine branches. The implemented decoders
are `FlowDecoder` (pi0.5 / GR00T N1.7 / LingBot-VLA) and `ParallelDecoder`
(OpenVLA-OFT), which are `RLDecoder`s, plus `CosmosDiffusionDecoder` (planning-only
Cosmos Policy) and `AutoregressiveDecoder` (recurrent token decode, ActiveVLN), which
are plain `ActionDecoder`s.

## EngineCore

`EngineCore` is the synchronous execution path. A stateless request runs `encode_prefix`
and then the decoder's `produce_chunk`; a recurrent request runs a transaction through
`SessionStore`.

## SessionStore

`SessionStore` manages opaque recurrent memory with explicit checkout, commit,
rollback, reset, and cancel. A session is identified by `SessionKey` (`env_id`,
`episode_id`, `rollout_id`). The engine never reads the KV layout; a transaction
commits only after prefill, token append, and parse/pack all succeed, and an exception
preserves the previously committed memory.

## AsyncEngine

`AsyncEngine` aggregates ready requests within a `max_wait_ms` window up to
`max_batch_size`, and pads each batch to the nearest `batch_buckets` size so the same
CUDA graph can be replayed. Parallel environments finish their step at different times,
which is what makes the window useful. A synchronous vectorised rollout, where the whole
batch steps together, goes straight to `EngineCore.execute`.

## CUDA-graph capture

Policies declare static-shape capture through policy metadata (`supports_cuda_graph`
and `cuda_graph_kind`). The denoising loop runs `N` times with exactly the same shapes,
so the engine captures it at three levels, keyed by `(batch bucket, N)`, lazily, and
valid across weight refits because the recorded parameter pointers do not change:

- **Single step** (`DenoiseGraph`). One denoising step is captured and a Python loop
  drives `N` replays; the prefix is copied into the static buffer once per prediction.
- **Full loop** (`LoopGraph`). The whole `N`-step integration, including the in-graph
  Euler step and the precomputed time schedule, is captured once and replayed once, so
  the host cost between steps goes from `O(N)` to `O(1)`. The gain on its own is
  limited, since single-step capture already removes most launch overhead; its value is
  that it collapses the CPU cost of denoising into a single submission, which is what
  makes stream pipelining possible.
- **SDE loop** (`SdeLoopGraph`). The whole stochastic SDE integration, including
  per-step noise injection and the trajectory and velocity records that RL
  log-probability needs, is captured once. Noise is drawn outside the graph into a
  static buffer so the captured path stays bit-comparable with the eager reference.
  Per-step noise scales and score-correction coefficients live in a device-side
  coefficient buffer, so changing the selected noise step is a buffer copy rather than a
  re-capture. The time direction — noise at `t=0` or at `t=1` — follows from the sign of
  the flow schedule's step size, so one implementation serves both the pi0.5 and the
  GR00T convention.

Prefix capture is configured by each policy. For example, π0.5 has an optional
`prefix_cuda_graph` path in its native runtime.
`GraphManager` selects the decode strategy through `cuda_graph_kind`.
Compare graph and eager outputs using the same inputs, noise, dtype, and
schedule. The model benchmarks report the numerical comparison for each backend.

For PI0.5, `attention="eager"` also selects the reference projection layout: camera
views are encoded separately, and Q/K/V and gate/up projections remain separate. Fused
backends retain the batched and fused routes. This distinction matters for
rollout/actor log-probability comparisons on selectively cast checkpoints: changing GEMM
shapes can change rounding even with the same attention formula. The generic
`denoise_step` computes the current time conditioning and AdaRMS projections inside the
graph, and therefore cannot reuse a warmup result by input `data_ptr` — the same static
buffer receives different values each step. For a fixed schedule, the native runtime
passes precomputed `modulations` explicitly instead of relying on the generic path's
address-based cache.

## Prefix K/V reuse

`encode_prefix` returns a `PrefixState` holding the attention K/V of every layer, and
the decoder reuses it for all `N` denoising steps, so the compute-bound VLM backbone
runs once per prediction rather than once per step. For planning, `expand_prefix`
broadcasts the same prefix to `N` sampled candidates, so the backbone runs once for the
whole group.

Each new environment step needs a fresh prefix because its images have changed. In
trajectory-level group sampling (GRPO and similar), where each candidate is an
independent environment trajectory, the prefix is redundant only on the first frame
after a reset.

## RL rollouts and weight updates

`GenerationBackend` exposes the minimum an RL trainer needs: `generate` (deterministic,
taking the CUDA-graph fast path), `generate_with_logprob` (stochastic sampling plus
log-probability), `best_of_n`, and `refit` (an in-place weight update that keeps captured
graphs valid). `VLAPolicy.refit_state_dict()`
plus `commit_refit(version=...)` add a zero-copy two-phase protocol: the training
framework transfers the weights, applies sharding, and maps parameter names;
EmbodiInfer validates and commits the policy version. A commit runs the
policy's `on_refit` runtime refresh before publishing the new
version. If the refresh fails, the previous version stays visible, but tensors already
written are not rolled back, so the caller must repair or discard that policy and retry.

Log-probability uses the SDE form: the deterministic ODE sampler is rewritten as a
stochastic process

```text
x_{k+1} = μ_k + σ(t_k)·sqrt(|Δt|)·ε_k

log p += log N( x_{k+1}; μ_k, σ(t_k)²·|Δt|·I )
```

where the transition mean `μ_k = x_k + v_θ·Δt − c_k·(x_k + γ_k·v_θ)` carries the
score-correction term, and `σ(t)` is a configurable noise schedule (`σ = 0` makes the
step exactly the deterministic ODE step). The coefficients `γ_k` and `c_k` follow from
the flow schedule's time direction and are matched op by op against the target RL
framework's native generator, including dtype promotion order — so numerical comparison must include both rollout and actor recomputation. A differentiable `flow_logprob_recompute` is
also provided; it re-scores a stored trajectory under the current parameters with a
usable gradient, for a trainer-side PPO/GRPO loss.

## Prefill/denoise stream pipelining

Denoising (CUDA-graph replay, almost no CPU) and the next batch's prefill (eager) can
overlap on two CUDA streams (`execute_pipelined`). Concurrent prefill allocates memory
through `torch.cuda.MemPool`; otherwise the caching allocator races across streams and
breaks losslessness. The mechanism only helps asynchronous rollout or serving, where
requests arrive spread out. Splitting a synchronous rollout batch into a pipeline was
measured to be a net loss, because denoising is insensitive to batch size and splitting
inflates the total denoising time. It is off by default.

## Package layout

```text
embodiinfer/policies          VLAPolicy + VLAPolicyConfig + ActionDecoder/RLDecoder + model adapters
embodiinfer/models           engine-agnostic nets by family — video_dit / video_vae /
                       text_encoders / schedulers (diffusion + flow math)
embodiinfer/layers            operator contracts + registry + backend routing
embodiinfer/backend           concrete Torch/Triton implementations, capability probes,
                       warmup, graph-safe execution
embodiinfer/engine            EngineConfig · EngineCore · transactional SessionStore ·
                       AsyncEngine · graph
embodiinfer/engine/parallel   data-parallel replicas/dispatch + tensor-parallel
                       primitives/sharding plans
embodiinfer/engine/rollout    GenerationBackend · flow/categorical log-probability ·
                       weight sync (rollout/demo/ = toy trainer + env)
embodiinfer/engine/serve      EmbodiInfer API + HTTP, WirelessComm, and WebSocket frontends
```

Tooling lives at the top level: `benchmarks/`, `examples/`, `scripts/`, `tests/`.

`embodiinfer/models/` holds reusable, engine-agnostic components. They must not import
`policies` or `engine`, and they do not know about `Observation` or `ActionChunk`; they
are organised by model family and composed by a policy, the way a Diffusers pipeline
composes its components. A network welded to one policy's forward — pi0.5's Gemma,
OpenVLA-OFT's Llama, and so on — is not promoted there and stays in its
`policies/<name>/`.

### Layer rules

| Layer | Directory | Owns | Constraint |
|---|---|---|---|
| policy | `embodiinfer/policies/<name>/` | checkpoint loading, the forward, collate/pad, `flow_schedule` | model-specific logic stays here; implements the base contracts |
| engine | `embodiinfer/engine/` | scheduling, execution, CUDA graphs | depends only on the public policy contracts (`VLAPolicy`, `PrefixState`, `flow_schedule`, `encode_prefix`, `denoise_step`, `supports_cuda_graph`, `cuda_graph_kind`, `allocate_static_prefix`, `copy_prefix_into`) |
| operator routing | `embodiinfer/layers/` | attention and other Protocols, registry, backend selection | no concrete operator implementations; callers depend on protocols and registered names |
| compute backend | `embodiinfer/backend/` | Torch/Triton implementations, capability probes, warmup | must not depend on models, policies, engine, or environment semantics |
| rollout | `embodiinfer/engine/rollout/` | RL rollout surface, log-probability, weight sync | depends on the engine, never on a concrete policy |
| serve | `embodiinfer/engine/serve/` | model-neutral inference API and frontends | communication and engine calls only; simulator and robot protocols belong to the deployment runtime |

## Deployment interface

EmbodiRun submits images, state, an instruction, and a session ID over HTTP or
WirelessComm. EmbodiInfer returns an action chunk, timing, and the policy
revision. EmbodiRun then maps those actions to device commands and manages the
control loop. See [Serving](serving.md) for the network contract.

## Adding a model

To add a model, implement its policy and declare the supported capabilities:

1. Implement `VLAPolicy.encode_prefix`.
2. Implement the serving `ActionDecoder`.
3. Inherit `RLDecoder` only when the generic policy-gradient contract is met.
4. Declare `supports_cuda_graph` and `cuda_graph_kind` only when static shapes are
   capturable.
5. Declare `is_recurrent` when state crosses calls, and let the engine's `SessionStore`
   manage the transaction lifecycle.

See [Contributing](contributing.md) for the proposal, parity, and documentation flow a
new adapter follows.

## Numerical discipline

VLA models are precision-sensitive. An optimisation is therefore lossless by default and
must come with a reproducible criterion.

- **bit-exact** — `max_i ||Δa_i||_inf = 0`. Applies to rewrites with the same
  mathematics and the same floating-point accumulation order, such as full-loop graph
  capture against step-by-step eager execution, or a static buffer against a freshly
  allocated tensor each step. This is the default target for engine-layer changes.
- **numerically equivalent** — `max_i ||Δa_i||_inf <= ε`, typically `ε ~ 1e-6`, with the
  difference attributable to floating-point reordering such as a changed reduction
  order. Using this level requires stating the source and why it is acceptable.
- **Same precision conditions** — any comparison must use the same dtype, the same
  `torch.set_float32_matmul_precision` setting, and the same attention semantics.
  Changing the precision conditions is not a lossless optimisation and is a separate
  change.

RL integration adds a criterion of its own: rollout-side log-probability is re-scored by
the actor, and the importance-ratio quantiles are compared against the native backend's
own noise floor at `θ₀`.

Measured results, including the engine-mechanism and RL-integration numbers, are on the
[Benchmark](benchmark.md) page.

## Limitations

- Cosmos Policy supports best-of-N planning and per-step graph capture. Generic
  RL rollout, whole-loop capture, and tree-search planning are not implemented.
- Log-probability is an SDE surrogate. Its deviation from the exact marginal likelihood
  is not quantified; its usefulness in PPO rests on the ratio-at-`θ₀` criterion and on
  end-to-end training health.
- The NCCL cross-process path for weight sync is a sketch. The RLinf integration uses
  the framework's own synchronisation, which updates weights in place and stays
  compatible with captured graphs.
- CUDA graphs require static shapes: batching is bucketed, and instruction length is
  padded to a fixed value.
- ActiveVLN uses eager execution, `B=1`, and explicit sessions. It has no
  `RLDecoder` implementation; real-checkpoint GPU parity validation is pending.
- Prefix capture, camera batching, and attention implementations depend on the
  policy and selected backend. Consult the model's benchmark configuration before
  applying an optimization or reusing a numerical result.

## References

- pi_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models.
  arXiv:2510.25889.
- RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of
  Vision-Language-Action Models. arXiv:2510.06710.
- openpi remote inference: `github.com/Physical-Intelligence/openpi`.
- Isaac-GR00T: `github.com/NVIDIA/Isaac-GR00T`.
