# 0008 — Asynchronous inference: RTC and VLASH

- Status: Draft
- Author: EmbodiInfer contributors
- Date: 2026-09-30

## 1. Summary

Add an engine-layer feature domain, `embodiinfer/engine/async_inference/`, implementing the two
published asynchronous-inference algorithms for action-chunking VLA policies: **Real-Time
Chunking (RTC)** and **VLASH**. Both address the same physical fact — in an asynchronous loop the
chunk a policy computes is not the chunk the robot is executing, because the robot keeps moving
during the forward pass. RTC reconciles the *actions* (hold the committed prefix, guide the
remainder toward it); VLASH reconciles the *state* (roll the robot state forward under the
actions already issued and condition on the execution-time state). The expected gain is that a
deployment can run inference concurrently with execution — removing action stalls and reducing
reaction latency — without the instability that naive asynchronous switching causes.

The algorithms are opt-in per request, carried in one `metadata["async"]` block, and neither
changes the behaviour of an existing synchronous request.

## 2. Motivation and current gap

The engine already separates prefill from a reusable denoise loop
(`embodiinfer/engine/core.py:302` `execute` → `_prefill` → `_integrate`), and already aggregates
ready requests into batches (`embodiinfer/engine/async_engine.py`). What does not exist is any
notion of *what the robot is doing while the model runs*:

- `AsyncEngine` batches *independent* requests that happened to arrive together. It is
  request-level batching, not concurrent inference-with-execution: the engine never learns that a
  request is a re-plan for an episode that is still executing the previous chunk.
- `FlowDecoder.integrate` (`embodiinfer/policies/decoder.py:256`) integrates the flow field from
  the observation-time state with no knowledge of a previously issued chunk. Under asynchronous
  execution that chunk's unexecuted prefix is a *commitment* the new chunk contradicts.
- `RawPolicyRequest` (`embodiinfer/engine/serve/contracts.py:44`) carries an open
  `metadata: Mapping[str, Any]`, and `PolicyService.step`
  (`embodiinfer/engine/serve/service.py:99`) already threads it through unchanged — but nothing
  reads it, so there is no way for a deployment to say "this request is 3 control steps late".
- `docs/proposals/0002-gr00t-adapter.md` lists "RTC (real-time chunking) inpainting" as a
  follow-up extension point, i.e. the gap is already acknowledged in the tree.

The gap is therefore an *interface* gap as much as an algorithmic one: the timing knowledge that
RTC and VLASH need (how many control steps of latency elapsed, which actions are still committed)
lives on the execution side, in a different process, behind the versioned HTTP/WirelessComm API.

## 3. Goals and non-goals

Goals:

- Implement RTC prefix guidance faithfully enough to be compared against the reference, with a
  true Jacobian-vector product correction (see §6), for every flow policy the engine supports
  (pi0.5, GR00T, LingBot-VLA) and both flow time directions.
- Implement VLASH state roll-forward for both action-space semantics (absolute and delta), plus
  the server-side application of it to a request's `state`.
- Define a versioned, validated wire contract for both, inside the existing step `metadata`.
- Wire both into the real execution paths: RTC through `ActionDecoder.produce_chunk` /
  `EngineCore.execute`, VLASH through `PolicyService.step` (model-agnostic) and the pi0.5 serving
  adapter (which owns the model-space prefix).
- Pin both with tests, including cross-repository golden vectors shared with EmbodiRun.

Non-goals:

- **VLASH's temporal-offset fine-tuning and its shared-observation attention packing.** Those are
  training-time changes and belong to a trainer, not to an inference engine. This proposal
  implements the *inference* half only; a deployment that skips the fine-tuning gets VLASH's
  mechanism but not its full published accuracy, and the docs must say so.
- **Action quantization.** Grouping micro-actions into macro-actions changes what the robot
  executes and is owned by the execution runtime (EmbodiRun), not by the inference engine.
- **The asynchronous execution loop itself** (launching inference while executing, measuring the
  real delay). That is EmbodiRun's side of the same feature; this proposal only defines the
  interface it speaks.
- Automatic delay estimation inside the engine. The engine is told the delay; it does not measure
  it.
- Any change to `AsyncEngine`'s scheduling. RTC/VLASH compose with it but do not require it.

## 4. Design

Two layers, because the two algorithms attach at different points in the request lifecycle.

### 4.1 Engine layer: `embodiinfer/engine/async_inference/`

- `rtc.py` — the numeric core: `PrefixAttentionSchedule`, `RTCGuidanceConfig`, `prefix_weights`,
  `build_rtc_guidance`, `guidance_strength`, `rtc_guided_velocity`, plus the fixed-prefix
  helpers (`hard_prefix_mask`, `clamp_prefix`) and the direction adapters (`flow_noise_end`,
  `action_ness`).
- `vlash.py` — `ActionSpaceSemantics`, `roll_state_forward`, `VlashStatePlan`,
  `apply_vlash_state_plan`.
- `contracts.py` — `ASYNC_SCHEMA`, `RTCPlan`, `AsyncInferencePlan`, `parse_async_plan`, and the
  validation limits.
- `serving.py` — `plan_from_request`, `apply_request_async_state`, `batch_rtc_guidance`.

The module depends only on `torch` and plain mappings. It has no model, checkpoint, or robot
knowledge, which is what lets the same arithmetic serve pi0.5, GR00T and LingBot-VLA, and lets
EmbodiRun mirror it in a different language of tensors.

### 4.2 RTC: where the conditioning enters the decode

`RTC` needs per-step intervention inside the denoising loop, so the capability is declared on the
decoder rather than branched on in the engine:

- `ActionDecoder.supports_rtc_guidance` (default `False`).
- `ActionDecoder.produce_chunk(..., *, guidance=None)`; a decoder that does not declare support
  calls `ActionDecoder.reject_rtc_guidance` and raises `UnsupportedAsyncGuidanceError`. It must
  not silently decode without the correction, because the resulting chunk would contradict the
  actions already committed to the robot.
- `FlowDecoder.integrate(..., *, guidance=None)` implements two conditioning modes: soft
  guidance (the paper's method) and hard fixed-prefix clamping. When guidance is active it runs
  eagerly and ignores `graphs` — the soft correction must be differentiated through the velocity
  field, which a captured graph cannot express, and the hard variant mutates the state between
  steps. `Pi05FlowDecoder.integrate` forwards the argument and falls back to the generic eager
  loop, because the native cached-schedule path is also not differentiable.
- `EngineCore.execute(..., *, rtc_guidance=None)` threads it through `_Staged` and `_integrate`.
  When guidance is present the batch is *not* padded to a graph bucket (nothing will replay), so
  one conditioning row maps to one real request.
- `EngineCore.execute_pipelined` runs conditioned batches eagerly and refuses to overlap them,
  because the MemPool isolation that keeps a concurrent prefill from aliasing a graph replay does
  not apply to an eager loop.

### 4.3 RTC: where the committed prefix comes from

The prefix that must be held is the previous chunk in **model space** (normalized, padded to
`max_action_dim`), because it is compared against the flow state inside the denoiser. This is
exactly why the reference keeps a separate `original_queue` beside the post-processed `queue`.
A client only ever sees post-processed actions, so the server is the only participant that can
hold the model-space prefix.

Two sources, in precedence order:

1. An explicit `prev_chunk_left_over` in the request. This is the escape hatch for a client that
   legitimately holds model-space actions (a co-located trainer, or a test pinning a known
   prefix).
2. The pi0.5 serving adapter's per-session cache of the last issued chunk
   (`Pi05ServingAdapter._last_model_chunk`, `_resolve_rtc_plan`). The request then needs to carry
   only `inference_delay`, and the prefix is `cached[delay:]` — the reference's
   `ActionQueue.get_left_over` after `delay` consumptions. `reset` drops the entry so a new
   episode is never held to the previous one's chunk.

Absence is never an error: on the first inference of an episode there is no committed prefix and
conditioning is a no-op, which is also the reference's behaviour.

### 4.4 VLASH: where the rolled state is applied

The roll is arithmetic on the request's own `state` mapping plus the pending actions, so it is
applied once in `PolicyService.step` before any adapter sees the request — and every policy in
the catalog inherits it, including non-flow ones. The client supplies *ingredients*, not a
verdict: `state_fields` (with the action-vector column order), `pending_actions`, `delay`, and
`action_space`. The server performs the roll, so both sides run the same arithmetic instead of
one trusting the other's precomputed override.

### 4.5 Wire contract

`embodiinfer.async.v1`, inside the existing step `metadata`:

```json
{
  "async": {
    "schema": "embodiinfer.async.v1",
    "rtc": {
      "prev_chunk_left_over": [[0.1, 0.2]],
      "inference_delay": 3,
      "execution_horizon": 8,
      "prefix_attention_schedule": "linear",
      "max_guidance_weight": 10.0,
      "hard_prefix": false
    },
    "vlash": {
      "state_fields": ["joint_0", "joint_1"],
      "pending_actions": [[0.0, 0.1], [0.0, 0.1]],
      "delay": 2,
      "action_space": "absolute"
    }
  }
}
```

Carrying it in `metadata` rather than widening the step schema is deliberate: the step schema is
frozen, every existing client keeps working byte-for-byte, and a malformed block is reported as
`invalid_observation` instead of degrading silently to synchronous behaviour — a silent downgrade
under an async deployment would be a control hazard, not a convenience.

### 4.6 Alternatives considered

**A separate `RTCDecoder` wrapper.** Rejected: the conditioning is per-*request timing*, not
per-decoder, so a wrapper would still need a channel to receive the current delay and prefix; a
declared capability plus one optional keyword is smaller and keeps the failure mode explicit.

**Client-side prefix cache (return raw model-space chunks to the client).** Rejected: it would
push model-space, checkpoint-specific tensors across the HTTP boundary and make every client
responsible for a model's normalization and padding. It also contradicts EmbodiRun's rule that
it owns no model semantics.

**Compute the VLASH roll on the client only and send the resulting state.** Rejected as the
*only* mechanism: it would make correctness unverifiable across the boundary (the server can only
trust the number). Keeping the ingredients server-side means the same golden vectors can be
asserted on both sides. The explicit `prev_chunk_left_over` / raw-ingredient form keeps the door
open for a client that wants to send a precomputed value.

**Make RTC guidance a graph-replayable kernel.** Rejected for now: the correction is a
Jacobian-vector product through the velocity field, so a captured replay would have to capture
the backward pass. That is a real optimization opportunity, but it is not needed to establish
correctness and it would couple this change to graph-capture internals.

## 5. Model-agnosticism verdict

Engine layer. The new subpackage depends only on `torch`, on plain number mappings, and on one
public policy contract (`FlowVLAPolicy.denoise_step`, used as a callable). It contains no model
name, no checkpoint layout, and no private policy field.

Two places do consult a *declaration*, which is the sanctioned pattern (cf. `cuda_graph_kind`):

- `ActionDecoder.supports_rtc_guidance` selects whether guidance may be applied.
- `FlowVLAPolicy.flow_schedule`'s `dt` sign supplies the flow direction, via `flow_noise_end`.
  Hardcoding the pi0.5 direction would have been a model assumption; deriving it is the same rule
  `models/schedulers/flow.sde_coefficients` already uses.

The only model-specific code is the pi0.5 adapter's committed-prefix cache, which lives in
`policies/pi05/serving.py` where pi0.5 assumptions belong.

## 6. Losslessness and precision criterion

**The synchronous path is bit-exact.** With `rtc_guidance=None` and no `async` block, no new
tensor operation, no dtype promotion, and no shape change occurs. Tests assert
`torch.equal` against the pre-change engine output for the same seed, and the pi0.5 adapter calls
`self._core.execute(batched)` with no extra keyword when no plan requests RTC.

**RTC is not a lossless optimization of the synchronous path; it is a different computation**, so
the criterion is equality with an independent closed form rather than with the unguided decode:

- *Reference*: Physical Intelligence `real-time-chunking-kinetix` `src/model.py` `realtime_action`
  (JAX), whose guidance is a `jax.vjp` through the denoiser.
  Reference path for the constants: the same expressions in the LeRobot port
  (`policies/rtc/modeling_rtc.py`).
- *Compared quantity*: one guided velocity for an affine velocity field `v(x) = Wx + b`, whose
  Jacobian-vector product is analytic.
- *Threshold*: `atol=1e-5`, float32.
- *Reproduction*: `tests/test_async_rtc_vlash.py::test_guided_velocity_matches_closed_form_true_vjp`.
- *Known difference*: the LeRobot PyTorch port evaluates `v_t` *before* marking `x_t` as requiring
  grad, which collapses `d x1/d x` to the identity and degrades the correction to the residual.
  This implementation follows the JAX original. The divergence is pinned by
  `test_guided_velocity_differs_from_residual_only_correction` so it cannot regress unnoticed.

**RTC prefix weights** are compared bit-for-bit (`torch.equal`) against the reference table,
including the documented example (`start=2, end=6, total=10 → 1 1 4/5 3/5 2/5 1/5 0 0 0 0`) and
the degenerate `end <= start` case.

**VLASH roll-forward** is exact arithmetic; it is compared to golden vectors at `atol=1e-6`
against float32 values, and to the upstream behaviour that the future state at the end of the
executing chunk equals that chunk's last action for absolute semantics.

**Cross-repository agreement.** `tests/fixtures/async_inference_golden.json` is committed
byte-identically to EmbodiInfer and EmbodiRun; each repository's test suite asserts the same
tables with its own implementation and tensor library. A drift on either side of the HTTP
boundary fails a test on that side.

## 7. Implementation plan

New: `embodiinfer/engine/async_inference/{__init__,rtc,vlash,contracts,serving}.py`.
Changed:

- `embodiinfer/policies/decoder.py` — `supports_rtc_guidance`, `reject_rtc_guidance`, guidance
  plumbing in `produce_chunk`, `FlowDecoder.integrate`/`_integrate_rtc`, rejection in
  `AutoregressiveDecoder` and `ParallelDecoder`.
- `embodiinfer/policies/pi05/runtime.py` — `Pi05FlowDecoder.integrate` forwards guidance and
  falls back to the eager loop.
- `embodiinfer/policies/cosmos/modeling_cosmos.py` — rejects guidance explicitly.
- `embodiinfer/engine/core.py` — `rtc_guidance` through `_Staged`, `_prefill`, `_integrate`,
  `execute`, `_pipeline_step`, `execute_pipelined`.
- `embodiinfer/engine/serve/service.py` — apply the VLASH state plan in `step`.
- `embodiinfer/policies/pi05/serving.py` — resolve the committed prefix, build batch guidance,
  cache the issued model-space chunk, declare the capability, drop the cache on `reset`.
- `embodiinfer/exceptions.py` — `UnsupportedAsyncGuidanceError`.

Backward compatibility: every new parameter is keyword-only and defaults to the current
behaviour; the wire schema is untouched.

## 8. Test plan

`tests/test_async_rtc_vlash.py` (CPU, mock policy, no checkpoint):

- RTC: reference weight table (golden), schedule edge cases, `start`/`end` precedence, guidance
  coefficient limits and symmetry, padding/clamping invariants, the closed-form VJP check for
  both flow directions, dtype preservation, no-prefix no-op, shape-mismatch failure.
- Decoder capability: flow declares support; a non-flow decoder rejects guidance; an absent
  prefix is not guidance and must not raise.
- Engine: `rtc_guidance=None` is bit-identical to the pre-change path; a hand-rolled guided loop
  reproduces the engine's output; guidance changes the chunk; hard prefix pins the committed rows
  exactly; recurrent policies reject guidance; `execute_pipelined` stays sequential under
  guidance.
- VLASH: absolute/delta semantics, delay 0 / full buffer, batched and rank-preserving, golden
  vectors, every validation error, dtype preservation, plan application rewriting only named
  fields, plan validation.
- Contracts: absent block, both halves parsed, schema/`NaN`/shape/emptiness rejections,
  `apply_request_async_state` no-op and roll-forward, batch guidance stacking, zero-fill and
  mixed-delay rejection.

CUDA/checkpoint coverage is deliberately out of scope for this change: RTC's eager path and the
native pi0.5 path differ only in which velocity callable is used, which the CPU tests already
exercise on a mock. A real-weight pi0.5 run is a follow-up once the checkpoint environment is
available, and must be reported with the checkpoint revision if it is claimed.

## 9. Benchmark plan

Not a performance proposal, but two measurements are needed before anyone may claim a win, and
neither may be generalized beyond its conditions:

- **Overhead of conditioned decoding.** Same checkpoint, dtype, batch size, warmup and step
  count; measured with `--num-steps` fixed; report eager-unguided vs eager-soft-guided vs
  hard-prefix on the same GPU. Expectation: soft guidance costs one extra backward pass per step
  and is therefore roughly 2–3× the unguided decode; hard prefix should be close to unguided.
  The gain disappears at high batch size where decode is compute-bound and at large `num_steps`
  where the per-step overhead amortizes differently.
- **End-to-end effect on reaction latency.** This requires EmbodiRun's execution side and a
  closed loop; it is an EmbodiRun benchmark, not an EmbodiInfer one.

Report hardware, dtype, batch size, warmup, measured iterations, checkpoint and revision.

## 10. Risks and limitations

- **Soft guidance is slow by construction.** It differentiates through the velocity field every
  step. A deployment that cannot afford it should use the hard-prefix mode, which is a different
  (weaker) approximation — the docs must not present them as interchangeable.
- **VLASH without offset fine-tuning is incomplete.** The published accuracy assumes the model
  was fine-tuned to use the state input. Feeding a future state to a checkpoint that under-uses
  proprioception may help little; the reference found that models largely rely on vision. This
  proposal does not measure that, and must not claim VLASH's published numbers.
- **The pi0.5 prefix cache is per-session process state.** It is bounded by the session limit and
  dropped on `reset`, but a session that is never reset or closed holds one chunk tensor. A
  multi-process or multi-replica deployment must keep a session affine to one replica, which is
  already an invariant of the data-parallel engine.
- **Mixed delays within one batch are rejected, not merged.** A batch shares one execution clock,
  so merging different delays would silently mis-condition a chunk. If a future caller needs it,
  it must batch by delay.
- **The LeRobot/reference divergence is unresolved upstream.** If the PyTorch reference is fixed
  to compute the true VJP, our tests already match it; if the intent was the degenerate form,
  this implementation is the one that must change. The divergence is documented and pinned so the
  choice is explicit rather than accidental.
- **Graph capture is not supported on the guided path.** A deployment that depends on whole-loop
  CUDA-graph replay to hit its latency target cannot use soft RTC; capturing the backward pass is
  future work.
