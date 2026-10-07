# 0008 — Reproducible kernel tuning with Humanize2

- Status: Accepted
- Author: EmbodiInfer maintainers
- Date: 2026-10-02

Implementation: repository CLI, Humanize2 Flow, evaluator, source/evidence archive,
and isolated tool lock are present. CPU/fake-agent coverage and NVIDIA RTX 5090
Triton/CUDA C++ builder tests are verified. Thor compilation and performance
still require validation on that target machine.

For the Chinese workflow guide, required task inputs, and stopping conditions,
see [Section 11](#11-自动化调优使用说明).

## 1. Summary

Add repository developer tooling for a serial, agent-driven kernel optimization
loop. Humanize2 (`hmz`) owns coding-agent execution, budgets, traces, and resume;
EmbodiInfer supplies the task contract, trusted evaluation, promotion rules, and
saved operator artifacts. Formal measurement belongs under `benchmarks/` and
orchestration under `scripts/`. This is model-neutral tooling, not an engine API.

## 2. Motivation and current gap

`embodiinfer/backend/triton/norm.py:302` and
`embodiinfer/backend/triton/split_kv_attention.py:179` contain concrete kernels and
hand-chosen launch configurations. `embodiinfer/layers/attention.py:78` provides
runtime registration, but the repository has no general operator task format,
automated search workflow, or durable archive of verified candidates. A task can
describe one of these kernels or an independent operator without importing model
or engine internals into the tuning framework.

## 3. Goals and non-goals

- Accept NVlabs KDA task packages and Triton or CUDA C++ multi-file solutions.
- Run on one target machine, with an explicitly selected coding agent and model.
- Freeze precision, workload, timing, and acceptance conditions before search.
- Save every attempt and export the best verified source with its evidence.
- Support bounded execution, interruptions, and explicit resume.

Runtime registration, deployment, SSH, model-level performance claims, and
automatic installation of GPU dependencies are outside this change. Hardware
neutrality describes orchestration: a backend must support the selected hardware;
CUDA and the default FlashInfer adapter require a compatible NVIDIA environment.

## 4. Design

Use the [NVlabs wishlist task template](https://github.com/NVlabs/kda/tree/2186cd90dfb3b11f1ec250b4e3f321d72d7e7ef0/example)
and [minimal flow](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/docs/agent-flow.md).
Tasks contain `README.md`, `definition.json`, `workloads.jsonl`, `baseline.py`,
and optionally `benchmark.py`. The definition embeds the mathematical reference
`run`; `baseline.py::run` supplies the initial performance baseline. The JSON
schemas remain FlashInfer Trace schemas. Project settings live in `tuning.yaml`.

Implement a local Humanize2 Flow against
[`hmz` revision b70d4427](https://github.com/humanfia/humanize/tree/b70d4427f11308442aef20f9b2233dfbbc34a39e).
The Flow asks an existing coding agent to inspect the task, write a hypothesis
and executable plan, produce a candidate, then respond to evaluator feedback.
Agent turns perform source/JSON checks and return the candidate; GPU compilation,
numerical verification, and timing run only in the separate evaluator. A working
GPU in the evaluator does not imply that the agent's command environment grants
GPU or compiler-cache access. Plans describe validation criteria without trying
to execute that validation inside an agent turn.
It delegates all agent sessions and cancellation to Humanize2. Domain helpers
validate tasks, launch a bounded evaluator process, persist candidate evidence,
and atomically publish the best pointer. No independent coding-agent subprocess
implementation is introduced.

The default evaluator reuses FlashInfer Bench 0.1.2 definitions, workload
materialization, and solution building. Additional gates enforce the configured
precision contract, workload completeness, graph replay correctness when selected,
and paired baseline/candidate measurements. A task-owned `benchmark.py` can
implement the documented structured evaluation interface for other hardware or
special input semantics. Exit status or printed PASS alone cannot promote a kernel.

Each task fixes eager or CUDA Graph timing. Promotion requires the weighted mean
latency to fall by at least 3% and every workload to regress by at most 5%, against
the incumbent and initial baseline. Weights default to equal. Repeat paired
measurements before promotion. Stop at 20 attempts, two hours of active execution,
or five consecutive failures/non-improvements; all limits are configurable.

Every timed sample starts from a cold L2 (a 256 MiB buffer is zeroed) with
freshly copied inputs. CUDA Graph samples queue the flush and the input copy
ahead of the start event and do not synchronize per sample, so the events bracket
only the replay's device execution. FlashInfer's `do_bench` synchronizes first;
its start event then precedes the host launch, adding several microseconds of
launch latency and scheduler jitter to every sample. On an RTX 5090 that
inflated microsecond kernels to about 10 µs and let three identical solutions
differ by up to 53%, far above the 3% promotion threshold. Without it, and with
1000 iterations (the generated-task default), identical solutions differed by at
most 4.5% and mostly under 2%. A single trivial kernel replays in about 4.1 µs
under this protocol, the device's floor; tasks whose baseline is already there
leave an agent nothing to win, so preflight prints every baseline latency.
Eager timing still uses FlashInfer's `time_runnable`, including launch cost.

Runs snapshot task files and record source digests, evaluator identity, hardware
and software fingerprints, precision settings, agent settings, attempts, raw
measurements, and decisions. Resume requires matching task/tool/environment
identity. Interrupted attempts remain in the archive and consume an attempt.
Export reconstructs source files plus the official Solution JSON and evidence.
Task and artifact hashes detect accidental modifications; they are not an OS
security sandbox for coding agents or kernels running as the same user.

Humanize2 needs Python 3.12+. Install it in a separate tooling environment; the
evaluator interpreter points to the prepared target-machine runtime (for example
Thor). EmbodiInfer's Python 3.10 floor and model dependency groups stay intact.

An independent lightweight controller was considered and rejected because the
approved design explicitly reuses Humanize2. Runtime auto-selection was rejected
because source review and separate integration are required before deployment.

## 5. Model-agnosticism verdict

This is developer automation and formal measurement, outside the inference
package. No engine, policy, or layer interface changes. Operator semantics belong
entirely to each task's definition and workloads. Backend and hardware details
belong to the evaluator and generated solution, not the orchestration loop.

## 6. Losslessness and precision criterion

### Calibrated numerical contracts for discovered operators

Model discovery records execution settings separately from acceptance rules.
Pure copies, indexing, exact elementwise operations, RNG transitions, and unknown
semantics retain bit-exact validation. An explicit registry selects floating
matrix multiplication, linear, reductions, softmax, and supported attention/norm
calls for tolerance validation; dtype alone never relaxes an unknown operator.
Existing hand-written/catalog contracts and archived runs keep their contracts.

Before task generation, `calibrate --captured CAPTURE --output CALIBRATED` runs
in the matching model environment, using only the captured production operator
and a trusted FP64 mathematical reference. Each shape/layout/argument signature
gets separate output bounds for real recorded inputs, seeded random inputs,
zeros, and alternating-sign cancellation inputs. Attention masks, scale, causal
and GQA semantics are preserved. Real-input calibration requires complete input
fixtures; capture must fail explicitly if its fixture budget cannot hold them.
No candidate source is accepted by the calibration API.

For each floating output, let B be the production baseline, R the FP64 reference,
and u the output dtype's epsilon. Fix rtol=u and
atol=max(u*RMS(R), 2*max(max(abs(B-R)-u*abs(R), 0)), dtype.tiny).
Every candidate element must satisfy abs(C-R) <= atol+rtol*abs(R). The one-epsilon
scale floor allows rounding near zero; the factor of two budgets measured
baseline error. These are explicit engineering policy constants, not a proof of
an error bound for unseen inputs. Non-floating outputs, layout, aliases, mutation
and RNG constraints remain exact. Nonfinite reference/baseline/output values
fail calibration/validation. FP64 outputs retain exact validation: there is no
higher-precision oracle in this implementation.

The immutable calibrated capture records reference identity, policy version,
seeds, profiles, dtype, layouts, scalar arguments, numerical flags, environment,
observed baseline error and each derived threshold. Generated tasks copy this
evidence into `numerics.json`; task/run hashes and the all-task preflight bind it
before any agent starts. Evaluation uses those thresholds without recalibrating.
Missing coverage, changed settings/seeds, edited evidence, or a different target
environment require a new calibration/batch. Calibration failure never widens a
threshold in response to a candidate failure.

Implementation belongs to `scripts/kernel_tuning/discovery/numerics.py` (static
policy/contract), `benchmarks/kernel_tuning/calibration.py` (trusted reference and
measurement), discovery recorder/task generation, and the replay evaluator.
There are no engine, policy, backend, dependency-pin or model-level action-parity
changes. CPU tests exercise selection, calibration, per-workload bounds, attention
semantics, precision restoration, real/random/boundary coverage, and tampering;
CUDA/real-checkpoint validation remains a separate target-machine requirement.

A single dtype-wide 1e-2 threshold was rejected because shape and accumulation
behavior matter. Online recalibration during candidate evaluation was rejected
because acceptance must be frozen. Full fixtures and FP64 references increase
preparation storage/time, and unsupported numerical references stay exact until
an explicit adapter is added. Existing incomplete captures must be recaptured;
an old task/run is never silently migrated to a looser contract.

Default correctness compares output shapes, dtypes, and exact tensor bytes against
`definition.json`'s reference on identical seeded inputs. A task may explicitly
declare fixed absolute/relative tolerance with a reason. The agent cannot change
that contract, workloads, dtypes, or float32 matmul precision. Graph tasks must
also replay with changed inputs and compare against fresh reference outputs.
The evaluator command and full numerical contract are saved with every run.

## 7. Implementation plan

Add `scripts/kernel_tuning/` for task validation, artifact storage, Humanize2 Flow,
and the `python -m scripts.kernel_tuning` developer entry point. Add
`benchmarks/kernel_tuning/` for measurement and a self-contained task example.
Document isolated setup in `CONTRIBUTING.md`; keep the optional tool's dependency
metadata separate from runtime requirements. Results default to the already
ignored `results/kernel_tuning/`. Nothing automatically registers saved kernels.

## 8. Test plan

CPU tests use fake coding agents and evaluators to cover actual Flow execution,
resume, failed attempts, budgets, task tampering, metric completeness, precision
contracts, regressions, atomic best publication, and export. Static checks cover
all new Python. GPU evaluator tests are explicitly marked and skip without their
optional dependencies and hardware. CPU validation does not launch a real coding
agent or tune kernels. Real agent searches and GPU validation run explicitly on
a prepared target machine, with their own saved task and measurement conditions.

## 9. Benchmark plan

After deployment to a prepared GPU machine, each task records hardware, software,
dtype, shapes, seeds, warmup, iterations, trials, timing mode, and baseline source.
Paired trials evaluate the full declared workload set. Record failed and neutral
attempts as well as improvements. Kernel-level results do not establish model
latency or action parity; those require the existing model benchmark separately.

## 10. Risks and limitations

Humanize2 and FlashInfer are evolving dependencies; use pinned versions and fail
explicitly on incompatible contracts. Arbitrary operators may need custom input
generation or timing through `benchmark.py`. Input coverage bounds correctness;
finite tests cannot prove an arbitrary generated kernel correct. Thermal state,
other GPU users, and clock changes can affect timing; paired remeasurement reduces
but does not remove this uncertainty. User-selected coding agents may execute
commands without interactive approval; run in a disposable workspace/account
appropriate to the task. Actual Thor compilation and performance remain target
machine validation work. The pinned Humanize2 journal uses POSIX primitives: real
searches run on the Linux target machine; local Windows checks and export are
supported. CPU tests on Windows adapt only upstream atomic journal file writing.

### Task-owned benchmark interface

`benchmark.py` is trusted task code, hashed and snapshotted before search. It
exports `create_adapter(task)`, where `task` is the import-safe `TaskPackage` from
`scripts.kernel_tuning.contracts`. The returned object implements:

```python
def environment(self) -> dict:
    # Actual device, compiler/runtime/library versions, and numerical switches.
    # Fields under identity must remain stable on resume.
    return {"identity": {...}, "conditions": {...}}

def evaluate(self, solutions: dict) -> dict:
    # Maps baseline/incumbent/candidate to official Solution objects.
    # Check all three against definition.reference using identical inputs.
    # Alternate their timing order within each paired round.
    return {"checks": [...], "rounds": [...]}

# Optional, only called when tuning.yaml enables profile; never promotion timing.
def profile(self, solutions: dict, output_dir: Path) -> dict:
    ...
```

The controller requires one check per `(workload UUID, role)`:

```json
{
  "workload": "a-declared-uuid",
  "role": "candidate",
  "passed": true,
  "seeds": [0, 1, 2],
  "precision": {
    "mode": "bit_exact", "atol": 0.0, "rtol": 0.0, "reason": "",
    "matmul_precision": "highest", "allow_tf32": false
  },
  "max_abs_error": 0.0,
  "max_rel_error": 0.0,
  "graph_replay": false
}
```

`seeds` and the complete `precision` object must equal the task settings (use
`dataclasses.asdict(task.settings.precision)`). Every check must pass; bit-exact
errors must be zero. For `cuda_graph`, `graph_replay` must be true **after checking
changed valid input tensors and fresh reference outputs**. A tolerance task uses
`abs(actual-reference) <= atol + rtol*abs(reference)` for every floating element,
exact integer/bool outputs, and finite values. The default adapter rejects input
mutation and incorrect output shape/dtype/device. Entrypoints must launch on the
caller's current stream or join any helper-stream work before returning, while
preserving its device, stream, and numerical settings. Input mutation, constrained
random values, non-contiguous inputs, statistical operators, fixed graph fixtures
requiring domain-specific perturbation, and other accelerators use the custom
adapter. It must implement the same frozen measurement contract.

Each element of `rounds` maps **every** UUID to
`{"baseline": [ms, ...], "incumbent": [ms, ...], "candidate": [ms, ...]}`.
List lengths must equal `timing.trials`; the number of rounds must equal
`timing.paired_rounds` (at least two). All timings must be finite and positive.
Within each round the gate averages trials per workload, then computes
`sum(weight * workload_latency) / sum(weight)` for each role. Candidate weighted
latency must improve by the configured fraction against both comparison roles,
and each workload must meet both regression caps. Every round must pass.

The worker attaches a nonce, request digest, task/source digests, environment
before/after, and exit status. Controller-side validation rejects mismatches and
incomplete evidence. A custom adapter remains responsible for truthful measurements
and numerical checks: the controller cannot infer GPU correctness from a boolean.
The default adapter supplies these checks for deterministic pure tensor operators.

### Archive and reproduction

`run` creates a unique directory under `results/kernel_tuning/`; `resume`,
`status`, and `export` take that directory. It contains the frozen task and tool
source, baseline Solution, every scratch attempt, candidate Solutions/source trees/plans,
worker requests/commands/stdout/stderr/results, decisions, and environment data.
`best.json` appears only after promotion. Winning evidence is hashed and checked
before export; source export writes the exact UTF-8 bytes embedded in the measured
Solution. Optional NCU profiling runs separately on the first workload; profile
failure never changes promotion.

Humanize2 owns resumable FlowState and the agent/session trace. The manifest is an
atomically written mirror. The latest framework journal in the dedicated workspace
wins on resume, including after an unclean exit. Attempt reservation is journaled
before scratch preparation. Interrupted attempts count toward candidate/patience
budgets; active elapsed time is checkpointed every second (an unclean kill may
lose the last heartbeat interval, excluding a stalled host). Downtime is not
charged. Exhausted runs require a new task/run. Task, hardware/software, evaluator,
and installed orchestration runtime identities must match to resume.

`humanize.json` locates the original framework epic and copied flow/resume journals
under `humanize/`; CLI-managed sessions remain in the original epic. The tooling
`uv.lock` records the isolated environment, and the manifest records the actual
installed hmz source digest and origin. Exports include the task, tool source/lock,
baseline, winning evidence, source tree, and checksums. Export remains available
after a repository tool update if archived source/evidence hashes still match.

To reproduce a measurement on a prepared target, use the saved tool's
`benchmarks/kernel_tuning/evaluate.py` with the archived request. Adjust absolute
task/solution/output locations when moving the bundle; retain task/solution digests,
nonce, runtime, and numerical/timing conditions. Use the evaluator Python recorded
in the task. The new response's request digest reflects relocated paths. This
executes evaluation, not agent search. Actual Thor compilation and performance
are not established by CPU/fake-agent tests.

### Generated catalog tasks and batch tuning

Hand-writing a task package per operator does not scale to every core kernel.
`scripts/kernel_tuning/operators.py` declares each core Triton operator once:
its eager Torch reference, the production call used as the baseline, the
model-derived workload shapes, and the numerical contract. `generate --catalog` renders a
complete task package from an entry. The baseline is the production kernel
itself: the defining repository module and the repository modules it imports
are copied byte-for-byte under `vendor/` (absolute `embodiinfer.` imports are
rewritten to relative ones; package `__init__` files are never copied), and
`baseline.py` calls it as the engine does. `generated.json` records the
repository revision and source digests. Structured integer inputs such as
segment offsets are written as safetensors data inside the task.

The catalog covers the Triton paths of pi0.5 (`ada_rms_norm`, `gated_residual`,
`gated_gelu`, `rotate_qk`, `split_kv_attention`), the Qwen2.5-VL vision tower of
the Qwen R2R low-level and panoramic policies (`rotate_half_rope`,
`segmented_attention`), and StreamVLN's Qwen2 decode path (`rms_norm`,
`add_rms_norm`, `swiglu`). ActiveVLN runs the Transformers vision tower and an
eager attention backend, so it currently calls none of these kernels. Workloads
come from committed captures (see below); the catalog's estimates are only the
fallback (`--estimated`). All generated tasks time with CUDA Graphs.

Contracts follow the production kernel, not an aspiration. `gated_residual`
and `rotate_qk` are bit-exact against Torch. The other kernels already round
differently from eager Torch, so their contract is a tolerance with a stated
reason: `atol = 2**-10` and `rtol = k * 2**-8`, where `k` counts BF16 roundings
the production kernel and the eager reference do not share (2 when only the
final casts differ; 3 for `rms_norm` and `swiglu`; 4 for `gated_gelu` and
`add_rms_norm`). The `gpu`-marked
`test_production_baseline_meets_generated_contract` checks every production
baseline against its contract, including changed-input CUDA Graph replay, and
prints the fraction of the bound it uses. Every baseline passed on an RTX 4060
Laptop GPU (Torch 2.6, Triton 3.2; at most 0.85 of its bound) and on an RTX
5090 (Torch 2.12, Triton 3.7; at most 0.90), and the evaluator preflight passed
for all ten tasks on both. Re-run both on each target before trusting a
contract there.

`tune-all --catalog` generates the selected operators into a new batch directory under
`results/kernel_tuning/batches/`, measures every production baseline with the
evaluator (preflight), and then runs each operator serially in its own
`run`/`resume` subprocess. A failed operator does not stop the batch; an
interrupt does, and `tune-all --resume BATCH` continues it, resuming existing
run archives. Promoted kernels are exported to `exports/<operator>`, and
`summary.md`/`summary.json` report attempts, promotions, and the improvement
over the production baseline from the weakest paired round. Machine-specific
settings (`evaluator_python`, `device`, budgets) are passed with `--set` at
generation time, and `--hardware-notes` copies target-hardware notes into each
task as `HARDWARE.md` for the agent.

### Model-scoped operator discovery

The model workflow selects exactly one canonical policy ID before preparation.
During the existing preparation execution, a scoped ATen observer and wrappers
around existing backend entrypoints collect complete input/output metadata.
Existing compound kernels remain single operators; this change introduces no
new fusion, quantization, runtime registration, or engine API.

The observed scope is end-to-end inference, not a list of policy methods. It
opens at the model-agnostic entries every deployment path uses: `EngineCore`
construction and execution, the `GenerationBackend` rollout calls, the policy's
`collate`, and its serving adapter's construction and `infer`/`infer_batch`.
Everything below an entry is recorded, including decoder-specific recurrent
decoding, RL sampling, request preprocessing and action postprocessing. Scope is
tracked per thread and the observer is installed on whichever thread enters, so
the async engine's executor and batched-serving workers are covered. Model
weights read by arithmetic outside every entry (a driver calling policy
internals directly) and a run without any entry are capture errors, so a
bypassed path cannot silently produce a smaller operator set.

The capture freezes the model type, supplied checkpoint/configuration descriptors, Torch version,
numerical settings, operator overloads, tensor shapes/strides, argument trees,
aliasing and mutation contracts, and every observed workload (no top-eight
truncation or estimated-shape fallback). Quantized calls and their internal
floating-point operations are excluded by invocation scope, including dynamically
created quantized projections and first-forward packing. Metadata-only operations
and host-side work are reported separately from device computation. Discovery
covers the executed preparation paths, not unvisited branches.

Uncompiled preparation is required; compilation is rejected rather than silently
changing the model's attention or execution path. CUDA Graph metadata is collected
during warmup/capture and calls are counted on replay. Bounded fixtures may be saved
outside graph capture; cases requiring unavailable fixtures are coverage gaps.
Unknown external operators, unrepresentable arguments/state, and unsupported
contracts also remain explicit gaps. Every eligible workload must have a task and
all production baselines must pass preflight before any model batch starts an
agent. No partial-model or skip-preflight escape hatch is provided.

ATen tasks replay the exact captured overload as their frozen bit-exact reference
and production baseline. Backend tasks snapshot their existing implementation.
The existing Humanize2 search, evaluator, paired promotion rules, archive and
export remain in use. Layout/state-aware task adapters extend evaluation without
weakening the default pure-operator checks. Inputs, current stream, numerical
settings, changed-input graph replay, and declared mutations are validated.

This is repository tooling under scripts/kernel_tuning and measurement under
benchmarks/kernel_tuning. A static operator whitelist was rejected because it
cannot discover model calls; full-model graph compilation was rejected because
it changes operator boundaries and is outside the no-new-fusion scope. Existing
catalog tasks remain available explicitly as a legacy workflow. Legacy captures
cannot establish model coverage and must be recaptured for model tuning.

CPU regression coverage must exercise uncatalogued operators, model identity,
all shapes/layouts, quantized scopes (including dynamic modules), unsupported
cases, immutable capture artifacts, and the all-baselines preflight barrier.
GPU/checkpoint validation remains explicit in the matching model environment;
CPU/fake-agent passes do not establish GPU correctness or performance.

#### Commands and artifacts

Run preparation in the selected model's own runtime. The bundled driver
`scripts.kernel_tuning.discovery.drive` constructs the policy and engine with the
serving transports' arguments and replays recorded inputs: serving requests go
through the policy's adapter as HTTP requests would, engine observations through
`collate` and `EngineCore.execute`, with one session per recurrent episode.
`benchmarks/kernel_tuning/discovery_inputs.py` exports such inputs from the pi05,
Qwen navigation and StreamVLN benchmark data; any script that drives one of the
entries above may be used instead. Compilation must be disabled explicitly (for
example `--no-cuda-graph` for StreamVLN, whose graph mode compiles its prefill).
The capture does not launch an extra forward or change the attention backend.
Canonical policy IDs are those accepted by the factory, for example `pi05`,
`streamvln`, or `qwen2.5-vl-3b-r2r-low-level`; catalog group names are not model IDs.

```bash
/absolute/runtime/python -m benchmarks.kernel_tuning.discovery_inputs pi05 --config PI05_BENCHMARK.yaml --output pi05-inputs.pt --serving-config pi05-serving.json
/absolute/runtime/python -m scripts.kernel_tuning capture --model pi05 --fixture-bytes 8589934592 --output results/kernel_tuning/captures/pi05 -- -m scripts.kernel_tuning.discovery.drive --policy pi05 --checkpoint CHECKPOINT --adapter-config pi05-serving.json --inputs pi05-inputs.pt
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning generate --model pi05 --captured results/kernel_tuning/captures/pi05 --list
/absolute/runtime/python -m scripts.kernel_tuning calibrate --captured results/kernel_tuning/captures/pi05 --output results/kernel_tuning/captures/pi05-calibrated
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning tune-all --model pi05 --captured results/kernel_tuning/captures/pi05-calibrated --agent 'HARNESS/MODEL:EFFORT' --set evaluator_python=/absolute/runtime/python
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning tune-all --resume results/kernel_tuning/batches/BATCH_ID
```

`--dry-run` builds and validates all tasks without evaluation. `--preflight-only`
evaluates all baselines without agents; if that batch is later resumed, supply
its agent at creation time. Model preflight failures prevent every agent from
starting. After all baselines pass, an individual search failure is recorded and
other prepared tasks may continue. Resume rechecks capture/task/tool identities
and reruns model preflight. The single-task `run`/`resume` paths also check the
parent model batch. There is no model-mode `--force`, `--estimated`, operator
filter, multiple-model selection, or `--skip-preflight` option.

Capture artifacts are `manifest.json` (actual policy type, builder configuration,
revision, Torch version, numerical settings, completeness and hashes),
`operators.json` (overload/backend identity, signature, every workload and counts),
`coverage.json` (ready/excluded_quantized/metadata_only/host_only/blocked counts),
and bounded `fixtures/`. A model batch snapshots this directory under `capture/`,
creates a task per call signature under `tasks/<operator-id>/`, and retains
`batch.json`, preflight records, run archives, exports and summaries. Each task
includes `replay.json` with storage/layout/mutation contracts and model identity;
its baseline calls the exact ATen overload or snapshotted backend implementation.
Different shapes and strides remain workloads of the same signature; overload,
scalar/static arguments, tensor rank/dtype and output structure distinguish tasks.

The default 64 MiB fixture budget is explicit (`--fixture-bytes`); the example
raises the maximum to 8 GiB. Integer/index storage and all inputs of supported
numerically calibrated operators require real fixtures, even for large weights.
For other exact tasks, floating storage up to 1 MiB is captured when budget
permits and other floating storage uses seeded generation. First observed
fixtures represent each structural case; calibration tests those real values,
fresh random values, zeros and cancellation inputs with masks/indices preserved.
This is finite validation, not proof for every input. Required fixture exhaustion
is a coverage gap. `calibrate` adds `calibration.json` to a new capture; generated
tasks carry each operator's frozen bounds as `numerics.json`. See the numerical
policy in Section 6 for the formula, environment binding and limitations.

Replay preserves shared storage, strides and offsets. It checks output values,
layouts, input/output and output/output aliases, declared mutations and unchanged
input storage. Timing resets backing storage before every sample, outside timed
events, in both eager and graph modes. The default is eager; `timing.mode=cuda_graph`
also requires successful changed-input A/B/A replay. NCU profiling currently
requires a dedicated replay-aware adapter. Numerical overrides cannot change
the selected exact contract or the frozen calibrated tolerance contract.

Default-generator `aten.randn.default` and `aten.randn.generator` with
`generator=None` have an eager replay contract. Capture stores the opaque CPU
and target CUDA generator states immediately before and after the existing call,
within the same required-fixture budget. Seed 0 replays the actual captured
position; other validation seeds use independent default-generator states.
Validation compares output bytes and the complete post-call generator states,
rejecting extra draws or rewinds even when output values match. State resets
happen before each warmup/sample, outside timing events, and evaluation restores
the caller's Torch RNG states on success or failure. Explicit generator objects,
other random overloads, and RNG tasks using CUDA Graph timing remain gaps.

SDPA's dispatcher RNG tag is conditional: `dropout_p=0` (including its schema
default) does not require an RNG replay contract. Discovery resolves positional,
keyword and omitted dropout arguments without changing the invocation, backend,
mask, scale or attention semantics. Nonzero dropout remains a coverage gap until
its own state contract is implemented; other nondeterministic tags still apply.
The same rule covers the backend overloads (flash, efficient, cuDNN, CPU) that
model code reaches under `no_grad`. Without dropout their `philox_seed` and
`philox_offset` outputs are left uninitialized, so replay records them as
`undefined_outputs` and checks every other output exactly; no other operator may
declare undefined outputs.

Replay tasks declare `definition_schema=embodiinfer-replay-v1`. Their local
`ReplayDefinition` extends the pinned FlashInfer tensor schema with `float64`,
retains upstream structural validation, and uses the same builders with native
double-precision tensors. Captured fixtures, input materialization and output
checks preserve float64 without a cast or relaxed tolerance. Ordinary FlashInfer
definitions and their accepted dtype set are unchanged.

Existing RoPE cache writes and greedy-sampling workspace mutations have explicit
contracts; graph GQA and prefill entries remain individual existing operators.
`GreedyWorkspace` has a typed tensor-state recipe; arbitrary Python objects still
require an adapter. Dtypes unsupported by the replay schema remain task
construction gaps rather than silently converted inputs.

Known quantization scopes are `QuantizedLinear` (FP8/INT8/NVFP4), conversion and
packing helpers, their backend functions, and explicit native quantization
dispatcher operators. Normal integer indices are not a quantization signal.
Unknown dispatcher extensions and raw Triton launches are reported as gaps;
random operators outside the default-generator randn contract, opaque Python
arguments, and calls without tensor outputs also require an adapter. Custom
C++/CUDA calls that bypass both the dispatcher and
these entrypoints need explicit instrumentation before claiming coverage.
Capture is a single-process preparation facility; it does not certify unvisited
branches, other worker processes, or checkpoint-level action parity. Existing
fused kernels count as one call; no new fused operators are proposed or generated.

### Captured catalog workloads

Catalog shapes are estimates. `capture --catalog` replaces them with production traffic:
it runs an unmodified model script in the model's runtime with every catalog
kernel wrapped, converts each call to the operator's task axes, scalars, and
fixed integer inputs, and writes one JSONL row per distinct case with its call
count. Calls outside a task's semantics (another dtype, broadcast tables,
custom attention scaling) are counted as skipped with the reason. Kernels
launched while a CUDA Graph is captured are attached to that graph and counted on
every replay, so production graph configurations can be captured; compilation
should be disabled. Captures committed under `benchmarks/kernel_tuning/captures/`
replace the catalog estimates by default, using each operator's most frequent
cases weighted by call count; `--captured FILE` selects another capture and
`--estimated` restores the estimates.

Shapes depend on the configuration and the inputs, never on weight values.
`skeleton REPO --revision REV` therefore builds a random-weight copy of a
Hugging Face checkpoint: it downloads the small files and synthesizes each
safetensors file from its header, read with an HTTP range request. Benchmark
scripts then run unchanged on it, with synthetic frames where the policy resizes
images to a fixed size anyway. Random weights do not reproduce generated text, so
decode lengths are fixed in the capture configuration.

`benchmarks/kernel_tuning/captures/pi05-libero10.jsonl` comes from
`benchmarks/pi05-benchmark` on 200 LIBERO-10 frames (10 tasks), with the
production Triton paths enabled (`native_inference`, Triton prefix and denoise
attention). It corrected three estimates. The prefix holds two cameras plus a
48- or 64-token language bucket (560 or 576 tokens, not 968). `split_kv_attention`
also runs once per layer over the whole prefix with no cached keys (`P=0`,
`S=576`). The checkpoint's AdaRMS modulation is BF16, not FP32; the catalog now
declares BF16 (recalibrated: at most 0.79 of its bound on an RTX 5090). At B=1,
four of the five pi0.5 kernels measure about 4 µs on an RTX 5090, the device's
launch floor; only `split_kv_attention` (16-35 µs) leaves room to tune.

The Qwen R2R captures (`qwen-r2r-low.jsonl`, `qwen-r2r-panoramic.jsonl`) use
skeleton checkpoints and synthetic R2R/RxR episodes (the trajectory release is
gated; episode lengths follow the benchmark READMEs) with `attention_backend:
triton` and CUDA Graphs on: only the graph path runs EmbodiInfer's Triton vision
forward. Each call encodes 4 history frames plus the current one: 1,980 patches
in 46 ragged windows (low-level) or 7,704 patches in 172 windows (panoramic),
not the estimated uniform 1,024/4,096. The rotary tables are BF16, which makes
the production RoPE bit-exact; the catalog now declares both. These shapes put
`segmented_attention` at 357 µs and 1,405 µs and `rotate_half_rope` at 18 µs and
61 µs on an RTX 5090. The StreamVLN capture (`streamvln-r2r-rxr.jsonl`, skeleton
weights, decode capped at the five tokens of its fast action path) sees only
single-row decode calls: `swiglu` never runs on prefill chunks, and all three
StreamVLN kernels sit at the launch floor.

```bash
PYTHONPATH=/path/to/EmbodiInfer /path/to/model-runtime/python -m scripts.kernel_tuning capture --catalog --output pi05.jsonl -- benchmark.py --config capture-config.yaml
python -m scripts.kernel_tuning tune-all --catalog --model pi05 --captured pi05.jsonl --agent 'HARNESS/MODEL:EFFORT'
```
