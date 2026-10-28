# 0008 — Agent-driven kernel optimization: profiling to whole-model benchmarks

- Status: Accepted for the existing tuning contracts; end-to-end extensions are partially implemented.
- Author: EmbodiInfer maintainers
- Date: 2026-10-02

This is the canonical document for the former proposals 0008 and 0009. It combines
the Humanize2 search/evaluation contract with the hotspot-to-model workflow.
Implementation status below is checked against the current source; consolidation
does not approve pending numerical or runtime changes.
For executable examples, see [automated tuning workflow](#11-automated-tuning-workflow).

## 1. Summary

The workflow is **profile → select → register if needed → capture → calibrate →
tune → install → verify → benchmark**. Start with the model's production latency,
choose worthwhile registered operators, let a coding agent improve their kernels,
and measure the resulting model with a hardware profile. A faster isolated kernel
is an intermediate result; success requires whole-model correctness and latency
evidence under fixed conditions.

Humanize2 (`hmz`) owns agent execution, budgets, traces and resume. Repository
scripts prepare tasks, validate evaluator evidence and install exported sources.
Formal measurement lives in `benchmarks/`; model-neutral dispatch lives in
`layers/`; concrete kernels and profile loading live in `backend/`. The engine
has no tuning scheduler or model-name branches.

| Stage | Current entry point | Current scope |
|---|---|---|
| Profile | `profile --output DIR -- DRIVER` | GPU attribution by stage, registry operator, attention backend and module |
| Select | Read `hotspots.md` and choose operators | Manual; automatic selection and fusion planning are future work |
| Register | `register new/run/resume/status/finalize` | Agent-assisted registration of one caller-selected operator; pi0.5 routing convention |
| Capture | `capture-ops --model MODEL --ops OPS ...` | Real arguments and counts of selected registered operators, in eager mode |
| Calibrate | `calibrate --captured DIR --output NEW_DIR` | Freeze supported numerical contracts before search |
| Tune | `tune-all`, or `run` for a prepared task | Humanize2 candidate loop with baseline preflight, correctness and paired timing |
| Install | `install EXPORT --profile NAME` | Python/Triton exports from registry captures only |
| Verify | `verify --profile NAME --output REPORT -- DRIVER_ARGS` | Whole-model FP32-reference/production-Torch/candidate comparison |
| Benchmark | Existing model benchmark scripts | Production end-to-end latency; separate configurations for each comparison row |

There is **no `optimize`, `select`, or `benchmark` subcommand** in
`scripts.kernel_tuning` today. The stages are invoked separately. `tune-all`
resumes search batches, and `register resume` resumes registration; neither
resumes the entire end-to-end workflow. pi0.5 is the current integrated policy.
Qwen R2R/StreamVLN registry integration and committed hardware profiles remain
outstanding; their benchmark-input/catalog support does not imply profile support.

## 2. Motivation and current gap

An isolated operator can improve without reducing model latency: its contribution
may be small, it may already be near launch overhead, or its tuned layout may
never be selected in production. Profiling and a final model benchmark connect
search effort to the actual inference path.

The registry workflow uses an explicit Torch reference and stable model call
sites so verified kernels can reach inference without editing the policy on every
tuning run. The independent catalog supports investigations of hand-maintained
kernels, but its exports require separate runtime integration. Full-model ATen
discovery has been removed; it did not produce installable registry call sites.

Current boundaries are visible in `scripts/kernel_tuning/__main__.py` (commands),
`regions.py` (registry capture), `install.py` (accepted exports), and
`hotspots.py` (attribution). The remaining gap is automated region selection,
new fusion integration, shape-range coverage and orchestration across stages.

## 3. Goals and non-goals

- Optimize a selected model on a prepared target machine with explicit agent,
  checkpoint, inputs, dtype, precision settings and execution configuration.
- Freeze workload coverage, numerical gates, timing and acceptance rules before
  search; retain all attempts and export exact verified source with evidence.
- Select installed kernels through optional hardware profiles with layout checks
  and fallback; evaluate the complete profile at model level.
- Keep the existing catalog and custom-task workflows usable, with
  their original contracts and artifact formats.

Robot/simulator communication, task success metrics, deployment orchestration,
automatic merge, automatic GPU-stack installation and changes to model mathematics
are outside this workflow. Installing a kernel does not make it the default.
Multi-GPU, broader batch sizes and unobserved shape ranges need their own coverage
and measurements; a B=1 capture establishes none of these automatically.

## 4. Design

### 4.1 Freeze the experiment and profile the production path

Record checkpoint/revision, evaluation input identity, dtype, attention semantics,
matmul precision and TF32 flags, seeds/noise, batch shape, decode/denoise schedule,
compile settings and CUDA Graph mode. Use the existing model benchmark for the
uninstrumented baseline latency, then run `profile` with the same production
settings to explain where GPU time goes.

`hotspots.py` instruments model-neutral inference entries and uses
`torch.profiler` correlation IDs to attribute kernels to policy/decoder stages,
registry calls, attention backends and module ranges. Repeated layer indices fold
to `layers.*`. CUDA Graph captures record node labels; replay uses those labels
when node counts agree, otherwise attribution stays at the replay-time range.
Compiled regions are attributed to their enclosing range; Python ranges are
skipped inside tracing. The implementation does not automatically rerun eagerly.

Outputs are `hotspots.json`, `hotspots.md`, and optional `trace.json`. They report
GPU time/share, kernel counts, small-kernel fractions and graph attribution
coverage. GPU activity sums and profiler-instrumented times are not an
uninstrumented end-to-end latency measurement. Inspect attribution coverage before
selecting regions; any separate eager diagnostic run must be labelled as such.

### 4.2 Select and, if necessary, register an operator

Selection is currently manual. Use time share, call frequency, kernel size and
production dispatch coverage to choose operators. A rough estimate after tuning
is `sum(call_count * (baseline_latency - candidate_latency))`; compilation,
overlap and graph execution mean this is not a measured model speedup.

An existing registered operator proceeds to capture. For an unregistered
mathematical operator, `register` drives an agent through plan, implementation,
CPU checks and repair. The caller supplies the operator and exact source
expressions; the tool does not select hotspots or fuse adjacent operations.
Its initial integration convention is pi0.5's `_op` helper and explicit
`KernelSelection` parameters. The full specification, commands and gate details
are maintained in the [registration guide](../../scripts/kernel_tuning/registration/README.md).

Registration preserves the original expression, ATen arithmetic order, casts and
shapes. Its CPU gates cover file scope, reference/registration/routing structure,
Ruff, existing and new tests, and new tests failing by assertion on BASE.
Evidence binds source bytes; interruptions and repairs preserve journaled state.
The output is a branch, `patch.diff` and `SUMMARY.md`, followed by source review
and the target-GPU bit-exact integration gate in Section 6. CPU checks do not
prove whole-model equivalence. Commit/push are explicit finalization options;
merging is separate.

New fusion regions require a separate reviewed code change with a bit-exact
Torch default before tuning. Automatic fusion selection and implementation remain
planned, as described in Section 7.

### 4.3 Capture real workloads and calibrate

`capture-ops` records registry calls on real model inputs, with selected operators
specified by `--ops`. Run capture eagerly with CUDA Graphs and compilation
disabled: replayed/compiled regions do not invoke the registry from Python.
The driver must execute at least `warmup + steps` inference calls. Every requested
operator must be observed, and the fixture budget must hold its required inputs.

The result uses the model-capture format (`manifest.json`, `operators.json`,
`coverage.json`, `fixtures/`) with `capture: registry`. Each task uses the Torch
reference in `embodiinfer/backend/torch/ops.py` as baseline and the implementation
seen during capture as a starting point. Workload weights are observed call
counts. Each case's output contract (shapes, strides, offsets, dtypes and
aliases) is taken from the call the model actually made, not from the Torch
reference: downstream operators' recorded inputs come from that call, so a tuned
kernel must return the same layout or the next tuned kernel's layout gate stops
matching. The reference must agree on everything except strides and offsets;
where its layout differs, the case also records `reference_outputs` and the
generated baseline copies the reference result into the production layout
(values unchanged; the copy counts toward baseline time). A production output
that aliases an input, has an offset or is not dense cannot be rebuilt this way
and is reported as a capture error. Registry tasks currently default to CUDA
Graph kernel timing, with an
uncompiled Torch reference; this is not a per-region Inductor comparison.
Production compilation is restored for model verification and benchmarking.

Calibrate supported tolerance contracts in the model runtime before generating
tasks. `calibrate` writes a new immutable capture with `calibration.json`;
generated tasks carry `numerics.json`. See Section 6 for exact/tolerance selection.
The capture's layouts must also occur in the compiled production path. Otherwise
the installed kernel falls back even if its isolated benchmark improves.

### 4.4 Search, evaluate and promote

#### Project skills and diagnostic feedback

The flow loads optional KernelWiki and ncu-report-skill directories from
`scripts/kernel_tuning/flow/skills/`, using Humanize2's skill mechanism.
The pinned Thor/Orin submodules under the repository's `skills/` can supply those
directories through local links. Each run records the carried SKILL.md digests
and rejects resume after they change. Missing optional skills do not block tuning.
Repository instructions and the frozen numerical contract take precedence over
generic skill recipes. Registration snapshots preserve submodule gitlink commits.

With `profile: true`, the evaluator collects the baseline before planning and
every evaluated candidate after the promotion decision. The remote-style
`profile_command` selects NCU and any explicit privilege wrapper; reports and
details are handed to the next chaser under `profiles/`, as described below.
Collection failures remain evidence and do not revoke promotion. Collection uses
the evaluator timeout and whole-run budget, without a separate report-count cap.

Migration: `diagnostics.*` settings and the old budgeted collector are retired.
Replace `diagnostics.ncu` with `profile_command: [/absolute/path/to/ncu]` and
remove the remaining `diagnostics` mapping before freezing a new task. Older runs
retain their snapshotted tools; they cannot resume under changed controller code.
The standalone `benchmarks/kernel_tuning/ncu_analysis.py` remains available for
CPU analysis of archived reports or detached native captures, using the pinned
NCU skill submodule. It is not an automatic step in the tuning flow.

A task contains `README.md`, `definition.json`, `workloads.jsonl`, `baseline.py`,
`tuning.yaml`, and optionally trusted `benchmark.py`; replay tasks also include
layout/fixture contracts. Task and solution schemas reuse NVlabs KDA/FlashInfer
formats. Humanize2 is pinned in the isolated tooling environment.

Humanize2 revision `58dd43a` fixes the Codex event-order issue upstream: an `idle`
notification does not end the turn, and successful `turn/completed` clears earlier
reconnect errors. Failed or interrupted completion still fails. The former local
event-filter shim is removed; bounded event-replay tests exercise success, failure,
interruption and waiting for completion through the pinned driver.

The agent reads the task and recent candidate source/evidence, writes a hypothesis
and executable plan, and produces a candidate. A separate evaluator compiles it,
checks every workload and times baseline, incumbent and candidate. Feedback drives
the next attempt, including rejected attempts. Agent turns do source/JSON checks;
the evaluator owns GPU compilation and validation.

The flow (`scripts/kernel_tuning/flow/`) drives two chasers in turn, in the manner
of Humanize2's `flame_chase`: attempt *n* is planned and implemented by
`first_chaser` when *n* is odd and `second_chaser` otherwise, each in a fresh
session that starts from the incumbent and every earlier attempt's evidence.
`--agent` takes one hmz spec for both chasers or two comma-separated specs, e.g.
a Claude Code and a Codex harness, whose different habits (rewrites against
refinements) the alternation exploits. The KDA skills `KernelWiki` and
`ncu-report-skill` travel with the flow when linked into its untracked `skills/`
directory; each run records the digest of every carried skill and refuses to
resume after they change. NVlabs' originals are B200-specific; device-routed forks
(SM87/Orin, SM89, SM90, SM100, SM103, SM110/Thor) keep the same names, and the plan
prompt asks agents to select the target matching `environment.json`.

A turn is stopped after `agent_timeout_seconds`. A stopped turn whose deliverable
(`PLAN.md`, then `solution.json`) already exists still counts, and the attempt
continues in a fresh session; without it the attempt fails. An evaluator failure
reaches the next agent with the worker's final exception line, and a calibrated
tolerance failure names the workload, input profile and how far the output
exceeded its bound.

With `profile: true` the evaluator collects Nsight Compute evidence (`--set full`,
restricted by NVTX to one warmed call of the heaviest workload) for the baseline
and for every evaluated candidate. Each attempt receives copies for the incumbent
and the previous attempt under `profiles/` (`ncu.ncu-rep`, `details.txt`); agents
never run the profiler. `profile_command` names NCU and an optional privilege
wrapper (`[sudo, -n, /path/to/ncu]` where counters need root). The worker is then
started through `env` with an allow-list of build variables and its own caches and
home, never secrets, because its command line is visible to other users. A
privileged NCU runs candidate code with those privileges; use it only where
counters cannot be opened otherwise, and only with agents that cannot invoke it.

`tune-all` generates tasks and preflights all baselines before opening any model
batch agent. Missing fixtures, unsupported contracts or a failed baseline block
the batch. A later individual search failure is recorded and other tasks may
continue. Each task runs in its own `run`/`resume` process, one at a time unless
`--parallel N` (also accepted with `--resume`) starts N at once. Agent turns then
overlap, but every evaluator worker (probe, check, timing, profile) holds a
per-device `flock` in the temporary directory, so measurements never share the
GPU; the evaluation timeout starts once the lock is held, and `command.json`
records the wait. Agents spend most of a run's wall time, so on one device this
multiplies throughput without changing what is measured. A preflighted item may
be set to `skipped` in `batch.json` to leave it out of the batch.

Promotion requires at least 3% weighted latency improvement and at most 5%
regression on every workload, against both the initial baseline and incumbent,
in every paired round. These defaults are configurable before the task freezes.
All checks must pass; a process exit code or printed PASS is insufficient.
Defaults stop at 20 attempts, two active hours or five consecutive
failures/non-improvements. Interruptions consume attempts; downtime is excluded.
Exhausted runs need a new task/run. Changed task/tool/environment identity
prevents resume.

CUDA Graph measurements flush a 256 MiB buffer and refresh inputs before the
start event; events bracket device replay, without per-sample synchronization.
Eager timing includes launch cost. Compare only under the same timing mode,
warmup, iterations and paired-round protocol. Optional NCU profiling is separate
from promotion. [Appendix A](#appendix-a-task-owned-benchmark-interface) specifies
custom evaluators; [Appendix B](#appendix-b-archive-and-reproduction) specifies
archives and reproduction.

### 4.5 Install and resolve a hardware profile

`install` accepts an exported registry task from `capture-ops`, with a
Python/Triton solution and an operator already registered in the current tree.
Catalog exports and CUDA C++ solutions are not installable through
this command. It writes `embodiinfer/backend/triton/tuned/<profile>/<op>/<variant>/`
and `profile.json`. An operator's layouts are often split across tasks (FP32 vision
and BF16 language GEMMs, prefix and decode rotations), so each installed task becomes
one variant and the generated `<op>/__init__.py` sends a call to the variant that
verified its exact layout. A variant whose layouts overlap an installed one, a
reinstall of the same task, or rebuilding an older single-kernel package requires
`--replace`, which drops the overlapping variants.

The manifest records device name, compute capability, Torch/Triton match keys,
source digest, verified layout count and kernel-search provenance. CUDA version
and full experiment conditions belong in the archived evidence as well. The
installer generates `applies` from verified argument shapes, dtypes, strides and
constants. Size-one and empty-tensor strides are normalized; other layouts fall
back to the policy's preferred implementation.

`embodiinfer/layers/ops.py` exposes `register_op`, `register_lazy_op` and
`KernelSelection`. A policy prepares its selection before compiled execution,
then resolves a cached callable. `kernel_profile` supports:

| Value | Behavior |
|---|---|
| `none` (default) | Preserve existing Torch/Triton preferences |
| `torch` | Force registered operators to their Torch references |
| `auto` | Select a profile matching device/capability/Torch/Triton, otherwise use `none` |
| Explicit name | Select that profile; a missing or incompatible profile raises |

Only operators present in a profile are overridden. Profile loading verifies
source digests. Backend modules remain pure tensor code with no engine/policy
imports. Registration and profile loading are runtime contracts; search is
repository tooling.

### 4.6 Verify the complete model, then benchmark

After installation, `verify` runs the same recorded inputs through an FP32 Torch
reference, a production-dtype Torch baseline and the selected profile, with the
same per-input seeds. It applies the model gate in Section 6 and writes a JSON
report. Its arguments after `--` are driver arguments, not a Python module name.

Installation precedes this check so the candidate can be exercised. **`install`
does not certify model correctness, and `verify` does not automatically remove a
failed install or update `profile.json`.** Retain the report with input/checkpoint
identity and kernel evidence; do not publish a failed profile as verified.

Next use the model's existing benchmark under the frozen production settings.
Compare `kernel_profile: torch`, existing Triton with `kernel_profile: none`,
and the candidate profile; record attention choices separately because the
registry switch does not replace every backend in the model. Measure end-to-end
mean/P50/P95, available stage splits, and per-operator ablations using separate
profile configurations. Confirm that measured calls actually select tuned kernels.
Profiler timings and weighted kernel gains cannot substitute for this run.

### 4.7 Alternatives and compatibility

- A separate agent controller would duplicate Humanize2's sessions, budgets and
  journals; reuse its Flow instead.
- Full ATen discovery did not establish installable registry call sites and has
  been removed. Capture selected registered operators with `capture-ops`; shared
  replay contracts and the inference driver remain available (Appendix C).
- Existing Triton is a useful starting point/comparison row; the registry task's
  baseline remains its frozen Torch reference. Inductor production performance
  is measured at model level, not assumed from kernel timing.
- Uncached runtime dispatch adds launch-path overhead; prepare selections before
  tracing and resolve cached callables.
- Generated policy changes and default profile promotion require ordinary review;
  neither search nor installation automatically merges them.

## 5. Model-agnosticism verdict

Task orchestration, evaluation, registry dispatch and hardware matching are
model-neutral. Model inputs use existing drivers; model-specific call sites and
fused-region semantics stay in the owning policy. Reusable mathematical
references/kernels live under `backend/`, which does not import `models`,
`policies` or `engine`. The inference engine does not acquire agent dependencies,
training adapters, environment control or benchmark loops.

## 6. Losslessness and precision criterion

Bit-exact comparison is the default. Registration/routing changes must preserve
outputs against BASE with `kernel_profile: none`, including whole-model target-GPU
outputs (`max |delta action| = 0`). Compare the same checkpoint, dtype, attention,
matmul precision/TF32, batch, seeds/noise and decode/denoise schedule. The tolerant
`verify` command below is not a substitute for this integration gate.

Calibrated tolerances are an explicit contract for supported arithmetic whose
accumulation/rounding differs. This document preserves those existing contracts;
it does not relax assertions or authorize approximate kernels merely because a
candidate fails.

### Operator gate

Registry capture records execution settings separately from acceptance rules.
Floating registry calls use their declared Torch reference for FP64 calibration.
Mutation, non-floating outputs, FP64 and unsupported dtypes retain bit-exact
validation. Existing hand-written/catalog contracts retain their own rules.

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
constraints remain exact. Nonfinite reference/baseline/output values
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
measurement), registry capture/task generation, and the replay evaluator.
Calibration itself changes no runtime API or model-level action gate. CPU tests
exercise selection, calibration, per-workload bounds, attention
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

For registry captures, B is the registered Torch reference in the production
dtype. Unknown numerical semantics retain the exact contract. Dtype alone is
never sufficient to choose a tolerance.

### Model gate

`scripts/kernel_tuning/verify.py` uses FP32 Torch output R, production-dtype
Torch output B and candidate-profile output C. For each continuous output name,
compute max-absolute error and RMSE against R for every input, then require:

| Statistic of candidate error | Limit |
|---|---|
| Mean max-absolute error | `max(1e-4, 1.25 * mean baseline max-absolute error)` |
| Mean RMSE | `max(1e-5, 1.25 * mean baseline RMSE)` |
| Worst max-absolute error | `max(1e-4, 2 * worst baseline max-absolute error)` |

Discrete outputs must equal B on every input. Mismatched output structures and
incomparable baseline/reference runs fail explicitly. Freeze the evaluation set,
seed base (default 1000), checkpoint and settings before candidate verification;
thresholds derive from the baseline and fixed policy constants, never candidate
error. Record the actual sample count: `--samples` can shorten the run and does
not establish full-set coverage. For pi0.5, the intended integration evaluation
includes its frozen cross-task references plus 200 benchmark frames; the CLI does
not assemble or enforce that set automatically.

When using Inductor, preserve the same compiler configuration/cache conditions;
use `TORCHINDUCTOR_DETERMINISTIC=1` for reproducible parity as in the pi0.5
validation workflow. Any unavailable deterministic behavior must be reported,
not hidden by widening thresholds. FP32 calibration, operator correctness and
whole-model parity each have distinct reference paths and evidence.

## 7. Implementation status and remaining design

Implemented components are the Humanize2 Flow/archive, registry
capture, calibration, evaluator, batch tuning, GPU attribution, registry/profile
loader, install, model verification, the CPU registration controller, and an
opt-in `--unfenced` agent mode for hosts without Landlock (Section 11.1). The
pi0.5 call sites are integrated. Two hardware profiles are committed for Jetson
AGX Orin: `orin_pi05` (Section 9.1) and `orin_pi05_r2` (Section 9.2).
`kernel_profile` still defaults to `none`, so a profile is used only when
selected by name or by `auto` on matching hardware; `auto` takes the first
matching profile in name order, which on Orin is `orin_pi05`.

Remaining work from the merged design:

1. Rank recoverable latency using measured bandwidth/compute/launch floors,
   rather than GPU time alone; detect adjacent memory-bound fusion candidates.
2. Add an agent selection flow producing a reviewable `plan.json` (proposed
   default: at most five regions and a 1% end-to-end contribution threshold).
   These are design targets, not currently enforced CLI options.
3. Integrate new fused Torch regions through reviewed, bit-exact code changes;
   extend registry use to Qwen R2R and StreamVLN.
4. Cover declared shape ranges, batches and layouts instead of only captured
   signatures; validate fallback and dispatch coverage.
5. Add a resumable cross-stage `optimize` orchestrator and bind model verification,
   evaluation-set identity and benchmark evidence into profile provenance.
   Today these artifacts must be retained together by the operator.

The proposed roofline bound is `max(bytes / measured bandwidth, FLOPs / measured
peak)`; recoverable time subtracts the larger of that bound and launch floors.
Fusion estimates account for eliminated intermediate traffic. These estimates
must be checked by model benchmarks before becoming performance claims.

## 8. Test plan

CPU tests under `tests/kernel_tuning/` cover task validation, Humanize2 fake-agent
execution, failed/interrupted attempts, budgets/resume, evidence tampering,
preflight barriers, replay layouts/aliases/input integrity, calibration, hotspot attribution,
installation, model-gate arithmetic and registration repair. Fake-agent sessions
also check that runs keep Humanize2's default permission unless `unfenced` is
recorded, and that batches and registration freeze that choice. Registry tests
also cover preference, fallback, profile mismatch and digest checks.

GPU and checkpoint tests must use their markers and matching isolated model
runtime. Required evidence includes production-baseline correctness, changed-input
CUDA Graph replay, registration bit-exact integration, profile hit/fallback paths,
and whole-model verification. CPU/fake-agent passes prove neither GPU kernel
correctness nor speed. Test future selection/fusion logic when implemented;
do not describe those planned tests as existing coverage.

## 9. Benchmark plan

Follow the owning model's benchmark protocol, for pi0.5
[`benchmarks/pi05-benchmark/README.md`](../../benchmarks/pi05-benchmark/README.md).
A target experiment is one prepared GPU, B=1, BF16, ten denoise steps, fixed
LIBERO-10 inputs, production compile settings and full-loop CUDA Graphs. The
standard config selects 1,600 frames; shortened runs must report their actual
coverage. Record GPU/driver/CUDA/Torch/Triton, checkpoint/revision, source digests,
dtype, attention, precision, seeds, batch, warmup, repetitions, step counts and
all runtime switches for each row.

Report Torch, existing Triton and tuned-profile rows with mean/P50/P95 and stage
splits, plus operator ablations. An additional Inductor `max-autotune` row is
useful only where it compiles and passes the same numerical gate. Keep raw
conditions and measurements with the report. A win on one GPU, shape or batch
is not evidence for another.

### 9.1 `orin_pi05` on Jetson AGX Orin

Measured 2026-10-15 on Jetson AGX Orin 32 GB (sm_87, `nvpmodel` MAXN), L4T
R36.4.7, CUDA 12.6, Torch 2.10.0, Triton 3.6.0, checkpoint `pi05_base`
(`config.json` SHA256 `36786971…`). Inputs are 200 recorded observations from
the first two LIBERO-10 task files (dataset revision `f13aa24a…`, input file
SHA256 `94e5cad5…`). B=1, ten denoise steps, seed 1000 with one FP32 CUDA noise
draw per input rounded to BF16. Mixed precision: the SigLIP vision tower and
multimodal projector run in FP32, everything else in BF16. `native_inference`,
`native_embeddings`, Triton prefix and denoise attention, `compile_backend:
none`, full denoise-loop CUDA Graph, `TORCHINDUCTOR_DETERMINISTIC=1`. Ten warmup
calls, then the 200 inputs three times (600 timed calls). Rows differ only in
`kernel_profile`. All rows ran the source tree that adds this profile, in which
pi0.5 also routes its projections through the `linear` registry operator and its
SigLIP tanh GELU through `gelu_tanh`.

| `kernel_profile` | Mean (ms) | P50 (ms) | P95 (ms) | vs `torch` |
|---|---|---|---|---|
| `torch` | 825.5 | 825.3 | 827.5 | — |
| `none` (existing Triton) | 609.8 | 609.7 | 611.1 | −26.1% |
| `orin_pi05` | 537.8 | 537.5 | 539.9 | −34.9% (−11.8% vs `none`) |

Deviations from the protocol above: an experiment-local evaluator replaced
`benchmarks/pi05-benchmark/benchmark.py`, which does not pass
`native_embeddings` and fixes camera keys for the LIBERO fine-tuned checkpoint;
200 inputs instead of 1,600; no prefill/decode split. The prefix encoder runs
eagerly instead of under Inductor because the tuned kernels' layout gates call
`Tensor.storage_offset()`, which Dynamo cannot trace under `fullgraph`; the
whole denoise loop is still captured in one CUDA Graph.

Every captured layout dispatched to its tuned kernel: no `rotate_qk` or
`split_kv_attention` call fell back. `linear` falls back for the layouts that
were not tuned (FP32 vision layers and the fused gate/up projections). Against
`torch`, the first action of every chunk (6,400 values) differs by at most
0.0156 (RMSE 0.0016). The `none` row differs from `torch` by a similar amount
(0.0195 / 0.0016). The whole-model `verify` gate (Section 6) was not run for
this profile.

Operators, from the promoted attempt's paired evaluator rounds against the
task's Torch baseline (calls per inference in parentheses):

| Operator (run) | Workload | Torch (ms) | Tuned (ms) |
|---|---|---|---|
| `split_kv_attention` (`20261015T055303Z-8eeab7204c`) | prefix, 672 keys (18) | 3.62 | 0.49 |
| | denoise, 672 + 50 keys (180) | 0.908 | 0.070 |
| `linear` (`20261015T090245Z-031b8d3bd3`) | prefix `down_proj`, 672×16384→2048 (18) | 6.45 | 3.77 |
| | expert `down_proj`, 50×4096→1024 (180) | 0.076 | 0.078 |
| `rotate_qk` prefix (`20261015T134002Z-6a72e1be9e`) | 672 tokens (18) | 0.434 | 0.049 |
| `rotate_qk` decode (`20261015T150031Z-03be0c2142`) | 50 tokens (180) | 0.071 | 0.013 |

The tuned expert `down_proj` is about 3% slower than Torch. It was promoted
because the weighted mean over both workloads improved.
`rotate_qk` combines two separately tuned exports with disjoint verified
layouts. Its package dispatches each call to the variant whose `applies`
accepts it. The decode export was tuned after capture recorded the production
output layout (a contiguous query). Replacing the existing Triton decode
rotation with it changed end-to-end latency by less than 1 ms and left actions
bit-identical.

### 9.2 `orin_pi05_r2` on Jetson AGX Orin

`orin_pi05_r2` holds eight kernels from a second Orin round, tuned with the
two-chaser flow (Section 4.4). The chasers were a Claude Code and a Codex harness,
both on DeepSeek-V4.1-Flash. They carried the device-routed `KernelWiki` and
`ncu-report-skill` forks and Nsight Compute evidence collected through `sudo -n`,
and ran `--unfenced`. Every prefix-side task covers both prompt buckets that
occur with this checkpoint (prefix length 672 or 712), and the attention task
also covers the denoise shapes of both, so one kernel serves either bucket.

The round ran two batches over the same capture:

- The first batch ran the ten operators one at a time and reached eight of them
  before it was stopped: 110 minutes, 16 candidates, patience 6, 15-minute turns
  and 10-minute evaluations per task.
- The second batch took the five tasks the first left without a kernel. These
  were the three `linear` tasks and two it had not reached. They ran at once
  (`--parallel 5`) with 5 hours, no candidate or patience limit, and 30-minute
  limits. A host reboot stopped it after 286 of its 300 minutes, and its best
  attempts were exported from the run archives.

Measured 2026-10-27 under the conditions of Section 9.1, with the following
differences:

- The source tree was `e131b3e` plus this profile.
- The GPU clock was not locked: a reboot had cleared the lock, leaving
  `nvhost_podgov` at 306–930.75 MHz.
- Rows ran in the order `none`, `orin_pi05_r2`, `orin_pi05`, `orin_pi05_r2`,
  `none`, each with ten warmup calls and 600 timed calls. The mean is over the
  runs of each profile.

`none` and `orin_pi05` reproduced their Section 9.1 means within 1.0 ms.

| `kernel_profile` | Runs | Mean (ms) | P50 (ms) | P95 (ms) | vs `none` |
|---|---|---|---|---|---|
| `none` (existing Triton) | 2 | 610.8 | 609.8 / 609.8 | 611.1 / 612.9 | — |
| `orin_pi05` | 1 | 537.6 | 537.4 | 538.9 | −12.0% |
| `orin_pi05_r2` | 2 | 515.0 | 514.3 / 514.6 | 516.2 / 516.9 | −15.7% (−4.2% vs `orin_pi05`) |

**Numerical gate.** The Section 6 gate (`scripts.kernel_tuning.verify.gate`)
passed over the 200 inputs with shared noise, worst ratio 0.854. The reference
was `kernel_profile: torch` in the same mixed precision, because an FP32
reference does not fit in the 32 GB unified memory. For the action chunk:

| Measure | `orin_pi05_r2` | Limit | `none` |
|---|---|---|---|
| Mean max-abs error | 0.00513 | 0.00605 | 0.00484 |
| Mean RMSE | 0.00167 | 0.00196 | 0.00156 |
| Worst max-abs error | 0.0254 | 0.0391 | 0.0195 |

**Dispatch.** The check used six inputs covering both buckets and the first 20
recorded inputs; the 20 recorded inputs all fall in the 672 bucket. Every call
of the six tuned operators dispatched to a tuned kernel. `linear` falls back only
for the layouts without a promoted kernel:

- the FP32 vision projections,
- the fused 2048→32768 gate/up projections,
- the expert 1024→8192 gate/up.

Operators, from the promoted attempt's paired evaluator rounds against the
task's Torch baseline (calls per inference in parentheses; the chaser that wrote
the kernel in brackets):

| Operator (run) | Workload | Torch (ms) | Tuned (ms) |
|---|---|---|---|
| `split_kv_attention` (`20261026T192726Z-44b77e65fd`) [Claude Code] | prefix, 672 / 712 keys (18) | 3.61 / 3.81 | 0.690 / 0.859 |
| | denoise, 672 / 712 + 50 keys (180) | 0.905 / 0.934 | 0.092 / 0.100 |
| `linear` (`20261027T083101Z-f3a3fe58e8`) [Codex] | prefix `down_proj`, 672 / 712×16384→2048 (18) | 6.43 / 6.61 | 3.06 / 3.09 |
| | expert `down_proj`, 50×4096→1024 (180) | 0.0765 | 0.0765 |
| `rotate_qk` prefix (`20261027T041343Z-ef7e804e63`) [Claude Code] | 672 / 712 tokens (18) | 0.432 / 0.461 | 0.050 / 0.052 |
| `rotate_qk` decode (`20261027T022339Z-cc147a0ad6`) [Claude Code] | 50 tokens (180) | 0.071 | 0.0125 |
| `gated_gelu` prefix (`20261027T003333Z-273b3ff7e2`) [Claude Code] | 672 / 712×16384 (18) | 1.27 / 1.35 | 0.354 / 0.375 |
| `gated_gelu` expert (`20261027T055939Z-b94ceefecd`) [Codex] | 50×4096 (180) | 0.0387 | 0.0163 |
| `gated_residual` (`20261027T083103Z-13ee35774e`) [Claude Code] | 50×1024 (360) | 0.0164 | 0.0093 |
| `gelu_tanh` (`20261027T083104Z-63d432d46f`) [Claude Code] | SigLIP 256×4304, FP32 (54) | 0.0586 | 0.0546 |

No candidate was promoted for the two remaining `linear` tasks:

- BF16 gate/up: 8 evaluated attempts, all slower than cuBLAS, which already
  reaches about 79% of the 26.7 TFLOP/s BF16 tensor peak at 930.75 MHz.
- FP32 vision projections: 6 attempts in the first batch and 7 in the second.
  They either exceeded the calibrated bounds, produced non-finite output, or
  were slower than cuBLAS.

**In-model attribution.** These per-operator gains overstate the end-to-end
saving. Weighted by calls, they predict about 300 ms per inference, against 96 ms
measured. One inference was profiled with Nsight Compute using an
experiment-local driver. NCU fixes the clock at its base, so durations were
converted from cycles to 930.75 MHz. The kernel sums are 589.6 ms (`none`) and
498.3 ms (`orin_pi05_r2`), against 610.8 and 515.0 ms end to end; the rest is
mostly launch gaps in the CUDA Graph. By stage:

| Stage | `none` (ms) | `orin_pi05_r2` (ms) |
|---|---|---|
| SigLIP vision tower and projector, FP32 | 248.5 | 248.6 |
| Gemma prefix | 257.2 | 181.0 |
| Action expert (ten steps) and input/output | 83.8 | 68.7 |

The prefix gain is mostly `down_proj` (−61 ms) and `gated_gelu` (−12 ms).
`split_kv_attention` accounts for most of the overestimate. Its task baseline is
the registry's Torch reference: 3.6 ms per prefix call and 0.9 ms per denoise
call. `none`, however, already runs the existing Triton split-KV kernels, at
0.49 ms per prefix call and 0.155 ms per denoise call in the model. Against that
path the tuned kernel saves 14.7 ms in the denoise loop and costs 3.3 ms in the
prefix, at 0.67 ms against 0.49 ms per call. A task should therefore be measured
against the implementation its production configuration actually runs.

## 10. Risks and limitations

- Captured layouts are finite; a production compile can change strides and cause
  fallback. Check dispatch coverage and benchmark the actual production path.
- Region selection and fusion remain manual. Large unregistered GEMMs cannot be
  optimized by `capture-ops` until their routing/reference contract exists.
- Profile environment matching is strict; upgrades require new validation.
- CUPTI/graph attribution may be incomplete. Diagnostic eager execution changes
  the experiment and must be distinguished from production timing.
- Finite numerical tests do not prove correctness on every input or task success.
  Thermal state, other GPU users and clocks can affect timing despite pairing.
- Profile installation and verification are separate; failed verification has
  no automatic rollback or publication gate in the current installer.
- Humanize2/FlashInfer use pinned optional dependencies. Real Humanize2 execution
  requires POSIX and Python 3.12+ in a separate tool environment; model runtimes
  retain their own dependency pins and Python support floor.
- Agents and kernels execute with process permissions. Workspace bounds and
  artifact digests are integrity checks, not an OS security sandbox. Humanize2's
  default session fence needs Linux Landlock; `--unfenced` (frozen per run or
  batch) drops it on hosts without Landlock and leaves isolation to a dedicated
  account or container.

## 11. Automated tuning workflow

The following pi0.5 example connects the currently implemented stages. Run these
commands from the repository root, replacing paths, checkpoint, agent and profile
names with values for the target machine. Prepare real model weights and benchmark
data first. Random-weight skeletons support structural exploration but do not
establish real-model correctness or performance.

### 11.1 Prepare environments and freeze inputs

Use Python 3.12+ for the tooling environment and the model's separate runtime for
model execution and GPU evaluation. Keep the incompatible pi0.5 and Qwen dependency
groups in separate environments. See
[CONTRIBUTING](../../CONTRIBUTING.md#optional-kernel-tuning-tools) for installation
and evaluator dependencies.

```bash
uv sync --project scripts/kernel_tuning --python 3.12 --frozen
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning --help
```

Copy `benchmarks/pi05-benchmark/config.yaml` into the ignored results directory,
set the local checkpoint, dataset and output paths, and save it as
`results/kernel_tuning/pi05-benchmark.yaml`. Export the inputs and serving
configuration. Set `compile_backend`, `kernel_profile` and attention options in
that configuration's `policy_kwargs` to match the experiment; the exporter does
not copy every model switch automatically. The input count must cover the
subsequent profiling warmup and measurement window.

```bash
/absolute/runtime/python -m benchmarks.kernel_tuning.discovery_inputs pi05 \
  --config results/kernel_tuning/pi05-benchmark.yaml \
  --samples 200 \
  --output results/kernel_tuning/pi05-inputs.pt \
  --serving-config results/kernel_tuning/pi05-serving.json
```

#### Agent hosts without Landlock (`--unfenced`)

By default Humanize2 fences every coding-agent session: the session workdir is
writable, the rest of the machine is read-only, and the network stays on. On
Linux the fence is enforced with Landlock, which requires kernel 5.13 or later
built with `CONFIG_SECURITY_LANDLOCK=y` and listed as active. Check the target
host before starting agents:

```bash
uname -r                                  # 5.13 or later
cat /sys/kernel/security/lsm              # must list landlock
zcat /proc/config.gz 2>/dev/null | grep LANDLOCK || grep LANDLOCK /boot/config-$(uname -r)
```

If Landlock is compiled in but inactive, add it to the kernel's `lsm=` boot
parameter. Some vendor kernels omit it entirely; for example, Jetson L4T R36
ships with `# CONFIG_SECURITY_LANDLOCK is not set`, and enabling it means
rebuilding the kernel and its out-of-tree modules. Without Landlock, every agent
session is refused with
`HarnessSandboxed: ... fencing it from outside needs Landlock`.

On such hosts, pass `--unfenced` when the run is created. Its sessions are then
granted every Humanize2 scope, which needs no fence:

| Command | Where to pass it | On resume |
| --- | --- | --- |
| `run TASK --agent ... --unfenced` | when the run is created | `resume RUN` reuses it; no flag is accepted |
| `tune-all ... --agent ... --unfenced` | when the batch is created; every operator run inherits it | `tune-all --resume BATCH` rejects `--unfenced` |
| `register run RUN --agent ... --unfenced` | when the reviewed sites are confirmed | `register resume RUN` reuses it; another `register run` is refused |

The choice is recorded as `unfenced` in the run's `manifest.json` (and the batch's
`batch.json`, whose `summary.md` notes it), and a warning is printed to stderr
before agents start. Runs created without the flag keep Humanize2's default
fence exactly, so the flag cannot widen an existing run.

An unfenced agent can read and write everything its account can, including the
repository checkout, the run archive and the evaluator source. Digests still stop
a tuning run whose task, controller/evaluator source or winning evidence changed,
and a registration run whose specification, BASE checkout or context changed, but
they are integrity checks, not a sandbox. Run unfenced agents only under a
dedicated account without `sudo`, `docker` group membership or access to other
users' files, or inside a container that mounts only what the run needs. Give
that account the device groups the evaluator needs (for example `video` and
`render` on Jetson), its own tooling environment and GPU runtime, and its own
coding-agent login.

A refused session is recorded as a failed attempt, not as a run error. A tuning
run can therefore still finish with `status: completed` and
`stop_reason: max_candidates`, and registration retries it as agent feedback
until its attempts are `exhausted`. The batch `summary.md` shows only the stop
reason. After starting on a new host, check the first attempt for
`HarnessSandboxed` before leaving a long run unattended:
`attempts[].decision.reasons` in the run's `manifest.json` (for a batch, under
`runs/`), or `attempts/*/feedback.json` for registration.

### 11.2 Profile and select operators

Profile with the production configuration. This example uses BF16, ten denoise
steps and a full-loop CUDA Graph. The compile and attention settings in
`pi05-serving.json` must match the baseline.

```bash
/absolute/runtime/python -m scripts.kernel_tuning profile \
  --output results/kernel_tuning/profile-pi05 --warmup 10 --steps 20 --trace \
  -- -m scripts.kernel_tuning.discovery.drive --policy pi05 \
  --checkpoint /absolute/checkpoint --device cuda:0 --dtype bfloat16 \
  --num-steps 10 --capture-full-loop \
  --adapter-config results/kernel_tuning/pi05-serving.json \
  --inputs results/kernel_tuning/pi05-inputs.pt
```

Read `hotspots.md` and manually select registered operators with room for
improvement. The commands below use `gated_residual` as an example; this does not
imply that it is a worthwhile tuning target. For an unregistered operator, follow
the [registration guide](../../scripts/kernel_tuning/registration/README.md),
complete code review and CPU checks, and verify whole-model bit-exact parity on
the target GPU before capturing workloads.

### 11.3 Capture workloads and calibrate

Copy the serving configuration to `pi05-eager.json`. Set
`policy_kwargs.compile_backend` and `policy_kwargs.kernel_profile` to `"none"`,
while preserving the checkpoint, dtype, attention settings and other mathematical
semantics. Also disable CUDA Graphs during capture so Python registry calls can
be recorded.

```bash
/absolute/runtime/python -m scripts.kernel_tuning capture-ops \
  --model pi05 --ops gated_residual --warmup 2 --steps 8 \
  --fixture-bytes 8589934592 --output results/kernel_tuning/captures/pi05-ops \
  -- -m scripts.kernel_tuning.discovery.drive --policy pi05 \
  --checkpoint /absolute/checkpoint --device cuda:0 --dtype bfloat16 \
  --num-steps 10 --no-cuda-graph \
  --adapter-config results/kernel_tuning/pi05-eager.json \
  --inputs results/kernel_tuning/pi05-inputs.pt

/absolute/runtime/python -m scripts.kernel_tuning calibrate \
  --captured results/kernel_tuning/captures/pi05-ops \
  --output results/kernel_tuning/captures/pi05-ops-calibrated
```

`--fixture-bytes` sets a storage limit, 8 GiB in this example; it does not
preallocate that space. If calibration fails, inspect input coverage and the
reference contract. Do not relax thresholds to make a candidate pass.
`capture-ops --ops` selects the scope of this capture. Subsequent batch tuning
must cover every task in that capture, which does not imply coverage of every
ATen operation in the model.

### 11.4 Run the agent, resume and export

```bash
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning tune-all \
  --model pi05 --captured results/kernel_tuning/captures/pi05-ops-calibrated \
  --agent 'HARNESS/MODEL:EFFORT[,HARNESS/MODEL:EFFORT]' \
  --set evaluator_python=/absolute/runtime/python

# Resume with the printed Batch path; omit agent, selection options and --set.
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning tune-all \
  --resume results/kernel_tuning/batches/BATCH_ID
```

Use `--dry-run` to generate tasks only, or `--preflight-only` to measure baselines
without starting agents. On a host whose kernel lacks Landlock, add `--unfenced`
when creating the batch and run it under an isolated account; see
[Agent hosts without Landlock](#agent-hosts-without-landlock---unfenced). If a
preflight-only batch will later resume search, specify its agent (and
`--unfenced`, if needed) when creating it. A normal run starts agents only after
every baseline passes. Default limits apply per task: 20 attempts, two active hours,
or five consecutive failures/non-improvements.

Inspect the batch's `summary.md` / `summary.json` and each task's run evidence.
Promoted results are automatically exported to `exports/<task-name>`; use the
actual directory names in the batch. For individual runs, use `status RUN` to
inspect progress and `export RUN DESTINATION` to export the result. Proceed to
installation only when a candidate has been promoted.

### 11.5 Install and verify the whole model

Restore the production serving configuration. Install only a registry export
produced by the preceding stage. Profile names must be Python identifiers;
`EXPORT_DIRECTORY` below denotes the actual export directory.

```bash
/absolute/runtime/python -m scripts.kernel_tuning install EXPORT_DIRECTORY --profile my_gpu

TORCHINDUCTOR_DETERMINISTIC=1 /absolute/runtime/python -m scripts.kernel_tuning verify \
  --profile my_gpu --output results/kernel_tuning/verify-my_gpu.json \
  --seed 1000 -- --policy pi05 --checkpoint /absolute/checkpoint \
  --device cuda:0 --dtype bfloat16 --num-steps 10 --capture-full-loop \
  --adapter-config results/kernel_tuning/pi05-serving.json \
  --inputs results/kernel_tuning/pi05-inputs.pt
```

`verify` switches internally between FP32 Torch, production-dtype Torch and the
selected profile. It does not automatically roll back an installation that fails
verification or write verification results into `profile.json`. After the report
passes, confirm that production layouts select the tuned kernels, and retain the
checkpoint identity, input digest and evidence from each stage. This tolerance
gate does not replace the bit-exact gate for registration changes.

### 11.6 Benchmark the whole model and prepare deliverables

Create three configurations from the same pi0.5 benchmark configuration, setting
`kernel_profile` to `torch`, `none` (the existing Triton path), and `my_gpu`,
respectively. Give each a distinct `output` path. Keep all other experimental
conditions identical, including attention selection, inputs, seed, warmup,
repeats, compilation, CUDA Graphs and denoise steps. `kernel_profile` controls
registered operators only.

```bash
TORCHINDUCTOR_DETERMINISTIC=1 /absolute/runtime/python \
  benchmarks/pi05-benchmark/benchmark.py --config results/kernel_tuning/bench-torch.yaml
TORCHINDUCTOR_DETERMINISTIC=1 /absolute/runtime/python \
  benchmarks/pi05-benchmark/benchmark.py --config results/kernel_tuning/bench-triton.yaml
TORCHINDUCTOR_DETERMINISTIC=1 /absolute/runtime/python \
  benchmarks/pi05-benchmark/benchmark.py --config results/kernel_tuning/bench-profile.yaml
```

Deliver reviewable kernel/profile sources, candidate and baseline operator
evidence, the whole-model correctness report, and all three benchmark
configurations with raw results. Profiles containing multiple operators should
also include per-operator ablations. Report hardware and software versions,
checkpoint, dataset coverage and timing boundaries. Only model-level measurements
establish end-to-end gains. Installation, committing and enabling a profile by
default are separate actions. Keep raw experiment data in the results directory
and formal reports under `benchmarks/`; keep run logs out of `docs/`.

## Appendix A. Task-owned benchmark interface

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

## Appendix B. Archive and reproduction

`run` creates a unique directory under `results/kernel_tuning/`; `resume`,
`status`, and `export` take that directory. It contains the frozen task and tool
source, baseline Solution, every scratch attempt, candidate Solutions/source trees/plans,
worker requests/commands/stdout/stderr/results, decisions, and environment data.
`best.json` appears only after promotion. Winning evidence is hashed and checked
before export; source export writes the exact UTF-8 bytes embedded in the measured
Solution. Optional NCU profiling runs separately on the highest-weight workload
(the first when weights tie); profile
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

## Appendix C. Other task sources

### Registry capture artifacts and shared tooling

`capture-ops` selects one canonical policy and observes registry calls during its
measured inference window. `--ops` chooses the operator scope; subsequent
generation, calibration and tuning cover every recorded workload of that scope.
It records no arbitrary ATen operations. Run the driver eagerly with compilation
and CUDA Graphs disabled so registry calls reach Python. Capture does not certify
unvisited paths, other shapes or checkpoint-level action parity.

`scripts.kernel_tuning.discovery.drive` remains the shared driver for `profile`,
`capture-ops` and `verify`. It constructs the policy and engine using serving
arguments and replays recorded requests or observations, retaining recurrent
episode sessions. `benchmarks.kernel_tuning.discovery_inputs` exports these
inputs using the pi05, Qwen navigation and StreamVLN benchmark loaders. Those
historical module paths are retained; they do not enable full-model discovery.

The capture contains:

- `manifest.json`: `capture: registry`, policy identity/configuration, revision,
  Torch version, numerical settings, capture window, completeness and file hashes.
- `operators.json`: registry/Torch-reference/starting-implementation identities,
  argument trees, observed shapes/strides/dtypes, aliases and call counts.
- `coverage.json`: operator statuses and explicit capture errors.
- `fixtures/`: real input storage, bounded by `--fixture-bytes` (default 1 GiB).

Calibration writes a new capture with `calibration.json`; generated tasks carry
the per-operator bounds in `numerics.json`. Inputs, seeds, masks and numerical
settings follow Section 6. First-observed real fixtures represent each structural
case; random and boundary profiles provide finite additional validation.

Tasks use the frozen Torch reference as their baseline and include
`replay.json` for storage/layout/alias contracts. Production output layouts are
preserved, including the reference relayout described in Section 4.3. Replay
checks every output and rejects input mutation, resets storage outside timed
events, and validates changed-input graph replay. Registry tasks default to
CUDA Graph timing; eager timing remains available. NCU profiling uses the same
replay adapter to materialize fresh inputs before one eager call under NVTX.
This diagnostic call does not reproduce CUDA Graph replay latency, even when
paired promotion measurements use graph timing.

The shared `discovery.contracts`, `tasks`, `tensors` and `numerics` modules remain
in place for registry capture, calibration, task generation, batch preflight and
evaluation. The `embodiinfer-replay-v1` float64 definition extension also remains;
it preserves double precision without changing the ordinary FlashInfer schema.

The full-model `capture --model` entry, ATen/backend observer, discovery-only
task/oracle branches, RNG-transition replay and greedy-workspace recipes have
been removed. New model captures and replay tasks must identify registry
operators. Old discovery artifacts are rejected with `capture-ops` migration
guidance; recapture through registered call sites. `capture --catalog` remains
an independent, explicit JSONL workflow.

Calibration binds the numerical-policy/oracle source digest. This cleanup changes
that digest without changing registry reference arithmetic or threshold formulas:
existing raw registry captures can be recalibrated into a new directory before
generating new tasks/batches. Do not edit old hashes or widen thresholds to reuse
old evidence. Archived runs retain their snapshotted tooling for reproduction.

### Generated catalog tasks and batch tuning

The legacy catalog in `scripts/kernel_tuning/operators.py` supplies hand-maintained
operator shapes, entry points, Torch references and fixed numerical contracts.
`generate --catalog` snapshots production backend sources as the baseline;
`capture --catalog` records legacy JSONL workload shapes/counts, not registry
fixtures or full-model coverage. `tune-all --catalog` uses the same Humanize2
search, promotion and export loop. These exports require separate integration;
`install` accepts registry exports only.

The catalog covers pi0.5 norm/activation/RoPE/split-KV kernels, Qwen2.5-VL vision
RoPE/segmented attention, and StreamVLN norm/activation kernels. `--estimated`
explicitly opts into estimated shapes. Existing fixed tolerances remain defined
by the catalog and must pass baseline preflight on the target runtime; they are
not silently replaced by calibrated registry contracts.

```bash
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning generate \
  --catalog gated_residual --output results/kernel_tuning/tasks
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning check \
  results/kernel_tuning/tasks/gated_residual
```

Catalog capture must load EmbodiInfer's backend source from the checkout and run
the relevant path; importing an unrelated copy or disabling the path that calls
the target kernel gives incomplete coverage. Use recorded workloads for real
measurement. Synthetic/estimated workloads and skeleton checkpoints do not prove
real-checkpoint performance or model correctness.
