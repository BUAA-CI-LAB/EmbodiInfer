# 0009 — Hotspot-driven Triton tuning with installable hardware profiles

- Status: Draft
- Author: EmbodiInfer maintainers
- Date: 2026-10-09

## 1. Summary

Close the loop from kernel search to model latency. Profile a model in its
production configuration, rank code regions by the time a better kernel could
recover, let a coding agent choose a few regions (including new fusion
boundaries) and write Triton implementations that must beat the model's Torch
implementation, then install the verified kernels in the repository under a
hardware profile (for example `rtx4090`). At runtime an operator registry
selects them behind an off-by-default switch and falls back to the existing
implementation whenever a kernel does not apply. Success is measured as model
end-to-end latency under a frozen model-level numerical gate, not as isolated
kernel speedups. The registry is a model-agnostic `layers/` contract; region
definitions and call sites stay in policies; tuned kernels are `backend/` code.
New work builds on this proposal instead of the per-operator workflow of
[0008](0008-kernel-tuning.md); its Humanize2 flow, evaluator timing protocol and
run archive are reused.

## 2. Motivation and current gap

- Tuned kernels never reach the model. 0008 exports verified sources, but
  "saved operators require a separate review/integration before runtime use"
  and there is no place to plug them in.
- There is no general dispatch point. Attention
  (`embodiinfer/layers/attention.py:74`) and quantized linear
  (`embodiinfer/layers/linear.py:253`) have registries; norm, activation and
  RoPE are chosen by per-model branches that import `embodiinfer.backend.triton`
  directly (`embodiinfer/policies/pi05/modeling_pi05.py:122`, `:161`, `:314`,
  `:669`; `embodiinfer/models/streamvln/backbone.py:12`;
  `embodiinfer/models/qwen25_vl/vision.py:47`).
- Isolated operators are the wrong unit. At B=1, four of five pi0.5 Triton
  kernels and all StreamVLN kernels already run near the ~4 µs device floor
  (0008, "Captured catalog workloads"). Remaining time sits in a few large
  kernels and in chains of small memory-bound kernels between GEMMs; only
  changing fusion boundaries reduces kernel count and intermediate traffic.
- Tuning everything is wasteful. Model discovery produced 94 pi0.5 tasks, most
  without measurable model-level effect.
- The comparison target is wrong. 0008 compares a candidate with the same ATen
  call or the existing Triton kernel. The goal is now for Triton to beat the
  model's Torch implementation, which in production may be compiled by Inductor
  (`_inference_helper`, `embodiinfer/policies/pi05/modeling_pi05.py:65`).

## 3. Goals and non-goals

Goals:

- One resumable pipeline per model and hardware: profile → select → tune →
  install → verify → benchmark. Only a new fusion region needs a code change,
  made once and merged like any other change.
- A hotspot report that attributes device time to model regions and estimates
  recoverable time; tuning only the selected regions.
- Regions may span several existing operators (new fusion boundaries).
- Baseline: the region's Torch implementation under the model's production
  settings (dtype, Inductor when configured, CUDA Graphs). Existing hand-written
  Triton kernels are optional starting points, not the baseline.
- Tuned kernels committed in the repository per hardware profile with
  provenance, selectable at runtime, reusable by others, and later promotable to
  the default Triton implementation through a normal review.

Non-goals:

- Making any tuned kernel the default (a separate change with evidence).
- Lower precision, quantization or changed model mathematics. A region may
  reorder floating-point work, but not change dtype or matmul precision.
- Automatic merging of agent-written policy changes; they are reviewed as PRs.
- Multi-GPU and batch > 1 tuning.
- Compatibility with the 0008 discovery/catalog task formats. Those commands
  stay as they are and are not extended.

## 4. Design

### 4.1 Operator registry (`embodiinfer/layers/ops.py`)

An operator is named by its mathematics (`ada_rms_norm`, `gated_residual`,
`add_rms_norm`, `rotate_qk`), never by a model. A policy-local fused region is
registered by its policy under a namespaced name (`pi05.residual_ada_rms_norm`).

Every existing Torch/Triton switch point is migrated to the registry once, as an
ordinary code change: the registry call becomes the model's default code path,
with today's implementations registered as `torch` and `triton`. Tuning runs
never edit these call sites; they only add implementations to a profile.

```python
register_op(op, impl, fn)                       # "torch" reference, "triton", ...
register_lazy_op(op, impl, module, function)    # import on first use
selection = KernelSelection(kernel_profile)     # one per policy
selection.prepare()                             # resolve everything eagerly
selection.resolve(op, prefer) -> Callable       # cached lookup
```

- Torch references live in `embodiinfer/backend/torch/ops.py`; the existing
  Triton kernels are registered unchanged.
- A policy builds its selection on first use on its device and drops it on
  refit, device or dtype change. `prepare()` resolves every operator before any
  `torch.compile` region, so graph tracing only reads the cache.
- Torch references run uncompiled. pi0.5's per-helper Inductor wrappers were
  first created while Dynamo traced the prefix encoder, which stores the raw
  function, so they never ran compiled outside that graph; keeping the
  references uncompiled preserves those numerics. Compiling them is a separate,
  measured change.
- `prefer` keeps the policy's current choice (`torch` or `triton`, from its
  existing flags). A selected profile overrides only operators it contains.
- A tuned callable is wrapped with its `applies(*args)` predicate (dtype,
  contiguity, shape divisibility, size limits). When it returns false the call
  goes to the preferred implementation. StreamVLN's `supports_*` checks
  (`embodiinfer/models/streamvln/backbone.py:125`) follow the same pattern.
- With no profile selected, resolution returns the current callables: behavior
  is bit-exact with the code before this change.

Runtime switch: `kernel_profile: none | auto | <name> | torch` in the policy
configuration (default `none`). `auto` picks the profile whose device name,
compute capability and Triton/Torch versions match the current process, else
`none` with one log line. An explicit name that does not match raises. `torch`
forces every registered operator to its Torch reference and is the baseline
configuration for benchmarks.

### 4.2 Hardware profiles (`embodiinfer/backend/triton/tuned/<profile>/`)

Each profile directory holds one module per tuned operator and `profile.json`:

- match keys: GPU name, compute capability, Torch/Triton/CUDA versions tested;
- per operator: module/function, `applies` predicate, covered workloads and
  weights, source digest, kernel-level and model-level gate results, the run id
  and the digest of its evidence archive;
- the model, checkpoint revision and configuration the profile was verified on.

Sources and this small manifest are tracked in Git. Evidence archives stay
outside the tracked tree, as with existing benchmark snapshots. Loading verifies
source digests, so a hand-edited kernel is reported instead of silently used.
Tuned modules are pure tensor functions and import nothing from `models`,
`policies` or `engine`.

### 4.3 Hotspot analysis (`scripts.kernel_tuning profile`)

Runs the model's inference driver in its production configuration on recorded
inputs, in two passes:

1. Timing: production settings, CUDA events, end-to-end and prefill/decode
   segments as the model benchmarks measure them.
2. Attribution: `torch.profiler` (CUPTI) with a `record_function` range around
   every registry call and every `nn.Module.forward`. Ranges are normalized
   across repeated layers (`layers.*`). When per-node kernels inside CUDA Graphs
   are not reported, the pass reruns without graphs and records that it did.

Per region the report gives device time, share of end-to-end time, calls, kernel
count, bytes moved and FLOPs from tensor metadata, and a lower bound
`max(bytes / measured bandwidth, FLOPs / measured peak)` from a device
microbenchmark. Recoverable time is `time - max(lower bound, floor × kernels)`.
Fusion candidates are adjacent producer→consumer chains of memory-bound regions;
their bound counts the eliminated intermediate traffic once. Outputs:
`hotspots.json` and `hotspots.md`.

### 4.4 Selection and new fusion regions

A Humanize2 selection flow gives the agent the hotspot report and the model
source. It returns `plan.json`: at most K regions (default 5), each either an
already registered operator or a proposed new fusion region, with reason and
estimated saving. Regions under 1% of end-to-end time are rejected. A person may
edit the plan or supply one with `--plan`.

A registered operator goes straight to tuning. A new fusion region does not
exist in the model yet, so it needs a one-time code change: the agent writes the
region's Torch reference (a composition of the existing operations), registers
it, and replaces the covered call sites with one registry call. The change is
accepted only if the model with `kernel_profile: none` is bit-exact against the
unchanged model on the evaluation inputs (§6). It then lands in the repository
as a normal reviewed change; from that point the region is an ordinary
registered operator for every hardware profile and later runs reuse it.

Workloads for every selected operator are captured by recording the registry
operator's real arguments and call counts during a production run.

### 4.5 Region tuning

A region task contains the Torch reference, the baseline (that reference under
the policy's compile settings, for example Inductor with the options used by
`_inference_helper`), an optional starting kernel, the captured workloads with
call-count weights, and frozen numerical bounds (§6). The 0008 Humanize2 flow,
CUDA Graph cold-L2 timing, paired promotion rule (≥ 3% weighted, ≤ 5% per
workload) and archive are reused. A candidate must also export `applies`, which
the evaluator checks is true on every workload. Expected model saving is
`Σ count × Δt`.

### 4.6 Install, verify, benchmark

`install` writes promoted sources and updates `profile.json`. It refuses an
operator that is not registered by the current source, so a fusion region's code
change must be merged first. The installed kernel applies only to the argument
layouts the evaluator measured (shape, dtype and strides; strides of size-one
dimensions and empty tensors are ignored) and otherwise falls back. Capture runs
eagerly, so a policy must give registry arguments the same layout with and
without `torch.compile`: on Jetson Thor the pi0.5 prefix K/V cache was a view of
the fused projection in eager mode but contiguous under Inductor, and a tuned
split-KV kernel was never selected until the cache was made contiguous in both.
`verify` runs the
model with `kernel_profile: <name>` against `kernel_profile: torch` on the
evaluation set and applies the model-level gate; an operator is kept only if the
whole profile passes. `benchmark` reports Torch, existing Triton and the profile
on the model's benchmark protocol, plus a per-operator ablation. The pipeline
command `optimize --model M --config C --profile P --agent A` runs all stages
with resume under `results/kernel_tuning/optimize/<id>/`.

### 4.7 Alternatives considered

- Add selection to the 0008 per-ATen discovery: it cannot express fusion and its
  baseline is the same ATen call, not the Torch implementation. Rejected.
- Rely on Inductor (`max-autotune`) for fusion: that is the baseline being beaten.
  It is reported as an extra benchmark row where it compiles.
- Keep tuned kernels outside the repository and download them: rejected, because
  the team wants reviewable, shareable sources that can become defaults.
- Uncached per-call resolution: rejected for launch-path overhead; resolution
  is a dictionary lookup after `prepare()` and is skipped entirely by CUDA Graph
  replay.
- Apply agent policy patches without review: rejected.

## 5. Model-agnosticism verdict

The registry, profile loader and `kernel_profile` switch are model-agnostic
`layers/` code: operators are named by mathematics and resolution uses only the
current device and versions. Tuned kernels are backend code. Region definitions,
call sites and policy-namespaced fused operators are policy-layer and optional;
with `kernel_profile: none` no path changes. The profiler uses module hooks and
registry ranges only; model-specific inputs come from the existing drivers.

## 6. Losslessness and precision criterion

Triton kernels reorder accumulation, so bit-exact output is not generally
achievable. This proposal requests explicit approval of calibrated tolerances,
following the frozen thresholds already used for the pi0.5 native path
(`benchmarks/pi05-benchmark/README.md`).

Exact gates:

- `kernel_profile: none` is bit-exact with the code before this change.
- Integrating a region with its Torch reference as default is bit-exact against
  the unpatched model (`max_i ||Δa_i||_inf = 0`).
- Precision conditions are identical: dtype, `float32_matmul_precision`, TF32
  flags and attention semantics.

Region level (during tuning): with FP64 reference `R` of the Torch reference,
Torch baseline output `B` and candidate `C` in the production dtype with epsilon
`u`, each floating output must satisfy `|C - R| <= atol + u·|R|` with
`atol = max(u·RMS(R), 2·max(|B - R| - u·|R|, 0), tiny)`, the 0008 formula with
the Torch implementation as `B`. Inputs are the captured fixtures plus seeded
random, zero and alternating-sign cancellation inputs. Integer, boolean, layout,
aliasing and mutation results are exact; non-finite values fail. In words: a
candidate may be at most about twice as wrong as Torch.

Model level (required for install):

- Continuous outputs (action chunks): compute an FP32 Torch reference from the
  same weights and inputs, and measure each run's max-abs and RMSE error per
  observation. Over the whole evaluation set, the candidate's mean max-abs and
  mean RMSE may exceed those of the `kernel_profile: torch` run by at most 25%,
  and its worst max-abs may be at most twice Torch's worst (floors `1e-4` and
  `1e-5`). Thresholds are frozen before any candidate runs. A per-observation
  bound of twice Torch's error was tried first and rejected: on Jetson Thor,
  40 pi0.5 observations, the existing Triton path failed it on 2 observations
  where Torch happened to land within 1e-3 of the reference, while its mean
  errors equalled Torch's and two Torch runs themselves differed per observation.
- Discrete outputs (generated tokens or discrete actions): sequences must equal
  the `kernel_profile: torch` run on every evaluation input. A difference fails;
  there is no automatic waiver.
- Evaluation set: the model benchmark's frozen cross-task references (pi0.5:
  the 10 references in `compare.py`) plus 200 benchmark frames.

Inductor's runtime kernel-configuration choice makes a fresh compilation of
the same Torch path produce different outputs: on Jetson Thor, two pi0.5
processes without a shared compile cache differed by up to 4.6e-2 in normalized
actions. Parity checks therefore run with `TORCHINDUCTOR_DETERMINISTIC=1`, which
made fresh compilations bit-exact, and model-level thresholds are derived from
the same compiled baseline that is timed.

The thresholds, reference identity and evaluation-set digest are saved in the
optimize run and summarized in `profile.json`; `scripts.kernel_tuning verify`
reproduces them.

## 7. Implementation plan

1. Registry and profiles: `embodiinfer/layers/ops.py`, profile loader,
   `kernel_profile` switch. One-time migration of the existing switch points to
   the registry as the default code path: pi0.5's `gated_residual`,
   `ada_rms_norm`, `gated_gelu` and `rotate_qk` (`split_kv_attention` already
   uses the attention registry and gains profile lookup), then the Qwen R2R and
   StreamVLN norm, activation and RoPE call sites. Install one kernel by hand and
   run `verify`/`benchmark` on an RTX 4090 to prove the loop.
2. `profile` command and hotspot report.
3. Tasks for registered operators with Torch baselines, tuning, automated
   `install`.
4. Agent selection and new fusion regions (one-time code changes, §4.4).
5. Further hardware profiles.

Status (2026-10-12): steps 1–3 are implemented for pi0.5; the loop was proven
on Jetson Thor instead of an RTX 4090 (§9), and the `thor` profile ships one
tuned operator. Qwen R2R and StreamVLN call sites, step 4 and an RTX 4090
profile are not done; neither is shape-range tuning (§10).

New tooling lives in `scripts/kernel_tuning/` and `benchmarks/kernel_tuning/`.
Existing 0008 commands keep working and are not extended. No public symbol is
removed.

## 8. Test plan

CPU: registry resolution and preference; `applies` fallback; profile matching
and mismatch handling for `auto` and explicit names; digest tamper detection;
install round trip; hotspot aggregation and fusion-chain detection on synthetic
profiler traces; plan validation; region and model gate arithmetic, including
frozen thresholds and discrete-output equality.

GPU (marked): pi0.5 with `kernel_profile: none` bit-exact against the previous
code; tuned and fallback paths inside CUDA Graph capture; the integration
bit-exact gate; `verify` on real weights.

## 9. Benchmark plan

pi0.5 on one RTX 4090, B=1, LIBERO-10 1,600 frames, BF16, Inductor and full
denoise-loop CUDA Graph, the protocol in `benchmarks/pi05-benchmark/README.md`.
Rows: `kernel_profile: torch`, current Triton, the tuned profile, and Inductor
`max-autotune` where it compiles; end-to-end mean/P50/P95 and prefill/decode
split; per-operator ablation. Expected gain is bounded by the recoverable time
in the hotspot report; it disappears where regions are already at the bandwidth
or launch floor. Batch > 1 and other hardware are reported only when measured.

First measurement (2026-10-12): Jetson Thor (sm_110, MAXN, `jetson_clocks`),
Torch 2.13.0+cu132, Triton 3.7.1, `pi05_libero_finetuned_v044`, B=1, BF16,
10 denoise steps, LIBERO-10 1,600 frames after 10 warmup calls, Inductor,
prefix and full-loop CUDA Graphs, native prefix/denoise attention.

| `kernel_profile` | Mean / P50 (ms) | Prefill (ms) | Decode (ms) |
|---|---|---|---|
| `torch` path (`native_inference: false`) | 179.9 / 179.9 | 63.9 | 111.4 |
| `none` (existing Triton) | 98.8 / 99.7 | 45.4 | 49.0 |
| `thor` | 87.6 / 85.4 | 45.2 | 37.2 |

The `thor` profile contains one operator, `split_kv_attention`, tuned by
`codex/gpt-6-astra:high` (run `20261012T084037Z-cf9243b720`, 12 attempts). Its
task timing (CUDA Graph, cold L2) against the Torch reference and the existing
Triton kernel:

| Workload (calls per step) | Torch (µs) | Existing Triton (µs) | Tuned (µs) |
|---|---|---|---|
| prefix, 576 keys (18) | 1,079 | ~178 | 139 |
| denoise, 576 + 50 keys (180) | 304 | ~120 | 38 |

Under the benchmark configuration every dispatch decision selected the tuned
kernel (72 prefix and 720 denoise calls executed before graph capture; replays
reuse them). `verify` on 40
recorded inputs passed (worst ratio 0.80 of the §6 limits). With
`kernel_profile: none`, 200 benchmark frames were bit-identical to the previous
code on both the native and the Torch path (`TORCHINDUCTOR_DETERMINISTIC=1`).

## 10. Risks and limitations

- Agent kernels are often shape-specialized (a 2026-10-07 `split_kv_attention`
  candidate asserted `P % PART_LENGTH == 0`); `applies` and fallback mitigate,
  but uncovered shapes run the old path. The shipped kernel covers the captured
  B=1 shapes with a 576-token prefix only; the 48-token text bucket, B > 1 and
  other checkpoints fall back. Follow-up: tasks over declared shape ranges
  (buckets, batch sizes, several layouts per shape), kernels with runtime strides
  and scales, and range-based `applies`.
- Region selection is manual: the hotspot report ranks time, but a person picks
  the operators. Follow-up: rank by `Σ count × (t − roofline floor)`, detect
  chains of memory-bound kernels between GEMMs as fusion candidates, and let an
  agent turn a candidate into a registered operator by PR, with a bit-exact
  `kernel_profile: none` parity test as the proof.
- GEMMs (the largest share of pi0.5 time on Thor) are not registry operators.
- Agent runs depend on a relay that drops streams; failed attempts consume the
  search budget.
- Profiles are tied to Triton/Torch versions; a version upgrade disables them
  under `auto` until re-verified.
- Attribution under CUDA Graphs depends on CUPTI support; the eager fallback pass
  can shift shares slightly.
- Finite evaluation sets do not prove correctness for every input or task
  success rate.
- Generated code adds review load; each install goes through a PR.
- Coding agents run commands without interactive approval, as in 0008.
