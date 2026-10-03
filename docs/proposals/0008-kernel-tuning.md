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
model-derived workload shapes, and the numerical contract. `generate` renders a
complete task package from an entry. The baseline is the production kernel
itself: the defining repository module and the repository modules it imports
are copied byte-for-byte under `vendor/` (absolute `embodiinfer.` imports are
rewritten to relative ones; package `__init__` files are never copied), and
`baseline.py` calls it as the engine does. `generated.json` records the
repository revision and source digests. Structured integer inputs such as
segment offsets are written as safetensors data inside the task.

The catalog covers the Triton paths of pi0.5 (`ada_rms_norm`, `gated_residual`,
`gated_gelu`, `rotate_qk`, `split_kv_attention`), ActiveVLN's Qwen2.5-VL vision
tower (`rotate_half_rope`, `segmented_attention`), and StreamVLN's Qwen2 decode
path (`rms_norm`, `add_rms_norm`, `swiglu`). Shapes are representative of those
models, not traced traffic. All generated tasks time with CUDA Graphs.

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

`tune-all` generates the selected operators into a new batch directory under
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
