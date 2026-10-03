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

## 11. 自动化调优使用说明

### 准备任务包

每个算子使用独立、自包含的任务目录。可复制
`benchmarks/kernel_tuning/tasks/gated_residual/`，再替换以下五个必需文件：

| 文件 | 人工提供的内容 |
|---|---|
| `README.md` | 调用场景、负载来源、布局与特化限制、基线来源及精度约定的理由。 |
| `definition.json` | 输入输出名称、顺序、shape、dtype、约束和可执行的数学参考 `run`。 |
| `workloads.jsonl` | 每行一个未评测的 Trace，包含唯一 UUID、实际 shape 和输入来源。 |
| `baseline.py` | 要超越的当前实现，入口为 `run`；参数顺序与 definition 的 inputs 一致。 |
| `tuning.yaml` | 语言、agent、评测 Python、设备、精度、计时、晋级规则和搜索预算。 |

例如 RMSNorm 需要先确定普通、残差融合或自适应版本、`eps`、
`weight` 或 `1 + weight`、归约精度和中间舍入顺序，再提供目标部署的
batch/token/hidden 负载。正确性参考与性能基线承担不同职责，不能只给出算子名称。
输入支持随机生成、标量和任务目录内的 safetensors；本地 helper 也应随任务保存。
非连续输入、原地修改或特殊数据生成/硬件语义需要任务自己的 `benchmark.py`，
接口见上面的 Task-owned benchmark interface。

### 准备环境并启动

真实调优在 Linux/POSIX 目标机器上执行。先准备兼容该 GPU 的运行环境，
将 `evaluator_python` 设置为该环境的绝对 Python 路径，安装评测依赖；
独立安装并登录 coding-agent CLI。账号凭据保留在 CLI 的配置中，不写入任务包。
Humanize2 工具环境和安装步骤见 `CONTRIBUTING.md` 的 Optional kernel tuning tools。
Thor 应使用 Thor 兼容的 GPU 软件栈，不能直接套用其他显卡的依赖版本。

以下命令从仓库根目录执行。把任务路径替换为自己的任务，agent 占位符替换为
已配置的实际 harness/model/effort：

```bash
uv sync --project scripts/kernel_tuning --python 3.12 --frozen
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning check benchmarks/kernel_tuning/tasks/gated_residual
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning check benchmarks/kernel_tuning/tasks/gated_residual --environment
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning run benchmarks/kernel_tuning/tasks/gated_residual --agent 'HARNESS/MODEL:EFFORT'
```

`check` 只做结构检查；`check --environment` 额外探测依赖与硬件。
`run` 和 `resume` 才会调用真实 agent 并执行编译、正确性检查与 GPU 测量。

### 自动循环与停止条件

1. Controller 冻结任务与工具源码，记录评测环境，检查初始基线。
2. Humanize2 调用 coding agent 读取任务、当前最优实现和反馈，先写 `PLAN.md`。
3. Agent 根据计划生成 Triton/CUDA 源码和完整的官方 `solution.json`，只做静态检查。
4. 独立 evaluator 编译候选；使用相同种子对参考、基线、当前最优和候选进行校验，
   再对基线、当前最优和候选进行配对计时。
5. Controller 根据固定门槛决定是否晋级，保存本轮源码和证据，把反馈交给下一轮。

默认精度为逐字节一致。允许浮点误差时，人工须事先明确设置
`precision.mode: tolerance`、`atol`、`rtol` 和 `reason`，agent 不能修改这些条件。
默认使用两个配对测量轮次、等权负载；每轮平均延迟须同时相对初始基线和当前
最优改善至少 3%，且任何单个负载的回退不超过 5%。显式 `weights` 必须覆盖全部 UUID。

默认最多生成 20 个候选、累计执行两小时，或连续五次失败/未改善后停止；
每轮 agent 和评测也有独立超时。这些上限均在 `tuning.yaml` 中配置。
失败和中断尝试保留并计入预算；耗尽预算后须建立新任务/run，不能修改冻结任务继续搜索。

### 查看、续跑与保存结果

`run` 会输出新建的 `results/kernel_tuning/RUN_ID/` 路径。该目录保存任务和工具快照、
环境信息、所有候选源码/计划、评测请求及原始结果、晋级决定和 Humanize2 编排记录。
只有候选通过晋级门槛后才生成 `best.json`；没有改善的 run 也会完整保存。

```bash
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning status results/kernel_tuning/RUN_ID
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning resume results/kernel_tuning/RUN_ID
uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning export results/kernel_tuning/RUN_ID results/exported_operator
```

续跑要求任务、工具、编排环境和评测环境身份一致，并有剩余预算。
`status` 与 `export` 不调用 agent；导出包包含最优算子的 Solution、完整源码、冻结任务、
工具/锁文件、原始基线、胜出证据和 SHA256 清单。结果位于 Git 忽略的 `results/`，
需要另行保存。导出不会自动注册到推理 runtime，实际集成仍需代码审查和模型验证。
