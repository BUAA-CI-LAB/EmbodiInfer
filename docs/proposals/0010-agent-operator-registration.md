# 0010 — 智能体驱动的算子注册流程

- Status: Draft
- Author: EmbodiInfer maintainers
- Date: 2026-10-13

## 1. 概要

本提案实现 [0009](0009-hotspot-kernel-tuning.md) §4.4 和实现计划第 4 步：调用方选定一个尚未进入算子注册表的算子或融合区域后，由编码智能体完成以下改动：

- 写出它的 Torch 参考实现；
- 在 `embodiinfer/layers/ops.py` 中注册；
- 把模型调用点改为经过注册表；
- 补齐测试。

控制器随后依次运行静态检查、CPU 测试、命中检查和 `kernel_profile: none` 逐位一致检查。任何一项失败，都把结构化反馈交回智能体修改，直到全部通过或预算用尽。最终产物是一个可以直接提交审阅的分支和证据包。

整个流程位于 `scripts/kernel_tuning/registration/`，以 `python -m scripts.kernel_tuning register …` 调用，属于开发者工具，不进入运行时。它只负责把算子变成"可调优、可安装"的注册算子，不做调优，也不自动合并。注册代码经人工审阅合并后，再进入 0009 已实现的调优流程（`capture-ops` → `calibrate` → `tune-all`/`run` → `install` → `verify`）。

## 2. 动机与现状缺口

- **只有注册算子能调优并接入模型。** `capture-ops` 只记录经过 `KernelSelection.resolve` 的调用（`scripts/kernel_tuning/regions.py:152-163`）。`install` 拒绝其他任务，也拒绝当前源码里没有注册的算子（`scripts/kernel_tuning/install.py:140-144`）。
- **注册目前完全靠人工。** pi0.5 的 5 个算子都是在 commit `449ebe2` 中手工迁移的，每个算子要做四件事：
  1. 把 Torch 实现搬到 `embodiinfer/backend/torch/ops.py`；
  2. 在 `embodiinfer/layers/ops.py:160` 注册；
  3. 把调用点改为 `_op(kernels, name, use_triton)(...)`（`embodiinfer/policies/pi05/modeling_pi05.py:88`），并把 `self._kernels` 传下去；
  4. 在 Thor 上验证 `kernel_profile: none` 逐位一致。
- **最大的热点在注册表之外。** 0009 §10 记录，Thor 上 pi0.5 耗时最多的是 GEMM，它们不在注册表里，新流程完全碰不到。0009 设想由智能体把热点区域写成新的注册算子并提 PR（§4.4），但这一步没有实现。
- **改调用点的风险很具体。** 仓库多处强调形状和表达式会改变舍入：
  - 合并 Q/K/V 会改变归约形状，从而改变舍入（`modeling_pi05.py:664-666`）；
  - `F.linear` 和 `x @ w.T` 会走不同的 ATen 分解；
  - eager 与 Inductor 下参数布局可能不同（commit `ecde612`）。

  这些都需要固定的门禁来兜底，不能依赖智能体自觉。

## 3. 目标与非目标

目标：

- 调用方显式选择要注册的算子：给出算子名、种类（包装或融合）和调用点。调用点可以直接写源码位置，也可以从热点报告挑选区域，再由工具解析成源码位置。
- 一条可恢复的流水线：选择 → 定位调用点 → 记录基线 → 智能体修改 → 静态门禁 → GPU 门禁 → 反馈循环 → 打包。每一步都有明确的产物和通过条件。
- 尽量复用现有工具：`profile`、`capture-ops`、`verify` 的输出比较逻辑、Humanize2 flow、`run_lock`/`atomic_json`。只为现有工具做不到的事新增代码。
- 验收由控制器负责。智能体写的测试只是补充，判定依据是控制器对比基线版本的逐位一致结果。

非目标：

- 调优或安装 kernel（属于 0009 的 B、C 部分）。
- 改变数学、精度或舍入。注册改动必须在 `kernel_profile: none` 下与基线逐位一致。
- 自动合并、自动推送或自动开 PR。默认只生成分支和 PR 描述；提交、推送、开 PR 都需要调用方显式传参。
- 修改 `KernelSelection`、调优档案（tuned profile）、`scripts/` 或依赖。这类改动走普通 PR。
- Qwen R2R、StreamVLN 的首次接入。它们还没有 `kernels` 传参链，第一次需要人工建立；建立之后同样可以使用本流程。

## 4. 设计

### 4.1 总览

```
 调用方                 控制器（工具环境）                     智能体（Humanize2）     模型运行时（GPU 子进程）
   │  R0 candidates ──► 读 hotspots.json，列出未覆盖热点
   │  R1 new ─────────► 校验规格、定位调用点、建 worktree、打包上下文
   │  ◄── sites.md 等待确认
   │  R2 run ─────────► 基线录制 ─────────────────────────────────────────────────► snapshot ×2、trace-ops
   │                   R3 ───────────────────────────────► plan / implement
   │                   R4 静态门禁（范围、结构、ruff、pytest）
   │                   R5 GPU 门禁 ─────────────────────────────────────────────► trace-ops、capture-ops、
   │                   │                                                         snapshot、audit-layouts
   │                   R6 失败 → feedback.json ──────────► repair（同一会话）
   │                   R7 全部通过 → 证据、可选覆盖率/测速、PR.md
   │  finalize ───────► （显式 --commit/--push）提交到 registry/<op>-<id> 分支
   │  R8 人工审阅、合并 → 进入 0009 的调优流程
```

| 步骤 | 工作 | 执行者 | 现有工具 | 产物 |
|---|---|---|---|---|
| R0 | 列出未被注册表覆盖的热点 | 控制器 | 读取 `profile` 的 `hotspots.json` | `candidates.md` |
| R1 | 规格校验、调用点定位、worktree、上下文 | 控制器 | `git worktree`；运行时内 `available_ops()` | `spec.json`、`sites.json`、`context/` |
| R2 | 基线输出与 ATen 序列，确定性预检 | 运行时子进程 | 复用 `verify` 的取输出逻辑（抽成 `snapshot`） | `golden/`、`trace/base.jsonl` |
| R3 | 写计划、改代码、写测试 | 智能体 | Humanize2 flow 机制（同 `flow.py`） | worktree 改动、`REGISTRATION.json` |
| R4 | 范围、结构、ruff、pytest | 控制器 | `ruff`、`pytest` | `attempts/N/static.json` |
| R5 | ATen 序列、命中、逐位一致、布局审计 | 运行时子进程 | `capture-ops`（现有）；`snapshot`、`trace-ops`、`audit-layouts`（新增） | `attempts/N/gpu.json` |
| R6 | 汇总失败交回智能体 | 控制器 + 智能体 | 同一 Humanize2 会话 | `attempts/N/feedback.json` |
| R7 | 终检、证据、可选覆盖率和测速、PR 描述 | 控制器 | `profile`（现有）、`benchmarks/pi05-benchmark/benchmark.py`（现有） | `evidence.json`、`PR.md`、`patch.diff` |
| R8 | 审阅与合并 | 人 | GitHub | 合并后的注册代码 |

### 4.2 调用方式：选择算子

所有命令都在目标 GPU 机器上运行，入口是工具环境（Python ≥ 3.12 + Humanize2，与 `tune-all` 相同）。加载模型的步骤通过 `--set runtime_python=…` 指定的模型运行时解释器在子进程里执行，与 `tune-all --set evaluator_python=…` 的做法一致。

```bash
TOOL="uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning"

# R0（可选）：看看哪些热点不在注册表里
$TOOL register candidates --hotspots HOT/hotspots.json --min-share 0.01

# R1：为选定的算子建立一次运行（规格文件和命令行参数二选一）
$TOOL register new --spec linear.yaml --output results/kernel_tuning/register
$TOOL register new --model pi05 --op linear --kind wrap \
    --site 'embodiinfer/policies/pi05/modeling_pi05.py::_fused_qkv::F.linear(x, weight)' \
    --region 'stage:denoise_step/module:expert.layers.*.mlp.down_proj' --hotspots HOT/hotspots.json \
    --parity-config parity.yaml --output results/kernel_tuning/register
#   命令结束时打印 sites.md，由调用方确认或编辑 sites.json；--yes 跳过确认。

# R2–R7：运行循环；--until 可以在某个阶段后停下
$TOOL register run RUN --agent 'codex/MODEL:EFFORT' \
    --set runtime_python=/abs/runtime/python --set test_python=/abs/dev/python \
    [--until baseline|static|parity|package]
$TOOL register resume RUN
$TOOL register status RUN

# 收尾：只有显式传参才会提交或推送
$TOOL register finalize RUN [--commit] [--push]
```

规格文件（`linear.yaml`）示例：

```yaml
model: pi05
op: linear              # 新算子名：Python 标识符、尚未注册、按数学含义命名
kind: wrap              # wrap：把一处已有表达式包成算子；fuse：把相邻几步合成一个算子
description: >
  F.linear(x, weight, bias)：weight 为 [out, in]，bias 可为 None；不改变 dtype。
sites:                  # 调用点，三种写法可以混用
  - source: embodiinfer/policies/pi05/modeling_pi05.py
    function: _fused_qkv
    expression: "F.linear(x, weight)"
  - source: embodiinfer/policies/pi05/modeling_pi05.py
    function: _mlp
    expression: "mlp.down_proj(activated)"
  - region: "stage:denoise_step/module:expert.layers.*.self_attn.o_proj"   # 从 hotspots.json 选
preserve:               # 必须保持原样的分支；R1 会自动检测并补充候选
  - "QuantizedLinear：继续走 layers/linear.py 的量化注册表"
  - "张量并行：RowParallelLinear / ColumnParallelLinear 保持模块调用"
  - "torch.is_grad_enabled() 或 training 时的原始调用"
parity:                 # GPU 逐位一致检查的配置；每个配置独占一个进程和一份编译缓存
  checkpoint: /ckpt/pi05_libero_finetuned_v044
  inputs: /data/pi05-inputs.pt          # discovery.drive 的录制输入
  samples: 40                           # 循环里使用的输入数
  final_inputs: /data/pi05-frames.pt    # R7 终检使用，建议 ≥ 200 帧
  configs:
    eager:    {adapter_config: pi05-eager.json,    dtype: bfloat16, num_steps: 10, no_cuda_graph: true}
    native:   {adapter_config: pi05-native.json,   dtype: bfloat16, num_steps: 10, capture_full_loop: true}
    inductor: {adapter_config: pi05-inductor.json, dtype: bfloat16, num_steps: 10, capture_full_loop: true}
    fp32:     {adapter_config: pi05-native.json,   dtype: float32,  num_steps: 10, no_cuda_graph: true}
tests:                  # 本模型相关的 CPU 测试；不写时使用模型默认表
  - tests/test_kernel_ops.py
  - tests/test_pi05_components.py
  - tests/test_pi05_cpu_routes.py
  - tests/test_tensor_parallel.py
budget:
  max_attempts: 5
  agent_timeout_seconds: 1800
  max_seconds: 14400
```

`parity.configs` 中的每一项展开为 `scripts.kernel_tuning.discovery.drive` 的参数（`--adapter-config`、`--dtype`、`--num-steps`、`--capture-full-loop`、`--no-cuda-graph` 等），因此可以覆盖策略的每条执行路径。对 pi0.5 来说，`adapter_config` 里的 `policy_kwargs` 决定走哪条路径，相关开关有 `attention`、`compile_backend`、`native_inference`、`prefix_cuda_graph`、`denoise_attention`、`prefix_attention`（`modeling_pi05.py:414-430`）。调用点处在哪条路径上，就必须有一个配置走到它。例如 `attention: eager` 走的是分开的 Q/K/V GEMM（`modeling_pi05.py:664-668`），融合投影的改动不会经过它。

每个模型在 `scripts/kernel_tuning/registration/models/<model>.yaml` 里提供默认值：CPU 测试表、常见的 preserve 模式（例如 `QuantizedLinear`、TP 线性层）、parity 配置模板，以及作为示例的迁移 commit（pi0.5 是 `449ebe2`）。规格文件可以覆盖其中任意一项。

### 4.3 运行目录与状态

```
results/kernel_tuning/register/<run_id>/
  manifest.json        状态、各阶段结果、尝试次数、耗时；atomic_json 写入，run_lock 加锁
  spec.json            冻结的规格（含基线 commit、模型、parity 配置的摘要）
  sites.json / sites.md  已确认的调用点清单
  context/             提供给智能体的只读上下文
  base/                基线 commit 的 git worktree（detached），只供基线录制使用
  worktree/            候选分支 registry/<op>-<run_id> 的 git worktree，智能体只能改这里
  golden/<cfg>-{1,2}.pt  基线输出（两次，用于确定性预检）
  trace/base.jsonl     基线 eager 推理的 ATen 调用序列
  attempts/<N>/        每次尝试的 prompt、回复、PLAN.md、patch.diff、static.json、gpu.json、feedback.json
  cache/               每个子进程独立的 TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR
  evidence.json  PR.md  patch.diff
```

`resume` 从 `manifest.json` 记录的阶段继续，已经通过的阶段不会重跑。规格或基线 commit 发生变化时必须新建一次运行，不允许在原运行上续跑。

### 4.4 各步骤详细说明

#### R0 候选列表（可选，不调用智能体）

- **工作：** 读取 `profile` 的输出，列出没有被 `op:` 范围覆盖的区域，按 `us_per_step` 排序，并标明各区域的定位精度：
  - `module:<path>` 区域可以映射到源码，比如 `expert.layers.*.mlp.down_proj` → 属性 `mlp.down_proj`；
  - `kernels:gemm`、`kernels:inductor` 这类按 kernel 名分类的区域，来自编译区域或无标签的图回放，无法自动映射到源码，只能由调用方手写 `site`。
- **复用的现有工具：** `profile` 命令和它生成的 `hotspots.json`，字段见 `scripts/kernel_tuning/hotspots.py:249-296`。
  ```bash
  /abs/runtime/python -m scripts.kernel_tuning profile --output HOT -- \
      -m scripts.kernel_tuning.discovery.drive --policy pi05 --checkpoint CKPT \
      --adapter-config pi05-native.json --inputs pi05-inputs.pt --dtype bfloat16 \
      --num-steps 10 --capture-full-loop
  ```
- **新增：** `registration/candidates.py`，只读 JSON。
- **说明：** 这一步只提供参考，选哪个算子仍由调用方决定。按 roofline 下限估算可回收时间是 0009 的后续工作，本步骤不实现。

#### R1 建立运行（`register new`，不调用智能体，不需要 GPU）

1. **规格校验**
   - `op` 必须是 Python 标识符。现有工具链要求参考实现函数名等于算子名，并在 `embodiinfer.backend.torch.ops` 中（`scripts/kernel_tuning/regions.py:215-217`、`scripts/kernel_tuning/discovery/tasks.py:95-98`）；安装时还用算子名作包目录名（`install.py:177`）。
   - `op` 不能已经注册：在运行时子进程里执行 `from embodiinfer.layers import available_ops`，检查 `op` 不在结果里；同时检查 `backend/torch/ops.py` 没有同名函数。
   - 模型名必须能被 `embodiinfer.policies.factory` 构造。
   - 主检出目录必须干净（`git status --porcelain` 为空），然后记录 `base = git rev-parse HEAD`。
2. **定位调用点**（新增 `registration/sites.py`，基于 `ast`）
   - `source/function/expression`：在函数体内查找与 `expression` 的 `ast.unparse` 规范化形式相同的表达式，记录文件、行号、列号。
   - `region`：取 `module:` 段的最后一级属性名（如 `down_proj`），在该模型策略源码中查找形如 `<…>.down_proj(<…>)` 的调用，列为候选。这只是启发式匹配，必须由调用方确认。
   - 对每个调用点收集包裹它的 `if` 条件，例如 `isinstance(..., QuantizedLinear)`、`torch.is_grad_enabled()`、`self.attention == "eager"`、`self.tensor_parallel.enabled`，作为 preserve 候选。再结合模型默认表，标出模块可能在运行时被替换的位置，例如 TP 会把 `o_proj`/`down_proj` 换成 `RowParallelLinear`（`embodiinfer/engine/parallel/pi05.py:48,52`）。
   - 同时检查调用点所在函数能否拿到 `KernelSelection`：方法里可以用 `self._kernels`，自由函数则需要新增 `kernels` 参数并沿调用链传递。
3. **工作区**
   ```bash
   git worktree add --detach RUN/base BASE
   git worktree add -b registry/<op>-<run_id> RUN/worktree BASE
   ```
   主检出目录在整个过程中保持不动。
4. **上下文包 `context/`**（只读）：
   - `AGENTS.md`；
   - `layers/ops.py`、`backend/torch/ops.py`、各调用点所在文件；
   - `tests/test_kernel_ops.py`，作为测试写法示例；
   - 示例迁移 commit 的 diff（`git show 449ebe2 -- embodiinfer/`）；
   - 0009 §4.1、§6 的摘录；
   - 第 4.5 节的规则全文；
   - `sites.json`。
5. **输出与确认：** 写出 `sites.md`（每个调用点附源码片段和 preserve 条件），然后退出，等调用方确认或修改 `sites.json` 后再执行 `register run`。规格里出现 `region` 时不允许 `--yes` 跳过确认。

#### R2 基线录制（GPU；运行时子进程，在 `base/` worktree 中执行）

- **工作：**
  1. **确定性预检。** 每个 parity 配置用两份全新的编译缓存各跑一次，比较两次输出。只要有任何差异就停止，报告"该配置的基线不可复现，无法做逐位一致检查"。0009 §6 记录过：不加 `TORCHINDUCTOR_DETERMINISTIC=1` 时，Inductor 两次全新编译的结果可能不同。
  2. **保存基线输出** `golden/<cfg>-1.pt`。
  3. **记录 ATen 调用序列**：用一个只走 eager 路径的配置，对 1 个输入记录完整的 ATen 调用序列，每条记录包含算子名，以及每个张量参数的形状、步长、dtype 和设备。
- **复用现有工具：** `verify` 里按 `kernel_profile` 收集输出的逻辑。`scripts/kernel_tuning/verify.py:181-245` 的 `_run` 已经为每个输入固定随机种子（`seed + index`），能构造 `EngineCore` 或 serving adapter，并用 `flatten_outputs` 展开输出。把它抽成公共函数 `collect_outputs(...)`，`verify` 和新命令共用，`verify` 的行为不变。
- **新增的运行时命令：**
  ```bash
  # 在 RUN/base 中执行
  env PYTHONPATH=RUN/base TORCHINDUCTOR_DETERMINISTIC=1 \
      TORCHINDUCTOR_CACHE_DIR=RUN/cache/base-inductor-1 TRITON_CACHE_DIR=RUN/cache/base-inductor-1-triton \
      /abs/runtime/python -m scripts.kernel_tuning snapshot --profile none --samples 40 \
      --output RUN/golden/inductor-1.pt -- \
      --policy pi05 --checkpoint CKPT --adapter-config pi05-inductor.json \
      --inputs pi05-inputs.pt --dtype bfloat16 --num-steps 10 --capture-full-loop

  env PYTHONPATH=RUN/base /abs/runtime/python -m scripts.kernel_tuning trace-ops \
      --samples 1 --output RUN/trace/base.jsonl -- <eager 配置的 drive 参数>
  ```
  - `snapshot`：保存每个输入的 `flatten_outputs` 结果，以及 `embodiinfer.__file__`、源码树哈希和环境信息（GPU、Torch、Triton、CUDA、`torch.get_float32_matmul_precision()`）。
  - `trace-ops`：沿用 `scripts/kernel_tuning/discovery/recorder.py:121-149` 的 `TorchDispatchMode` 观察方式，但只记录调用序列，不保存夹具。
- **导入路径校验：** 运行时环境里 EmbodiInfer 是可编辑安装，指向主检出目录。所以每个子进程都要把 `PYTHONPATH` 设为对应 worktree，让它排在 site-packages 的 `.pth` 之前；控制器再检查输出里记录的 `embodiinfer.__file__` 确实位于该 worktree 下。不满足就直接失败，否则比较的其实是同一份代码，门禁就失效了。

#### R3 智能体修改（Humanize2 flow `operator_registration`）

- **机制：** 与 `scripts/kernel_tuning/flow.py` 相同：一个可恢复的 `@flow`，一个 `coder` 智能体，工作环境是 `LocalEnv` 指向的 `worktree/`，单次超时用 `Budget`，`ctx.state` 中保存进度台账（ledger）。`__main__._execute` 中启动 `Hmz(...).run(flow=..., agents={"coder": AGENT}, params={"run_dir": RUN})` 的方式原样复用。不同之处：
  - 本流程追求得到一个正确的补丁，而不是在多个候选里挑最好的一个，所以第一个通过全部门禁的修改即为结果；
  - 修改在同一个 worktree 上逐步进行，失败后在同一会话里修复（见 R6）。
- **第一轮 `plan`：** 智能体阅读 `context/` 和 `sites.json`，写出 `PLAN.md`，内容包括：
  1. 算子的数学定义，以及它与原表达式逐个 ATen 调用的对应关系；
  2. 函数签名，即哪些是张量参数、哪些是常量；
  3. 每个调用点的改法，以及 `kernels` 参数如何传到该点；
  4. 每个 preserve 分支如何保持不变；
  5. 新测试清单；
  6. 可能破坏逐位一致的风险点。
- **第二轮 `implement`：** 在 `worktree/` 中按第 4.5 节的规则修改代码和测试，并写出 `REGISTRATION.json`：
  ```json
  {
    "op": "linear",
    "reference": "embodiinfer/backend/torch/ops.py::linear",
    "signature": {"tensors": ["x", "weight", "bias?"], "constants": []},
    "sites": [{"id": "S1", "file": "embodiinfer/policies/pi05/modeling_pi05.py", "function": "_fused_qkv",
               "line": 255, "call": "_op(kernels, \"linear\", False)(x, weight)"}],
    "preserved": [{"site": "S1", "branch": "isinstance(projection, QuantizedLinear)", "reason": "量化注册表"}],
    "expected_calls_per_inference": {"S1": 216},
    "tests": ["tests/test_kernel_ops.py::test_linear_reference_matches_f_linear"]
  }
  ```
  `expected_calls_per_inference` 是智能体声明的每次推理调用次数，由 R5 的命中检查核对。上例按 2 个 tower × 18 层，加上专家 tower 每个去噪步的调用估算，仅作示意。
- **允许的操作：** 运行 `ruff` 和 CPU `pytest`。
- **禁止的操作：** 运行 GPU 代码、`git commit`、安装依赖、修改 `context/` 或运行目录。

#### R4 静态门禁（控制器；CPU；按顺序执行，前一项失败时后面不再执行）

| 门禁 | 检查内容 | 方式 |
|---|---|---|
| G1 范围 | `git -C worktree diff --name-status BASE` 只涉及允许修改的文件：`embodiinfer/backend/torch/ops.py`、`embodiinfer/layers/ops.py`、`sites.json` 涉及的源文件、`tests/test_kernel_ops.py`、`tests/test_registry_<op>.py`。已有测试只能新增，不能删除或修改断言行。`pyproject.toml`、`uv.lock`、`scripts/`、`embodiinfer/backend/triton/tuned/` 一律不能改 | 新增 `registration/workspace.py` |
| G2 结构 | 1. `backend/torch/ops.py` 中存在 `def <op>`，有类型注解和 docstring，且只导入 `torch` 和标准库，不依赖 `models`/`policies`/`engine`；<br>2. `layers/ops.py` 用 `register_lazy_op("<op>", "torch", "embodiinfer.backend.torch.ops", "<op>")` 注册；<br>3. 每个调用点改为注册表调用，原表达式只出现在 preserve 分支里；<br>4. 在运行时子进程中：`available_ops()["<op>"]` 包含 `torch`，`get_op("<op>") is backend.torch.ops.<op>`，`KernelSelection("none").prepare()` 成功，`KernelSelection("torch").resolve("<op>")` 返回 Torch 参考实现 | `ast` 检查 + 运行时子进程 |
| G3 格式 | 改动文件通过 `ruff check` 和 `ruff format --check` | 现有 `ruff` |
| G4 测试 | 规格中的 `tests` 和 `REGISTRATION.json` 列出的新测试全部通过。另外，把新测试放到 `base/` 上运行，必须失败，用来证明这些测试确实依赖新算子 | `test_python -m pytest … -q` |

智能体写的测试至少要覆盖三类：

1. 参考实现与原表达式在 CPU 随机输入（FP32 和 BF16）上 `torch.equal`；
2. 用假调优档案确认调用点经过注册表，写法见 `tests/test_kernel_ops.py:135-150`；
3. preserve 分支（量化、张量并行、训练和求梯度）不经过注册表。

这些测试只是补充。验收依据是 R5 中由控制器执行、对比基线版本的检查，智能体无法通过改测试绕过它。

#### R5 GPU 门禁（运行时子进程，在 `worktree/` 中执行；按开销从低到高排列）

| 门禁 | 检查内容 | 调用方式 |
|---|---|---|
| G5 ATen 序列 | 候选与基线的 eager ATen 调用序列完全相同，包括算子名、形状、步长、dtype 和顺序。失败时报告第一个分歧点及前后 5 条记录 | 新增 `trace-ops`，与 R2 参数相同，`PYTHONPATH=RUN/worktree` |
| G6 命中 | 新算子被记录到；`errors` 为空；每个调用点的调用次数等于 `expected_calls_per_inference`，且不为 0 | 现有 `capture-ops`，加一个新的小选项 `--record-callers`，按调用者的文件和行号计数：<br>`python -m scripts.kernel_tuning capture-ops --model pi05 --ops linear --warmup 1 --steps 2 --record-callers --output RUN/attempts/N/capture -- -m scripts.kernel_tuning.discovery.drive <eager 参数> --no-cuda-graph` |
| G7 逐位一致 | 对每个 parity 配置，候选在 `kernel_profile: none` 下的输出与 `golden/<cfg>-1.pt` 完全相等（见第 6 节） | 新增 `snapshot` 和 `snapshot-compare`，环境变量与 R2 相同，使用新的缓存目录 |
| G8 布局审计（只警告） | 在生产配置下，比较 Python 实际执行（未编译、未在图捕获中）的注册调用的参数布局，与 G6 在 eager 下记录的布局是否一致。比较前按 `install.py` 的规则规范化：忽略长度为 1 的维度和空张量的步长。只在编译区域内执行的调用记为"无法观测" | 新增 `audit-layouts`，复用 `regions.py` 包装 `resolve` 的方式，只记录布局 |

G5 不是必需的，但能把"某个输出差了多少"细化成"第 N 个 ATen 调用从 `aten.linear` 变成了 `aten.matmul`"。它的开销只有一次 eager 推理，所以放在最前面执行。

G8 只发警告，因为布局不一致不影响 `kernel_profile: none` 的正确性，但会让之后安装的调优 kernel 在生产中一次也选不中（commit `ecde612` 的情况）。警告会进入反馈。智能体可以把参数规范成连续布局等形式，但前提是 G5 和 G7 仍然通过；否则这个警告写进 PR 描述，由审阅者决定。

#### R6 反馈循环

- 任何门禁失败时，控制器写出 `attempts/N/feedback.json`：
  ```json
  {"gate": "G7", "config": "inductor", "summary": "输出与基线不一致",
   "details": {"input": 3, "output": ".actions", "max_abs": 0.0078125, "mismatched": 42, "total": 350},
   "hints": ["G5 已通过：eager 序列相同，差异只出现在 Inductor 配置，检查编译区域内的参数布局或 dtype"]}
  ```
  其他门禁的 `details` 分别是：G1 的违规文件列表、G2 失败的断言、G3/G4 的输出摘要（截断到固定长度）、G5 的序列差异、G6 的调用次数对比。
- 下一次尝试在同一个 Humanize2 会话中发出 `repair` prompt，附上 `feedback.json`，智能体在原 worktree 上继续修改。如果连续两次在同一门禁失败，就新开一个会话重新规划，worktree 保留。
- 每次尝试结束时，把 `git -C worktree diff BASE` 存档为 `attempts/N/patch.diff`。
- 停止条件：
  - 全部门禁通过，进入 R7；
  - 达到 `max_attempts` 或 `max_seconds`，状态为 `exhausted`；
  - R2 确定性预检失败，或运行环境发生变化（GPU、Torch、Triton 版本，或基线 worktree 被改动），状态为 `failed`。这类问题不是智能体能修复的。

#### R7 终检与打包

1. **终检：** 用 `parity.final_inputs`（建议 ≥ 200 帧）对每个配置重跑 R2 基线和 G7，全部逐位一致才继续。循环中只用 `samples` 个输入，是为了控制每轮的 GPU 时间。
2. **覆盖率（可选，`--coverage`）：** 用现有 `profile` 命令在生产配置下分别跑基线和候选，报告时间从哪些 `module:` 区域移到了 `op:<op>` 区域，以及注册表覆盖的时间占比。编译区域内的调用不会有 `op:` 标签（`hotspots.py` 的模块文档字符串里有说明），报告会注明这一点。
3. **测速（可选，`--bench`）：** 用 `benchmarks/pi05-benchmark/benchmark.py`，在相同配置和 `kernel_profile: none` 下，分别跑基线和候选。结果只报告、不卡门禁。`_op` 的分派开销只影响 eager 调用，图回放不受影响；commit `449ebe2` 测得 native 路径是 99.35 ms 对 98.47 ms。
4. **证据 `evidence.json`：** 基线 commit；候选源码树哈希；规格摘要；每个门禁的结果；parity 配置、输入摘要和数量；环境信息；尝试次数和耗时。
5. **`PR.md`：** 按 `.github/PULL_REQUEST_TEMPLATE.md` 的结构填写：
   - Summary：算子定义、改动的调用点、保持不变的分支；
   - Type of change：勾选 "Kernel, layer, or numerical change"；
   - Verification：实际运行的命令和结果，以及未运行的检查；
   - Parity：基线、配置、输入数、`max|Δ| = 0`；
   - G8 警告和覆盖率摘要。
6. **`finalize`：**
   - 不带参数时，只打印 worktree 路径、分支名和建议的提交说明。
   - `--commit`：在 worktree 中提交到 `registry/<op>-<run_id>`。提交说明使用约定前缀，例如 `feat: route <sites> through the <op> registry operator`，并遵守 AGENTS.md 的 Git 约定。
   - `--push`：执行 `git push -u origin <branch>`。
   - 本工具不提供合并功能。

#### R8 人工审阅与合并

审阅者重点看三件事：参考实现是否真是原数学，没有偷换表达式；preserve 分支是否完整；`expected_calls_per_inference` 是否合理。

合并后进入 0009 的调优流程：在合并后的代码上运行 `capture-ops --ops <op>`，然后 `calibrate`、`tune-all`/`run`、`install`、`verify`。G6 产生的抓取结果只有在源码哈希没变时才能直接复用（`scripts/kernel_tuning/discovery/tasks.py:99-102` 会做校验）。审阅中只要改过代码，就要重新抓取。

### 4.5 智能体必须遵守的规则

以下规则写进 prompt，并由 R4 和 R5 在机制上检查：

1. **参考实现的位置：** 写在 `embodiinfer/backend/torch/ops.py`，函数名等于算子名。参数只能是张量或常量（`int`/`float`/`bool`/`str`/`None`），返回张量或张量元组，不得原地修改输入。安装生成的 `applies()` 只能判断这两类参数（`install.py:108-122`）。
2. **逐字保留算术：** 使用相同的 ATen 调用、相同的顺序和相同的中间 dtype 转换。不允许把 `F.linear` 改写成 `x @ w.T`，也不允许增删 `.float()`、`.to()`、`.contiguous()` 或合并、拆分 GEMM，除非 `REGISTRATION.json` 中说明理由，并且 G5、G7 仍然通过。
3. **注册方式：** 用 `register_lazy_op("<op>", "torch", "embodiinfer.backend.torch.ops", "<op>")`。只有 Torch 实现的算子不能放进 `layers/ops.py:160` 那个同时注册 torch 和 triton 的循环。
4. **调用点写法：** 使用策略现有的 helper，例如 pi0.5 的 `_op(kernels, "<op>", False)`。只有 Torch 实现时 `prefer` 必须是 `"torch"`，否则 `get_op` 会抛出 KeyError。`KernelSelection` 用参数一路传下去，不使用全局变量。
5. **preserve 分支：** 原样保留，只用类型判断（例如 `type(module) is nn.Linear`），不按模型名分支。
6. **不改的内容：** 不修改 `KernelSelection`、调优档案、`scripts/`、依赖，以及已有测试的断言。
7. **公开函数：** 带类型注解和说明契约的 docstring（AGENTS.md 的要求）。

### 4.6 融合区域（`kind: fuse`）

融合区域的流程与包装相同，额外有三条要求：

- `sites` 中必须给出相邻的一组表达式，例如 `_mlp` 中 gate/up GEMM 后接 `gated_gelu`。参考实现就是这组表达式按原顺序组成的函数，结果必须与原表达式序列逐位一致（G5、G7）。
- 名称优先按数学含义取，例如 `gate_up_gelu`。确实只属于某个模型的区域，用 `<policy>_<name>` 形式的标识符，例如 `pi05_residual_ada_rms_norm`。0009 §4.1 写的 `pi05.residual_ada_rms_norm` 带点号，与现有工具链对算子名的要求冲突（R1 第 1 项），因此改为下划线形式，并建议同步修订 0009 的命名示例。
- 融合可能删掉原来单独存在的注册调用。例如被融合的 `gated_gelu` 不再单独出现，`expected_calls_per_inference` 要反映这个变化。调优档案里如果已经装过被融合掉的算子，这些 kernel 不再生效，PR 描述中需要写明。

### 4.7 环境与执行位置

- **控制器：** 工具环境，Python ≥ 3.12 + Humanize2，POSIX，与 `tune-all` 相同（`scripts/kernel_tuning/__main__.py:19-31`）。
- **模型运行时：** 通过 `runtime_python` 指定，是模型依赖组的环境，例如 `uv sync --python 3.12 --frozen --no-dev --group pi05`。所有加载模型的步骤（R1 的注册检查、R2、G2 的运行时部分、R5、R7）都以子进程运行，带超时和进程组清理，做法同 `scripts/kernel_tuning/runner.py`。
- **CPU 测试：** 通过 `test_python` 指定，默认等于 `runtime_python`；只能在核心开发环境运行的测试可以单独指定。
- **机器：** 整个流程在目标 GPU 机器上运行，parity 结果只对这台机器有效，证据中会记录硬件信息。智能体的沙箱限制与 0009 相同：Thor 上 AppArmor 禁止普通用户创建用户命名空间，Codex 以无沙箱方式在低权限账号下运行。

### 4.8 备选方案

- **规则化改写（codemod），不用智能体。** 单一表达式的包装可以用 AST 替换完成，但 `kernels` 传参链、preserve 分支和融合都需要理解代码。本提案由智能体修改、由门禁兜底；将来可以给 `wrap` 加一条 codemod 快速路径，产物同样经过 R4/R5。
- **运行时子图替换（torch.fx 或 Dynamo pattern matcher），不改源码。** 这样不需要 PR，但 eager 路径不经过 fx 图，替换也不可审阅，违背 0009"注册调用就是模型默认代码路径"的设计。拒绝。
- **只做端到端逐位一致，不做 ATen 序列。** 失败信息太粗（只知道"差了 0.0078"），每轮都要付全部配置的 GPU 成本。两者都保留，ATen 序列放在前面执行。
- **把注册和调优串成一条命令。** 注册改动要经过人工审阅，审阅中改动参考实现会让已有的抓取作废。所以 `register` 只做到 PR 为止，调优在合并之后进行。

## 5. 模型无关性结论

- **工具本身：** 本工具属于 `scripts/` 下的开发者工具，不进入运行时。控制器、门禁和 flow 是通用的，只依赖公开的注册表接口（`available_ops`、`get_op`、`KernelSelection`）、`discovery.drive` 的驱动参数、`profile` 和 `capture-ops` 的输出格式。
- **模型相关的知识**（调用点、preserve 模式、parity 配置、CPU 测试表、示例 commit）放在规格文件和 `registration/models/<model>.yaml` 中，代码里不按模型名分支。
- **智能体产生的运行时改动** 与 `449ebe2` 的分层相同：
  - `backend/torch/ops.py` 里是纯张量函数；
  - `layers/ops.py` 里加一行注册；
  - 调用点留在策略层。

  引擎层不受影响。
- **对新模型的要求：** 必须先具备 `kernels` 传参链（一个 `KernelSelection` 和类似 `_op` 的 helper）。这个一次性的接入不属于本流程。

## 6. 无损判据

判据是逐位一致，不设容差。

- **参考：** 基线 commit（`spec.json` 中记录的 `base`）在同一 parity 配置、同一输入、同一随机种子（`seed + index`，与 `verify` 相同）下运行，`kernel_profile: none`。
- **比较对象：** `flatten_outputs` 展开后的每一个输出叶子，每个输入都比较。浮点输出（如动作块）要求 `max|Δ| = 0`；整数、布尔、字符串输出要求完全相等。输出结构（叶子名称和长度）不同也算失败。
- **配置：** `parity.configs` 中的每一项，至少覆盖每个调用点所在的执行路径，并包含生产配置（pi0.5 为 Inductor + 完整去噪 CUDA Graph）。所有运行都设置 `TORCHINDUCTOR_DETERMINISTIC=1`，每次使用全新的 `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR`；`float32_matmul_precision` 保持默认，并记录在证据中。
- **前置条件：** 基线自身两次全新运行的结果完全相同（R2 确定性预检）。不满足时这次运行失败，不会放宽标准。
- **辅助判据：** eager ATen 调用序列完全相同（G5）。
- **复现方式：** `register run` 的 G5/G7，或者手动执行 `snapshot` + `snapshot-compare`，命令见 R2/R5。
- **为什么可以要求逐位一致：** 注册改动只是把同一串 ATen 调用移到另一个函数里执行。Dynamo 会内联这层包装，编译出的图不变；eager 下执行的 kernel 序列也不变。所以任何差异都说明数学或布局被改了，这正是门禁要拦截的。

## 7. 实现计划

新增文件（均位于 `scripts/kernel_tuning/`）：

| 文件 | 内容 |
|---|---|
| `registration/__init__.py` | 包说明 |
| `registration/spec.py` | 规格的 dataclass、YAML 和命令行解析、校验；冻结后写入 `spec.json` |
| `registration/sites.py` | AST 调用点定位、`region` → 源码映射、preserve 条件提取 |
| `registration/candidates.py` | R0：从 `hotspots.json` 列出未覆盖区域 |
| `registration/workspace.py` | `git worktree` 的创建和清理、diff、G1 范围检查 |
| `registration/store.py` | 运行目录、`manifest.json`、阶段状态；复用 `artifacts.atomic_json` 和 `run_lock` |
| `registration/launch.py` | 以 `runtime_python`/`test_python` 启动子进程：设置 `PYTHONPATH`、缓存目录和环境变量，处理超时，校验导入路径 |
| `registration/gates.py` | G1–G4、G5–G8 的判定和 `feedback.json` 的生成 |
| `registration/prompts.py` | plan、implement、repair 三种 prompt |
| `registration/flow.py` | Humanize2 flow `operator_registration` |
| `registration/package.py` | `evidence.json`、`PR.md`、`finalize` |
| `registration/models/pi05.yaml` | pi0.5 默认值：CPU 测试、preserve 模式、parity 配置模板、示例 commit `449ebe2` |
| `snapshot.py` | 运行时命令 `snapshot`、`snapshot-compare`（复用抽出的 `collect_outputs`） |
| `optrace.py` | 运行时命令 `trace-ops`（只记录调用序列的 `TorchDispatchMode`） |

对现有代码的改动：

- `verify.py`：把 `_run` 抽成公共的 `collect_outputs`，`verify` 的行为和输出不变。
- `regions.py`：`capture-ops` 增加 `--record-callers`（按调用者的文件和行号计数），以及只记录布局的审计模式 `audit-layouts`。默认行为不变。
- `__main__.py`：增加 `register` 子命令组（`candidates`、`new`、`run`、`resume`、`status`、`finalize`）和运行时命令（`snapshot`、`snapshot-compare`、`trace-ops`、`audit-layouts`）。
- 文档：在 `CONTRIBUTING.md` 的 "Optional kernel tuning tools" 一节补充 `register` 用法；在 0009 §7 指向本提案，并按 §4.6 修订融合算子的命名示例。

不改动的部分：运行时包 `embodiinfer/`（智能体产生的改动经各自的 PR 合入）、`pyproject.toml`、`uv.lock`（工具环境已经有 `pyyaml`，见 `scripts/kernel_tuning/pyproject.toml`）。

## 8. 测试计划

CPU 测试（`tests/kernel_tuning/test_registration_*.py`，不需要 GPU，也不需要智能体）：

- **规格：** 算子名合法性、与已注册算子重名、缺少 parity 配置、`region` 未确认时拒绝 `--yes`。
- **调用点定位：** 在固定的示例源码上定位 `source/function/expression`；`region` → 候选调用点；提取 preserve 条件，包括 `isinstance` 和 `torch.is_grad_enabled`。
- **G1 范围检查：** 在合成的 git 仓库上，分别验证允许的文件、越界文件、删除已有测试断言、修改 `pyproject.toml` 的情况。
- **G2 结构检查：** 参考实现缺失、导入了 `embodiinfer.policies`、错误地放进 torch+triton 循环、原表达式残留在非 preserve 路径。
- **G4 有效性检查：** 新测试在基线上通过时判为无效。
- **`snapshot-compare`：** 完全相同时通过；差一个 ULP、输出结构不同、离散输出不同时都失败。
- **`trace-ops`：** 在 CPU 上的小模块中记录序列，并在分歧点生成差异报告。
- **flow：** 用 Humanize2 自带的假智能体（同 `tests/kernel_tuning/test_flow.py`）跑完"失败 → 反馈 → 修复 → 通过"一轮，并验证 `resume` 不会重复已完成的阶段。缺少 `hmz` 时跳过。

GPU 测试（标记 `gpu`，并使用 pi0.5 checkpoint 标记）：

- 端到端回放 `449ebe2`：在 `449ebe2^` 上，以 `gated_gelu` 的同名替身（例如 `gated_gelu_v2`，避免与已注册算子冲突）为目标运行一次 `register run --until parity`，要求在预算内通过 G5–G7。
- `linear` 作为第一个真实目标，规格见第 4.2 节，产出一个待审阅的 PR。

## 9. 基准计划

注册改动本身不应改变性能，因此不设性能门禁。

`--bench` 在 Jetson Thor 上用 0009 §9 的协议（B=1、BF16、10 步去噪、LIBERO-10、Inductor + 完整去噪 CUDA Graph），在 `kernel_profile: none` 下分别跑基线和候选，报告平均值、P50 和 prefill/decode 拆分。`_op` 的分派只在 eager 调用中产生开销；参考 `449ebe2` 的测量，native 路径约有 1% 的差异。

工具本身的成本也要在第一次运行时记录：每轮尝试的 GPU 时间（主要是 G7 各配置的模型加载和编译），以及完成一个算子所需的尝试次数，作为确定 `max_attempts` 默认值的依据。

## 10. 风险与限制

- **覆盖面有限。** parity 只覆盖配置里列出的路径。没有被执行到的分支（TP、量化、B > 1、其他 checkpoint）只靠 CPU 测试和审阅来保证。证据中会列出未覆盖的分支。
- **调用点定位是启发式的。** `region` 到源码的映射只按属性名匹配；编译区域里的 `kernels:*` 热点无法自动定位。因此必须经过人工确认（R1）。
- **审阅仍然必要。** 智能体的补丁可能逐位一致但结构不好，例如传参链过长或重复判断。0009 的非目标之一就是不自动合并智能体写的策略改动。
- **GPU 成本。** 每轮尝试大约是"配置数 × 一次模型加载和编译"。通过按开销排序门禁（G5 在前）、循环中只用 `samples` 个输入、终检才用全量输入来控制。
- **确定性依赖 Inductor 设置。** 某些配置即使设置了 `TORCHINDUCTOR_DETERMINISTIC=1` 仍无法复现时，这个配置就不能用于逐位一致检查，这次运行会失败并报告原因，不会放宽标准。
- **导入路径。** 可编辑安装可能让子进程导入主检出目录里的代码。通过 `PYTHONPATH` 和 `embodiinfer.__file__` 校验来防范；校验不通过时直接失败。
- **G8 观测不到编译区域内部。** 只在编译区域里执行的调用无法审计布局，只能等调优后的 `install` 和命中统计来暴露问题。
- **工具链对算子的约束保留不变。** 算子名必须是标识符，参考实现必须在 `backend/torch/ops.py`。真正的模型专属融合如果需要放在策略目录，就要先泛化 `regions.py`、`discovery/tasks.py`、`install.py` 的相关检查，那是单独的改动。
- **中转站断流。** 与 0009 相同，中转站断流会浪费尝试预算。
