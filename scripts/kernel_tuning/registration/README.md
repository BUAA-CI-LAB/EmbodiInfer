# 算子注册流程（智能体闭环）

本目录是 `python -m scripts.kernel_tuning register …` 工具链的规格说明，代码尚未实现。

本工具链自动完成"把一个算子接入算子注册表"这一步：

1. 调用方选定算子；
2. 工具定位调用点；
3. 智能体修改代码、注册算子、编写测试；
4. 控制器运行检查；
5. 失败时把信息反馈给智能体重做；
6. 全部通过后产出一个待审阅的分支。

不在本流程内的工作：

- 调优和安装 kernel：使用现有的 `capture-ops`、`calibrate`、`tune-all`/`run`、`install`、`verify`。
- GPU 上的整模型逐位一致验证：按 [0009 §6](../../../docs/proposals/0009-hotspot-kernel-tuning.md) 的要求，合并前在目标 GPU 上单独做一次。
- 合并：由人工完成。

本流程的检查全部在 CPU 上运行，因此不需要 GPU。

## 1. 流程总览

```
选定算子 ──► 定位调用点 ──► 智能体修改 ──► 检查 ──通过──► 产出分支
 (步骤1)      (步骤2)         (步骤3)      (步骤4)          (步骤6)
                 │                ▲            │
            调用方确认             └── 反馈 ◄───┘ 失败 (步骤5)
```

| 步骤 | 工作 | 执行者 | 使用的现有工具 | 产物 |
|---|---|---|---|---|
| 1 选定算子 | 校验规格，记录基线 commit，建立 worktree | 控制器 | `git worktree`；在测试环境中调用 `embodiinfer.layers.available_ops` | `spec.json`、`base/`、`worktree/` |
| 2 定位调用点 | 找到要改的表达式，以及需要保持原样的分支 | 控制器，调用方确认 | 无（`ast`） | `sites.json`、`sites.md` |
| 3 智能体修改 | 写参考实现、注册、改调用点、写测试 | 智能体 | Humanize2 flow 机制（同 `scripts/kernel_tuning/flow.py`） | worktree 中的改动、`PLAN.md`、`REGISTRATION.json` |
| 4 检查 | 范围、结构、格式、测试、测试有效性 | 控制器 | `ruff`、`pytest` | `attempts/<N>/checks.json` |
| 5 反馈重做 | 把失败信息交回智能体 | 控制器 + 智能体 | 同一个 Humanize2 会话 | `attempts/<N>/feedback.json` |
| 6 产出分支 | 汇总改动，按需提交和推送 | 控制器 | `git` | 分支 `registry/<op>-<run_id>`、`patch.diff`、`SUMMARY.md` |

## 2. 命令

控制器运行在调优工具环境里（Python ≥ 3.12 + Humanize2，POSIX），与 `tune-all` 相同。需要导入 `embodiinfer` 的检查，通过 `test_python` 指定的解释器以子进程运行。

```bash
TOOL="uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning"

# 步骤 1、2：为选定的算子建立一次运行，打印 sites.md 后停下，等待调用方确认
$TOOL register new --spec linear.yaml --output results/kernel_tuning/register

# 步骤 3–5：智能体修改、检查、反馈，直到通过或预算用尽
$TOOL register run RUN --agent 'HARNESS/MODEL:EFFORT' --set test_python=/abs/dev/python

$TOOL register resume RUN   # 中断后继续
$TOOL register status RUN   # 查看当前阶段和每次尝试的结果

# 步骤 6：默认只打印分支和改动；提交、推送都需要显式传参
$TOOL register finalize RUN [--commit] [--push]
```

`register new` 也可以不用规格文件，直接用命令行参数：

```bash
$TOOL register new --model pi05 --op linear \
    --site 'embodiinfer/policies/pi05/modeling_pi05.py::_fused_qkv::F.linear(x, weight)' \
    --site 'embodiinfer/policies/pi05/modeling_pi05.py::_mlp::mlp.down_proj(activated)' \
    --output results/kernel_tuning/register
```

## 3. 规格文件

规格文件决定这次运行注册哪个算子、改哪些调用点。

```yaml
model: pi05                    # 调用点所属的策略
op: linear                     # 新算子名，要求见步骤 1
description: >                 # 算子的数学含义，写进智能体的上下文
  F.linear(x, weight, bias)：weight 为 [out, in]，bias 可为 None；不改变 dtype。
sites:                         # 要改的调用点：文件、所在函数、原表达式
  - source: embodiinfer/policies/pi05/modeling_pi05.py
    function: _fused_qkv
    expression: "F.linear(x, weight)"
  - source: embodiinfer/policies/pi05/modeling_pi05.py
    function: _mlp
    expression: "mlp.down_proj(activated)"
preserve:                      # 必须保持原样的分支；步骤 2 会自动补充候选
  - "QuantizedLinear 继续走 embodiinfer/layers/linear.py 的量化注册表"
  - "张量并行下的 RowParallelLinear / ColumnParallelLinear 保持模块调用"
tests:                         # 每次检查都要通过的已有 CPU 测试
  - tests/test_kernel_ops.py
  - tests/test_pi05_components.py
  - tests/test_pi05_cpu_routes.py
  - tests/test_tensor_parallel.py
budget:
  max_attempts: 5              # 最多尝试次数
  agent_timeout_seconds: 1800  # 每轮智能体调用的超时
  max_seconds: 7200            # 整次运行的时间上限
```

## 4. 各步骤

### 步骤 1：选定算子（`register new`）

控制器依次检查以下各项，任何一项不满足就报错退出：

1. **算子名合法。** `op` 必须是 Python 标识符，并按数学含义命名，不能用模型名。原因是现有调优工具要求参考实现位于 `embodiinfer.backend.torch.ops`，且函数名与算子名相同（`scripts/kernel_tuning/regions.py:215-217`、`scripts/kernel_tuning/discovery/tasks.py:95-98`）；`install` 还会用算子名作包目录名（`scripts/kernel_tuning/install.py:177`）。
2. **算子尚未注册。** 以 `PYTHONPATH=<仓库根目录>` 在 `test_python` 中运行下面的命令，`op` 不能出现在结果里：
   ```bash
   python -c "from embodiinfer.layers import available_ops; print(sorted(available_ops()))"
   ```
   同时 `embodiinfer/backend/torch/ops.py` 中不能已有同名函数。
3. **工作目录干净。** `git status --porcelain` 为空，然后记录 `BASE=$(git rev-parse HEAD)`。
4. **建立两个 worktree。** 主检出目录在整个过程中不被修改：
   ```bash
   git worktree add --detach RUN/base "$BASE"                        # 基线，步骤 4 的有效性检查用
   git worktree add -b registry/<op>-<run_id> RUN/worktree "$BASE"   # 智能体只改这里
   ```
5. **冻结规格。** 规格和 `BASE` 一起写入 `RUN/spec.json`。

### 步骤 2：定位调用点（`register new` 的后半段）

对每个 `site`，控制器用 `ast` 完成以下工作：

- **定位表达式。** 在 `function` 的函数体中查找与 `expression` 规范化后（`ast.unparse`）相同的表达式，记录行号和列号。找不到，或者找到多处又无法区分时，直接报错。
- **收集 preserve 候选。** 收集包裹该表达式的 `if` 条件，例如 `isinstance(..., QuantizedLinear)`、`torch.is_grad_enabled()`、`self.attention == "eager"`、`self.tensor_parallel.enabled`，与规格中的 `preserve` 合并。
- **检查能否拿到 `KernelSelection`。**
  - 调用点在策略方法里时，可以直接用 `self._kernels`（`embodiinfer/policies/pi05/modeling_pi05.py:574`）。
  - 调用点在自由函数里时，需要新增 `kernels` 参数。控制器会列出同一文件中调用这个函数的位置，智能体要把参数一路传过去。

结果写入 `sites.json`，并生成给人看的 `sites.md`，内容是每个调用点的源码片段和 preserve 条件。然后命令停下。调用方确认无误、或直接修改 `sites.json` 之后，再运行 `register run`。传 `--yes` 可以跳过确认。

### 步骤 3：智能体修改（`register run`）

**运行机制**与现有调优流程相同：控制器用 Humanize2 运行一个可恢复的 flow `operator_registration`，结构参照 `scripts/kernel_tuning/flow.py`；启动方式与 `scripts/kernel_tuning/__main__.py` 中的 `_execute` 一致：

```python
Hmz(workspace=RUN / "workspace").run(
    flow=Path("scripts/kernel_tuning/registration/flow.py"),
    task=f"Register {op} and route its call sites through the registry",
    agents={"coder": AGENT},
    params={"run_dir": str(RUN)},
    budget={"duration": remaining_seconds, "graceful": False},
    resume=resume,
)
```

智能体的工作环境（`LocalEnv`）指向 `RUN/worktree`。只读上下文放在 `RUN/context/`，包括：

- `AGENTS.md`；
- `embodiinfer/layers/ops.py`、`embodiinfer/backend/torch/ops.py`，以及各调用点所在的文件；
- `tests/test_kernel_ops.py`，作为测试写法示例；
- 一次已完成迁移的 diff 作为示例：`git show 449ebe2 -- embodiinfer/`；
- `sites.json`；
- 下面的规则全文。

**每次尝试分两轮：**

1. **`plan`**：写出 `PLAN.md`，包括：
   - 算子的数学定义，以及它与原表达式的逐项对应关系；
   - 函数签名，即哪些参数是张量、哪些是常量；
   - 每个调用点的改法，以及 `kernels` 如何传到该点；
   - preserve 分支如何保持不变；
   - 要新增的测试。
2. **`implement`**：在 worktree 中完成修改，并写出 `REGISTRATION.json`：
   ```json
   {
     "op": "linear",
     "reference": "embodiinfer/backend/torch/ops.py::linear",
     "sites": [{"id": "S1", "file": "embodiinfer/policies/pi05/modeling_pi05.py",
                "function": "_fused_qkv", "call": "_op(kernels, \"linear\", False)(x, weight)"}],
     "preserved": [{"site": "S1", "branch": "isinstance(projection, QuantizedLinear)"}],
     "tests": ["tests/test_kernel_ops.py::test_linear_reference_matches_f_linear"]
   }
   ```

**智能体必须遵守的规则**（写进 prompt，并由步骤 4 检查）：

1. **参考实现。** 写在 `embodiinfer/backend/torch/ops.py`，函数名等于算子名，带类型注解和说明契约的 docstring，只导入 `torch` 和标准库。参数只能是张量或常量（`int`/`float`/`bool`/`str`/`None`），返回张量或张量元组，不得原地修改输入。
2. **逐字保留原算术。** 使用相同的 ATen 调用、顺序和中间 dtype 转换。例如不能把 `F.linear` 改写成 `x @ w.T`，不能增删 `.float()`、`.to()`、`.contiguous()`，也不能合并或拆分 GEMM。仓库多处说明过：表达式或形状一变，舍入就会变（`embodiinfer/policies/pi05/modeling_pi05.py:664-666`）。
3. **注册。** 单独写一行：
   ```python
   register_lazy_op("<op>", "torch", "embodiinfer.backend.torch.ops", "<op>")
   ```
   只有 Torch 实现的算子不能放进 `embodiinfer/layers/ops.py:160` 那个同时注册 torch 和 triton 的循环。
4. **调用点。** 使用策略现有的 helper，pi0.5 是 `_op(kernels, "<op>", False)(...)`。只有 Torch 实现时，偏好的实现必须是 `torch`，否则 `get_op` 会抛出 KeyError。`KernelSelection` 用参数传递，不使用全局变量。
5. **preserve 分支原样保留。** 判断方式只用类型（例如 `type(module) is nn.Linear`），不按模型名分支。
6. **测试至少覆盖三类：**
   - 参考实现与原表达式在 CPU 随机输入（FP32 和 BF16）上 `torch.equal`；
   - 调用点经过注册表，做法是用一个会记录调用的假调优档案，写法见 `tests/test_kernel_ops.py:135-150`；
   - preserve 分支不经过注册表。
7. **不能改的内容：** 已有测试的断言、`KernelSelection`、`embodiinfer/backend/triton/tuned/`、`scripts/`、`pyproject.toml`、`uv.lock`。
8. **允许和禁止的操作：** 可以运行 `ruff` 和 CPU `pytest`；不能运行 GPU 代码，不能 `git commit`，不能安装依赖。

### 步骤 4：检查

控制器按顺序执行以下检查，前一项失败时后面的不再执行。所有检查都在 CPU 上运行。凡是需要导入 `embodiinfer` 的检查，都以 `PYTHONPATH=RUN/worktree` 在 `test_python` 中运行，并确认导入的 `embodiinfer.__file__` 位于 worktree 内。这是因为开发环境里 EmbodiInfer 是可编辑安装，不设置的话会导入主检出目录中的旧代码。

| 检查 | 内容 | 方式 |
|---|---|---|
| C1 范围 | `git -C RUN/worktree diff --name-status $BASE` 只涉及：`embodiinfer/backend/torch/ops.py`、`embodiinfer/layers/ops.py`、`sites.json` 中的源文件、`tests/test_kernel_ops.py` 或新的 `tests/test_registry_<op>.py`。已有测试文件只允许新增内容，不允许删改原有行 | `git diff` |
| C2 结构 | ① `backend/torch/ops.py` 中有 `def <op>`，带注解和 docstring，且不导入 `embodiinfer.models`/`policies`/`engine`；② `layers/ops.py` 中有对应的 `register_lazy_op` 行；③ 每个调用点已改为注册表调用，原表达式只出现在 preserve 分支中；④ 在 `test_python` 中验证：`available_ops()["<op>"]` 包含 `torch`、`get_op("<op>")` 就是参考实现、`KernelSelection("none").prepare()` 成功、`KernelSelection("torch").resolve("<op>")` 返回参考实现 | `ast` + 子进程 |
| C3 格式 | 改动文件通过 `ruff check` 和 `ruff format --check` | `ruff` |
| C4 测试 | 规格 `tests` 中的测试和 `REGISTRATION.json` 列出的新测试全部通过 | `test_python -m pytest <files> -q`，在 `RUN/worktree` 中运行 |
| C5 测试有效性 | 把新测试复制到 `RUN/base`，以 `PYTHONPATH=RUN/base` 运行，结果必须失败，以此证明这些测试确实依赖本次改动 | `test_python -m pytest` |

### 步骤 5：反馈重做

任何检查失败时，控制器写出 `attempts/<N>/feedback.json`：

```json
{
  "check": "C2",
  "summary": "调用点 S2 仍直接调用 mlp.down_proj(activated)，且不在 preserve 分支内",
  "details": {"site": "S2", "file": "embodiinfer/policies/pi05/modeling_pi05.py", "line": 315}
}
```

`details` 的内容随检查而定：
- C1：越界的文件或被改动的已有测试行；
- C2：失败的具体检查项；
- C3、C4：命令输出摘要，截断到固定长度；
- C5：在基线上仍然通过的测试名。

**下一次尝试：**
- 默认在同一个 Humanize2 会话中发出 `repair` prompt，附上 `feedback.json`，智能体在原 worktree 上继续修改；
- 连续两次在同一项检查失败时，新开一个会话，从 `plan` 重新开始，但 worktree 保留。

每次尝试结束时，控制器把 `git -C RUN/worktree diff $BASE` 存为 `attempts/<N>/patch.diff`。

**停止条件：**
- 全部检查通过：进入步骤 6；
- 达到 `max_attempts` 或 `max_seconds`：状态记为 `exhausted`，worktree 保留供人工接手；
- 规格、`BASE` 或 `RUN/base` 被改动：状态记为 `failed`。

### 步骤 6：产出分支（`register finalize`）

全部检查通过后，控制器生成两个文件：

- `RUN/patch.diff`；
- `RUN/SUMMARY.md`，内容包括：
  - 算子定义；
  - 改动的调用点和保持不变的分支；
  - 每项检查的命令和结果；
  - 尝试次数；
  - 明确写出未做的验证：GPU 整模型逐位一致。

`finalize` 的行为：

- 不带参数时，只打印 worktree 路径、分支名和建议的提交说明。
- `--commit`：在 worktree 中提交到 `registry/<op>-<run_id>`。提交说明遵守 AGENTS.md 的 Git 约定，例如 `feat: route pi0.5 projections through the linear registry operator`。
- `--push`：执行 `git push -u origin registry/<op>-<run_id>`。

本工具不提供合并功能。合并前，还需要在目标 GPU 上按 0009 §6 确认 `kernel_profile: none` 时整模型输出与 `BASE` 逐位一致，方法与 commit `449ebe2` 相同。

## 5. 运行目录

```
results/kernel_tuning/register/<run_id>/
  manifest.json       当前阶段、每次尝试的结果、耗时（atomic_json 写入，run_lock 加锁）
  spec.json           冻结的规格和 BASE
  sites.json  sites.md
  context/            智能体的只读上下文
  base/               BASE 的 worktree（只读）
  worktree/           分支 registry/<op>-<run_id> 的 worktree
  workspace/          Humanize2 的工作区和日志
  attempts/<N>/       plan/implement/repair 的 prompt 和回复、PLAN.md、REGISTRATION.json、
                      checks.json、feedback.json、patch.diff
  patch.diff  SUMMARY.md
```

`resume` 会从 `manifest.json` 记录的阶段继续，已经通过的阶段不再重跑。规格变了就要新建一次运行。

## 6. 现有工具链的复用

| 现有部件 | 用在 | 用法 |
|---|---|---|
| Humanize2 flow 机制（`scripts/kernel_tuning/flow.py`、`__main__.py` 中的 `_execute`） | 步骤 3、5 | 新写一个 flow，沿用 `@flow(..., resumable=True)`、`AgentCollection`、`LocalEnv`、`Budget`、`ctx.state` 中的进度台账，以及 `Hmz(...).run(...)` 的启动方式 |
| `scripts/kernel_tuning/artifacts.py` 中的 `atomic_json`、`run_lock` | 运行目录 | 直接导入 |
| `scripts/kernel_tuning/runner.py` 的子进程处理（超时、进程组清理） | 步骤 1、4 的子进程 | 照搬这套做法，但启动的是 `test_python` |
| `embodiinfer.layers`（`available_ops`、`get_op`、`KernelSelection`） | 步骤 1、C2 | 在 `test_python` 子进程中导入 |
| `ruff`、`pytest` | C3–C5 | 命令行调用 |
| `git worktree` | 步骤 1、6 | 命令行调用 |

`profile`、`capture-ops`、`calibrate`、`tune-all`、`install`、`verify` 不在本流程内使用。它们属于调优阶段，或者需要 GPU。

## 7. 需要新写的代码

| 文件 | 内容 |
|---|---|
| `registration/__init__.py` | 包说明 |
| `registration/spec.py` | 规格的解析（YAML 或命令行）和校验，生成 `spec.json` |
| `registration/sites.py` | 用 `ast` 定位调用点、提取 preserve 条件、查找调用链 |
| `registration/workspace.py` | worktree 的创建和清理、diff、C1 检查 |
| `registration/checks.py` | C2–C5 检查，生成 `feedback.json` |
| `registration/prompts.py` | `plan`、`implement`、`repair` 三种 prompt |
| `registration/flow.py` | Humanize2 flow `operator_registration` |
| `registration/finalize.py` | 生成 `SUMMARY.md`，提交和推送 |
| `__main__.py` | 增加 `register` 子命令组：`new`、`run`、`resume`、`status`、`finalize` |
| `tests/kernel_tuning/test_registration.py` | CPU 测试，覆盖：规格校验；在示例源码上定位调用点；用合成仓库测 C1；构造缺参考实现、错进循环、原表达式残留等情况测 C2；C5 能识别在基线上也通过的测试；用 Humanize2 自带的假智能体（同 `tests/kernel_tuning/test_flow.py`）跑通"失败 → 反馈 → 修复 → 通过"并验证 `resume` |

工具环境已经包含 `pyyaml`（见 `scripts/kernel_tuning/pyproject.toml`），不需要新增依赖。

## 8. 限制

- **CPU 检查不能证明 GPU 上逐位一致。** 例如 pi0.5 的 native 路径只在 CUDA 上启用（`embodiinfer/policies/pi05/modeling_pi05.py:565-572`），它上面的调用点在 CPU 测试中执行不到。因此合并前仍需在 GPU 上验证一次（见步骤 6）。
- **调用点由调用方指定。** 工具只负责定位和确认，不会从热点报告中自动挑选。
- **每次只注册一个算子，不支持把多个相邻算子融合成一个。**
- **策略需要已有 `kernels` 传参链。** 必须已经有 `KernelSelection` 和类似 `_op` 的 helper，目前只有 pi0.5 具备。其他策略要先人工接入一次。
- **智能体写出的补丁可能检查全过但结构不好**（例如传参链过长），仍然需要人工审阅。
