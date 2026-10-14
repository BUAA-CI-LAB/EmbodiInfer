# 算子注册工具（`register`）

`python -m scripts.kernel_tuning register …` 把一个尚未注册的数学算子接入算子注册表
（`embodiinfer/layers/ops.py`），并把选定的调用点改为经过注册表调用。设计归属见
[0008 §4.2](../../../docs/proposals/0008-kernel-tuning.md#42-select-and-if-necessary-register-an-operator)。

工具由一个控制器驱动编码智能体，循环执行"计划 → 实现 → CPU 检查 → 反馈修复"，直到全部检查通过或预算用尽。
最终产物是一个待人工审阅的分支和 `patch.diff`。

注册只改调用路径，不改数值：参考实现必须逐字保留原表达式的算术。以下工作不在本工具范围内：

- 选择要注册的算子和调用点（由调用方指定）；
- 融合多个相邻算子；
- 调优、安装 kernel（`tune-all`/`run`、`install`、`verify` 等）；
- GPU 上的整模型逐位一致验证（合并前必须单独完成，见[合并前](#合并前)）；
- 合并分支（由人工完成）。

## 1. 前提条件

- **工具环境**：Python ≥ 3.12，并已安装 Humanize2 和 Ruff，与 `tune-all` 共用：
  `uv sync --project scripts/kernel_tuning --python 3.12 --frozen`。
  `run` 和 `resume` 只能在 POSIX 系统上执行；`new`、`status`、`finalize` 不启动智能体。
  启动后工具环境的包版本（`hmz`、`pydantic`、`pyyaml`、`psutil`）被记录下来，变化后需要新建运行。
- **测试解释器 `test_python`**：一个已准备好的模型运行环境，需包含 Torch、pytest，以及规格中 CPU
  测试需要的模型依赖。所有需要导入 `embodiinfer` 的检查都在这个解释器的子进程中执行。工具环境自带
  Torch 时可以省略，否则必须通过 `--set test_python=/absolute/python` 指定。
- **干净的仓库**：`git status --porcelain` 必须为空（包括未跟踪文件）。输出目录应放在仓库外或被
  `.gitignore` 忽略的位置，默认的 `results/kernel_tuning/register` 满足要求。
- **策略已接入注册表**：调用点所在文件必须已有模块级 `_op` helper，并能拿到 `KernelSelection`。
  目前只有 pi0.5（`embodiinfer/policies/pi05/modeling_pi05.py`）满足。其他策略需要先人工接入一次。

## 2. 快速开始

```bash
TOOL="uv run --project scripts/kernel_tuning python -m scripts.kernel_tuning"

# 1. 建立运行：校验规格、建立 worktree、定位调用点，打印 sites.md 后停下
$TOOL register new --spec linear.yaml --set test_python=/abs/runtime/python
#    输出最后一行为 "Run: <RUN 目录>"

# 2. 审阅 RUN/sites.md（需要时补充 RUN/sites.json），然后启动智能体循环
$TOOL register run RUN --agent 'HARNESS/MODEL:EFFORT'

# 中断后继续；查看状态
$TOOL register resume RUN
$TOOL register status RUN

# 3. 导出已验证的补丁；提交和推送都需要显式参数
$TOOL register finalize RUN [--commit] [--push]
```

所有命令都在仓库根目录执行。`run`/`resume` 的退出码在状态为 `passed` 或 `finalized` 时为 0，
否则为 1。

## 3. 规格

规格决定本次运行注册哪个算子、修改哪些调用点、用哪些测试回归，以及预算。

```yaml
model: pi05                    # 调用点所属的策略目录名
op: linear                     # 新算子名
description: >                 # 算子的数学含义，写入智能体上下文和 SUMMARY.md
  F.linear(x, weight, bias)：weight 为 [out, in]，bias 可为 None；不改变 dtype。
sites:                         # 要改的调用点：文件、所在函数、原表达式
  - source: embodiinfer/policies/pi05/modeling_pi05.py
    function: _fused_qkv
    expression: "F.linear(x, weight)"
  - source: embodiinfer/policies/pi05/modeling_pi05.py
    function: _mlp
    expression: "mlp.down_proj(activated)"
preserve:                      # 必须保持原样的路径，作为审阅说明交给智能体
  - "QuantizedLinear 继续走量化注册表"
tests:                         # 每次检查都要通过的已有 CPU 测试
  - tests/test_kernel_ops.py
  - tests/test_pi05_cpu_routes.py
budget:
  max_attempts: 5              # 最多尝试次数
  agent_timeout_seconds: 1800  # 单次智能体调用的超时
  max_seconds: 7200            # 整次运行的累计时长上限，中断后继续累计
```

校验规则（`spec.py`），不满足时在建立 worktree 之前就报错：

- 不允许未知字段。
- `model` 必须是标识符。
- `op` 必须是公开的 Python 标识符：不能是关键字，不能以 `_` 开头。
  它按数学含义命名，不能包含模型名。下游调优工具要求 `embodiinfer.backend.torch.ops` 中的参考函数与算子同名，
  `install` 也用它作目录名。
- `sites` 至少一项，且不能重复。
  - `source` 必须是 `embodiinfer/policies/<model>/` 下的 `.py` 相对路径。
  - `function` 是函数名，或 `Class.method` 形式的限定名。
  - `expression` 必须是一个调用表达式，按 `ast.unparse` 规范化后比较。
- `tests` 是 `tests/` 下的 pytest 节点，默认 `tests/test_kernel_ops.py`。这些文件在 BASE 中必须存在。
- `budget` 的值必须是正整数。

也可以不写规格文件，直接用命令行参数。此时 `description`、`preserve`、`tests` 和 `budget` 取默认值：

```bash
$TOOL register new --model pi05 --op linear \
    --site 'embodiinfer/policies/pi05/modeling_pi05.py::_fused_qkv::F.linear(x, weight)' \
    --site 'embodiinfer/policies/pi05/modeling_pi05.py::_mlp::mlp.down_proj(activated)'
```

## 4. 流程

### 4.1 `register new`：建立运行

`new` 的参数：

| 参数 | 含义 |
|---|---|
| `--spec FILE` 或 `--model/--op/--site` | 规格，二选一 |
| `--repository PATH` | 要注册进去的干净仓库，默认是控制器所在仓库。控制器代码有未提交改动时，可以指向另一个干净检出 |
| `--output DIR` | 运行目录的父目录，默认 `results/kernel_tuning/register` |
| `--set test_python=/abs/python` | 测试解释器，必须是已存在的绝对路径 |
| `--yes` | 在记录中标记已确认调用点；`new` 仍会停下，不会启动智能体 |

控制器依次执行以下步骤，任何一步失败都会报错，并撤销已经建立的 worktree 和分支：

1. 确认仓库干净，记录 `BASE = HEAD`。
2. 用 `ast` 定位每个调用点（见 4.2），并确认规格中的测试文件存在。
3. 生成 `run_id`（UTC 时间加随机后缀），建立 `RUN = <output>/<run_id>`。
4. **预检**：`backend/torch/ops.py` 中不能已有同名函数。然后在 `test_python` 中以 `PYTHONPATH=<仓库>`
   导入 `embodiinfer.layers.available_ops`，算子不能已经注册，日志写入 `preflight.log`。
   预检后再次确认仓库仍是干净的 `BASE`。
5. 建立两个 worktree，主检出在整个过程中不被修改：
   - `RUN/base`：detached 检出 `BASE`，作为只读基线；
   - `RUN/worktree`：新分支 `registry/<op>-<run_id>`，智能体只在这里修改。
6. 写入冻结的 `spec.json`（规格和 BASE）、`sites.json`、`sites.md` 和 `manifest.json`。
   此时状态为 `awaiting_confirmation`。

最后打印 `sites.md` 和 `Run: <RUN>`。

### 4.2 审阅调用点

`sites.json` 的每个调用点记录以下内容：

- **位置**：表达式在函数体内必须恰好匹配一处，不进入嵌套函数，记录行号和列号。
- **guard**：外层 `if` 的条件和分支。包含调用点的分支要改；不包含它的兄弟分支、以及调用点之前
  会提前 `return` 的分支，标记为 `preserve_body`，这些分支的代码必须原样保留。
- **`KernelSelection` 的来源**：
  - 函数已有 `kernels` 参数时用 `kernels`；
  - 方法所在文件用到 `self._kernels` 时用 `self._kernels`；
  - 否则需要新增 `kernels` 参数，并由同一文件中列出的调用方逐层传递。
- **调用方**：同一文件内直接或间接调用该函数的位置。
- **源码片段**：调用点前后几行。

`sites.md` 是同一份内容的可读版本。审阅时可以：

- 在 `sites.json` 的 `preserve` 中追加说明；
- 给调用点追加额外字段作为备注。

以下内容不能删改：调用点的 `id`、`source`、`function`、`expression`、`guards`，以及规格中已有的
`preserve` 条目。需要换调用点时，新建一次运行。

### 4.3 `register run`：启动智能体

`run RUN --agent HARNESS/MODEL:EFFORT [--set test_python=…] [--yes]` 执行一次即视为确认调用点，然后：

1. 恢复并校验运行目录（见 4.6）。
2. 用当前或新指定的 `test_python` 在 `RUN/base` 上重新做预检。
3. 将 `sites.json` 与 BASE 上重新定位的结果比对，然后冻结 agent、`test_python` 和 `sites.json` 哈希，状态变为 `ready`。
4. 建立只读上下文 `RUN/context/`，内容如下：
   - `AGENTS.md`；
   - `embodiinfer/backend/torch/ops.py`、`embodiinfer/layers/ops.py`；
   - `embodiinfer/policies/pi05/modeling_pi05.py`：已经通过 `_op` 和 `KernelSelection`
     走注册表的调用点，作为改法示例，取自 BASE；
   - `tests/test_kernel_ops.py`：测试写法示例；
   - 各调用点所在的源文件、`sites.json`、`RULES.md`。
5. 启动 Humanize2 flow `operator_registration`（`flow.py`）。Humanize2 的 workspace 设为 `RUN`，
   智能体的 `LocalEnv` 是从中派生出的 `RUN/worktree`；框架日志归档到 `RUN/workspace/`，
   实际的 epic 路径记录在 `humanize.json` 中。

`run` 只能执行一次。之后使用 `resume`，它不接受更换 agent 或 `test_python`。

**每次尝试的阶段：**

| 尝试 | 阶段 | 会话 |
|---|---|---|
| 第一次 | `plan` → `implement` → 检查 | 新会话 |
| 之后 | `repair` → 检查 | 沿用上一次的会话 |
| 同一项检查连续失败两次后 | `plan` → `implement` → 检查 | 新会话，worktree 保留 |
| 智能体调用出错之后（超时、没写 `PLAN.md` 等） | `repair` → 检查 | 新会话 |

- `plan`：只写 `PLAN.md`，内容包括数学定义及其与每个原表达式的逐项对应、参数中哪些是张量哪些是常量、
  `KernelSelection` 如何到达每个调用点、每个 preserve 分支如何保留、要新增哪些测试。
- `implement` / `repair`：修改代码，并写出 `REGISTRATION.json`：

  ```json
  {
    "op": "linear",
    "reference": "embodiinfer/backend/torch/ops.py::linear",
    "sites": [{"id": "S1", "file": "embodiinfer/policies/pi05/modeling_pi05.py",
               "function": "_fused_qkv", "call": "_op(kernels, \"linear\", False)(x, weight)"}],
    "preserved": [{"site": "S1", "branch": "exact original guard"}],
    "tests": ["tests/test_registry_linear.py::test_reference"]
  }
  ```

每轮智能体调用结束后，`PLAN.md` 必须存在且不为空。单轮超时取 `agent_timeout_seconds` 与剩余总时长中较小的一个。
每个 prompt 都包含冻结的规格、规则全文和上一次的反馈，新开的会话也能独立接手。

**规则摘要**（全文见 `prompts.py` 中的 `RULES`，由检查强制执行）：

1. 参考实现写在 `embodiinfer/backend/torch/ops.py`，函数名等于算子名。带类型注解和说明契约的
   docstring，只导入 `torch` 和标准库。参数只能是张量或常量，返回张量或张量元组，不原地修改输入。
2. 逐字保留原算术：ATen 调用、顺序、形状、中间 dtype 转换和 `contiguous` 调用都不变。
   例如不能把 `F.linear` 改写成 `x @ w.T`，不能合并或拆分 GEMM。
3. 在 `layers/ops.py` 中单独写一行顶层的
   `register_lazy_op("<op>", "torch", "embodiinfer.backend.torch.ops", "<op>")`，
   不放进同时注册 Torch 和 Triton 的循环。
4. 调用点改为 `_op(kernels, "<op>", False)(...)`，即偏好 Torch 实现。`KernelSelection` 通过参数显式传递，不使用全局变量。
5. preserve 分支原样保留。区分普通 `nn.Linear` 时用 `type(module) is nn.Linear`，不按模型名分支。
6. 新增测试覆盖三类行为：
   - 参考实现与原表达式在 CPU FP32 和 BF16 随机输入上 `torch.equal`；
   - 调用点经过一个会记录调用的假调优档案；
   - preserve 分支不经过注册表。
7. 允许修改的范围见 C1。禁止删改已有测试，禁止安装依赖、运行 GPU 代码、执行 `git commit`/`push`/merge、
   修改 Git 元数据或运行目录中控制器拥有的文件。

### 4.4 CPU 检查（C1–C5）

控制器按顺序执行检查，前一项失败时后面的不再执行。所有检查子进程都满足以下条件：

- 以 `PYTHONPATH=<检出目录>` 运行，并断言导入的 `embodiinfer` 确实来自该检出目录（开发环境中
  EmbodiInfer 通常是可编辑安装）；
- 隐藏 CUDA/ROCm 设备，禁止自动加载 pytest 插件；
- 超时或取消时终止整个进程组。

| 检查 | 内容 |
|---|---|
| C1 范围 | 改动文件取自 `git diff <BASE>` 加上所有未跟踪文件（包括被忽略的，缓存目录除外）。只允许出现：`embodiinfer/backend/torch/ops.py`、`embodiinfer/layers/ops.py`、调用点源文件、`tests/test_kernel_ops.py`、`tests/test_registry_<op>.py`，以及 `PLAN.md`、`REGISTRATION.json`。不允许删除文件。已有测试文件只能新增内容，原有行和顶层语句不变。`KernelSelection` 类不变 |
| C2 结构 | ① `REGISTRATION.json` 指向正确的算子和参考实现；覆盖每个调用点恰好一次；列出的测试是 BASE 中不存在的新测试函数，位于上述两个测试文件中，不能是文件或单个参数化实例。② 参考函数是模块级同步函数，带 docstring 和合法注解，没有 `*args`/`**kwargs`，所在文件只导入 `torch` 和标准库，函数内没有原地修改。③ `layers/ops.py` 中恰好有一行对应的顶层 `register_lazy_op`。④ 每个调用点都包含报告的 `_op(..., False)` 调用；`kernels` 是函数参数，同一文件内的调用链逐层传递它；guard 及 preserve 分支与 BASE 相同；原表达式只出现在 preserve 分支中，或出现在 `type(x) is nn.Linear` 不成立的那一支。⑤ 在 `test_python` 中确认：`available_ops()` 包含该算子的 `torch` 实现，`get_op` 返回参考实现，`KernelSelection("none").prepare()` 成功，`KernelSelection("torch").resolve` 返回参考实现 |
| C3 格式 | 改动文件通过 `ruff check` 和 `ruff format --check`（在工具环境中运行） |
| C4 测试 | 在 `RUN/worktree` 中以 `-m "not gpu"` 运行：规格中的测试一起跑；每个新测试单独跑。新测试必须实际执行，不能被跳过 |
| C5 有效性 | 把新测试文件复制到 `RUN/base`，逐个运行新测试。每个新测试的所有用例都必须执行，并以断言失败（退出码 1）。通过、跳过、收集错误或依赖缺失都不能作为有效证据 |

检查结果写入 `attempts/<N>/checks.json`，并与当时源码的指纹绑定。中断后恢复时，只有源码字节未变，
已通过的检查才会复用。检查过程中源码发生变化会被判为失败。

### 4.5 反馈、重试和停止

检查失败时，控制器写出 `attempts/<N>/feedback.json`，下一次尝试的 prompt 会附上它：

```json
{"check": "C2", "summary": "S2: original expression remains outside a preserve branch at line 315", "details": {}}
```

智能体调用本身出错时，`check` 为 `agent`。每次尝试结束时，当时的补丁都会保存为 `attempts/<N>/patch.diff`。

| 结果 | 状态 |
|---|---|
| C1–C5 全部通过 | `passed`，记录已验证的源码指纹，并写出 `RUN/patch.diff` 和 `RUN/SUMMARY.md` |
| 达到 `max_attempts` 或 `max_seconds` | `exhausted`，worktree 保留，供人工接手 |
| 冻结输入被改动（规格、BASE、`RUN/base`、分支、提交历史、已审阅的调用点、只读上下文），或控制器、运行环境出错 | `failed`，不会转为反馈交给智能体重试 |
| Ctrl-C 或取消 | `interrupted`，可用 `resume` 继续 |

### 4.6 `resume` 和 `status`

- `status RUN` 打印 `manifest.json`，不加锁，也不启动智能体。
- `resume RUN` 根据 `manifest.json` 和 Humanize2 的状态台账继续：
  - 已完成的阶段不重跑；
  - 已用时长每秒落盘一次，中断不会重置预算；
  - 状态已是 `passed`、`finalized`、`failed` 或 `exhausted` 时直接返回。

所有修改运行目录的命令都持有 `RUN/.run.lock`。C5 往 `RUN/base` 复制测试之前，会先把这些文件记入
`manifest.json` 的 `base_overlay`；任何失败、取消或下一次启动时，都会先用 BASE 还原这些文件。

### 4.7 `register finalize`：导出和提交

`finalize` 只接受状态为 `passed` 或 `finalized`、且源码指纹与通过检查时一致的运行。
它会重新写出 `RUN/patch.diff` 和 `RUN/SUMMARY.md`。补丁不包含 `PLAN.md` 和 `REGISTRATION.json`。

| 参数 | 行为 |
|---|---|
| 无 | 打印 worktree 路径、分支名、建议的提交说明和产物路径，不修改 Git |
| `--commit` | 只暂存通过 C1 的文件（暂存区里有其他文件时报错），在 worktree 中以 `feat: route <model> calls through the <op> registry operator` 提交。提交使用当前 Git 身份，会运行仓库的 hook。状态变为 `finalized` |
| `--push` | 需要同时传 `--commit`，或此前已经提交。推送前再次确认文件字节与已验证的一致，然后执行 `git push -u origin registry/<op>-<run_id>` |

本工具不提供合并功能。

#### 合并前

CPU 检查不能证明 GPU 上逐位一致。例如 pi0.5 的 native 路径只在 CUDA 上启用，CPU 测试执行不到其中的调用点。
合并前必须在目标 GPU 上按 [0008 §6](../../../docs/proposals/0008-kernel-tuning.md#6-losslessness-and-precision-criterion)
确认：`kernel_profile: none` 时，整模型输出与 BASE 逐位一致。两边要用相同的 checkpoint、dtype、
attention 语义、随机种子和噪声、matmul precision、batch 形状和 decode/denoise 步数，并设置
`TORCHINDUCTOR_DETERMINISTIC=1`。`SUMMARY.md` 的末尾也会写明这一项尚未完成。

## 5. Git 的使用和清理

智能体执行命令时没有交互确认，控制器因此用 Git 做隔离和取证：

- **隔离**：两个 worktree 共享主仓库的对象库，主检出始终不被修改。
- **基线**：`RUN/base` 固定在 BASE。C5 需要在未修改的代码上运行新测试；还原 overlay 时用
  `git restore --source=<BASE>`。
- **防篡改**：每轮都核对 `RUN/base` 的 HEAD 和文件哈希、`RUN/worktree` 的分支和 HEAD。
  智能体自行提交会导致运行失败。
- **范围和证据**：C1 用 `git diff` 和 `git ls-files --others` 列出改动；补丁用 `git diff --binary`
  生成，再补上未跟踪的新文件。
- **交付**：只有在显式传入 `--commit`/`--push` 时才提交或推送。

运行结束后，worktree 和分支会留在仓库里，需要手动清理：

```bash
git worktree remove --force RUN/worktree
git worktree remove RUN/base
git branch -D registry/<op>-<run_id>      # 已推送或已合并后再删
rm -rf RUN
# 如果先删除了 RUN 目录：git worktree prune
```

## 6. 运行目录

```
<output>/<run_id>/
  manifest.json        状态、尝试记录、已用时长、冻结的 agent/test_python、各类哈希
  spec.json            冻结的规格和 BASE
  sites.json sites.md  调用点及审阅说明
  preflight.log        new 的注册表预检；run 的预检写入 test-runtime.log
  context/             智能体的只读上下文（校验哈希）
  base/                BASE 的 detached worktree
  worktree/            分支 registry/<op>-<run_id> 的 worktree
  workspace/           归档的 Humanize2 日志；humanize.json 记录 epic 路径
  attempts/<N>/        <stage>-prompt.txt、<stage>-reply.txt、PLAN.md、REGISTRATION.json、
                       checks.json、feedback.json、patch.diff，以及 registry/ruff/tests/baseline 的日志和 JUnit 报告
  patch.diff SUMMARY.md 通过后的导出
  .run.lock
```

状态的变化顺序为：`awaiting_confirmation` → `ready` → `running` → `passed` → `finalized`。
中途可能变为 `exhausted`、`failed` 或 `interrupted`。

## 7. 代码布局

| 文件 | 内容 |
|---|---|
| `spec.py` | 规格的解析（YAML 或命令行）和校验 |
| `sites.py` | 用 `ast` 定位调用点、提取 guard 和调用方，审阅前后比对，生成 `sites.md` |
| `workspace.py` | `RegistrationRun`：worktree、冻结输入的校验、只读上下文、子进程执行、改动列表、补丁、C1 |
| `checks.py` | C2–C5 以及检查调度 |
| `prompts.py` | 规则和 `plan`/`implement`/`repair` prompt |
| `flow.py` | Humanize2 flow `operator_registration` |
| `finalize.py` | `SUMMARY.md`、补丁导出、提交和推送 |
| `cli.py` | `register` 子命令、环境检查、Humanize2 启动 |

CPU 测试位于 `tests/kernel_tuning/test_registration.py`，使用仓库的临时克隆、真实的 CPU 子进程和
Humanize2 自带的假智能体，不需要 checkpoint 或 GPU。

## 8. 限制

- CPU 检查和有限的测试不能证明 GPU 上逐位一致，也不能保证智能体写的测试覆盖了全部行为。
- 每次运行只注册一个算子，不支持融合。调用点只能位于 `embodiinfer/policies/<model>/` 下，
  且所在文件必须已有 `_op` 集成。`kernels` 传递链的检查只覆盖同一文件内的调用方。
- 检查全部通过的补丁仍可能结构不佳（例如传参链过长），需要人工审阅。
- 目录边界、哈希校验和隐藏加速设备都不构成操作系统级的沙箱。智能体和测试以当前用户的权限运行，
  必要时请在隔离的账户或容器中运行。
