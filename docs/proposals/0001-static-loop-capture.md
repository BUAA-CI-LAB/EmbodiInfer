# 0001 — 全循环 CUDA-graph 捕获与静态调度预计算

- 状态：Implemented
- 日期：2026-07-06

## 1. 摘要

将 flow-matching 去噪循环由「单步捕获 + Python 驱动的 N 次重放」改为「整条 N 步积分循环一次性捕获、单次重放」。这是**引擎层、模型无关**的优化：它把每步残留的主机侧（host）开销（时间张量构造、输入拷贝、逐步 launch、Euler 更新）从 $O(N)$ 降到 $O(1)$。目标为逐位无损（bit-exact），预期收益集中在小 batch 的 launch-bound 区，是现有单步 CUDA graph 之上的进一步增量。

## 2. 动机与现状差距

现状去噪循环在 `embodiinfer/engine/core.py:73-76`：

```python
for t_val, dt in self.policy.flow_schedule(num_steps):
    t = torch.full((bucket,), t_val, device=self.device, dtype=self.dtype)  # 每步 host op
    v = graph.run(x, t) if graph is not None else self.policy.denoise_step(x, t, prefix)
    x = x + v * dt                                                          # Python 驱动的 Euler
```

现有 CUDA graph（`embodiinfer/engine/graph.py` 的 `DenoiseGraph`）捕获的是**单步** `denoise_step`，循环仍由 Python `for` 驱动。因此单步捕获消除了「步内」的逐算子 launch，但保留了「步间」的主机开销，每次预测累计 $N$ 份：

1. `torch.full((bucket,), t_val, ...)` —— 每步在 host 构造时间张量；
2. `DenoiseGraph.run` 内 `self._x.copy_(x)` 与 `self._t.copy_(t)` —— 每步两次 H2D 拷贝（`graph.py:67-68`）；
3. `self._graph.replay()` —— 每步一次重放 launch；
4. `x = x + v * dt` —— 每步一次 Euler 更新的 launch。

关键观察：`flow_schedule(num_steps)` 返回的 $\{(t_k, dt_k)\}_{k=0}^{N-1}$ 在循环开始前**完全确定**（对 pi0.5 为 $t_k = 1 + k\,dt,\ dt = -1/N$，见 `policies/pi05/modeling_pi05.py:198-200`），唯一沿链传递的状态是 $x$。即去噪积分是一条**静态控制流**：

$$x_{k+1} = x_k + dt_k \cdot v_\theta(x_k, t_k;\ \text{prefix}),\qquad x_0 = \text{noise}.$$

静态控制流意味着整条循环可被一次捕获、一次重放。

## 3. 目标与非目标

**目标**

- 引擎层、模型无关地消除去噪循环的步间主机开销。
- 逐位无损（§6）。
- 与现有单步捕获、无 graph 的 eager 路径并存，通过开关切换，关闭时零行为差异。

**非目标**

- 不改精度口径（dtype / matmul precision / attention 语义均不动）。
- 不做 continuous batching（收益条件过窄，另议）。
- 不做任何模型特异的算子融合（违反 §2.4，收益不可跨模型继承）。
- 不引入 `torch.compile`（见 §4 备选 C）。

## 4. 设计

**核心**：新增 `LoopGraph`，捕获整条 $N$ 步积分。捕获时在私有流上按 `flow_schedule` 展开 $N$ 步循环体（`denoise_step` + 图内 Euler 更新 + 图内时间取值），录制为单一 CUDA graph。每次预测：拷入初始 noise、`set_prefix` 一次、重放一次、读出 $x_N$。

**静态调度预计算**：循环前将 $\{t_k\}$ 一次性构造为静态 device 张量 `t_all`（形状 `[N, bucket]`），$\{dt_k\}$ 作为捕获期常量。捕获后 `t_all` 地址固定，图内按步索引，`torch.full` 与逐步 `copy_(t)` 被完全消除。

**图内 Euler**：递推 $x_{k+1} = x_k + dt_k v_k$ 在图内对静态缓冲原地进行。Euler 更新是引擎侧通用逻辑（不依赖任何具体模型），因此**无需 policy 新增接口**，模型无关性得以保持。

**接口与共存**（改动集中在 `engine/`，不触及 policy 协议）：

- `engine/graph.py`：新增 `LoopGraph`，复用现有 `allocate_static_prefix` / `copy_prefix_into` / `set_prefix` 机制与「私有流 warmup 后捕获」的 allocator-safe 模式（`graph.py:46-55`）。`GraphManager` 按 `(bucket, num_steps)` 缓存（`num_steps` 进入键，因循环长度已入图）。
- `engine/core.py`：当开关开启且 `policy.supports_cuda_graph` 为真时，取 `LoopGraph` 并以「拷入 noise → set_prefix → 单次重放 → 读出」替换 `core.py:73-76` 的 Python 循环；否则回退现有单步或 eager 路径。
- `engine/config.py`：新增 `capture_full_loop: bool`，默认 `False`；在精度与基准验证通过后再评估是否置为默认开。

**备选方案与取舍**

- **A. SGLang 式 overlap scheduler（双缓冲，CPU 调度与 GPU 前向重叠）**。overlap 是为 LLM decode 那种**数据依赖的动态控制流**设计的（下一步依赖上一步采样结果，无法整体捕获，只能把主机调度藏到 GPU 计算之后）。flow 去噪是静态控制流，可整体捕获，全循环捕获比双缓冲更彻底且实现更简单，无需维护双缓冲与同步。overlap 仅在「无法整体捕获」的情形（动态步数、图内含不可捕获算子）才作为回退手段。据此选择全循环捕获。
- **B. 保留现状（单步捕获 + Python 驱动）**。作为 `LoopGraph` 不可用时的回退路径保留，不删除。
- **C. `torch.compile(mode="reduce-overhead")`**。功能上等价于 CUDA graph，但对捕获边界与 graph break 的控制力弱、可控性差；手写捕获路径可控且其单步形态已在既有工作中验证为无损。故不引入 compile。

## 5. 模型无关性判定

引擎层功能。仅依赖既有公共协议：`flow_schedule`、`denoise_step`、`supports_cuda_graph`、`allocate_static_prefix`、`copy_prefix_into`。图内 Euler 与时间预计算为引擎侧通用逻辑，不含任何模型假设。任何满足 `supports_cuda_graph` 的 flow policy（当前 pi0.5，未来 Cosmos/WAM）自动受益，无需 policy 侧改动。

## 6. 无损性与精度判据

**级别**：bit-exact。全循环重放与现状路径执行相同数学、相同浮点累加顺序（相同的逐步 `denoise_step`、相同的 Euler 系数、相同的时间取值），差异应严格为零。

**判据**：固定初始 noise，比对 `LoopGraph` 路径与「单步 eager 路径」及「现有单步 CUDA graph 路径」的输出动作，要求

$$\max_i \lVert \Delta a_i \rVert_\infty = 0,$$

对 batch $\in \{1,2,4,8,16,32\}$ 逐档验证。

**条件记录**：dtype、`matmul precision` 口径、$N$、场景与上述保持一致；捕获路径与对照路径除「是否整体捕获」外的一切设置相同。

## 7. 实现计划

| 文件 | 改动 |
|---|---|
| `embodiinfer/engine/graph.py` | 新增 `LoopGraph`（展开捕获 $N$ 步 + 图内 Euler + 静态 `t_all`）；`GraphManager` 键改为 `(bucket, num_steps)` |
| `embodiinfer/engine/core.py` | `capture_full_loop` 开启时切换到单次重放路径；否则回退 |
| `embodiinfer/engine/config.py` | 新增 `capture_full_loop: bool = False`，注释其对应设计主张 |

向后兼容：开关默认关，关闭时 `core.py` 走现有路径，行为逐位不变；`LoopGraph` 不可用时回退单步捕获。

## 8. 测试计划

- **CPU（mock policy，默认 CI 口径）**：验证静态调度预计算的正确性——`t_all` 与逐步 `torch.full(t_val)` 序列逐位相等；以 mock policy 在 Python 展开循环上验证「预计算路径」与「现状逐步路径」数值一致（CPU 无 CUDA graph，此处只验预计算与循环等价逻辑，不验捕获本身）。
- **GPU（`@pytest.mark.gpu`）**：真实捕获 parity，达成 §6 的 `max|Δaction| = 0`；`(bucket, num_steps)` 缓存命中与回退分支的行为。
- **pi0.5（`@pytest.mark.pi05`）**：真实权重下的端到端 parity（`LoopGraph` 对单步 eager），复用 `tests/test_pi05_parity.py` 的门控方式（`EMBODIINFER_PI05_CKPT`）。

## 9. 基准计划

- **对照**：同 policy、同精度口径、同 $N$、同场景下，`eager` / 单步 CUDA graph / `LoopGraph` 三条路径的 obs/s 与逐步延迟。
- **预期收益区间**：在单步 CUDA graph 之上，进一步消除步间的 $O(N)$ 主机往返（$N=10$ 时约 9 次循环迭代的 host 开销）。收益集中在小 batch 的 launch-bound 区（真实并行 env 数少的场景）；随 batch 增大进入 compute-bound，与单步捕获一样收敛至 ~1×。
- **诚实边界**：这是单步 CUDA graph 之上的**边际增量**，非数量级改进；大 batch 无收益属预期。基准脚本与日志归档到工作目录 `dev/scripts` 与 `dev/logs`（不进 git）。

## 9.1 实测结果（2026-07-06，单 H200，eager，TF32）

**正确性**：达成 §6 判据。mock/small 与 pi0.5 真权重（4.14B）在全 batch
（$B \in \{1,2,4,8,16,32\}$）均为 $\max_i \lVert \Delta a_i \rVert_\infty = 0$
（full-loop 对 eager、对 per-step graph 皆逐位一致）。

**收益（负面结论）**：full-loop 相对既有的单步 CUDA graph **无可测增量**——pi0.5 全
batch `loop/per_step = 1.00x`（mock 1.01–1.02x）。单步 CUDA graph 已吃满 launch-bound
收益（pi0.5 $B=1$：eager 7.22 → per-step 14.17 obs/s，即 1.97x）；full-loop 额外消除的
步间 host 开销（$N$ 次时间张量构造、Euler launch、$N$ 次重放并为一次）在真实模型每步 GPU
约 7 ms 下占比趋近零。mock 上出现的 1–2% 源于 mock 每步过快（144 obs/s）放大了 host 占比。
据此判断：**launch/host 开销的瓶颈在步内（已由单步 graph 解决），步间不构成瓶颈。**

**保留决定**：机制正确、逐位无损，且构成后续 prefill+denoise 全链捕获的一半（图内 Euler +
静态调度）。作为独立优化零增量，故 `capture_full_loop` 默认关闭。

## 10. 风险与局限

- **图数量**：按 `(bucket, num_steps)` 缓存使组合增多，抬高首次捕获延迟与显存占用。缓解：`num_steps` 通常取 policy 默认值（pi0.5 为 10），组合有限；缓存惰性建立。
- **捕获安全性**：图内 Euler 需对静态缓冲原地递推，沿用现有「私有流 warmup 后捕获」的 allocator-safe 模式；若某 policy 的 `denoise_step` 含捕获不安全算子，`supports_cuda_graph` 已在上游门控，此时回退单步或 eager。
- **CPU 不可验捕获**：CUDA graph 仅限 CUDA 设备，CPU 测试只覆盖预计算与循环等价逻辑，捕获 parity 必须在 GPU 环境完成（§8）。
- **收益依场景**：若目标 rollout 的并行 env 数落在 compute-bound 区，本优化收益趋近于零；是否推进应结合目标场景的并行度。

## 2026-09-17：混合精度 checkpoint 的完整去噪图

此次扩展解决 CUDA `dtype="auto"` 把全部权重转成首个参数精度的问题。
π0.5 微调 checkpoint 的 Gemma 权重为 BF16，视觉、归一化、动作和时间投影为
FP32；统一转换既改变动作，也改变显存需求。引擎现在保留每个参数和 buffer
的 dtype，通过模型无关的 `VLAPolicy.execution_dtype` 属性确定观测和去噪状态
精度。默认属性仍返回首个参数精度，π0.5 返回动作投影精度。CPU 保持 FP32，
显式 `dtype="bfloat16"` 等仍表示统一转换。KV 是 policy 私有状态，静态图缓存
按实际 prefix KV 的 dtype 分配，含已有 compact-layout 路径。

配套的 policy 选项 `native_embeddings=True` 使用 EmbodiInfer 的逐相机 SigLIP、文本
嵌入、默认 RoPE、时间编码和掩码计算。默认 `False` 保留现有 LeRobot 嵌入及
批量相机/编译路径；不改已有 `native_inference` 专用 runtime 的默认选择。
`vision_attention` 默认 SDPA，与 Gemma 的 `attention` 独立。
`LeRobotPi05Adapter` 提供 B=1 的 `predict_action_chunk` / `reset`，返回截取
真实动作维数后的归一化张量，由调用方沿用 checkpoint 后处理。它显式启用
原生嵌入、Gemma eager 和完整去噪图；不包含机器人通信、动作执行或任务指标。

备选方案是把模型全部转成 FP32 或 BF16，再让输入、KV 服从同一种精度。
这虽省去 dtype 区分，却无法保留当前 checkpoint 的原生算术，因此未采用。
保留 LeRobot 整体视觉 forward 仍是兼容选项；原生嵌入通过开关启用，便于对照
且不改变现有融合优化的路径。非默认 RoPE 在原生嵌入路径明确报错。

验证要求：相同图像、状态、文本、FP32 初始 noise、matmul precision、B=1、
10 次去噪和 50 步输出。eager 与完整图比较全部归一化和反归一化动作，要求
`max|Δaction| = 0`；Gemma SDPA 的不同归约顺序单独报告，不作为无损替代。
CPU 测试覆盖混合精度采样、实际 KV dtype、五个非持久 buffer（含共享别名）
恢复和错误路径；小型 LeRobot 组件对照覆盖 Gemma、SigLIP、时间和掩码；
CUDA 测试用不同观测连续重放图，检验 prefix 更新和完整去噪结果。
录制数据复现入口为 `benchmarks/pi05-benchmark/compare_recorded.py`，完整条件
和 AGX 实测结果见该目录 README 的 AGX 小节。

限制：Engine 仍将整个 policy 移到 CUDA，不能保留已有 CPU 词嵌入 offload；
这次部署测量约增加 1 GiB 的 GPU 分配。原生嵌入不承诺不同硬件/融合后端逐位
一致；源码迁移后的实权重回放与此前部署分支的性能测量分别记录。
