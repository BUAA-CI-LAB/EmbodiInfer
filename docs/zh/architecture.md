# 推理引擎架构

EmbodiInfer 用统一接口连接推理引擎和模型适配器。
引擎负责批处理调度、CUDA graph、并行执行、rollout 和会话管理；
适配器负责各模型的观测编码和动作解码。

## 从观测到动作

流匹配 VLA 将观测 `o = (images, language, state)` 映射为未来的
动作块 `a ∈ R^(H×A)`。计算分为两个阶段。

1. **多模态前缀编码。** 图像、语言和机器人自身状态由
   视觉编码器和 VLM 主干网络编码为特征序列
   `c ∈ R^(P×D)`。该阶段受计算限制。
2. **去噪。** 动作头从噪声出发，在 `N` 步内积分速度场
   `v_θ(x_t, t | c)` —— 确定性 Euler，或随机 SDE 形式 ——
   每一步都关注固定的前缀 `c`。

根据这一计算结构，可以从三个方面优化推理：

- **捕获静态计算。** 去噪步数 `N` 固定，各步张量形状相同，时间调度可预先计算。
  去噪循环因此适合捕获为 CUDA graph，后续重复回放，消除逐个算子从主机启动的开销。
  小批次推理中，这部分开销尤其明显。
- **复用前缀。** 同一次预测的 `N` 个去噪步骤使用相同观测，前缀及注意力 K/V 只需计算一次。
  best-of-N 规划中的多个候选也可共用这份前缀。
- **按请求节奏调度。** 并行仿真器通常同时产生观测，适合合并请求执行。
  异步 rollout 的请求到达时间较分散，则可在不同 CUDA 流上重叠执行当前批次的 prefill 和上一批次的去噪。

引擎据此提供跨环境批处理、去噪 CUDA graph、前缀 K/V 复用和 prefill/去噪流水线，
并为强化学习提供权重同步与对数概率接口。

## 应用、引擎与训练框架的分工

应用负责提供观测并执行返回的动作，EmbodiInfer 负责模型计算。
接入强化学习时，训练框架还负责更新权重和计算训练目标。

部署推理端点时使用[网络服务](serving.md)，直接嵌入应用或训练框架时使用 [Python API](api.md)。
[基准测试指南](benchmark.md)记录了在特定环境和数值条件下
与参考实现的对比。

## 策略与解码器接口

模型通过 `VLAPolicy` 向引擎提供多模态前缀编码（`encode_prefix`）、观测整理与填充、
CUDA graph 前缀流水线和 `decoder`。

`ActionDecoder` 定义推理所需的 `init_state` 和 `produce_chunk`，并提供默认 `decode` 封装。
`RLDecoder` 在此基础上增加 on-policy 训练能力：动作采样和可微的对数概率重计算，
用于通用策略梯度 rollout。

解码方式由策略对象提供，引擎不按模型名称分支。已实现的解码器
包括 `FlowDecoder`（pi0.5 / GR00T N1.7 / LingBot-VLA）和 `ParallelDecoder`
（OpenVLA-OFT），它们是 `RLDecoder`；此外还有 `CosmosDiffusionDecoder`（仅规划
的 Cosmos Policy）和 `AutoregressiveDecoder`（有状态 token 解码，ActiveVLN），它们
是普通的 `ActionDecoder`。

## 同步执行：EngineCore {#enginecore}

`EngineCore` 是同步执行路径。无状态请求运行 `encode_prefix`，
然后运行解码器的 `produce_chunk`；有状态请求通过
`SessionStore` 运行一个事务。

## 会话状态：SessionStore {#sessionstore}

`SessionStore` 管理模型会话状态，提供 checkout、commit、rollback、reset 和 cancel 操作。
会话通过 `SessionKey`（`env_id`、`episode_id`、`rollout_id`）标识，引擎无需了解内部 KV 布局。
只有 prefill、token 追加、结果解析和封装全部成功，才提交事务；异常时保留上一次已提交的状态。

## 异步批处理：AsyncEngine {#asyncengine}

`AsyncEngine` 在 `max_wait_ms` 窗口内收集就绪请求，每批最多 `max_batch_size` 个。
批次按 `batch_buckets` 填充到最近的可容纳桶大小，以复用 CUDA graph。
这一收集窗口适合各环境推进速度不同的情况；整个批次同时步进的同步向量化 rollout 直接调用 `EngineCore.execute`。

## CUDA Graph 捕获与重放 {#cuda-graph-capture}

策略通过 `supports_cuda_graph` 和 `cuda_graph_kind` 声明图捕获能力。
去噪循环的 `N` 步使用相同形状，引擎按 `(batch bucket, N)` 缓存图，并在首次使用时捕获。
原地更新权重（refit）不改变参数指针，因此已捕获的图仍可使用。支持三种捕获方式：

- **单步**（`DenoiseGraph`）。捕获一个去噪步，由 Python 循环
  驱动 `N` 次回放；前缀在每次预测时复制到静态缓冲区一次。
- **完整循环**（`LoopGraph`）。整个 `N` 步积分（包括图内
  Euler 步和预先计算的时间调度）捕获一次、回放一次，因此
  步间主机开销从 `O(N)` 降至 `O(1)`。单步捕获已消除大部分启动开销，
  完整循环的额外收益主要是把 CPU 工作压缩为一次提交，便于与其他 CUDA 流重叠执行。
- **SDE 循环**（`SdeLoopGraph`）。整个随机 SDE 积分（包括
  每步噪声注入以及 RL 对数概率所需的轨迹和速度记录）
  捕获一次。噪声在图外抽取到
  静态缓冲区，因此捕获的路径与 eager 参考保持逐位可比。
  每步噪声缩放和分数修正系数存放在设备侧
  系数缓冲区中，因此更改所选的噪声步只需复制缓冲区，而无需
  重新捕获。时间方向 —— 噪声在 `t=0` 还是 `t=1` —— 由
  flow schedule 步长的符号决定，因此同一实现同时服务于 pi0.5 和
  GR00T 约定。

前缀捕获由各策略配置，例如 π0.5 原生运行时可启用 `prefix_cuda_graph`。
`GraphManager` 通过 `cuda_graph_kind` 选择解码策略。
比较图重放和 eager 输出时，保持输入、噪声、dtype 和时间调度一致。
各模型的基准测试提供后端数值对比。

对于 PI0.5，`attention="eager"` 还会选择参考投影布局：相机
视图分别编码，Q/K/V 以及 gate/up 投影保持分离。融合
后端保留批量和融合路径。这一区别对于
在选择性转换的检查点上进行 rollout/actor 对数概率比较很重要：即使注意力公式相同，
改变 GEMM 形状也可能改变舍入。通用的
`denoise_step` 在图内计算当前时间条件和 AdaRMS 投影，
因此无法按输入 `data_ptr` 复用预热结果 —— 同一个静态
缓冲区每一步接收到不同的值。对于固定 schedule，原生运行时会
显式传入预先计算的 `modulations`，而不是依赖通用路径的
基于地址的缓存。

## 前缀 K/V 复用

`encode_prefix` 返回一个 `PrefixState`，其中保存每一层的注意力 K/V，
解码器在所有 `N` 个去噪步中复用它，因此受计算限制的 VLM 主干网络
每次预测只运行一次，而不是每步运行一次。对于规划，`expand_prefix`
将同一前缀广播到 `N` 个采样候选，因此主干网络对整个
group 只运行一次。

进入新的环境步后，图像发生变化，需要重新编码前缀。
GRPO 等轨迹级分组采样的候选对应独立环境轨迹，只有重置后的首帧前缀相同，可以共用。

## RL rollout 与权重更新

`GenerationBackend` 暴露 RL 训练器所需的最小接口：`generate`（确定性，
走 CUDA graph 快速路径）、`generate_with_logprob`（随机采样加
对数概率）、`best_of_n` 和 `refit`（原地权重更新，使已捕获的
图保持有效）。`VLAPolicy.refit_state_dict()`
加上 `commit_refit(version=...)` 增加了零拷贝的两阶段协议：训练
框架传输权重、应用分片并映射参数名；
EmbodiInfer 校验并提交策略版本。提交会在发布新
版本之前运行策略的 `on_refit` 运行时刷新。如果刷新失败，
之前的版本仍然可见，但已写入的张量
不会回滚，因此调用方必须修复或丢弃该策略并重试。

对数概率使用 SDE 形式：确定性 ODE 采样器被改写为
随机过程

```text
x_{k+1} = μ_k + σ(t_k)·sqrt(|Δt|)·ε_k

log p += log N( x_{k+1}; μ_k, σ(t_k)²·|Δt|·I )
```

其中转移均值 `μ_k = x_k + v_θ·Δt − c_k·(x_k + γ_k·v_θ)` 携带
分数修正项，`σ(t)` 是可配置的噪声 schedule（`σ = 0` 使该
步恰好为确定性 ODE 步）。系数 `γ_k` 和 `c_k` 由
flow schedule 的时间方向得出，并逐算子与目标 RL
框架的原生生成器匹配，包括 dtype 提升顺序 —— 因此数值比较必须同时包含 rollout 和 actor 重计算。还提供了可微的 `flow_logprob_recompute`；
它在当前参数下对已存储的轨迹重新评分，并给出
可用的梯度，供训练器侧的 PPO/GRPO 损失使用。

## Prefill 与去噪并行执行

`execute_pipelined` 可在两个 CUDA 流上重叠执行当前批次的去噪和下一批次的 prefill。
去噪使用 CUDA graph，几乎不占 CPU；prefill 使用 eager，并通过 `torch.cuda.MemPool`
分配内存，避免跨流的缓存分配竞争影响数值一致性。
该功能默认关闭，适合请求到达时间分散的异步 rollout 或在线服务。
实测中，将同步 rollout 批次拆开反而更慢：去噪耗时对批次大小不敏感，拆分增加了总去噪时间。

## 代码结构

```text
embodiinfer/policies          VLAPolicy + VLAPolicyConfig + ActionDecoder/RLDecoder + model adapters
embodiinfer/models           engine-agnostic nets by family — video_dit / video_vae /
                       text_encoders / schedulers (diffusion + flow math)
embodiinfer/layers            operator contracts + registry + backend routing
embodiinfer/backend           concrete Torch/Triton implementations, capability probes,
                       warmup, graph-safe execution
embodiinfer/engine            EngineConfig · EngineCore · transactional SessionStore ·
                       AsyncEngine · graph
embodiinfer/engine/parallel   data-parallel replicas/dispatch + tensor-parallel
                       primitives/sharding plans
embodiinfer/engine/rollout    GenerationBackend · flow/categorical log-probability ·
                       weight sync (rollout/demo/ = toy trainer + env)
embodiinfer/engine/serve      EmbodiInfer API + HTTP, WirelessComm, and WebSocket frontends
```

工具位于顶层：`benchmarks/`、`examples/`、`scripts/`、`tests/`。

`embodiinfer/models/` 存放可复用的、与引擎无关的组件。它们不得导入
`policies` 或 `engine`，也不感知 `Observation` 或 `ActionChunk`；它们
按模型族组织，由策略按需组合。与特定策略的前向计算紧密耦合的网络，例如 pi0.5 的 Gemma、
OpenVLA-OFT 的 Llama，仍放在对应的
`policies/<name>/` 中。

### 模块职责与依赖

| 层级 | 目录 | 负责 | 约束 |
|---|---|---|---|
| policy | `embodiinfer/policies/<name>/` | 检查点加载、前向、collate/pad、`flow_schedule` | 模型特定逻辑留在这里；实现基础契约 |
| engine | `embodiinfer/engine/` | 调度、执行、CUDA graph | 仅依赖公开的策略契约（`VLAPolicy`、`PrefixState`、`flow_schedule`、`encode_prefix`、`denoise_step`、`supports_cuda_graph`、`cuda_graph_kind`、`allocate_static_prefix`、`copy_prefix_into`） |
| 算子路由 | `embodiinfer/layers/` | 注意力及其他 Protocol、注册表、后端选择 | 不含具体算子实现；调用方依赖协议和已注册名称 |
| 计算后端 | `embodiinfer/backend/` | Torch/Triton 实现、能力探测、预热 | 不得依赖模型、策略、引擎或环境语义 |
| rollout | `embodiinfer/engine/rollout/` | RL rollout 接口、对数概率、权重同步 | 依赖引擎，绝不依赖具体策略 |
| serve | `embodiinfer/engine/serve/` | 模型中立的推理 API 和前端 | 仅做通信和引擎调用；仿真器和机器人协议属于部署运行时 |

## 与 EmbodiRun 通信

EmbodiRun 通过 HTTP 或 WirelessComm 提交图像、状态、指令和
会话 ID。EmbodiInfer 返回动作块、计时和策略
版本。随后 EmbodiRun 将这些动作映射为设备命令并管理
控制循环。协议用法见[推理服务](serving.md)。

## 接入新模型

要添加模型，实现其策略并声明支持的能力：

1. 实现 `VLAPolicy.encode_prefix`。
2. 实现服务用的 `ActionDecoder`。
3. 仅当满足通用策略梯度契约时才继承 `RLDecoder`。
4. 仅当静态形状可捕获时，才声明 `supports_cuda_graph` 和
   `cuda_graph_kind`。
5. 当状态跨调用时声明 `is_recurrent`，并让引擎的 `SessionStore`
   管理事务生命周期。

新适配器的设计提案、数值一致性验证和文档要求见
[贡献指南（英文）](https://embodiinfer.readthedocs.io/en/latest/contributing/)。

## 数值一致性要求

VLA 模型对精度敏感。因此优化默认必须是无损的，并且
必须附带可复现的判据。

- **逐位一致（bit-exact）** —— `max_i ||Δa_i||_inf = 0`。适用于数学
  相同且浮点累加顺序相同的改写，例如完整循环图
  捕获对比逐步 eager 执行，或静态缓冲区对比每一步新
  分配的张量。这是引擎层改动的默认目标。
- **数值等价（numerically equivalent）** —— `max_i ||Δa_i||_inf <= ε`，通常 `ε ~ 1e-6`，
  差异可归因于浮点重排序，例如归约顺序改变。使用这一级别
  要求说明来源以及为何可接受。
- **相同精度条件** —— 任何比较都必须使用相同的 dtype、相同的
  `torch.set_float32_matmul_precision` 设置以及相同的注意力语义。
  改变精度条件不属于无损优化，而是一项单独的
  改动。

RL 集成还增加了它自己的判据：rollout 侧的对数概率由
actor 重新评分，重要性比分位数则与原生后端在 `θ₀` 处
自身的噪声底进行比较。

测量结果（包括引擎机制和 RL 集成的数字）见
[基准测试](benchmark.md)页面。

## 当前实现范围

- Cosmos Policy 支持 best-of-N 规划和逐步骤图捕获。通用
  RL rollout、完整循环捕获和树搜索规划尚未实现。
- 对数概率是 SDE 代理。它与精确边缘似然的偏差
  尚未量化；其在 PPO 中的有用性取决于 `θ₀` 处比值的判据以及
  端到端训练的健康状况。
- 跨进程 NCCL 权重同步目前只有原型。RLinf 集成使用
  框架自身的同步机制，该机制原地更新权重并保持
  与已捕获图的兼容。
- CUDA graph 要求静态形状：批处理按桶分组，指令长度
  填充到固定值。
- ActiveVLN 使用 eager 执行、`B=1` 和显式会话。它没有
  `RLDecoder` 实现；真实检查点的 GPU parity 校验尚未完成。
- 前缀捕获、相机批处理和注意力实现取决于
  策略和所选后端。在应用优化或复用数值结果之前，
  请查阅该模型的基准测试配置。

## 参考文献

- pi_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models.
  arXiv:2510.25889.
- RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of
  Vision-Language-Action Models. arXiv:2510.06710.
- openpi 远程推理：`github.com/Physical-Intelligence/openpi`。
- Isaac-GR00T: `github.com/NVIDIA/Isaac-GR00T`.
