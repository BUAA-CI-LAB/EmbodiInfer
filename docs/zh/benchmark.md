# 性能测试与复现

本页汇总离线推理、引擎优化和强化学习集成的性能结果，并提供测试配置与复现入口。

## 选择基准测试

| 目录 | 模型 | 默认数据 |
|---|---|---|
| [streamvln-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/README.md) | StreamVLN | R2R/RxR 各前 48 个 episode，全帧回放 |
| [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | PI0.5 | LIBERO-10，1,600 帧 |
| [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) | Qwen low / panoramic / NaViDA | R2R/RxR 各前 48 个 episode，每段 4 帧 |
| [cosmos-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/cosmos-benchmark/README.md) | Cosmos Policy | 与 PI0.5 相同的 LIBERO-10，1,600 帧 |

## 测试数据与计时范围

各测试均使用离线回放，采样数量可配置。统一计时包括预处理、推理和结果输出到 CPU，
不包括模型加载、磁盘数据解码和预热。Panoramic 的输入由 RGB 数据构造，形状与模型要求匹配；
Cosmos 测量动作生成耗时。

每个目录都自带 `setup_env.py`，用于创建隔离的虚拟环境；
`benchmark.py` 负责数据采样、计时和推理。Qwen Low/Panoramic 使用 SDPA + Inductor
+ CUDA Graph。每个 README 都给出确切的运行命令。

## 离线推理性能

<!-- offline-results:start -->
以下结果于 2026 年 9 月 7 日在 Jetson AGX Thor 和 RTX 4090 上测得。
各模型逐个运行，使用 B=1、BF16 和对应测试配置中的优化，不计加载、编译和预热耗时。

| 模型 / 数据 | 每卡调用数 | Thor 平均 ms | 4090 平均 ms | Thor calls/s | 4090 calls/s | 4090 / Thor 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PI0.5 / LIBERO-10 | 1,600 | 165.10 | 78.72 | 6.0569 | 12.7037 | 2.10× |
| Cosmos Policy / LIBERO-10 | 1,600 | 1,333.35 | 468.58 | 0.7500 | 2.1341 | 2.85× |
| StreamVLN / R2R | 2,997 | 491.75 | 168.94 | 2.0335 | 5.9193 | 2.91× |
| StreamVLN / RxR | 3,879 | 502.57 | 169.86 | 1.9898 | 5.8872 | 2.96× |
| Qwen Low / R2R | 192 | 250.67 | 135.04 | 3.9892 | 7.4053 | 1.86× |
| Qwen Low / RxR | 192 | 266.48 | 144.15 | 3.7527 | 6.9371 | 1.85× |
| Qwen Panoramic / R2R | 192 | 834.41 | 449.62 | 1.1984 | 2.2241 | 1.86× |
| Qwen Panoramic / RxR | 192 | 833.35 | 447.97 | 1.2000 | 2.2323 | 1.86× |
| NaViDA / R2R | 192 | 1,145.17 | 414.42 | 0.8732 | 2.4130 | 2.76× |
| NaViDA / RxR | 192 | 1,140.57 | 413.53 | 0.8768 | 2.4182 | 2.76× |

每个目录的 README 包含延迟分位数、输出吞吐、峰值内存、
加载和预热时间，以及 graph 重放状态。该表比较了两个平台
在 B=1 下各自使用的 Torch CUDA 构建和驱动。
<!-- offline-results:end -->

## 复现结果

按所选目录的 README 准备运行环境、检查点和数据集，再执行测量命令。

记录结果时，同时保存代码版本、GPU、驱动、依赖版本、检查点、输入样本、精度、批次大小，
以及预热和正式测量次数。比较加速比前，先检查对应的数值一致性结果。

## 用合成策略测试引擎

```bash
python benchmarks/benchmark.py --preset small --sweep   # hf vs embodiinfer-eager vs embodiinfer-graph
```

该脚本在 CUDA 可用时，比较合成策略在单请求 eager、批处理 eager 和
graph 执行下的表现。CPU 回退会走软件路径。若要测量模型吞吐，
请使用上面某个特定于模型的基准测试。

关于 Qwen2.5-VL-3B 和 NaViDA 的 CUDA-graph 对比，参见
[`benchmarks/3B-navigation/`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/3B-navigation/README.md)。

双 GPU 数据并行和张量并行的对比见
[并行](parallelism.md) 页面。

## 各项优化与强化学习集成结果

以下测试分别评估单项优化，保持模型、权重、注意力实现和精度一致。
除特别说明外，单项测试使用一张 H200，RLinf 端到端测试使用四张 H200。

### pi0.5（4.14B，LeRobot `pi05_base` 权重）

- **数值一致性。** 内置前向实现在 fp32 下与 LeRobot 参考实现逐位一致
  （`max|Δaction| = 0`），并且 CUDA-graph 路径在相同噪声下与 eager
  逐位一致。
- **同条件对比。** 与 RLinf 使用的 openpi `PI0Pytorch` 比较，在相同权重、eager、TF32 和 `K = 10` 下，
  EmbodiInfer eager 在 `B = 1..64` 的加速比为 1.01–1.09×。小批次略快；
  批次增大、计算成为主要瓶颈后，两者性能接近。
- **CUDA graph 加速**（eager → graph，TF32）。去噪循环在 `B = 1` 时为 2.29×，
  在 `B = 4` 时为 1.32×，在 `B = 8` 时为 1.07×，随着 batch 增大而收敛；
  引擎端到端加速比在 `B = 1` 时为 2.01×。
- **RLinf 端到端**（LIBERO PPO，4×H200，4 env）。Rollout 生成耗时 11.3–12.6 s/it，
  而原生 openpi 后端为 13.5–14.3 s/it，PPO 指标（ratio、
  approx_kl、clip_fraction、success）处于同一水平。`θ₀` 处的 ratio 分位数落在
  原生后端自身的噪声底内，这来自原生 rollout 与 actor 重算之间的
  精度路径差异。捕获的 graph 在权重同步后
  仍然有效。

### GR00T N1.7（内置 DiT 动作头 + Qwen3-VL 主干，bf16，`N = 4`）

- **数值一致性。** 已通过与 Isaac-GR00T 的对比验证。RLinf 集成的
  `θ₀` 处 ratio 在**每个分位数上都恰好为 1.0**（逐位一致），value 路径
  逐位一致，eval 路径的确定性偏差为零。
- **GPU 去噪耗时**（graph 对比 eager，同步计时）。在 `B = 2/4/8` 时加速比为
  2.83× / 2.53× / 1.89×（15.2/17.2/23.5 ms 对比 43.0/43.6/44.5 ms），
  并在多台机器上复现。
- **完整预测流程**（对比原生 eager 后端，并固定 CPU 预处理
  线程数）。在 `B = 4/8/16` 时为 1.549× / 1.288× / 1.115×。
- **RLinf 端到端**（LIBERO PPO，4×H200，8 env）。graph 和 eager 后端
  耗时接近，差异处于测量波动范围内（中位数 14.5 对比 14.7 s/it）。
  此配置主要耗时在仿真步进和 CPU 工作上，模型调用约占 epoch 的十分之一，
  因此 GPU 加速没有明显降低整体耗时。worker 内的计时与单独测量模型时一致。
  权重卸载导致指针变化、原图失效时，会自动重新捕获；这一机制已在训练循环中验证。

### OpenVLA-OFT（Llama-2 7B + 内置 timm 双塔视觉，单次前向 + 分类头，bf16）

OpenVLA-OFT 通过 `ActionDecoder` 接口在单次前向
中预测动作 token。

- **数值一致性。** 内置 Llama-2 前向与 HF 逐位一致
  （`max|Δ| = 0`）；视觉加投影器与官方实现逐位一致，
  且动作 token 的 argmax 相同；完整链路相对官方动作 logits
  达到 `max|Δ| ≈ 1.6e-4`，这是跨实现的 fp32 噪声。
- **RLinf 初始参数 `θ₀` 下的 ratio。** BF16 分类对数概率对 logits 的舍入误差较敏感，
  流匹配的高斯对数概率则相对稳定。把 `encode_prefix` 的 prefill 与 decode 拆开执行时，
  BF16 舍入使 ratio 落在 `[0.67, 1.57]`。改用与原生算子对齐的单次完整前向后，
  ratio **恰好为 1.0**，对数概率差 `Δ = 0`，两侧参数 key 集合相同。
- **CUDA graph。** `ForwardGraph` 捕获 decode 段，其中 56 个动作 query
  关注缓存的 KV。graph 对比 eager 为 `max|Δ| = 0`，并且在 7B 模型
  计算受限、56 个 token 的情况下，decode 段获得 1.09× 的增益。
- **RLinf 端到端**（LIBERO GRPO，EGL 渲染与模型 CUDA 分开
  放置）。完整前向 rollout 在每个 token 上都与 actor 严格同策略
  （`ratio = 1`、`ratio_abs = 0`、`approx_kl = 0`、`clip_fraction = 0`）。predict
  段的耗时约为原生的 2×，这被受观测限制的 rollout 墙钟时间掩盖，
  因此端到端两者持平。

### LingBot-VLA（Qwen2.5-VL-3B + 窄 Qwen2 MoT 专家，pi0 风格 flow-SDE，bf16，`N = 10`）

- **数值一致性。** 内置 MoT 前向在 VL prefill 阶段生成 KV，专家模块对固定的前缀 KV 计算注意力。
  相比原生参考，速度场预测的平均误差为 `mean|Δ| = 1.4e-3`，尚未达到逐位一致。
- **注意力实现。** 原生 MoT 共享注意力是带 fp32 核心的
  自定义 `our_eager_attention_forward`，而内置模型中并未实现 flash 分支，
  因此 EmbodiInfer 必须用 `eager` 后端与之匹配。使用 `sdpa` 时，
  较小的标准差放大了速度场误差，使 `θ₀` 处 ratio 偏差达到 `±16%`。使用
  `eager` 时收窄到 q01–q99 范围 `[0.987, 1.015]`（`±1.5%`），
  中位数恰好为 1.0。
- **RLinf 权重同步。** 内置模块沿用 actor 的 `nn.Module` 树结构注册参数，
  全部 1,555 个 key 一致，无需名称映射。
- **RLinf 端到端**（RoboTwin click_bell GRPO，分开放置）。原生和
  EmbodiInfer 后端都报告 `ratio = 1.0`；
  `success_once` 在原生下从 0.88→0.97，而 EmbodiInfer 为 0.92→0.98；`approx_kl`
  原生为 5.5–9e-4，对照为 1.3–6.2e-3，略高，可追溯到前向
  残差，与离线的 `±1.5%` 一致，且仍在 GRPO clip 之内；rollout
  和 predict 持平，为 55 ms 对比 53–58 ms。

### Cosmos Policy（Cosmos-Predict2 2B 视频扩散 DiT + Wan2.1 VAE，bf16，`N = 5` 步去噪）

Cosmos 使用扩散解码器进行动作生成和 best-of-N 规划。
以下对比以 cosmos-policy 的推理实现
作为参考。

- **正确性**（一块 H200，bf16，固定初始噪声，与 cosmos-policy 的原生
  `generate_samples_from_batch` 对齐，LIBERO Predict2-2B）。内置 DiT 前向
  `max||Δ||_inf = 0.032`（均值 3.4e-3，其中网络输出的均值和标准差
  与原生实现逐位一致）；Wan2.1 VAE 编码 `max||Δ||_inf = 0.024`；
  端到端 latent `max||Δ||_inf = 0.024`；动作帧（归一化）1.9e-3；value
  帧 7e-4。
- **Best-of-N 规划。** `encode_prefix`（VAE 加文本）只运行
  **一次**，并由 `expand(N)` 广播；`N` 条扩散轨迹联合对动作帧
  和 value 帧去噪，再选择价值最高的候选。前缀编码耗时 68.9 ms，单个候选
  （5 步 DiT）为 159.6 ms，因此规划延迟在 `N = 1/2/4/8` 时为
  229/350/580/1032 ms。
- **前缀复用收益**（同一次引擎测试，`N = 4`）。复用耗时 579 ms
  （VAE 加文本编码一次），而每个候选重新编码为 909 ms，节省
  330 ms，约 36%。DiT 对所有帧进行双向注意力，且没有
  clean-frame KV 缓存，因此复用仅限于 encode 阶段。
- **CUDA graph。** Cosmos 声明 `supports_cuda_graph=True`，并设置
  `cuda_graph_kind="diffusion_step"`，`GraphManager` 会在与 flow 和 OFT
  相同的 `engine/graph.py` 分发中，把每步的 `denoise`（preconditioning、DiT、帧替换）
  捕获为 `CosmosDenoiseGraph`。静态 prefix 只拷入一次，
  并在每一步重放。graph 对比 eager 得到 latent `max||Δ||_inf = 0`
  （逐位一致），5 步的 `sample_latent` 从 157.6 ms 降到 138.2 ms，即
  1.14×。2B DiT 主要受计算量限制，图捕获只减少每步启动开销；
  2ab 多步算法的主机端 float64 运算仍在图外，进一步限制了总体收益。
  单独测量 DiT 前向时，加速比为 1.36×。
- **完整动作生成。** `Vvla("cosmos").act(obs)` 先通过 `collate` 将 `Observation` 转成 `CosmosBatch`：
  两路相机输入从 `[0,1]` 映射到 `[-1,1]`，缩放机器人自身状态，并为 T5 交叉注意力读取预计算的指令表示。
  随后运行 `encode_prefix` 和扩散去噪，返回形状为 `ActionChunk (16,7)`、使用数据集尺度的动作块，
  同样支持批处理。未预计算的指令由内置 `T5TextEncoder`（`google-t5/t5-11b`）编码。

### 如何选择优化方式

- CUDA graph 更适合小批次、算子启动开销占比较高的场景；批次增大后，加速比逐渐接近 1×。
  能否降低整个 rollout 的耗时，取决于 GPU 计算在流程中的占比：本次测试中，pi0.5 收益明显，GR00T 较小。
- CPU 预处理测试应固定线程数，并与实际部署一致。RL worker 通常为单线程，
  过多线程竞争可能使 CPU 计时偏差超过一个数量级，导致误判瓶颈。
