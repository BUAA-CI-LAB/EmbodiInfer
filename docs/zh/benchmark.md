# 性能测试与复现

本页汇总离线推理、闭环导航和强化学习集成的性能结果，并提供测试配置与复现入口。

最新 π0.5 动图与两组完整性能表见 [README](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/README.zh-CN.md#性能)及
[π0.5 benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md)。
下方按工作负载提供各模型的详细报告与复现入口。

## 最新结果

每个模型在每台设备上最近一次的 batch=1 结果。各行的工作负载、权重和计时范围不同，只在行内对照；完整条件以所链接的报告为准。
日期为报告记录的测量日期；报告未记录时为结果发布的月份。

| 模型 | 设备 | 工作负载 · 精度 | EmbodiInfer | 对照 | 日期 | 报告 |
|---|---|---|---:|---|---|---|
| π0.5 | Jetson AGX Thor | FlashRT v0.2.0 测速脚本 · NVFP4 / FP8 / BF16 | 20.04 / 33.60 / 61.63 ms | FlashRT 21.07 / 40.70 / 69.24（FP16）ms | 2026-11 | [PI0.5：Thor 与 RTX 4090](pi05-performance.md) |
| π0.5 | RTX 4090 | FlashRT v0.2.0 测速脚本 · FP8 / BF16 | 17.06 / 30.08 ms | FlashRT 22.06 / 31.93（FP16）ms | 2026-11 | [PI0.5：Thor 与 RTX 4090](pi05-performance.md) |
| π0.5 | AGX Orin 32GB | 200 帧 LIBERO-10 · BF16 | 302.0 ms | 现有 Triton kernel 396.6 ms | 2026-11-02 | [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| π0.5 | Orin NX 16GB | `RLinf-Pi05-LIBERO-SFT` · W8A8 prefix + W4A16 action | 321.9–338.3 ms | — | 2026-10-12 | [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| π0.5 | DGX Spark | `RLinf-Pi05-LIBERO-SFT` · NVFP4 prefix + FP8 action | 65.76 ms | 原 EmbodiInfer BF16 109.33 ms | 2026-10-10 | [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| GR00T N1.7 | RTX 4090 | LIBERO-10，1,600 帧 · BF16 | 31.30 ms | PhyAI 33.97 ms | 2026-09-09 | [gr00t-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| GR00T N1.7 | Jetson AGX Thor | LIBERO-10，1,600 帧 · BF16，原生优化前 | 92.47 ms | — | 2026-09-08 | [gr00t-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| OpenVLA-OFT | Thor / RTX 4090 | LIBERO-10，1,600 帧 · BF16 | 149.36 / 63.90 ms | — | 2026-09-08 | [openvla-oft-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/openvla-oft-benchmark/README.md) |
| LingBot-VLA | Thor / RTX 4090 | RoboTwin2.0，1,600 帧 · BF16 | 212.17 / 88.36 ms | — | 2026-09-08 | [lingbot-vla-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/lingbot-vla-benchmark/README.md) |
| DM0.5 | Thor / RTX 4090 | LIBERO-10，1,600 帧 · BF16 | 260.93 / 140.37 ms | — | 2026-09-08 | [dm05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/dm05-benchmark/README.md) |
| Cosmos Policy | Thor / RTX 4090 | LIBERO-10，1,600 帧 · BF16，N=1 | 1,323.97 / 444.62 ms | — | 2026-09-08 | [cosmos-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/cosmos-benchmark/README.md) |
| ActiveVLN | RTX 4090 | 128 个 R2R 首帧 · BF16 | 55.39 ms | SGLang 177.22 ms，vLLM 117.26 ms | 2026-11-02 | [RTX 4090](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/rtx-4090/README.md) |
| ActiveVLN | Jetson AGX Thor | 128 个 R2R 首帧 · BF16 / INT8 | 116.57 / 114.74 ms | — | 2026-10 | [Thor](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/thor/README.md) |
| ActiveVLN | Orin NX 16GB | 128 集 R2R 闭环 · BF16 / INT8 | SR 43.75% / 49.22%；模型 691.34 / 537.55 ms | — | 2026-10 | [Orin NX](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/orin-nx/README.md) |
| StreamVLN | RTX 4090 | 128 个 R2R 首帧 · BF16 | 91.25 ms | 官方实现 276.84 ms | 2026-11-02 | [RTX 4090](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/rtx-4090/README.md) |
| StreamVLN | Jetson AGX Thor | 128 个 R2R 首帧 · BF16 / INT8 | 173.19 / 180.10 ms | — | 2026-10 | [Thor](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/thor/README.md) |
| StreamVLN | Orin NX 16GB | 128 集 R2R 闭环 · INT8 | SR 54.69%；模型 1,359.43 ms | — | 2026-10 | [Orin NX](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/orin-nx/README.md) |
| StreamVLN | AGX Orin 32GB | 48 集 R2R 回放 · BF16 / INT8 | 970.95 / 2,970.90 ms | — | 2026-09 | [streamvln-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-quant-benchmark/README.md) |
| Qwen R2R low | Thor / RTX 4090 | 192 次 R2R 调用 · BF16 | 212.51 / 118.75 ms | — | 2026-09-08 | [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) |
| Qwen R2R panoramic | Thor / RTX 4090 | 192 次 R2R 调用 · BF16 | 773.61 / 398.82 ms | — | 2026-09-08 | [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) |
| NaViDA | Thor / RTX 4090 | 192 次 R2R 调用 · BF16 | 1,145.61 / 377.54 ms | — | 2026-09-08 | [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) |

双卡数据并行与张量并行见[并行](parallelism.md)页面，ActiveVLN 在 RLinf 中的 rollout 见[下文](#activevln-rlinf-integration)。

## 选择基准测试

| 目录 / 工作流 | 模型 | 工作负载与报告 |
|---|---|---|
| [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | π0.5 | FlashRT 测速、Thor / 4090 预设、LIBERO 闭环与其他平台结果 |
| [gr00t-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | GR00T N1.7 | LIBERO-10，1,600 帧；4090 原生优化、竞品对比与 batch，Thor 基线 |
| [openvla-oft-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/openvla-oft-benchmark/README.md) | OpenVLA-OFT | LIBERO-10，1,600 帧；categorical action-token 检查点 |
| [lingbot-vla-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/lingbot-vla-benchmark/README.md) | LingBot-VLA | RoboTwin2.0，1,600 帧；三相机与双臂关节 |
| [dm05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/dm05-benchmark/README.md) | DM0.5 | LIBERO-10，1,600 帧；独立 OpenDM 环境 |
| [cosmos-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/cosmos-benchmark/README.md) | Cosmos Policy | LIBERO-10，1,600 帧；动作生成 |
| [streamvln-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/README.md) | StreamVLN | R2R / RxR 回放、128 首帧、官方实现对比与闭环导航 |
| [activevln-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/README.md) | ActiveVLN | R2R / RxR 回放、SGLang / vLLM 对比、首帧与闭环导航 |
| [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) | Qwen R2R low / panoramic · NaViDA | R2R / RxR 前 48 集，每集 4 条录制帧 |
| [3B-navigation](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/3B-navigation/README.md) | Qwen R2R low / panoramic · NaViDA | 真实图像的 eager / CUDA Graph 一致性、算子与编译缓存检查工具 |
| [pi05-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) | π0.5 | AGX Orin BF16 / INT8 的 B=1 LIBERO 回放 |
| [streamvln-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-quant-benchmark/README.md) | StreamVLN | 按设备 / 精度记录的 B=1 R2R / RxR 全帧回放，早于 10 月的推理优化 |
| [kernel tuning](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/proposals/0008-kernel-tuning.md) | 注册算子 / π0.5 | 捕获算子、数值校准、调优与 profiling；开发者工作流 |

PI0.5 在 Jetson AGX Thor 与 RTX 4090 上与 FlashRT、APXinf、Embodied.cpp、OpenPI 的对比，以及 LIBERO
闭环视频，见 [PI0.5：Thor 与 RTX 4090 性能对比](pi05-performance.md)。

## 测试数据与计时范围

采样数量可配置。除报告另有说明外，计时从已解码的 CPU 观测开始，到 CPU 动作输出结束，包括预处理、推理和结果输出，
不包括模型加载、磁盘数据解码和预热。多数结果为离线回放录制观测；闭环导航结果由 EmbodiRun 的 Habitat runner 运行，
另行报告 SR/SPL。Panoramic 的输入由 RGB 数据构造，形状与模型要求匹配；Cosmos 测量动作生成耗时。

原有 benchmark 目录提供 `setup_env.py` 创建隔离环境；
ActiveVLN 使用已有 `activevln` 依赖组。各目录的
`benchmark.py` 负责数据采样、计时和推理。Qwen Low/Panoramic 使用 SDPA + Inductor
+ CUDA Graph。每个 README 都给出确切的运行命令。

<a id="coverage-gaps"></a>

## 测试覆盖缺口

下表是现有报告尚未覆盖或尚不完整的实验。

| 范围 | 已有证据 | 待补 |
|---|---|---|
| GR00T / Thor | [4090 原生优化与 Thor 基线](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | 原生优化版本的 Thor 复测与设备专用精度实验。 |
| OpenVLA-OFT / LingBot-VLA / Cosmos / DM0.5 | 上方各模型的 Thor / 4090 BF16 基线 | 模型专用优化、适用量化档及同条件竞品对照。 |
| 除 π0.5 外的 VLA | 仅有 Thor / 4090 | AGX Orin、Orin NX、DGX Spark 的模型级测量；多卡执行。 |
| StreamVLN | [Thor](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/thor/README.md) 与 [4090](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/rtx-4090/README.md) 首帧；Thor、4090 与 [NX](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/orin-nx/README.md) 闭环；10 月优化前测得的量化回放 | Thor RxR 与 BF16 B=8 闭环；NX 首帧；AGX Orin 及 FP8 / NVFP4 在当前代码上的测量；DGX Spark。 |
| ActiveVLN | [基准索引](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/README.md)与下方 RLinf 实验记录 | NX 首帧 / RxR 闭环，Thor RxR / 多 batch 闭环；RxR B=4 的显存限制；AGX Orin / DGX Spark；完整训练、收敛及待通过的 logprob 一致性验证。 |
| Qwen R2R low / panoramic · NaViDA | Thor / 4090 离线回放及 A800 DP/TP | Orin 平台、量化与更多 batch 测量。 |

新模型和部署工作见[模型路线图](models.md#roadmap)及
[EmbodiRun 覆盖与规划](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/zh/support-matrix.md)。
Orin FP8 / FP4、RTX 4090 FP4 不计为当前硬件配置的待补档位。

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

Qwen2.5-VL-3B 和 NaViDA 的逐样本 eager / CUDA Graph 一致性检查见
[`benchmarks/3B-navigation/`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/3B-navigation/README.md)。

双 GPU 数据并行和张量并行的对比见
[并行](parallelism.md) 页面。

<a id="activevln-rlinf-integration"></a>

## ActiveVLN / RLinf 集成

**每训练步的 8 次批量推理调用合计（秒）**

| 硬件 | Batch | SGLang（s） | EmbodiInfer（s） | 加速比 |
|---|---:|---:|---:|---:|
| 2× RTX 4090 | 2 | 2.255 | 0.872 | 2.59× |
| 2× RTX 4090 | 4 | 3.148 | 1.259 | 2.50× |
| 2× RTX 4090 | 8 | 4.789 | 2.057 | 2.33× |
| 2× H20 | 2 | 1.654 | 0.639 | 2.59× |
| 2× H20 | 4 | 2.358 | 1.043 | 2.26× |
| 2× H20 | 8 | 4.125 | 1.916 | 2.15× |

**完整 rollout（秒 / 训练步）**

| 硬件 | Batch | SGLang（s） | EmbodiInfer（s） | 加速比 |
|---|---:|---:|---:|---:|
| 2× RTX 4090 | 2 | 5.001 | 3.484 | 1.44× |
| 2× RTX 4090 | 4 | 6.690 | 4.944 | 1.35× |
| 2× RTX 4090 | 8 | 10.878 | 7.987 | 1.36× |
| 2× H20 | 2 | 3.392 | 2.322 | 1.46× |
| 2× H20 | 4 | 5.292 | 3.707 | 1.43× |
| 2× H20 | 8 | 9.404 | 7.576 | 1.24× |

来源：[RLinf #1789](https://github.com/RLinf/RLinf/pull/1789)，下游提交
[`d909549`](https://github.com/RLinf/RLinf/commit/d9095497b4f1eb2b5419a9ee63e725ca03df53f2)。
`Arvil/Qwen2.5-VL-3B_rl_r2r_4000`、BF16，重复运行 R2R episode 1；
一张 GPU 承担 actor / rollout，一张 GPU 负责 Habitat-Sim 渲染。
预热后测量 5 个训练步并取均值；第一张表是每步 8 次调用合计，**不是单次请求延迟**。
推理包括预处理、生成、输出打包和待应用权重；完整 rollout 另含环境交互与轨迹传输。
收益包含运行时和下游集成的共同优化。

该下游 PR 尚未合入；完整 HabitatLab 训练与收敛未验证，BF16 增量 / 全历史 logprob
最大差 0.119415，超过 0.02 的验证门槛。

### A800 · 独立的 GRPO 集成验证

[EmbodiInfer #16](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/16) 记录了另一套工作负载：
2×A800-SXM4-80GB，一卡 actor / rollout、另一卡 HabitatSim；BF16、同一 R2R 检查点，
episode 1、2 环境、group size 2、8 turns、global batch 16、microbatch 1。
每个后端 6 个 iteration，丢弃首个，统计后 5 个。

| HF actor / rollout | 训练 iteration 均值 ± 样本标准差 | Rollout 阶段 | Actor 阶段 |
|---|---:|---:|---:|
| HF / SGLang | 20.64 ± 0.15 s | 15.27 s | 4.63 s |
| HF / EmbodiInfer | 19.73 ± 0.41 s | 14.77 s | 4.43 s |

使用外部 HabitatSim HTTP 验证桥，并非完整官方 HabitatLab/Ray 服务；轨迹不同，
这些阶段耗时不证明 SR/SPL 或训练收敛等价。运行时版本 `f20e92e`。
它与上面的 4090 / H20 表是独立实验。[完整配置与验证范围](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/16)。
