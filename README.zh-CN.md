<p align="center">
  <img src="assets/embodiinfer-logo.svg" alt="EmbodiInfer" width="440">
</p>

<h3 align="center">一个引擎，统一具身推理与 RL rollout。</h3>
<p align="center">
  <a href="https://embodiinfer.readthedocs.io/zh-cn/latest/">文档</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#支持的模型">模型</a> ·
  <a href="#vla-与-vln-演示">演示</a> ·
  <a href="#性能">性能</a> ·
  <a href="#部署与-recipe">部署 / Recipe</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://embodiinfer.readthedocs.io/"><img src="https://readthedocs.org/projects/embodiinfer/badge/?version=latest" alt="Documentation"></a>
</p>

**EmbodiInfer 是面向具身模型的推理与 RL rollout 引擎。** 通过统一 Python 接口运行操作策略、世界动作模型和有状态导航策略，
结合模型专用优化与共享的批处理、会话和多 GPU 执行能力。

**从 EmbodiInfer 开始：** 查看模型、性能与 Python / 服务接口；需要连接机器人或仿真器时，
使用 [EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.zh-CN.md) 的一键部署案例、Recipe 与部署框架。

| 入口 | 内容 |
|---|---|
| **EmbodiInfer · 模型推理** | 检查点、策略适配、算子优化、批处理、多 GPU 与 RL rollout。 |
| [**EmbodiRun · 部署与执行**](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.zh-CN.md) | 机器人 / 仿真器接入、服务编排、动作执行、录制与案例 Recipe。 |

![应用、RL 训练器和 EmbodiRun 通过共享引擎调用各模型的策略适配器](docs/assets/engine-overview.svg)

## VLA 与 VLN 演示

### VLA · π0.5 / LIBERO

[![RTX 4090 上 OpenPI、FlashRT 与 EmbodiInfer 的 LIBERO 闭环对比](docs/assets/pi05/libero-4090-comparison.webp)](benchmarks/pi05-benchmark/README.md)

RTX 4090，任务 6 / 初始状态 2，三个引擎各自独占 GPU、均为 10 步去噪。
**1× 原速**；三个引擎均完成该次任务。录像对应单次闭环，性能汇总与成功率见下表及[报告](docs/zh/pi05-performance.md)。

<details>
<summary>Jetson AGX Thor · OpenPI / FlashRT / EmbodiInfer（2× 播放）</summary>

![Thor 上三引擎的 LIBERO 闭环对比](docs/assets/pi05/libero-thor-comparison.webp)

相同任务初始状态，每个控制步重新决策。画面计时为真实时间，三个引擎均完成该次任务。
[测试条件、成功率与原视频](benchmarks/pi05-benchmark/README.md)。

</details>

### VLN · ActiveVLN / Habitat

[![EmbodiRun 中 EmbodiInfer、SGLang 与 vLLM 的 ActiveVLN 原速对比](https://raw.githubusercontent.com/BUAA-CI-LAB/EmbodiRun/aea124bf8f7eda613daa50098181d532fccd1903/docs/assets/demos/activevln-engines.webp)](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/zh/demos/activevln-engines.md)

RTX 4090、BF16、B=1，R2R 同一 episode 的 **1× 原速**录像；
[EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.zh-CN.md) 统一负责观测、HTTP 请求与动作执行。
单集任务耗时为 EmbodiInfer **1.979 s**、SGLang 2.736 s、vLLM 2.682 s，
三者最终距目标均为 0.673 m。[录像条件与接入状态](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/zh/demos/activevln-engines.md)。

## 性能

[模型推理对比](#vla--π05-多引擎对比) · [RL rollout](#rl-rollout--activevln--rlinf) · [模型 × 硬件大表](#模型--硬件测试覆盖)

### VLA · π0.5 多引擎对比

| 设备 | EmbodiInfer 预设 | EmbodiInfer | FlashRT | APXinf | Embodied.cpp | OpenPI |
|---|---|---:|---:|---:|---:|---:|
| Jetson AGX Thor | `nvfp4-nvfp4` | **20.04 ms** | 21.07 ms (FP4) | — | — | — |
| Jetson AGX Thor | `fp8-fp8` | **33.60 ms** | 40.70 ms | 40.88 ms | — | — |
| Jetson AGX Thor | `bf16` | **61.63 ms** | 69.24 ms (FP16) | 67.12 ms | 223.97 ms | 261.18 ms |
| RTX 4090 | `fp8-fp8` | **17.06 ms** | 22.06 ms | 40.62 ms (INT8) | — | — |
| RTX 4090 | `bf16` | **30.08 ms** | 31.93 ms (FP16) | 32.06 ms | 57.48 ms | — |

FlashRT v0.2.0 的 `examples/thor/pi05.py`，仅替换各引擎加载和 `run()`：
相同 `RLinf-Pi05-LIBERO-SFT` 权重，8 条 LIBERO 观测轮流输入，13-token prompt，
B=1、双路 224×224 相机、10 步去噪、动作长度 10。每轮预热 200 次、计时 1,000 次，
取 **3 轮交替运行的 P50 平均值**；Thor FP4 为另一次同设置的交替运行。
Thor 使用 MAXN 并锁定频率，4090 使用默认频率；各引擎的精度差异已标在表内，`—` 表示未报告。
[完整配置、数值核验与复现命令](docs/zh/pi05-performance.md) ·
[基准脚本与原始报告入口](benchmarks/pi05-benchmark/README.md)。

**LIBERO-spatial 500 次评测：** OpenPI 官方权重与官方评测入口下，
EmbodiInfer `fp8-fp8`、2 步去噪 / ρ=3 为 **495/500（99.0%），13.34 ms / 75.0 Hz**；
OpenPI BF16、10 步为 494/500（98.8%）。
这是独立的少步配置，与上表 10 步测速分别记录。
[评测与标定方法](docs/zh/pi05-performance.md)。

### VLN · ActiveVLN 原生引擎对比

| Batch | EmbodiInfer 均值 / P95 | SGLang SRT 均值 / P95 | vLLM 均值 / P95 |
|---:|---:|---:|---:|
| 1 | **55.39 / 57.88 ms** | 177.22 / 218.77 ms | 117.26 / 131.11 ms |
| 2 | **73.60 / 77.25 ms** | 214.48 / 252.51 ms | 152.81 / 163.19 ms |
| 4 | **112.02 / 113.83 ms** | 280.95 / 319.75 ms | 223.69 / 232.43 ms |
| 8 | **220.13 / 290.67 ms** | 364.19 / 407.99 ms | 315.69 / 323.11 ms |

RTX 4090 24 GB、BF16、`Arvil/Qwen2.5-VL-3B_rl_r2r_4000@1609873`，
R2R val-unseen 固定清单前 128 集各取首帧；128 帧预热一遍、正式测量一遍。
计时为 **CPU RGB → 整批完整 CPU actions**，含预处理；不含加载、预热、HTTP 与仿真，
SGLang 含本机 SRT 进程通信。各批清空历史、内容缓存与已学习的投机候选。
各引擎使用独立兼容环境及各自优化，动作输出并非全部相同；本表仅比较性能。
B=8 的首批尖峰保留在统计中。[配置、输出核验、首批与后续批次](benchmarks/activevln-benchmark/rtx-4090/README.md)。

| 更多结果 | 入口 |
|---|---|
| StreamVLN 对比官方实现，B=1/2/4/8 | [RTX 4090 报告](benchmarks/streamvln-benchmark/rtx-4090/README.md) |
| ActiveVLN 首帧、长历史回放与历史闭环 SR/SPL | [基准索引](benchmarks/activevln-benchmark/README.md) · [Thor](benchmarks/activevln-benchmark/thor/README.md) · [Orin NX](benchmarks/activevln-benchmark/orin-nx/README.md) |
| π0.5 / GR00T 的历史 BF16 优化与 batch 扫描 | [π0.5](benchmarks/pi05-benchmark/README.md#其他平台与早期结果) · [GR00T](benchmarks/gr00t-benchmark/README.md) |
| 量化、世界动作模型与多 GPU | [π0.5 量化](benchmarks/pi05-quant-benchmark/README.md) · [StreamVLN 量化](benchmarks/streamvln-quant-benchmark/README.md) · [所有基准](docs/zh/benchmark.md) · [并行](docs/zh/parallelism.md) |


### RL rollout · ActiveVLN / RLinf

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
[完整条件及独立 A800 测量](docs/zh/benchmark.md#activevln-rlinf-integration)。

### 模型 × 硬件测试覆盖

**✓ 已有记录 · ◐ 局部 / 历史或实验性集成 · — 暂无已发布结果**。
这是已记录实验的覆盖盘点，涵盖历史冻结版本，不等同于当前 main 在所有平台复测通过；
每格链接到对应报告，不同模型 / 平台 / 测试口径之间不换算加速比。

| 模型 | Jetson AGX Thor | RTX 4090 | AGX Orin 32GB | Orin NX 16GB | DGX Spark | H200 | A800 ×2 | H20 ×2 |
|---|---|---|---|---|---|---|---|---|
| π0.5 | [✓ FP4 / FP8 / BF16 + LIBERO](benchmarks/pi05-benchmark/README.md) | [✓ BF16 / FP8 + LIBERO](benchmarks/pi05-benchmark/README.md) | [✓ H10 / H50](benchmarks/pi05-benchmark/README.md) | [✓ W8A8 / W4A16](benchmarks/pi05-benchmark/README.md) | [✓ 预设 1.66×](benchmarks/pi05-benchmark/README.md) | [✓ 机制 / RLinf](docs/zh/benchmark.md) | [✓ DP / TP](docs/zh/parallelism.md) | — |
| GR00T N1.7 | [◐ BF16 旧基线](benchmarks/gr00t-benchmark/README.md) | [✓ 原生优化 / batch](benchmarks/gr00t-benchmark/README.md) | — | — | — | [✓ 机制 / RLinf](docs/zh/benchmark.md) | — | — |
| OpenVLA-OFT | [◐ BF16 旧基线](benchmarks/openvla-oft-benchmark/README.md) | [◐ BF16 旧基线](benchmarks/openvla-oft-benchmark/README.md) | — | — | — | [✓ 机制 / GRPO](docs/zh/benchmark.md) | — | — |
| LingBot-VLA | [◐ BF16 eager](benchmarks/lingbot-vla-benchmark/README.md) | [◐ BF16 eager](benchmarks/lingbot-vla-benchmark/README.md) | — | — | — | [✓ 机制 / GRPO](docs/zh/benchmark.md) | — | — |
| Cosmos Policy | [◐ BF16 旧基线](benchmarks/cosmos-benchmark/README.md) | [◐ BF16 旧基线](benchmarks/cosmos-benchmark/README.md) | — | — | — | [✓ best-of-N / 图](docs/zh/benchmark.md) | — | — |
| DM0.5 | [◐ BF16 旧基线](benchmarks/dm05-benchmark/README.md) | [◐ BF16 旧基线](benchmarks/dm05-benchmark/README.md) | — | — | — | — | — | — |
| StreamVLN | [✓ 首帧 / 闭环 / 量化](benchmarks/streamvln-benchmark/thor/README.md) | [✓ 原生 / 官方 / 闭环](benchmarks/streamvln-benchmark/rtx-4090/README.md) | [◐ BF16 / INT8 离线](benchmarks/streamvln-quant-benchmark/README.md) | [◐ R2R INT8 闭环](benchmarks/streamvln-benchmark/orin-nx/README.md) | — | — | — | — |
| Qwen R2R low | [◐ BF16 离线](benchmarks/qwenvl-benchmark/README.md) | [✓ 离线 / 图](benchmarks/3B-navigation/R2R_VLNCE_THROUGHPUT_REPORT.md) | — | — | — | — | [✓ DP / TP](docs/zh/parallelism.md) | — |
| Qwen R2R panoramic | [◐ BF16 离线](benchmarks/qwenvl-benchmark/README.md) | [✓ 离线 / 图](benchmarks/3B-navigation/R2R_VLNCE_THROUGHPUT_REPORT.md) | — | — | — | — | [✓ DP / TP](docs/zh/parallelism.md) | — |
| NaViDA | [◐ BF16 离线](benchmarks/qwenvl-benchmark/README.md) | [✓ 离线；图未过精度](benchmarks/3B-navigation/R2R_VLNCE_THROUGHPUT_REPORT.md) | — | — | — | — | [✓ DP / TP](docs/zh/parallelism.md) | — |
| ActiveVLN | [✓ 首帧 / R2R 闭环](benchmarks/activevln-benchmark/thor/README.md) | [✓ 首帧 / 引擎 / 闭环](benchmarks/activevln-benchmark/rtx-4090/README.md) | — | [◐ R2R BF16 / INT8 闭环](benchmarks/activevln-benchmark/orin-nx/README.md) | — | — | [◐ RLinf 验证](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/16) | [◐ RLinf rollout](https://github.com/RLinf/RLinf/pull/1789) |

从 2026-11-02 的 11×7 盘点更新：纳入 [#22](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/22)
的最新 π0.5 数据、[#18](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/18) 的 VLN 引擎对比，
以及 A800 / H20 的实验性 RLinf 集成。H200 列主要是引擎机制与 RL；
A800 的 DP/TP 与 RLinf 验证为不同工作负载。H20 结果来自仍待合入的下游 PR。
Orin 的 FP8 / FP4、4090 的 FP4 不计为本表缺口。
[完整口径与基准索引](docs/zh/benchmark.md)。

## 支持的模型

✓ 已实现 · ◐ 实验性 · ○ 计划支持；具体检查点与验证范围见[模型参考](docs/zh/models.md)。

| 类别 | 模型 | Python 推理 | HTTP / WirelessComm | 能力与限制 |
|---|---|---|---|---|
| VLA | π0.5 | ✓ | ✓ | 跨会话批处理；BF16 / 设备专用量化预设；RL decoder。 |
| VLA | GR00T N1.7 | ✓ | — | 批处理与 RL decoder；使用基准专用环境。 |
| VLA | OpenVLA-OFT | ✓ | — | 单次 forward 动作解码与 RL decoder。 |
| VLA | LingBot-VLA | ✓ | — | Flow matching 与 RL decoder。 |
| VLA | DM0.5 | ✓ | ✓ | 需独立 OpenDM 环境；网络服务保持 B=1。 |
| VLN | StreamVLN | ✓ | ✓ | 循环历史；专用张量批处理；通用网络服务保持 B=1。 |
| VLN | Qwen R2R low / panoramic · NaViDA | ✓ | — | 显式循环会话与策略专用解码。 |
| VLN | ActiveVLN | ◐ | — | 专用循环 / 张量批处理 API；通用服务适配仍待合入，真实权重 GPU parity 待完成。 |
| 世界动作 | Cosmos Policy | ✓ | — | 扩散动作生成与候选规划；不提供通用 RL decoder。 |
| 计划 | SmolVLA · OpenVLA 基础模型 | ○ | ○ | 模型适配、输入处理与服务接入待实现。 |

`—` 表示无通用网络适配器；部署案例的专用服务不等同于通用服务支持。
[完整能力、依赖与安装矩阵](docs/zh/models.md#capabilities-and-installation) ·
[模型路线图](docs/zh/models.md#roadmap)。

## 快速开始

### 无需权重的首次运行

使用 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/) 0.12.x：

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiInfer.git
cd EmbodiInfer
uv sync --frozen
uv run python examples/quickstart.py
```

示例可在 CPU 上运行，使用合成策略演示单条和批量请求，并打印动作形状，无需下载模型权重。
Linux lock 包含 CUDA 版 Torch，CPU 主机的安装下载量也可能较大。

### 加载 π0.5

使用 CUDA GPU 和 π0.5 专用 profile：

```bash
uv sync --python 3.12 --frozen --no-dev --group pi05
uv run --no-sync python examples/pi05_inference.py \
  --ckpt lerobot/pi05_base --envs 1
```

该示例加载真实权重，但输入仍为合成观测，用来演示引擎 API。
真实图像与状态接入见[服务教程](docs/zh/serving.md#send-your-first-observation)。

### 选择接入方式

| 用途 | 接口 | 入口 |
|---|---|---|
| 在应用内推理 | `EmbodiInfer.act(observation)` 或 `EmbodiInfer.act(observations)` | [Python API](docs/zh/api.md) |
| 为机器人运行时提供网络推理 | HTTP / WirelessComm 会话 | [服务与首次观测请求](docs/zh/serving.md) |
| RL rollout 与权重更新 | Rollout / refit 接口 | [RL 集成](docs/zh/api.md#generating-rl-rollouts) |
| 多 GPU 执行 | 数据并行 / 张量并行 | [并行指南](docs/zh/parallelism.md) |

多个客户端共享 π0.5 服务时，为 HTTP 或 WirelessComm 启动器添加
`--max-batch 3 --max-wait-ms 5`。独立会话的请求可组成最多三个请求的批次；
5 ms 收集窗口结束时，未满批次也会执行。该窗口不包括排队和推理时间。
配置步骤见[服务指南](docs/zh/serving.md#多客户端批处理-share-one-service-across-clients)。

发行包名为 `embodiinfer`；Python import、`embodiinfer-*` 命令行入口与 `embodiinfer.policy.*` 通信 schema 统一使用同一命名。

## 部署与 Recipe

[EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/README.zh-CN.md) 提供一键案例入口与部署框架。
按 Recipe 准备检查点、场景或硬件后，使用统一的 `embodirun example` 命令完成配置、检查、启动与停止。

| 场景 | 部署入口 | 当前状态 |
|---|---|---|
| VLA · SO-101 / π0.5 | [抓取 Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/so101_grasping.md) · [多后端对比](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/engine_comparison/README.md) | 已有真机录像；按检查点与硬件配置。 |
| VLN · MicroDuck / MuJoCo | [导航 Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md) | 专用仿真 Recipe，需准备场景、模型与 GPU。 |
| VLN · Habitat / 三引擎 | [录像与配置](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/zh/demos/activevln-engines.md) | 实验性；可复用 vLLM 封装入口的 GPU 验证待完成。 |
| VLN · Go2 / ActiveVLN | [真机演示](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/zh/demos/go2-activevln.md) | 已有录像；适配实现仍在 [Run #13](https://github.com/BUAA-CI-LAB/EmbodiRun/pull/13)。 |
| Agent · XLeRobot | [移动抓取 Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/README.md) | 软件演练与监督下硬件流程。 |

[完整覆盖矩阵与路线图](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/zh/support-matrix.md) ·
[全部 Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md)。

## 为什么选择 EmbodiInfer？

优化模型执行路径，让机器人推理与 RL rollout 共用一套引擎。

<table>
<tr>
<td width="50%" valign="top">
<h3>⚡ 捕获、编译、回放</h3>
<p><b>CUDA Graphs · torch.compile / Inductor</b></p>
<p>回放静态 prefix 和解码路径，编译模型计算。针对不同策略配置执行路径，优化去噪与 token 生成中的重复计算。</p>
<a href="docs/zh/architecture.md#cuda-graph-capture">执行路径 →</a>
</td>
<td width="50%" valign="top">
<h3>🎚️ 为 GPU 选择合适精度</h3>
<p><b>BF16 · FP8 · INT8 · NVFP4</b></p>
<p>π0.5 与 StreamVLN 提供硬件专用配置：AGX Orin 使用 INT8，RTX 4090 与 Thor 使用 FP8，Thor 还提供 NVFP4。</p>
<a href="docs/zh/models.md#optimization-profiles">精度与后端配置 →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🔥 深入算子的原生优化</h3>
<p><b>Triton attention · 模型算子融合</b></p>
<p>专用 attention kernel，以及融合的归一化、旋转位置编码和门控激活，为受支持的原生模型路径加速。</p>
<a href="docs/zh/models.md#optimization-profiles">原生优化 →</a>
</td>
<td valign="top">
<h3>🧠 复用上下文，保留历史</h3>
<p><b>Prefix KV 复用 · 有状态会话</b></p>
<p>多步去噪复用固定观测上下文；导航策略保留 episode 历史，通过事务更新状态，并支持显式重置和取消。</p>
<a href="docs/zh/architecture.md#sessionstore">上下文与会话管理 →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🚀 从单次请求到多环境并行</h3>
<p><b>请求批处理 · 多 GPU 执行</b></p>
<p>批量提交观测、聚合异步请求，或将任务分发到多个模型副本。受支持的策略还提供张量并行。</p>
<a href="docs/zh/parallelism.md">并行执行 →</a>
</td>
<td valign="top">
<h3>🔄 融入 RL 训练循环</h3>
<p><b>动作采样 · log probability · 权重更新</b></p>
<p>通过支持 RL 的 decoder 收集 rollout、更新策略权重。保留自己的训练器与学习目标，复用模型执行能力。</p>
<a href="docs/zh/api.md#generating-rl-rollouts">RL 集成 →</a>
</td>
</tr>
</table>

根据策略与硬件在[能力参考](docs/zh/models.md#capabilities-and-installation)中选择配置。
适配器负责模型专用优化，共享调度器保持模型无关。

**Agent 调优 kernel：** 由 coding agent 针对特定 GPU 和软件栈调优算子 kernel。在 Jetson AGX Orin 上，其他条件相同时，
`orin_pi05_r2` profile 将 π0.5 的延迟从手写 kernel 的 396.6 ms 降到 **302.0 ms**（BF16，batch size 1；
[测量条件](docs/proposals/0008-kernel-tuning.md#93-orin_pi05_r2-in-uniform-bf16)、[示例](examples/pi05_inference.py)）。
[调优工作流](docs/proposals/0008-kernel-tuning.md)目前是面向开发者的工具，已接入 π0.5，后续将支持用户针对自己的硬件和环境调优。其设计借鉴了
[NVIDIA Kernel Design Agents (KDA)](https://github.com/NVlabs/kda)（[致谢说明](THIRD_PARTY_NOTICES.md#kernel-tuning-workflow-references)）。

## 文档

| 内容 | 指南 |
|---|---|
| 安装模型环境 | [安装](docs/zh/installation.md) |
| 运行推理 | [快速开始](docs/zh/quickstart.md) · [示例](examples/) |
| 部署服务 | [服务](docs/zh/serving.md) |
| 使用多张 GPU | [并行](docs/zh/parallelism.md) |
| 集成或扩展引擎 | [架构](docs/zh/architecture.md) · [Python API](docs/zh/api.md) |
| 了解模型能力 | [模型](docs/zh/models.md) |
| 评估性能 | [基准](docs/zh/benchmark.md) |

### 路线图

| 方向 | 状态与后续工作 |
|---|---|
| 原生推理与设备预设 | π0.5 Thor / 4090 与 LIBERO 评测已合入；其他模型与平台按基准逐项扩展。 |
| ActiveVLN 通用服务 | [#19](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/19) / [#20](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/20) 仍为 Draft，通用服务待合入；专用 API 与录像可查阅。 |
| 新模型 | SmolVLA、OpenVLA 基础模型的适配与数值验证。 |
| Agent 算子调优 | 已有 π0.5 开发者工作流；面向用户的自动硬件 / 环境调优待实现。 |
| 部署案例 | 在 EmbodiRun 完善 Go2 等 Recipe、独立部署验证与扩展性测量。 |

## 参与贡献

开发流程、模型接入与数值验证要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
问题与建议请提交 [GitHub issue](https://github.com/BUAA-CI-LAB/EmbodiInfer/issues)，
漏洞请按 [SECURITY.md](SECURITY.md) 私下报告。社区遵循[行为准则](CODE_OF_CONDUCT.md)。

## 致谢与引用

- **量化方法：** 感谢 Guangxuan Xiao 等人的
  [SmoothQuant（ICML 2023）](https://proceedings.mlr.press/v202/xiao23c.html) 与
  [开源实现](https://github.com/mit-han-lab/smoothquant)。Thor NVFP4 校准借鉴其通道缩放方法，
  并结合本项目的 NVFP4 打包与 BF16 离群通道旁路；[实现与测量](docs/zh/pi05-performance.md)分别说明。
- **基准与参考引擎：** 感谢 [FlashRT](https://github.com/flashrt-project/FlashRT)
  提供开源 VLA 推理与 PI0.5 测速流程；本项目多引擎表使用其 v0.2.0 harness。
  感谢 [APXinf](https://github.com/infinigence/ApxInf)、
  [Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp)、
  [OpenPI](https://github.com/Physical-Intelligence/openpi)、
  [LeRobot](https://github.com/huggingface/lerobot)、
  [SGLang](https://github.com/sgl-project/sglang) 与 [vLLM](https://github.com/vllm-project/vllm)
  提供模型参考与对照实现。
- **模型、数据与算子生态：** 感谢上游模型作者、
  [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、
  [Habitat](https://github.com/facebookresearch/habitat-lab)、
  [RLinf](https://github.com/RLinf/RLinf)、
  [PyTorch](https://github.com/pytorch/pytorch)、[Triton](https://github.com/triton-lang/triton)、
  [CUTLASS](https://github.com/NVIDIA/cutlass)、
  [FlashAttention](https://github.com/Dao-AILab/flash-attention)、
  [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 与
  [NVIDIA KDA](https://github.com/NVlabs/kda)。

各项目具体用途、派生代码与许可证归属见[第三方声明](THIRD_PARTY_NOTICES.md)；
性能比较对应表中固定版本与配置。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。
权重和数据集保留上游许可证。
