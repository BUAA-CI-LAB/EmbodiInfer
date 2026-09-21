<p align="center">
  <img src="assets/embodiinfer-logo.svg" alt="EmbodiInfer" width="440">
</p>

<h3 align="center">一个引擎，统一具身推理与 RL rollout。</h3>
<p align="center">
  <a href="https://embodiinfer.readthedocs.io/zh-cn/latest/">文档</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#支持的模型">模型</a> ·
  <a href="#性能">性能</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://embodiinfer.readthedocs.io/"><img src="https://readthedocs.org/projects/embodiinfer/badge/?version=latest" alt="Documentation"></a>
</p>

**EmbodiInfer 是面向具身模型的推理与 RL rollout 引擎。** 通过统一 Python 接口运行操作策略、世界动作模型和有状态导航策略，
结合模型专用优化与共享的批处理、会话和多 GPU 执行能力。

它可以嵌入应用或训练器，也可以向 [EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun) 提供预测，由后者负责机器人部署与执行。

![应用、RL 训练器和 EmbodiRun 通过共享引擎调用各模型的策略适配器](docs/assets/engine-overview.svg)

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

## 性能

### 单张 RTX 4090 上的延迟优化

与原有 EmbodiInfer 路径相比，原生推理优化将 **π0.5 的平均端到端延迟降低 47.7%**，
将 **GR00T N1.7 的平均端到端延迟降低 31.2%**：

| 策略 | 优化前 | 优化后 | 优化后吞吐 |
|---|---:|---:|---:|
| [π0.5](benchmarks/pi05-benchmark/README.md#pi05-原生优化2026-09-09) | 74.33 ms | **38.89 ms** | **25.71 observations/s** |
| [GR00T N1.7](benchmarks/gr00t-benchmark/README.md#4090-原生优化2026-09-09) | 45.51 ms | **31.30 ms** | **31.95 observations/s** |

测量日期为 2026 年 9 月 9 日，使用 RTX 4090、BF16、batch size 1，每个模型回放 1,600 条 LIBERO-10 观测。
π0.5 使用 `pi05_libero_finetuned_v044_gitcode` 和 10 步去噪；GR00T 使用 `nvidia/GR00T-N1.7-LIBERO` 和 4 步去噪。
优化前后均开启 Inductor 和去噪 CUDA graph。基线使用 SDPA；优化路径增加原生推理与 prefix graph，
π0.5 的 prefix 与去噪 attention 使用 Triton，GR00T 使用 SDPA。

使用 10 条跨任务观测预热后，计时从已解码 CPU 观测到 CPU 动作输出。上面的报告包含权重详情、环境、命令、数值检查及 batch 扫描。
在同一 GPU 上，优化后的 GR00T 在 **batch size 32** 时达到 **59.33 observations/s**。

更多结果：[导航与世界动作模型](docs/zh/benchmark.md)、[π0.5 量化](benchmarks/pi05-quant-benchmark/README.md)、
[StreamVLN 量化](benchmarks/streamvln-quant-benchmark/README.md)、[多 GPU](docs/zh/parallelism.md)。

## 支持的模型

✓ **已实现** · ◐ **实验性** · ○ **计划支持**

<table>
<tr>
<th align="left">🦾 操作策略</th>
<th align="left">🧭 导航策略</th>
<th align="left">🌐 世界动作模型</th>
</tr>
<tr>
<td valign="top">
<p>✓ <b>π0.5</b><br>✓ <b>GR00T N1.7</b><br>✓ <b>OpenVLA-OFT</b><br>✓ <b>LingBot-VLA</b><br>✓ <b>DM0.5</b></p>
<p>操作任务的动作预测<br>与 RL decoder 接口。</p>
</td>
<td valign="top">
<p>✓ <b>StreamVLN</b><br>✓ <b>Qwen R2R</b> · low / panoramic<br>✓ <b>NaViDA</b><br>◐ ActiveVLN</p>
<p>Episode 历史管理<br>与有状态推理。</p>
</td>
<td valign="top">
<p>✓ <b>Cosmos Policy</b></p>
<p>基于扩散的动作生成<br>与候选规划。</p>
</td>
</tr>
</table>

**网络服务：** π0.5、DM0.5、StreamVLN 支持 HTTP / WirelessComm，batch size 为 1。
各类模型均提供 Python 入口。[能力参考](docs/zh/models.md#capabilities-and-installation)列出批处理、CUDA graph、
RL 接口和环境要求。ActiveVLN 真实检查点的 GPU parity 仍待验证。

### 计划支持的模型

- [ ] **SmolVLA** — 模型适配、输入处理与网络服务接入。
- [ ] **OpenVLA** — 原始模型推理，与 OpenVLA-OFT 分开实现。

实现步骤见[模型路线图](docs/zh/models.md#roadmap)，机器人和仿真器接入见
[EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun#support-at-a-glance)。

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
| 在应用内推理 | `Vvla.act(observation)` 或 `Vvla.act(observations)` | [Python API](docs/zh/api.md) |
| 为机器人运行时提供网络推理 | HTTP / WirelessComm 会话 | [服务与首次观测请求](docs/zh/serving.md) |
| RL rollout 与权重更新 | Rollout / refit 接口 | [RL 集成](docs/zh/api.md#generating-rl-rollouts) |
| 多 GPU 执行 | 数据并行 / 张量并行 | [并行指南](docs/zh/parallelism.md) |

网络启动器支持 π0.5、DM0.5 和 StreamVLN，batch size 为 1；其他策略通过 Python API 调用。
各模型支持情况见[能力表](docs/zh/models.md#capabilities-and-installation)。

发行包名为 `embodiinfer`，Python import 仍为 `embodiinfer`。
旧 `vvla-*` 命令别名与 `vvla.policy.*` 通信 schema 保持兼容。

## 文档

[安装](https://embodiinfer.readthedocs.io/zh-cn/latest/installation/) ·
[快速开始](https://embodiinfer.readthedocs.io/zh-cn/latest/quickstart/) ·
[服务](https://embodiinfer.readthedocs.io/zh-cn/latest/serving/) ·
[并行](https://embodiinfer.readthedocs.io/en/latest/parallelism/) ·
[架构](https://embodiinfer.readthedocs.io/en/latest/architecture/) ·
[Python API](https://embodiinfer.readthedocs.io/en/latest/api/)

中文站点覆盖全部正文页面；治理与法律页（贡献指南、行为准则、许可证）保留英文原文。

## 参与贡献

开发流程、模型接入与数值验证要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
问题与建议请提交 [GitHub issue](https://github.com/BUAA-CI-LAB/EmbodiInfer/issues)，
漏洞请按 [SECURITY.md](SECURITY.md) 私下报告。社区遵循[行为准则](CODE_OF_CONDUCT.md)。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。
权重和数据集保留上游许可证。
