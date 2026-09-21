<div class="hero" markdown>

<h1 class="hero-title">
  <img src="assets/logo.svg" alt="EmbodiInfer" class="hero-logo">
</h1>

**面向具身模型的推理与 RL rollout 引擎。**

在本地运行策略、对外提供推理服务，或接入 RL 训练器。

[安装](installation.md){ .md-button .md-button--primary }
[快速开始](quickstart.md){ .md-button }
[GitHub](https://github.com/BUAA-CI-LAB/EmbodiInfer){ .md-button }

</div>

EmbodiInfer 可以运行视觉—语言—动作策略、世界动作模型和有状态导航策略，
通过 Python 与网络接口提供模型加载、批处理执行、CUDA graph 加速和有状态会话管理。
可用的优化和 rollout 能力取决于所选策略。

它可独立使用，也可配合 [EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun) 部署到机器人或仿真器。

## 从这里开始

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __安装并运行模型__

    ---

    用 uv 安装，试用 Python API，加载第一个检查点。

    [:octicons-arrow-right-24: 安装](installation.md)

    [:octicons-arrow-right-24: 快速开始](quickstart.md)

-   :material-server-network:{ .lg .middle } __部署服务与接入训练__

    ---

    启动推理服务、把请求分发到多张 GPU，或接入 RL 训练器。

    [:octicons-arrow-right-24: 服务](serving.md)

    [:octicons-arrow-right-24: 并行](parallelism.md)

-   :material-chart-line:{ .lg .middle } __查看性能测试__

    ---

    对比模型延迟、引擎优化和 RL rollout 耗时。

    [:octicons-arrow-right-24: 性能](benchmark.md)

    [:octicons-arrow-right-24: 支持的模型](models.md)

-   :material-sitemap:{ .lg .middle } __了解引擎设计__

    ---

    跟踪一个请求经过预处理、模型执行与解码的完整路径。

    [:octicons-arrow-right-24: 架构](architecture.md)

    [:octicons-arrow-right-24: Python API](api.md)

</div>

## 选择模型与运行环境

[模型列表](models.md)列出各模型的检查点、依赖组和优化选项。
π0.5 的 HTTP 和 WirelessComm 服务支持[多客户端批处理](serving.md#share-one-service-across-clients)。
有状态导航策略会在同一轮任务内保留历史状态，供后续推理使用。

机器人或仿真器部署请搭配 [EmbodiRun](https://embodirun.readthedocs.io/)。

## 社区

- [参与贡献](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/CONTRIBUTING.md) —— 开发环境、
  测试与 Pull Request 流程。
- [行为准则](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/CODE_OF_CONDUCT.md) ——
  本项目采用的 Contributor Covenant 2.1。
- [安全政策](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/SECURITY.md) ——
  了解如何私下报告漏洞。
- [许可证（英文）](https://embodiinfer.readthedocs.io/en/latest/license/) ——
  Apache-2.0 与第三方组件声明。

仓库 README 提供[英文版](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/README.md)和
[简体中文版](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/README.zh-CN.md)。
