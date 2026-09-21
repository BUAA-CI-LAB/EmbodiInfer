# 快速开始

先用无需检查点的测试策略运行引擎 API，再加载真实模型。
平台要求和独立环境配置见[安装指南](installation.md)。

## 安装并验证引擎

克隆仓库并运行示例：

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiInfer.git
cd EmbodiInfer
uv sync --frozen
uv run python examples/quickstart.py
```

示例使用合成测试策略 `mock_flow_vla`，无需下载检查点。
它先打印单次推理返回的动作块（action chunk）形状，再打印批处理返回的八个动作块。
有 CUDA 时选择 CUDA，否则在 CPU 上运行。

## 调用 Python 接口

`Vvla.act(observation)` 返回一个动作块；传入观测列表则返回动作块列表。
每个 `Observation` 携带图像、状态和语言输入，其布局与预处理由所选策略定义。

[完整示例](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/quickstart.py)
根据合成策略的配置构造输入。对于真实策略，请使用其检查点的相机映射、归一化和 tokenizer。

多个客户端调用 π0.5 时，HTTP 和 WirelessComm 服务可将请求合并为一个批次执行。
批次大小和请求收集窗口的设置见[推理服务](serving.md)。

## 加载 π0.5 模型

π0.5 需要 Python 3.12+、CUDA GPU 和对应依赖组：

```bash
uv sync --python 3.12 --frozen --no-dev --group pi05
uv run --no-sync python examples/pi05_inference.py \
  --ckpt lerobot/pi05_base --envs 1
```

检查点必须在本地可访问，或能通过模型 hub 获取。
示例加载权重后，对合成观测执行推理，并打印返回的动作块。

要使用真实观测，可以在 Python 应用中集成检查点的预处理，或者按[服务指南](serving.md)
配置模型服务。已有的 LeRobot 应用可以使用
[π0.5 适配器](api.md#lerobot-pi05-adapter)。

## 下一步

- [模型](models.md)：选择策略并了解其输入与会话要求。
- [服务](serving.md)：启动 HTTP 或 WirelessComm 推理。
- [多 GPU 并行](parallelism.md)：配置模型副本和多卡执行。
- [RL rollout](api.md#rl-rollout)：生成动作及对数概率，并更新权重。
- [有状态会话](api.md#activevln-sessions)：保留和重置每轮任务的模型状态。
