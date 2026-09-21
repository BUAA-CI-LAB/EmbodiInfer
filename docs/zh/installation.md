# 安装

从源码安装 EmbodiInfer，并按模型选择依赖组。
安装包名为 `embodiinfer`，在 Python 中通过 `embodiinfer` 导入。

## 环境要求

- uv 0.12.x，在仓库根目录运行。
- 核心包支持 Python 3.10 及以上；`pi05` 依赖组需要 Python 3.12 及以上。
  可用 `--python` 指定解释器。
- `uv.lock` 固定依赖版本。Linux 上的 Torch 和 torchvision 使用官方 CUDA 13.0（`cu130`）软件源，
  其他平台使用 PyPI 发行包。

## 用 uv 安装

在仓库根目录使用 uv 0.12.x：

```bash
uv sync --frozen                              # 核心包 + 开发工具
uv sync --frozen --extra serve                # + websocket 服务
uv sync --frozen --extra wireless             # + WirelessComm 策略服务
uv sync --python 3.12 --frozen --no-dev --group pi05  # pi0.5 运行时
uv sync --frozen --no-dev --group activevln   # ActiveVLN 运行时
uv sync --frozen --no-dev --group qwen25-vln  # Qwen2.5-VL 导航运行时
```

## 模型依赖组

各模型的运行环境通过 `pi05`、`openvla-oft`、`lingbot-vla`、`activevln`、`streamvln`、
`qwen25-vln` 和 `cosmos` 等依赖组选择。需要 WebSocket 服务时，再加上 `--extra serve`。

- `pi05` —— pi0.5 运行时，需要 Python 3.12 及以上。
- `pi05-openpi` —— OpenPI PyTorch、Orbax 检查点准备和分词工具，与 `pi05` 一起安装。
- `openvla-oft`、`lingbot-vla`、`activevln`、`streamvln`、`qwen25-vln`、`cosmos` ——
  对应模型的运行依赖。
- `dm05` —— 仅用于选择模型环境，不安装额外依赖；OpenDM 需单独准备。

互斥的模型依赖组不能装入同一环境，uv 会拒绝不兼容的 Torch 或 Transformers 版本组合。
核心包仍然支持 Python 3.10 及以上。

## 可选组件

- `serve` —— websocket 服务。
- `wireless` —— WirelessComm 策略服务。

HTTP 服务包含在核心包里。无论使用哪种传输，都要安装所选模型的依赖并准备其检查点。

## CUDA 与 PyTorch 版本

CUDA 13.0 锁定环境使用 `torch<2.11`。四个离线基准测试目录各自维护已验证的
Torch 2.13 环境：Thor 使用 CUDA 13.2，4090 使用 CUDA 12.9，按各目录的 `setup_env.py`
和 requirements 文件安装。`--group benchmark` 用于安装公开数据集的读取依赖。

## 加载 OpenPI 与 Orbax 检查点

加载 OpenPI PyTorch 或 Orbax 检查点时，在 PI0.5 环境中安装相应的准备和分词工具：

```bash
uv sync --python 3.12 --frozen --no-dev --group pi05 --group pi05-openpi
```

## 准备 GR00T 参考输出

使用 NVIDIA 上游环境生成 GR00T 参考输出，用于数值一致性检查。具体步骤见
[GR00T benchmark 指南](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md)。

## DM0.5 与 OpenDM

DM0.5 需要单独的 OpenDM 环境，因为它的 Torch 和 Transformers 栈与其他适配器冲突。
在已准备好的 OpenDM 环境中，安装本项目但不要替换该环境的依赖：

```bash
uv pip install --python /path/to/opendm-env/bin/python -e . --no-deps
```

通过 EmbodiRun 部署时，在模型的 `environment_packages` 中指定可安装的 OpenDM 源码或包，
无需在 Control 节点安装。Host 会先按锁文件同步核心环境，再安装这些额外依赖；
因此 OpenDM 及其附加依赖应在该字段中声明，以便同步环境后重新安装。

## 验证安装

启动器 `embodiinfer-http-serve`（或 `embodiinfer-serve`）和 `embodiinfer-wireless-serve`
随项目一起安装。模型专用服务需要匹配的依赖组和检查点。

包内提供无需检查点的合成测试策略 `mock_flow_vla`，可用于验证
引擎、数据并行和 rollout；有 CUDA 时还可验证 CUDA graph 路径。运行通用 mock 基准测试：

```bash
python benchmarks/benchmark.py --preset small --sweep   # hf vs embodiinfer-eager vs embodiinfer-graph
```

该测试比较单请求 eager、批处理 eager 和批处理 CUDA graph 的执行性能。
有 CUDA 时使用 CUDA，否则在 CPU 上运行。
