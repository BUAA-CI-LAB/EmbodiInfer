# StreamVLN · Orin NX 16 GB

## 固定清单前 128 集（2026-10-28 汇总）

从既有导航原始记录中，按固定 256 集清单顺序提取相同的前 128 集；未重新运行推理。各组均为 batch=1，使用 Habitat 闭环观测，非离线录制轨迹回放。

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | INT8 | 70 / 128 | 54.69% | 46.21% | 1359.43 / 3957.49 | 2861 | 92.06 | [r2r-int8.json](r2r-int8.json) |

模型延迟从解码后的 CPU RGB 到 CPU actions，排除加载、预热、HTTP 和模拟器；整集耗时为所选各集实际耗时之和。不同精度会改变闭环轨迹和调用数，这些耗时不是固定输入的无损加速比。INT8 是单独的精度实验。

R2R 完整 256 集清单 SHA256：`ea4c07eeff3eaa2b5cb4085269404b04d5ef19aea8a3db0a44e2f8f576e9f7d8`；前缀首尾 ID 为 `17` / `1710`。同数据集各组的 episode ID、初始位置、朝向与 RGB 哈希已经配对核验。

## 条件与复现入口

Jetson Orin NX 16 GB 共享内存；实测 MAXN、GPU 上限 918 MHz，未启用 Super 模式。Torch 2.10.0、CUDA 12.6、Triton 3.4.0、Transformers 4.51.3。权重位于 NX 本地存储，Habitat 在同局域网的 RTX 4090 上渲染。

该行来自既有 `v9` 串行服务，后续 NX 新优化尚未完成验证；不能作为当前合并源码的性能验收。

```bash
CONFIG=benchmarks/streamvln-benchmark/orin-nx/r2r-int8.json
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocator_configuration"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES=0 python benchmarks/streamvln-benchmark/serve.py \
  --config "$CONFIG" --ready-file artifacts/service-ready.json \
  --token-file artifacts/service-token --port 18767
```

先创建本地 `artifacts/` 和私有 token 文件；将配置中的权重、33 帧预热图片、完整形状预热清单等 `assets/` 路径改成已准备的本地文件。4090 使用原生张量服务 B=1，NX v9 的配置使用串行入口。

模型服务 ready 后，使用 EmbodiRun 的 `benchmarks/streamvln-navigation/benchmark.py`，传入同一份配置、ready、token 和 256 集清单，并显式设置 `--simulator-gpu 0 --batch-size 1 --episode-count 256`。运行方法见 EmbodiRun 的 `benchmarks/README.md`。NX 通过 HTTP 连接同局域网模拟器，模型权重不从 4090 远程读取。

本目录配置保留实测执行参数；仅将机器私有路径替换为相对示例路径，并校正文案中的动作语法描述，因此文件哈希不同于原始配置。上表是既有冻结源码的观测结果，当前剥离并整合的源码只进行了 CPU 检查，尚未重新进行 GPU 验证。原始报告、逐集轨迹、冻结源码和未完成实验均保留在本地，不随此分支提交。

## 来源校验

### R2R INT8 — `nx-int8-matmul-v9`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`db691edf21871fa401d863fea768d8011b6d5bd5b317fb3267990a2c4699b9ac`。
- 前 128 集报告 SHA256：`db13096e8554e40d8765196c2fffe87eb80bb620971a423aa141a9367adf4999`。
- 前 128 集清单 SHA256：`573668f064aa3c5437aaa320c0983c1730e94f3c8741df5b3af53863b325ac5b`。
- 原始配置 SHA256：`4706d89b82a418854122a66c06a7f4a68e6fd849d6e4b90db177ee764e332791`。
- 实测 Python 源码摘要：`060cd5b1bc3ebe19a7012ea7a08b4897144aabad4961372c5edbaca7da26244a`。
- 权重版本：`f1f76c66083c362ddfcd2610167f9c4e4a46c027`。
