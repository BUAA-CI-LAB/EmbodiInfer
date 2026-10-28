# StreamVLN · RTX 4090

## 固定清单前 128 集（2026-10-28 汇总）

从既有导航原始记录中，按固定 256 集清单顺序提取相同的前 128 集；未重新运行推理。各组均为 batch=1，使用 Habitat 闭环观测，非离线录制轨迹回放。

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | BF16 | 71 / 128 | 55.47% | 47.41% | 105.72 / 252.07 | 2891 | 26.58 | [r2r-bf16.json](r2r-bf16.json) |
| R2R | INT8 | 70 / 128 | 54.69% | 46.14% | 68.56 / 147.99 | 2945 | 24.87 | [r2r-int8.json](r2r-int8.json) |

模型延迟从解码后的 CPU RGB 到 CPU actions，排除加载、预热、HTTP 和模拟器；整集耗时为所选各集实际耗时之和。不同精度会改变闭环轨迹和调用数，这些耗时不是固定输入的无损加速比。INT8 是单独的精度实验。

R2R 完整 256 集清单 SHA256：`ea4c07eeff3eaa2b5cb4085269404b04d5ef19aea8a3db0a44e2f8f576e9f7d8`；前缀首尾 ID 为 `17` / `1710`。同数据集各组的 episode ID、初始位置、朝向与 RGB 哈希已经配对核验。

## 条件与复现入口

RTX 4090 24 GB，推理与 Habitat 渲染使用同一张 GPU 0；Torch 2.10.0、CUDA 12.8、Triton 3.6.0、Transformers 4.51.3。

BF16 实测冻结源为 `v10`，INT8 为 `v17`；INT8 使用 `triton_native`。树预算 31、刷新周期 64；图像/历史 GPU 处理、CUDA Graph、融合算子与打包 prefill 的具体开关见配置。

```bash
CONFIG=benchmarks/streamvln-benchmark/rtx-4090/r2r-int8.json
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocator_configuration"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES=0 python benchmarks/streamvln-benchmark/serve_batch.py \
  --config "$CONFIG" --ready-file artifacts/service-ready.json \
  --token-file artifacts/service-token --port 18767
```

先创建本地 `artifacts/` 和私有 token 文件；将配置中的权重、33 帧预热图片、完整形状预热清单等 `assets/` 路径改成已准备的本地文件。4090 使用原生张量服务 B=1，NX v9 的配置使用串行入口。

模型服务 ready 后，使用 EmbodiRun 的 `benchmarks/streamvln-navigation/benchmark.py`，传入同一份配置、ready、token 和 256 集清单，并显式设置 `--simulator-gpu 0 --batch-size 1 --episode-count 256`。运行方法见 EmbodiRun 的 `benchmarks/README.md`。NX 通过 HTTP 连接同局域网模拟器，模型权重不从 4090 远程读取。

本目录配置保留实测执行参数；仅将机器私有路径替换为相对示例路径，并校正文案中的动作语法描述，因此文件哈希不同于原始配置。上表是既有冻结源码的观测结果，当前剥离并整合的源码只进行了 CPU 检查，尚未重新进行 GPU 验证。原始报告、逐集轨迹、冻结源码和未完成实验均保留在本地，不随此分支提交。

## 来源校验

### R2R BF16 — `4090-bf16-final-v15`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`2b647f873bedc1cdcef31c04cce8d57a1cbaeb3ff700306d4b2cce60de884673`。
- 前 128 集报告 SHA256：`d7ef406c359f44b26aa68f585c28dcc08f8ed8025089bd01025bdfccfc309fdc`。
- 前 128 集清单 SHA256：`573668f064aa3c5437aaa320c0983c1730e94f3c8741df5b3af53863b325ac5b`。
- 原始配置 SHA256：`ce2c1c38b039b9f85fd01d251d06a2313b30a8526f3fc0a9add5f75c361daf99`。
- 实测 Python 源码摘要：`c4024d25b9cad34139134af6ed5cb6af8b2484ed4646b9d936a2d1db679b48ea`。
- 权重版本：`f1f76c66083c362ddfcd2610167f9c4e4a46c027`。

### R2R INT8 — `4090-int8-final-v15`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`c7ad3dc80d8aa494627d0f2d4f64aa01be630816e59bc434947565e6380f8db7`。
- 前 128 集报告 SHA256：`7574e3411045c48521c7d093d147133537eec8b44f2d74a720ae46bcb61d96ac`。
- 前 128 集清单 SHA256：`573668f064aa3c5437aaa320c0983c1730e94f3c8741df5b3af53863b325ac5b`。
- 原始配置 SHA256：`aa45cf51ed92c3680547617aa449067e5aca54b9024d790176c1261e6da2fe8a`。
- 实测 Python 源码摘要：`c2dafd5109b3dcbe0cd2322b47457caa97ddd095e5e9286cd9889c4c57121a25`。
- 权重版本：`f1f76c66083c362ddfcd2610167f9c4e4a46c027`。
