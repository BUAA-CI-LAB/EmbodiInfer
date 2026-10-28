# ActiveVLN · RTX 4090

## 固定清单前 128 集（2026-10-28 汇总）

从既有导航原始记录中，按固定 256 集清单顺序提取相同的前 128 集；未重新运行推理。各组均为 batch=1，使用 Habitat 闭环观测，非离线录制轨迹回放。

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | BF16 | 58 / 128 | 45.31% | 41.27% | 49.18 / 68.81 | 3848 | 12.08 | [r2r-bf16.json](r2r-bf16.json) |
| R2R | INT8 | 57 / 128 | 44.53% | 39.87% | 47.87 / 66.86 | 3852 | 11.67 | [r2r-int8.json](r2r-int8.json) |
| RXR | BF16 | 61 / 128 | 47.66% | 38.09% | 71.82 / 129.74 | 5322 | 18.83 | [rxr-bf16.json](rxr-bf16.json) |
| RXR | INT8 | 65 / 128 | 50.78% | 42.11% | 68.04 / 131.10 | 5643 | 19.57 | [rxr-int8.json](rxr-int8.json) |

模型延迟从解码后的 CPU RGB 到 CPU actions，排除加载、预热、HTTP 和模拟器；整集耗时为所选各集实际耗时之和。不同精度会改变闭环轨迹和调用数，这些耗时不是固定输入的无损加速比。INT8 是单独的精度实验。

R2R 完整 256 集清单 SHA256：`ea4c07eeff3eaa2b5cb4085269404b04d5ef19aea8a3db0a44e2f8f576e9f7d8`；前缀首尾 ID 为 `17` / `1710`。同数据集各组的 episode ID、初始位置、朝向与 RGB 哈希已经配对核验。

## 条件与复现入口

RTX 4090 24 GB，推理与 Habitat 渲染使用同一张 GPU 0；Torch 2.10.0、CUDA 12.8、Triton 3.6.0、Transformers 4.51.3。

R2R 权重为 `Arvil/Qwen2.5-VL-3B_rl_r2r_4000`；4090 的 RxR 使用独立的 `Arvil/Qwen2.5-VL-3B_rl_rxr_4000_step350`。两种已测 checkpoint 都使用 R2R 动作语法（0.25 m 前进、15° 转向），RxR 30° 的旧结果未纳入。预热 33 次；校准 INT8 的 `calibration_sha256` 见对应配置。

这里使用共同的模型服务入口 `../orin-nx/serve.py`（同时支持 4090 和 NX）；旧 `run.py` / `config.json` 属于历史实验，4090 旧入口只提供 BF16。新的 `r2r-*.json` / `rxr-*.json` 应直接传入服务入口。

```bash
BENCH=benchmarks/activevln-benchmark
CONFIG="$BENCH/rtx-4090/r2r-int8.json"
export PYTHONPATH="$PWD"
export ACTIVEVLN_BENCHMARK_TOKEN="$(cat artifacts/service-token)"
export PYTORCH_CUDA_ALLOC_CONF="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocator_config"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES=0 python "$BENCH/orin-nx/serve.py" \
  --source "$PWD" --config "$CONFIG" --ready-file artifacts/service-ready.json \
  --port 18767 --calibration assets/calibration-maxima.pt
```

先创建本地 `artifacts/` 和私有 token 文件，并将配置中的 `assets/` 路径改成已准备的本地文件；`--calibration` 必须指向与配置哈希一致的文件，BF16 省略该参数。NX 的 BF16 派生权重须保持记录的精确转换。

模型服务 ready 后，使用 EmbodiRun 的 `benchmarks/activevln-navigation/benchmark.py`，传入同一份配置、ready、token 和 256 集清单，并显式设置 `--simulator-gpu 0 --batch-size 1 --episode-count 256`。运行方法见 EmbodiRun 的 `benchmarks/README.md`。NX 通过 HTTP 连接同局域网模拟器，模型权重不从 4090 远程读取。

本目录配置保留实测执行参数；仅将机器私有路径替换为相对示例路径，并校正文案中的动作语法描述，因此文件哈希不同于原始配置。上表是既有冻结源码的观测结果，当前剥离并整合的源码只进行了 CPU 检查，尚未重新进行 GPU 验证。原始报告、逐集轨迹、冻结源码和未完成实验均保留在本地，不随此分支提交。

## 来源校验

### R2R BF16 — `active-4090-bf16-r2r`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`87ae4941c875eacbb6b14dfe1fdddf75fd411854a81ae096e5501820fa55121a`。
- 前 128 集报告 SHA256：`d1ef81b055c1dea70a230702fdd6b33d3d4358c8c4520746f57cc15dd6051179`。
- 前 128 集清单 SHA256：`573668f064aa3c5437aaa320c0983c1730e94f3c8741df5b3af53863b325ac5b`。
- 原始配置 SHA256：`d8874876f3ba7f95e523c4057e2cd8c06f6da81afdabb1c13a4cc4602c5481ff`。
- 实测 Python 源码摘要：`f3689b159726dadf7f25c4ca320345f45897aaf8201cd6c8a0639073848a2675`。
- 权重版本：`160987313e3e869705f42400d1b8f28177044518`。

### R2R INT8 — `active-4090-int8-r2r`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`b862d1876214f657668a81a5ebaa0d7298aa398d73ec71a0c448995f8605b35e`。
- 前 128 集报告 SHA256：`202b0f482731be15a425921f24d2ba8a8656e022ddd6ccfb6c4b3bee5e5ef13a`。
- 前 128 集清单 SHA256：`573668f064aa3c5437aaa320c0983c1730e94f3c8741df5b3af53863b325ac5b`。
- 原始配置 SHA256：`4adfe3f9edba9d51426f1a0ee471962a149828d85e7f83143a5945d5185e80d6`。
- 实测 Python 源码摘要：`f3689b159726dadf7f25c4ca320345f45897aaf8201cd6c8a0639073848a2675`。
- 权重版本：`160987313e3e869705f42400d1b8f28177044518`。

### RXR BF16 — `active-4090-bf16-rxr-protocol-v18`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`93f38732f0c61bf5e703d0a651b5ca9cc0e46a3e5ce16a71665a851649b6748c`。
- 前 128 集报告 SHA256：`82b325c1eac572f07482814e8fc5dab49940c1f09ce7288b9a24c81dd1b22d89`。
- 前 128 集清单 SHA256：`f07dfa4e4a386e21aa2f8f4c7066dd8ea1f6a6781656ad69c3a8e324577a6bce`。
- 原始配置 SHA256：`5bcce090589d2cc4ca215c2868935b86079802770e7551c715d608ecfc9cbc3c`。
- 实测 Python 源码摘要：`f3689b159726dadf7f25c4ca320345f45897aaf8201cd6c8a0639073848a2675`。
- 权重版本：`476f8410abb7749da8f5fed810c020984d578564`。

### RXR INT8 — `active-4090-int8-rxr-protocol-v18`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`2f8a0b1839d21ad1c0d97676f85fe6b74ee27f9b20d3a71f85dcb9a134fafe24`。
- 前 128 集报告 SHA256：`cb7ffe33b4a5086614014451cc2902124ff72fcdd64c8561c4823c52bcfe0b87`。
- 前 128 集清单 SHA256：`f07dfa4e4a386e21aa2f8f4c7066dd8ea1f6a6781656ad69c3a8e324577a6bce`。
- 原始配置 SHA256：`233afd07ba038bcf6c6d8091873a6ddbc73c4f56f0270e227373515de48cf295`。
- 实测 Python 源码摘要：`f3689b159726dadf7f25c4ca320345f45897aaf8201cd6c8a0639073848a2675`。
- 权重版本：`476f8410abb7749da8f5fed810c020984d578564`。
