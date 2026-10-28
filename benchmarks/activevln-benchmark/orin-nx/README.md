# ActiveVLN · Orin NX 16 GB

## 固定清单前 128 集（2026-10-28 汇总）

从既有导航原始记录中，按固定 256 集清单顺序提取相同的前 128 集；未重新运行推理。各组均为 batch=1，使用 Habitat 闭环观测，非离线录制轨迹回放。

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | BF16 | 56 / 128 | 43.75% | 39.02% | 691.34 / 990.21 | 3916 | 56.41 | [r2r-bf16.json](r2r-bf16.json) |
| R2R | INT8 | 63 / 128 | 49.22% | 42.06% | 537.55 / 833.65 | 3679 | 43.05 | [r2r-int8.json](r2r-int8.json) |

模型延迟从解码后的 CPU RGB 到 CPU actions，排除加载、预热、HTTP 和模拟器；整集耗时为所选各集实际耗时之和。不同精度会改变闭环轨迹和调用数，这些耗时不是固定输入的无损加速比。INT8 是单独的精度实验。

R2R 完整 256 集清单 SHA256：`ea4c07eeff3eaa2b5cb4085269404b04d5ef19aea8a3db0a44e2f8f576e9f7d8`；前缀首尾 ID 为 `17` / `1710`。同数据集各组的 episode ID、初始位置、朝向与 RGB 哈希已经配对核验。

## 条件与复现入口

Jetson Orin NX 16 GB 共享内存；实测 MAXN、GPU 上限 918 MHz，未启用 Super 模式。Torch 2.10.0、CUDA 12.6、Triton 3.4.0、Transformers 4.51.3。权重位于 NX 本地存储，Habitat 在同局域网的 RTX 4090 上渲染。

R2R 权重为 `Arvil/Qwen2.5-VL-3B_rl_r2r_4000`；4090 的 RxR 使用独立的 `Arvil/Qwen2.5-VL-3B_rl_rxr_4000_step350`。两种已测 checkpoint 都使用 R2R 动作语法（0.25 m 前进、15° 转向），RxR 30° 的旧结果未纳入。预热 33 次；校准 INT8 的 `calibration_sha256` 见对应配置。

这里使用共同的模型服务入口 `../orin-nx/serve.py`（同时支持 4090 和 NX）；旧 `run.py` / `config.json` 属于历史实验，4090 旧入口只提供 BF16。新的 `r2r-*.json` / `rxr-*.json` 应直接传入服务入口。

本轮 INT8 原始实验中断于 164 / 256，以上仅统计已完整验证的前 128 集，不能标为 256 集完成。原始日志最后一条不完整记录不属于该前缀，汇总时排除；保留原始记录。

```bash
BENCH=benchmarks/activevln-benchmark
CONFIG="$BENCH/orin-nx/r2r-int8.json"
export PYTHONPATH="$PWD"
export ACTIVEVLN_BENCHMARK_TOKEN="$(cat artifacts/service-token)"
export PYTORCH_CUDA_ALLOC_CONF="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocator_config"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES=0 python "$BENCH/orin-nx/serve.py" \
  --source "$PWD" --config "$CONFIG" --ready-file artifacts/service-ready.json \
  --port 18767 --calibration assets/calibration-maxima.pt --streamed-loading
```

先创建本地 `artifacts/` 和私有 token 文件，并将配置中的 `assets/` 路径改成已准备的本地文件；`--calibration` 必须指向与配置哈希一致的文件，BF16 省略该参数。NX 的 BF16 派生权重须保持记录的精确转换。

模型服务 ready 后，使用 EmbodiRun 的 `benchmarks/activevln-navigation/benchmark.py`，传入同一份配置、ready、token 和 256 集清单，并显式设置 `--simulator-gpu 0 --batch-size 1 --episode-count 256`。运行方法见 EmbodiRun 的 `benchmarks/README.md`。NX 通过 HTTP 连接同局域网模拟器，模型权重不从 4090 远程读取。

本目录配置保留实测执行参数；仅将机器私有路径替换为相对示例路径，并校正文案中的动作语法描述，因此文件哈希不同于原始配置。上表是既有冻结源码的观测结果，当前剥离并整合的源码只进行了 CPU 检查，尚未重新进行 GPU 验证。原始报告、逐集轨迹、冻结源码和未完成实验均保留在本地，不随此分支提交。

## 来源校验

### R2R BF16 — `active-nx-bf16-r2r`

- 原始完成数：256 / 256；原始状态：`complete`。
- 原始报告 SHA256：`4d8f286900383dfafb26a10f37ec077f670ca013a177b0cebb06bd178c1e1230`。
- 前 128 集报告 SHA256：`fc22fda52e16269207b92a283df18ee6cbf10c80d3dcb37e6062f126cc0fa852`。
- 前 128 集清单 SHA256：`573668f064aa3c5437aaa320c0983c1730e94f3c8741df5b3af53863b325ac5b`。
- 原始配置 SHA256：`c82e8bc5bae08a144d120dd39e829c51fdf35195972486e748f1624307dee62b`。
- 实测 Python 源码摘要：`f3689b159726dadf7f25c4ca320345f45897aaf8201cd6c8a0639073848a2675`。
- 权重版本：`160987313e3e869705f42400d1b8f28177044518`。

### R2R INT8 — `active-nx-int8-r2r`

- 原始完成数：164 / 256；原始状态：`infrastructure_failure`。
- 原始报告 SHA256：`b8e4e7e089a55530b0433137a698d21fbc975bed0c364bfe32d245fd72eeeee3`。
- 前 128 集报告 SHA256：`2d946dec5ecc9187413cb00405e5bce4ec85a9271860c43cf6310eb834f22339`。
- 前 128 集清单 SHA256：`573668f064aa3c5437aaa320c0983c1730e94f3c8741df5b3af53863b325ac5b`。
- 原始配置 SHA256：`79cd35c675e7ca6ed4dbab2784be7ffaff07c4495d864dada93f9b477d64ecef`。
- 实测 Python 源码摘要：`f3689b159726dadf7f25c4ca320345f45897aaf8201cd6c8a0639073848a2675`。
- 权重版本：`160987313e3e869705f42400d1b8f28177044518`。

## 历史 256 集实验（不同运行）

R2R val-unseen，BF16 / 校准 INT8 各 256 集、batch=1；Orin NX 推理，RTX 4090 渲染。
两组使用相同的按场景比例抽样清单，配置见 [config.json](config.json)。

```bash
BENCH=benchmarks/activevln-benchmark/orin-nx
DATASET=r2r
python "$BENCH/run.py" bf16 --dataset "$DATASET" \
  --checkpoint "$CKPT" --data-root "$WARMUP_R2R" \
  --manifest "$MANIFESTS/selected-$DATASET-navigation-256.json" --output "$RUN_ROOT"
# INT8：把 bf16 改为 int8，使用新的输出目录，并加 --calibration "$CALIBRATION"
```

`CKPT` 使用已准备的 BF16 快照；`CALIBRATION` 是保留的 `calibration-maxima.pt`，
脚本校验哈希。INT8 量化 204 个文本/视觉 MLP 层，其余保持 BF16。

转发 NX 服务端口，将该组输出的配置、ready、token 和 episode 清单复制到模拟器机器的
`RUN_ROOT`，然后运行：

```bash
EMBODIRUN=/path/to/EmbodiRun HABITAT_PYTHON=/path/to/habitat/python \
SCENES=/path/to/scene_datasets SIMULATOR_GPU=0 \
  bash "$BENCH/navigate.sh" "$DATASET" "$RUN_ROOT" http://127.0.0.1:18767 "$RUN_ROOT/navigation"
```

| 精度 | 完成集数 | 模型 E2E mean / P95 | SR | SPL | NE / Oracle SR | 峰值 CUDA allocated / reserved | 整集耗时合计 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 256 / 256 | 672.31 / 956.86 ms | 46.09% | 0.4184 | 5.838 m / 50.00% | 9.67 / 10.31 GiB | 90.71 min |
| INT8 | 256 / 256 | 505.42 / 786.50 ms | 46.48% | 0.4148 | 5.720 m / 51.56% | 6.96 / 8.01 GiB | 65.05 min |

INT8 平均模型 E2E 降低 **24.8%**，SR 增加 0.39 个百分点，SPL 降低 0.0036。
两组均无 OOM、非法动作或图执行回退。

模型 E2E 从已解码 CPU RGB 到 CPU actions，不含预热、HTTP 和模拟器；整集耗时为
256 集实际耗时之和。两组闭环轨迹、调用数不同（BF16 6480 / INT8 5858），耗时差异不是固定输入的加速比。
完整 HTTP 耗时、逐集 token/action 轨迹与日志保留在本地，不提交 Git。
