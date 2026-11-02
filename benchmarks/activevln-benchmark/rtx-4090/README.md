# ActiveVLN · RTX 4090 24 GB

## 首帧性能

| 数据集 | 精度 | Batch | 批调用数 | 整批 mean / P95 (ms) | 摊销 ms/观测 | 观测/s | 峰值 allocated (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | BF16 | 1 | 128 | 45.74 / 54.56 | 45.74 | 21.86 | 7.35 |
| R2R | BF16 | 2 | 64 | 64.33 / 71.35 | 32.16 | 31.09 | 7.89 |
| R2R | BF16 | 4 | 32 | 103.15 / 109.27 | 25.79 | 38.78 | 8.51 |
| R2R | BF16 | 8 | 16 | 185.52 / 193.95 | 23.19 | 43.12 | 9.76 |
| R2R | INT8 | 1 | 128 | 41.60 / 48.86 | 41.60 | 24.04 | 4.70 |
| R2R | INT8 | 2 | 64 | 54.15 / 61.36 | 27.08 | 36.93 | 5.24 |
| R2R | INT8 | 4 | 32 | 81.74 / 88.41 | 20.44 | 48.93 | 5.86 |
| R2R | INT8 | 8 | 16 | 137.73 / 148.48 | 17.22 | 58.09 | 7.11 |
| RxR | BF16 | 1 | 128 | 50.52 / 59.23 | 50.52 | 19.79 | 7.46 |
| RxR | BF16 | 2 | 64 | 72.31 / 82.15 | 36.16 | 27.66 | 7.91 |
| RxR | BF16 | 4 | 32 | 117.47 / 128.80 | 29.37 | 34.05 | 8.53 |
| RxR | BF16 | 8 | 16 | 217.57 / 231.84 | 27.20 | 36.77 | 9.79 |
| RxR | INT8 | 1 | 128 | 43.08 / 50.43 | 43.08 | 23.21 | 4.80 |
| RxR | INT8 | 2 | 64 | 58.66 / 66.84 | 29.33 | 34.10 | 5.26 |
| RxR | INT8 | 4 | 32 | 89.71 / 100.45 | 22.43 | 44.59 | 5.88 |
| RxR | INT8 | 8 | 16 | 160.29 / 170.46 | 20.04 | 49.91 | 7.14 |

### 场景

R2R / RxR val-unseen，各取与 [Thor](../thor/README.md) 相同的固定 128 集首帧和原指令，每集从空历史开始。同一组输入预热一遍后正式测量一遍；不渲染、不计算 SR/SPL，也不覆盖长历史复用。

2026-11-01，单张 RTX 4090 24 GB，驱动 580.82.07，功耗上限 450 W；Torch 2.10.0 / CUDA 12.8 / Triton 3.6.0 / Transformers 4.51.3，CPU 线程数为 4。使用与 Thor 相同的权重版本和首帧脚本。投机解码、GPU 预处理、CUDA Graph、融合算子和 packed prefill 开启；本轮图缺失回退为 0。

计时从解码后的 CPU RGB 到 CPU actions，包含完整动作生成；不含加载、预热、PNG 解码、HTTP 和仿真。整批延迟与摊销每观测耗时分别统计。

权重：R2R 使用 `Arvil/Qwen2.5-VL-3B_rl_r2r_4000@1609873`，RxR 使用 `Arvil/Qwen2.5-VL-3B_rl_rxr_4000_step350@476f841`。

4090 的 KV 工作区配置为 4,096 tokens，池容量配置为 4,096 × batch；Thor 使用 128,000 × batch。此调整缩小未使用的预分配，以适配 24 GB 显存；首帧指令长度、生成上限、图捕获桶和解码机制保持一致。batch=1 使用 resident KV，batch>1 使用独立会话 KV 与增量提交；INT8 为同一校准的 204 个文本/视觉 MLP 投影。

### 复现

测量代码为 [40f364a](https://github.com/yufoo1/EmbodiInfer/tree/40f364aef40de263c2ca42754b141bfc7e15ffa0)。在 EmbodiInfer 根目录执行。`VLN_CONFIGS` 指向仓库外的实验配置目录，先调整配置中的本地路径，准备权重、校准文件和固定输入；`VLN_INPUTS` 包含 `cache-manifest.json`、`r2r/`、`rxr/`。配置及原始报告保留在实验工作区，输出目录须不存在。

```bash
export VLN_CONFIGS=/path/to/local-vln-configs
export VLN_INPUTS=/path/to/vln-firstframe/inputs
CONFIG="$VLN_CONFIGS/rtx4090-activevln-bf16-r2r-b1.json"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/vln_firstframe.py \
  --config "$CONFIG" --inputs "$VLN_INPUTS" \
  --output results/rtx-4090/activevln/firstframe/r2r-bf16-b1
```

## 闭环导航（历史数据）

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | BF16 | 58 / 128 | 45.31% | 41.27% | 49.18 / 68.81 | 3848 | 12.08 | [r2r-bf16.json](r2r-bf16.json) |
| R2R | INT8 | 57 / 128 | 44.53% | 39.87% | 47.87 / 66.86 | 3852 | 11.67 | [r2r-int8.json](r2r-int8.json) |
| RXR | BF16 | 61 / 128 | 47.66% | 38.09% | 71.82 / 129.74 | 5322 | 18.83 | [rxr-bf16.json](rxr-bf16.json) |
| RXR | INT8 | 65 / 128 | 50.78% | 42.11% | 68.04 / 131.10 | 5643 | 19.57 | [rxr-int8.json](rxr-int8.json) |

### 实验场景

R2R / RxR val-unseen，batch=1，固定 256 集清单的前 128 集。前进 0.25 m、转向 15°，最多 500 步；STOP 时距目标小于 3 m 记为成功。

RTX 4090 24 GB，推理与 Habitat 渲染使用同一张 GPU 0。Torch 2.10.0 / CUDA 12.8 / Triton 3.6.0 / Transformers 4.51.3。权重版本和优化参数见配置。

模型延迟从解码后的 CPU RGB 到 CPU actions，不含加载、预热、HTTP 和仿真；整集耗时为 128 集耗时之和。数据来自冻结版本，当前分支未重测。

### 复现

在 EmbodiInfer 根目录执行；先修改配置中的本地路径，准备预热数据、完整 256 集清单、`artifacts/` 及私有 `artifacts/service-token`。INT8 校准文件须匹配配置中的哈希；BF16 改用 `r2r-bf16.json` 并去掉 `--calibration`。RxR 使用 `rxr-*.json`，下方 `--datasets` 改为 `rxr`，动作语法仍为 15°。

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

服务 ready 后，在渲染机设置 `EMBODIRUN`、`HABITAT_PYTHON`、`SCENES`、`MANIFESTS` 和 `CONFIG`，使用同一份配置、ready 与 token：

```bash
PYTHONPATH="$EMBODIRUN/src" python "$EMBODIRUN/benchmarks/activevln-navigation/benchmark.py" \
  --root "$MANIFESTS" --datasets r2r --episode-count 256 --batch-size 1 \
  --habitat-python "$HABITAT_PYTHON" --scenes "$SCENES" \
  --config "$CONFIG" --endpoint http://127.0.0.1:18767 \
  --service-ready artifacts/service-ready.json --token-file artifacts/service-token \
  --simulator-gpu 0 --output artifacts/navigation
```

按固定清单顺序统计前 128 集，与上表保持一致。
