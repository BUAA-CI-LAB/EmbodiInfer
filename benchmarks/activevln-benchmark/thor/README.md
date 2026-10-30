# ActiveVLN · Jetson AGX Thor 128 GB

## 首帧性能

| 数据集 | 精度 | Batch | 批调用数 | 整批 mean / P95 (ms) | 摊销 ms/观测 | 观测/s | 峰值 allocated (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | BF16 | 1 | 128 | 116.57 / 142.78 | 116.57 | 8.58 | 11.52 |
| R2R | BF16 | 2 | 64 | 179.00 / 203.90 | 89.50 | 11.17 | 16.32 |
| R2R | BF16 | 4 | 32 | 250.39 / 269.53 | 62.60 | 15.98 | 25.49 |
| R2R | BF16 | 8 | 16 | 421.27 / 435.19 | 52.66 | 18.99 | 43.76 |
| R2R | INT8 | 1 | 128 | 114.74 / 131.73 | 114.74 | 8.72 | 8.85 |
| R2R | INT8 | 2 | 64 | 184.20 / 202.14 | 92.10 | 10.86 | 13.65 |
| R2R | INT8 | 4 | 32 | 327.25 / 348.34 | 81.81 | 12.22 | 22.83 |
| R2R | INT8 | 8 | 16 | 618.77 / 646.10 | 77.35 | 12.93 | 41.11 |
| RxR | BF16 | 1 | 128 | 124.40 / 148.73 | 124.40 | 8.04 | 11.68 |
| RxR | BF16 | 2 | 64 | 174.24 / 204.48 | 87.12 | 11.48 | 16.38 |
| RxR | BF16 | 4 | 32 | 291.69 / 336.42 | 72.92 | 13.71 | 25.52 |
| RxR | BF16 | 8 | 16 | 500.91 / 536.31 | 62.61 | 15.97 | 43.79 |
| RxR | INT8 | 1 | 128 | 132.77 / 156.79 | 132.77 | 7.53 | 9.01 |
| RxR | INT8 | 2 | 64 | 216.56 / 253.40 | 108.28 | 9.24 | 13.73 |
| RxR | INT8 | 4 | 32 | 388.97 / 449.79 | 97.24 | 10.28 | 22.86 |
| RxR | INT8 | 8 | 16 | 753.06 / 829.67 | 94.13 | 10.62 | 41.14 |

### 场景

R2R / RxR val-unseen，各取固定清单的 128 集首帧和原指令，每集从空历史开始。同一组输入预热一遍后正式测量一遍；不计算 SR/SPL，也不覆盖长历史复用。

Thor MAXN，GPU 1575 MHz；Torch 2.13.0+cu132 / CUDA 13.2 / Triton 3.7.1 / Transformers 4.51.3。权重和图片均在本地。投机解码、GPU 预处理、CUDA Graph、融合算子和 packed prefill 开启；本轮图缺失回退为 0。

计时从解码后的 CPU RGB 到 CPU actions，包含完整动作生成；不含加载、预热、PNG 解码、HTTP 和仿真。整批延迟与摊销每观测耗时分别统计。

### 复现

在 EmbodiInfer 根目录执行。`VLN_CONFIGS` 指向仓库外保存的实验配置目录，包含 `activevln/firstframe/` 和 `activevln/closed-loop/`。先按本地配置准备权重及校准文件；`assets/vln-firstframe/` 保存 `cache-manifest.json`、`r2r/`、`rxr/` 和固定清单 `manifests/`。配置和原始报告均保留在实验工作区。输出目录须不存在。

```bash
export VLN_CONFIGS=/path/to/local-vln-configs
CONFIG="$VLN_CONFIGS/activevln/firstframe/r2r-bf16-b1.json"
PYTHONPATH=. python benchmarks/vln_firstframe.py \
  --config "$CONFIG" --inputs assets/vln-firstframe \
  --output results/thor/activevln/firstframe/r2r-bf16-b1
```

## 闭环导航（历史数据）

| 数据集 | 精度 | Batch | 成功 / 128 | SR | SPL | 整批 mean / P95 (ms) | 批调用数 | 整集耗时合计 (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | BF16 | 1 | 55 / 128 | 42.97% | 38.69% | 140.65 / 204.85 | 3947 | 17.87 |
| R2R | INT8 | 1 | 56 / 128 | 43.75% | 39.50% | 129.72 / 191.84 | 4393 | 18.97 |

### 场景

同一台 Thor 推理、局域网 RTX 4090 运行 Habitat，固定 R2R val-unseen 128 集完整闭环。前进 0.25 m、转向 15°、最多 500 步；STOP 时距目标小于 3 m 记为成功。模型计时口径同上；整集耗时为 128 集耗时之和，包含仿真及通信，多 batch 时不等于运行墙钟时长。

历史闭环原始数据保留在实验工作区，上表独立汇总，不并入首帧统计。两种精度的实际轨迹及调用数可能不同。

本次闭环仅完成 R2R batch=1。首帧 batch=1 使用 resident KV，batch>1 使用独立会话 KV 与增量提交。INT8 为校准后的 204 个文本/视觉 MLP 投影，其余层保留 BF16；Thor SM110 使用同一算术内核。

### 复现

使用同一 `VLN_CONFIGS` 本地配置目录，准备预热数据及私有 `artifacts/service-token`。在 Thor 的 EmbodiInfer 根目录启动服务：

```bash
BENCH=benchmarks/activevln-benchmark
CONFIG="$VLN_CONFIGS/activevln/closed-loop/r2r-int8-b1.json"
export PYTHONPATH="$PWD:$PWD/$BENCH"
export PYTORCH_CUDA_ALLOC_CONF="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocator_config"])' "$CONFIG")"
export ACTIVEVLN_BENCHMARK_TOKEN="$(cat artifacts/service-token)"
CUDA_VISIBLE_DEVICES=0 python "$BENCH/orin-nx/serve.py" \
  --source "$PWD" --config "$CONFIG" --ready-file artifacts/service-ready.json \
  --calibration assets/calibration/activevln-r2r.pt --port 18767
```

服务 ready 后将端口转发至渲染机，并复制同一份 config、ready、token。在渲染机设置 `EMBODIRUN`、`HABITAT_PYTHON`、`SCENES`、`MANIFESTS`、`CONFIG` 后执行；`MANIFESTS` 为固定 128 集清单目录，batch 与配置一致：

```bash
PYTHONPATH="$EMBODIRUN/src" python "$EMBODIRUN/benchmarks/activevln-navigation/benchmark.py" \
  --root "$MANIFESTS" --datasets r2r --episode-count 128 --batch-size 1 \
  --habitat-python "$HABITAT_PYTHON" --scenes "$SCENES" \
  --config "$CONFIG" --endpoint http://127.0.0.1:18767 \
  --service-ready artifacts/service-ready.json --token-file artifacts/service-token \
  --simulator-gpu 0 --output artifacts/navigation-new
```

BF16 使用对应配置；ActiveVLN 同时去掉 `--calibration`。保留原始输出，复跑写入新目录。
