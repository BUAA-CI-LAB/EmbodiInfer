# StreamVLN · Jetson AGX Thor 128 GB

## 首帧性能

| 数据集 | 精度 | Batch | 批调用数 | 整批 mean / P95 (ms) | 摊销 ms/观测 | 观测/s | 峰值 allocated (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | BF16 | 1 | 128 | 173.19 / 230.20 | 173.19 | 5.77 | 15.77 |
| R2R | BF16 | 2 | 64 | 289.81 / 359.25 | 144.90 | 6.90 | 16.47 |
| R2R | BF16 | 4 | 32 | 411.20 / 462.92 | 102.80 | 9.73 | 17.89 |
| R2R | BF16 | 8 | 16 | 745.83 / 797.24 | 93.23 | 10.73 | 20.70 |
| R2R | INT8 | 1 | 128 | 180.10 / 208.72 | 180.10 | 5.55 | 9.66 |
| R2R | INT8 | 2 | 64 | 312.70 / 346.96 | 156.35 | 6.40 | 10.36 |
| R2R | INT8 | 4 | 32 | 584.76 / 627.74 | 146.19 | 6.84 | 11.77 |
| R2R | INT8 | 8 | 16 | 1181.95 / 1239.81 | 147.74 | 6.77 | 14.58 |
| RxR | BF16 | 1 | 128 | 191.98 / 248.21 | 191.98 | 5.21 | 15.86 |
| RxR | BF16 | 2 | 64 | 313.64 / 376.52 | 156.82 | 6.38 | 16.58 |
| RxR | BF16 | 4 | 32 | 492.22 / 574.89 | 123.05 | 8.13 | 18.01 |
| RxR | BF16 | 8 | 16 | 928.32 / 1038.16 | 116.04 | 8.62 | 20.85 |
| RxR | INT8 | 1 | 128 | 219.65 / 279.72 | 219.65 | 4.55 | 9.71 |
| RxR | INT8 | 2 | 64 | 389.04 / 462.60 | 194.52 | 5.14 | 10.43 |
| RxR | INT8 | 4 | 32 | 727.39 / 846.12 | 181.85 | 5.50 | 11.83 |
| RxR | INT8 | 8 | 16 | 1495.88 / 1702.91 | 186.98 | 5.35 | 14.67 |

### 场景

R2R / RxR val-unseen，各取固定清单的 128 集首帧和原指令，每集从空历史开始。同一组输入预热一遍后正式测量一遍；不计算 SR/SPL，也不覆盖长历史复用。

Thor MAXN，GPU 1575 MHz；Torch 2.13.0+cu132 / CUDA 13.2 / Triton 3.7.1 / Transformers 4.51.3。权重和图片均在本地。投机解码、GPU 预处理、CUDA Graph、融合算子和 packed prefill 开启；本轮图缺失回退为 0。

计时从解码后的 CPU RGB 到 CPU actions，包含完整动作生成；不含加载、预热、PNG 解码、HTTP 和仿真。整批延迟与摊销每观测耗时分别统计。

### 复现

在 EmbodiInfer 根目录执行。`VLN_CONFIGS` 指向仓库外保存的实验配置目录，包含 `streamvln/firstframe/` 和 `streamvln/closed-loop/`。先按本地配置准备权重及校准文件；`assets/vln-firstframe/` 保存 `cache-manifest.json`、`r2r/`、`rxr/` 和固定清单 `manifests/`。配置和原始报告均保留在实验工作区。输出目录须不存在。

```bash
export VLN_CONFIGS=/path/to/local-vln-configs
CONFIG="$VLN_CONFIGS/streamvln/firstframe/r2r-bf16-b1.json"
PYTHONPATH=. python benchmarks/vln_firstframe.py \
  --config "$CONFIG" --inputs assets/vln-firstframe \
  --output results/thor/streamvln/firstframe/r2r-bf16-b1
```

## 闭环导航（历史数据）

| 数据集 | 精度 | Batch | 成功 / 128 | SR | SPL | 整批 mean / P95 (ms) | 批调用数 | 整集耗时合计 (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | BF16 | 1 | 67 / 128 | 52.34% | 45.13% | 231.48 / 465.41 | 2839 | 35.26 |
| R2R | BF16 | 2 | 73 / 128 | 57.03% | 49.08% | 348.33 / 694.95 | 1236 | 54.31 |
| R2R | BF16 | 4 | 74 / 128 | 57.81% | 49.96% | 595.93 / 1086.12 | 645 | 110.48 |
| R2R | INT8 | 1 | 70 / 128 | 54.69% | 46.59% | 246.53 / 783.08 | 3156 | 38.07 |
| R2R | INT8 | 2 | 68 / 128 | 53.12% | 45.14% | 421.96 / 924.62 | 1418 | 64.90 |
| R2R | INT8 | 4 | 71 / 128 | 55.47% | 46.79% | 804.50 / 1587.13 | 806 | 152.50 |
| R2R | INT8 | 8 | 64 / 128 | 50.00% | 43.05% | 1561.78 / 2384.89 | 386 | 284.64 |

### 场景

同一台 Thor 推理、局域网 RTX 4090 运行 Habitat，固定 R2R val-unseen 128 集完整闭环。前进 0.25 m、转向 15°、最多 500 步；STOP 时距目标小于 3 m 记为成功。模型计时口径同上；整集耗时为 128 集耗时之和，包含仿真及通信，多 batch 时不等于运行墙钟时长。

历史闭环原始数据保留在实验工作区，上表独立汇总，不并入首帧统计。两种精度的实际轨迹及调用数可能不同。

BF16 batch=8 的旧闭环计时因资源干扰疑点归档待复测，未列入有效结果；RxR 尚无完整闭环结果。B4/B8 的服务收集窗口为 10000 ms，B1/B2 为 1000 ms；满批立即执行，尾批等待计入整集耗时，不计入模型耗时。

### 复现

使用同一 `VLN_CONFIGS` 本地配置目录，准备预热数据及私有 `artifacts/service-token`。在 Thor 的 EmbodiInfer 根目录启动服务：

```bash
BENCH=benchmarks/streamvln-benchmark
CONFIG="$VLN_CONFIGS/streamvln/closed-loop/r2r-int8-b1.json"
export PYTHONPATH="$PWD:$PWD/$BENCH"
export PYTORCH_CUDA_ALLOC_CONF="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocator_config"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES=0 python "$BENCH/serve_batch.py" \
  --config "$CONFIG" --ready-file artifacts/service-ready.json \
  --token-file artifacts/service-token --max-wait-ms 1000 --port 18767
```

服务 ready 后将端口转发至渲染机，并复制同一份 config、ready、token。在渲染机设置 `EMBODIRUN`、`HABITAT_PYTHON`、`SCENES`、`MANIFESTS`、`CONFIG` 后执行；`MANIFESTS` 为固定 128 集清单目录，batch 与配置一致：

```bash
PYTHONPATH="$EMBODIRUN/src" python "$EMBODIRUN/benchmarks/streamvln-navigation/benchmark.py" \
  --root "$MANIFESTS" --datasets r2r --episode-count 128 --batch-size 1 \
  --habitat-python "$HABITAT_PYTHON" --scenes "$SCENES" \
  --config "$CONFIG" --endpoint http://127.0.0.1:18767 \
  --service-ready artifacts/service-ready.json --token-file artifacts/service-token \
  --simulator-gpu 0 --output artifacts/navigation-new
```

BF16 使用对应配置；ActiveVLN 同时去掉 `--calibration`。保留原始输出，复跑写入新目录。
