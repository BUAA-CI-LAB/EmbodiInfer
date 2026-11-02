# StreamVLN · RTX 4090 24 GB

## R2R 首帧：官方实现对比（2026-11-02）

| Batch | EmbodiInfer mean / P95 (ms) | 官方 mean / P95 (ms) | 观测/s（EmbodiInfer / 官方） | mean 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 91.25 / 93.74 | 276.84 / 289.67 | 10.96 / 3.61 | 3.03× |
| 2 | 132.20 / 138.94 | 414.05 / 432.91 | 15.13 / 4.83 | 3.13× |
| 4 | 214.90 / 223.88 | 530.92 / 550.44 | 18.61 / 7.53 | 2.47× |
| 8 | 381.75 / 391.61 | 768.72 / 780.07 | 20.96 / 10.41 | 2.01× |

### 首个正式批次与后续批次

从上表同一次测量的逐批记录拆分，未重跑。首批指完整预热后的第一个正式 batch；后续批次仍是其他 episode 的独立首帧。batch=1/2/4/8 的后续部分分别包含 127/63/31/15 批、127/126/124/120 个观测；所有配置统一按这个规则拆分，不再筛除任何批次。上表仍保留完整 128 帧结果。

FPS 为完整观测/s：首批按 `1000 × batch / 首批整批耗时(ms)` 计算；后续按剩余观测总数除以剩余批次总耗时计算。

| 引擎 | Batch | 首批 FPS | 后续 FPS | 全量 FPS |
| --- | ---: | ---: | ---: | ---: |
| EmbodiInfer | 1 | 10.95 | 10.96 | 10.96 |
| EmbodiInfer | 2 | 14.95 | 15.13 | 15.13 |
| EmbodiInfer | 4 | 17.97 | 18.64 | 18.61 |
| EmbodiInfer | 8 | 20.44 | 20.99 | 20.96 |
| 官方 | 1 | 3.58 | 3.61 | 3.61 |
| 官方 | 2 | 4.88 | 4.83 | 4.83 |
| 官方 | 4 | 7.11 | 7.55 | 7.53 |
| 官方 | 8 | 10.29 | 10.41 | 10.41 |

后续批次含不同输入，仅用于展示批次分布。

### 场景与配置

单张 RTX 4090 24 GB、BF16，R2R val-unseen、seed=42 的冻结 256 集清单前 128 集，各取首帧，batch=1/2/4/8 分别计时 128/64/32/16 次真实批调用。CPU RGB → 完整 CPU actions，包含预处理；不含磁盘、加载、预热、重置和渲染。128 帧完整预热一遍再测一遍，每批清空历史和已学习的投机候选，保留图与编译结果；EmbodiInfer 配置 `firstframe.reset_speculative_prior=true`。下方历史首帧数据未采用这个候选重置规则，不混用。

两侧使用相同 StreamVLN 权重 `f1f76c6`、输入、prompt 和 greedy 解码，repetition penalty=1.05、max new tokens=128；每组均得到 128 个有效输出、共 1,024 tokens。按 batch=1/2/4/8，官方与 EmbodiInfer 动作序列一致的帧数为 125/126/127/127（各 128 帧）。这里只比较性能，不报告导航精度。

EmbodiInfer 基于 `2a3f053`，开启语法树投机、GPU 预处理、CUDA Graph、融合算子、GQA 和 packed prefill；本轮图回退为 0。[官方源码](https://github.com/InternRobotics/StreamVLN/tree/e48f6ff7e9201d93aae003e8f64b04c00cec13bc) 保持不变，使用原生 `generate`、FlashAttention 2、KV cache、`num_logits_to_keep=1` 和左侧 padding。

驱动 580.82.07，450 W，CPU 线程数 4。EmbodiInfer：Torch 2.10.0 / CUDA 12.8 / Transformers 4.51.3 / Triton 3.6.0；官方独立环境：Torch 2.1.2+cu121 / Transformers 4.45.1 / FlashAttention 2.5.9.post1。这是各引擎可用环境的端到端比较。

### 复现

在仓库根目录执行，`VLN_CONFIGS`、`VLN_INPUTS`、`VLN_RESULTS` 指向仓库外的配置、固定输入和输出目录；配置中的权重、SigLIP 和清单路径需对应本机。`EMBODI_PYTHON`、`OFFICIAL_PYTHON` 分别使用上述独立环境，`STREAMVLN_SOURCE` 指向上述官方版本。配置与原始数据保留在实验目录，不提交到仓库。

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for B in 1 2 4 8; do
  PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.75 \
  "$EMBODI_PYTHON" benchmarks/vln_firstframe.py \
    --config "$VLN_CONFIGS/rtx4090-streamvln-bf16-r2r-freshprior-b$B.json" \
    --inputs "$VLN_INPUTS" --output "$VLN_RESULTS/embodi-stream-b$B"
  "$OFFICIAL_PYTHON" benchmarks/vln_firstframe_reference.py --engine official \
    --official-source "$STREAMVLN_SOURCE" \
    --config "$VLN_CONFIGS/rtx4090-streamvln-bf16-r2r-official-fa2-b$B.json" \
    --inputs "$VLN_INPUTS" --output "$VLN_RESULTS/official-stream-b$B"
done
```

## 首帧性能（历史数据，2026-11-01）

| 数据集 | 精度 | Batch | 批调用数 | 整批 mean / P95 (ms) | 摊销 ms/观测 | 观测/s | 峰值 allocated (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | BF16 | 1 | 128 | 74.27 / 89.63 | 74.27 | 13.46 | 15.77 |
| R2R | BF16 | 2 | 64 | 114.16 / 129.17 | 57.08 | 17.52 | 16.48 |
| R2R | BF16 | 4 | 32 | 195.89 / 213.87 | 48.97 | 20.42 | 17.90 |
| R2R | BF16 | 8 | 16 | 360.18 / 380.50 | 45.02 | 22.21 | 20.71 |
| R2R | INT8 | 1 | 128 | 52.32 / 62.79 | 52.32 | 19.11 | 9.66 |
| R2R | INT8 | 2 | 64 | 80.35 / 92.23 | 40.18 | 24.89 | 10.37 |
| R2R | INT8 | 4 | 32 | 134.32 / 146.66 | 33.58 | 29.78 | 11.77 |
| R2R | INT8 | 8 | 16 | 239.58 / 251.54 | 29.95 | 33.39 | 14.59 |
| RxR | BF16 | 1 | 128 | 84.03 / 102.22 | 84.03 | 11.90 | 15.87 |
| RxR | BF16 | 2 | 64 | 134.21 / 163.33 | 67.11 | 14.90 | 16.59 |
| RxR | BF16 | 4 | 32 | 240.95 / 280.05 | 60.24 | 16.60 | 18.03 |
| RxR | BF16 | 8 | 16 | 445.32 / 504.46 | 55.67 | 17.96 | 20.86 |
| RxR | INT8 | 1 | 128 | 57.04 / 68.89 | 57.04 | 17.53 | 9.71 |
| RxR | INT8 | 2 | 64 | 90.98 / 109.59 | 45.49 | 21.98 | 10.43 |
| RxR | INT8 | 4 | 32 | 155.30 / 178.48 | 38.82 | 25.76 | 11.83 |
| RxR | INT8 | 8 | 16 | 317.40 / 442.90 | 39.67 | 25.20 | 14.67 |

### 场景

R2R / RxR val-unseen，各取与 [Thor](../thor/README.md) 相同的固定 128 集首帧和原指令，每集从空历史开始。同一组输入预热一遍后正式测量一遍；不渲染、不计算 SR/SPL，也不覆盖长历史复用。

2026-11-01，单张 RTX 4090 24 GB，驱动 580.82.07，功耗上限 450 W；Torch 2.10.0 / CUDA 12.8 / Triton 3.6.0 / Transformers 4.51.3，CPU 线程数为 4。使用与 Thor 相同的权重版本和首帧脚本。投机解码、GPU 预处理、CUDA Graph、融合算子和 packed prefill 开启；本轮图缺失回退为 0。

计时从解码后的 CPU RGB 到 CPU actions，包含完整动作生成；不含加载、预热、PNG 解码、HTTP 和仿真。整批延迟与摊销每观测耗时分别统计。

R2R / RxR 使用同一 StreamVLN 权重，revision `f1f76c66083c362ddfcd2610167f9c4e4a46c027`；INT8 使用 `triton_native` 后端。

### 复现

测量代码为 [40f364a](https://github.com/yufoo1/EmbodiInfer/tree/40f364aef40de263c2ca42754b141bfc7e15ffa0)。在 EmbodiInfer 根目录执行。`VLN_CONFIGS` 指向仓库外的实验配置目录，先调整配置中的本地路径，准备权重、校准文件和固定输入；`VLN_INPUTS` 包含 `cache-manifest.json`、`r2r/`、`rxr/`。配置及原始报告保留在实验工作区，输出目录须不存在。

```bash
export VLN_CONFIGS=/path/to/local-vln-configs
export VLN_INPUTS=/path/to/vln-firstframe/inputs
CONFIG="$VLN_CONFIGS/rtx4090-streamvln-bf16-r2r-b1.json"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/vln_firstframe.py \
  --config "$CONFIG" --inputs "$VLN_INPUTS" \
  --output results/rtx-4090/streamvln/firstframe/r2r-bf16-b1
```

## 闭环导航（历史数据）

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | BF16 | 71 / 128 | 55.47% | 47.41% | 105.72 / 252.07 | 2891 | 26.58 | [r2r-bf16.json](r2r-bf16.json) |
| R2R | INT8 | 70 / 128 | 54.69% | 46.14% | 68.56 / 147.99 | 2945 | 24.87 | [r2r-int8.json](r2r-int8.json) |

### 实验场景

R2R val-unseen，batch=1，固定 256 集清单的前 128 集。前进 0.25 m、转向 15°，最多 500 步；STOP 时距目标小于 3 m 记为成功。

RTX 4090 24 GB，推理与 Habitat 渲染使用同一张 GPU 0。Torch 2.10.0 / CUDA 12.8 / Triton 3.6.0 / Transformers 4.51.3。权重版本和优化参数见配置。

模型延迟从解码后的 CPU RGB 到 CPU actions，不含加载、预热、HTTP 和仿真；整集耗时为 128 集耗时之和。BF16 / INT8 分别使用 v10 / v17 冻结版本。数据来自冻结版本，当前分支未重测。

### 复现

在 EmbodiInfer 根目录执行；先修改配置中的本地路径，准备预热数据、完整 256 集清单、`artifacts/` 及私有 `artifacts/service-token`。BF16 改用 `r2r-bf16.json`。

```bash
CONFIG=benchmarks/streamvln-benchmark/rtx-4090/r2r-int8.json
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocator_configuration"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES=0 python benchmarks/streamvln-benchmark/serve_batch.py \
  --config "$CONFIG" --ready-file artifacts/service-ready.json \
  --token-file artifacts/service-token --port 18767
```

服务 ready 后，在渲染机设置 `EMBODIRUN`、`HABITAT_PYTHON`、`SCENES`、`MANIFESTS` 和 `CONFIG`，使用同一份配置、ready 与 token：

```bash
PYTHONPATH="$EMBODIRUN/src" python "$EMBODIRUN/benchmarks/streamvln-navigation/benchmark.py" \
  --root "$MANIFESTS" --datasets r2r --episode-count 256 --batch-size 1 \
  --habitat-python "$HABITAT_PYTHON" --scenes "$SCENES" \
  --config "$CONFIG" --endpoint http://127.0.0.1:18767 \
  --service-ready artifacts/service-ready.json --token-file artifacts/service-token \
  --simulator-gpu 0 --output artifacts/navigation
```

按固定清单顺序统计前 128 集，与上表保持一致。
