# ActiveVLN · RTX 4090 24 GB

## R2R 首帧：原生引擎对比（2026-11-02）

整批 mean / P95，单位 ms；吞吐量为 `1000 × batch / mean`。

| Batch | EmbodiInfer | SGLang SRT | vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 55.39 / 57.88 | 177.22 / 218.77 | 117.26 / 131.11 |
| 2 | 73.60 / 77.25 | 214.48 / 252.51 | 152.81 / 163.19 |
| 4 | 112.02 / 113.83 | 280.95 / 319.75 | 223.69 / 232.43 |
| 8 | 220.13 / 290.67 | 364.19 / 407.99 | 315.69 / 323.11 |

### 首个正式批次与后续批次

从上表同一次测量的逐批记录拆分，未重跑。首批指完整预热后的第一个正式 batch；后续批次仍是其他 episode 的独立首帧。batch=1/2/4/8 的后续部分分别包含 127/63/31/15 批、127/126/124/120 个观测；所有配置统一按这个规则拆分，不再筛除任何批次。上表仍保留完整 128 帧结果。

FPS 为完整观测/s：首批按 `1000 × batch / 首批整批耗时(ms)` 计算；后续按剩余观测总数除以剩余批次总耗时计算。

| 引擎 | Batch | 首批 FPS | 后续 FPS | 全量 FPS |
| --- | ---: | ---: | ---: | ---: |
| EmbodiInfer | 1 | 17.78 | 18.06 | 18.06 |
| EmbodiInfer | 2 | 27.69 | 27.17 | 27.18 |
| EmbodiInfer | 4 | 35.05 | 35.73 | 35.71 |
| EmbodiInfer | 8 | 14.21 | 40.55 | 36.34 |
| SGLang SRT | 1 | 5.99 | 5.64 | 5.64 |
| SGLang SRT | 2 | 10.16 | 9.31 | 9.32 |
| SGLang SRT | 4 | 13.45 | 14.26 | 14.24 |
| SGLang SRT | 8 | 24.79 | 21.80 | 21.97 |
| vLLM | 1 | 7.86 | 8.53 | 8.53 |
| vLLM | 2 | 12.49 | 13.10 | 13.09 |
| vLLM | 4 | 18.32 | 17.87 | 17.88 |
| vLLM | 8 | 25.42 | 25.34 | 25.34 |

后续批次含不同输入，仅用于展示分布；首批尖峰原因尚未定位。

### 场景与配置

单张 RTX 4090 24 GB、BF16，R2R val-unseen、seed=42 的冻结 256 集清单前 128 集，各取首帧；batch=1/2/4/8 分别计时 128/64/32/16 个请求批次，直到整批最后一个输出返回；原生引擎保留自己的动态调度。128 帧完整预热一遍后测一遍，每批清空历史、内容缓存和已学习的投机候选，保留图与编译结果；EmbodiInfer 配置 `firstframe.reset_speculative_prior=true`。计时从 CPU RGB 到完整 CPU actions，包含预处理；SGLang 包含本机 SRT 进程通信，不含磁盘、加载、预热、重置、HTTP 和渲染。

三侧使用 `Arvil/Qwen2.5-VL-3B_rl_r2r_4000@1609873`，相同输入、prompt、greedy 解码、max new tokens=512。repetition penalty 统一为 1.0，避免引擎对输入前缀是否参与惩罚的语义差异；每帧 88 个图像 tokens；每组均有 128 个有效输出、2,688 个生成 tokens。下方 2026-11-01 的首帧数据采用不同的 repetition penalty 与投机候选重置规则，与本表不混用。

| 引擎 | 本次启用的优化 |
| --- | --- |
| EmbodiInfer | 语法树投机（budget=31）、GPU 预处理、CUDA Graph、融合算子、GQA、packed prefill；图回退为 0 |
| SGLang SRT | 原生 NGRAM（12 draft tokens）、torch.compile、full prefill / target-verify 图、ViT 图、FlashInfer 文本注意力、GPU 图像处理与 CUDA IPC、overlap 调度 |
| vLLM | 原生 NGRAM_GPU（16 draft tokens）、optimization level=3、torch.compile、文本 FULL_AND_PIECEWISE / 视觉图、GPU 归一化、FlashAttention；沿用已验证的同步调度配置（async_scheduling=False） |

驱动 580.82.07，450 W，CPU 线程数 4。EmbodiInfer 基于 `2a3f053`，Torch 2.10.0 / CUDA 12.8 / Transformers 4.51.3 / Triton 3.6.0。SGLang 使用 [2351c4d](https://github.com/yufoo1/SGLang-for-RLinf/tree/2351c4d3593f56ff11b7dab7219816aa0ec70750) 的原生 SRT Qwen2.5-VL 路径，未调用训练 actor；Torch 2.14.1 / CUDA 13.0 / Transformers 5.19.0 / Triton 3.8.0。vLLM 0.30.0，Torch 2.13.0 / Transformers 5.17.0 / Triton 3.7.1。三者使用各自独立的兼容环境。

EmbodiInfer batch=8 的首个正式批次为 563.03 ms，其余 15 批为 194.2–199.9 ms；统计保留全部批次，未剔除异常值。按 batch=1/2/4/8，与 EmbodiInfer 动作序列一致的帧数为 SGLang 99/100/100/102、vLLM 108/113/110/110（各 128 帧）；这是输出核验，不是导航精度。只比较性能，不计算 SR/SPL。

### 复现

在仓库根目录执行。`VLN_CONFIGS`、`VLN_INPUTS`、`VLN_RESULTS` 分别指向仓库外的配置、固定输入和输出目录；先修改配置中的权重与清单路径。`EMBODI_PYTHON`、`VLLM_PYTHON`、`SGLANG_PYTHON` 分别指向已配置上述依赖和源码路径的独立环境 Python；SGLang 另需 `torch-memory-saver==0.0.10`、`sglang-kernel==0.4.10` 和 `flashinfer-python==0.7.2`。配置、逐帧输出和启动日志保留在实验目录，不提交到仓库。

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VLLM_ENABLE_V1_MULTIPROCESSING=0
for B in 1 2 4 8; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7 \
  "$EMBODI_PYTHON" benchmarks/vln_firstframe.py \
    --config "$VLN_CONFIGS/rtx4090-activevln-bf16-r2r-rep1-freshprior-b$B.json" \
    --inputs "$VLN_INPUTS" --output "$VLN_RESULTS/embodi-active-b$B"
  "$VLLM_PYTHON" benchmarks/vln_firstframe_reference.py --engine vllm \
    --config "$VLN_CONFIGS/rtx4090-activevln-bf16-r2r-rep1-b$B.json" \
    --inputs "$VLN_INPUTS" --output "$VLN_RESULTS/vllm-active-b$B"
  "$SGLANG_PYTHON" benchmarks/vln_firstframe_reference.py --engine sglang \
    --config "$VLN_CONFIGS/rtx4090-activevln-bf16-r2r-rep1-b$B.json" \
    --inputs "$VLN_INPUTS" --output "$VLN_RESULTS/sglang-active-b$B"
done
```

## 首帧性能：BF16 / INT8（2026-11-01）

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

R2R / RxR val-unseen，各取与 [Thor](../thor/README.md) 相同的固定 128 集首帧和原指令，每集从空历史开始，与 Thor 页同协议。同一组输入预热一遍后正式测量一遍；不渲染、不计算 SR/SPL，也不覆盖长历史复用。

2026-11-01，单张 RTX 4090 24 GB，驱动 580.82.07，功耗上限 450 W；Torch 2.10.0 / CUDA 12.8 / Triton 3.6.0 / Transformers 4.51.3，CPU 线程数为 4。使用与 Thor 相同的权重版本和首帧脚本。投机解码、GPU 预处理、CUDA Graph、融合算子和 packed prefill 开启；图缺失回退为 0。

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

## 闭环导航（R2R / RxR，B=1）

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
