# ActiveVLN · Orin NX 16 GB

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | BF16 | 56 / 128 | 43.75% | 39.02% | 691.34 / 990.21 | 3916 | 56.41 | [r2r-bf16.json](r2r-bf16.json) |
| R2R | INT8 | 63 / 128 | 49.22% | 42.06% | 537.55 / 833.65 | 3679 | 43.05 | [r2r-int8.json](r2r-int8.json) |

## 实验场景

R2R val-unseen，batch=1，固定 256 集清单的前 128 集。前进 0.25 m、转向 15°，最多 500 步；STOP 时距目标小于 3 m 记为成功。

Orin NX 16 GB，MAXN、GPU 上限 918 MHz（非 Super）；本地权重，同局域网 RTX 4090 渲染。Torch 2.10.0 / CUDA 12.6 / Triton 3.4.0 / Transformers 4.51.3。权重版本和优化参数见配置。

模型延迟从解码后的 CPU RGB 到 CPU actions，不含加载、预热、HTTP 和仿真；整集耗时为 128 集耗时之和。数据来自冻结版本，当前分支未重测。

## 复现

在 EmbodiInfer 根目录执行；先修改配置中的本地路径，准备预热数据、完整 256 集清单、`artifacts/` 及私有 `artifacts/service-token`。INT8 校准文件须匹配配置中的哈希；BF16 改用 `r2r-bf16.json` 并去掉 `--calibration`。

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

NX 服务 ready 后，将 18767 端口转发到渲染机；在渲染机设置 `EMBODIRUN`、`HABITAT_PYTHON`、`SCENES`、`MANIFESTS` 和 `CONFIG`，使用同一份配置、ready 与 token：

```bash
PYTHONPATH="$EMBODIRUN/src" python "$EMBODIRUN/benchmarks/activevln-navigation/benchmark.py" \
  --root "$MANIFESTS" --datasets r2r --episode-count 256 --batch-size 1 \
  --habitat-python "$HABITAT_PYTHON" --scenes "$SCENES" \
  --config "$CONFIG" --endpoint http://127.0.0.1:18767 \
  --service-ready artifacts/service-ready.json --token-file artifacts/service-token \
  --simulator-gpu 0 --output artifacts/navigation
```

按固定清单顺序统计前 128 集，与上表保持一致。
