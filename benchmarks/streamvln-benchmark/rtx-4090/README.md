# StreamVLN · RTX 4090

| 数据集 | 精度 | 成功 / 128 | SR | SPL | 模型 mean / P95 (ms) | 模型调用数 | 整集耗时合计 (min) | 配置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R | BF16 | 71 / 128 | 55.47% | 47.41% | 105.72 / 252.07 | 2891 | 26.58 | [r2r-bf16.json](r2r-bf16.json) |
| R2R | INT8 | 70 / 128 | 54.69% | 46.14% | 68.56 / 147.99 | 2945 | 24.87 | [r2r-int8.json](r2r-int8.json) |

## 实验场景

R2R val-unseen，batch=1，固定 256 集清单的前 128 集。前进 0.25 m、转向 15°，最多 500 步；STOP 时距目标小于 3 m 记为成功。

RTX 4090 24 GB，推理与 Habitat 渲染使用同一张 GPU 0。Torch 2.10.0 / CUDA 12.8 / Triton 3.6.0 / Transformers 4.51.3。权重版本和优化参数见配置。

模型延迟从解码后的 CPU RGB 到 CPU actions，不含加载、预热、HTTP 和仿真；整集耗时为 128 集耗时之和。BF16 / INT8 分别使用 v10 / v17 冻结版本。数据来自冻结版本，当前分支未重测。

## 复现

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
